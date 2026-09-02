#!/usr/bin/env python3
"""拉片分析 Web 服务：上传视频 -> 选择模型 -> 后台分析 -> 在线查看拉片结果。

用法：
    python3 web.py                  # 默认 127.0.0.1:8000
    python3 web.py --port 9000
"""

import argparse
import json
import logging
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import uvicorn
import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from lapian.pipeline import Pipeline

log = logging.getLogger("lapian.web")

BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config.yaml"
PROMPTS_PATH = BASE_DIR / "prompts.yaml"
JOBS_DIR = BASE_DIR / "jobs"       # 每个任务一个子目录：原视频 + output/
STATIC_DIR = BASE_DIR / "static"

JOB_WORKERS = 2  # 同时运行的分析任务数，超出的排队等待


class JobStore:
    """任务注册表：内存索引 + job.json 落盘，重启后历史任务可恢复查看。"""

    def __init__(self, root: Path):
        self.root = root
        self.lock = threading.RLock()
        self.jobs: dict[str, dict] = {}
        root.mkdir(parents=True, exist_ok=True)
        for meta in sorted(root.glob("*/job.json")):
            try:
                job = json.loads(meta.read_text(encoding="utf-8"))
            except Exception:
                continue
            if job.get("status") in ("queued", "running"):
                job["status"] = "error"
                job["error"] = "服务重启导致任务中断，点“重试”可利用缓存断点续跑"
            self.jobs[job["id"]] = job

    def add(self, job: dict) -> None:
        with self.lock:
            self.jobs[job["id"]] = job
        self.save(job["id"])

    def get(self, job_id: str) -> dict | None:
        with self.lock:
            job = self.jobs.get(job_id)
            return dict(job) if job else None

    def list(self) -> list[dict]:
        with self.lock:
            return sorted((dict(j) for j in self.jobs.values()),
                          key=lambda j: j["created_at"], reverse=True)

    def update(self, job_id: str, **fields) -> None:
        with self.lock:
            if job_id not in self.jobs:
                return
            self.jobs[job_id].update(fields)
            data = dict(self.jobs[job_id])
        (self.root / job_id / "job.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def save(self, job_id: str) -> None:
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return
            data = dict(job)
        (self.root / job_id / "job.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def delete(self, job_id: str) -> None:
        with self.lock:
            self.jobs.pop(job_id, None)
        shutil.rmtree(self.root / job_id, ignore_errors=True)


store = JobStore(JOBS_DIR)
executor = ThreadPoolExecutor(max_workers=JOB_WORKERS)


def run_job(job_id: str) -> None:
    """在后台线程跑拉片流水线，进度实时写入任务状态。"""
    job = store.get(job_id)
    if not job:
        return
    job_dir = JOBS_DIR / job_id

    def progress(stage: str, detail: str = "") -> None:
        store.update(job_id, stage=stage, detail=detail)

    store.update(job_id, status="running", stage="prepare", detail="准备中", error="")
    try:
        pipeline = Pipeline(CONFIG_PATH, PROMPTS_PATH, model_override=job["model"])
        result = pipeline.run(job_dir / job["video_file"], job_dir / "output",
                              progress_cb=progress)
        records = json.loads(result.read_text(encoding="utf-8"))
        store.update(job_id, status="done", stage="done", detail="",
                     finished_at=time.time(), shot_count=len(records))
    except Exception as e:
        log.exception("任务 %s 失败", job_id)
        store.update(job_id, status="error", stage="", detail="",
                     error=str(e), finished_at=time.time())


app = FastAPI(title="拉片分析服务")


def _load_cfg() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


@app.get("/api/models")
def list_models():
    cfg = _load_cfg()
    return {"default": cfg["model"], "models": cfg.get("models") or [cfg["model"]]}


@app.post("/api/jobs")
def create_job(file: UploadFile = File(...), model: str = Form(...)):
    cfg = _load_cfg()
    allowed = cfg.get("models") or [cfg["model"]]
    if model not in allowed:
        raise HTTPException(400, f"模型不在可选列表中: {model}")
    suffix = Path(file.filename or "video.mp4").suffix.lower() or ".mp4"
    job_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True)
    video_name = f"input{suffix}"
    with (job_dir / video_name).open("wb") as f:
        shutil.copyfileobj(file.file, f, length=1024 * 1024)
    store.add({
        "id": job_id,
        "name": file.filename or "video",
        "model": model,
        "status": "queued",
        "stage": "", "detail": "", "error": "",
        "video_file": video_name,
        "created_at": time.time(),
    })
    executor.submit(run_job, job_id)
    return store.get(job_id)


@app.get("/api/jobs")
def list_jobs():
    return store.list()


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = store.get(job_id)
    if not job:
        raise HTTPException(404, "任务不存在")
    return job


@app.get("/api/jobs/{job_id}/result")
def job_result(job_id: str):
    job = store.get(job_id)
    if not job:
        raise HTTPException(404, "任务不存在")
    if job["status"] != "done":
        raise HTTPException(409, "任务尚未完成")
    result_file = JOBS_DIR / job_id / "output" / "shots.json"
    if not result_file.exists():
        raise HTTPException(404, "结果文件缺失")
    return json.loads(result_file.read_text(encoding="utf-8"))


@app.post("/api/jobs/{job_id}/retry")
def retry_job(job_id: str):
    job = store.get(job_id)
    if not job:
        raise HTTPException(404, "任务不存在")
    if job["status"] in ("queued", "running"):
        raise HTTPException(409, "任务正在进行中")
    if not (JOBS_DIR / job_id / job["video_file"]).exists():
        raise HTTPException(400, "原始视频已丢失，无法重试")
    store.update(job_id, status="queued", stage="", detail="", error="")
    executor.submit(run_job, job_id)
    return store.get(job_id)


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    job = store.get(job_id)
    if not job:
        raise HTTPException(404, "任务不存在")
    if job["status"] in ("queued", "running"):
        raise HTTPException(409, "任务进行中，无法删除")
    store.delete(job_id)
    return {"ok": True}


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


# 分镜截图 / shots.json / report.md 等产物直接通过 /jobs/<id>/output/... 访问
app.mount("/jobs", StaticFiles(directory=JOBS_DIR), name="jobs")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description="拉片分析 Web 服务")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
