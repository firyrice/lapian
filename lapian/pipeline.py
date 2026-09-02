"""流水线：切块 -> 并行端到端分析 -> 截帧回填 -> 汇总输出。支持断点续跑。"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

from .analyze import analyze_chunk
from .client import LLMClient
from .media import cut_clip, extract_frame, probe, split_chunks

log = logging.getLogger("lapian.pipeline")


def _fmt_ts(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    h, m = divmod(int(m), 60)
    return f"{h:02d}:{m:02d}:{s:04.1f}"


class Pipeline:
    def __init__(self, config_path: Path, prompts_path: Path, model_override: str = None):
        self.cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        self.prompts = yaml.safe_load(Path(prompts_path).read_text(encoding="utf-8"))
        self.model = model_override or self.cfg["model"]
        api = self.cfg["api"]
        self.client = LLMClient(api["base_url"], api["api_key"],
                                timeout=api.get("timeout", 600),
                                max_retries=api.get("max_retries", 3))

    # ---- 各环节（中间结果落盘，重跑自动跳过） ----

    def _analyze_chunks(self, chunks, chunk_dir: Path, progress_cb=None) -> list[dict]:
        """并行分析所有块，返回合并后的分镜列表。"""
        workers = self.cfg.get("parallel", {}).get("workers", 4)

        def do_one(chunk):
            cache = chunk_dir / f"chunk_{chunk.index:03d}.json"
            if cache.exists():
                log.info("块 %d 已有缓存，跳过", chunk.index)
                return json.loads(cache.read_text(encoding="utf-8"))
            min_dur = self.cfg.get("postprocess", {}).get("min_shot_duration", 2.0)
            prod = self.cfg.get("production", {})
            shots = analyze_chunk(self.client, chunk, self.model, self.prompts["analyze"],
                                  min_shot_duration=min_dur,
                                  available_material_types=prod.get("available_material_types"),
                                  target_aspect_ratio=prod.get("target_aspect_ratio", "source"))
            cache.write_text(json.dumps(shots, ensure_ascii=False, indent=2), encoding="utf-8")
            return shots

        all_shots = []
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(do_one, c): c for c in chunks}
            for fut in as_completed(futures):
                chunk = futures[fut]
                try:
                    all_shots.extend(fut.result())
                except Exception as e:
                    log.error("块 %d 分析失败: %s", chunk.index, e)
                    raise
                done += 1
                if progress_cb:
                    progress_cb("analyze", f"{done}/{len(chunks)} 块完成")
        all_shots.sort(key=lambda s: s["start"])
        return all_shots

    def _attach_screenshots(self, shots: list[dict], video_path: Path, shot_dir: Path,
                            progress_cb=None) -> None:
        """按分镜起止时间截取首帧/中间帧/尾帧三张画面（shot_NNN_1/2/3.jpg），回填 screenshots 字段。

        首/尾帧向内收约 0.15s，避开切换瞬间的黑场/叠化帧；已存在的帧直接跳过（断点续跑）。
        """
        sc = self.cfg.get("screenshot", {})
        quality = sc.get("quality", 3)
        workers = sc.get("workers", 4)
        duration = probe(video_path)["duration"]

        def do_one(item) -> None:
            i, shot = item
            start = min(shot["start"], max(duration - 0.3, 0.0))
            end = min(shot["end"], duration)
            pad = min(0.15, max(end - start, 0.0) * 0.05)
            times = (start + pad, (start + end) / 2, end - pad)
            paths = []
            for k, ts in enumerate(times, 1):
                out = shot_dir / f"shot_{i:03d}_{k}.jpg"
                ts = min(max(ts, 0.0), duration - 0.05)
                if out.exists() or extract_frame(video_path, ts, out, quality):
                    paths.append(str(out.relative_to(shot_dir.parent)))
                else:
                    log.warning("分镜 %d 第 %d 帧截帧失败（ts=%.1f）", i, k, ts)
            shot["screenshots"] = paths

        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(do_one, item) for item in enumerate(shots, 1)]
            for fut in as_completed(futures):
                fut.result()
                done += 1
                if progress_cb:
                    progress_cb("screenshot", f"{done}/{len(shots)}")

    def _attach_clips(self, shots: list[dict], video_path: Path, shot_dir: Path,
                      progress_cb=None) -> None:
        """按分镜起止时间从原视频切出短片段（shot_NNN.mp4），回填 clip 字段。

        重编码较耗时，并行处理；已存在的片段直接跳过（断点续跑/补齐场景）。
        """
        cc = self.cfg.get("clip", {})
        if not cc.get("enabled", True):
            for shot in shots:
                shot["clip"] = ""
            return
        max_width = cc.get("max_width", 720)
        crf = cc.get("crf", 26)
        ab = cc.get("audio_bitrate", "64k")
        workers = cc.get("workers", 4)
        duration = probe(video_path)["duration"]

        def do_one(item) -> None:
            i, shot = item
            out = shot_dir / f"shot_{i:03d}.mp4"
            if out.exists():
                shot["clip"] = str(out.relative_to(shot_dir.parent))
                return
            start = min(shot["start"], max(duration - 0.3, 0.0))
            end = min(shot["end"], duration)
            if end - start < 0.2:
                shot["clip"] = ""
                return
            ok = cut_clip(video_path, start, end, out,
                          max_width=max_width, crf=crf, audio_bitrate=ab)
            if not ok:
                log.warning("分镜 %d 视频切分失败（%.1f~%.1f）", i, start, end)
            shot["clip"] = str(out.relative_to(shot_dir.parent)) if ok else ""

        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(do_one, item) for item in enumerate(shots, 1)]
            for fut in as_completed(futures):
                fut.result()
                done += 1
                if progress_cb:
                    progress_cb("clip", f"{done}/{len(shots)}")

    # ---- 汇总输出 ----

    @staticmethod
    def _final_record(i: int, shot: dict) -> dict:
        return {
            "分镜编号": i,
            "口播文字": shot["asr_text"],
            "画面截图": shot.get("screenshots", []),
            "分镜视频": shot.get("clip", ""),
            "画面描述": shot["description"],
            # 复刻配方；老缓存（chunk_results/）没有这个字段，用 get 兜住
            "制作方法": shot.get("production", {}),
            "开始时间": _fmt_ts(shot["start"]),
            "结束时间": _fmt_ts(shot["end"]),
        }

    @staticmethod
    def _production_lines(prod: dict) -> list[str]:
        """把制作方法渲染成 report.md 里给人读的小节（不塞 JSON）。"""
        if not prod:
            return []
        lines = [f"**制作方法** · form: `{prod.get('form', '')}`\n"]
        materials = prod.get("materials") or []
        if materials:
            lines.append("素材：\n")
            for m in materials:
                head = " / ".join(x for x in (m.get("id"), f"`{m.get('type')}`", m.get("role")) if x)
                lines.append(f"- {head} —— {m.get('desc', '')}")
                for label, key in (("检索/生成", "query"), ("规格", "specs"), ("处理", "treatment")):
                    if m.get(key):
                        lines.append(f"  - {label}：{m[key]}")
            lines.append("")
        else:
            lines.append("素材：无（纯代码画）\n")
        text_layers = prod.get("text_layers") or []
        if text_layers:
            lines.append("屏显文字：\n")
            for t in text_layers:
                tail = " ".join(x for x in (t.get("role"), t.get("style")) if x)
                lines.append(f"- 「{t.get('content', '')}」{f'（{tail}）' if tail else ''}")
            lines.append("")
        for label, key in (("构图", "layout"), ("时间轴", "timeline"),
                           ("音效", "sfx"), ("复刻备注", "notes")):
            if prod.get(key):
                lines.append(f"{label}：{prod[key]}\n")
        return lines

    def _write_outputs(self, shots: list[dict], out_dir: Path) -> None:
        records = [self._final_record(i, s) for i, s in enumerate(shots, 1)]
        (out_dir / "shots.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

        lines = ["# 拉片分析报告\n"]
        for r in records:
            lines.append(f"## 分镜 {r['分镜编号']}（{r['开始时间']} ~ {r['结束时间']}）\n")
            if r["画面截图"]:
                lines.append(" ".join(f"![分镜{r['分镜编号']}]({p})" for p in r["画面截图"]) + "\n")
            if r["分镜视频"]:
                lines.append(f"🎬 [分镜视频]({r['分镜视频']})\n")
            lines.append(f"**口播文字**：{r['口播文字'] or '（无）'}\n")
            lines.append(f"**画面描述**：{r['画面描述']}\n")
            lines.extend(self._production_lines(r["制作方法"]))
            lines.append("---\n")
        (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    # ---- 主流程 ----

    def run(self, input_video: Path, out_dir: Path, progress_cb=None) -> Path:
        """progress_cb(stage, detail)：可选进度回调，stage ∈ split/analyze/screenshot/clip/write。"""
        def report(stage: str, detail: str = "") -> None:
            if progress_cb:
                try:
                    progress_cb(stage, detail)
                except Exception:
                    log.debug("进度回调异常", exc_info=True)

        out_dir.mkdir(parents=True, exist_ok=True)
        chunk_video_dir = out_dir / "chunks"
        chunk_json_dir = out_dir / "chunk_results"
        chunk_json_dir.mkdir(exist_ok=True)
        shot_dir = out_dir / "shots"

        log.info("[1/5] 视频切块 ...")
        cc = self.cfg["chunk"]
        chunks = split_chunks(input_video, chunk_video_dir,
                              chunk_len=cc["length_seconds"], max_width=cc["max_width"],
                              crf=cc["crf"], audio_bitrate=cc["audio_bitrate"],
                              fps=cc.get("fps", 5),
                              progress_cb=lambda i, n: report("split", f"{i}/{n} 块"))

        log.info("[2/5] 逐块端到端分析（模型: %s） ...", self.model)
        report("analyze", f"共 {len(chunks)} 块")
        shots = self._analyze_chunks(chunks, chunk_json_dir, progress_cb)
        if not shots:
            raise RuntimeError("模型未输出任何分镜")

        log.info("[3/5] 截取 %d 个分镜的代表性画面 ...", len(shots))
        self._attach_screenshots(shots, input_video, shot_dir, progress_cb)

        log.info("[4/5] 切分 %d 个分镜的短片段 ...", len(shots))
        self._attach_clips(shots, input_video, shot_dir, progress_cb)

        log.info("[5/5] 汇总输出 ...")
        report("write", "生成报告")
        self._write_outputs(shots, out_dir)
        log.info("完成！结果目录: %s", out_dir)
        return out_dir / "shots.json"
