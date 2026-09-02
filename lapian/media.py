"""ffmpeg / ffprobe 封装：视频探查、切块、精确截帧。"""

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("lapian.media")


@dataclass
class Chunk:
    index: int
    path: Path
    offset: float    # 本块在原视频中的起始时间（秒）
    duration: float  # 本块实际时长（秒）


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    log.debug("run: %s", " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, capture_output=True, text=True, check=True)


def probe(video_path: Path) -> dict:
    """返回 {duration, width, height}。"""
    out = run([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height:format=duration",
        "-of", "json", str(video_path),
    ]).stdout
    info = json.loads(out)
    stream = info["streams"][0]
    return {
        "duration": float(info["format"]["duration"]),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
    }


def split_chunks(video_path: Path, out_dir: Path, chunk_len: int,
                 max_width: int = 960, crf: int = 26, audio_bitrate: str = "64k",
                 fps: int = 5, progress_cb=None) -> list[Chunk]:
    """把视频切成固定时长的 mp4 块（重编码保证切口精确、缩小体积控制 token）。

    fps 上限不影响模型理解（网关按 ~1 帧/秒采样），只用于减小上传体积。
    progress_cb(done, total)：每块就绪后回调（含命中缓存的块）。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    duration = probe(video_path)["duration"]
    spans = []  # (offset, actual_duration)；尾部不足 0.5s 的残片并入上一块
    offset = 0.0
    while offset < duration - 0.5:
        spans.append((offset, min(chunk_len, duration - offset)))
        offset += chunk_len

    chunks = []
    for idx, (off, actual) in enumerate(spans):
        out_path = out_dir / f"chunk_{idx:03d}.mp4"
        if not out_path.exists():
            run([
                "ffmpeg", "-y", "-loglevel", "error",
                "-ss", f"{off:.3f}", "-t", f"{actual:.3f}", "-i", str(video_path),
                "-vf", f"scale='min({max_width},iw)':-2,fps={fps}",
                "-c:v", "libx264", "-preset", "fast", "-crf", str(crf),
                "-c:a", "aac", "-b:a", audio_bitrate,
                str(out_path),
            ])
        chunks.append(Chunk(index=idx, path=out_path, offset=off, duration=actual))
        if progress_cb:
            progress_cb(idx + 1, len(spans))
    log.info("视频 %.1fs 切为 %d 块（每块 %ds）", duration, len(chunks), chunk_len)
    return chunks


def cut_clip(video_path: Path, start: float, end: float, out_path: Path,
             max_width: int = 720, crf: int = 26, audio_bitrate: str = "64k") -> bool:
    """从原视频精确切出 [start, end] 区间的小视频，供网页逐分镜回看。

    重编码保证切口精确（流拷贝会对齐到关键帧）；faststart 让浏览器即点即播。
    失败返回 False。
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(video_path),
            "-vf", f"scale='min({max_width},iw)':-2",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
            "-c:a", "aac", "-b:a", audio_bitrate,
            "-movflags", "+faststart",
            str(out_path),
        ])
    except subprocess.CalledProcessError:
        return False
    return out_path.exists() and out_path.stat().st_size > 0


def extract_frame(video_path: Path, ts: float, out_path: Path, quality: int = 3) -> bool:
    """从原视频在 ts 秒处精确截取一帧。失败（黑帧区间外/时间戳越界）返回 False。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", f"{ts:.3f}", "-i", str(video_path),
            "-frames:v", "1", "-q:v", str(quality),
            str(out_path),
        ])
    except subprocess.CalledProcessError:
        return False
    return out_path.exists() and out_path.stat().st_size > 0
