"""LLM 网关调用封装：视频上传、重试、JSON 解析兜底。"""

import base64
import json
import logging
import re
import time
from pathlib import Path

import openai

log = logging.getLogger("lapian.client")


class VideoDroppedError(Exception):
    """网关静默丢弃了视频内容（usage 中无 video_tokens，或模型自称没收到视频）。"""


# 模型未收到视频时的典型措辞
_VIDEO_MISSING_PAT = re.compile(
    r"没有上传|未上传|未检测到视频|没有.{0,6}视频|无法.{0,6}视频|请上传|请提供视频|"
    r"no video|not.*video.*provided", re.IGNORECASE)


def _video_tokens(resp) -> int:
    """从 usage.prompt_tokens_details 提取 video_tokens（网关扩展字段）。"""
    try:
        details = resp.model_dump().get("usage", {}).get("prompt_tokens_details") or {}
        return int(details.get("video_tokens") or 0)
    except Exception:
        return 0


class LLMClient:
    def __init__(self, base_url: str, api_key: str, timeout: int = 600, max_retries: int = 3):
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.max_retries = max_retries

    def analyze_video(self, video_path: Path, prompt: str, model: str) -> str:
        """把视频文件 + prompt 发给模型，返回文本响应。

        两道防线检测网关静默丢视频（该网关存在此随机故障）：
        1. usage.prompt_tokens_details.video_tokens 为 0/缺失
        2. 响应文本自称未收到视频
        命中即重试；网络/5xx/限流同样指数退避重试。
        """
        b64 = base64.b64encode(Path(video_path).read_bytes()).decode()
        content = [
            {"type": "text", "text": prompt},
            {"type": "file", "file": {
                "filename": Path(video_path).name,
                "file_data": f"data:video/mp4;base64,{b64}",
            }},
        ]
        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": content}],
                )
                text = resp.choices[0].message.content or ""
                if _video_tokens(resp) <= 0:
                    raise VideoDroppedError("usage 中 video_tokens 为 0，视频被网关丢弃")
                if _VIDEO_MISSING_PAT.search(text):
                    raise VideoDroppedError(f"模型自称未收到视频: {text[:80]}")
                return text
            except Exception as e:  # VideoDroppedError / 网络错误 / 网关 5xx / 限流统一重试
                last_err = e
                wait = 2 ** attempt * 5
                log.warning("请求失败（第 %d/%d 次）: %s；%ds 后重试", attempt, self.max_retries, e, wait)
                if attempt < self.max_retries:
                    time.sleep(wait)
        raise RuntimeError(f"模型请求失败，已重试 {self.max_retries} 次: {last_err}")

    def chat(self, prompt: str, model: str) -> str:
        """纯文本调用（无视频），用于全片统筹类任务。网络/5xx/限流统一指数退避重试。"""
        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                last_err = e
                wait = 2 ** attempt * 5
                log.warning("文本请求失败（第 %d/%d 次）: %s；%ds 后重试", attempt, self.max_retries, e, wait)
                if attempt < self.max_retries:
                    time.sleep(wait)
        raise RuntimeError(f"文本请求失败，已重试 {self.max_retries} 次: {last_err}")


def parse_json(text: str) -> list:
    """从模型输出中稳健提取 JSON 数组。

    依次尝试：直接解析 -> 去掉 markdown 代码围栏 -> 截取第一个 [...] 片段。
    """
    text = text.strip()
    candidates = [text]
    # 去掉 ```json ... ``` 围栏
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        candidates.insert(0, fence.group(1).strip())
    # 截取第一个 [ 到最后一个 ]
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])

    for cand in candidates:
        try:
            data = json.loads(cand)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            continue
    raise ValueError(f"无法从模型输出解析 JSON 数组，输出前 300 字符: {text[:300]}")


def parse_json_obj(text: str) -> dict:
    """从模型输出中稳健提取 JSON 对象（与 parse_json 同思路，目标是 {...}）。"""
    text = text.strip()
    candidates = [text]
    # 去掉 ```json ... ``` 围栏
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        candidates.insert(0, fence.group(1).strip())
    # 截取第一个 { 到最后一个 }
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])

    for cand in candidates:
        try:
            data = json.loads(cand)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue
    raise ValueError(f"无法从模型输出解析 JSON 对象，输出前 300 字符: {text[:300]}")
