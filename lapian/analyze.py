"""环节核心：把视频块发给 Gemini 端到端分析，归一化输出分镜数据。"""

import logging
import re
from pathlib import Path

from .client import LLMClient, parse_json
from .media import Chunk

log = logging.getLogger("lapian.analyze")

REQUIRED_KEYS = ("start", "end", "asr_text", "shot_type", "description", "representative_time", "production")

FORMS = ("raw", "mg-compose", "mg-pure")
ALL_MATERIAL_TYPES = ("a-roll", "local", "cloud", "web", "meme", "gen-img", "gen-vid")
# 无口播分镜的合法类型（有口播的一律归一为「口播镜」）
SPECIAL_SHOT_TYPES = ("片头", "章节卡", "转场卡", "片尾", "纯画面")
# 特殊镜（片头/转场卡等）天然短，时长下限放宽到此值；口播镜仍用 min_shot_duration
SPECIAL_SHOT_MIN_DURATION = 0.5

_MATERIAL_STR_KEYS = ("type", "role", "desc", "query", "specs", "treatment")
_PRODUCTION_STR_KEYS = ("layout", "timeline", "sfx", "notes")
# 分镜级「视频编导构思」八个子字段（与 prompts.yaml【视频编导构思规格】对齐，改动要同步）
_DIRECTING_STR_KEYS = ("一句话阐述", "叙事功能", "表达目标", "视觉策略", "声画关系", "对位锚点", "情绪节奏", "取舍基准")
# 全片级「编导构思」字段（与 prompts.yaml 的 concept 环节对齐，改动要同步）
_CONCEPT_STR_KEYS = ("一句话立意", "节奏设计")
_CONCEPT_SECTION_KEYS = ("章节名", "覆盖分镜", "叙事任务", "情绪基调")
_CONCEPT_GROUPS = {"素材要求": ("国家", "年代", "素材类型", "素材风格"),
                   "MG画风": ("职能与密度", "配色", "字体", "动画样式", "版式与底纹")}


def _to_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _s(v) -> str:
    return str(v).strip() if v is not None else ""


# timeline 每条以「秒数 + 空格」开头，用 ｜ 分隔；括号里的 (0.4s) 是时长不是时间点，不匹配
_TL_TS_PAT = re.compile(r"(?:(?<=^)|(?<=｜))(\s*)(\d+(?:\.\d+)?)(?=[\s，,])")


def _rebase_timeline(timeline: str, shot_start: float, duration: float, shot_label: str) -> str:
    """把 timeline 里误写成【视频绝对时间】的秒数改回【相对本镜起点】。

    模型常把分镜起止时间直接抄进 timeline（分镜 6.6~14.1 秒，timeline 就从 6.6 起）。
    下游按「本镜起点 = 0」排动效，不改则每条动效都错位。
    仅在「超出本镜时长、但恰好落在本镜绝对区间内」时才改，避免误伤正确输出。
    """
    ts = [float(m.group(2)) for m in _TL_TS_PAT.finditer(timeline)]
    if not ts or shot_start <= 0:
        return timeline
    tol = 0.6
    looks_absolute = (max(ts) > duration + tol
                      and min(ts) >= shot_start - tol
                      and max(ts) <= shot_start + duration + tol)
    if not looks_absolute:
        # 已是相对时间但仍超出本镜时长：多为超长分镜里模型算漂了。改不了（改就是瞎猜），只报警
        if max(ts) > duration + tol:
            log.warning("%s 的 timeline 最大秒数 %.1fs 超出本镜时长 %.1fs，动效尾部可能溢出",
                        shot_label, max(ts), duration)
        return timeline
    log.warning("%s 的 timeline 用了视频绝对时间（起点 %.1fs），已改为相对本镜起点",
                shot_label, shot_start)
    return _TL_TS_PAT.sub(
        lambda m: f"{m.group(1)}{max(0.0, float(m.group(2)) - shot_start):.1f}", timeline)


def _normalize_production(raw, allowed_types: list[str], shot_label: str,
                          shot_start: float = 0.0, duration: float = 0.0) -> dict:
    """归一化「制作方法」字段：补齐缺失键、过滤越界素材通道、修正 form × material 耦合。

    模型输出不可信，这里做防御性修正（与时间戳截断修正同思路），保证下游拿到的结构稳定。
    shot_start / duration 用于把 timeline 里误写的绝对时间改回相对本镜起点（均为块内相对值）。
    """
    if not isinstance(raw, dict):
        log.warning("%s 缺少或非法 production 字段（%r），输出空骨架", shot_label, type(raw).__name__)
        raw = {}

    materials = []
    for m in raw.get("materials") or []:
        if not isinstance(m, dict):
            log.warning("%s 的某份素材不是对象，丢弃: %r", shot_label, m)
            continue
        mtype = _s(m.get("type"))
        if mtype not in allowed_types:
            log.warning("%s 的素材通道 %r 不在可用清单内，丢弃该素材", shot_label, mtype)
            continue
        item = {"id": _s(m.get("id")) or f"m{len(materials) + 1}"}
        item.update({k: _s(m.get(k)) for k in _MATERIAL_STR_KEYS})
        materials.append(item)

    form = _s(raw.get("form"))
    if form not in FORMS:
        inferred = "mg-pure" if not materials else ("raw" if len(materials) == 1 else "mg-compose")
        log.warning("%s 的 form %r 非法，按素材数推断为 %s", shot_label, form, inferred)
        form = inferred

    # 两轴耦合约束：raw 恰 1 个素材 / mg-compose ≥1 / mg-pure 空
    if form == "mg-pure" and materials:
        log.warning("%s 为 mg-pure 却带了 %d 份素材，改判为 mg-compose", shot_label, len(materials))
        form = "mg-compose"
    elif form == "raw" and len(materials) > 1:
        log.warning("%s 为 raw 却带了 %d 份素材，改判为 mg-compose", shot_label, len(materials))
        form = "mg-compose"
    elif form in ("raw", "mg-compose") and not materials:
        log.warning("%s 为 %s 却没有可用素材，改判为 mg-pure", shot_label, form)
        form = "mg-pure"

    text_layers = []
    for t in raw.get("text_layers") or []:
        if isinstance(t, dict):
            text_layers.append({k: _s(t.get(k)) for k in ("content", "role", "style")})
        elif t:  # 模型偶发直接给字符串
            text_layers.append({"content": _s(t), "role": "", "style": ""})

    out = {"form": form, "materials": materials, "text_layers": text_layers}
    out.update({k: _s(raw.get(k)) for k in _PRODUCTION_STR_KEYS})
    return out


def _normalize_directing(raw, shot_label: str) -> dict:
    """归一化「视频编导构思」：模型偶发漏给或给成字符串，兜底为空骨架，保证下游结构稳定。

    软词表字段（叙事功能/视觉策略等）不做枚举硬校验——模型自造短语也保留，
    只在 prompt 层引导向词表收敛，避免误删有信息量的输出。
    """
    if isinstance(raw, str):
        # 模型把构思压成了一句话：塞进「一句话阐述」，其余留空
        return {k: (raw.strip() if k == "一句话阐述" else "") for k in _DIRECTING_STR_KEYS}
    if not isinstance(raw, dict):
        if raw is not None:
            log.warning("%s 的 directing 字段非法（%r），输出空骨架", shot_label, type(raw).__name__)
        return {k: "" for k in _DIRECTING_STR_KEYS}
    return {k: _s(raw.get(k)) for k in _DIRECTING_STR_KEYS}


def normalize_concept(raw) -> dict:
    """归一化全片编导构思（concept 环节输出）：补齐缺失键，保证产物结构稳定。"""
    if not isinstance(raw, dict):
        log.warning("全片编导构思输出不是对象（%r），输出空骨架", type(raw).__name__)
        raw = {}
    out = {k: _s(raw.get(k)) for k in _CONCEPT_STR_KEYS}
    sections = raw.get("章节结构")
    out["章节结构"] = [
        {k: _s(item.get(k)) for k in _CONCEPT_SECTION_KEYS}
        for item in sections if isinstance(item, dict)
    ] if isinstance(sections, list) else []
    for group, keys in _CONCEPT_GROUPS.items():
        g = raw.get(group)
        out[group] = ({k: _s(g.get(k)) for k in keys} if isinstance(g, dict)
                      else {k: "" for k in keys})
    return out


def analyze_chunk(client: LLMClient, chunk: Chunk, model: str, prompt_template: str,
                  max_parse_retries: int = 2, min_shot_duration: float = 2.0,
                  available_material_types: list[str] = None,
                  target_aspect_ratio: str = "source") -> list[dict]:
    """分析单个视频块，返回【绝对时间戳】的分镜列表。JSON 解析失败会重问模型。"""
    allowed_types = list(available_material_types or ALL_MATERIAL_TYPES)
    prompt = (prompt_template
              .replace("{chunk_duration}", f"{chunk.duration:.0f}")
              .replace("{available_material_types}", " / ".join(allowed_types))
              .replace("{target_aspect_ratio}", target_aspect_ratio))
    shots = None
    for attempt in range(1, max_parse_retries + 1):
        raw = client.analyze_video(chunk.path, prompt, model)
        try:
            shots = parse_json(raw)
            break
        except ValueError as e:
            log.warning("块 %d 输出解析失败（第 %d/%d 次）: %s", chunk.index, attempt, max_parse_retries, e)
    if shots is None:
        raise RuntimeError(f"块 {chunk.index} 多次输出非法 JSON，放弃")

    results = []
    for i, s in enumerate(shots):
        if not isinstance(s, dict):
            log.warning("块 %d 第 %d 个分镜不是对象，跳过: %r", chunk.index, i, s)
            continue
        start = _to_float(s.get("start")) + chunk.offset
        end = _to_float(s.get("end")) + chunk.offset
        rep = _to_float(s.get("representative_time"), -1) + chunk.offset

        # 时间戳合法性修正：截断到本块范围
        lo, hi = chunk.offset, chunk.offset + chunk.duration
        start = max(lo, min(start, hi))
        end = max(lo, min(end, hi))
        if end <= start:
            end = min(start + 0.5, hi)
        if not (start <= rep <= end):
            rep = (start + end) / 2  # 代表帧时间非法时回退为分镜中点

        asr = str(s.get("asr_text") or "").strip()
        # shot_type 归一化：有口播一律「口播镜」；无口播必须在特殊类型枚举内，否则回退「纯画面」
        shot_type = _s(s.get("shot_type"))
        if asr:
            shot_type = "口播镜"
        elif shot_type not in SPECIAL_SHOT_TYPES:
            if shot_type and shot_type != "口播镜":
                log.warning("块 %d 第 %d 个分镜的 shot_type %r 非法，回退为 纯画面",
                            chunk.index, i + 1, shot_type)
            shot_type = "纯画面"

        results.append({
            "start": round(start, 1),
            "end": round(end, 1),
            "representative_time": round(rep, 1),
            "asr_text": asr,
            "shot_type": shot_type,
            "directing": _normalize_directing(s.get("directing"), f"块 {chunk.index} 第 {i + 1} 个分镜"),
            "description": str(s.get("description") or "").strip(),
            # timeline 的秒数按【块内相对时间】判断（模型给的 start 也是块内相对）
            "production": _normalize_production(
                s.get("production"), allowed_types, f"块 {chunk.index} 第 {i + 1} 个分镜",
                shot_start=start - chunk.offset, duration=end - start),
            "chunk_index": chunk.index,
        })

    # 过滤退化分镜（模型偶发在块边界输出 start≈end 的幻影分镜）
    # 口播镜下限 min_shot_duration；无口播的特殊镜（片头/转场卡等）天然更短，下限放宽
    before = len(results)
    results = [s for s in results
               if s["end"] - s["start"] >= (SPECIAL_SHOT_MIN_DURATION
                                            if s["shot_type"] != "口播镜" else min_shot_duration)]
    if len(results) < before:
        log.info("块 %d 过滤掉 %d 个时长不足的退化分镜", chunk.index, before - len(results))

    results.sort(key=lambda s: s["start"])
    log.info("块 %d（偏移 %.0fs）分析出 %d 个分镜", chunk.index, chunk.offset, len(results))
    return results
