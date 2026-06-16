"""
Hipodcast MVP —— 播客逐字稿 AI 加工工具（Web 验证版）

四步闭环：来源 → 逐字稿 → AI 加工（书面化/总结/翻译）→ 导出 Markdown

- AI 加工：DeepSeek（你已有 key，现在就能用）
- 音频转写：阿里云 DashScope Paraformer（按音频链接转写，之后配 key 启用）
- 长稿（如 4 小时播客）自动分段处理，再拼接 —— 解决模型一次输出长度有限的问题

运行：  python app.py   然后浏览器打开 http://127.0.0.1:8000
"""

import os
import re
import json
import time
import asyncio

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "").strip()

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
# DeepSeek V4：deepseek-v4-flash(便宜快) / deepseek-v4-pro(更强)，
# 两者都是 1M 上下文、384K 输出帽。旧别名 deepseek-chat 只有 8K 输出且将弃用，已不用。
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash").strip()

# 单次输出上限。V4 实际可达 384K，这里设一个够覆盖单块、又不过分的值即可。
DEEPSEEK_MAX_TOKENS = int(os.getenv("DEEPSEEK_MAX_TOKENS", "16384"))

# 每个分块的大致字数上限。书面化是 1:1 改写，输出≈输入，所以块由两头夹定：
#   上限——别让单块输出超过 max_tokens；下限——块越小越不容易"偷懒漏内容"。
# V4 输出帽很大，截断已不是威胁，所以默认块比旧版大很多，拼接缝更少。
CHUNK_CHARS = int(os.getenv("CHUNK_CHARS", "6000"))        # 普通模式
HIFI_CHUNK_CHARS = int(os.getenv("HIFI_CHUNK_CHARS", "3000"))  # 高保真模式(更小块更准)
CONTEXT_TAIL = int(os.getenv("CONTEXT_TAIL", "400"))       # 传给下一块的衔接字数

app = FastAPI(title="Hipodcast MVP")


# --------------------------------------------------------------------------
# 工具：把长文本按句子边界切成若干块
# --------------------------------------------------------------------------
def _split_sentences(text: str):
    """按中英文句末标点切句（保留标点）。"""
    return [p for p in re.split(r"(?<=[。！？!?\.])\s*", text) if p]


def split_into_chunks(text: str, max_chars: int = CHUNK_CHARS):
    """优先按段落切，段落过大再按句子切，最后拼成不超过 max_chars 的块。
    尽量在自然边界（段落/句子）断开，避免把一句话切两半。"""
    text = text.strip()
    if not text:
        return []
    # 先按段落（空行或换行）拆成基本单元
    paragraphs = [p for p in re.split(r"\n\s*\n|\n", text) if p.strip()]
    units = []
    for para in paragraphs:
        if len(para) <= max_chars:
            units.append(para)
        else:
            # 段落本身就超长 -> 退化到按句子切
            units.extend(_split_sentences(para))

    chunks, buf = [], ""
    for u in units:
        sep = "\n" if buf else ""
        if len(buf) + len(sep) + len(u) > max_chars and buf:
            chunks.append(buf)
            buf = u
        else:
            buf += sep + u
    if buf.strip():
        chunks.append(buf)
    return chunks


# --------------------------------------------------------------------------
# DeepSeek 调用
# --------------------------------------------------------------------------
async def deepseek_chat(system_prompt: str, user_content: str) -> str:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError(
            "还没有配置 DEEPSEEK_API_KEY。请把 .env.example 复制成 .env 并填入你的 DeepSeek key。"
        )
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.3,
        "max_tokens": DEEPSEEK_MAX_TOKENS,  # 显式设大，避免默认值把长输出截断
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=300) as client:
        r = await client.post(DEEPSEEK_URL, headers=headers, json=payload)
        if r.status_code != 200:
            raise RuntimeError(f"DeepSeek 接口报错 {r.status_code}: {r.text[:500]}")
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()


# --------------------------------------------------------------------------
# 提示词
# --------------------------------------------------------------------------
# 书面化的核心提示词 —— 这是产品的灵魂，所有规则都为"忠实完整 + 干净可读"服务。
PROMPT_POLISH_BASE = """你是一名顶尖的播客文字整理编辑。任务：把下面这段口语逐字稿整理成高质量的书面文本，供人阅读、或交给 AI 做进一步分析。

【铁律 · 必须遵守】
1. 忠实第一：完整保留原文所有信息点、观点、论证、事实、数字、案例、人名与专有名词。绝不删减、绝不概括成摘要、绝不添加原文没有的内容或你自己的评论。
2. 去口语噪音：删掉语气词与口头禅（嗯、啊、那个、就是说、对吧）、无意义重复、说了一半又重来的开头、明显的卡顿。
3. 修顺不改意：在不改变原意的前提下，理顺语序、补全被省略的成分、合并破碎的短句，使其读起来是完整通顺的书面句子。
4. 纠错有度：结合上下文修正语音识别明显的同音错别字（如"原理"误作"愿意"）；若某处实在拿不准，保留原词并在其后标注 [?]，不要臆造。
5. 结构化：合理分段；当话题明显切换时，可加一个简短小标题（用 Markdown 的 ## ）分节。
6. 保留"人味"与表达习惯：必须保留说话人标志性的措辞、惯用词、个人化的比喻、幽默感、风格化的口头表达、句式节奏和第一人称视角。只清除"无意义的语流噪音"（纯填充词、结巴、说错重来），不要把带个人风格的表达一并抹平，更不要替换成更"标准"却更平庸的说法。判断标准：去掉它读起来更顺且不损失风格 → 去；去掉它就少了"那个人的味道" → 留。

直接输出整理后的正文，不要任何开场白、说明或结尾总结。"""

# 只保留两档，对应清洗强度的两端，含义直观、无需纠结
POLISH_STYLES = {
    "voiced": "【风格】保留原声（轻清洗）：只去掉纯粹的填充词和结巴重来，最大限度保留口语原貌、口头禅、语气、节奏和具体例子，尽量贴近『他本人在说话』。",
    "readable": "【风格】深度整理：在不丢任何信息的前提下，把内容整理成结构清晰、逻辑连贯、好读的文章——合理分段、用小标题分节、适当补充过渡句。",
}


def prompt_polish(style: str = "voiced") -> str:
    extra = POLISH_STYLES.get(style, POLISH_STYLES["voiced"])
    return PROMPT_POLISH_BASE + "\n\n" + extra

PROMPT_SUMMARY_MAP = (
    "请阅读下面这段内容，用简洁的要点列表（3-6 条）概括其核心信息，"
    "只基于原文、不要编造。直接输出要点。"
)

PROMPT_SUMMARY_REDUCE = (
    "下面是一篇长内容各部分的要点汇总。请整合成一份结构化总结，用 Markdown 格式：\n"
    "1. 开头一段话总览全篇；\n"
    "2. 按主题分小标题列出关键要点；\n"
    "3. 最后列出『关键结论 / 行动项』（如果有）。\n"
    "只基于提供的要点，不要编造。"
)


def prompt_translate(target_lang: str) -> str:
    return (
        f"请把下面的文字准确翻译成{target_lang}，忠实原意、保持段落结构、表达自然流畅，"
        "不要添加任何解释或注释，直接输出译文。"
    )


# --------------------------------------------------------------------------
# 核心：分段加工
# --------------------------------------------------------------------------
async def run_process(text, mode, target_lang, high_fidelity, style):
    """统一的分段处理。这是个异步生成器，逐步 yield 进度事件：
        {"type":"start","total":N}
        {"type":"progress","done":k,"total":N,"label":"..."}
        {"type":"done","result":...,"chunks":N,"warnings":[...]}
    这样前端就能实时显示"第 k/N 块"。"""
    max_chars = HIFI_CHUNK_CHARS if high_fidelity else CHUNK_CHARS

    if mode in ("polish", "translate"):
        system_prompt = prompt_polish(style) if mode == "polish" else prompt_translate(target_lang)
        check_fidelity = mode == "polish"
        chunks = split_into_chunks(text, max_chars)
        total = len(chunks)
        yield {"type": "start", "total": total}
        results, warnings, prev_tail = [], [], ""
        for i, chunk in enumerate(chunks):
            if prev_tail:
                user_content = (
                    f"【前文结尾，仅供衔接参考，不要重复输出】\n{prev_tail}\n\n"
                    f"【需要处理的正文】\n{chunk}"
                )
            else:
                user_content = chunk
            out = await deepseek_chat(system_prompt, user_content)
            results.append(out)
            prev_tail = out[-CONTEXT_TAIL:]
            if check_fidelity and len(chunk) >= 200 and len(out) < len(chunk) * 0.5:
                warnings.append(
                    f"第 {i+1}/{total} 块：输出字数仅为输入的 {len(out)/len(chunk):.0%}，"
                    "可能有内容被压缩，建议开高保真模式重试。"
                )
            yield {"type": "progress", "done": i + 1, "total": total,
                   "label": f"精修中… 第 {i+1}/{total} 块"}
        yield {"type": "done", "result": "\n\n".join(results),
               "chunks": total, "warnings": warnings}

    elif mode == "summarize":
        chunks = split_into_chunks(text, max_chars)
        if len(chunks) <= 1:
            yield {"type": "start", "total": 1}
            out = await deepseek_chat(PROMPT_SUMMARY_REDUCE,
                                      "请直接对下面内容做结构化总结：\n\n" + text)
            yield {"type": "progress", "done": 1, "total": 1, "label": "总结中…"}
            yield {"type": "done", "result": out, "chunks": 1, "warnings": []}
            return
        total = len(chunks) + 1  # 各块提要点 + 最后汇总
        yield {"type": "start", "total": total}
        partials = []
        for i, chunk in enumerate(chunks):
            partials.append(await deepseek_chat(PROMPT_SUMMARY_MAP, chunk))
            yield {"type": "progress", "done": i + 1, "total": total,
                   "label": f"提炼要点… 第 {i+1}/{len(chunks)} 块"}
        merged = "\n\n".join(f"【第 {i+1} 部分要点】\n{p}" for i, p in enumerate(partials))
        final = await deepseek_chat(PROMPT_SUMMARY_REDUCE, merged)
        yield {"type": "progress", "done": total, "total": total, "label": "汇总成稿…"}
        yield {"type": "done", "result": final, "chunks": len(chunks), "warnings": []}

    else:
        raise RuntimeError(f"未知操作：{mode}")


# --------------------------------------------------------------------------
# 阿里云 DashScope —— 录音文件识别（按音频 URL 转写）
# --------------------------------------------------------------------------
DASHSCOPE_SUBMIT = (
    "https://dashscope.aliyuncs.com/api/v1/services/audio/asr/transcription"
)
DASHSCOPE_TASK = "https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"


async def transcribe_audio_url(audio_url: str) -> dict:
    if not DASHSCOPE_API_KEY:
        raise RuntimeError(
            "还没有配置 DASHSCOPE_API_KEY，无法做音频转写。"
            "你可以先用『粘贴文字稿』那一栏体验 AI 加工；"
            "想用音频转写，请申请阿里云 DashScope key 并填进 .env。"
        )
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json",
        "X-DashScope-Async": "enable",
    }
    payload = {
        "model": "paraformer-v2",
        "input": {"file_urls": [audio_url]},
        "parameters": {"language_hints": ["zh", "en"]},
    }
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(DASHSCOPE_SUBMIT, headers=headers, json=payload)
        if r.status_code != 200:
            raise RuntimeError(f"提交转写任务失败 {r.status_code}: {r.text[:500]}")
        task_id = r.json()["output"]["task_id"]

        # 轮询任务结果
        poll_headers = {"Authorization": f"Bearer {DASHSCOPE_API_KEY}"}
        deadline = time.time() + 1800  # 最多等 30 分钟
        while time.time() < deadline:
            await asyncio.sleep(5)
            t = await client.get(DASHSCOPE_TASK.format(task_id=task_id),
                                 headers=poll_headers)
            out = t.json().get("output", {})
            status = out.get("task_status")
            if status == "SUCCEEDED":
                results = out.get("results", [])
                if not results or "transcription_url" not in results[0]:
                    raise RuntimeError("转写完成但未返回结果链接。")
                tr = await client.get(results[0]["transcription_url"])
                return {"transcript": _extract_transcript_text(tr.json())}
            if status == "FAILED":
                raise RuntimeError(f"转写任务失败：{out.get('message', out)}")
        raise RuntimeError("转写超时（音频可能过长），请稍后重试或换更短的音频。")


def _extract_transcript_text(data: dict) -> str:
    """从 Paraformer 结果 JSON 里抽出纯文本。"""
    texts = []
    for tr in data.get("transcripts", []):
        if tr.get("text"):
            texts.append(tr["text"])
            continue
        for sent in tr.get("sentences", []):
            if sent.get("text"):
                texts.append(sent["text"])
    return "\n".join(texts).strip() or "(未识别到文字)"


# --------------------------------------------------------------------------
# 路由
# --------------------------------------------------------------------------
@app.get("/api/config")
async def config():
    """前端用来判断哪些功能已就绪。"""
    return {
        "deepseek_ready": bool(DEEPSEEK_API_KEY),
        "dashscope_ready": bool(DASHSCOPE_API_KEY),
    }


@app.post("/api/transcribe")
async def api_transcribe(audio_url: str = Body(..., embed=True)):
    try:
        return await transcribe_audio_url(audio_url.strip())
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.post("/api/process")
async def api_process(
    text: str = Body(..., embed=True),
    mode: str = Body(..., embed=True),
    target_lang: str = Body("英文", embed=True),
    high_fidelity: bool = Body(False, embed=True),
    style: str = Body("voiced", embed=True),
):
    """流式处理：以 NDJSON（每行一个 JSON）逐步返回进度，最后一行是结果。
    前端边读边显示『第 k/N 块』。"""
    text = (text or "").strip()
    if not text:
        return JSONResponse(status_code=400, content={"error": "请先提供逐字稿文本。"})

    async def gen():
        try:
            async for ev in run_process(text, mode, target_lang, high_fidelity, style):
                yield json.dumps(ev, ensure_ascii=False) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "error": str(e)}, ensure_ascii=False) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.get("/")
async def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))


app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")


if __name__ == "__main__":
    import uvicorn
    print("\n  Hipodcast 已启动，用浏览器打开： http://127.0.0.1:8000\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)
