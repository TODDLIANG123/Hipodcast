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
import time
import asyncio

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "").strip()

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"

# 每个分块的大致字数上限。播客长稿会被切成若干这么大的块，逐块处理。
# 调小 -> 更稳但更慢更费；调大 -> 更快，但太大可能超出模型单次输出长度。
CHUNK_CHARS = 3500

app = FastAPI(title="Hipodcast MVP")


# --------------------------------------------------------------------------
# 工具：把长文本按句子边界切成若干块
# --------------------------------------------------------------------------
def split_into_chunks(text: str, max_chars: int = CHUNK_CHARS):
    """按中英文句末标点切句，再把句子拼成不超过 max_chars 的块。"""
    text = text.strip()
    if not text:
        return []
    # 在句末标点后插入分割点（保留标点）
    parts = re.split(r"(?<=[。！？!?\.\n])", text)
    chunks, buf = [], ""
    for p in parts:
        if not p:
            continue
        if len(buf) + len(p) > max_chars and buf:
            chunks.append(buf)
            buf = p
        else:
            buf += p
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
PROMPT_POLISH = (
    "你是一名专业的中文文字编辑。下面是一段播客口语逐字稿，可能含口头禅、重复、语气词、"
    "口误。请把它改写成书面化、通顺、易读的文字：去掉语气词和无意义重复，理顺语序，"
    "适当分段。要求：忠实保留原意和所有信息点，不要遗漏、不要自行添加观点、不要做总结。"
    "直接输出改写后的正文，不要任何前后说明。"
)

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
async def process_polish_or_translate(text: str, system_prompt: str) -> dict:
    """书面化 / 翻译：逐块处理，带上一块结尾做衔接，再拼接。"""
    chunks = split_into_chunks(text)
    results = []
    prev_tail = ""
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
        prev_tail = out[-200:]
    return {"result": "\n\n".join(results), "chunks": len(chunks)}


async def process_summary(text: str) -> dict:
    """总结：先对每块各自提要点（map），再汇总成稿（reduce）。"""
    chunks = split_into_chunks(text)
    if len(chunks) <= 1:
        # 短文直接出结构化总结
        out = await deepseek_chat(PROMPT_SUMMARY_REDUCE,
                                  "请直接对下面内容做结构化总结：\n\n" + text)
        return {"result": out, "chunks": len(chunks)}
    partials = []
    for chunk in chunks:
        partials.append(await deepseek_chat(PROMPT_SUMMARY_MAP, chunk))
    merged = "\n\n".join(f"【第 {i+1} 部分要点】\n{p}" for i, p in enumerate(partials))
    final = await deepseek_chat(PROMPT_SUMMARY_REDUCE, merged)
    return {"result": final, "chunks": len(chunks)}


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
):
    text = (text or "").strip()
    if not text:
        return JSONResponse(status_code=400, content={"error": "请先提供逐字稿文本。"})
    try:
        if mode == "polish":
            return await process_polish_or_translate(text, PROMPT_POLISH)
        elif mode == "summarize":
            return await process_summary(text)
        elif mode == "translate":
            return await process_polish_or_translate(text, prompt_translate(target_lang))
        else:
            return JSONResponse(status_code=400, content={"error": f"未知操作：{mode}"})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.get("/")
async def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))


app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")


if __name__ == "__main__":
    import uvicorn
    print("\n  Hipodcast 已启动，用浏览器打开： http://127.0.0.1:8000\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)
