# 🎙️ Hipodcast MVP（验证版）

一个用来**验证想法**的小网页：把播客逐字稿 → **书面化 / 总结 / 翻译** → **导出 Markdown**（可丢进 Obsidian 或喂给 AI）。

这是 Web 验证版，不是最终的鸿蒙 App。目的是用最小成本确认「AI 加工后的稿子对我到底有没有用」。

## 它能做什么

| 步骤 | 说明 | 需要的 key |
| --- | --- | --- |
| ① 来源 | 粘贴文字稿，或填音频链接自动转写 | 转写需阿里云 key |
| ② 逐字稿 | 显示/编辑文字 | — |
| ③ AI 加工 | 书面化改写 / 提炼总结 / 翻译 | DeepSeek |
| ④ 导出 | 一键下载 `.md` 文件 | — |

> **长稿（如 4 小时播客）会自动分段处理再拼接**，绕开「模型一次输出长度有限」的限制。总结用「先分段提要点、再汇总成稿」。

---

## 怎么跑起来（傻瓜步骤）

### 1. 装 Python
到 https://www.python.org/downloads/ 下载安装（安装时勾选 **Add Python to PATH**）。

### 2. 下载本项目代码
把这个分支的代码下载到电脑某个文件夹（用 GitHub 网页的「Download ZIP」即可），解压。

### 3. 填入你的 key
- 把 `.env.example` 复制一份，改名为 `.env`
- 用记事本打开 `.env`，把你的 DeepSeek key 填在 `DEEPSEEK_API_KEY=` 后面
  （音频转写的阿里云 key 现在可以先空着，等要用再填）

### 4. 安装依赖并启动
打开「终端 / 命令提示符」，进入项目文件夹，依次运行：

```bash
pip install -r requirements.txt
python app.py
```

看到提示后，用浏览器打开： **http://127.0.0.1:8000**

### 5. 开始体验
- 先点「粘贴文字稿」，把任意一段文字贴进第 2 步的框里；
- 点「书面化改写 / 提炼总结 / 翻译」试试效果；
- 满意就点「导出 Markdown」。

---

## key 怎么申请
- **DeepSeek（AI 加工，现在就要）**：https://platform.deepseek.com/ → 控制台 → API keys
- **阿里云 DashScope（音频转写，之后要）**：https://dashscope.console.aliyun.com/ → 开通 → 创建 API-KEY

## 注意
- `.env` 里是你的真实 key，**不要上传到 GitHub**（已在 `.gitignore` 里排除）。
- 音频转写按「音频链接」工作，播客每集的 mp3 链接天然适用；本地文件上传后续再加。
