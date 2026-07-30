# Gemini 网页版后端（脚本契约 v1）
#
# fixed-flow.py 的契约版：入口从 `run(task)` 换成 `chat(request)`，返回值从
# `{"ok": ..., "reply": ...}` 换成 `{"content": ...}`，并在等待期间轮询页面
# 吐出增量，让客户端能逐字看到回答。
#
# 用法：NEVOFLUX_OPENAI_MODELS='gemini-web=/opt/nevoflux/gemini-web.py'
#
# 选择器随 Google 改版会腐烂——出错时看返回的 `message` 里的步骤号定位。

# ---- 以下四个函数抄自 webui_backend.py 模板（Monty 无 import） ----------------


def is_err(r):
    return isinstance(r, dict) and r.get("__tool_error")


def fail(step, label, err):
    detail = err.get("error", "failed") if isinstance(err, dict) else str(err)
    return {
        "error": {
            "message": "step %s (%s) failed: %s" % (step, label, detail),
            "type": "server_error",
            "code": "script_step_failed",
        }
    }


def slice_reply(markdown, start_marker, end_markers):
    if start_marker not in markdown:
        return ""
    after = markdown.split(start_marker, 1)[1]
    for em in end_markers:
        if em in after:
            after = after.split(em, 1)[0]
    return after.strip()


# UI 残渣标记：从最具体到最宽泛，否则短标记会在正文里误命中
END_MARKERS = [
    "\n\nPro\n\nGemini is AI",
    "\nGemini is AI and can make mistakes.",
    "\nEdit prompt\n",
    "\nGood response\n",
    "\nBad response\n",
    "\nGoogle apps",
    "\nUse microphone",
    "\nNew chat\n",
    "\nSearch chats\n",
]

INPUT_SELECTORS = [
    "[aria-label='Enter a prompt for Gemini']",
    "rich-textarea div[contenteditable]",
    ".ql-editor",
]

SEND_SELECTORS = ["[aria-label='Send message']", "button[aria-label='Send message']"]
DONE_SELECTORS = ["[aria-label='Good response']", "button[aria-label='Good response']"]


def chat(request):
    prompt = request.get("task", "")
    if not prompt:
        return {
            "error": {
                "message": "empty prompt: the last user message carries the task",
                "type": "invalid_request_error",
                "code": "empty_prompt",
            }
        }

    emit_progress("打开 Gemini")
    nav = browser_navigate(url="https://gemini.google.com/app")
    if is_err(nav):
        return fail(1, "navigate to gemini.google.com/app", nav)
    tab = nav["tab_id"]

    # navigate 打开的是**非活动**标签页，要激活后交互才会落在它上面
    browser_activate_tab(tab_id=tab)

    emit_progress("新建对话")
    browser_wait_for(selector="a[aria-label='New chat']", state="visible", timeout_ms=10000, tab_id=tab)
    r = browser_click(selector="a[aria-label='New chat']", tab_id=tab)
    if is_err(r):
        browser_click(selector="a[href='/app']", tab_id=tab)

    browser_wait_for(selector=INPUT_SELECTORS[0], state="visible", timeout_ms=8000, tab_id=tab)

    emit_progress("填入提示词")
    filled = None
    for sel in INPUT_SELECTORS:
        r = browser_input(selector=sel, text=prompt, mode="fill", verify=True, tab_id=tab)
        if not is_err(r):
            filled = sel
            break
    if filled is None:
        return fail(4, "fill prompt into editor", "all selectors failed")

    emit_progress("发送")
    browser_wait_for(selector=SEND_SELECTORS[0], state="visible", timeout_ms=3000, tab_id=tab)
    sent = None
    for sel in SEND_SELECTORS:
        r = browser_click(selector=sel, tab_id=tab)
        if not is_err(r):
            sent = sel
            break
    if sent is None:
        return fail(5, "click Send message", "all selectors failed")

    emit_progress("等待回答")
    # 轮询而不是干等：网页版逐字渲染，差量就是新吐出的字，这样客户端
    # 能看到流式效果而不是几十秒静默后一次性蹦出全文。
    sent_len = 0
    reply = ""
    for _ in range(90):
        done = False
        for sel in DONE_SELECTORS:
            probe = browser_wait_for(selector=sel, state="visible", timeout_ms=1000, tab_id=tab)
            if not is_err(probe):
                done = True
                break

        md = browser_get_markdown(tab_id=tab)
        if is_err(md):
            return fail(7, "extract response text", md)
        text = md.get("markdown", "") if isinstance(md, dict) else str(md)

        reply = slice_reply(text, "## Gemini said", END_MARKERS)
        if not reply:
            # 回退：没有 "## Gemini said" 标记时，从用户提示词之后开始取
            if "You said" in text:
                after_user = text.split("You said", 1)[1]
                idx = after_user.find(prompt.strip())
                after_prompt = after_user[idx + len(prompt.strip()):] if idx >= 0 else after_user
                for em in END_MARKERS:
                    if em in after_prompt:
                        after_prompt = after_prompt.split(em, 1)[0]
                reply = after_prompt.strip()

        if len(reply) > sent_len:
            emit_text(reply[sent_len:])
            sent_len = len(reply)

        if done:
            break

    return {"content": reply}
