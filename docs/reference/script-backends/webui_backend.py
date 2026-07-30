# 网页版 LLM 后端模板（脚本契约 v1）
#
# Monty 沙箱没有模块系统，`import` 不可用——所以这不是一个库，而是**一段可以
# 复制粘贴的函数块**。新写一个后端时，把这些函数抄进你的脚本，改选择器即可。
#
# 契约要点（完整说明见 docs/reference/script-backend-contract.md）：
#   def chat(request) -> {"content": str} | {"tool_calls": [...]} | {"error": {...}}
#   emit_text(str)      正文增量，客户端逐字看到
#   emit_progress(str)  进度提示，不进正文（OpenAI 侧是 SSE 注释帧）


def is_err(r):
    """工具调用返回的是错误信封而不是结果时为 True。"""
    return isinstance(r, dict) and r.get("__tool_error")


def fail(step, label, err, tab_id=None):
    """构造契约错误。`step`/`label` 让失败可定位——网页版后端会随对方
    改版而腐烂，出错时说清楚死在哪一步比什么都重要。"""
    detail = err.get("error", "failed") if isinstance(err, dict) else str(err)
    return {
        "error": {
            "message": "step %s (%s) failed: %s" % (step, label, detail),
            "type": "server_error",
            "code": "script_step_failed",
        }
    }


def click_first(selectors, tab_id):
    """依次尝试一组选择器，返回命中的那个；全失败返回 None。
    网页版 UI 经常小改，单一选择器太脆。"""
    for sel in selectors:
        r = browser_click(selector=sel, tab_id=tab_id)
        if not is_err(r):
            return sel
    return None


def fill_first(selectors, text, tab_id):
    """依次尝试一组输入框选择器，返回命中的那个；全失败返回 None。"""
    for sel in selectors:
        r = browser_input(selector=sel, text=text, mode="fill", verify=True, tab_id=tab_id)
        if not is_err(r):
            return sel
    return None


def slice_reply(markdown, start_marker, end_markers):
    """从整页 markdown 里切出回答正文。

    `end_markers` 按**从最具体到最宽泛**排列——先削掉长而独特的 UI 残渣，
    再削短的，否则短标记会在正文里误命中（比如正文里出现单独一行 "Pro"）。
    """
    if start_marker not in markdown:
        return ""
    after = markdown.split(start_marker, 1)[1]
    for em in end_markers:
        if em in after:
            after = after.split(em, 1)[0]
    return after.strip()


def poll_reply(tab_id, start_marker, end_markers, done_selectors, max_rounds=60):
    """轮询页面，把新增的正文作为增量吐出去，直到回答完成。

    返回最终正文。增量靠"本轮切出的正文比上轮长"来判断——网页版是逐字
    渲染的，所以差量就是新吐出的字。

    `done_selectors` 里任何一个出现即视为回答结束（如"点赞"按钮）。
    轮询而不是干等，是这个后端能做到真流式的唯一途径。
    """
    sent = ""
    reply = ""
    for _ in range(max_rounds):
        for sel in done_selectors:
            probe = browser_wait_for(selector=sel, state="visible", timeout_ms=1000, tab_id=tab_id)
            if not is_err(probe):
                md = browser_get_markdown(tab_id=tab_id)
                text = md.get("markdown", "") if isinstance(md, dict) else str(md)
                reply = slice_reply(text, start_marker, end_markers)
                if len(reply) > len(sent):
                    emit_text(reply[len(sent):])
                return reply

        md = browser_get_markdown(tab_id=tab_id)
        text = md.get("markdown", "") if isinstance(md, dict) else str(md)
        reply = slice_reply(text, start_marker, end_markers)
        if len(reply) > len(sent):
            emit_text(reply[len(sent):])
            sent = reply
    return reply
