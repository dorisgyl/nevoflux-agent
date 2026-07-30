# 脚本后端契约 v1

headless 服务把 OpenAI 请求翻译成一次**脚本调用**，再把脚本的返回值翻译回
OpenAI 响应。这份文档写给脚本作者。

参考实现：`docs/reference/script-backends/gemini-web.py`（Gemini 网页版）
可复制的函数块：`docs/reference/script-backends/webui_backend.py`

---

## 1. 两个入口

```python
def chat(request):   # 完整契约入口，优先
    ...

def run(task):       # 老入口，只收一个字符串
    ...
```

服务端**静态扫描**脚本源码里有没有 `def chat(`。用扫描而不是运行时探测，是
因为 `try/except NameError` 会把 `chat()` 内部真实的 NameError 一并吞掉，
静默降级到 `run()`，这类 bug 极难定位。

注释里的 `# def chat(...)`、以及 `chat_helper` / `mychat` 这类名字都不算。

## 2. `request` 长什么样

```python
{
  "contract_version": 1,
  "protocol": "openai",              # 或 "mcp"
  "model": "gemini-web",

  "messages": [                      # 完整历史；OpenAI 侧填满
    {"role": "system|user|assistant|tool",
     "content": "压平后的纯文本",       # 恒为 str
     "content_parts": [...],          # 原始内容块，无则 []
     "tool_call_id": "call_1"}        # 仅 tool 角色
  ],
  "arguments": {},                   # MCP tools/call 的原始参数；OpenAI 侧为 {}

  "task": "最后一条非空 user 消息的文本",   # 便利字段，等价于 run(task) 的入参
  "tools": [...],                    # 客户端声明的工具
  "tool_choice": "auto",
  "stream": True,
  "params": {"temperature": 0.7, "max_tokens": 4096},
  "metadata": {"task_id": "task-7"},
}
```

`messages` 与 `arguments` **恒存在**，各自可能是退化形态——脚本不必关心自己
被 OpenAI 前端还是 MCP 前端调用。

**实现约束**：请求以 Python 字面量的形式注入脚本，JSON 的 `true` / `false` /
`null` 会被渲染成 `True` / `False` / `None`。字符串内容不受影响（逐 token
渲染，不是文本替换），所以正文里出现 "true" 这个词是安全的。

## 3. 返回什么

```python
return {"content": "回答正文"}

return {"tool_calls": [
    {"id": "call_1", "name": "read_file", "arguments": {"path": "/tmp/a"}}
]}

return {"error": {"message": "...", "type": "server_error", "code": "..."}}
```

三选一，优先级 `error` > `tool_calls` > `content`。可选字段：

- `finish_reason`：`stop` / `length` / `tool_calls` / `content_filter`。不给则
  按主体推断（有 tool_calls 就是 `tool_calls`，否则 `stop`）。
- `usage`：`{"prompt_tokens": .., "completion_tokens": .., "total_tokens": ..}`

`tool_calls[].arguments` 在契约里是**对象**。OpenAI 线格式要求字符串化的
JSON，这一步由网关完成——脚本**不要**自己 `json.dumps`。

## 4. 流式：`emit_text` / `emit_progress`

```python
emit_text("增量文本")        # → chat.completion.chunk 的 delta.content
emit_progress("正在等待…")   # → SSE 注释帧（客户端忽略，但连接保活）
```

`emit_progress` 是为 MCP 预留的：OpenAI 侧降级成注释帧，MCP 侧将映射到
`notifications/progress`，脚本无需改动。

非流式请求下 `emit_text` 照样可用，增量会被拼起来。**显式 `return` 的
`content` 优先，没有才用拼接结果。**

没有增量通道时（例如 CLI 直接跑脚本），两个函数是空操作，不会因"函数未
定义"而崩。

## 5. legacy 行为

`run(task)` 的返回值**按原样处理**：字符串取本身，其它 `str()` 化。不做
`reply` / `text` 之类的启发式提取——要干净输出就迁移到 `chat()`。

这意味着老脚本返回 `{"ok": true, "reply": "正文"}` 时，客户端拿到的是整坨
JSON 字符串，而不是 `正文`。

## 6. 模型路由

```
NEVOFLUX_OPENAI_MODELS='gemini-web=/opt/nevoflux/gemini-web.py,agent='
```

- 值为空 = 该模型不走脚本，走真 agent 循环
- `GET /v1/models` 列出全部键
- 配了这个变量后，未知 model 名返回 404 `model_not_found`

**未配置时**回退单后端模式：广播 `NEVOFLUX_HEADLESS_MODEL`（默认
`nevoflux-script`），脚本取 `NEVOFLUX_HEADLESS_SCRIPT`，且**接受任意 model
名**——发 `"model":"gpt-4"` 的存量客户端不受影响。

## 7. 超时与取消

预算只有一个来源：任务墙钟 `NEVOFLUX_WALL_CLOCK_SECS`。沙箱预算由它推导
（减 10 秒余量），不再吃执行器的 180 秒默认值。

流式请求的客户端断开时，取消标志被置位，脚本在**下一个工具调用边界**停止。
所以长轮询要用 `browser_wait_for` 这类工具调用来推进，纯计算的死循环无法
被取消。

## 8. 错误如何映射到 HTTP

| 情况 | HTTP | type / code |
|---|---|---|
| 请求体畸形、无 user 消息 | 400 | `invalid_request_error` |
| model 不在路由表 | 404 | `model_not_found` |
| 脚本抛异常 / 沙箱失败 | 502 | `server_error` / `script_error` |
| 脚本返回 `{"error": ...}` | 502 | 用脚本给的 code |
| 超时 | 504 | `timeout` |

流已经开始之后（SSE 发出响应头后状态码不可改）：发一个 `{"error": {...}}`
数据帧再 `[DONE]` 收尾，不粗暴断连——断连在客户端只表现为连接重置，诊断
信息全失。
