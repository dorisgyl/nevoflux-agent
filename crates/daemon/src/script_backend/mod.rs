//! 脚本后端契约（P8）：外挂程序与网关之间的接口。
//!
//! 这一层是**传输无关**的——OpenAI 前端与（第二期的）MCP 前端都翻译成同一组
//! 类型再交给脚本。它不认识 HTTP，也不负责执行；执行仍由
//! [`crate::agent::code_mode`] 完成。

pub mod contract;
pub mod entry;
pub mod stream;

pub use contract::{
    OutcomeBody, OutcomeError, ScriptMessage, ScriptOutcome, ScriptParams, ScriptRequest,
    ScriptToolCall, Usage,
};
pub use entry::{build_invocation, detect_entry, to_python_literal, EntryPoint};
pub use stream::{Delta, DeltaSink, FinishPayload};
