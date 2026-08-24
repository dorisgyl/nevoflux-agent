//! 第一次要说话时,决定在哪个后端上说。
//!
//! 选择逻辑本身在 `nevoflux_tts::ep`(纯函数,可测);这里负责它需要的两样脏东西:
//! 问运行时**这台机器上有什么设备**,以及**真的建一次引擎跑一句**来量速度。
//!
//! ## 为什么探测在这里,而不是 daemon 启动时
//!
//! 绝大多数会话不开语音,而一次探测最坏要加载几百兆权重。启动时做,等于让每个
//! 从不说话的用户为语音付这笔钱;放在 `engine()` 的 `OnceLock` 里,只有真的要
//! 出声的那一刻才付,而且一个进程只付一次。
//!
//! ## 赢的那个引擎不会被重建
//!
//! 探测建出来的引擎直接留用。否则「量一次、再建一次」= 两次几百兆的加载,而
//! 第二次除了慢没有任何新信息。

use std::path::Path;
use std::sync::OnceLock;
use std::time::Instant;

use nevoflux_tts::ep::{self, Ep, NoBackend, Outcome, Selection, Trial};
use nevoflux_tts::moss::MossEngine;

/// 探测用的那句话。
///
/// 短(几秒音频),但必须走完整条路:分词、prefill、自回归解码、codec 解码。
/// 拿一句更短的去量,量到的是加载开销而不是合成速度。
const PROBE_TEXT: &str = "你好，这是一次速度测量。";

/// 配置里 `[speech] execution_provider` 的取值。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Setting {
    /// 探测决定。默认。
    Auto,
    /// 钉死一个。只为排障存在 —— 「自动选错了」和「这个后端本身有问题」要能分开。
    Pinned(Ep),
}

impl Setting {
    pub fn parse(s: Option<&str>) -> Setting {
        match s.map(str::trim) {
            None | Some("") | Some("auto") => Setting::Auto,
            Some(other) => match Ep::parse(other) {
                Some(ep) => Setting::Pinned(ep),
                None => {
                    tracing::warn!(
                        target: "speech",
                        value = other,
                        "unknown execution_provider; falling back to auto"
                    );
                    Setting::Auto
                }
            },
        }
    }
}

/// 探测结果,连同赢家的引擎。
pub struct Probed {
    pub selection: Selection,
    pub engine: Option<MossEngine>,
}

/// 跑一句,返回 RTF。
fn measure(engine: &MossEngine) -> Result<f32, String> {
    let voice = engine
        .voices()
        .first()
        .map(|v| v.voice.clone())
        .ok_or_else(|| "引擎里一个音色都没有".to_string())?;
    let started = Instant::now();
    let audio = engine
        .speak(&voice, PROBE_TEXT, 0)
        .map_err(|e| e.to_string())?;
    let seconds = audio.seconds() as f32;
    if seconds <= 0.0 {
        return Err("合成出来的音频长度是零".into());
    }
    Ok(started.elapsed().as_secs_f32() / seconds)
}

/// 选一个后端并把赢家的引擎一起带回来。
///
/// ## 只在有得选的时候才量
///
/// 第一版对每个能建出 session 的候选都跑一次探测合成。代价被低估了:实测在这台
/// 机器上是**加载 9.3 秒 + 探测合成 9.0 秒**,而这 18 秒全落在用户第一句话的
/// 前面 —— 用户的原话是「很久才有声音」。
///
/// 现在的规矩:按顺序建,**第一个建得出来的就是引擎**;只有当它是 GPU 时才额外
/// 量一次 CPU 来对照。理由是这个对照只回答一个问题 ——「GPU 真的比 CPU 快吗」。
/// 第一个建得出来的已经是 CPU 时,这个问题不存在,那次合成就是纯浪费。
///
/// 代价仍在,但落在**该付的那种机器上**:GPU 真能用的机器多花一次加载加两次短
/// 合成,换来「别用一块比 CPU 还慢的显卡」这个结论。
pub fn probe(dir: &Path, threads: usize, setting: Setting, budget: f32) -> Probed {
    let devices = ep::devices();
    for d in &devices {
        tracing::info!(
            target: "speech",
            vendor = %d.vendor,
            ep = %d.ep,
            id = d.id,
            gpu = d.is_gpu,
            "ONNX Runtime device"
        );
    }

    // 钉死时不探测:钉死的意义就是「别问了,就用这个」。它建不出来会在
    // `engine()` 那里如常报错,而不是悄悄换一个 —— 排障时最怕的就是悄悄换。
    if let Setting::Pinned(ep) = setting {
        tracing::info!(target: "speech", %ep, "execution provider pinned by config");
        return Probed {
            selection: Selection {
                ep,
                rtf: f32::NAN,
                trials: Vec::new(),
            },
            engine: None,
        };
    }

    // 第一关:谁建得出 session。建不出的留下原因 —— 「没用上 GPU」与「用上了
    // 但没变快」是两件事,分不清就没法修。
    let mut trials = Vec::new();
    let mut winner = None;
    for candidate in ep::order(ep::has_nvidia(&devices)) {
        match MossEngine::load(dir, threads, candidate) {
            Err(e) => {
                let why = e.to_string();
                tracing::info!(target: "speech", ep = %candidate, why = %why, "backend unavailable");
                trials.push(Trial {
                    ep: candidate,
                    outcome: Outcome::Unavailable(why),
                });
            }
            Ok(engine) => {
                winner = Some((candidate, engine));
                break;
            }
        }
    }

    let Some((first_ep, first_engine)) = winner else {
        let no = NoBackend { trials };
        tracing::error!(target: "speech", why = %no, "no usable inference backend");
        return Probed {
            selection: Selection {
                ep: Ep::Cpu,
                rtf: f32::NAN,
                trials: no.trials,
            },
            engine: None,
        };
    };

    // CPU 已经是第一个建得出来的:没有更慢的对照对象,不必量。
    if first_ep == Ep::Cpu {
        trials.push(Trial {
            ep: Ep::Cpu,
            outcome: Outcome::Available,
        });
        return Probed {
            selection: Selection {
                ep: Ep::Cpu,
                rtf: f32::NAN,
                trials,
            },
            engine: Some(first_engine),
        };
    }

    // GPU 过了门。跟 CPU 比一次 —— 「能用」不等于「更快」,而虚拟机里那块
    // 只报 DX12 却没有算力的适配器正是靠这一步被淘汰的。
    //
    // 已经建好的那个引擎留用,不重建:一次加载是几百兆。
    let mut built = vec![(first_ep, first_engine)];
    let raced = ep::choose(&[first_ep, Ep::Cpu], budget, |candidate| {
        let at = match built.iter().position(|(e, _)| *e == candidate) {
            Some(i) => i,
            None => match MossEngine::load(dir, threads, candidate) {
                Ok(e) => {
                    built.push((candidate, e));
                    built.len() - 1
                }
                Err(e) => {
                    let why = e.to_string();
                    tracing::info!(target: "speech", ep = %candidate, why = %why, "backend unavailable");
                    return Err(why);
                }
            },
        };
        let rtf = measure(&built[at].1)?;
        tracing::info!(target: "speech", ep = %candidate, rtf, "backend measured");
        Ok(rtf)
    });

    match raced {
        Ok(mut selection) => {
            let at = built.iter().position(|(e, _)| *e == selection.ep);
            let engine = at.map(|i| built.swap_remove(i).1);
            // 门那一关的失败记录排在前面,race 的结果接在后面 —— 读的人要的是
            // 完整的一路经过。
            trials.extend(std::mem::take(&mut selection.trials));
            selection.trials = trials;
            Probed { selection, engine }
        }
        Err(no) => {
            // GPU 建出来了却量不出来(合成本身失败),CPU 也没成。把 GPU 留着用,
            // 总比没有声音强,但原因要留下。
            tracing::warn!(target: "speech", why = %no, "backend race produced nothing measurable");
            trials.extend(no.trials);
            let engine = built
                .into_iter()
                .find(|(e, _)| *e == first_ep)
                .map(|(_, e)| e);
            Probed {
                selection: Selection {
                    ep: first_ep,
                    rtf: f32::NAN,
                    trials,
                },
                engine,
            }
        }
    }
}
/// 这个进程最终用的是哪个后端。设置页与排障读它。
static CHOSEN: OnceLock<(Ep, String)> = OnceLock::new();

pub fn record(selection: &Selection) {
    let _ = CHOSEN.set((selection.ep, selection.summary()));
}

/// 探测选中的后端。Kokoro 用它,而不是自己再探一次:两个引擎跑在同一台机器
/// 同一个运行时上,分头得出不同结论只会让「到底用没用 GPU」更难回答。
pub fn chosen_ep() -> Ep {
    CHOSEN.get().map(|(ep, _)| *ep).unwrap_or(Ep::Cpu)
}

/// 一行人话,给界面用。还没探测过时是 `None` —— 「还没说过话」与「用的是 CPU」
/// 是两件事。
pub fn chosen() -> Option<&'static str> {
    CHOSEN.get().map(|(_, s)| s.as_str())
}
