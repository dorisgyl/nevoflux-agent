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

#[cfg(test)]
mod gpu_choice_tests {
    use super::*;

    /// `GPU_CHOICE` 是进程级的,所以这些测试不能并行跑。
    fn exclusive() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// 关掉要压过一切 —— 包括 config.toml 里钉死的 GPU。
    ///
    /// 这个开关存在的全部理由就是显卡出问题时有一条一定回得去的路;如果它还要
    /// 服从配置,那条路就不是一定回得去。
    #[test]
    fn refusing_the_gpu_pins_the_cpu_whatever_the_config_says() {
        let _g = exclusive();
        set_gpu_allowed(Some(false));
        for cfg in [None, Some("auto"), Some("cuda"), Some("directml")] {
            assert_eq!(Setting::resolve(cfg), Setting::Pinned(Ep::Cpu), "{cfg:?}");
        }
        set_gpu_allowed(None);
    }

    /// 允许**不等于**强制。打开之后仍然由探测决定 —— 这台机器上 CUDA 实测
    /// 2.63x 对 CPU 1.17x,硬选 GPU 会让它更慢。
    #[test]
    fn allowing_the_gpu_leaves_the_choice_to_the_probe() {
        let _g = exclusive();
        set_gpu_allowed(Some(true));
        assert_eq!(Setting::resolve(None), Setting::Auto);
        assert_eq!(Setting::resolve(Some("auto")), Setting::Auto);
        // 配置钉死的仍然生效:排障要能钉。
        assert_eq!(Setting::resolve(Some("cpu")), Setting::Pinned(Ep::Cpu));
        set_gpu_allowed(None);
    }

    /// 真实合成失败之后,那个后端不该再被选中 —— 哪怕开关还开着、配置里还钉着它。
    ///
    /// 回归的是一次真实故障:DirectML 通过了探测、赢了裁判,然后每一句话都在
    /// `ConvTranspose` 上报 80070057。当时唯一的出路是有人去读日志再去关开关;
    /// 这条规则的意义就是不再需要那个人。
    #[test]
    fn a_backend_that_failed_for_real_is_not_chosen_again() {
        let _g = exclusive();
        clear_demotion();
        set_gpu_allowed(Some(true));
        assert_eq!(Setting::resolve(Some("directml")), Setting::Pinned(Ep::DirectMl));

        demote(Ep::DirectMl, "ConvTranspose: 80070057");
        // 开关仍然开着、配置仍然钉着 directml,但事实已经出来了。
        assert_eq!(Setting::resolve(Some("directml")), Setting::Pinned(Ep::Cpu));
        assert_eq!(Setting::resolve(None), Setting::Pinned(Ep::Cpu));
        // 原因要留着 ——「为什么没用 GPU」得答得出来。
        let (ep, why) = demoted().expect("记下来了");
        assert_eq!(ep, Ep::DirectMl);
        assert!(why.contains("80070057"), "{why}");

        clear_demotion();
        set_gpu_allowed(None);
    }

    /// CPU 是地板,降级它没有意义 —— 没有地方可退。
    #[test]
    fn the_cpu_is_never_demoted() {
        let _g = exclusive();
        clear_demotion();
        demote(Ep::Cpu, "不该被记下");
        assert!(demoted().is_none());
    }

    /// 没表态时和从前完全一样,由配置说了算。
    #[test]
    fn saying_nothing_leaves_the_config_in_charge() {
        let _g = exclusive();
        set_gpu_allowed(None);
        assert_eq!(gpu_allowed(), None);
        assert_eq!(Setting::resolve(None), Setting::parse(None));
        assert_eq!(Setting::resolve(Some("cpu")), Setting::parse(Some("cpu")));
    }
}

/// 探测用的那句话。
///
/// 短(几秒音频),但必须走完整条路:分词、prefill、自回归解码、codec 解码。
/// 拿一句更短的去量,量到的是加载开销而不是合成速度。
const PROBE_TEXT: &str = "你好，这是一次速度测量。";

/// 第二句,明显更长。**它测的不是速度,是形状。**
///
/// 这条存在的理由是一次真实的故障:DirectML 通过了单句探测、在裁判那里赢了
/// CPU,然后**每一句真实的话都崩** —— `ConvTranspose` 报 80070057(参数错误)。
/// 用户听到的是全程无声,而日志里每句一条错误。
///
/// 原因写在 ORT 自己的文档里:DirectML 对**动态输入尺寸**支持不好。一句固定
/// 短句只走出一种形状,过了不说明别的形状也过。而真实的句子长短不一。
///
/// 所以门要用两种长度去敲。两次都成才算这个后端能用,这比「能建出 session」强
/// 得多。
///
/// **短,而且只给 GPU 候选跑。** 第一版这句写了 53 个字,而 CPU 上一次探测合成
/// 按 rtf=2.89 算就是约 49 秒 —— 设置页那句「Checking which engine will speak…」
/// 要等 77 秒,大头就是它。而 CPU **从来没有形状问题**:这道门是为 DirectML 那
/// 类对动态尺寸挑剔的后端设的,让地板陪着跑一遍最贵的检查,纯是浪费。
const PROBE_TEXT_LONG: &str =
    "换一个长度再算一次。";

/// 设置页里那个「用 GPU」开关的答案,进程级。
///
/// 0 = 没表态(由 config.toml 决定),1 = 允许,2 = 拒绝。
///
/// 是进程级而不是每会话,因为引擎本身就是进程级的 `OnceLock` —— 一台机器上
/// 只会探测一次、只会有一个后端在跑,把这个偏好做成每会话的会造出一个不存在
/// 的选择。原子量而不是锁:读它的地方在合成的热路径上。
static GPU_CHOICE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// 运行期被判定不能用的后端。空串表示没有。
///
/// 与 [`GPU_CHOICE`] 分开:那个是用户的意愿,这个是机器给出的事实,而事实要能
/// 压过意愿 —— 一个每句话都崩的后端,不该因为开关还开着就继续被选中。
static DEMOTED: std::sync::Mutex<Option<(Ep, String)>> = std::sync::Mutex::new(None);

/// 记下「这个后端真用起来是坏的」。
///
/// 来自一次真实故障:DirectML 通过了探测、在裁判那里赢了 CPU,然后每一句真实的
/// 话都在 `ConvTranspose` 上报 80070057,用户听到的是全程无声。门已经加强到用
/// 两种形状去敲(见 `PROBE_TEXT_LONG`),但门再严也只是抽样 —— 真正的证据是
/// 合成本身失败了,而那个证据出现时就该有人认账。
///
/// 只对非 CPU 后端有意义:CPU 是地板,它坏了没有地方可退。
pub fn demote(ep: Ep, why: impl Into<String>) {
    if ep == Ep::Cpu {
        return;
    }
    let why = why.into();
    let mut slot = DEMOTED.lock().unwrap_or_else(|e| e.into_inner());
    if slot.is_none() {
        tracing::warn!(
            target: "speech",
            %ep, why = %why,
            "backend failed on real input; falling back to the CPU from here on"
        );
        *slot = Some((ep, why));
    }
}

/// 忘掉降级。给测试用,也给「换了驱动想再试一次」用 —— 降级是这个进程得出的
/// 结论,不是对这台机器的判决,重启就该重新给一次机会。
pub fn clear_demotion() {
    *DEMOTED.lock().unwrap_or_else(|e| e.into_inner()) = None;
}

/// 被降级的那个后端,连同原因。原因要留着 —— 「为什么没用 GPU」得答得出来。
pub fn demoted() -> Option<(Ep, String)> {
    DEMOTED
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
}

/// 记下用户的选择。`None` 表示没表态。
pub fn set_gpu_allowed(v: Option<bool>) {
    let n = match v {
        None => 0,
        Some(true) => 1,
        Some(false) => 2,
    };
    GPU_CHOICE.store(n, std::sync::atomic::Ordering::Relaxed);
}

/// 用户的选择,没表态时是 `None`。
pub fn gpu_allowed() -> Option<bool> {
    match GPU_CHOICE.load(std::sync::atomic::Ordering::Relaxed) {
        1 => Some(true),
        2 => Some(false),
        _ => None,
    }
}

/// 配置里 `[speech] execution_provider` 的取值。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Setting {
    /// 探测决定。默认。
    Auto,
    /// 钉死一个。只为排障存在 —— 「自动选错了」和「这个后端本身有问题」要能分开。
    Pinned(Ep),
}

impl Setting {
    /// 配置里写的,叠上用户在设置页里的表态。
    ///
    /// 关掉 GPU 要能压过一切:那个开关存在的全部理由,就是显卡或驱动出问题时
    /// 有一条一定回得去的路。所以「拒绝」直接钉 CPU,不再看配置。
    ///
    /// 「允许」不等于「强制用 GPU」—— 它只是不阻止,余下交给探测。硬把一个
    /// 比 CPU 还慢的后端选上去(实测 CUDA 在这台机器上是 2.63x 对 1.17x),
    /// 是把一个开关变成一个陷阱。
    pub fn resolve(cfg: Option<&str>) -> Setting {
        // 机器给出的事实压过一切。一个已经被真实合成证伪的后端,不该因为开关
        // 还开着、或者配置里钉着它,就再被选一次。
        if demoted().is_some() {
            return Setting::Pinned(Ep::Cpu);
        }
        match gpu_allowed() {
            Some(false) => Setting::Pinned(Ep::Cpu),
            _ => Setting::parse(cfg),
        }
    }

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
fn measure(engine: &MossEngine, ep: Ep) -> Result<f32, String> {
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
    let rtf = started.elapsed().as_secs_f32() / seconds;

    // 换一种长度再来一次 —— 只给非 CPU 后端。计时到此为止:这一次问的是
    // 「算不算得出来」,不是「多快」,所以它不进 RTF。
    if ep != Ep::Cpu {
        let second = engine
            .speak(&voice, PROBE_TEXT_LONG, 0)
            .map_err(|e| format!("换一种输入长度就失败了:{e}"))?;
        if second.seconds() <= 0.0 {
            return Err("换一种输入长度之后合成出来的音频长度是零".into());
        }
    }

    Ok(rtf)
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
        let rtf = measure(&built[at].1, candidate)?;
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
