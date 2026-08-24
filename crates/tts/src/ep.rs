//! 选一个 ONNX 执行提供者(EP),按实测,不按写死的表。
//!
//! ## 为什么不是一张硬件表
//!
//! 「有 N 卡就用 CUDA」这类规则要靠一张表活着,而表会过期:缺 cuDNN 的 N 卡、
//! 虚拟机里那块只报 DX12 却没有算力的显示适配器、驱动比运行时新一代的卡 ——
//! 每一种都能让规则挑中一个**能初始化但更慢**的后端。而这条链路上「更慢」不是
//! 一句警告,是用户听到的断断续续。
//!
//! 所以这里三层,各管一件事:
//!
//! 1. **排序**(便宜的提示):运行时自己报告有哪些设备([`devices`]),据此决定
//!    先试谁。排错了也只是多试一个。
//! 2. **门**:候选必须真的建得出 session。缺 dll、缺驱动、显卡不支持,都在这里
//!    被挡掉,而且**挡掉的原因要留下来** —— 「没用上 GPU」和「用上了但没变快」
//!    是两件事,分不清就没法修。
//! 3. **裁判**:过了门的候选各跑一小段真合成,量 RTF,谁快用谁。CPU 永远在候选
//!    里,所以「GPU 反而更慢」会自动被淘汰,不需要谁去预判。
//!
//! 这与仓库里已有的判据同构:MOSS 快不快本来就是量出来的(`measured_rtf` /
//! `rtf_budget` / 设置页的 Re-measure),这里只是把同一件事多做一层 —— 量的从
//! 「这台机器快不快」变成「这台机器上哪个后端最快」。

use std::fmt;

/// 一个执行提供者。
///
/// 只列真正接了线的三个。加一个新的(CoreML、ROCm)是加一个 variant 加一条
/// 注册分支,不需要动选择逻辑 —— 那正是这一层存在的意义。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ep {
    Cpu,
    DirectMl,
    Cuda,
}

impl Ep {
    pub fn name(self) -> &'static str {
        match self {
            Ep::Cpu => "cpu",
            Ep::DirectMl => "directml",
            Ep::Cuda => "cuda",
        }
    }

    /// 配置里写死用哪个时的解析。`auto` 交给探测。
    pub fn parse(s: &str) -> Option<Ep> {
        match s.trim().to_ascii_lowercase().as_str() {
            "cpu" => Some(Ep::Cpu),
            "directml" | "dml" => Some(Ep::DirectMl),
            "cuda" => Some(Ep::Cuda),
            _ => None,
        }
    }
}

impl fmt::Display for Ep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// 一个候选的下场。三种,而且不能合并:补法完全不同。
#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    /// 建不出 session:缺 provider dll、缺 CUDA/cuDNN、驱动太老、拿不到显示适配器。
    Unavailable(String),
    /// 建得出,但没跟谁比过速度 —— 没有对照的必要时就不比。
    ///
    /// 「没量」和「量了是 0」必须分得开:前者是省下了一次几秒的合成,后者是坏了。
    Available,
    /// 建得出,而且量过:这是 RTF(合成 1 秒音频要花几秒)。
    Measured(f32),
}

/// 一个候选试过之后留下的记录。上报与排障要的就是这个 —— 「为什么没用 GPU」
/// 必须能回答。
#[derive(Debug, Clone, PartialEq)]
pub struct Trial {
    pub ep: Ep,
    pub outcome: Outcome,
}

impl Trial {
    pub fn describe(&self) -> String {
        match &self.outcome {
            Outcome::Unavailable(why) => format!("{}: 不可用({why})", self.ep),
            Outcome::Available => format!("{}: 可用(未比较)", self.ep),
            Outcome::Measured(rtf) => format!("{}: {rtf:.2}x", self.ep),
        }
    }
}

/// 选中的结果,连同一路上试过的所有候选。
#[derive(Debug, Clone, PartialEq)]
pub struct Selection {
    pub ep: Ep,
    pub rtf: f32,
    pub trials: Vec<Trial>,
}

impl Selection {
    /// 一行人话,进日志、进设置页、进 `voice_done`。
    pub fn summary(&self) -> String {
        let rest: Vec<String> = self.trials.iter().map(Trial::describe).collect();
        let head = if self.rtf.is_finite() {
            format!("{} @ {:.2}x", self.ep, self.rtf)
        } else {
            // 没比过就别报一个数字。`NaN @ 0.00x` 会被当成「量出来是零」。
            format!("{}", self.ep)
        };
        format!("{head}({})", rest.join("; "))
    }
}

/// 一个候选都没成 —— 连 CPU 都建不出 session。这时候不该猜,该报。
#[derive(Debug, Clone, PartialEq)]
pub struct NoBackend {
    pub trials: Vec<Trial>,
}

impl fmt::Display for NoBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let rest: Vec<String> = self.trials.iter().map(Trial::describe).collect();
        write!(f, "没有可用的推理后端({})", rest.join("; "))
    }
}

/// 运行时报告的一个设备。
///
/// 这是「这台机器上有什么」的答案来源。问的是 ONNX Runtime 自己枚举出来的东西,
/// 而不是我们维护的一张厂商表 —— 表会过期,运行时不会。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceInfo {
    pub vendor: String,
    pub ep: String,
    pub id: u32,
    pub is_gpu: bool,
}

/// 枚举运行时看得见的设备。环境起不来时返回空 —— 那本身也是一种答案,而且
/// 后面的门会把每个候选都挡掉,原因不会丢。
pub fn devices() -> Vec<DeviceInfo> {
    // `commit()` 只是「这次调用有没有装上全局环境」,已经装过就返回 false ——
    // 那不是失败,所以不看它的返回值,直接去拿当前环境。
    ort::environment::init().commit();
    let Ok(env) = ort::environment::current() else {
        return Vec::new();
    };
    env.devices()
        .map(|d| DeviceInfo {
            vendor: d.vendor().unwrap_or_default().to_string(),
            ep: d.ep().unwrap_or_default().to_string(),
            id: d.id(),
            is_gpu: matches!(d.ty(), ort::memory::DeviceType::GPU),
        })
        .collect()
}

/// 设备里有没有 NVIDIA 的 GPU。只用来排候选顺序 —— 有卡不等于 CUDA 能用。
pub fn has_nvidia(devices: &[DeviceInfo]) -> bool {
    devices
        .iter()
        .any(|d| d.is_gpu && d.vendor.to_ascii_lowercase().contains("nvidia"))
}

/// 试的顺序。
///
/// 只是提示,不是判决:真正决定的是门与裁判。这里唯一的目的是**别把最可能赢的
/// 那个放到最后** —— 第一个就达标时后面的候选根本不会被加载,而加载一次 MOSS
/// 是几百兆的事。
///
/// `has_nvidia` 由运行时报告的设备厂商得出(见 [`devices`]),不是猜的。
/// **每个后端都在候选里,提示只决定先后。**
///
/// 第一版写成「有 N 卡才把 CUDA 放进候选」,那是把提示变成了准入门槛,而它当场
/// 就咬人了:换上 ORT 的 CUDA 版运行时之后,`devices()` 只报告一个 Intel CPU
/// —— 设备枚举在不同的运行时构建里报出来的东西不一样,而那台机器上明明插着一块
/// T4。结果 CUDA 一次都没被试过,日志里也看不出为什么。
///
/// 能不能用由门决定(建不建得出 session),快不快由实测决定。提示错了最多是多试
/// 一个候选,而它**永远不该让某个后端连试的机会都没有**。
pub fn order(has_nvidia: bool) -> Vec<Ep> {
    // CPU 恒在最后:它是地板,不是竞争者。
    if has_nvidia {
        vec![Ep::Cuda, Ep::DirectMl, Ep::Cpu]
    } else {
        vec![Ep::DirectMl, Ep::Cuda, Ep::Cpu]
    }
}

/// 按顺序试,挑一个。
///
/// `trial` 建 session、跑一小段、返回 RTF;建不出就返回失败原因。
///
/// **第一个达标就停**:这不是省事,是省一次几百兆的模型加载。达标的定义是
/// `rtf <= budget`,与 MOSS/Kokoro 回落用的是同一个预算。
///
/// 一个都不达标时选**实测最快**的那个,而不是顺序上的第一个 —— 顺序是提示,
/// 实测是证据,证据在就不该再听提示的。
pub fn choose(
    order: &[Ep],
    budget: f32,
    mut trial: impl FnMut(Ep) -> Result<f32, String>,
) -> Result<Selection, NoBackend> {
    let mut trials = Vec::new();
    let mut best: Option<(Ep, f32)> = None;

    for &ep in order {
        match trial(ep) {
            Err(why) => trials.push(Trial {
                ep,
                outcome: Outcome::Unavailable(why),
            }),
            Ok(rtf) => {
                trials.push(Trial {
                    ep,
                    outcome: Outcome::Measured(rtf),
                });
                if best.is_none_or(|(_, b)| rtf < b) {
                    best = Some((ep, rtf));
                }
                if rtf <= budget {
                    break;
                }
            }
        }
    }

    match best {
        Some((ep, rtf)) => Ok(Selection { ep, rtf, trials }),
        None => Err(NoBackend { trials }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 达标就停 —— 后面的候选一个都不该被加载。
    #[test]
    fn a_candidate_within_budget_ends_the_search() {
        let mut tried = Vec::new();
        let sel = choose(&[Ep::Cuda, Ep::DirectMl, Ep::Cpu], 0.85, |ep| {
            tried.push(ep);
            Ok(0.30)
        })
        .unwrap();
        assert_eq!(sel.ep, Ep::Cuda);
        assert_eq!(tried, vec![Ep::Cuda], "达标之后不该再加载别的后端");
    }

    /// 建不出来的候选要留下**原因**。「没用上 GPU」和「用上了没变快」是两件事。
    #[test]
    fn an_unavailable_candidate_records_why_and_moves_on() {
        let sel = choose(&[Ep::Cuda, Ep::Cpu], 0.85, |ep| match ep {
            Ep::Cuda => Err("cudart64_12.dll not found".into()),
            _ => Ok(0.50),
        })
        .unwrap();
        assert_eq!(sel.ep, Ep::Cpu);
        assert_eq!(
            sel.trials[0].outcome,
            Outcome::Unavailable("cudart64_12.dll not found".into())
        );
        assert!(
            sel.summary().contains("cudart64_12.dll"),
            "{}",
            sel.summary()
        );
    }

    /// 全都超预算:选实测最快的,而不是顺序第一个。
    #[test]
    fn when_nothing_meets_the_budget_the_fastest_measured_wins() {
        let sel = choose(&[Ep::Cuda, Ep::DirectMl, Ep::Cpu], 0.85, |ep| match ep {
            Ep::Cuda => Ok(2.4),
            Ep::DirectMl => Ok(1.1),
            Ep::Cpu => Ok(1.6),
        })
        .unwrap();
        assert_eq!(sel.ep, Ep::DirectMl);
        assert_eq!(sel.trials.len(), 3, "都不达标时三个都要试过");
    }

    /// GPU 能用但比 CPU 慢 —— 这正是虚拟机里那块显示适配器会干的事。
    #[test]
    fn a_gpu_that_is_slower_than_the_cpu_loses() {
        let sel = choose(&[Ep::DirectMl, Ep::Cpu], 0.85, |ep| match ep {
            Ep::DirectMl => Ok(3.0),
            Ep::Cpu => Ok(1.5),
            Ep::Cuda => unreachable!("不在候选里"),
        })
        .unwrap();
        assert_eq!(sel.ep, Ep::Cpu);
    }

    /// 连 CPU 都建不出来:明确报错,不要猜一个回去。
    #[test]
    fn nothing_usable_is_an_error_not_a_guess() {
        let err = choose(&[Ep::Cuda, Ep::Cpu], 0.85, |_| Err("boom".into())).unwrap_err();
        assert_eq!(err.trials.len(), 2);
        assert!(err.to_string().contains("boom"), "{err}");
    }

    /// 没有 N 卡时不该白试 CUDA —— 提示层只做这一件事。
    #[test]
    fn ordering_puts_the_likely_winner_first() {
        assert_eq!(order(true), vec![Ep::Cuda, Ep::DirectMl, Ep::Cpu]);
        assert_eq!(order(false), vec![Ep::DirectMl, Ep::Cuda, Ep::Cpu]);
    }

    /// 提示只排序,不准入。
    ///
    /// 回归:第一版在没枚举到 N 卡时把 CUDA 从候选里删掉,而 ORT 的 CUDA 版
    /// 运行时在一台插着 T4 的机器上只报告了一个 Intel CPU —— 于是 CUDA 一次都
    /// 没被试过,且日志里看不出为什么。能不能用是门说了算,不是提示。
    #[test]
    fn every_backend_stays_a_candidate_whatever_the_hint_says() {
        for hint in [true, false] {
            let o = order(hint);
            for ep in [Ep::Cuda, Ep::DirectMl, Ep::Cpu] {
                assert!(o.contains(&ep), "hint={hint} 时 {ep} 被挡在候选之外");
            }
            assert_eq!(o.last(), Some(&Ep::Cpu), "CPU 永远兜底,排最后");
        }
    }

    /// 「没量过」要和「量出来是 0」分得开 —— 前者是省下了一次几秒的合成
    /// (第一个建得出来的就是 CPU 时没有对照对象),后者是坏了。
    #[test]
    fn an_unmeasured_winner_does_not_report_a_number() {
        let sel = Selection {
            ep: Ep::Cpu,
            rtf: f32::NAN,
            trials: vec![
                Trial {
                    ep: Ep::Cuda,
                    outcome: Outcome::Unavailable("providers_cuda.dll missing".into()),
                },
                Trial {
                    ep: Ep::Cpu,
                    outcome: Outcome::Available,
                },
            ],
        };
        let line = sel.summary();
        assert!(!line.contains("0.00x"), "别把没量过报成 0:{line}");
        assert!(line.contains("cpu"), "{line}");
        assert!(line.contains("providers_cuda.dll"), "原因要留下:{line}");
    }

    #[test]
    fn config_can_pin_a_backend_for_diagnosis() {
        assert_eq!(Ep::parse("cuda"), Some(Ep::Cuda));
        assert_eq!(Ep::parse("DML"), Some(Ep::DirectMl));
        assert_eq!(Ep::parse(" cpu "), Some(Ep::Cpu));
        assert_eq!(Ep::parse("auto"), None, "auto 交给探测,不是一个 EP");
    }
}
