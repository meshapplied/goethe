// Watchdog liveness policy (injectable clock, unit-tested).

pub const CHEAP_FAIL_DEAD_SECS: i64 = 30;
pub const DEEP_INTERVAL_SECS: i64 = 60;
pub const DEEP_FAIL_STREAK_MAX: u32 = 3;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WatchdogAction {
    None,
    MarkDead { reason: &'static str },
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LivenessState {
    pub cheap_fail_since: Option<i64>,
    pub deep_fail_streak: u32,
    pub last_deep_at: i64,
    pub cheap_ok_once: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProbeInput {
    pub now: i64,
    pub pid_alive: bool,
    pub cheap_ok: bool,
    pub deep_ok: Option<bool>,
}

pub fn step(state: &mut LivenessState, input: ProbeInput) -> WatchdogAction {
    if !input.pid_alive {
        return WatchdogAction::MarkDead {
            reason: "process exited",
        };
    }
    if input.cheap_ok {
        state.cheap_ok_once = true;
        state.cheap_fail_since = None;
    } else if state.cheap_ok_once {
        if state.cheap_fail_since.is_none() {
            state.cheap_fail_since = Some(input.now);
        } else if input.now - state.cheap_fail_since.unwrap_or(input.now) >= CHEAP_FAIL_DEAD_SECS {
            return WatchdogAction::MarkDead {
                reason: "cheap probe failed 30s",
            };
        }
    }
    let do_deep = input.now - state.last_deep_at >= DEEP_INTERVAL_SECS;
    if do_deep {
        state.last_deep_at = input.now;
        if let Some(ok) = input.deep_ok {
            if ok {
                state.deep_fail_streak = 0;
            } else if state.cheap_ok_once {
                state.deep_fail_streak += 1;
                if state.deep_fail_streak >= DEEP_FAIL_STREAK_MAX {
                    return WatchdogAction::MarkDead {
                        reason: "deep health failed 3x",
                    };
                }
            }
        }
    }
    WatchdogAction::None
}

pub fn deep_due(state: &LivenessState, now: i64) -> bool {
    now - state.last_deep_at >= DEEP_INTERVAL_SECS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_503_does_not_kill() {
        let mut st = LivenessState::default();
        let a = step(
            &mut st,
            ProbeInput {
                now: 0,
                pid_alive: true,
                cheap_ok: false,
                deep_ok: None,
            },
        );
        assert_eq!(a, WatchdogAction::None);
        let a2 = step(
            &mut st,
            ProbeInput {
                now: 5,
                pid_alive: true,
                cheap_ok: true,
                deep_ok: None,
            },
        );
        assert_eq!(a2, WatchdogAction::None);
        assert!(st.cheap_fail_since.is_none());
    }

    #[test]
    fn cheap_fail_30s_kills() {
        let mut st = LivenessState {
            cheap_ok_once: true,
            ..Default::default()
        };
        step(
            &mut st,
            ProbeInput {
                now: 0,
                pid_alive: true,
                cheap_ok: false,
                deep_ok: None,
            },
        );
        let a = step(
            &mut st,
            ProbeInput {
                now: 31,
                pid_alive: true,
                cheap_ok: false,
                deep_ok: None,
            },
        );
        assert!(matches!(a, WatchdogAction::MarkDead { .. }));
    }

    #[test]
    fn deep_three_streak_kills() {
        let mut st = LivenessState {
            last_deep_at: -100,
            ..Default::default()
        };
        for t in [0, 60, 120] {
            step(
                &mut st,
                ProbeInput {
                    now: t,
                    pid_alive: true,
                    cheap_ok: true,
                    deep_ok: Some(false),
                },
            );
        }
        let a = step(
            &mut st,
            ProbeInput {
                now: 180,
                pid_alive: true,
                cheap_ok: true,
                deep_ok: Some(false),
            },
        );
        assert!(matches!(a, WatchdogAction::MarkDead { .. }));
    }
}
