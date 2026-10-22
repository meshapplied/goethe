// Startup readiness timeout: stall detection instead of a fixed wall clock.

pub const LOG_STALL_SECS: i64 = 180;
pub const HARD_CAP_DEFAULT_SECS: i64 = 30 * 60;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StartupFail {
    ProcessGone,
    LogStalled,
    HardCap,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StartupStep {
    Continue,
    Ready,
    Failed(StartupFail),
}

#[derive(Debug, Clone, Default)]
pub struct StallTracker {
    pub started_at: i64,
    pub last_log_len: u64,
    pub last_log_growth_at: i64,
}

impl StallTracker {
    pub fn new(now: i64, initial_log_len: u64) -> Self {
        Self {
            started_at: now,
            last_log_len: initial_log_len,
            last_log_growth_at: now,
        }
    }
}

pub fn poll_startup_step(
    tracker: &mut StallTracker,
    now: i64,
    pid_alive: bool,
    log_len: u64,
    health_ready: bool,
    hard_cap_secs: i64,
) -> StartupStep {
    if health_ready {
        return StartupStep::Ready;
    }
    if !pid_alive {
        return StartupStep::Failed(StartupFail::ProcessGone);
    }
    if log_len > tracker.last_log_len {
        tracker.last_log_len = log_len;
        tracker.last_log_growth_at = now;
    }
    if now - tracker.last_log_growth_at >= LOG_STALL_SECS {
        return StartupStep::Failed(StartupFail::LogStalled);
    }
    if now - tracker.started_at >= hard_cap_secs {
        return StartupStep::Failed(StartupFail::HardCap);
    }
    StartupStep::Continue
}

pub fn hard_cap_secs_from_env() -> i64 {
    std::env::var("GOETHE_HEALTH_POLL_TIMEOUT_SEC")
        .ok()
        .and_then(|s| s.parse::<i64>().ok())
        .unwrap_or(HARD_CAP_DEFAULT_SECS)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cold_start_keeps_going_while_log_grows() {
        let mut t = StallTracker::new(0, 100);
        assert_eq!(
            poll_startup_step(&mut t, 150, true, 5000, false, 1800),
            StartupStep::Continue
        );
    }

    #[test]
    fn stall_after_180s_no_growth() {
        let mut t = StallTracker::new(0, 100);
        let _ = poll_startup_step(&mut t, 0, true, 100, false, 1800);
        assert_eq!(
            poll_startup_step(&mut t, 181, true, 100, false, 1800),
            StartupStep::Failed(StartupFail::LogStalled)
        );
    }

    #[test]
    fn health_ready_wins() {
        let mut t = StallTracker::new(0, 0);
        assert_eq!(
            poll_startup_step(&mut t, 5, true, 0, true, 1800),
            StartupStep::Ready
        );
    }

    #[test]
    fn hard_cap_from_env_default() {
        assert_eq!(hard_cap_secs_from_env(), HARD_CAP_DEFAULT_SECS);
    }
}
