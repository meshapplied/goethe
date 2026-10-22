// Pure watchdog restart window and backoff decisions (injectable clock in tests).

pub const RESTART_WINDOW_SECS: i64 = 600;
pub const MAX_RESTARTS_IN_WINDOW: i64 = 3;

pub fn restart_backoff_secs(instance_restart_count: i64) -> u64 {
    match instance_restart_count {
        0 => 5,
        1 => 15,
        _ => 45,
    }
}

pub struct WindowState {
    pub window_start: i64,
    pub count: i64,
}

pub fn effective_window_count(now: i64, window_start: i64, count: i64) -> i64 {
    if now.saturating_sub(window_start) > RESTART_WINDOW_SECS {
        0
    } else {
        count
    }
}

pub enum RestartDecision {
    Allow {
        new_window_start: i64,
        new_count: i64,
    },
    DenyFailed,
}

pub fn decide_restart(now: i64, window_start: Option<i64>, count: i64) -> RestartDecision {
    let (start, cnt) = match window_start {
        Some(s) => (s, effective_window_count(now, s, count)),
        None => (now, 0),
    };
    if cnt >= MAX_RESTARTS_IN_WINDOW {
        return RestartDecision::DenyFailed;
    }
    let new_start = if window_start.is_some() && now.saturating_sub(start) <= RESTART_WINDOW_SECS {
        start
    } else {
        now
    };
    RestartDecision::Allow {
        new_window_start: new_start,
        new_count: cnt + 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backoff_schedule() {
        assert_eq!(restart_backoff_secs(0), 5);
        assert_eq!(restart_backoff_secs(1), 15);
        assert_eq!(restart_backoff_secs(2), 45);
        assert_eq!(restart_backoff_secs(9), 45);
    }

    #[test]
    fn window_expiry_resets_count() {
        assert_eq!(effective_window_count(1000, 100, 2), 0);
        assert_eq!(effective_window_count(500, 100, 2), 2);
    }

    #[test]
    fn allows_three_restarts_then_denies() {
        let t0 = 1_000_000i64;
        let d1 = decide_restart(t0, None, 0);
        let RestartDecision::Allow {
            new_window_start: s1,
            new_count: c1,
        } = d1
        else {
            panic!("expected allow");
        };
        assert_eq!(c1, 1);
        let d2 = decide_restart(t0 + 10, Some(s1), c1);
        let RestartDecision::Allow { new_count: c2, .. } = d2 else {
            panic!("expected allow");
        };
        assert_eq!(c2, 2);
        let d3 = decide_restart(t0 + 20, Some(s1), c2);
        let RestartDecision::Allow { new_count: c3, .. } = d3 else {
            panic!("expected allow");
        };
        assert_eq!(c3, 3);
        assert!(matches!(
            decide_restart(t0 + 30, Some(s1), c3),
            RestartDecision::DenyFailed
        ));
    }

    #[test]
    fn expired_window_allows_again() {
        let start = 0i64;
        let now = RESTART_WINDOW_SECS + 1;
        assert!(matches!(
            decide_restart(now, Some(start), 3),
            RestartDecision::Allow { .. }
        ));
    }
}
