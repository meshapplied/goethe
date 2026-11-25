// Routed inference on the node connection: the headers the platform adds and the usage trailer the node sends back.

pub const REQUEST_ID_HEADER: &str = "x-goethe-request-id";
pub const KEY_ID_HEADER: &str = "x-goethe-key-id";
pub const INSTANCE_HEADER: &str = "x-goethe-instance";
pub const USER_HEADER: &str = "x-goethe-user";
pub const DISPATCH_HEADER: &str = "x-goethe-dispatch-ms";
pub const USAGE_TRAILER: &str = "goethe-usage";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Usage {
    pub prompt: u64,
    pub completion: u64,
    pub ttft_ms: u64,
}

impl Usage {
    pub fn header(&self) -> String {
        format!("prompt={},completion={},ttft_ms={}", self.prompt, self.completion, self.ttft_ms)
    }

    pub fn parse(text: &str) -> Option<Usage> {
        // The trailer value back into numbers; None when a field is missing, repeated, unknown or not a number.
        let mut got: [Option<u64>; 3] = [None; 3];
        for part in text.split(',') {
            let (name, value) = part.trim().split_once('=')?;
            let i = ["prompt", "completion", "ttft_ms"].iter().position(|n| *n == name)?;
            if got[i].replace(value.parse().ok()?).is_some() {
                return None;
            }
        }
        Some(Usage { prompt: got[0]?, completion: got[1]?, ttft_ms: got[2]? })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usage_trailer_round_trips_and_refuses_junk() {
        let u = Usage { prompt: 12, completion: 340, ttft_ms: 87 };
        assert_eq!(Usage::parse(&u.header()), Some(u));
        for bad in ["", "prompt=1,completion=2", "prompt=1,completion=2,ttft_ms=x", "prompt=1,prompt=1,ttft_ms=3", "a=1,completion=2,ttft_ms=3"] {
            assert_eq!(Usage::parse(bad), None, "{bad}");
        }
    }
}
