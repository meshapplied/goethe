// IPv4-first HTTP agent shared by the registry and Hugging Face clients.

use std::net::{SocketAddr, ToSocketAddrs};
use std::time::Duration;

struct Ipv4First;

impl ureq::Resolver for Ipv4First {
    fn resolve(&self, netloc: &str) -> std::io::Result<Vec<SocketAddr>> {
        // Hosts with an IPv6 black hole stall on every AAAA record, so try IPv4 first.
        let mut addrs: Vec<SocketAddr> = netloc.to_socket_addrs()?.collect();
        addrs.sort_by_key(|a| a.is_ipv6());
        Ok(addrs)
    }
}

pub fn agent() -> ureq::Agent {
    // Agent with IPv4-first resolution and a 5 s connect timeout.
    ureq::AgentBuilder::new()
        .resolver(Ipv4First)
        .timeout_connect(Duration::from_secs(5))
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ureq::Resolver;

    #[test]
    fn ipv4_sorts_before_ipv6() {
        // Loopback resolves to both families on most hosts; IPv4 must come first.
        let addrs = Ipv4First.resolve("localhost:80").unwrap();
        if let Some(first_v6) = addrs.iter().position(|a| a.is_ipv6()) {
            assert!(addrs[..first_v6].iter().all(|a| a.is_ipv4()));
            assert!(addrs[first_v6..].iter().all(|a| a.is_ipv6()));
        }
    }
}
