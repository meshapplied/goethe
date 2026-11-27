// How a client calls the gateway in each API schema it speaks (OpenAI, Anthropic, LiteLLM): one request and one SDK call, for the menus to show.

pub const NAMES: [&str; 3] = ["openai", "anthropic", "litellm"];

use std::net::{IpAddr, Ipv4Addr, SocketAddr};

pub const KEY: &str = "API_KEY";
pub const MODEL: &str = "MODEL_ID";

fn gateway_ip(bind: &str, vpn: Option<IpAddr>, public: impl FnOnce() -> Option<IpAddr>) -> IpAddr {
    // The address clients use: the one the gateway listens on (its VPN address, or 127.0.0.1 when local only); on every address, this machine's VPN address, else its public one.
    match bind.parse::<IpAddr>() {
        Ok(ip) if !ip.is_unspecified() => ip,
        Ok(_) => vpn.or_else(public).unwrap_or(IpAddr::V4(Ipv4Addr::LOCALHOST)),
        Err(_) => IpAddr::V4(Ipv4Addr::LOCALHOST),
    }
}

pub fn base_url(home: &std::path::Path) -> String {
    let vpn = crate::nginx::interface_addrs()
        .unwrap_or_default()
        .into_iter()
        .find(|a| a.ip.is_ipv4() && crate::snapshot::scope_of(a.ip, crate::netif::is_tunnel(&a.name)) == "vpn")
        .map(|a| a.ip);
    let ip = gateway_ip(&crate::nginx::public_bind(home), vpn, || crate::client::link::lookup_public_ip(home).ok());
    format!("https://{}/v1", SocketAddr::new(ip, crate::nginx::public_port_u16(home)))
}

pub fn example(schema: &str, base: &str, model: &str, key: &str) -> Vec<String> {
    // The lines for one schema: its base URL, a curl request (self-signed certificate, so -k) and the SDK call.
    let root = base.trim_end_matches("/v1");
    let hello = r#"[{"role": "user", "content": "Hello!"}]"#;
    let lines: Vec<String> = match schema {
        "anthropic" => vec![
            format!("Base URL   {root}"),
            String::new(),
            format!("curl -k {base}/messages \\"),
            format!("  -H \"x-api-key: {key}\" \\"),
            "  -H \"anthropic-version: 2023-06-01\" \\".to_string(),
            "  -H \"Content-Type: application/json\" \\".to_string(),
            format!("  -d '{{\"model\": \"{model}\", \"max_tokens\": 256, \"messages\": {hello}}}'"),
            String::new(),
            "import anthropic".to_string(),
            format!("client = anthropic.Anthropic(base_url=\"{root}\", api_key=\"{key}\")"),
            format!("client.messages.create(model=\"{model}\", max_tokens=256, messages={hello})"),
        ],
        "litellm" => vec![
            format!("Base URL   {root}"),
            String::new(),
            format!("curl -k {root}/chat/completions \\"),
            format!("  -H \"x-litellm-api-key: {key}\" \\"),
            "  -H \"Content-Type: application/json\" \\".to_string(),
            format!("  -d '{{\"model\": \"{model}\", \"messages\": {hello}}}'"),
            String::new(),
            "import litellm".to_string(),
            format!("litellm.completion(model=\"litellm_proxy/{model}\", api_base=\"{root}\", api_key=\"{key}\", messages={hello})"),
        ],
        _ => vec![
            format!("Base URL   {base}"),
            String::new(),
            format!("curl -k {base}/chat/completions \\"),
            format!("  -H \"Authorization: Bearer {key}\" \\"),
            "  -H \"Content-Type: application/json\" \\".to_string(),
            format!("  -d '{{\"model\": \"{model}\", \"messages\": {hello}}}'"),
            String::new(),
            "from openai import OpenAI".to_string(),
            format!("client = OpenAI(base_url=\"{base}\", api_key=\"{key}\")"),
            format!("client.chat.completions.create(model=\"{model}\", messages={hello})"),
        ],
    };
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_address_is_the_vpn_then_local_then_public() {
        let vpn: Option<IpAddr> = Some("100.64.1.2".parse().unwrap());
        let public = || Some("203.0.113.7".parse().unwrap());
        assert_eq!(gateway_ip("100.64.1.2", None, public).to_string(), "100.64.1.2", "listening on the VPN");
        assert_eq!(gateway_ip("127.0.0.1", None, public).to_string(), "127.0.0.1", "local only");
        assert_eq!(gateway_ip("0.0.0.0", vpn, public).to_string(), "100.64.1.2", "every address: the VPN first");
        assert_eq!(gateway_ip("0.0.0.0", None, public).to_string(), "203.0.113.7", "every address, no VPN: the public one");
    }

    #[test]
    fn each_schema_uses_the_route_and_key_header_the_gateway_reads() {
        let base = "https://127.0.0.1:8443/v1";
        let text = |s: &str| example(s, base, "MODEL", "API_KEY").join("\n");
        let openai = text("openai");
        assert!(openai.contains("curl -k https://127.0.0.1:8443/v1/chat/completions") && openai.contains("Authorization: Bearer API_KEY"));
        let anthropic = text("anthropic");
        assert!(anthropic.contains("/v1/messages") && anthropic.contains("x-api-key: API_KEY") && anthropic.contains("anthropic-version"));
        assert!(anthropic.contains("base_url=\"https://127.0.0.1:8443\""), "the Anthropic SDK adds /v1 itself");
        let litellm = text("litellm");
        assert!(litellm.contains("curl -k https://127.0.0.1:8443/chat/completions") && litellm.contains("x-litellm-api-key: API_KEY"));
        assert!(NAMES.iter().all(|n| example(n, base, "m", "k").iter().any(|l| l.contains("\"m\""))));
    }
}
