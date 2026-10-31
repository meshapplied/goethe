// Trace bodies: a streamed reply rebuilt into its non-streaming body, and the canonical message form used to link turns.

use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, HashMap};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Api {
    Chat,
    Completions,
    Messages,
    Responses,
}

impl Api {
    pub fn from_path(tail: &str) -> Option<Api> {
        // Only generation routes are traced; embeddings, rerank and token counts are not.
        match tail {
            "/chat/completions" => Some(Api::Chat),
            "/completions" => Some(Api::Completions),
            "/messages" => Some(Api::Messages),
            "/responses" => Some(Api::Responses),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Api::Chat => "chat.completions",
            Api::Completions => "completions",
            Api::Messages => "messages",
            Api::Responses => "responses",
        }
    }

    pub fn has_messages(self) -> bool {
        matches!(self, Api::Chat | Api::Messages)
    }
}

pub fn assemble(api: Api, sse: bool, raw: &[u8]) -> Value {
    // The body a non-streaming call would have returned; unparseable bytes are kept as text.
    if !sse {
        return serde_json::from_slice(raw).unwrap_or_else(|_| Value::String(String::from_utf8_lossy(raw).into_owned()));
    }
    let events = sse_events(raw);
    match api {
        Api::Chat => chat(&events),
        Api::Completions => completions(&events),
        Api::Messages => messages(&events),
        Api::Responses => responses(&events),
    }
}

fn sse_events(raw: &[u8]) -> Vec<Value> {
    raw.split(|&b| b == b'\n')
        .filter_map(|line| line.strip_suffix(b"\r").unwrap_or(line).strip_prefix(b"data:"))
        .map(<[u8]>::trim_ascii)
        .filter(|d| *d != b"[DONE]")
        .filter_map(|d| serde_json::from_slice(d).ok())
        .collect()
}

fn append(obj: &mut Map<String, Value>, field: &str, s: &str) {
    match obj.get_mut(field) {
        Some(Value::String(cur)) => cur.push_str(s),
        _ => {
            obj.insert(field.to_string(), Value::String(s.to_string()));
        }
    }
}

fn copy_head(out: &mut Map<String, Value>, e: &Value, keys: &[&str]) {
    for k in keys {
        if let Some(v) = e.get(*k).filter(|v| !v.is_null()) {
            out.entry(k.to_string()).or_insert_with(|| v.clone());
        }
    }
}

fn chat(events: &[Value]) -> Value {
    // Deltas per choice index: text fields concatenate, tool calls merge by their own index.
    let mut out = Map::new();
    let mut choices: BTreeMap<u64, Map<String, Value>> = BTreeMap::new();
    for e in events {
        if let Some(err) = e.get("error") {
            out.insert("error".into(), err.clone());
            continue;
        }
        copy_head(&mut out, e, &["id", "created", "model", "system_fingerprint"]);
        if let Some(u) = e.get("usage").filter(|u| !u.is_null()) {
            out.insert("usage".into(), u.clone());
        }
        for c in e.get("choices").and_then(Value::as_array).into_iter().flatten() {
            let idx = c.get("index").and_then(Value::as_u64).unwrap_or(0);
            let ch = choices.entry(idx).or_insert_with(|| {
                let mut m = Map::new();
                m.insert("index".into(), json!(idx));
                m.insert("message".into(), json!({"role": "assistant"}));
                m
            });
            if let (Some(d), Some(Value::Object(msg))) = (c.get("delta"), ch.get_mut("message")) {
                if let Some(r) = d.get("role").and_then(Value::as_str) {
                    msg.insert("role".into(), json!(r));
                }
                for field in ["content", "reasoning_content"] {
                    if let Some(s) = d.get(field).and_then(Value::as_str) {
                        append(msg, field, s);
                    }
                }
                for tc in d.get("tool_calls").and_then(Value::as_array).into_iter().flatten() {
                    merge_tool_call(msg, tc);
                }
            }
            for k in ["finish_reason", "matched_stop"] {
                if let Some(v) = c.get(k).filter(|v| !v.is_null()) {
                    ch.insert(k.into(), v.clone());
                }
            }
        }
    }
    let choices: Vec<Value> = choices
        .into_values()
        .map(|mut ch| {
            if let Some(Value::Object(msg)) = ch.get_mut("message") {
                msg.entry("content").or_insert(Value::Null);
                if let Some(Value::Array(calls)) = msg.get_mut("tool_calls") {
                    for call in calls.iter_mut().filter_map(Value::as_object_mut) {
                        call.remove("index");
                    }
                }
            }
            ch.entry("finish_reason").or_insert(Value::Null);
            Value::Object(ch)
        })
        .collect();
    out.insert("object".into(), json!("chat.completion"));
    out.insert("choices".into(), Value::Array(choices));
    Value::Object(out)
}

fn merge_tool_call(msg: &mut Map<String, Value>, tc: &Value) {
    let idx = tc.get("index").and_then(Value::as_u64).unwrap_or(0);
    let calls = msg.entry("tool_calls").or_insert_with(|| json!([]));
    let Some(calls) = calls.as_array_mut() else {
        return;
    };
    let pos = match calls.iter().position(|c| c.get("index").and_then(Value::as_u64) == Some(idx)) {
        Some(p) => p,
        None => {
            calls.push(json!({"index": idx, "type": "function", "function": {"name": "", "arguments": ""}}));
            calls.len() - 1
        }
    };
    let Some(call) = calls[pos].as_object_mut() else {
        return;
    };
    for k in ["id", "type"] {
        if let Some(v) = tc.get(k).and_then(Value::as_str) {
            call.insert(k.into(), json!(v));
        }
    }
    if let (Some(f), Some(Value::Object(dst))) = (tc.get("function"), call.get_mut("function")) {
        for k in ["name", "arguments"] {
            if let Some(s) = f.get(k).and_then(Value::as_str) {
                append(dst, k, s);
            }
        }
    }
}

fn completions(events: &[Value]) -> Value {
    let mut out = Map::new();
    let mut choices: BTreeMap<u64, Map<String, Value>> = BTreeMap::new();
    for e in events {
        if let Some(err) = e.get("error") {
            out.insert("error".into(), err.clone());
            continue;
        }
        copy_head(&mut out, e, &["id", "created", "model"]);
        if let Some(u) = e.get("usage").filter(|u| !u.is_null()) {
            out.insert("usage".into(), u.clone());
        }
        for c in e.get("choices").and_then(Value::as_array).into_iter().flatten() {
            let idx = c.get("index").and_then(Value::as_u64).unwrap_or(0);
            let ch = choices.entry(idx).or_insert_with(|| {
                let mut m = Map::new();
                m.insert("index".into(), json!(idx));
                m.insert("text".into(), json!(""));
                m
            });
            if let Some(s) = c.get("text").and_then(Value::as_str) {
                append(ch, "text", s);
            }
            if let Some(v) = c.get("finish_reason").filter(|v| !v.is_null()) {
                ch.insert("finish_reason".into(), v.clone());
            }
        }
    }
    out.insert("object".into(), json!("text_completion"));
    out.insert("choices".into(), Value::Array(choices.into_values().map(Value::Object).collect()));
    Value::Object(out)
}

fn messages(events: &[Value]) -> Value {
    // Anthropic events: message_start, content blocks with their deltas, then message_delta with stop reason and usage.
    let mut msg = Map::new();
    let mut blocks: Vec<Value> = Vec::new();
    let mut partial: HashMap<usize, String> = HashMap::new();
    for e in events {
        let block_at = |e: &Value| e.get("index").and_then(Value::as_u64).map(|i| i as usize);
        match e.get("type").and_then(Value::as_str).unwrap_or("") {
            "message_start" => {
                if let Some(Value::Object(m)) = e.get("message") {
                    msg = m.clone();
                }
            }
            "content_block_start" => {
                if let (Some(i), Some(b)) = (block_at(e), e.get("content_block")) {
                    if blocks.len() <= i {
                        blocks.resize(i + 1, Value::Null);
                    }
                    blocks[i] = b.clone();
                }
            }
            "content_block_delta" => {
                let (Some(i), Some(d)) = (block_at(e), e.get("delta")) else {
                    continue;
                };
                let field = match d.get("type").and_then(Value::as_str).unwrap_or("") {
                    "text_delta" => "text",
                    "thinking_delta" => "thinking",
                    "signature_delta" => "signature",
                    "input_json_delta" => {
                        let s = d.get("partial_json").and_then(Value::as_str).unwrap_or("");
                        partial.entry(i).or_default().push_str(s);
                        continue;
                    }
                    _ => continue,
                };
                if let (Some(s), Some(Value::Object(b))) = (d.get(field).and_then(Value::as_str), blocks.get_mut(i)) {
                    append(b, field, s);
                }
            }
            "message_delta" => {
                if let Some(Value::Object(d)) = e.get("delta") {
                    for (k, v) in d {
                        msg.insert(k.clone(), v.clone());
                    }
                }
                if let Some(Value::Object(u)) = e.get("usage") {
                    let usage = msg.entry("usage").or_insert_with(|| json!({}));
                    if let Some(dst) = usage.as_object_mut() {
                        for (k, v) in u {
                            dst.insert(k.clone(), v.clone());
                        }
                    }
                }
            }
            "error" => {
                msg.insert("error".into(), e.get("error").cloned().unwrap_or(Value::Null));
            }
            _ => {}
        }
    }
    for (i, s) in partial {
        if let Some(Value::Object(b)) = blocks.get_mut(i) {
            let input = if s.trim().is_empty() { json!({}) } else { serde_json::from_str(&s).unwrap_or(Value::String(s)) };
            b.insert("input".into(), input);
        }
    }
    msg.insert("content".into(), Value::Array(blocks.into_iter().filter(|b| !b.is_null()).collect()));
    Value::Object(msg)
}

fn responses(events: &[Value]) -> Value {
    // The Responses API ends with an event carrying the whole response object.
    events
        .iter()
        .rev()
        .find_map(|e| e.get("response").filter(|r| r.is_object()))
        .cloned()
        .unwrap_or_else(|| json!({}))
}

pub fn reply_message(api: Api, body: &Value) -> Option<Value> {
    // The assistant message a client would append to its history after this reply.
    match api {
        Api::Chat => body.pointer("/choices/0/message").cloned(),
        Api::Messages => body.get("content").map(|c| json!({"role": "assistant", "content": c})),
        _ => None,
    }
}

pub fn canonical(msg: &Value) -> String {
    // Role, visible text and tool calls with key-sorted arguments; ids and reasoning are left out, as clients resend history without them.
    let role = msg.get("role").and_then(Value::as_str).unwrap_or("");
    let mut text = Vec::new();
    let mut calls = Vec::new();
    content_parts(msg.get("content"), &mut text, &mut calls);
    for tc in msg.get("tool_calls").and_then(Value::as_array).into_iter().flatten() {
        let f = tc.get("function").unwrap_or(&Value::Null);
        let args = match f.get("arguments") {
            Some(Value::String(s)) => serde_json::from_str(s).unwrap_or_else(|_| Value::String(s.trim().to_string())),
            Some(v) => v.clone(),
            None => Value::Null,
        };
        calls.push(json!([f.get("name").and_then(Value::as_str).unwrap_or(""), sorted(&args)]));
    }
    json!([role, text.join("\n"), calls]).to_string()
}

fn content_parts(content: Option<&Value>, text: &mut Vec<String>, calls: &mut Vec<Value>) {
    match content {
        Some(Value::String(s)) => text.push(s.trim().to_string()),
        Some(Value::Array(parts)) => {
            for p in parts {
                match p.get("type").and_then(Value::as_str).unwrap_or("") {
                    "text" | "input_text" | "output_text" => {
                        text.push(p.get("text").and_then(Value::as_str).unwrap_or("").trim().to_string());
                    }
                    "tool_use" => calls.push(json!([
                        p.get("name").and_then(Value::as_str).unwrap_or(""),
                        sorted(p.get("input").unwrap_or(&Value::Null))
                    ])),
                    "tool_result" => {
                        let mut inner = Vec::new();
                        content_parts(p.get("content"), &mut inner, calls);
                        text.push(format!("[tool_result]{}", inner.join("\n")));
                    }
                    "thinking" | "redacted_thinking" => {}
                    other => text.push(format!("[{other}]{}", sorted(p))),
                }
            }
        }
        _ => {}
    }
}

pub fn sorted(v: &Value) -> Value {
    // Rebuild objects with keys inserted in order, so the text is the same whether or not serde_json keeps insertion order.
    match v {
        Value::Object(m) => {
            let mut keys: Vec<&String> = m.keys().collect();
            keys.sort();
            Value::Object(keys.into_iter().map(|k| (k.clone(), sorted(&m[k]))).collect())
        }
        Value::Array(a) => Value::Array(a.iter().map(sorted).collect()),
        other => other.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sse(events: &[Value]) -> Vec<u8> {
        let mut out = String::new();
        for e in events {
            out.push_str(&format!("data: {e}\n\n"));
        }
        out.push_str("data: [DONE]\n\n");
        out.into_bytes()
    }

    #[test]
    fn chat_stream_rebuilds_content_reasoning_and_tool_calls() {
        let raw = sse(&[
            json!({"id": "c1", "created": 7, "model": "m", "choices": [{"index": 0, "delta": {"role": "assistant", "reasoning_content": "think "}}]}),
            json!({"id": "c1", "choices": [{"index": 0, "delta": {"reasoning_content": "more"}}]}),
            json!({"id": "c1", "choices": [{"index": 0, "delta": {"content": "Hi"}}]}),
            json!({"id": "c1", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {"name": "read", "arguments": "{\"pa"}}]}}]}),
            json!({"id": "c1", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "th\": \"a\"}"}}]}}]}),
            json!({"id": "c1", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}),
            json!({"id": "c1", "choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 9}}),
        ]);
        let body = assemble(Api::Chat, true, &raw);
        assert_eq!(body["object"], "chat.completion");
        assert_eq!(body["usage"]["completion_tokens"], 9);
        let msg = &body["choices"][0]["message"];
        assert_eq!(msg["content"], "Hi");
        assert_eq!(msg["reasoning_content"], "think more");
        assert_eq!(msg["tool_calls"][0], json!({"id": "call_1", "type": "function", "function": {"name": "read", "arguments": "{\"path\": \"a\"}"}}));
        assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
    }

    #[test]
    fn anthropic_stream_rebuilds_blocks_and_stop_reason() {
        let raw = sse(&[
            json!({"type": "message_start", "message": {"id": "m1", "role": "assistant", "content": [], "usage": {"input_tokens": 4}}}),
            json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hel"}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "lo"}}),
            json!({"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "tu", "name": "ls", "input": {}}}),
            json!({"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{\"dir\":"}}),
            json!({"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "\".\"}"}}),
            json!({"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 12}}),
            json!({"type": "message_stop"}),
        ]);
        let body = assemble(Api::Messages, true, &raw);
        assert_eq!(body["content"][0]["text"], "Hello");
        assert_eq!(body["content"][1]["input"], json!({"dir": "."}));
        assert_eq!(body["stop_reason"], "tool_use");
        assert_eq!(body["usage"], json!({"input_tokens": 4, "output_tokens": 12}));
    }

    #[test]
    fn canonical_ignores_ids_reasoning_and_argument_formatting() {
        let produced = json!({"role": "assistant", "content": "  ok\n", "reasoning_content": "hmm",
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{\"b\": 1, \"a\": 2}"}}]});
        let resent = json!({"role": "assistant", "content": "ok",
            "tool_calls": [{"id": "x", "type": "function", "function": {"name": "f", "arguments": "{\"a\":2,\"b\":1}"}}]});
        assert_eq!(canonical(&produced), canonical(&resent));
        let anthropic = json!({"role": "assistant", "content": [{"type": "thinking", "thinking": "hmm"}, {"type": "text", "text": "ok"},
            {"type": "tool_use", "id": "tu", "name": "f", "input": {"b": 1, "a": 2}}]});
        assert_eq!(canonical(&produced), canonical(&anthropic));
        assert_ne!(canonical(&produced), canonical(&json!({"role": "assistant", "content": "ok"})));
    }
}
