// Background note updates for in-session learning: one hidden model call at a time per model, never on the request path.

use crate::mux::Mux;
use goethe_core::gateway_cache::Route;
use goethe_core::inflight;
use goethe_core::learn::{Job, AGENT_MAX_TOKENS, USER_MAX_TOKENS};
use http_body_util::{BodyExt, Full, Limited};
use hyper::body::Bytes;
use hyper::header::{AUTHORIZATION, CONTENT_TYPE};
use hyper::{Method, Request, StatusCode};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::Semaphore;

const CALL_TIMEOUT: Duration = Duration::from_secs(600);
const REPLY_CAP: usize = 4 << 20;
const TICK: Duration = Duration::from_secs(5);

#[derive(Default)]
pub struct Gates(Mutex<HashMap<String, Arc<Semaphore>>>);

impl Gates {
    fn of(&self, model: &str) -> Arc<Semaphore> {
        let mut all = self.0.lock().unwrap_or_else(|e| e.into_inner());
        Arc::clone(all.entry(model.to_string()).or_insert_with(|| Arc::new(Semaphore::new(1))))
    }
}

struct Reply {
    text: String,
    prompt: u64,
    completion: u64,
}

fn visible(content: &str) -> &str {
    // A model without a reasoning parser can return its thinking inside the content.
    content.rsplit_once("</think>").map_or(content, |(_, rest)| rest).trim()
}

fn usable(content: &str, cut: bool) -> Option<&str> {
    // Thinking that never closed holds no answer; a reply cut at the token limit loses its unfinished last line.
    if content.contains("<think>") && !content.contains("</think>") {
        return None;
    }
    let text = visible(content);
    let text = if cut { text.rsplit_once('\n').map_or("", |(done, _)| done) } else { text };
    Some(text.trim()).filter(|t| !t.is_empty())
}

async fn post(m: &Mux, route: &Route, body: &Value) -> Result<(StatusCode, Bytes), String> {
    let mut builder = Request::builder()
        .method(Method::POST)
        .uri(format!("http://127.0.0.1:{}/v1/chat/completions", route.port))
        .header(CONTENT_TYPE, "application/json");
    if let Some(k) = &route.engine_key {
        builder = builder.header(AUTHORIZATION, format!("Bearer {k}"));
    }
    let raw = serde_json::to_vec(body).map_err(|e| e.to_string())?;
    let req = builder.body(Full::new(Bytes::from(raw))).map_err(|e| e.to_string())?;
    let call = async {
        let resp = m.client.request(req).await.map_err(|e| e.to_string())?;
        let status = resp.status();
        let bytes = Limited::new(resp.into_body(), REPLY_CAP).collect().await.map_err(|e| e.to_string())?.to_bytes();
        Ok::<_, String>((status, bytes))
    };
    tokio::time::timeout(CALL_TIMEOUT, call).await.map_err(|_| "timed out".to_string())?
}

async fn ask(m: &Mux, route: &Route, model: &str, prompt: &(&str, String), max_tokens: u64) -> Result<Reply, String> {
    // One tool-less completion with thinking off; a runtime that refuses the template option gets the plain request.
    let mut body = json!({
        "model": model,
        "messages": [{"role": "system", "content": prompt.0}, {"role": "user", "content": prompt.1}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": false,
        "chat_template_kwargs": {"enable_thinking": false},
    });
    let (mut status, mut bytes) = post(m, route, &body).await?;
    if status == StatusCode::BAD_REQUEST {
        if let Some(o) = body.as_object_mut() {
            o.remove("chat_template_kwargs");
        }
        (status, bytes) = post(m, route, &body).await?;
    }
    if !status.is_success() {
        return Err(format!("engine answered {status}"));
    }
    let v: Value = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    let content = v.pointer("/choices/0/message/content").and_then(Value::as_str).unwrap_or("");
    let cut = v.pointer("/choices/0/finish_reason").and_then(Value::as_str) == Some("length");
    let text = usable(content, cut).ok_or("the model returned no usable text")?;
    Ok(Reply {
        text: text.to_string(),
        prompt: v.pointer("/usage/prompt_tokens").and_then(Value::as_u64).unwrap_or(0),
        completion: v.pointer("/usage/completion_tokens").and_then(Value::as_u64).unwrap_or(0),
    })
}

fn live_route(m: &Mux, model: &str) -> Option<Route> {
    m.cache.exact().routes(model).iter().find(|r| !r.draining).cloned()
}

async fn run(m: &'static Mux, job: Job) {
    // Waits its turn, then runs both curators. These calls hold no in-flight slot, so an unload never waits for them; it fails them and the messages stay.
    let gate = m.learn_gates.of(&job.model);
    let Ok(_permit) = gate.acquire().await else {
        job.failed(0, 0);
        return;
    };
    let Some(route) = live_route(m, &job.model) else {
        job.failed(0, 0);
        return;
    };
    let (mut prompt, mut completion) = (0, 0);
    for (i, text) in job.user_texts.iter().enumerate() {
        match ask(m, &route, &job.model, &job.user_prompt(text), USER_MAX_TOKENS).await {
            Ok(r) => {
                job.apply_user(&r.text, i);
                prompt += r.prompt;
                completion += r.completion;
            }
            Err(e) => {
                eprintln!("goethe learn: note update failed for {}: {e}", job.model);
                job.failed(prompt, completion);
                return;
            }
        }
    }
    match ask(m, &route, &job.model, &job.agent_prompt, AGENT_MAX_TOKENS).await {
        Ok(r) => {
            job.apply_agent(&r.text);
            job.commit(prompt + r.prompt, completion + r.completion);
        }
        Err(e) => {
            eprintln!("goethe learn: note update failed for {}: {e}", job.model);
            job.failed(prompt, completion);
        }
    }
}

pub fn start(m: &'static Mux, job: Job) {
    tokio::spawn(run(m, job));
}

pub fn spawn_ticker(m: &'static Mux) {
    // Saves notes every few seconds, off the request path, and updates sessions that went quiet when their model is idle.
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(TICK);
        loop {
            tick.tick().await;
            let store = Arc::clone(&m.learn);
            let _ = tokio::task::spawn_blocking(move || store.tick()).await;
            let snap = m.cache.exact();
            for c in m.learn.idle() {
                if snap.learn().applies(&c.model) && inflight::count(&c.model) == 0 && !snap.routes(&c.model).is_empty() {
                    if let Some(job) = m.learn.begin_idle(&c, snap.context_length(&c.model)) {
                        start(m, job);
                    }
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thinking_is_removed_from_content() {
        assert_eq!(visible("<think>ADD goal ::: no</think>\nNONE\n"), "NONE");
        assert_eq!(visible("  NONE "), "NONE");
        assert_eq!(usable("<think>never closed", false), None);
        assert_eq!(usable("", false), None);
        assert_eq!(usable("ADD goal ::: a\nADD goal ::: unfinis", true), Some("ADD goal ::: a"));
        assert_eq!(usable("ADD goal ::: unfinis", true), None);
        assert_eq!(usable("NONE", false), Some("NONE"));
    }
}
