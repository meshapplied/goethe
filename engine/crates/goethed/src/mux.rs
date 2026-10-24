// API multiplexer on 127.0.0.1 (persisted mux port): OpenAI, Anthropic and LiteLLM routes, key auth, metering and upstream auth.

use goethe_core::db;
use goethe_core::gateway_cache::{GatewayCache, Route, Snapshot};
use goethe_core::hw;
use goethe_core::inflight;
use goethe_core::metrics;
use goethe_core::nginx;
use goethe_core::paths::goethe_home;
use goethe_core::placement;
use goethe_core::security::{self, ResolvedKey};
use http_body_util::{BodyExt, Full, LengthLimitError, Limited};
use hyper::body::{Body, Bytes, Frame, Incoming, SizeHint};
use hyper::header::{HeaderMap, HeaderName, HeaderValue, ALLOW, AUTHORIZATION, CONTENT_LENGTH, CONTENT_TYPE, RETRY_AFTER};
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::client::legacy::connect::HttpConnector;
use hyper_util::client::legacy::Client;
use hyper_util::rt::{TokioExecutor, TokioIo, TokioTimer};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::convert::Infallible;
use std::fmt::Write as _;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};
use tokio::net::TcpListener;

type MuxBody = http_body_util::combinators::BoxBody<Bytes, hyper::Error>;
type Upstream = Client<HttpConnector, Full<Bytes>>;
type MuxRef = &'static Mux;

const HEADER_READ_TIMEOUT: Duration = Duration::from_secs(10);
const UPSTREAM_IDLE: Duration = Duration::from_secs(4);
const UPSTREAM_TIMEOUT: Duration = Duration::from_secs(600);
const PROBE_TIMEOUT: Duration = Duration::from_secs(2);
const READY_TTL: Duration = Duration::from_secs(1);
const JSON_USAGE_CAP: usize = 8 << 20;
const SSE_LINE_CAP: usize = 1 << 20;
const X_REQUEST_ID: HeaderName = HeaderName::from_static("x-request-id");

struct Mux {
    cache: GatewayCache,
    client: Upstream,
    max_body: usize,
    rpm: Mutex<HashMap<String, Instant>>,
    ready: Mutex<HashMap<String, (Instant, bool)>>,
}

static MUX: OnceLock<Mux> = OnceLock::new();

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

pub fn spawn_mux() -> Result<tokio::task::JoinHandle<()>, String> {
    // Bind the daemon's persisted mux port now so startup fails fast, then serve it on the runtime.
    let home = goethe_home();
    let port = nginx::persisted_mux_port(&home)?;
    raise_nofile_limit();
    let listener = std::net::TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], port))).map_err(|e| {
        format!("mux bind failed on 127.0.0.1:{port} ({e}); set GOETHE_MUX_PORT to a free port")
    })?;
    listener.set_nonblocking(true).map_err(|e| e.to_string())?;
    let listener = TcpListener::from_std(listener).map_err(|e| e.to_string())?;
    let max_body = nginx::persist_max_body_mb(&home) as usize * 1024 * 1024;
    let mux = MUX.get_or_init(|| Mux {
        cache: GatewayCache::open(&home),
        client: upstream_client(),
        max_body,
        rpm: Mutex::new(HashMap::new()),
        ready: Mutex::new(HashMap::new()),
    });
    let _ = mux.cache.exact();
    Ok(tokio::spawn(serve(mux, listener)))
}

fn raise_nofile_limit() {
    // Lift the soft RLIMIT_NOFILE to the hard limit so connection bursts do not hit EMFILE.
    let mut lim = libc::rlimit { rlim_cur: 0, rlim_max: 0 };
    unsafe {
        if libc::getrlimit(libc::RLIMIT_NOFILE, &mut lim) == 0 && lim.rlim_cur < lim.rlim_max {
            lim.rlim_cur = lim.rlim_max;
            libc::setrlimit(libc::RLIMIT_NOFILE, &lim);
        }
    }
}

fn upstream_client() -> Upstream {
    // Pooled keep-alive client; idle sockets close before SGLang's 5 s keep-alive would drop them.
    let mut http = HttpConnector::new();
    http.set_nodelay(true);
    http.set_connect_timeout(Some(Duration::from_secs(2)));
    Client::builder(TokioExecutor::new())
        .pool_idle_timeout(UPSTREAM_IDLE)
        .pool_max_idle_per_host(64)
        .pool_timer(TokioTimer::new())
        .build(http)
}

async fn serve(m: MuxRef, listener: TcpListener) {
    // Accept forever; transient accept errors (EMFILE, ECONNABORTED) back off 50 ms instead of ending the mux.
    loop {
        let stream = match listener.accept().await {
            Ok((stream, _)) => stream,
            Err(e) => {
                eprintln!("goethe mux: accept failed: {e}");
                tokio::time::sleep(Duration::from_millis(50)).await;
                continue;
            }
        };
        let _ = stream.set_nodelay(true);
        tokio::spawn(async move {
            let svc = service_fn(move |req| async move { Ok::<_, Infallible>(handle(m, req).await) });
            let _ = http1::Builder::new()
                .timer(TokioTimer::new())
                .header_read_timeout(HEADER_READ_TIMEOUT)
                .serve_connection(TokioIo::new(stream), svc)
                .await;
        });
    }
}

async fn handle(m: MuxRef, req: Request<Incoming>) -> Response<MuxBody> {
    let rid = request_id(req.headers());
    let anthropic = is_anthropic(req.uri().path());
    let mut resp = dispatch(m, req, &rid).await;
    if anthropic {
        resp = anthropic_error(resp);
    }
    if let Ok(v) = HeaderValue::from_str(&rid) {
        resp.headers_mut().insert(X_REQUEST_ID, v);
    }
    resp
}

async fn dispatch(m: MuxRef, req: Request<Incoming>, rid: &str) -> Response<MuxBody> {
    // Health and metrics need no key; every API route is authenticated.
    let path = req.uri().path().to_string();
    let get = req.method() == Method::GET;
    match path.as_str() {
        "/metrics" | "/health" | "/health/ready" | "/health/liveliness" | "/health/liveness" | "/health/readiness"
            if !get =>
        {
            method_not_allowed(&Method::GET)
        }
        p if p.starts_with("/health/admin") && !get => method_not_allowed(&Method::GET),
        "/metrics" => match tokio::task::spawn_blocking(metrics::render_prometheus).await {
            Ok(text) => {
                let mut resp = Response::new(full(text));
                resp.headers_mut().insert(
                    CONTENT_TYPE,
                    HeaderValue::from_static("text/plain; version=0.0.4; charset=utf-8"),
                );
                resp
            }
            Err(_) => error(StatusCode::INTERNAL_SERVER_ERROR, "metrics unavailable", "server_error", "internal"),
        },
        "/health" => json_response(StatusCode::OK, json!({"status": "ok"})),
        "/health/ready" => health_ready(m, req.uri().query()).await,
        "/health/liveliness" | "/health/liveness" => json_response(StatusCode::OK, json!("I'm alive!")),
        "/health/readiness" => json_response(StatusCode::OK, json!({"status": "healthy"})),
        p if p.starts_with("/health/admin") => health_admin(m, req.headers()).await,
        _ => api(m, req, &path, rid).await,
    }
}

enum Api {
    Models,
    Model(String),
    ModelInfo { group: bool },
    Proxy(String),
}

fn api_tail(path: &str) -> &str {
    // LiteLLM serves every OpenAI route with and without the /v1 prefix.
    path.strip_prefix("/v1").filter(|t| t.starts_with('/')).unwrap_or(path)
}

fn is_anthropic(path: &str) -> bool {
    matches!(api_tail(path), "/messages" | "/messages/count_tokens")
}

fn api_route(path: &str) -> Option<(Method, Api)> {
    // OpenAI, Anthropic Messages and LiteLLM paths; proxied calls go to the engine's /v1 path.
    let tail = api_tail(path);
    match tail {
        "/chat/completions" | "/completions" | "/embeddings" | "/responses" | "/rerank" | "/messages"
        | "/messages/count_tokens" => Some((Method::POST, Api::Proxy(format!("/v1{tail}")))),
        "/models" => Some((Method::GET, Api::Models)),
        "/model/info" => Some((Method::GET, Api::ModelInfo { group: false })),
        "/model_group/info" => Some((Method::GET, Api::ModelInfo { group: true })),
        t => t.strip_prefix("/models/").map(|id| (Method::GET, Api::Model(percent_decode(id, false)))),
    }
}

async fn api(m: MuxRef, req: Request<Incoming>, path: &str, rid: &str) -> Response<MuxBody> {
    // Method check, key auth, per-key rpm, then the model list or a proxied call.
    let Some((allowed, route)) = api_route(path) else {
        return not_found();
    };
    if req.method() != allowed {
        return method_not_allowed(&allowed);
    }
    let key = match authenticate(m, req.headers()) {
        Ok(k) => k,
        Err((message, code)) => return error(StatusCode::UNAUTHORIZED, message, "auth_error", code),
    };
    if let Some(retry) = gcra_check(&mut lock(&m.rpm), &key.key_id, key.rpm_limit, Instant::now()) {
        metrics::inc_rate_limited(&key.key_id);
        return rate_limited(retry);
    }
    match route {
        Api::Models => list_models(m, &key),
        Api::Model(id) => get_model(m, &key, &id),
        Api::ModelInfo { group } => model_info(m, &key, group),
        Api::Proxy(upstream) => route_post(m, req, &upstream, &key, rid).await,
    }
}

fn authenticate(
    m: &Mux,
    headers: &HeaderMap,
) -> Result<Arc<ResolvedKey>, (&'static str, &'static str)> {
    // Client token to a cached key (a miss rechecks the DB); expired keys are refused.
    let key = client_token(headers)
        .and_then(|t| m.cache.resolve_token(t))
        .ok_or(("Invalid or missing Goethe API key", "invalid_api_key"))?;
    if security::key_expired(&key.expires_at) {
        return Err(("key expired", "key_expired"));
    }
    Ok(key)
}

fn client_token(headers: &HeaderMap) -> Option<&str> {
    // Bearer in Authorization or x-litellm-api-key, else a bare key in x-litellm-api-key, x-api-key (Anthropic) or api-key (Azure).
    let get = |name: &str| headers.get(name).and_then(|v| v.to_str().ok());
    get(AUTHORIZATION.as_str())
        .and_then(parse_bearer)
        .or_else(|| get("x-litellm-api-key").map(|v| parse_bearer(v).unwrap_or(v.trim())))
        .or_else(|| get("x-api-key").or_else(|| get("api-key")).map(str::trim))
        .filter(|t| !t.is_empty() && !t.contains(|c: char| c.is_ascii_whitespace()))
}

fn parse_bearer(raw: &str) -> Option<&str> {
    // "Bearer <token>": case-insensitive scheme, any surrounding or separating whitespace, one token.
    let (scheme, rest) = raw.trim().split_once(|c: char| c.is_ascii_whitespace())?;
    let token = rest.trim();
    (scheme.eq_ignore_ascii_case("bearer") && !token.is_empty() && !token.contains(|c: char| c.is_ascii_whitespace()))
        .then_some(token)
}

fn gcra_check(tats: &mut HashMap<String, Instant>, key_id: &str, rpm: i64, now: Instant) -> Option<u64> {
    // GCRA: up to rpm requests at once, refilled at rpm per minute; Some(retry-after secs) when limited.
    if rpm <= 0 {
        return None;
    }
    let rpm = u32::try_from(rpm).unwrap_or(u32::MAX);
    let interval = Duration::from_secs(60) / rpm;
    let burst = interval * (rpm - 1);
    let tat = tats.get(key_id).copied().filter(|t| *t > now).unwrap_or(now);
    let ahead = tat.duration_since(now);
    if ahead > burst {
        return Some((ahead - burst).as_secs_f64().ceil().max(1.0) as u64);
    }
    if tats.len() > 10_000 {
        tats.retain(|_, t| *t > now);
    }
    tats.insert(key_id.to_string(), tat + interval);
    None
}

fn model_object(snap: &Snapshot, id: &str) -> Value {
    json!({"id": id, "object": "model", "created": snap.created(id), "owned_by": "goethe"})
}

fn revoked() -> Response<MuxBody> {
    error(StatusCode::UNAUTHORIZED, "Invalid or missing Goethe API key", "auth_error", "invalid_api_key")
}

fn scoped_models<'a>(snap: &'a Snapshot, key: &ResolvedKey) -> Vec<&'a str> {
    snap.serving_models().into_iter().filter(|id| security::model_allowed(&key.model_scope, id)).collect()
}

fn list_models(m: &Mux, key: &ResolvedKey) -> Response<MuxBody> {
    // Serving models in the key's scope; the exact snapshot also refuses a key revoked a moment ago.
    let snap = m.cache.exact();
    if !snap.has_key(key) {
        return revoked();
    }
    let data: Vec<Value> = scoped_models(&snap, key).into_iter().map(|id| model_object(&snap, id)).collect();
    metrics::record_request("", 200, 0.0, 0, 0);
    m.cache.record_usage(&key.key_id, 0, 0);
    json_response(StatusCode::OK, json!({"object": "list", "data": data}))
}

fn model_info(m: &Mux, key: &ResolvedKey, group: bool) -> Response<MuxBody> {
    // LiteLLM's /model/info and /model_group/info shapes for the serving models in the key's scope.
    let snap = m.cache.exact();
    if !snap.has_key(key) {
        return revoked();
    }
    let data: Vec<Value> = scoped_models(&snap, key)
        .into_iter()
        .map(|id| {
            if group {
                json!({"model_group": id, "providers": ["openai"]})
            } else {
                json!({
                    "model_name": id,
                    "litellm_params": {"model": format!("openai/{id}")},
                    "model_info": {"id": id, "created": snap.created(id), "owned_by": "goethe"},
                })
            }
        })
        .collect();
    metrics::record_request("", 200, 0.0, 0, 0);
    m.cache.record_usage(&key.key_id, 0, 0);
    json_response(StatusCode::OK, json!({"data": data}))
}

fn get_model(m: &Mux, key: &ResolvedKey, raw: &str) -> Response<MuxBody> {
    // One serving model by id or alias; 404 when not serving or outside the key's scope.
    let snap = m.cache.exact();
    if !snap.has_key(key) {
        return revoked();
    }
    let id = snap.resolve_model(raw);
    if scoped_models(&snap, key).contains(&id) {
        return json_response(StatusCode::OK, model_object(&snap, id));
    }
    model_not_found()
}

#[derive(Deserialize)]
struct RequestHead {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    stream: Option<Value>,
}

struct Finish {
    m: MuxRef,
    label: String,
    key_id: String,
    started: Instant,
}

impl Finish {
    fn now(self, resp: Response<MuxBody>) -> Response<MuxBody> {
        self.record(resp.status().as_u16(), 0, 0, None);
        resp
    }

    fn record(&self, code: u16, prompt: i64, completion: i64, ttft: Option<f64>) {
        // Metrics use the full duration to the last byte; key usage goes to the batched writer.
        metrics::record_request(&self.label, code, self.started.elapsed().as_secs_f64(), prompt, completion);
        if let Some(t) = ttft {
            metrics::record_ttft(&self.label, t);
        }
        self.m.cache.record_usage(&self.key_id, prompt, completion);
    }
}

async fn route_post(
    m: MuxRef,
    req: Request<Incoming>,
    path: &str,
    key: &ResolvedKey,
    rid: &str,
) -> Response<MuxBody> {
    // Parse the body once, resolve the model, take the in-flight guard before the drain check, then proxy.
    let started = Instant::now();
    let declared = req
        .headers()
        .get(CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok());
    if declared.is_some_and(|n| n > m.max_body as u64) {
        return too_large();
    }
    let body = match Limited::new(req.into_body(), m.max_body).collect().await {
        Ok(c) => c.to_bytes(),
        Err(e) if e.downcast_ref::<LengthLimitError>().is_some() => return too_large(),
        Err(_) => return bad_request("could not read the request body"),
    };
    let head: RequestHead = match serde_json::from_slice(&body) {
        Ok(h) => h,
        Err(e) => return bad_request(&format!("request body is not valid JSON: {e}")),
    };
    let Some(raw_model) = head.model.filter(|s| !s.is_empty()) else {
        return bad_request("model is required");
    };
    let mut snap = m.cache.current();
    if !snap.is_known(snap.resolve_model(&raw_model)) {
        snap = m.cache.exact();
    }
    let model = snap.resolve_model(&raw_model).to_string();
    let finish = Finish {
        m,
        label: if snap.is_known(&model) { model.clone() } else { metrics::UNKNOWN_MODEL.to_string() },
        key_id: key.key_id.clone(),
        started,
    };
    if !security::model_allowed(&key.model_scope, &model) {
        return finish.now(model_not_found());
    }
    let mut guard = inflight::track(&model);
    let snap = m.cache.exact();
    if !snap.has_key(key) {
        drop(guard);
        return finish.now(revoked());
    }
    let routes = snap.routes(&model);
    let live: Vec<&Route> = routes.iter().filter(|r| !r.draining).collect();
    if live.is_empty() {
        drop(guard);
        return finish.now(if routes.is_empty() { model_not_found() } else { draining() });
    }
    let ids: Vec<&str> = live.iter().map(|r| r.instance_id.as_str()).collect();
    let route = live[guard.pick(&ids).unwrap_or(0)].clone();
    let stream = matches!(head.stream, Some(Value::Bool(true)));
    proxy(m, path, &route, body, rid, stream, guard, finish).await
}

#[allow(clippy::too_many_arguments)]
async fn proxy(
    m: MuxRef,
    path: &str,
    route: &Route,
    body: Bytes,
    rid: &str,
    stream: bool,
    guard: inflight::InflightGuard,
    finish: Finish,
) -> Response<MuxBody> {
    // Forward with the engine's own key (never the client's) and meter the reply as it streams back.
    let mut builder = Request::builder()
        .method(Method::POST)
        .uri(format!("http://127.0.0.1:{}{path}", route.port))
        .header(CONTENT_TYPE, "application/json")
        .header(X_REQUEST_ID, rid);
    if let Some(k) = &route.engine_key {
        builder = builder.header(AUTHORIZATION, format!("Bearer {k}"));
    }
    let Ok(req) = builder.body(Full::new(body)) else {
        return finish.now(error(StatusCode::BAD_GATEWAY, "invalid upstream request", "upstream_error", "bad_gateway"));
    };
    let resp = match tokio::time::timeout(UPSTREAM_TIMEOUT, m.client.request(req)).await {
        Ok(Ok(r)) => r,
        Ok(Err(_)) => {
            return finish.now(error(StatusCode::BAD_GATEWAY, "engine unreachable", "upstream_error", "bad_gateway"));
        }
        Err(_) => {
            return finish.now(error(
                StatusCode::GATEWAY_TIMEOUT,
                "engine did not answer in time",
                "timeout_error",
                "gateway_timeout",
            ));
        }
    };
    let (mut parts, upstream) = resp.into_parts();
    for name in ["connection", "keep-alive", "proxy-connection", "transfer-encoding", "te", "trailer", "upgrade"] {
        parts.headers.remove(name);
    }
    let sse = match parts.headers.get(CONTENT_TYPE).and_then(|v| v.to_str().ok()) {
        Some(ct) => ct.starts_with("text/event-stream"),
        None => stream,
    };
    let code = parts.status.as_u16();
    let body = MeteredBody {
        inner: upstream,
        meter: UsageMeter::new(sse, finish.started),
        done: Some((finish, guard, code)),
        sse,
        cut: false,
        cut_event: if is_anthropic(path) { ANTHROPIC_CUT_EVENT } else { STREAM_CUT_EVENT },
    };
    Response::from_parts(parts, body.boxed())
}

struct MeteredBody {
    inner: Incoming,
    meter: UsageMeter,
    done: Option<(Finish, inflight::InflightGuard, u16)>,
    sse: bool,
    cut: bool,
    cut_event: &'static [u8],
}

const STREAM_CUT_EVENT: &[u8] = b"data: {\"error\":{\"message\":\"the engine stopped during this stream (model unloaded or engine crashed)\",\"type\":\"unavailable_error\",\"code\":\"engine_stopped\"}}\n\ndata: [DONE]\n\n";
const ANTHROPIC_CUT_EVENT: &[u8] = b"event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"api_error\",\"message\":\"the engine stopped during this stream (model unloaded or engine crashed)\"}}\n\n";

impl MeteredBody {
    fn finish(&mut self) {
        // Record once, at end of stream or when the client goes away; this also releases the in-flight slot.
        if let Some((finish, guard, code)) = self.done.take() {
            let (prompt, completion) = self.meter.totals();
            finish.record(code, prompt, completion, self.meter.ttft());
            drop(guard);
        }
    }
}

impl Body for MeteredBody {
    type Data = Bytes;
    type Error = hyper::Error;

    fn poll_frame(self: Pin<&mut Self>, cx: &mut Context) -> Poll<Option<Result<Frame<Bytes>, hyper::Error>>> {
        // Frames pass through while data is metered; a cut SSE stream ends with an error event in its API's shape, not a reset.
        let this = self.get_mut();
        if this.cut {
            return Poll::Ready(None);
        }
        let polled = Pin::new(&mut this.inner).poll_frame(cx);
        match &polled {
            Poll::Ready(Some(Ok(frame))) => {
                if let Some(data) = frame.data_ref() {
                    this.meter.feed(data);
                }
            }
            Poll::Ready(Some(Err(_))) if this.sse => {
                this.finish();
                this.cut = true;
                return Poll::Ready(Some(Ok(Frame::data(Bytes::from_static(this.cut_event)))));
            }
            Poll::Ready(None) | Poll::Ready(Some(Err(_))) => this.finish(),
            Poll::Pending => {}
        }
        polled
    }

    fn is_end_stream(&self) -> bool {
        self.cut || self.inner.is_end_stream()
    }

    fn size_hint(&self) -> SizeHint {
        self.inner.size_hint()
    }
}

impl Drop for MeteredBody {
    fn drop(&mut self) {
        self.finish();
    }
}

#[derive(Deserialize)]
struct Usage {
    #[serde(default, alias = "input_tokens")]
    prompt_tokens: Option<i64>,
    #[serde(default, alias = "output_tokens")]
    completion_tokens: Option<i64>,
}

#[derive(Deserialize)]
struct UsageEnvelope {
    #[serde(default)]
    usage: Option<Usage>,
}

#[derive(Deserialize)]
struct StreamChunk {
    #[serde(default, rename = "type")]
    kind: Option<String>,
    #[serde(default)]
    usage: Option<Usage>,
    #[serde(default)]
    message: Option<UsageEnvelope>,
    #[serde(default)]
    response: Option<UsageEnvelope>,
    #[serde(default)]
    choices: Vec<StreamChoice>,
}

impl StreamChunk {
    fn has_output(&self) -> bool {
        // OpenAI choices with text, an Anthropic content_block_delta, or a Responses "*.delta" event.
        self.choices.iter().any(StreamChoice::has_output)
            || self.kind.as_deref().is_some_and(|k| k == "content_block_delta" || k.ends_with(".delta"))
    }
}

#[derive(Deserialize)]
struct StreamChoice {
    #[serde(default)]
    delta: Option<Delta>,
    #[serde(default)]
    text: Option<String>,
}

#[derive(Deserialize)]
struct Delta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    tool_calls: Option<serde::de::IgnoredAny>,
}

impl StreamChoice {
    fn has_output(&self) -> bool {
        let filled = |s: &Option<String>| s.as_deref().is_some_and(|t| !t.is_empty());
        filled(&self.text)
            || self
                .delta
                .as_ref()
                .is_some_and(|d| filled(&d.content) || filled(&d.reasoning_content) || d.tool_calls.is_some())
    }
}

struct UsageMeter {
    sse: bool,
    started: Instant,
    buf: Vec<u8>,
    prompt: Option<i64>,
    completion: Option<i64>,
    chunks: i64,
    first_token: Option<Duration>,
    overflow: bool,
}

impl UsageMeter {
    fn new(sse: bool, started: Instant) -> Self {
        Self {
            sse,
            started,
            buf: Vec::new(),
            prompt: None,
            completion: None,
            chunks: 0,
            first_token: None,
            overflow: false,
        }
    }

    fn feed(&mut self, data: &[u8]) {
        // SSE: account each complete line as it arrives; JSON: buffer (capped) for one parse at the end.
        if self.overflow {
            return;
        }
        self.buf.extend_from_slice(data);
        if !self.sse {
            if self.buf.len() > JSON_USAGE_CAP {
                self.overflow = true;
                self.buf = Vec::new();
            }
            return;
        }
        let Some(last) = self.buf.iter().rposition(|&b| b == b'\n') else {
            if self.buf.len() > SSE_LINE_CAP {
                self.buf.clear();
            }
            return;
        };
        let complete: Vec<u8> = self.buf.drain(..=last).collect();
        for line in complete.split(|&b| b == b'\n') {
            self.sse_line(line);
        }
    }

    fn sse_line(&mut self, line: &[u8]) {
        // Reported usage wins (Anthropic splits it over message_start and message_delta); else each output chunk is one token.
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        let Some(data) = line.strip_prefix(b"data:") else {
            return;
        };
        let data = data.trim_ascii();
        if data == b"[DONE]" {
            return;
        }
        let Ok(chunk) = serde_json::from_slice::<StreamChunk>(data) else {
            return;
        };
        let has_output = chunk.has_output();
        for u in [chunk.usage, chunk.message.and_then(|e| e.usage), chunk.response.and_then(|e| e.usage)].into_iter().flatten() {
            self.prompt = u.prompt_tokens.or(self.prompt);
            self.completion = u.completion_tokens.or(self.completion);
        }
        if has_output {
            self.chunks += 1;
            if self.first_token.is_none() {
                self.first_token = Some(self.started.elapsed());
            }
        }
    }

    fn totals(&mut self) -> (i64, i64) {
        // (prompt, completion): JSON usage, or for SSE the usage chunk, else the count of output chunks.
        if !self.sse {
            return serde_json::from_slice::<UsageEnvelope>(&self.buf)
                .ok()
                .and_then(|e| e.usage)
                .map_or((0, 0), |u| (u.prompt_tokens.unwrap_or(0), u.completion_tokens.unwrap_or(0)));
        }
        let rest = std::mem::take(&mut self.buf);
        self.sse_line(&rest);
        if self.prompt.is_none() && self.completion.is_none() {
            return (0, self.chunks);
        }
        (self.prompt.unwrap_or(0), self.completion.unwrap_or(self.chunks))
    }

    fn ttft(&self) -> Option<f64> {
        self.first_token.map(|d| d.as_secs_f64())
    }
}

fn request_id(headers: &HeaderMap) -> String {
    // Keep a sane client X-Request-Id (1-128 visible ASCII), else mint a process-unique random one.
    static PREFIX: OnceLock<String> = OnceLock::new();
    static SEQ: AtomicU64 = AtomicU64::new(0);
    if let Some(id) = headers
        .get(X_REQUEST_ID)
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
        .filter(|s| !s.is_empty() && s.len() <= 128 && s.bytes().all(|b| b.is_ascii_graphic()))
    {
        return id.to_string();
    }
    let prefix = PREFIX.get_or_init(|| {
        let mut b = [0u8; 8];
        unsafe {
            libc::getrandom(b.as_mut_ptr().cast(), b.len(), 0);
        }
        b.iter().fold(String::from("req-"), |mut s, x| {
            let _ = write!(s, "{x:02x}");
            s
        })
    });
    format!("{prefix}{:08x}", SEQ.fetch_add(1, Ordering::Relaxed))
}

async fn health_ready(m: MuxRef, query: Option<&str>) -> Response<MuxBody> {
    // Ready when the route cache has a live instance and a cheap keyed engine probe succeeds (throttled per model).
    let Some(raw) = query.and_then(|q| query_param(q, "model")).filter(|s| !s.is_empty()) else {
        return bad_request("model query parameter required");
    };
    let snap = m.cache.exact();
    let model = snap.resolve_model(&raw).to_string();
    let route = snap.routes(&model).iter().find(|r| !r.draining).cloned();
    let ready = match route {
        Some(r) => probe_ready(m, &model, &r).await,
        None => false,
    };
    let (status, word) = if ready {
        (StatusCode::OK, "ready")
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, "not_ready")
    };
    json_response(status, json!({"status": word, "model": model}))
}

async fn probe_ready(m: &Mux, model: &str, route: &Route) -> bool {
    // At most one engine probe per model per READY_TTL; callers during a probe reuse the last answer.
    {
        let mut cache = lock(&m.ready);
        let last = cache.get(model).copied();
        if let Some((_, ok)) = last.filter(|(at, _)| at.elapsed() < READY_TTL) {
            return ok;
        }
        cache.insert(model.to_string(), (Instant::now(), last.is_some_and(|(_, ok)| ok)));
    }
    let mut builder = Request::builder()
        .method(Method::GET)
        .uri(format!("http://127.0.0.1:{}/v1/models", route.port));
    if let Some(k) = &route.engine_key {
        builder = builder.header(AUTHORIZATION, format!("Bearer {k}"));
    }
    let ok = match builder.body(Full::new(Bytes::new())) {
        Ok(req) => matches!(
            tokio::time::timeout(PROBE_TIMEOUT, m.client.request(req)).await,
            Ok(Ok(r)) if r.status().is_success()
        ),
        Err(_) => false,
    };
    lock(&m.ready).insert(model.to_string(), (Instant::now(), ok));
    ok
}

fn query_param(query: &str, name: &str) -> Option<String> {
    query.split('&').find_map(|pair| {
        let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
        (percent_decode(k, true) == name).then(|| percent_decode(v, true))
    })
}

fn percent_decode(s: &str, form: bool) -> String {
    // %XX escapes (and '+' as space in query strings); malformed escapes are kept literally.
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        let hex = |c: u8| (c as char).to_digit(16);
        match b[i] {
            b'%' if i + 2 < b.len() => match (hex(b[i + 1]), hex(b[i + 2])) {
                (Some(h), Some(l)) => {
                    out.push((h * 16 + l) as u8);
                    i += 3;
                }
                _ => {
                    out.push(b'%');
                    i += 1;
                }
            },
            b'+' if form => {
                out.push(b' ');
                i += 1;
            }
            c => {
                out.push(c);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

async fn health_admin(m: MuxRef, headers: &HeaderMap) -> Response<MuxBody> {
    // Admin keys only (anything else gets a plain 404); DB and NVML work runs on the blocking pool.
    let key = client_token(headers).and_then(|t| m.cache.resolve_token(t));
    if !key.is_some_and(|k| k.is_admin && !security::key_expired(&k.expires_at)) {
        return admin_not_found();
    }
    match tokio::task::spawn_blocking(admin_report).await {
        Ok(Some(v)) => json_response(StatusCode::OK, v),
        _ => admin_not_found(),
    }
}

fn admin_report() -> Option<Value> {
    // GPU memory split (managed vs foreign), instances with in-flight counts, and installed-but-idle models.
    let home = goethe_home();
    let con = db::open(&home).ok()?;
    let managed = placement::managed_pids(&con);
    let snaps = hw::list_gpus_for_placement(&managed).ok()?;
    let names = gpu_names_by_index(&home);
    let mut gpus = Vec::new();
    for snap in snaps {
        let idx = snap.index;
        let (total_mb, used_mb) = hw::refresh_gpu_vram(idx)
            .map(|(t, u, _, _)| (t, u))
            .unwrap_or((0, 0));
        let used_managed_mb = hw::managed_vram_listed_mb(idx, &managed).unwrap_or(0);
        gpus.push(json!({
            "index": idx,
            "name": names.get(&idx).cloned().unwrap_or_default(),
            "total_mb": total_mb,
            "used_managed_mb": used_managed_mb,
            "used_foreign_mb": used_mb.saturating_sub(used_managed_mb),
        }));
    }
    let mut instances = Vec::new();
    if let Ok(mut stmt) = con.prepare(
        "SELECT instance_id, model_id, pid, status, gpu_indices, internal_port FROM loaded_instances ORDER BY model_id, instance_id",
    ) {
        if let Ok(rows) = stmt.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, i64>(2)?,
                r.get::<_, String>(3)?,
                r.get::<_, String>(4)?,
                r.get::<_, i64>(5)?,
            ))
        }) {
            for (instance_id, model_id, pid, status, gpu_indices, internal_port) in rows.flatten() {
                instances.push(json!({
                    "in_flight": inflight::instance_count(&instance_id),
                    "instance_id": instance_id,
                    "model_id": model_id,
                    "pid": pid,
                    "status": status,
                    "gpu_indices": gpu_indices,
                    "internal_port": internal_port,
                }));
            }
        }
    }
    let mut inactive_models = Vec::new();
    if let Ok(mut stmt) = con.prepare(
        "SELECT model_id FROM installed_models WHERE model_id NOT IN \
         (SELECT DISTINCT model_id FROM loaded_instances WHERE status = 'RUNNING') ORDER BY model_id",
    ) {
        if let Ok(rows) = stmt.query_map([], |r| r.get::<_, String>(0)) {
            inactive_models = rows.flatten().collect();
        }
    }
    Some(json!({"gpus": gpus, "instances": instances, "inactive_models": inactive_models}))
}

fn gpu_names_by_index(home: &std::path::Path) -> HashMap<u32, String> {
    let Ok(data) = std::fs::read_to_string(home.join("hardware_state.json")) else {
        return HashMap::new();
    };
    let Ok(v) = serde_json::from_str::<Value>(&data) else {
        return HashMap::new();
    };
    let Some(arr) = v.get("gpus").and_then(|g| g.as_array()) else {
        return HashMap::new();
    };
    arr.iter()
        .map(|g| {
            let idx = g.get("index").and_then(|x| x.as_u64()).unwrap_or(0) as u32;
            let name = g.get("name").and_then(|x| x.as_str()).unwrap_or("").to_string();
            (idx, name)
        })
        .collect()
}

fn full(bytes: impl Into<Bytes>) -> MuxBody {
    Full::new(bytes.into()).map_err(|never| match never {}).boxed()
}

fn json_response(status: StatusCode, v: Value) -> Response<MuxBody> {
    let (mut parts, body) = Response::new(full(serde_json::to_vec(&v).unwrap_or_default())).into_parts();
    parts.status = status;
    parts.headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    Response::from_parts(parts, body)
}

#[derive(Clone)]
struct GoetheError(String);

fn error(status: StatusCode, message: &str, kind: &str, code: &str) -> Response<MuxBody> {
    let mut resp = json_response(
        status,
        json!({"error": {"message": message, "type": kind, "param": null, "code": code}}),
    );
    resp.extensions_mut().insert(GoetheError(message.to_string()));
    resp
}

fn anthropic_error(resp: Response<MuxBody>) -> Response<MuxBody> {
    // Goethe's own errors on Anthropic routes in the Messages API shape; engine replies pass through untouched.
    let Some(GoetheError(message)) = resp.extensions().get::<GoetheError>().cloned() else {
        return resp;
    };
    let kind = match resp.status().as_u16() {
        400 | 405 => "invalid_request_error",
        401 => "authentication_error",
        403 => "permission_error",
        404 => "not_found_error",
        413 => "request_too_large",
        429 => "rate_limit_error",
        503 => "overloaded_error",
        _ => "api_error",
    };
    let (parts, _) = resp.into_parts();
    let body = json!({"type": "error", "error": {"type": kind, "message": message}});
    Response::from_parts(parts, full(serde_json::to_vec(&body).unwrap_or_default()))
}

fn not_found() -> Response<MuxBody> {
    error(StatusCode::NOT_FOUND, "Not Found", "invalid_request_error", "not_found")
}

fn model_not_found() -> Response<MuxBody> {
    error(StatusCode::NOT_FOUND, "Model Not Found", "invalid_request_error", "model_not_found")
}

fn admin_not_found() -> Response<MuxBody> {
    not_found()
}

fn bad_request(message: &str) -> Response<MuxBody> {
    error(StatusCode::BAD_REQUEST, message, "invalid_request_error", "bad_request")
}

fn too_large() -> Response<MuxBody> {
    error(StatusCode::PAYLOAD_TOO_LARGE, "Request body too large", "invalid_request_error", "request_too_large")
}

fn method_not_allowed(allowed: &Method) -> Response<MuxBody> {
    let mut resp = error(StatusCode::METHOD_NOT_ALLOWED, "Method Not Allowed", "invalid_request_error", "method_not_allowed");
    if let Ok(v) = HeaderValue::from_str(allowed.as_str()) {
        resp.headers_mut().insert(ALLOW, v);
    }
    resp
}

fn rate_limited(retry_after_secs: u64) -> Response<MuxBody> {
    let mut resp = error(StatusCode::TOO_MANY_REQUESTS, "Rate limit exceeded", "rate_limit_error", "rate_limit_exceeded");
    resp.headers_mut().insert(RETRY_AFTER, HeaderValue::from(retry_after_secs));
    resp
}

fn draining() -> Response<MuxBody> {
    let mut resp = error(StatusCode::SERVICE_UNAVAILABLE, "Model is draining", "unavailable_error", "model_draining");
    resp.headers_mut().insert(RETRY_AFTER, HeaderValue::from_static("5"));
    resp
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bearer_parsing() {
        assert_eq!(parse_bearer("Bearer abc"), Some("abc"));
        assert_eq!(parse_bearer("bearer abc"), Some("abc"));
        assert_eq!(parse_bearer("BEARER   abc  "), Some("abc"));
        assert_eq!(parse_bearer("  Bearer\tabc"), Some("abc"));
        assert_eq!(parse_bearer("Bearer"), None);
        assert_eq!(parse_bearer("Bearer "), None);
        assert_eq!(parse_bearer("Basic abc"), None);
        assert_eq!(parse_bearer("Bearer a b"), None);
        assert_eq!(parse_bearer("Bearerabc"), None);
    }

    fn meter_feed(sse: bool, parts: &[&[u8]]) -> UsageMeter {
        let mut m = UsageMeter::new(sse, Instant::now());
        for p in parts {
            m.feed(p);
        }
        m
    }

    #[test]
    fn sse_usage_chunk_wins() {
        let mut m = meter_feed(
            true,
            &[
                b"data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\n\n",
                b"data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"}}],\"usage\":null}\n\nda",
                b"ta: {\"choices\":[{\"delta\":{\"content\":\" there\"}}]}\r\n\r\n",
                b"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":11,\"total_tokens\":18}}\n\n",
                b"data: [DONE]\n\n",
            ],
        );
        assert_eq!(m.totals(), (7, 11));
        assert!(m.ttft().is_some());
    }

    #[test]
    fn sse_counts_delta_chunks_without_usage() {
        let mut m = meter_feed(
            true,
            &[
                b"data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n",
                b"data: {\"choices\":[{\"delta\":{\"content\":\"model=m \"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"con",
                b"tent\":\" t1\\n\"}}]}\n\n: keepalive\n\n",
                b"data: {\"choices\":[{\"text\":\"x\"}]}\n\n",
                b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0}]}}]}\n\n",
                b"data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"r\"}}]}",
            ],
        );
        assert_eq!(m.totals(), (0, 5));
    }

    #[test]
    fn json_usage_is_read_once_at_the_end() {
        let mut m = meter_feed(false, &[b"{\"id\":\"x\",\"choices\":[],", b"\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2}}"]);
        assert_eq!(m.totals(), (3, 2));
        assert_eq!(m.ttft(), None);
        let mut bad = meter_feed(false, &[b"not json"]);
        assert_eq!(bad.totals(), (0, 0));
    }

    #[test]
    fn anthropic_and_responses_usage_is_metered() {
        let mut a = meter_feed(
            true,
            &[
                b"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":9,\"output_tokens\":1}}}\n\n",
                b"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}\n\n",
                b"event: message_delta\ndata: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":4}}\n\n",
            ],
        );
        assert_eq!(a.totals(), (9, 4));
        assert!(a.ttft().is_some());
        let mut r = meter_feed(
            true,
            &[
                b"data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hi\"}\n\n",
                b"data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":5,\"output_tokens\":2}}}\n\n",
            ],
        );
        assert_eq!(r.totals(), (5, 2));
        let mut j = meter_feed(false, &[b"{\"type\":\"message\",\"usage\":{\"input_tokens\":3,\"output_tokens\":8}}"]);
        assert_eq!(j.totals(), (3, 8));
    }

    #[test]
    fn litellm_routes_and_key_headers() {
        let proxied = |p: &str| match api_route(p) {
            Some((Method::POST, Api::Proxy(up))) => Some(up),
            _ => None,
        };
        assert_eq!(proxied("/chat/completions").as_deref(), Some("/v1/chat/completions"));
        assert_eq!(proxied("/v1/chat/completions").as_deref(), Some("/v1/chat/completions"));
        assert_eq!(proxied("/v1/messages").as_deref(), Some("/v1/messages"));
        assert_eq!(proxied("/responses").as_deref(), Some("/v1/responses"));
        assert!(matches!(api_route("/models"), Some((Method::GET, Api::Models))));
        assert!(matches!(api_route("/v1/models/org%2Fm"), Some((_, Api::Model(id))) if id == "org/m"));
        assert!(matches!(api_route("/model/info"), Some((_, Api::ModelInfo { group: false }))));
        assert!(matches!(api_route("/v1/model_group/info"), Some((_, Api::ModelInfo { group: true }))));
        assert!(api_route("/v1").is_none() && api_route("/v1x/models").is_none() && api_route("/metrics").is_none());
        assert!(is_anthropic("/messages") && is_anthropic("/v1/messages/count_tokens") && !is_anthropic("/v1/responses"));

        let with = |pairs: &[(&'static str, &'static str)]| {
            let mut h = HeaderMap::new();
            for (k, v) in pairs {
                h.insert(*k, HeaderValue::from_static(v));
            }
            h
        };
        assert_eq!(client_token(&with(&[("authorization", "Bearer a")])), Some("a"));
        assert_eq!(client_token(&with(&[("x-litellm-api-key", "Bearer b")])), Some("b"));
        assert_eq!(client_token(&with(&[("x-litellm-api-key", "b2")])), Some("b2"));
        assert_eq!(client_token(&with(&[("x-api-key", "c")])), Some("c"));
        assert_eq!(client_token(&with(&[("api-key", "d")])), Some("d"));
        assert_eq!(client_token(&with(&[("authorization", "Basic z"), ("x-api-key", "c")])), Some("c"));
        assert_eq!(client_token(&with(&[("x-api-key", "a b")])), None);
        assert_eq!(client_token(&HeaderMap::new()), None);
    }

    #[tokio::test]
    async fn goethe_errors_take_the_anthropic_shape_on_messages() {
        let body = |r: Response<MuxBody>| async move {
            serde_json::from_slice::<Value>(&r.into_body().collect().await.unwrap().to_bytes()).unwrap()
        };
        let resp = anthropic_error(rate_limited(3));
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(resp.headers()[RETRY_AFTER], "3");
        assert_eq!(
            body(resp).await,
            json!({"type": "error", "error": {"type": "rate_limit_error", "message": "Rate limit exceeded"}})
        );
        let engine = anthropic_error(json_response(StatusCode::OK, json!({"ok": true})));
        assert_eq!(body(engine).await, json!({"ok": true}));
    }

    #[test]
    fn gcra_limits_per_key_and_skips_unlimited() {
        let mut t = HashMap::new();
        let now = Instant::now();
        assert_eq!(gcra_check(&mut t, "k", 2, now), None);
        assert_eq!(gcra_check(&mut t, "k", 2, now + Duration::from_millis(5)), None);
        let retry = gcra_check(&mut t, "k", 2, now + Duration::from_millis(10)).expect("third request limited");
        assert!((1..=30).contains(&retry), "retry {retry}");
        assert_eq!(gcra_check(&mut t, "other", 2, now), None, "keys are independent");
        assert_eq!(gcra_check(&mut t, "k", 2, now + Duration::from_secs(31)), None, "refills over time");
        for _ in 0..1000 {
            assert_eq!(gcra_check(&mut t, "free", 0, now), None);
        }
        assert!(!t.contains_key("free"), "unlimited keys keep no state");
    }

    #[test]
    fn query_values_are_url_decoded() {
        assert_eq!(query_param("model=org%2Fname", "model").as_deref(), Some("org/name"));
        assert_eq!(query_param("a=1&model=my+model&x", "model").as_deref(), Some("my model"));
        assert_eq!(query_param("x=1", "model"), None);
        assert_eq!(percent_decode("100%", true), "100%");
        assert_eq!(percent_decode("%zz%41", true), "%zzA");
        assert_eq!(percent_decode("a+b%2Fc", false), "a+b/c");
    }

    #[test]
    fn request_ids_pass_through_or_are_minted() {
        let mut h = HeaderMap::new();
        h.insert(X_REQUEST_ID, HeaderValue::from_static("abc-123"));
        assert_eq!(request_id(&h), "abc-123");
        h.insert(X_REQUEST_ID, HeaderValue::from_static("has space"));
        let minted = request_id(&h);
        assert!(minted.starts_with("req-") && minted.len() == 28, "{minted}");
        assert_ne!(request_id(&HeaderMap::new()), minted);
    }
}
