// API multiplexer on 127.0.0.1 (persisted mux port): OpenAI, Anthropic and LiteLLM routes, key auth, metering, tracing, session notes and upstream auth.

use crate::curator::{self, Gates};
use goethe_core::db;
use goethe_core::engines;
use goethe_core::gateway_cache::{GatewayCache, Route, Snapshot};
use goethe_core::hw;
use goethe_core::inflight;
use goethe_core::learn::{self, Store};
use goethe_core::metrics;
use goethe_core::nginx;
use goethe_core::paths::goethe_home;
use goethe_core::placement;
use goethe_core::public_share;
use goethe_core::security::{self, ResolvedKey};
use goethe_core::trace::{self, Capture, Recorder};
use goethe_core::trace_assemble::Api as TraceApi;
use goethe_core::usage_log::{self, Dims, Sample};
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
use std::net::{IpAddr, SocketAddr};
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::task::{Context, Poll};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::net::{TcpListener, UnixListener};
use tokio::sync::oneshot;

type MuxBody = http_body_util::combinators::BoxBody<Bytes, hyper::Error>;
pub type Upstream = Client<HttpConnector, Full<Bytes>>;
pub type MuxRef = &'static Mux;

const HEADER_READ_TIMEOUT: Duration = Duration::from_secs(10);
const UPSTREAM_IDLE: Duration = Duration::from_secs(4);
const PROBE_TIMEOUT: Duration = Duration::from_secs(2);
const READY_TTL: Duration = Duration::from_secs(1);
const JSON_USAGE_CAP: usize = 8 << 20;
const SSE_LINE_CAP: usize = 1 << 20;
const REBUILD_CAP: usize = 64 << 20;
const X_REQUEST_ID: HeaderName = HeaderName::from_static("x-request-id");
const X_SESSION_ID: HeaderName = HeaderName::from_static("x-session-id");
const ROUTER_MS_CAP: u64 = 600_000;

pub struct Mux {
    pub cache: GatewayCache,
    pub client: Upstream,
    max_body: usize,
    rpm: Mutex<HashMap<String, Instant>>,
    ready: Mutex<HashMap<String, (Instant, bool)>>,
    trace: Recorder,
    pub learn: Arc<Store>,
    pub learn_gates: Gates,
}

static MUX: OnceLock<Mux> = OnceLock::new();

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

pub fn spawn_mux() -> Result<tokio::task::JoinHandle<()>, String> {
    // Bind the daemon's persisted mux port now so startup fails fast, then serve it and the public listener in one task, so aborting it (upgrade drain) stops both.
    let home = goethe_home();
    let port = nginx::persisted_mux_port(&home)?;
    raise_nofile_limit();
    let listener = std::net::TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], port))).map_err(|e| {
        format!("mux bind failed on 127.0.0.1:{port} ({e}); set GOETHE_MUX_PORT to a free port")
    })?;
    listener.set_nonblocking(true).map_err(|e| e.to_string())?;
    let listener = TcpListener::from_std(listener).map_err(|e| e.to_string())?;
    let public = public_share::bind_public_mux(&home)
        .and_then(|l| l.set_nonblocking(true).map(|()| l).map_err(|e| e.to_string()))
        .and_then(|l| UnixListener::from_std(l).map_err(|e| e.to_string()))
        .map_err(|e| eprintln!("goethe mux: public gateway listener unavailable ({e})"))
        .ok();
    let max_body = nginx::persist_max_body_mb(&home) as usize * 1024 * 1024;
    nginx::persist_gateway_limits(&home);
    let mux = MUX.get_or_init(|| Mux {
        cache: GatewayCache::open(&home),
        client: upstream_client(),
        max_body,
        rpm: Mutex::new(HashMap::new()),
        ready: Mutex::new(HashMap::new()),
        trace: Recorder::start(&home),
        learn: Store::start(&home),
        learn_gates: Gates::default(),
    });
    let _ = mux.cache.exact();
    curator::spawn_ticker(mux);
    Ok(tokio::spawn(async move {
        match public {
            Some(public) => {
                tokio::join!(serve(mux, listener), serve_public(mux, public));
            }
            None => serve(mux, listener).await,
        }
    }))
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
    // Accept forever; transient accept errors back off 50 ms instead of ending the mux.
    loop {
        match listener.accept().await {
            Ok((stream, _)) => {
                let _ = stream.set_nodelay(true);
                tokio::spawn(connection(m, stream, false));
            }
            Err(e) => accept_failed(e).await,
        }
    }
}

async fn serve_public(m: MuxRef, listener: UnixListener) {
    // The listener behind goethe api share: only nginx can open its socket.
    loop {
        match listener.accept().await {
            Ok((stream, _)) => {
                tokio::spawn(connection(m, stream, true));
            }
            Err(e) => accept_failed(e).await,
        }
    }
}

async fn accept_failed(e: std::io::Error) {
    eprintln!("goethe mux: accept failed: {e}");
    tokio::time::sleep(Duration::from_millis(50)).await;
}

async fn connection<S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin + Send + 'static>(m: MuxRef, stream: S, public: bool) {
    let svc = service_fn(move |req| async move { Ok::<_, Infallible>(handle(m, req, public).await) });
    let _ = http1::Builder::new().timer(TokioTimer::new()).header_read_timeout(HEADER_READ_TIMEOUT).serve_connection(TokioIo::new(stream), svc).await;
}

async fn handle(m: MuxRef, req: Request<Incoming>, public: bool) -> Response<MuxBody> {
    let rid = request_id(req.headers());
    let anthropic = is_anthropic(req.uri().path());
    let mut resp = if public {
        let path = req.uri().path().to_string();
        api(m, req, &path, &rid, true).await
    } else {
        dispatch(m, req, &rid).await
    };
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
        _ => api(m, req, &path, rid, false).await,
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
        | "/messages/count_tokens" | "/audio/transcriptions" => Some((Method::POST, Api::Proxy(format!("/v1{tail}")))),
        "/models" => Some((Method::GET, Api::Models)),
        "/model/info" => Some((Method::GET, Api::ModelInfo { group: false })),
        "/model_group/info" => Some((Method::GET, Api::ModelInfo { group: true })),
        t => t.strip_prefix("/models/").map(|id| (Method::GET, Api::Model(percent_decode(id, false)))),
    }
}

async fn api(m: MuxRef, req: Request<Incoming>, path: &str, rid: &str, public: bool) -> Response<MuxBody> {
    // Method check, key auth, the public gateway's window and key list, per-key rpm, then the model list or a proxied call.
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
    if public {
        let client = req.headers().get("x-real-ip").and_then(|v| v.to_str().ok()).and_then(|v| v.trim().parse().ok());
        match public_share::check(&goethe_home(), &key.key_id, key.is_admin, client) {
            public_share::Access::Allowed => {}
            public_share::Access::KeyExpired => {
                return error(StatusCode::FORBIDDEN, "This key's public sharing has ended", "forbidden", "key_public_ended")
            }
            public_share::Access::NotFromHere => {
                return error(StatusCode::FORBIDDEN, "This key may not be used from this address", "forbidden", "address_not_allowed")
            }
            public_share::Access::Closed => {
                return error(StatusCode::FORBIDDEN, "The public gateway is closed", "forbidden", "public_gateway_closed")
            }
            public_share::Access::KeyNotShared => {
                return error(StatusCode::FORBIDDEN, "This key is not allowed on the public gateway", "forbidden", "key_not_public")
            }
        }
    }
    if let Some(retry) = gcra_check(&mut lock(&m.rpm), &key.key_id, key.rpm_limit, Instant::now()) {
        metrics::inc_rate_limited(&key.key_id);
        let dims = Dims { model: usage_log::RATE_LIMITED.into(), api_key_id: key.key_id.clone(), ..Dims::default() };
        usage_log::record(&dims, &Sample::new(429, false, 0.0, None, 0, 0));
        return rate_limited(retry);
    }
    match route {
        Api::Models => list_models(m, &key),
        Api::Model(id) => get_model(m, &key, &id),
        Api::ModelInfo { group } => model_info(m, &key, group),
        Api::Proxy(upstream) => route_post(m, req, &upstream, Caller::Local { key: &key, public }, rid, None).await,
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
    dims: Dims,
    router_ms: Option<f64>,
    trailer: Option<oneshot::Sender<HeaderMap>>,
}

impl Finish {
    fn now(self, resp: Response<MuxBody>) -> Response<MuxBody> {
        self.record(resp.status().as_u16(), false, &Metered::default(), None);
        resp
    }

    fn record(&self, code: u16, cancelled: bool, u: &Metered, ttft: Option<f64>) {
        // Metrics and the usage log use the full duration to the last byte; key usage goes to the batched writer.
        let secs = self.started.elapsed().as_secs_f64();
        metrics::record_request(&self.label, code, secs, u.prompt, u.completion);
        let finish = match code {
            _ if cancelled => "cancelled",
            400.. => "error",
            _ => u.finish,
        };
        let dims = Dims {
            finish_reason: finish.into(),
            context_length_bucket: if u.prompt > 0 { usage_log::context_bucket(u.prompt).into() } else { String::new() },
            ..self.dims.clone()
        };
        let sample = Sample {
            code,
            cancelled,
            duration_s: secs,
            ttft_s: ttft,
            prompt: u.prompt,
            completion: u.completion,
            reasoning: u.reasoning,
            cached: u.cached,
            router_ms: self.router_ms,
        };
        usage_log::record(&dims, &sample);
        if let Some(t) = ttft {
            metrics::record_ttft(&self.label, t);
        }
        if !self.key_id.is_empty() {
            self.m.cache.record_usage(&self.key_id, u.prompt, u.completion);
        }
    }
}

pub struct Routed {
    pub request_id: String,
    pub key_id: String,
    pub user_id: String,
    pub dispatch_ms: Option<u64>,
    pub instance: Option<String>,
}

impl Routed {
    fn from_headers(h: &HeaderMap) -> Routed {
        // The ids the platform puts on a forwarded request; read only on the node connection, never on a local listener.
        let get = |name: &str| h.get(name).and_then(|v| v.to_str().ok()).map(str::trim).filter(|v| !v.is_empty() && v.len() <= 128).map(str::to_string);
        Routed {
            request_id: get(goethe_proto::REQUEST_ID_HEADER).unwrap_or_default(),
            key_id: get(goethe_proto::KEY_ID_HEADER).unwrap_or_default(),
            user_id: get(goethe_proto::USER_HEADER).unwrap_or_default(),
            dispatch_ms: get(goethe_proto::DISPATCH_HEADER).and_then(|v| v.parse().ok()),
            instance: get(goethe_proto::INSTANCE_HEADER),
        }
    }

    fn router_ms(&self, now_ms: u64) -> Option<f64> {
        // The platform's dispatch to our receipt, 0 to 10 minutes; a clock ahead of ours gives 0.
        self.dispatch_ms.map(|d| now_ms.saturating_sub(d).min(ROUTER_MS_CAP) as f64)
    }
}

#[derive(Clone, Copy)]
enum Caller<'a> {
    Local { key: &'a ResolvedKey, public: bool },
    Routed(&'a Routed),
}

impl Caller<'_> {
    fn key_id(&self) -> String {
        // Workspace key ids get a prefix so they never collide with this node's own key ids.
        match self {
            Caller::Local { key, .. } => key.key_id.clone(),
            Caller::Routed(r) if r.key_id.is_empty() => String::new(),
            Caller::Routed(r) => format!("ck:{}", r.key_id),
        }
    }
}

pub fn running() -> Option<MuxRef> {
    MUX.get()
}

pub fn gateway(m: MuxRef) -> goethe_core::client::Gateway {
    // The /v1 handler of the node connection (goethe-core client): the connection is the authentication, so its headers are trusted here and only here.
    Arc::new(move |req: Request<Incoming>| {
        let r = Routed::from_headers(req.headers());
        Box::pin(routed(m, req, r))
    })
}

async fn routed(m: MuxRef, req: Request<Incoming>, r: Routed) -> Response<MuxBody> {
    // A request the platform forwarded: no node-local key, address or rate check, otherwise the local path; a proxied answer ends with the goethe-usage trailer.
    let path = req.uri().path().to_string();
    let rid = if r.request_id.is_empty() { request_id(&HeaderMap::new()) } else { r.request_id.clone() };
    let (tx, rx) = oneshot::channel();
    let mut resp = match api_route(&path) {
        Some((allowed, _)) if req.method() != allowed => method_not_allowed(&allowed),
        Some((_, Api::Proxy(upstream))) => route_post(m, req, &upstream, Caller::Routed(&r), &rid, Some(tx)).await,
        Some((_, Api::Models)) => {
            let snap = m.cache.exact();
            let data: Vec<Value> = snap.serving_models().into_iter().map(|id| model_object(&snap, id)).collect();
            json_response(StatusCode::OK, json!({"object": "list", "data": data}))
        }
        _ => not_found(),
    };
    if is_anthropic(&path) {
        resp = anthropic_error(resp);
    }
    if let Ok(v) = HeaderValue::from_str(&rid) {
        resp.headers_mut().insert(X_REQUEST_ID, v);
    }
    let (parts, body) = resp.into_parts();
    Response::from_parts(parts, body.with_trailers(async move { rx.await.ok().map(Ok) }).boxed())
}

fn modality(path: &str) -> &'static str {
    match api_tail(path) {
        "/embeddings" => "embeddings",
        "/rerank" => "rerank",
        "/audio/transcriptions" => "audio",
        "/messages/count_tokens" => "count_tokens",
        _ => "chat",
    }
}

fn network_path(client: Option<IpAddr>) -> &'static str {
    // Where a direct request came from, by nginx's X-Real-IP: this machine, a private network, a Tailscale or WireGuard overlay (100.64.0.0/10, fd7a:115c:a1e0::/48), or the internet.
    let Some(ip) = client.map(|ip| ip.to_canonical()) else {
        return "local";
    };
    match ip {
        _ if ip.is_loopback() => "local",
        IpAddr::V4(a) if a.octets()[0] == 100 && a.octets()[1] & 0xc0 == 64 => "tailscale_wireguard",
        IpAddr::V6(a) if a.segments()[..3] == [0xfd7a, 0x115c, 0xa1e0] => "tailscale_wireguard",
        IpAddr::V4(a) if a.is_private() || a.is_link_local() => "lan",
        IpAddr::V6(a) if a.segments()[0] & 0xfe00 == 0xfc00 || a.segments()[0] & 0xffc0 == 0xfe80 => "lan",
        _ => "public",
    }
}

async fn route_post(
    m: MuxRef,
    req: Request<Incoming>,
    path: &str,
    caller: Caller<'_>,
    rid: &str,
    trailer: Option<oneshot::Sender<HeaderMap>>,
) -> Response<MuxBody> {
    // Parse the body once, resolve the model, take the in-flight guard before the drain check, then proxy; a routed request without a key id gets no session notes, which its callers would share.
    let started = Instant::now();
    let wall = SystemTime::now();
    let network = match caller {
        Caller::Local { public: true, .. } => "public",
        Caller::Local { .. } => network_path(req.headers().get("x-real-ip").and_then(|v| v.to_str().ok()).and_then(|v| v.trim().parse().ok())),
        Caller::Routed(_) => "routed",
    };
    let (key, routed) = match caller {
        Caller::Local { key, .. } => (Some(key), None),
        Caller::Routed(r) => (None, Some(r)),
    };
    let key_id = caller.key_id();
    let hint = session_hint(req.headers()).filter(|_| !key_id.is_empty());
    let audio = api_tail(path) == "/audio/transcriptions";
    let content_type = req.headers().get(CONTENT_TYPE).and_then(|v| v.to_str().ok()).unwrap_or("").to_string();
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
    let head: RequestHead = if audio {
        match form_field(&content_type, &body, "model") {
            Some(model) => RequestHead { model: Some(model), stream: None },
            None => return bad_request("a multipart/form-data body with a model field is required"),
        }
    } else {
        match serde_json::from_slice(&body) {
            Ok(h) => h,
            Err(e) => return bad_request(&format!("request body is not valid JSON: {e}")),
        }
    };
    let Some(raw_model) = head.model.filter(|s| !s.is_empty()) else {
        return bad_request("model is required");
    };
    let mut snap = m.cache.current();
    if !snap.is_known(snap.resolve_model(&raw_model)) {
        snap = m.cache.exact();
    }
    let model = snap.resolve_model(&raw_model).to_string();
    let label = if snap.is_known(&model) { model.clone() } else { metrics::UNKNOWN_MODEL.to_string() };
    let dims = Dims {
        model: label.clone(),
        engine: snap.engine(&model).map_or("", |e| e.id).into(),
        variant: snap.variant(&model).into(),
        api_key_id: key_id.clone(),
        user_id: routed.map(|r| r.user_id.clone()).unwrap_or_default(),
        modality: modality(path).into(),
        network_path: network.into(),
        ..Dims::default()
    };
    let now_ms = wall.duration_since(UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64);
    let router_ms = routed.and_then(|r| r.router_ms(now_ms));
    let finish = Finish { m, label, key_id: key.map(|k| k.key_id.clone()).unwrap_or_default(), started, dims, router_ms, trailer };
    if key.is_some_and(|k| !security::model_allowed(&k.model_scope, &model)) {
        return finish.now(model_not_found());
    }
    let mut guard = inflight::track(&model);
    let snap = m.cache.exact();
    if key.is_some_and(|k| !snap.has_key(k)) {
        drop(guard);
        return finish.now(revoked());
    }
    let routes = snap.routes(&model);
    let live: Vec<&Route> = routes.iter().filter(|r| !r.draining).collect();
    if live.is_empty() {
        drop(guard);
        return finish.now(if routes.is_empty() { model_not_found() } else { draining() });
    }
    let pinned = routed.and_then(|r| r.instance.as_deref()).and_then(|id| live.iter().position(|r| r.instance_id == id));
    let ids: Vec<&str> = match pinned {
        Some(i) => vec![live[i].instance_id.as_str()],
        None => live.iter().map(|r| r.instance_id.as_str()).collect(),
    };
    let route = live[pinned.unwrap_or(0) + guard.pick(&ids).unwrap_or(0)].clone();
    let engine = snap.engine(&model);
    if let Some(e) = engine.filter(|e| !e.routes.contains(&api_tail(path))) {
        drop(guard);
        return finish.now(unsupported("unsupported_task", None, format!("{path} is not served by {}", e.id)));
    }
    if let Some(why) = wrong_task(snap.task(&model), api_tail(path), &model) {
        drop(guard);
        return finish.now(unsupported("unsupported_task", None, why));
    }
    let openai = matches!(api_tail(path), "/chat/completions" | "/completions" | "/embeddings");
    if let Some((e, Err(param))) = engine.filter(|_| openai).map(|e| (e, engines::check_params(e, &body))) {
        drop(guard);
        return finish.now(unsupported("unsupported_parameter", Some(&param), format!("{param} is not supported by {}", e.id)));
    }
    let stream = matches!(head.stream, Some(Value::Bool(true)));
    let body = prepare_body(path, body);
    let (upstream_body, mode) = streamed(path, stream, &body);
    let (sent, retry, learn_note) = apply_note(m, &snap, path, &model, &key_id, hint.as_deref(), &upstream_body);
    let tap = snap.trace().records(&key_id).then(|| api_tail(path)).filter(|t| TraceApi::from_path(t).is_some()).map(|t| {
        let cfg = snap.trace();
        Box::new(Capture {
            dir: cfg.dir(&goethe_home()),
            max_gb: cfg.max_gb(),
            path: t.to_string(),
            request_id: rid.to_string(),
            key_id: key_id.clone(),
            instance_id: route.instance_id.clone(),
            model: model.clone(),
            session_hint: hint,
            learn_note,
            request: body.to_vec(),
            started_at: wall,
            duration_s: 0.0,
            ttft_s: None,
            status: 0,
            sse: false,
            complete: false,
            response: Vec::new(),
            truncated: false,
        })
    });
    let ctype = if audio { content_type.as_str() } else { "application/json" };
    proxy(m, path, &route, sent, retry, rid, mode, guard, finish, tap, ctype).await
}

fn form_field(content_type: &str, body: &[u8], name: &str) -> Option<String> {
    // The text value of one multipart/form-data field (RFC 7578): the one part whose Content-Disposition name matches exactly; a repeated field is refused, since parsers disagree on which copy wins.
    content_type.to_ascii_lowercase().starts_with("multipart/form-data").then_some(())?;
    let boundary = content_type.split(';').find_map(|p| p.trim().strip_prefix("boundary="))?.trim_matches('"');
    let delim = format!("\r\n--{boundary}");
    let find = |needle: &[u8], from: usize| body.get(from..)?.windows(needle.len()).position(|w| w == needle).map(|i| i + from);
    let mut head_start = find(&delim.as_bytes()[2..], 0)? + delim.len() - 2;
    let mut found = None;
    while body.get(head_start..head_start + 2) == Some(b"\r\n") {
        let head_end = find(b"\r\n\r\n", head_start)?;
        let next = find(delim.as_bytes(), head_end + 4)?;
        let named = std::str::from_utf8(&body[head_start..head_end]).ok().and_then(disposition_name) == Some(name);
        if named && found.replace(std::str::from_utf8(&body[head_end + 4..next]).ok()?.trim().to_string()).is_some() {
            return None;
        }
        head_start = next + delim.len();
    }
    found
}

fn disposition_name(head: &str) -> Option<&str> {
    // The name parameter of a part's Content-Disposition header (not filename).
    let (_, value) = head.split("\r\n").filter_map(|l| l.split_once(':')).find(|(k, _)| k.trim().eq_ignore_ascii_case("content-disposition"))?;
    value.split(';').find_map(|p| p.trim().strip_prefix("name=")).map(|n| n.trim_matches('"'))
}

struct NoteRetry {
    original: Bytes,
    key_id: String,
    session: String,
}

fn apply_note(
    m: MuxRef,
    snap: &Snapshot,
    path: &str,
    model: &str,
    key_id: &str,
    hint: Option<&str>,
    body: &Bytes,
) -> (Bytes, Option<NoteRetry>, Option<String>) {
    // Session notes need the model turned on and a client session id; anything unusual forwards the client's bytes unchanged.
    let unchanged = || (body.clone(), None, None);
    let (Some(session), true) = (hint, snap.learn().applies(model)) else {
        return unchanged();
    };
    let Some(api) = TraceApi::from_path(api_tail(path)).filter(|a| a.has_messages()) else {
        return unchanged();
    };
    let Some(step) = m.learn.observe(key_id, session, model, api, body, snap.context_length(model)) else {
        return unchanged();
    };
    if let Some(job) = step.job {
        curator::start(m, job);
    }
    if step.block.is_empty() {
        return unchanged();
    }
    let Some(spliced) = learn::splice(api, body, &step.block) else {
        return unchanged();
    };
    let retry = NoteRetry { original: body.clone(), key_id: key_id.to_string(), session: session.to_string() };
    (Bytes::from(spliced), Some(retry), Some(step.block))
}

fn session_hint(headers: &HeaderMap) -> Option<String> {
    // Optional client session id for the trace recorder: 1-128 visible ASCII, like X-Request-Id.
    headers
        .get(X_SESSION_ID)
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
        .filter(|s| !s.is_empty() && s.len() <= 128 && s.bytes().all(|b| b.is_ascii_graphic()))
        .map(str::to_string)
}

fn prepare_body(path: &str, body: Bytes) -> Bytes {
    // Anthropic rejects thinking budgets under 1024. Raise only that field; every other body is unchanged.
    if !is_anthropic(path) {
        return body;
    }
    let Ok(mut v) = serde_json::from_slice::<Value>(&body) else {
        return body;
    };
    let Some(budget) = v.get("thinking").and_then(|t| t.get("budget_tokens")).and_then(|n| n.as_u64()) else {
        return body;
    };
    if budget >= 1024 {
        return body;
    }
    if let Some(t) = v.get_mut("thinking") {
        if let Some(obj) = t.as_object_mut() {
            obj.insert("budget_tokens".to_string(), json!(1024));
        }
    }
    Bytes::from(serde_json::to_vec(&v).unwrap_or_else(|_| body.to_vec()))
}

#[allow(clippy::too_many_arguments)]
async fn proxy(
    m: MuxRef,
    path: &str,
    route: &Route,
    body: Bytes,
    retry: Option<NoteRetry>,
    rid: &str,
    mode: Mode,
    guard: inflight::InflightGuard,
    finish: Finish,
    mut tap: Option<Box<Capture>>,
    content_type: &str,
) -> Response<MuxBody> {
    // Forward with the engine's own key (never the client's), meter and maybe trace the reply; a 400 with a note resends the client's own request once.
    let send = |payload: Bytes| {
        let mut builder = Request::builder()
            .method(Method::POST)
            .uri(format!("http://127.0.0.1:{}{path}", route.port))
            .header(CONTENT_TYPE, content_type)
            .header(X_REQUEST_ID, rid);
        if let Some(k) = &route.engine_key {
            builder = builder.header(AUTHORIZATION, format!("Bearer {k}"));
        }
        builder.body(Full::new(payload)).map(|req| tokio::time::timeout(goethe_core::nginx::UPSTREAM_TIMEOUT, m.client.request(req)))
    };
    let Ok(call) = send(body) else {
        return finish.now(error(StatusCode::BAD_GATEWAY, "invalid upstream request", "upstream_error", "bad_gateway"));
    };
    let mut answered = call.await;
    if let (Some(r), Ok(Ok(first))) = (retry, &answered) {
        if first.status() == StatusCode::BAD_REQUEST {
            if let Ok(call) = send(r.original) {
                if let Some(c) = tap.as_mut() {
                    c.learn_note = None;
                }
                answered = call.await;
                if matches!(&answered, Ok(Ok(second)) if second.status().is_success()) {
                    m.learn.note_rejected(&r.key_id, &r.session);
                }
            }
        }
    }
    let resp = match answered {
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
        None => mode != Mode::Pass(false),
    };
    let rebuild = (mode == Mode::Rebuild && sse).then(|| TraceApi::from_path(api_tail(path))).flatten();
    let hide = (mode == Mode::HideUsage && sse).then(UsageFilter::default);
    if rebuild.is_some() || hide.is_some() {
        parts.headers.remove(CONTENT_LENGTH);
    }
    let code = parts.status.as_u16();
    let mut body = MeteredBody {
        inner: upstream,
        meter: UsageMeter::new(sse, finish.started),
        done: Some((finish, guard, code)),
        sse,
        cut: false,
        ended: false,
        tap,
        cut_event: if is_anthropic(path) { ANTHROPIC_CUT_EVENT } else { STREAM_CUT_EVENT },
        hide,
        defer: rebuild.is_some(),
    };
    match rebuild {
        Some(api) => rebuild_reply(parts, &mut body, api, is_anthropic(path)).await,
        None => Response::from_parts(parts, body.boxed()),
    }
}

async fn rebuild_reply(mut parts: hyper::http::response::Parts, body: &mut MeteredBody, api: TraceApi, anthropic: bool) -> Response<MuxBody> {
    // Read the whole engine stream (capped), assemble the body a non-streaming call returns off the async workers, then pick the status: an engine error inside the stream is an error reply, not a 200.
    let mut raw = Vec::new();
    while let Some(frame) = body.frame().await {
        let Some(data) = frame.ok().and_then(|f| f.into_data().ok()) else {
            continue;
        };
        raw.extend_from_slice(&data);
        if raw.len() > REBUILD_CAP {
            body.cut = true;
            body.settle(StatusCode::BAD_GATEWAY.as_u16());
            return error(StatusCode::BAD_GATEWAY, "the reply is too large to return as one body; send the request with stream: true", "server_error", "reply_too_large");
        }
    }
    let assembled = tokio::task::spawn_blocking(move || goethe_core::trace_assemble::assemble(api, true, &raw)).await.unwrap_or(Value::Null);
    let (status, reply) = rebuilt_reply(parts.status, assembled, anthropic);
    body.settle(status.as_u16());
    parts.status = status;
    parts.headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    Response::from_parts(parts, full(serde_json::to_vec(&reply).unwrap_or_default()))
}

fn rebuilt_reply(engine: StatusCode, assembled: Value, anthropic: bool) -> (StatusCode, Value) {
    // An error the engine put in the stream (or a failed Responses run) becomes the error reply in the API's shape, with the engine's code when it gave one, else 502.
    let failed = assembled.get("status").and_then(Value::as_str) == Some("failed");
    let err = assembled.get("error").filter(|e| !e.is_null()).cloned().or_else(|| failed.then(|| json!({"message": "the engine failed this response"})));
    let Some(err) = err.filter(|_| engine.is_success()) else {
        return (engine, assembled);
    };
    let status = err
        .get("code")
        .and_then(Value::as_u64)
        .and_then(|c| u16::try_from(c).ok())
        .and_then(|c| StatusCode::from_u16(c).ok())
        .filter(|c| c.is_client_error() || c.is_server_error())
        .unwrap_or(StatusCode::BAD_GATEWAY);
    let reply = if anthropic { json!({"type": "error", "error": err}) } else { json!({"error": err}) };
    (status, reply)
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Mode {
    Pass(bool),
    HideUsage,
    Rebuild,
}

fn streamed(path: &str, stream: bool, body: &Bytes) -> (Bytes, Mode) {
    // Every generation request streams from the engine, so time to first token and decode speed are measured one way: Rebuild returns a request the client did not stream as one body, HideUsage hides the usage chunk an OpenAI stream now asks for, Pass(stream) leaves the body as sent.
    let Some(api) = TraceApi::from_path(api_tail(path)) else {
        return (body.clone(), Mode::Pass(stream));
    };
    let Ok(Value::Object(mut v)) = serde_json::from_slice::<Value>(body) else {
        return (body.clone(), Mode::Pass(stream));
    };
    let openai = matches!(api, TraceApi::Chat | TraceApi::Completions);
    let asked = v.get("stream_options").and_then(|o| o.get("include_usage")) == Some(&Value::Bool(true));
    let mode = match (stream, openai && !asked) {
        (false, _) => Mode::Rebuild,
        (true, true) => Mode::HideUsage,
        (true, false) => return (body.clone(), Mode::Pass(true)),
    };
    v.insert("stream".into(), Value::Bool(true));
    if openai {
        match v.entry("stream_options").or_insert_with(|| json!({})).as_object_mut() {
            Some(o) => o.insert("include_usage".into(), Value::Bool(true)),
            None => return (body.clone(), Mode::Pass(stream)),
        };
    }
    match serde_json::to_vec(&v) {
        Ok(b) => (Bytes::from(b), mode),
        Err(_) => (body.clone(), Mode::Pass(stream)),
    }
}

#[derive(Default)]
struct UsageFilter {
    held: Vec<u8>,
    skip_blank: bool,
}

impl UsageFilter {
    fn pass(&mut self, data: &[u8], last: bool) -> Bytes {
        // Whole SSE lines go through except the usage-only chunk (no choices) and the blank line after it; a partial line waits for the next frame.
        self.held.extend_from_slice(data);
        let end = if last { self.held.len() } else { self.held.iter().rposition(|&b| b == b'\n').map_or(0, |i| i + 1) };
        let mut out = Vec::with_capacity(end);
        for line in self.held[..end].split_inclusive(|&b| b == b'\n') {
            let text = line.trim_ascii();
            if self.skip_blank && text.is_empty() {
                self.skip_blank = false;
                continue;
            }
            self.skip_blank = false;
            if text.strip_prefix(b"data:").is_some_and(usage_only) {
                self.skip_blank = true;
                continue;
            }
            out.extend_from_slice(line);
        }
        self.held.drain(..end);
        Bytes::from(out)
    }
}

fn usage_only(data: &[u8]) -> bool {
    serde_json::from_slice::<Value>(data.trim_ascii()).is_ok_and(|c| c.get("choices").and_then(Value::as_array).is_some_and(|a| a.is_empty()) && c.get("usage").is_some_and(|u| !u.is_null()))
}

struct MeteredBody {
    inner: Incoming,
    meter: UsageMeter,
    done: Option<(Finish, inflight::InflightGuard, u16)>,
    sse: bool,
    cut: bool,
    ended: bool,
    tap: Option<Box<Capture>>,
    cut_event: &'static [u8],
    hide: Option<UsageFilter>,
    defer: bool,
}

const STREAM_CUT_EVENT: &[u8] = b"data: {\"error\":{\"message\":\"the engine stopped during this stream (model unloaded or engine crashed)\",\"type\":\"unavailable_error\",\"code\":\"engine_stopped\"}}\n\ndata: [DONE]\n\n";
const ANTHROPIC_CUT_EVENT: &[u8] = b"event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"api_error\",\"message\":\"the engine stopped during this stream (model unloaded or engine crashed)\"}}\n\n";

impl MeteredBody {
    fn take_in(&mut self, data: &[u8]) {
        // Every byte from the engine is metered and, when traced, kept as the engine sent it.
        self.meter.feed(data);
        if let Some(c) = self.tap.as_mut() {
            if c.response.len() + data.len() > trace::RESPONSE_CAP {
                c.truncated = true;
            } else {
                c.response.extend_from_slice(data);
            }
        }
    }

    fn finish_unless_deferred(&mut self) {
        if !self.defer {
            self.finish();
        }
    }

    fn settle(&mut self, code: u16) {
        // A deferred body (rebuilt reply) records with the status the client finally gets.
        if let Some(d) = self.done.as_mut() {
            d.2 = code;
        }
        self.finish();
    }

    fn finish(&mut self) {
        // Record once, at end of stream or when the client goes away (neither end nor engine error: cancelled); this also releases the in-flight slot.
        if let Some((mut finish, guard, code)) = self.done.take() {
            let mut used = self.meter.totals();
            if self.cut {
                used.finish = "error";
            }
            let cancelled = !self.ended && !self.cut;
            finish.record(code, cancelled, &used, self.meter.ttft());
            drop(guard);
            if let Some(tx) = finish.trailer.take().filter(|_| self.ended && !self.cut) {
                let _ = tx.send(usage_trailer(&used, self.meter.ttft()));
            }
            if let Some(mut c) = self.tap.take() {
                c.duration_s = finish.started.elapsed().as_secs_f64();
                c.ttft_s = self.meter.ttft();
                c.status = code;
                c.sse = self.sse;
                c.complete = self.ended && !self.cut;
                finish.m.trace.submit(*c);
            }
        }
    }
}

fn usage_trailer(used: &Metered, ttft: Option<f64>) -> HeaderMap {
    let usage = goethe_proto::Usage { prompt: used.prompt.max(0) as u64, completion: used.completion.max(0) as u64, ttft_ms: ttft.map_or(0, |t| (t * 1000.0).round() as u64) };
    let mut map = HeaderMap::new();
    if let Ok(v) = HeaderValue::from_str(&usage.header()) {
        map.insert(goethe_proto::USAGE_TRAILER, v);
    }
    map
}

impl Body for MeteredBody {
    type Data = Bytes;
    type Error = hyper::Error;

    fn poll_frame(self: Pin<&mut Self>, cx: &mut Context) -> Poll<Option<Result<Frame<Bytes>, hyper::Error>>> {
        // Frames pass through while data is metered; a cut SSE stream ends with an error event in its API's shape, not a reset.
        let this = self.get_mut();
        if this.cut || (this.ended && (this.done.is_none() || this.defer)) {
            return Poll::Ready(None);
        }
        let polled = Pin::new(&mut this.inner).poll_frame(cx);
        match &polled {
            Poll::Ready(Some(Ok(frame))) => {
                if let Some(data) = frame.data_ref() {
                    this.take_in(data);
                }
                if this.inner.is_end_stream() {
                    this.ended = true;
                }
                if let (Some(f), Some(data)) = (this.hide.as_mut(), frame.data_ref()) {
                    return Poll::Ready(Some(Ok(Frame::data(f.pass(data, this.ended)))));
                }
            }
            Poll::Ready(Some(Err(_))) if this.sse => {
                this.cut = true;
                this.finish_unless_deferred();
                return Poll::Ready(Some(Ok(Frame::data(Bytes::from_static(this.cut_event)))));
            }
            Poll::Ready(None) => {
                this.ended = true;
                this.finish_unless_deferred();
                if let Some(rest) = this.hide.as_mut().map(|f| f.pass(&[], true)).filter(|b| !b.is_empty()) {
                    return Poll::Ready(Some(Ok(Frame::data(rest))));
                }
            }
            Poll::Ready(Some(Err(_))) => {
                this.cut = true;
                this.finish_unless_deferred();
            }
            Poll::Pending => {}
        }
        polled
    }

    fn is_end_stream(&self) -> bool {
        self.cut || self.inner.is_end_stream()
    }

    fn size_hint(&self) -> SizeHint {
        // A rebuilt or filtered body is not the engine's length; claiming it would make hyper send a wrong Content-Length.
        if self.defer || self.hide.is_some() {
            return SizeHint::default();
        }
        self.inner.size_hint()
    }
}

impl Drop for MeteredBody {
    fn drop(&mut self) {
        self.finish();
    }
}

#[derive(Deserialize)]
struct TokenDetails {
    #[serde(default)]
    reasoning_tokens: Option<i64>,
    #[serde(default)]
    cached_tokens: Option<i64>,
}

#[derive(Deserialize)]
struct Usage {
    #[serde(default, alias = "input_tokens")]
    prompt_tokens: Option<i64>,
    #[serde(default, alias = "output_tokens")]
    completion_tokens: Option<i64>,
    #[serde(default, alias = "output_tokens_details")]
    completion_tokens_details: Option<TokenDetails>,
    #[serde(default, alias = "input_tokens_details")]
    prompt_tokens_details: Option<TokenDetails>,
    #[serde(default)]
    cache_read_input_tokens: Option<i64>,
    #[serde(default)]
    cache_creation_input_tokens: Option<i64>,
}

impl Usage {
    fn prompt(&self) -> Option<i64> {
        // Anthropic counts cache reads and writes outside input_tokens; the prompt is all three.
        self.prompt_tokens.map(|p| p.saturating_add(self.cache_read_input_tokens.unwrap_or(0)).saturating_add(self.cache_creation_input_tokens.unwrap_or(0)))
    }

    fn cached(&self) -> Option<i64> {
        self.prompt_tokens_details.as_ref().and_then(|d| d.cached_tokens).or(self.cache_read_input_tokens)
    }

    fn reasoning(&self) -> Option<i64> {
        self.completion_tokens_details.as_ref().and_then(|d| d.reasoning_tokens)
    }
}

#[derive(Deserialize)]
struct Incomplete {
    #[serde(default)]
    reason: Option<String>,
}

#[derive(Deserialize)]
struct UsageEnvelope {
    #[serde(default)]
    usage: Option<Usage>,
    #[serde(default)]
    stop_reason: Option<String>,
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    incomplete_details: Option<Incomplete>,
}

fn stop_of<'a>(stop_reason: &'a Option<String>, incomplete: &'a Option<Incomplete>, status: &'a Option<String>) -> Option<&'a str> {
    // Anthropic's stop_reason, or a Responses status (an incomplete one by its reason).
    stop_reason
        .as_deref()
        .or_else(|| incomplete.as_ref().and_then(|d| d.reason.as_deref()))
        .or(status.as_deref().filter(|s| *s != "in_progress"))
}

#[derive(Deserialize)]
struct StopDelta {
    #[serde(default)]
    stop_reason: Option<String>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum TopDelta {
    // Anthropic's message_delta carries an object; a Responses text delta is a plain string, which must not fail the whole chunk.
    Stop(StopDelta),
    Other(serde::de::IgnoredAny),
}

#[derive(Deserialize)]
struct StreamChunk {
    #[serde(default, rename = "type")]
    kind: Option<String>,
    #[serde(default)]
    usage: Option<Usage>,
    #[serde(default)]
    stop_reason: Option<String>,
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    incomplete_details: Option<Incomplete>,
    #[serde(default)]
    message: Option<UsageEnvelope>,
    #[serde(default)]
    response: Option<UsageEnvelope>,
    #[serde(default)]
    delta: Option<TopDelta>,
    #[serde(default)]
    choices: Vec<StreamChoice>,
}

fn finish_word(raw: &str) -> &'static str {
    // One vocabulary over OpenAI finish_reason, Anthropic stop_reason and Responses status.
    match raw {
        "stop" | "end_turn" | "stop_sequence" | "completed" | "eos" => "stop",
        "length" | "max_tokens" | "max_output_tokens" | "model_context_window_exceeded" => "length",
        "tool_calls" | "tool_use" | "function_call" => "tool_calls",
        "content_filter" | "refusal" => "content_filter",
        _ => "other",
    }
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
    #[serde(default)]
    finish_reason: Option<String>,
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
    fn has_reasoning(&self) -> bool {
        self.delta.as_ref().and_then(|d| d.reasoning_content.as_deref()).is_some_and(|t| !t.is_empty())
    }

    fn has_output(&self) -> bool {
        let filled = |s: &Option<String>| s.as_deref().is_some_and(|t| !t.is_empty());
        filled(&self.text)
            || self
                .delta
                .as_ref()
                .is_some_and(|d| filled(&d.content) || filled(&d.reasoning_content) || d.tool_calls.is_some())
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
struct Metered {
    prompt: i64,
    completion: i64,
    reasoning: i64,
    cached: i64,
    finish: &'static str,
}

#[cfg(test)]
impl Metered {
    fn tokens(&self) -> (i64, i64) {
        (self.prompt, self.completion)
    }
}

#[derive(Default)]
struct UsageMeter {
    sse: bool,
    started: Option<Instant>,
    buf: Vec<u8>,
    prompt: Option<i64>,
    completion: Option<i64>,
    reasoning: Option<i64>,
    cached: Option<i64>,
    finish: Option<&'static str>,
    chunks: i64,
    reasoning_chunks: i64,
    first_token: Option<Duration>,
    overflow: bool,
}

impl UsageMeter {
    fn new(sse: bool, started: Instant) -> Self {
        Self { sse, started: Some(started), ..Self::default() }
    }

    fn absorb(&mut self, chunk: &StreamChunk) {
        // Reported usage and stop reasons win over earlier ones (Anthropic splits usage over message_start and message_delta).
        let top = (&chunk.usage, stop_of(&chunk.stop_reason, &chunk.incomplete_details, &chunk.status));
        let nested = [chunk.message.as_ref(), chunk.response.as_ref()]
            .into_iter()
            .flatten()
            .map(|e| (&e.usage, stop_of(&e.stop_reason, &e.incomplete_details, &e.status)));
        for (usage, stop) in std::iter::once(top).chain(nested) {
            if let Some(u) = usage {
                self.prompt = u.prompt().or(self.prompt);
                self.completion = u.completion_tokens.or(self.completion);
                self.cached = u.cached().or(self.cached);
                self.reasoning = u.reasoning().or(self.reasoning);
            }
            if let Some(stop) = stop {
                self.finish = Some(finish_word(stop));
            }
        }
        let reasons = chunk.choices.iter().filter_map(|c| c.finish_reason.as_deref());
        let delta = match &chunk.delta {
            Some(TopDelta::Stop(d)) => d.stop_reason.as_deref(),
            _ => None,
        };
        if let Some(stop) = reasons.chain(delta).last() {
            self.finish = Some(finish_word(stop));
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
        self.absorb(&chunk);
        if chunk.has_output() {
            self.chunks += 1;
            if chunk.choices.iter().any(StreamChoice::has_reasoning) {
                self.reasoning_chunks += 1;
            }
            if self.first_token.is_none() {
                self.first_token = self.started.map(|s| s.elapsed());
            }
        }
    }

    fn totals(&mut self) -> Metered {
        // JSON usage, or for SSE the usage chunk, else the count of output chunks; reasoning tokens as reported, else the count of reasoning chunks.
        if !self.sse {
            if let Ok(chunk) = serde_json::from_slice::<StreamChunk>(&self.buf) {
                self.absorb(&chunk);
            }
            self.buf = Vec::new();
        } else {
            let rest = std::mem::take(&mut self.buf);
            self.sse_line(&rest);
        }
        let counted = self.sse && self.prompt.is_none() && self.completion.is_none();
        Metered {
            prompt: self.prompt.unwrap_or(0),
            completion: if counted { self.chunks } else { self.completion.unwrap_or(if self.sse { self.chunks } else { 0 }) },
            reasoning: self.reasoning.unwrap_or(self.reasoning_chunks),
            cached: self.cached.unwrap_or(0),
            finish: self.finish.unwrap_or(""),
        }
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
        goethe_core::sys::random_bytes(&mut b);
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
        return not_found();
    }
    match tokio::task::spawn_blocking(admin_report).await {
        Ok(Some(v)) => json_response(StatusCode::OK, v),
        _ => not_found(),
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

fn unsupported(code: &str, param: Option<&str>, message: String) -> Response<MuxBody> {
    // One refusal shape for every engine: 400, the code, the parameter when one is at fault, a message naming the engine.
    let body = json!({"error": {"message": message, "type": "invalid_request_error", "param": param, "code": code}});
    let mut resp = json_response(StatusCode::BAD_REQUEST, body);
    resp.extensions_mut().insert(GoetheError(message));
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

fn wrong_task(task: Option<&str>, tail: &str, model: &str) -> Option<String> {
    // An embedding model asked to write text: say which route it answers instead of passing the engine's error on.
    let writes = matches!(tail, "/chat/completions" | "/completions" | "/responses" | "/messages");
    (task == Some("embed") && writes).then(|| format!("{model} is an embedding model; send it to /v1/embeddings"))
}

fn model_not_found() -> Response<MuxBody> {
    error(StatusCode::NOT_FOUND, "Model Not Found", "invalid_request_error", "model_not_found")
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

    #[test]
    fn the_model_of_a_transcription_upload_is_its_form_field() {
        let body = b"--XyZ\r\nContent-Disposition: form-data; name=\"file\"; filename=\"model.wav\"\r\nContent-Type: audio/wav\r\n\r\nRIFF..model..\r\n\
--XyZ\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\norg/asr\r\n--XyZ\r\nContent-Disposition: form-data; name=\"language\"\r\n\r\nen\r\n--XyZ--\r\n";
        assert_eq!(form_field("multipart/form-data; boundary=XyZ", body, "model").as_deref(), Some("org/asr"));
        assert_eq!(form_field("Multipart/Form-Data; boundary=\"XyZ\"", body, "language").as_deref(), Some("en"));
        assert_eq!(form_field("multipart/form-data; boundary=XyZ", body, "prompt"), None);
        assert_eq!(form_field("application/json", br#"{"model": "org/asr"}"#, "model"), None);
    }

    #[test]
    fn a_form_field_is_found_by_its_part_header_only_and_never_twice() {
        let ct = "multipart/form-data; boundary=B";
        let decoy = b"--B\r\nContent-Disposition: form-data; name=\"file\"; filename=\"model\"\r\n\r\nx name=\"model\"\r\n\r\nfake\r\n--B\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nreal\r\n--B--\r\n";
        assert_eq!(form_field(ct, decoy, "model").as_deref(), Some("real"), "bytes inside a file or a filename are not the field");
        let twice = b"--B\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\na\r\n--B\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nb\r\n--B--\r\n";
        assert_eq!(form_field(ct, twice, "model"), None, "which copy an engine reads is not ours to guess");
        assert_eq!(form_field(ct, b"--B\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nunterminated", "model"), None);
    }

    #[test]
    fn the_meter_reads_reasoning_cache_and_finish_from_every_api_shape() {
        let mut o = meter_feed(
            true,
            &[
                b"data: {\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"hm\"}}]}\n\n",
                b"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"tool_calls\"}]}\n\n",
                b"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":50,\"completion_tokens\":9,\"prompt_tokens_details\":{\"cached_tokens\":32},\"completion_tokens_details\":{\"reasoning_tokens\":4}}}\n\ndata: [DONE]\n\n",
            ],
        );
        assert_eq!(o.totals(), Metered { prompt: 50, completion: 9, reasoning: 4, cached: 32, finish: "tool_calls" });
        let mut a = meter_feed(
            true,
            &[
                b"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":10,\"cache_read_input_tokens\":90,\"output_tokens\":1}}}\n\n",
                b"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"x\"}}\n\n",
                b"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"max_tokens\"},\"usage\":{\"output_tokens\":7}}\n\n",
            ],
        );
        assert_eq!(a.totals(), Metered { prompt: 100, completion: 7, reasoning: 0, cached: 90, finish: "length" });
        let mut r = meter_feed(
            true,
            &[b"data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"usage\":{\"input_tokens\":5,\"output_tokens\":2,\"input_tokens_details\":{\"cached_tokens\":1},\"output_tokens_details\":{\"reasoning_tokens\":1}}}}\n\n"],
        );
        assert_eq!(r.totals(), Metered { prompt: 5, completion: 2, reasoning: 1, cached: 1, finish: "stop" });
        let mut j = meter_feed(false, &[b"{\"type\":\"message\",\"stop_reason\":\"end_turn\",\"usage\":{\"input_tokens\":3,\"output_tokens\":8}}"]);
        assert_eq!(j.totals(), Metered { prompt: 3, completion: 8, reasoning: 0, cached: 0, finish: "stop" });
        let mut e = meter_feed(false, &[b"{\"data\":[{\"embedding\":[0.1]}],\"usage\":{\"prompt_tokens\":4}}"]);
        assert_eq!(e.totals(), Metered { prompt: 4, ..Metered::default() });
        let mut huge = meter_feed(false, &[b"{\"usage\":{\"input_tokens\":9223372036854775807,\"cache_read_input_tokens\":9}}"]);
        assert_eq!(huge.totals().prompt, i64::MAX, "an absurd count saturates instead of overflowing");
    }

    #[test]
    fn requests_are_labelled_by_the_network_they_came_from() {
        let p = |s: &str| network_path(s.parse().ok());
        assert_eq!(network_path(None), "local");
        assert_eq!(
            [p("127.0.0.1"), p("::1"), p("::ffff:192.168.1.4"), p("10.0.0.2"), p("fd12::1"), p("100.64.1.9"), p("fd7a:115c:a1e0::5"), p("100.200.0.1"), p("8.8.8.8")],
            ["local", "local", "lan", "lan", "lan", "tailscale_wireguard", "tailscale_wireguard", "public", "public"]
        );
        assert_eq!(
            ["/v1/chat/completions", "/v1/messages", "/embeddings", "/v1/rerank", "/v1/audio/transcriptions"].map(modality),
            ["chat", "chat", "embeddings", "rerank", "audio"]
        );
    }

    fn meter_feed(sse: bool, parts: &[&[u8]]) -> UsageMeter {
        let mut m = UsageMeter::new(sse, Instant::now());
        for p in parts {
            m.feed(p);
        }
        m
    }

    #[test]
    fn every_generation_request_streams_from_the_engine_and_others_pass() {
        let mode = |path: &str, stream: bool, b: &[u8]| {
            let (sent, mode) = streamed(path, stream, &Bytes::copy_from_slice(b));
            (serde_json::from_slice::<Value>(&sent).unwrap_or(Value::Null), mode)
        };
        let (v, m) = mode("/v1/chat/completions", true, br#"{"stream":true,"stream_options":{"include_usage":false,"x":1}}"#);
        assert_eq!((m, v["stream_options"].clone()), (Mode::HideUsage, json!({"include_usage": true, "x": 1})), "other stream options are kept");
        let own = br#"{"stream":true,"stream_options":{"include_usage":true}}"#;
        assert_eq!(streamed("/v1/chat/completions", true, &Bytes::from_static(own)), (Bytes::from_static(own), Mode::Pass(true)), "a client that asked keeps its usage chunk");
        let (v, m) = mode("/v1/messages", false, br#"{"model":"m"}"#);
        assert_eq!((m, v["stream"].clone(), v.get("stream_options").is_none()), (Mode::Rebuild, json!(true), true), "Anthropic streams carry usage already");
        assert_eq!(mode("/v1/messages", true, br#"{"stream":true}"#).1, Mode::Pass(true));
        assert_eq!(mode("/v1/responses", false, br#"{"input":"x"}"#).1, Mode::Rebuild);
        let emb = br#"{"input":"x"}"#;
        assert_eq!(streamed("/v1/embeddings", false, &Bytes::from_static(emb)), (Bytes::from_static(emb), Mode::Pass(false)), "not a generation route");
    }

    #[test]
    fn only_the_usage_chunk_is_hidden_even_when_split_across_frames() {
        let mut f = UsageFilter::default();
        let a = f.pass(b"data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\ndata: {\"choices\":[],\"usa", false);
        let b = f.pass(b"ge\":{\"prompt_tokens\":7,\"completion_tokens\":2}}\n\ndata: [DONE]\n\n", true);
        let out = [a, b].concat();
        assert_eq!(String::from_utf8(out).unwrap(), "data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\ndata: [DONE]\n\n");
        let mut f = UsageFilter::default();
        let keep = b"data: {\"choices\":[{\"delta\":{}}],\"usage\":{\"completion_tokens\":2}}\n\n";
        assert_eq!(f.pass(keep, true).as_ref(), &keep[..], "a chunk with choices is the client's, usage or not");
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
        assert_eq!(m.totals().tokens(), (7, 11));
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
        assert_eq!(m.totals().tokens(), (0, 5));
    }

    #[test]
    fn json_usage_is_read_once_at_the_end() {
        let mut m = meter_feed(false, &[b"{\"id\":\"x\",\"choices\":[],", b"\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2}}"]);
        assert_eq!(m.totals().tokens(), (3, 2));
        assert_eq!(m.ttft(), None);
        let mut bad = meter_feed(false, &[b"not json"]);
        assert_eq!(bad.totals().tokens(), (0, 0));
        assert_eq!(meter_feed(false, &[b"And so, my fellow Americans"]).totals().tokens(), (0, 0), "a plain text transcript meters nothing");
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
        assert_eq!(a.totals().tokens(), (9, 4));
        assert!(a.ttft().is_some());
        let mut r = meter_feed(
            true,
            &[
                b"data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hi\"}\n\n",
                b"data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":5,\"output_tokens\":2}}}\n\n",
            ],
        );
        assert_eq!(r.totals().tokens(), (5, 2));
        assert!(r.ttft().is_some(), "a Responses text delta (a string, not an object) is output");
        let mut j = meter_feed(false, &[b"{\"type\":\"message\",\"usage\":{\"input_tokens\":3,\"output_tokens\":8}}"]);
        assert_eq!(j.totals().tokens(), (3, 8));
    }

    #[test]
    fn litellm_routes_and_key_headers() {
        let proxied = |p: &str| match api_route(p) {
            Some((Method::POST, Api::Proxy(up))) => Some(up),
            _ => None,
        };
        assert_eq!(wrong_task(Some("embed"), "/chat/completions", "org/e").as_deref(), Some("org/e is an embedding model; send it to /v1/embeddings"));
        assert_eq!(wrong_task(Some("embed"), "/embeddings", "org/e"), None);
        assert_eq!(wrong_task(None, "/chat/completions", "org/m"), None);
        assert_eq!(proxied("/chat/completions").as_deref(), Some("/v1/chat/completions"));
        assert_eq!(proxied("/v1/chat/completions").as_deref(), Some("/v1/chat/completions"));
        assert_eq!(proxied("/v1/messages").as_deref(), Some("/v1/messages"));
        assert_eq!(proxied("/responses").as_deref(), Some("/v1/responses"));
        assert_eq!(proxied("/v1/audio/transcriptions").as_deref(), Some("/v1/audio/transcriptions"));
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

    #[test]
    fn anthropic_budget_below_1024_is_raised_and_tools_stay() {
        let raw = concat!(
            r#"{"model":"m","thinking":{"type":"enabled","budget_tokens":128},"#,
            r#""tools":[{"name":"ping"}],"tool_choice":{"type":"tool","name":"ping"}}"#
        )
        .as_bytes();
        let out = prepare_body("/v1/messages", Bytes::from_static(raw));
        let v: Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(v["thinking"]["budget_tokens"], 1024);
        assert_eq!(v["tools"][0]["name"], "ping");
        assert_eq!(v["tool_choice"]["name"], "ping");
        let high = br#"{"thinking":{"budget_tokens":1024},"tools":[{"name":"ping"}]}"#;
        assert_eq!(prepare_body("/v1/messages", Bytes::from_static(high)).as_ref(), &high[..]);
        let openai = br#"{"tools":[{"type":"function","function":{"name":"ping"}}]}"#;
        assert_eq!(prepare_body("/v1/chat/completions", Bytes::from_static(openai)).as_ref(), &openai[..]);
    }

    async fn fake_engine(reply: &'static [u8], content_type: &'static str) -> (u16, Arc<Mutex<Vec<u8>>>) {
        // One connection that records the request body and answers with `reply`.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let seen = Arc::new(Mutex::new(Vec::<u8>::new()));
        let seen_task = seen.clone();
        tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let svc = service_fn(move |req: Request<Incoming>| {
                let seen_task = seen_task.clone();
                async move {
                    let bytes = req.into_body().collect().await.unwrap().to_bytes();
                    seen_task.lock().unwrap().clone_from(&bytes.to_vec());
                    let mut resp = Response::new(Full::new(Bytes::from_static(reply)));
                    resp.headers_mut().insert(CONTENT_TYPE, HeaderValue::from_static(content_type));
                    Ok::<_, Infallible>(resp)
                }
            });
            let _ = http1::Builder::new().serve_connection(TokioIo::new(stream), svc).await;
        });
        tokio::task::yield_now().await;
        (port, seen)
    }

    fn test_mux(name: &str) -> MuxRef {
        let dir = std::env::temp_dir().join(format!("goethe-mux-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        test_mux_at(&dir)
    }

    fn test_mux_at(dir: &std::path::Path) -> MuxRef {
        Box::leak(Box::new(Mux {
            cache: GatewayCache::open(dir),
            client: upstream_client(),
            max_body: 8 << 20,
            rpm: Mutex::new(HashMap::new()),
            ready: Mutex::new(HashMap::new()),
            trace: Recorder::start(dir),
            learn: Store::start(dir),
            learn_gates: Gates::default(),
        }))
    }

    async fn proxied(mux: MuxRef, port: u16, body: &[u8], mode: Mode) -> Response<MuxBody> {
        let route = Route { instance_id: "i".into(), port, engine_key: None, draining: false };
        let finish = Finish { m: mux, label: "m".into(), key_id: "k".into(), started: Instant::now(), dims: Dims::default(), router_ms: None, trailer: None };
        proxy(mux, "/v1/chat/completions", &route, Bytes::copy_from_slice(body), None, "req-1", mode, inflight::track("m"), finish, None, "application/json").await
    }

    async fn send(mux: MuxRef, port: u16, body: &[u8], mode: Mode) -> (Option<String>, Vec<u8>) {
        let resp = proxied(mux, port, body, mode).await;
        let ct = resp.headers().get(CONTENT_TYPE).and_then(|v| v.to_str().ok()).map(str::to_string);
        (ct, resp.into_body().collect().await.unwrap().to_bytes().to_vec())
    }

    #[tokio::test]
    async fn a_filtered_stream_does_not_claim_the_engines_length() {
        let sse = b"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"x\"}}]}\n\ndata: {\"choices\":[],\"usage\":{\"completion_tokens\":1}}\n\ndata: [DONE]\n\n";
        let (port, _) = fake_engine(sse, "text/event-stream").await;
        let resp = proxied(test_mux("hide-len"), port, br#"{"model":"m","stream":true}"#, Mode::HideUsage).await;
        assert!(resp.headers().get(CONTENT_LENGTH).is_none() && resp.body().size_hint().exact().is_none());
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        assert!(!body.windows(5).any(|w| w == b"usage") && body.ends_with(b"data: [DONE]\n\n"), "{body:?}");
    }

    #[tokio::test]
    async fn proxied_openai_body_keeps_tools() {
        let (port, seen) = fake_engine(b"{\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1}}", "application/json").await;
        let raw = br#"{"model":"m","tools":[{"type":"function","function":{"name":"ping"}}]}"#;
        assert_eq!(prepare_body("/v1/chat/completions", Bytes::from_static(raw)).as_ref(), &raw[..]);
        send(test_mux("tools"), port, raw, Mode::Pass(false)).await;
        let v: Value = serde_json::from_slice(&seen.lock().unwrap()).unwrap();
        assert_eq!(v["tools"][0]["function"]["name"], "ping");
    }

    #[tokio::test]
    async fn a_request_the_client_did_not_stream_comes_back_as_one_body() {
        let sse = b"data: {\"id\":\"c1\",\"created\":7,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hel\"}}]}\n\n\
data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"},\"finish_reason\":\"stop\"}]}\n\n\
data: {\"id\":\"c1\",\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2,\"total_tokens\":5}}\n\ndata: [DONE]\n\n";
        let (port, seen) = fake_engine(sse, "text/event-stream").await;
        let (upstream, mode) = streamed("/v1/chat/completions", false, &Bytes::from_static(br#"{"model":"m","messages":[]}"#));
        assert_eq!(mode, Mode::Rebuild);
        let (ct, body) = send(test_mux("rebuild"), port, &upstream, mode).await;
        let sent: Value = serde_json::from_slice(&seen.lock().unwrap()).unwrap();
        assert_eq!((sent["stream"].clone(), sent["stream_options"]["include_usage"].clone()), (json!(true), json!(true)), "the engine is asked for a stream with usage");
        assert_eq!(ct.as_deref(), Some("application/json"));
        let v: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["object"], "chat.completion");
        assert_eq!((v["choices"][0]["message"]["content"].clone(), v["choices"][0]["finish_reason"].clone()), (json!("Hello"), json!("stop")));
        assert_eq!(v["usage"]["completion_tokens"], 2);
    }

    #[tokio::test]
    async fn a_rebuilt_reply_past_the_cap_ends_in_an_error_instead_of_growing() {
        let line = b"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"x\"}}]}\n\n";
        let huge: &'static [u8] = Box::leak(line.repeat(REBUILD_CAP / line.len() + 2).into_boxed_slice());
        let (port, _) = fake_engine(huge, "text/event-stream").await;
        let resp = proxied(test_mux("rebuild-cap"), port, br#"{"model":"m","stream":true}"#, Mode::Rebuild).await;
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        assert_eq!(serde_json::from_slice::<Value>(&body).unwrap()["error"]["code"], "reply_too_large");
    }

    #[tokio::test]
    async fn an_engine_error_inside_a_rebuilt_stream_is_an_error_reply_not_a_200() {
        let sse = b"data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hel\"}}]}\n\n\
data: {\"error\":{\"message\":\"context too long\",\"type\":\"BadRequestError\",\"code\":400}}\n\ndata: [DONE]\n\n";
        let (port, _) = fake_engine(sse, "text/event-stream").await;
        let resp = proxied(test_mux("rebuild-err"), port, br#"{"model":"m","stream":true}"#, Mode::Rebuild).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let v: Value = serde_json::from_slice(&resp.into_body().collect().await.unwrap().to_bytes()).unwrap();
        assert_eq!(v, json!({"error": {"message": "context too long", "type": "BadRequestError", "code": 400}}));
        let (code, reply) = rebuilt_reply(StatusCode::OK, json!({"type": "message", "error": {"type": "api_error", "message": "stopped"}}), true);
        assert_eq!((code, reply["type"].clone()), (StatusCode::BAD_GATEWAY, json!("error")), "no numeric code: 502, Anthropic shape");
        let fine = json!({"object": "chat.completion"});
        assert_eq!(rebuilt_reply(StatusCode::OK, fine.clone(), false), (StatusCode::OK, fine));
    }

    #[tokio::test]
    async fn a_routed_request_needs_no_node_key_streams_back_with_the_usage_trailer_and_is_logged_as_routed() {
        let sse = b"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"}}]}\n\ndata: {\"choices\":[],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":2}}\n\ndata: [DONE]\n\n";
        let (port, seen) = fake_engine(sse, "text/event-stream").await;
        let dir = tempfile_dir("routed");
        let con = db::open(&dir).unwrap();
        con.execute_batch(&format!(
            "INSERT INTO loaded_instances (instance_id, model_id, pid, pgid, internal_port, gpu_indices, allocated_vram_gb_per_gpu, mem_fraction_static, ctx_len, max_conc, status)
             VALUES ('i-0', 'org/routed', 1, 1, 1, '0', 1, 0.8, 4096, 8, 'RUNNING'), ('i-1', 'org/routed', 1, 1, {port}, '0', 1, 0.8, 4096, 8, 'RUNNING');"
        ))
        .unwrap();
        let m = test_mux_at(&dir);
        let node = Arc::new(goethe_core::client::Node::new(dir.clone(), "n1", "h1"));
        let (ours, theirs) = tokio::io::duplex(1 << 16);
        tokio::spawn(goethe_core::client::serve(ours, node, gateway(m)));
        let (mut c, conn) = hyper::client::conn::http2::handshake(TokioExecutor::new(), TokioIo::new(theirs)).await.unwrap();
        tokio::spawn(conn);
        let now_ms = goethe_core::sys::unix_ms();
        let req = Request::post("http://node/v1/chat/completions")
            .header(goethe_proto::REQUEST_ID_HEADER, "r-1")
            .header(goethe_proto::KEY_ID_HEADER, "key1")
            .header(goethe_proto::USER_HEADER, "u1")
            .header(goethe_proto::DISPATCH_HEADER, (now_ms - 50).to_string())
            .header(goethe_proto::INSTANCE_HEADER, "i-1")
            .body(Full::new(Bytes::from_static(br#"{"model":"org/routed","stream":true}"#)))
            .unwrap();
        let resp = c.send_request(req).await.unwrap();
        assert_eq!((resp.status(), resp.headers()[X_REQUEST_ID].to_str().unwrap()), (StatusCode::OK, "r-1"));
        let got = resp.into_body().collect().await.unwrap();
        let trailer = got.trailers().and_then(|t| t.get(goethe_proto::USAGE_TRAILER)).and_then(|v| v.to_str().ok()).map(str::to_string);
        let usage = trailer.as_deref().and_then(goethe_proto::Usage::parse).expect("the answer ends with the usage trailer");
        assert_eq!((usage.prompt, usage.completion), (7, 2));
        let body = got.to_bytes();
        assert!(body.starts_with(b"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"}}]}") && body.ends_with(b"data: [DONE]\n\n"), "{body:?}");
        assert!(!seen.lock().unwrap().is_empty(), "the pinned instance answered");

        usage_log::flush(&dir);
        let rows: Vec<(String, i64, i64)> = con
            .prepare("SELECT k, router_n, router_ms_sum FROM req_minute")
            .unwrap()
            .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get(1)?, r.get(2)?)))
            .unwrap()
            .map(Result::unwrap)
            .filter(|(k, _, _)| k.starts_with("org/routed"))
            .collect();
        assert_eq!(rows.len(), 1, "{rows:?}");
        let (k, n, ms) = &rows[0];
        let parts: Vec<&str> = k.split('\u{1f}').collect();
        assert_eq!((parts[3], parts[4], parts[6]), ("ck:key1", "u1", "routed"));
        assert!(*n == 1 && (50..600_000).contains(ms), "router_ms {ms}");

        let models = c.send_request(Request::get("http://node/v1/models").body(Full::new(Bytes::new())).unwrap()).await.unwrap();
        let v: Value = serde_json::from_slice(&models.into_body().collect().await.unwrap().to_bytes()).unwrap();
        assert_eq!(v["data"][0]["id"], "org/routed", "no key needed on the connection");
        let missing = c.send_request(Request::post("http://node/v1/nope").body(Full::new(Bytes::new())).unwrap()).await.unwrap();
        assert_eq!(missing.status(), StatusCode::NOT_FOUND);
    }

    #[test]
    fn router_time_is_clamped_and_absent_without_a_dispatch_stamp() {
        let r = |d: Option<u64>| Routed { request_id: String::new(), key_id: String::new(), user_id: String::new(), dispatch_ms: d, instance: None };
        assert_eq!(r(Some(1_000)).router_ms(1_250), Some(250.0));
        assert_eq!(r(Some(2_000)).router_ms(1_000), Some(0.0), "a server clock ahead of ours");
        assert_eq!(r(Some(0)).router_ms(10_000_000), Some(600_000.0));
        assert_eq!(r(None).router_ms(5), None);
        assert_eq!(Caller::Routed(&r(None)).key_id(), "");
        assert_eq!(Caller::Routed(&Routed { key_id: "k".into(), ..r(None) }).key_id(), "ck:k");
    }

    fn tempfile_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("goethe-mux-{name}-{}-{}", std::process::id(), SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }
}
