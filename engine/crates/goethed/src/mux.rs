// OpenAI multiplexer on 127.0.0.1 (GOETHE_MUX_PORT, default 30000).

use futures_util::{stream, StreamExt};
use http_body_util::{BodyExt, Full, StreamBody};
use hyper::body::{Bytes, Frame, Incoming};
use hyper::header::{AUTHORIZATION, RETRY_AFTER};
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::client::legacy::Client;
use hyper_util::rt::{TokioExecutor, TokioIo};
use goethe_core::alias;
use goethe_core::db;
use goethe_core::hw;
use goethe_core::inflight;
use goethe_core::key_usage;
use goethe_core::metrics;
use goethe_core::paths::{goethe_home, goethe_mux_addr, goethe_mux_port};
use goethe_core::placement;
use goethe_core::security::{self, ResolvedKey};
use goethe_core::supervisor;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::{Mutex, OnceLock};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};
use tokio::net::TcpListener;

type MuxBody = http_body_util::combinators::BoxBody<Bytes, hyper::Error>;

static RPM_BUCKETS: OnceLock<Mutex<HashMap<String, Vec<Instant>>>> = OnceLock::new();

fn rpm_buckets() -> &'static Mutex<HashMap<String, Vec<Instant>>> {
    RPM_BUCKETS.get_or_init(|| Mutex::new(HashMap::new()))
}

pub fn spawn_mux() -> Result<tokio::task::JoinHandle<()>, String> {
    let port = goethe_mux_port();
    let addr: SocketAddr = goethe_mux_addr()
        .parse()
        .map_err(|e: std::net::AddrParseError| e.to_string())?;
    std::net::TcpListener::bind(addr).map_err(|e| {
        format!(
            "mux bind failed on 127.0.0.1:{port} ({e}); set GOETHE_MUX_PORT to a free port"
        )
    })?;
    Ok(tokio::spawn(async {
        if let Err(e) = run_mux().await {
            eprintln!("mux error: {e}");
        }
    }))
}

async fn run_mux() -> Result<(), String> {
    let addr: SocketAddr = goethe_mux_addr()
        .parse()
        .map_err(|e: std::net::AddrParseError| e.to_string())?;
    let listener = TcpListener::bind(addr).await.map_err(|e| e.to_string())?;
    loop {
        let (stream, _) = listener.accept().await.map_err(|e| e.to_string())?;
        stream.set_nodelay(true).ok();
        tokio::spawn(async move {
            let io = TokioIo::new(stream);
            let svc = service_fn(|req| async {
                Ok::<Response<MuxBody>, Infallible>(handle(req).await)
            });
            if http1::Builder::new()
                .serve_connection(io, svc)
                .await
                .is_err()
            {}
        });
    }
}

async fn handle(req: Request<Incoming>) -> Response<MuxBody> {
    let path = req.uri().path().to_string();
    if req.method() == Method::GET && path == "/metrics" {
        let body = metrics::render_prometheus();
        return Response::builder()
            .status(StatusCode::OK)
            .header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            .body(fixed_body(Bytes::from(body)))
            .unwrap();
    }
    if req.method() == Method::GET && path == "/health" {
        return json_response(StatusCode::OK, json!({"status": "ok"}));
    }
    if req.method() == Method::GET && path == "/health/ready" {
        return health_ready(req.uri().query()).await;
    }
    if req.method() == Method::GET && path.starts_with("/health/admin") {
        return health_admin(req.headers()).await;
    }
    if !path.starts_with("/v1/") {
        return empty_status(StatusCode::NOT_FOUND);
    }
    let Some(key) = resolve_key(req.headers()) else {
        return json_response(
            StatusCode::UNAUTHORIZED,
            json!({"error": {"message": "Invalid or missing Goethe API key", "type": "auth_error"}}),
        );
    };
    if security::key_expired(&key.expires_at) {
        return json_response(
            StatusCode::UNAUTHORIZED,
            json!({"error": {"message": "key expired", "type": "auth_error"}}),
        );
    }
    if let Some(retry) = check_rpm(&key) {
        metrics::inc_rate_limited(&key.key_id);
        return rate_limited(retry);
    }
    if req.method() == Method::GET && path == "/v1/models" {
        let resp = list_models(&key);
        metrics::record_request("", resp.status().as_u16(), 0.0, 0, 0);
        let _ = key_usage::record_request(&key.key_id, 0, 0);
        return resp;
    }
    if req.method() == Method::POST
        && (path == "/v1/chat/completions"
            || path == "/v1/completions"
            || path == "/v1/embeddings")
    {
        return route_post(req, &path, &key).await;
    }
    empty_status(StatusCode::NOT_FOUND)
}

fn resolve_key(headers: &hyper::HeaderMap) -> Option<ResolvedKey> {
    let raw = headers.get(AUTHORIZATION).and_then(|v| v.to_str().ok())?;
    let token = raw.strip_prefix("Bearer ")?;
    if token.is_empty() {
        return None;
    }
    security::resolve_token(token)
}

fn check_rpm(key: &ResolvedKey) -> Option<u64> {
    let window = Duration::from_secs(60);
    let now = Instant::now();
    let mut map = rpm_buckets().lock().ok()?;
    let bucket = map.entry(key.key_id.clone()).or_default();
    bucket.retain(|t| now.duration_since(*t) < window);
    if key.rpm_limit > 0 && bucket.len() as i64 >= key.rpm_limit {
        let retry = bucket
            .first()
            .map(|t| window.saturating_sub(now.duration_since(*t)).as_secs().max(1))
            .unwrap_or(1);
        return Some(retry);
    }
    bucket.push(now);
    None
}

fn rate_limited(retry_after_secs: u64) -> Response<MuxBody> {
    Response::builder()
        .status(StatusCode::TOO_MANY_REQUESTS)
        .header(RETRY_AFTER, retry_after_secs.to_string())
        .header("Content-Type", "application/json")
        .body(fixed_body(Bytes::from(
            serde_json::to_vec(&json!({
                "error": {
                    "message": "Rate limit exceeded",
                    "type": "rate_limit_error",
                    "code": "rate_limit_exceeded"
                }
            }))
            .unwrap_or_default(),
        )))
        .unwrap_or_else(|_| empty_status(StatusCode::TOO_MANY_REQUESTS))
}

fn list_models(key: &ResolvedKey) -> Response<MuxBody> {
    let home = goethe_home();
    let con = match db::open(&home) {
        Ok(c) => c,
        Err(_) => return json_response(StatusCode::INTERNAL_SERVER_ERROR, json!({"error": "db"})),
    };
    let mut stmt = match con.prepare(
        "SELECT DISTINCT model_id FROM loaded_instances WHERE status = 'RUNNING'",
    ) {
        Ok(s) => s,
        Err(_) => return json_response(StatusCode::INTERNAL_SERVER_ERROR, json!({"error": "db"})),
    };
    let rows = stmt.query_map([], |r| r.get::<_, String>(0));
    let mut data = Vec::new();
    if let Ok(rows) = rows {
        for id in rows.filter_map(|r| r.ok()) {
            if supervisor::model_draining(&id) {
                continue;
            }
            if !security::model_allowed(&key.model_scope, &id) {
                continue;
            }
            data.push(json!({"id": id, "object": "model"}));
        }
    }
    json_response(StatusCode::OK, json!({"object": "list", "data": data}))
}

fn resolve_model_id(raw: &str) -> String {
    let home = goethe_home();
    let Ok(con) = db::open(&home) else {
        return raw.to_string();
    };
    alias::resolve(&con, raw).unwrap_or_else(|| raw.to_string())
}

fn model_not_found() -> Response<MuxBody> {
    json_response(
        StatusCode::NOT_FOUND,
        json!({"error": {"message": "Model Not Found", "type": "invalid_request_error"}}),
    )
}

fn admin_not_found() -> Response<MuxBody> {
    json_response(
        StatusCode::NOT_FOUND,
        json!({"error": {"message": "Not Found", "type": "invalid_request_error"}}),
    )
}

async fn health_ready(query: Option<&str>) -> Response<MuxBody> {
    let model_raw = query
        .and_then(|q| {
            q.split('&').find_map(|pair| {
                let mut it = pair.splitn(2, '=');
                let k = it.next()?;
                if k == "model" {
                    Some(it.next().unwrap_or("").to_string())
                } else {
                    None
                }
            })
        })
        .filter(|s| !s.is_empty());
    let Some(model_raw) = model_raw else {
        return json_response(
            StatusCode::BAD_REQUEST,
            json!({"error": {"message": "model query parameter required", "type": "invalid_request_error"}}),
        );
    };
    let model = resolve_model_id(&model_raw);
    let home = goethe_home();
    let con = match db::open(&home) {
        Ok(c) => c,
        Err(_) => {
            return json_response(
                StatusCode::SERVICE_UNAVAILABLE,
                json!({"status": "not_ready", "model": model}),
            );
        }
    };
    let port: Option<u16> = con
        .query_row(
            "SELECT internal_port FROM loaded_instances WHERE model_id = ?1 AND status = 'RUNNING' LIMIT 1",
            [&model],
            |r| r.get(0),
        )
        .ok();
    let ready = match port {
        Some(p) => engine_health_ok(p).await,
        None => false,
    };
    if ready {
        json_response(
            StatusCode::OK,
            json!({"status": "ready", "model": model}),
        )
    } else {
        json_response(
            StatusCode::SERVICE_UNAVAILABLE,
            json!({"status": "not_ready", "model": model}),
        )
    }
}

async fn engine_health_ok(port: u16) -> bool {
    let url = format!("http://127.0.0.1:{port}/health");
    let client: Client<_, Full<Bytes>> = Client::builder(TokioExecutor::new()).build_http();
    let req = match Request::builder()
        .method(Method::GET)
        .uri(&url)
        .body(Full::new(Bytes::new()))
    {
        Ok(r) => r,
        Err(_) => return false,
    };
    match tokio::time::timeout(Duration::from_secs(2), client.request(req)).await {
        Ok(Ok(resp)) => resp.status().is_success(),
        _ => false,
    }
}

async fn health_admin(headers: &hyper::HeaderMap) -> Response<MuxBody> {
    let Some(key) = resolve_key(headers) else {
        return admin_not_found();
    };
    if security::key_expired(&key.expires_at) || !key.is_admin {
        return admin_not_found();
    }
    let home = goethe_home();
    let con = match db::open(&home) {
        Ok(c) => c,
        Err(_) => return admin_not_found(),
    };
    let managed = placement::managed_pids(&con);
    let snaps = match hw::list_gpus_for_placement(&managed) {
        Ok(s) => s,
        Err(_) => return admin_not_found(),
    };
    let names = gpu_names_by_index(&home);
    let mut gpus = Vec::new();
    for snap in snaps {
        let idx = snap.index;
        let (total_mb, used_mb) = hw::refresh_gpu_vram(idx)
            .map(|(t, u, _, _)| (t, u))
            .unwrap_or((0, 0));
        let used_managed_mb = hw::managed_vram_listed_mb(idx, &managed).unwrap_or(0);
        let used_foreign_mb = used_mb.saturating_sub(used_managed_mb);
        gpus.push(json!({
            "index": idx,
            "name": names.get(&idx).cloned().unwrap_or_default(),
            "total_mb": total_mb,
            "used_managed_mb": used_managed_mb,
            "used_foreign_mb": used_foreign_mb,
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
            for row in rows.filter_map(|r| r.ok()) {
                let (instance_id, model_id, pid, status, gpu_indices, internal_port) = row;
                instances.push(json!({
                    "instance_id": instance_id,
                    "model_id": model_id,
                    "pid": pid,
                    "status": status,
                    "gpu_indices": gpu_indices,
                    "internal_port": internal_port,
                    "in_flight": inflight::count(&model_id),
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
            inactive_models = rows.filter_map(|r| r.ok()).collect();
        }
    }
    json_response(
        StatusCode::OK,
        json!({
            "gpus": gpus,
            "instances": instances,
            "inactive_models": inactive_models,
        }),
    )
}

fn gpu_names_by_index(home: &std::path::Path) -> HashMap<u32, String> {
    let path = home.join("hardware_state.json");
    let Ok(data) = std::fs::read_to_string(path) else {
        return HashMap::new();
    };
    let Ok(v) = serde_json::from_str::<Value>(&data) else {
        return HashMap::new();
    };
    let Some(arr) = v.get("gpus").and_then(|g| g.as_array()) else {
        return HashMap::new();
    };
    let mut out = HashMap::new();
    for g in arr {
        let idx = g.get("index").and_then(|x| x.as_u64()).unwrap_or(0) as u32;
        let name = g
            .get("name")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string();
        out.insert(idx, name);
    }
    out
}

async fn route_post(req: Request<Incoming>, path: &str, key: &ResolvedKey) -> Response<MuxBody> {
    let started = Instant::now();
    let body = match req.into_body().collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(_) => return empty_status(StatusCode::BAD_REQUEST),
    };
    let raw_model = match parse_model(&body) {
        Some(m) => m,
        None => return empty_status(StatusCode::BAD_REQUEST),
    };
    let model = resolve_model_id(&raw_model);
    if !security::model_allowed(&key.model_scope, &model) {
        metrics::record_request(&model, 404, 0.0, 0, 0);
        let _ = key_usage::record_request(&key.key_id, 0, 0);
        return model_not_found();
    }
    if supervisor::model_draining(&model) {
        metrics::record_request(&model, 503, 0.0, 0, 0);
        let _ = key_usage::record_request(&key.key_id, 0, 0);
        return Response::builder()
            .status(StatusCode::SERVICE_UNAVAILABLE)
            .header(RETRY_AFTER, "5")
            .header("Content-Type", "application/json")
            .body(fixed_body(Bytes::from(
                serde_json::to_vec(&json!({
                    "error": {"message": "Model is draining", "type": "unavailable_error"}
                }))
                .unwrap_or_default(),
            )))
            .unwrap_or_else(|_| empty_status(StatusCode::SERVICE_UNAVAILABLE));
    }
    let port = match supervisor::internal_port_for_model(&model) {
        Some(p) => p,
        None => {
            metrics::record_request(&model, 404, 0.0, 0, 0);
            let _ = key_usage::record_request(&key.key_id, 0, 0);
            return model_not_found();
        }
    };
    let streaming = serde_json::from_slice::<Value>(&body)
        .ok()
        .and_then(|v| v.get("stream").and_then(|x| x.as_bool()))
        .unwrap_or(false);
    let guard = inflight::track(&model);
    let (resp, prompt_t, completion_t) = proxy_post(path, port, body, streaming, guard).await;
    metrics::record_request(
        &model,
        resp.status().as_u16(),
        started.elapsed().as_secs_f64(),
        prompt_t,
        completion_t,
    );
    let _ = key_usage::record_request(&key.key_id, prompt_t, completion_t);
    resp
}

struct GuardedDataStream<S> {
    guard: Option<inflight::InflightGuard>,
    inner: Pin<Box<S>>,
}

impl<S> stream::Stream for GuardedDataStream<S>
where
    S: stream::Stream<Item = Result<Frame<Bytes>, hyper::Error>> + Unpin + 'static,
{
    type Item = Result<Frame<Bytes>, hyper::Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.as_mut().get_mut();
        match Pin::new(&mut this.inner).poll_next(cx) {
            Poll::Ready(None) => {
                this.guard.take();
                Poll::Ready(None)
            }
            other => other,
        }
    }
}

async fn proxy_post(
    path: &str,
    port: u16,
    body: Bytes,
    streaming: bool,
    guard: inflight::InflightGuard,
) -> (Response<MuxBody>, i64, i64) {
    let url = format!("http://127.0.0.1:{port}{path}");
    let client: Client<_, Full<Bytes>> = Client::builder(TokioExecutor::new()).build_http();
    let req = match Request::builder()
        .method(Method::POST)
        .uri(&url)
        .header("Content-Type", "application/json")
        .body(Full::new(body))
    {
        Ok(r) => r,
        Err(_) => return (empty_status(StatusCode::BAD_GATEWAY), 0, 0),
    };
    match client.request(req).await {
        Ok(resp) => {
            let (parts, upstream) = resp.into_parts();
            if streaming {
                let inner = upstream
                    .into_data_stream()
                    .map(|item| item.map(Frame::data));
                let guarded = GuardedDataStream {
                    guard: Some(guard),
                    inner: Box::pin(inner),
                };
                let out = Response::from_parts(
                    parts,
                    BodyExt::boxed(StreamBody::new(guarded)),
                );
                (out, 0, 0)
            } else {
                let collected = upstream.collect().await;
                drop(guard);
                match collected {
                    Ok(buf) => {
                        let bytes = buf.to_bytes();
                        let (pt, ct) = parse_usage(&bytes);
                        (
                            Response::from_parts(parts, fixed_body(bytes)),
                            pt,
                            ct,
                        )
                    }
                    Err(_) => (empty_status(StatusCode::BAD_GATEWAY), 0, 0),
                }
            }
        }
        Err(_) => (empty_status(StatusCode::BAD_GATEWAY), 0, 0),
    }
}

fn parse_usage(bytes: &Bytes) -> (i64, i64) {
    let Ok(v) = serde_json::from_slice::<Value>(bytes) else {
        return (0, 0);
    };
    let Some(usage) = v.get("usage") else {
        return (0, 0);
    };
    let pt = usage
        .get("prompt_tokens")
        .and_then(|x| x.as_i64())
        .unwrap_or(0);
    let ct = usage
        .get("completion_tokens")
        .and_then(|x| x.as_i64())
        .unwrap_or(0);
    (pt, ct)
}

fn parse_model(body: &Bytes) -> Option<String> {
    let v: Value = serde_json::from_slice(body).ok()?;
    v.get("model")?.as_str().map(String::from)
}

fn json_response(status: StatusCode, v: Value) -> Response<MuxBody> {
    let body = serde_json::to_vec(&v).unwrap_or_default();
    Response::builder()
        .status(status)
        .header("Content-Type", "application/json")
        .body(fixed_body(Bytes::from(body)))
        .unwrap_or_else(|_| empty_status(status))
}

fn empty_status(status: StatusCode) -> Response<MuxBody> {
    Response::builder()
        .status(status)
        .body(fixed_body(Bytes::new()))
        .unwrap()
}

fn fixed_body(bytes: Bytes) -> MuxBody {
    BodyExt::boxed(StreamBody::new(stream::once(async move {
        Ok(Frame::data(bytes))
    })))
}
