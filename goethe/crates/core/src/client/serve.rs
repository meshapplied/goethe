// The node side of the reverse HTTP/2 connection (D9): goethe-server asks, the node answers live state, commands and routed requests.

use super::state::{self, Node};
use goethe_proto::SignedCommand;
use http_body_util::{BodyExt, Full, Limited};
use hyper::body::{Bytes, Frame, Incoming};
use hyper::header::{HeaderValue, CONTENT_TYPE};
use hyper::server::conn::http2;
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::{TokioExecutor, TokioIo, TokioTimer};
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::convert::Infallible;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::sync::mpsc;

pub type Body = http_body_util::combinators::BoxBody<Bytes, hyper::Error>;
pub type Gateway = Arc<dyn Fn(Request<Incoming>) -> Pin<Box<dyn Future<Output = Response<Body>> + Send>> + Send + Sync>;

const KEEPALIVE: Duration = Duration::from_secs(15);
const KEEPALIVE_TIMEOUT: Duration = Duration::from_secs(45);
const BODY_CAP: usize = 1 << 20;
const NDJSON: &str = "application/x-ndjson";

pub async fn serve<IO>(io: IO, node: Arc<Node>, gateway: Gateway) -> Result<(), hyper::Error>
where
    IO: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    // HTTP/2 with the node as the server on a connection the node opened; a ping every 15 s, and no answer within 45 s closes it.
    let svc = service_fn(move |req| {
        let (node, gateway) = (node.clone(), gateway.clone());
        async move { Ok::<_, Infallible>(route(node, gateway, req).await) }
    });
    http2::Builder::new(TokioExecutor::new())
        .timer(TokioTimer::new())
        .keep_alive_interval(KEEPALIVE)
        .keep_alive_timeout(KEEPALIVE_TIMEOUT)
        .serve_connection(TokioIo::new(io), svc)
        .await
}

async fn route(node: Arc<Node>, gateway: Gateway, req: Request<Incoming>) -> Response<Body> {
    // The routes of CONTRACTS 4.2; every other path, the local admin routes included, is 404.
    let path = req.uri().path().to_string();
    match (req.method().clone(), path.as_str()) {
        (Method::GET, "/state") => ndjson(state::open(node)),
        (Method::POST, "/cmd") => command(node, req).await,
        (Method::GET, p) if p.starts_with("/cmd/") => outcome(node, p["/cmd/".len()..].to_string()).await,
        (Method::POST, p) if p.starts_with("/v1/") => gateway(req).await,
        (Method::GET, "/v1/models") => gateway(req).await,
        _ => empty(StatusCode::NOT_FOUND),
    }
}

async fn command(node: Arc<Node>, req: Request<Incoming>) -> Response<Body> {
    // Run on the blocking pool; each event is one NDJSON line the moment it is emitted, and the command finishes even if the stream is cancelled.
    let Some(signed) = read::<SignedCommand>(req).await else {
        return empty(StatusCode::BAD_REQUEST);
    };
    let (tx, rx) = mpsc::channel(32);
    tokio::task::spawn_blocking(move || {
        super::exec::handle(&node.home, &signed, &mut |event| {
            let _ = tx.blocking_send(line(&event));
        })
    });
    ndjson(rx)
}

async fn outcome(node: Arc<Node>, cmd_id: String) -> Response<Body> {
    // The stored events of a command, for goethe-server after a reconnect.
    match tokio::task::spawn_blocking(move || super::exec::outcome(&node.home, &cmd_id)).await {
        Ok(Some(events)) => {
            let body: Vec<u8> = events.iter().flat_map(|e| line(e).to_vec()).collect();
            typed(Response::new(Full::new(Bytes::from(body)).map_err(|never| match never {}).boxed()))
        }
        _ => empty(StatusCode::NOT_FOUND),
    }
}

async fn read<T: DeserializeOwned>(req: Request<Incoming>) -> Option<T> {
    let body = Limited::new(req.into_body(), BODY_CAP).collect().await.ok()?.to_bytes();
    serde_json::from_slice(&body).ok()
}

pub(super) fn line<T: Serialize>(value: &T) -> Bytes {
    let mut out = serde_json::to_vec(value).unwrap_or_default();
    out.push(b'\n');
    Bytes::from(out)
}

fn empty(status: StatusCode) -> Response<Body> {
    let mut resp = Response::new(Full::new(Bytes::new()).map_err(|never| match never {}).boxed());
    resp.status_mut().clone_from(&status);
    resp
}

fn ndjson(rx: mpsc::Receiver<Bytes>) -> Response<Body> {
    typed(Response::new(Lines(rx).boxed()))
}

fn typed(mut resp: Response<Body>) -> Response<Body> {
    resp.headers_mut().insert(CONTENT_TYPE, HeaderValue::from_static(NDJSON));
    resp
}

struct Lines(mpsc::Receiver<Bytes>);

impl hyper::body::Body for Lines {
    type Data = Bytes;
    type Error = hyper::Error;

    fn poll_frame(mut self: Pin<&mut Self>, cx: &mut Context) -> Poll<Option<Result<Frame<Bytes>, hyper::Error>>> {
        self.0.poll_recv(cx).map(|line| line.map(|b| Ok(Frame::data(b))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hyper::client::conn::http2::{handshake, SendRequest};
    use serde_json::{json, Value};
    use std::io::{Read, Write};
    use std::path::Path;

    type Client = SendRequest<Full<Bytes>>;

    async fn connect(home: &Path, gateway: Gateway) -> (Client, Arc<Node>) {
        // goethe-server's side of the connection: an HTTP/2 client on the other end of an in-memory socket pair.
        let (ours, theirs) = tokio::io::duplex(1 << 16);
        let node = Arc::new(Node::new(home.to_path_buf(), "n1", "h1"));
        tokio::spawn(serve(ours, node.clone(), gateway));
        let (send, conn) = handshake(TokioExecutor::new(), TokioIo::new(theirs)).await.unwrap();
        tokio::spawn(conn);
        (send, node)
    }

    fn no_gateway() -> Gateway {
        Arc::new(|_| Box::pin(async { panic!("not routed") }))
    }

    async fn call(c: &mut Client, method: Method, path: &str, body: &str) -> Response<Incoming> {
        let req = Request::builder().method(method).uri(format!("http://node{path}")).body(Full::new(Bytes::from(body.to_string()))).unwrap();
        c.send_request(req).await.unwrap()
    }

    async fn text(resp: Response<Incoming>) -> String {
        String::from_utf8(resp.into_body().collect().await.unwrap().to_bytes().to_vec()).unwrap()
    }

    async fn next_line(body: &mut Incoming, buf: &mut Vec<u8>) -> Value {
        // One NDJSON line off the endless state stream, within 10 s.
        let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
        loop {
            if let Some(i) = buf.iter().position(|&b| b == b'\n') {
                let line: Vec<u8> = buf.drain(..=i).collect();
                return serde_json::from_slice(&line).unwrap();
            }
            let frame = tokio::time::timeout_at(deadline, body.frame()).await.expect("a state line in time").unwrap().unwrap();
            if let Some(d) = frame.data_ref() {
                buf.extend_from_slice(d);
            }
        }
    }

    fn fake_metrics(body: &'static str) -> u16 {
        // An engine /metrics endpoint answering every connection with `body`.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            for mut s in listener.incoming().flatten() {
                let mut req = [0u8; 4096];
                let _ = s.read(&mut req);
                let _ = write!(s, "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
            }
        });
        port
    }

    fn seed(home: &Path, port: u16) {
        // One running SGLang instance with an alias, and an unsent usage minute and error that must stay off this stream.
        let con = crate::db::open(home).unwrap();
        con.execute_batch(&format!(
            "INSERT INTO installed_models (model_id, local_path, architecture, static_vram_gb, kv_bytes_per_token, default_ctx, default_max_conc, default_tp)
             VALUES ('org/m', 'models/m', 'x', 1, 1, 4096, 8, 1);
             INSERT INTO loaded_instances (instance_id, model_id, pid, pgid, internal_port, gpu_indices, allocated_vram_gb_per_gpu, mem_fraction_static, ctx_len, max_conc, status)
             VALUES ('i-1', 'org/m', 1, 1, {port}, '0', 1, 0.8, 4096, 8, 'RUNNING');
             INSERT INTO model_aliases (alias, model_id) VALUES ('m', 'org/m');"
        ))
        .unwrap();
        std::fs::create_dir_all(home.join("models/m")).unwrap();
        std::fs::write(home.join("models/m/goethe_card.toml"), "runtime = \"sglang\"\n").unwrap();
        let now = chrono::Utc::now().timestamp();
        crate::usage_log::put_minute_for_test(&con, now - now.rem_euclid(60) - 600, "org/m", 5);
        std::fs::write(home.join("errors.log"), format!("{}\tINSTALL_DOWNLOAD\t0.3.6\tcould not download\n", now - 30)).unwrap();
    }

    #[tokio::test]
    async fn state_streams_hello_then_live_load_and_nothing_else() {
        let dir = tempfile::tempdir().unwrap();
        let port = fake_metrics("# HELP x\nsglang:num_queue_reqs{model_name=\"m\"} 3.0\nsglang:token_usage{model_name=\"m\"} 0.125\n");
        seed(dir.path(), port);
        let (mut c, _) = connect(dir.path(), no_gateway()).await;
        let resp = call(&mut c, Method::GET, "/state", "").await;
        assert_eq!((resp.status(), resp.headers()[CONTENT_TYPE].to_str().unwrap()), (StatusCode::OK, NDJSON));
        let mut body = resp.into_body();
        let mut buf = Vec::new();
        let hello = next_line(&mut body, &mut buf).await;
        assert_eq!(hello, json!({"type": "hello", "schema": "goethe.state.v1", "node_id": "n1", "install_hash": "h1", "goethe_version": env!("CARGO_PKG_VERSION"), "proto": 1}));
        let live = next_line(&mut body, &mut buf).await;
        assert_eq!(live["type"], "live", "only live load follows the hello: {live}");
        assert_eq!(
            live["instances"],
            json!([{"instance_id": "i-1", "model_id": "org/m", "aliases": ["m"], "status": "RUNNING", "in_flight": 0, "max_conc": 8, "queued": 3, "kv_cache_pct": 12.5}])
        );
        let mut guard = crate::inflight::track("org/m");
        guard.pick(&["i-1"]);
        let busy = next_line(&mut body, &mut buf).await;
        assert_eq!((busy["type"].as_str(), busy["instances"][0]["in_flight"].as_u64()), (Some("live"), Some(1)), "a new request shows at once");
        drop(guard);
        let con = crate::db::open(dir.path()).unwrap();
        assert_eq!((crate::db::setting(&con, "usage_sent_t"), crate::errcode::sent_to(&con).0), (None, 0), "usage and errors are the heartbeat's, never moved here");
    }

    #[tokio::test]
    async fn commands_stream_ndjson_and_other_paths_are_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let gateway: Gateway = Arc::new(|req: Request<Incoming>| {
            Box::pin(async move { Response::new(Full::new(Bytes::from(format!("gateway {} {}", req.method(), req.uri().path()))).map_err(|never| match never {}).boxed()) })
        });
        let (mut c, _) = connect(dir.path(), gateway).await;
        let signed = json!({"manifest_b64": "bQ", "body_b64": "Yg", "assertion": {"key_id": "k", "authenticator_data_b64": "YQ", "client_data_json_b64": "Yw", "signature_b64": "cw"}});
        let resp = call(&mut c, Method::POST, "/cmd", &signed.to_string()).await;
        assert_eq!((resp.status(), resp.headers()[CONTENT_TYPE].to_str().unwrap()), (StatusCode::OK, NDJSON));
        for l in text(resp).await.lines() {
            serde_json::from_str::<goethe_proto::CommandEvent>(l).unwrap();
        }
        assert_eq!(call(&mut c, Method::POST, "/cmd", r#"{"manifest_b64":"x"}"#).await.status(), StatusCode::BAD_REQUEST);
        assert_eq!(call(&mut c, Method::GET, "/cmd/0190-unknown", "").await.status(), StatusCode::NOT_FOUND);
        assert_eq!(text(call(&mut c, Method::POST, "/v1/chat/completions", "{}").await).await, "gateway POST /v1/chat/completions");
        assert_eq!(text(call(&mut c, Method::GET, "/v1/models", "").await).await, "gateway GET /v1/models");
        for (m, p) in [(Method::GET, "/health"), (Method::GET, "/metrics"), (Method::GET, "/health/admin"), (Method::GET, "/v1/chat/completions"), (Method::POST, "/ack"), (Method::POST, "/state"), (Method::GET, "/")] {
            assert_eq!(call(&mut c, m.clone(), p, "").await.status(), StatusCode::NOT_FOUND, "{m} {p}");
        }
    }
}
