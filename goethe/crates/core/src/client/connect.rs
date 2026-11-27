// The paid platform connection: dial the server the console named, prove the install key, and serve the node routes on the upgraded stream.

use super::{link, serve, Gateway, Node};
use http_body_util::{BodyExt, Empty};
use hyper::body::Bytes;
use hyper::client::conn::http1;
use hyper::header::{HeaderValue, AUTHORIZATION, CONNECTION, HOST, UPGRADE};
use hyper::{Method, Request, StatusCode};
use hyper_util::rt::TokioIo;
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::net::TcpStream;
use tokio::sync::watch;

const DIAL_TIMEOUT: Duration = Duration::from_secs(10);
const RETRY_MIN: Duration = Duration::from_secs(1);
const RETRY_MAX: Duration = Duration::from_secs(60);
const REFUSED_WAIT: Duration = Duration::from_secs(300);
const UNPAID_WAIT: Duration = Duration::from_secs(3600);
const NONCE_CAP: usize = 4096;

trait Io: AsyncRead + AsyncWrite + Unpin + Send {}
impl<T: AsyncRead + AsyncWrite + Unpin + Send> Io for T {}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Target {
    tls: bool,
    host: String,
    port: u16,
}

#[derive(Debug, PartialEq, Eq)]
enum Outcome {
    Closed,
    Unpaid,
    Refused,
    Failed(String),
}

fn target(url: &str) -> Result<Target, String> {
    // https anywhere, plain http on loopback only (the console URL policy); host and port from the authority.
    let url = link::check_url(url)?;
    let (scheme, rest) = url.split_once("://").ok_or("no scheme")?;
    let auth = rest.split('/').next().unwrap_or("");
    let (host, port) = match auth.strip_prefix('[') {
        Some(v6) => {
            let (h, tail) = v6.split_once(']').ok_or("bad IPv6 host")?;
            (h.to_string(), tail.strip_prefix(':').map(str::parse).transpose().map_err(|_| "bad port")?)
        }
        None => match auth.rsplit_once(':') {
            Some((h, p)) => (h.to_string(), Some(p.parse().map_err(|_| "bad port")?)),
            None => (auth.to_string(), None),
        },
    };
    let tls = scheme == "https";
    Ok(Target { tls, port: port.unwrap_or(if tls { 443 } else { 80 }), host })
}

async fn open(t: &Target) -> Result<Box<dyn Io>, String> {
    // TCP, then TLS against the web roots when the URL is https.
    let tcp = tokio::time::timeout(DIAL_TIMEOUT, TcpStream::connect((t.host.as_str(), t.port)))
        .await
        .map_err(|_| format!("{}:{} did not answer", t.host, t.port))?
        .map_err(|e| format!("{}:{}: {e}", t.host, t.port))?;
    if !t.tls {
        return Ok(Box::new(tcp));
    }
    let roots = tokio_rustls::rustls::RootCertStore { roots: webpki_roots::TLS_SERVER_ROOTS.to_vec() };
    let config = tokio_rustls::rustls::ClientConfig::builder().with_root_certificates(roots).with_no_client_auth();
    let name = tokio_rustls::rustls::pki_types::ServerName::try_from(t.host.clone()).map_err(|_| format!("{} is not a valid server name", t.host))?;
    let tls = tokio::time::timeout(DIAL_TIMEOUT, tokio_rustls::TlsConnector::from(Arc::new(config)).connect(name, tcp))
        .await
        .map_err(|_| format!("TLS to {} timed out", t.host))?
        .map_err(|e| format!("TLS to {}: {e}", t.host))?;
    Ok(Box::new(tls))
}

fn request(t: &Target, method: Method, path: &str, token: &str) -> Result<hyper::http::request::Builder, String> {
    let host = if (t.tls && t.port == 443) || (!t.tls && t.port == 80) { t.host.clone() } else { format!("{}:{}", t.host, t.port) };
    let bearer = HeaderValue::from_str(&format!("Bearer {token}")).map_err(|_| "the node token is not a valid header")?;
    Ok(Request::builder().method(method).uri(path).header(HOST, host).header(AUTHORIZATION, bearer))
}

fn refusal(status: StatusCode) -> Outcome {
    match status {
        StatusCode::PAYMENT_REQUIRED => Outcome::Unpaid,
        StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => Outcome::Refused,
        s => Outcome::Failed(format!("the platform answered {s}")),
    }
}

async fn nonce(t: &Target, token: &str) -> Result<String, Outcome> {
    // One-time value for the proof, on its own short connection.
    #[derive(Deserialize)]
    struct Nonce {
        nonce: String,
    }
    let fail = |e: String| Outcome::Failed(e);
    let (mut send, conn) = http1::handshake(TokioIo::new(open(t).await.map_err(fail)?)).await.map_err(|e| fail(e.to_string()))?;
    tokio::spawn(conn);
    let req = request(t, Method::GET, "/node/nonce", token).map_err(fail)?.body(Empty::<Bytes>::new()).map_err(|e| fail(e.to_string()))?;
    let resp = send.send_request(req).await.map_err(|e| fail(e.to_string()))?;
    if resp.status() != StatusCode::OK {
        return Err(refusal(resp.status()));
    }
    let body = http_body_util::Limited::new(resp.into_body(), NONCE_CAP).collect().await.map_err(|e| fail(e.to_string()))?.to_bytes();
    let n: Nonce = serde_json::from_slice(&body).map_err(|_| fail("the platform sent no nonce".into()))?;
    if n.nonce.is_empty() || n.nonce.len() > 256 || !n.nonce.chars().all(|c| c.is_ascii_graphic()) {
        return Err(fail("the platform sent a malformed nonce".into()));
    }
    Ok(n.nonce)
}

async fn once(url: &str, home: &Path, gateway: &Gateway) -> Outcome {
    // One connection: nonce, proof, 101, then the node routes until either side closes it.
    let Ok(Some(creds)) = link::load_credentials(home) else {
        return Outcome::Refused;
    };
    let key = match link::node_key(home) {
        Ok(k) => k,
        Err(e) => return Outcome::Failed(e),
    };
    let t = match target(url) {
        Ok(t) => t,
        Err(e) => return Outcome::Failed(e),
    };
    let token = creds.token.expose();
    let nonce = match nonce(&t, token).await {
        Ok(n) => n,
        Err(o) => return o,
    };
    let io = match open(&t).await {
        Ok(io) => io,
        Err(e) => return Outcome::Failed(e),
    };
    let Ok((mut send, conn)) = http1::handshake(TokioIo::new(io)).await else {
        return Outcome::Failed("HTTP handshake failed".into());
    };
    tokio::spawn(conn.with_upgrades());
    let proof = key.connect_proof(&nonce, &creds.node_id);
    let req = match request(&t, Method::POST, "/node/connect", token).and_then(|b| {
        b.header(CONNECTION, "upgrade")
            .header(UPGRADE, goethe_proto::CONNECT_UPGRADE)
            .header(goethe_proto::PROOF_HEADER, proof)
            .body(Empty::<Bytes>::new())
            .map_err(|e| e.to_string())
    }) {
        Ok(r) => r,
        Err(e) => return Outcome::Failed(e),
    };
    let resp = match send.send_request(req).await {
        Ok(r) => r,
        Err(e) => return Outcome::Failed(e.to_string()),
    };
    if resp.status() != StatusCode::SWITCHING_PROTOCOLS {
        return refusal(resp.status());
    }
    let upgraded = match hyper::upgrade::on(resp).await {
        Ok(u) => u,
        Err(e) => return Outcome::Failed(e.to_string()),
    };
    let node = Arc::new(Node::new(home.to_path_buf(), &creds.node_id, &key.hash()));
    let _ = serve(TokioIo::new(upgraded), node, gateway.clone()).await;
    Outcome::Closed
}

fn jitter(d: Duration) -> Duration {
    // Up to a quarter more, so a fleet that lost the server does not come back in step.
    let n = SystemTime::now().duration_since(UNIX_EPOCH).map(|t| t.subsec_nanos()).unwrap_or(0) as u64;
    d + Duration::from_millis(n % (d.as_millis() as u64 / 4 + 1))
}

pub async fn run(home: PathBuf, gateway: Gateway, mut url: watch::Receiver<Option<String>>) {
    // Hold the connection while the console names the platform (a paid workspace): reconnect with backoff, wait an hour after 402, five minutes after 401.
    let mut backoff = RETRY_MIN;
    let mut said = String::new();
    loop {
        let Some(server) = url.borrow_and_update().clone() else {
            if url.changed().await.is_err() {
                return;
            }
            continue;
        };
        let outcome = once(&server, &home, &gateway).await;
        let (wait, note) = match &outcome {
            Outcome::Closed => {
                backoff = RETRY_MIN;
                (RETRY_MIN, format!("platform: connection to {server} closed; reconnecting"))
            }
            Outcome::Unpaid => (UNPAID_WAIT, "platform: this workspace is not on a paid plan; routing stays off".to_string()),
            Outcome::Refused => (REFUSED_WAIT, "platform: the node was refused (token or install key)".to_string()),
            Outcome::Failed(e) => {
                backoff = (backoff * 2).min(RETRY_MAX);
                (backoff, format!("platform: {e}"))
            }
        };
        if note != said {
            crate::supervisor::append_goethed_log(&home, &format!("{note}\n"));
            said = note;
        }
        tokio::select! {
            _ = tokio::time::sleep(jitter(wait)) => {}
            changed = url.changed() => {
                if changed.is_err() {
                    return;
                }
                backoff = RETRY_MIN;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::api::Approved;
    use crate::client::link::{Secret, Workspace};
    use hyper::server::conn::http1 as server;
    use hyper::service::service_fn;
    use hyper_util::rt::TokioExecutor;
    use std::convert::Infallible;
    use tokio::sync::oneshot;

    fn linked(home: &Path) -> String {
        let approved = Approved { node_id: "n1".into(), node_token: Secret::new("mesh_node_cccccccccccccccc"), workspace: Workspace { id: "w1".into(), name: "Lab".into() }, trust_root: None };
        link::save_credentials(home, &approved.into_credentials("http://127.0.0.1:1")).unwrap();
        link::node_key(home).unwrap().public_b64()
    }

    async fn fake_server(nonce_status: StatusCode, install_pub: String, hello: oneshot::Sender<serde_json::Value>) -> String {
        // The platform's node side: a nonce, the proof check, 101, then an HTTP/2 client that reads the node's hello off GET /state.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let hello = Arc::new(std::sync::Mutex::new(Some(hello)));
        tokio::spawn(async move {
            loop {
                let (tcp, _) = listener.accept().await.unwrap();
                let (pub_b64, hello) = (install_pub.clone(), hello.clone());
                tokio::spawn(async move {
                    let svc = service_fn(move |req: Request<hyper::body::Incoming>| {
                        let (pub_b64, hello) = (pub_b64.clone(), hello.clone());
                        async move {
                            let bearer = req.headers().get(AUTHORIZATION).and_then(|v| v.to_str().ok()) == Some("Bearer mesh_node_cccccccccccccccc");
                            let mut resp = hyper::Response::new(http_body_util::Full::new(Bytes::from_static(br#"{"nonce":"nonce-1","expires_ms":60000}"#)));
                            match req.uri().path() {
                                "/node/nonce" if bearer => *resp.status_mut() = nonce_status,
                                "/node/connect" if bearer => {
                                    let proof = req.headers().get(goethe_proto::PROOF_HEADER).and_then(|v| v.to_str().ok()).unwrap_or("").to_string();
                                    if !goethe_proto::verify_connect(&pub_b64, "nonce-1", "n1", &proof) {
                                        resp.status_mut().clone_from(&StatusCode::UNAUTHORIZED);
                                        return Ok::<_, Infallible>(resp);
                                    }
                                    resp.status_mut().clone_from(&StatusCode::SWITCHING_PROTOCOLS);
                                    resp.headers_mut().insert(UPGRADE, HeaderValue::from_static(goethe_proto::CONNECT_UPGRADE));
                                    tokio::spawn(async move {
                                        let up = hyper::upgrade::on(req).await.unwrap();
                                        let (mut h2, conn) = hyper::client::conn::http2::handshake(TokioExecutor::new(), up).await.unwrap();
                                        tokio::spawn(conn);
                                        let r = h2.send_request(Request::get("http://node/state").body(Empty::<Bytes>::new()).unwrap()).await.unwrap();
                                        let mut body = r.into_body();
                                        let mut buf = Vec::new();
                                        while !buf.contains(&b'\n') {
                                            let frame = body.frame().await.unwrap().unwrap();
                                            buf.extend_from_slice(frame.data_ref().unwrap());
                                        }
                                        let line = buf.split(|&b| b == b'\n').next().unwrap().to_vec();
                                        if let Some(tx) = hello.lock().unwrap().take() {
                                            let _ = tx.send(serde_json::from_slice(&line).unwrap());
                                        }
                                    });
                                }
                                _ => *resp.status_mut() = StatusCode::UNAUTHORIZED,
                            }
                            Ok::<_, Infallible>(resp)
                        }
                    });
                    let _ = server::Builder::new().serve_connection(TokioIo::new(tcp), svc).with_upgrades().await;
                });
            }
        });
        url
    }

    fn no_gateway() -> Gateway {
        Arc::new(|_| Box::pin(async { panic!("not routed") }))
    }

    #[test]
    fn targets_take_https_anywhere_and_http_only_on_loopback() {
        assert_eq!(target("https://server.goethe.sh").unwrap(), Target { tls: true, host: "server.goethe.sh".into(), port: 443 });
        assert_eq!(target("http://127.0.0.1:9000").unwrap(), Target { tls: false, host: "127.0.0.1".into(), port: 9000 });
        assert_eq!(target("https://[::1]:8443").unwrap(), Target { tls: true, host: "::1".into(), port: 8443 });
        assert!(target("http://server.goethe.sh").is_err());
    }

    #[tokio::test]
    async fn the_node_proves_its_install_key_and_serves_its_routes_on_the_upgraded_stream() {
        let dir = tempfile::tempdir().unwrap();
        let install_pub = linked(dir.path());
        let (tx, rx) = oneshot::channel();
        let url = fake_server(StatusCode::OK, install_pub, tx).await;
        let home = dir.path().to_path_buf();
        let task = tokio::spawn(async move { once(&url, &home, &no_gateway()).await });
        let hello = tokio::time::timeout(Duration::from_secs(10), rx).await.expect("hello in time").unwrap();
        assert_eq!((hello["type"].as_str(), hello["node_id"].as_str()), (Some("hello"), Some("n1")));
        assert_eq!(hello["install_hash"].as_str(), Some(link::node_key(dir.path()).unwrap().hash().as_str()));
        task.abort();
    }

    #[tokio::test]
    async fn an_unpaid_workspace_or_a_wrong_key_never_gets_a_connection() {
        let dir = tempfile::tempdir().unwrap();
        let install_pub = linked(dir.path());
        let url = fake_server(StatusCode::PAYMENT_REQUIRED, install_pub, oneshot::channel().0).await;
        assert_eq!(once(&url, dir.path(), &no_gateway()).await, Outcome::Unpaid);
        let other = tempfile::tempdir().unwrap();
        linked(other.path());
        let url = fake_server(StatusCode::OK, link::node_key(dir.path()).unwrap().public_b64(), oneshot::channel().0).await;
        assert_eq!(once(&url, other.path(), &no_gateway()).await, Outcome::Refused, "another install's key does not pass the proof");
        assert_eq!(once(&url, tempfile::tempdir().unwrap().path(), &no_gateway()).await, Outcome::Refused, "an unlinked node does not dial");
    }
}
