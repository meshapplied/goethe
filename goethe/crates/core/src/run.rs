// goethe run: try a model by hand, by input modality: text (a chat, one reply, an embedding summary) or an audio file (a transcript). It talks to the model's own engine on 127.0.0.1 with its engine key from state.db, never through the gateway, and creates no API key.

use crate::gateway_cache::GatewayCache;
use crate::paths::goethe_home;
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{Mutex, Once};

pub struct Opts {
    pub keep: bool,
    pub plain: bool,
    pub thinking: bool,
}

pub(crate) struct Loaded {
    // The models this run loaded, unloaded when it ends unless --keep; a model that was already loaded is never touched.
    models: Vec<String>,
    keep: bool,
    unload: Box<dyn Fn(&str)>,
}

impl Loaded {
    fn new(keep: bool) -> Loaded {
        if !keep {
            watch_signals();
        }
        Loaded { models: Vec::new(), keep, unload: Box::new(unload_model) }
    }

    pub(crate) fn add(&mut self, model: &str) {
        self.models.push(model.to_string());
        if !self.keep {
            lock(&BY_THIS_RUN).push(model.to_string());
        }
    }

    fn forget(&mut self, model: &str) {
        self.models.retain(|m| m != model);
        lock(&BY_THIS_RUN).retain(|m| m != model);
    }
}

impl Drop for Loaded {
    fn drop(&mut self) {
        if !self.keep {
            self.models.iter().for_each(|m| (self.unload)(m));
            lock(&BY_THIS_RUN).clear();
        }
    }
}

static BY_THIS_RUN: Mutex<Vec<String>> = Mutex::new(Vec::new());
static SIGNAL_FD: AtomicI32 = AtomicI32::new(-1);
static WATCH: Once = Once::new();

fn lock(list: &Mutex<Vec<String>>) -> std::sync::MutexGuard<'_, Vec<String>> {
    list.lock().unwrap_or_else(|e| e.into_inner())
}

extern "C" fn on_signal(sig: libc::c_int) {
    // Async-signal-safe: one byte down the pipe; a thread does the unloading.
    let byte = sig as u8;
    unsafe { libc::write(SIGNAL_FD.load(Ordering::SeqCst), (&byte as *const u8).cast(), 1) };
}

fn watch_signals() {
    // SIGINT, SIGTERM and SIGHUP of goethe run unload what this run loaded, then leave with the shell's code for the signal.
    WATCH.call_once(|| {
        let mut fds = [0; 2];
        if unsafe { libc::pipe(fds.as_mut_ptr()) } != 0 {
            return;
        }
        SIGNAL_FD.store(fds[1], Ordering::SeqCst);
        let handler = on_signal as extern "C" fn(libc::c_int) as libc::sighandler_t;
        for sig in [libc::SIGINT, libc::SIGTERM, libc::SIGHUP] {
            unsafe { libc::signal(sig, handler) };
        }
        let read_end = fds[0];
        std::thread::spawn(move || signal_worker(read_end, &BY_THIS_RUN, &unload_model, &|code| std::process::exit(code)));
    });
}

fn signal_worker(read_end: i32, models: &Mutex<Vec<String>>, unload: &dyn Fn(&str), leave: &dyn Fn(i32)) {
    let mut byte = 0u8;
    while unsafe { libc::read(read_end, (&mut byte as *mut u8).cast(), 1) } != 1 {
        if std::io::Error::last_os_error().kind() != std::io::ErrorKind::Interrupted {
            return;
        }
    }
    eprintln!("\ninterrupted; unloading what this run loaded");
    let list = std::mem::take(&mut *lock(models));
    list.iter().for_each(|m| unload(m));
    leave(128 + byte as i32);
}

fn unload_model(model: &str) {
    match crate::rpc::call("unload", json!({"model_id": model, "all": false, "force": false})) {
        Ok(_) => eprintln!("unloaded {model}"),
        Err(e) => eprintln!("could not unload {model}: {e}"),
    }
}

pub fn load_need(loaded: bool, installed: bool, model: &str) -> Result<bool, String> {
    // Whether run must load the model first: no when it serves, yes when installed, else the install hint (run never installs).
    match (loaded, installed) {
        (true, _) => Ok(false),
        (false, true) => Ok(true),
        (false, false) => Err(format!("{model} is not installed: goethe install {model}")),
    }
}

pub const RUN_CTX: u64 = 100_000;

fn run_ctx(home: &std::path::Path, model: &str) -> Option<u64> {
    // The context run loads a model with: RUN_CTX, or the model's maximum when that is smaller; None (the load default) when unknown.
    let con = crate::db::open(home).ok()?;
    let max: i64 = con.query_row("SELECT COALESCE(max_ctx, default_ctx) FROM installed_models WHERE model_id = ?1", [model], |r| r.get(0)).ok()?;
    run_ctx_for(max)
}

fn run_ctx_for(max: i64) -> Option<u64> {
    (max > 0).then(|| RUN_CTX.min(max as u64))
}

pub(crate) struct Target {
    pub(crate) model: String,
    pub(crate) kind: String,
    pub(crate) runtime: String,
    pub(crate) llama: bool,
    pub(crate) instance: String,
    pub(crate) api: Api,
}

pub(crate) fn target(model: &str) -> Result<Target, String> {
    // A loaded model's engine, its kind (the engine's, else the card's) and runtime.
    let home = goethe_home();
    let snap = GatewayCache::open(&home).exact();
    let r = snap.routes(model).iter().find(|r| !r.draining).ok_or_else(|| format!("{model} is not loaded: goethe load {model}"))?;
    let api = Api { base: format!("http://127.0.0.1:{}/v1", r.port), key: r.engine_key.clone() };
    let (card_kind, llama) = card_task(&crate::paths::model_dir(&home, model));
    let kind = kind_of(&api.call("GET", "/models", None)?, model, card_kind);
    let runtime = crate::engine_overhead::model_runtime(&home, model);
    Ok(Target { model: model.to_string(), kind, runtime, llama, instance: r.instance_id.clone(), api })
}

fn interactive(input: &Option<String>, plain: bool) -> bool {
    // The chat screen: no INPUT, a terminal on both ends, and not --plain.
    use std::io::IsTerminal;
    input.is_none() && !plain && std::io::stdin().is_terminal() && std::io::stdout().is_terminal()
}

pub fn run(args: &[String], opts: Opts) -> Result<(), String> {
    // MODEL INPUT as given, or on a terminal without either a model picked from the installed ones; one that is not loaded is loaded first (and unloaded at the end unless --keep).
    let home = goethe_home();
    let mut snap = GatewayCache::open(&home).exact();
    let mut args = args.to_vec();
    if let Some(first) = args.first().map(|f| snap.resolve_model(f).to_string()) {
        args[0] = first;
    }
    let installed = crate::models::installed_ids(&home);
    if args.is_empty() && interactive(&None, opts.plain) {
        match crate::menu::pick_model()? {
            Some(m) => args.push(m),
            None => return Ok(()),
        }
    }
    let (model, input) = route(&args, &snap.serving_models(), &installed.iter().map(String::as_str).collect::<Vec<_>>())?;
    let mut loaded = Loaded::new(opts.keep);
    if load_need(snap.routes(&model).iter().any(|r| !r.draining), installed.contains(&model), &model)? {
        loaded.add(&model);
        let ctx = run_ctx(&home, &model);
        if let Err(e) = crate::install::run_load(json!({"model_id": model, "gpus": null, "ctx": ctx, "max_conc": null, "tp": null, "force": false, "yes": false, "extra_sglang_args": [], "async_finish": true}), false, false) {
            loaded.forget(&model);
            return Err(e);
        }
        snap = GatewayCache::open(&home).exact();
    }
    drop(snap);
    let t = target(&model)?;
    if interactive(&input, opts.plain) {
        return crate::chat::run(t, &mut loaded, opts.thinking);
    }
    if let Some(e) = mismatch(&model, &t.kind, input.as_deref()) {
        return Err(e);
    }
    let (api, llama) = (&t.api, t.llama);
    match (t.kind.as_str(), input) {
        ("embedding", Some(text)) => api.embed(&model, &text).map(|e| println!("{}", embed_line(&e))),
        ("speech", Some(path)) => api.transcribe(&model, std::path::Path::new(&path), llama).map(|text| println!("{}", for_stdout(&text))),
        ("speech", None) => {
            let wav = record()?;
            let done = api.transcribe(&model, &wav, llama);
            let _ = std::fs::remove_file(&wav);
            let _ = wav.parent().map(std::fs::remove_dir);
            done.map(|text| println!("{}", for_stdout(&text)))
        }
        (_, Some(text)) => api.chat(&model, &[json!({"role": "user", "content": text})], &mut |_| true).map(|reply| println!("{}", shown(&reply, opts.thinking))),
        (_, None) => {
            let mut msgs = Vec::new();
            loop {
                print!("> ");
                let _ = std::io::stdout().flush();
                let mut line = String::new();
                if std::io::stdin().read_line(&mut line).map_err(|e| e.to_string())? == 0 || line.trim() == "/exit" {
                    return Ok(());
                }
                msgs.push(json!({"role": "user", "content": line.trim_end()}));
                match api.chat(&model, &msgs, &mut |_| true) {
                    Ok(reply) => {
                        println!("{}", shown(&reply, opts.thinking));
                        msgs.push(json!({"role": "assistant", "content": reply}));
                    }
                    Err(e) => {
                        msgs.pop();
                        eprintln!("{e}");
                    }
                }
            }
        }
    }
}

fn for_stdout(text: &str) -> String {
    // A transcript as received for a pipe; on a terminal without control characters.
    use std::io::IsTerminal;
    if std::io::stdout().is_terminal() { crate::chat::clean(text) } else { text.to_string() }
}

fn shown(reply: &str, thinking: bool) -> String {
    use std::io::IsTerminal;
    crate::chat::plain_text(reply, thinking, std::io::stdout().is_terminal())
}

pub fn route(args: &[String], loaded: &[&str], installed: &[&str]) -> Result<(String, Option<String>), String> {
    // MODEL and INPUT from up to two words: a first word that is a loaded or installed model is the model; with one loaded model it may be left out.
    let only = || match loaded {
        [m] => Ok(m.to_string()),
        [] => Err("no model is loaded: goethe load MODEL".to_string()),
        many => Err(format!("several models are loaded ({}): goethe run MODEL [INPUT]", many.join(", "))),
    };
    match args {
        [m, i] => Ok((m.clone(), Some(i.clone()))),
        [w] if loaded.contains(&w.as_str()) || installed.contains(&w.as_str()) => Ok((w.clone(), None)),
        [w] => only().map(|m| (m, Some(w.clone()))),
        _ => only().map(|m| (m, None)),
    }
}

pub fn mismatch(model: &str, kind: &str, input: Option<&str>) -> Option<String> {
    // Text goes to text and embedding models, an audio file on disk to speech models; no INPUT is a chat or a recording; else one named line.
    let audio = input.is_some_and(|i| {
        let p = std::path::Path::new(i);
        p.is_file() && p.extension().is_some_and(|x| ["wav", "flac", "mp3", "ogg", "m4a"].iter().any(|e| x.eq_ignore_ascii_case(e)))
    });
    let (ok, a, want) = match kind {
        "speech" => (input.is_none() || audio, "a speech", "an audio file"),
        "embedding" => (input.is_some() && !audio, "an embedding", "text"),
        _ => (!audio, "a text", "text"),
    };
    (!ok).then(|| format!("{model} is {a} model: give it {want}"))
}

pub(crate) fn card_task(dir: &std::path::Path) -> (&'static str, bool) {
    // The kind a card says when its engine names none (llama.cpp: task embed is an embedding model, task transcribe a speech model, else text), and whether llama.cpp serves it.
    let card = std::fs::read_to_string(dir.join(crate::install::CARD_FILE)).unwrap_or_default().parse::<toml::Table>().unwrap_or_default();
    let task = card.get("task").and_then(|t| t.as_str()).unwrap_or_default();
    let kind = match task {
        "embed" => "embedding",
        "transcribe" => "speech",
        _ => "text",
    };
    (kind, card.get("runtime").and_then(|r| r.as_str()) == Some("llama.cpp"))
}

pub fn kind_of(models: &Value, model: &str, card: &str) -> String {
    // The engine's /v1/models kind (goethe-engine: text, embedding or speech); other engines name none: the card's (llama.cpp task embed, else text).
    let data = models["data"].as_array().cloned().unwrap_or_default();
    let entry = data.iter().find(|m| m["id"] == model).or(data.first());
    entry.and_then(|m| m["kind"].as_str()).unwrap_or(card).to_string()
}

pub(crate) struct Api {
    pub(crate) base: String,
    pub(crate) key: Option<String>,
}

impl Api {
    fn req(&self, method: &str, path: &str) -> ureq::Request {
        let req = crate::net::agent().request(method, &format!("{}{path}", self.base));
        match &self.key {
            Some(k) => req.set("Authorization", &format!("Bearer {k}")),
            None => req,
        }
    }

    pub(crate) fn call(&self, method: &str, path: &str, body: Option<Value>) -> Result<Value, String> {
        let req = self.req(method, path);
        reply(match body {
            Some(b) => req.set("Content-Type", "application/json").send_string(&b.to_string()),
            None => req.call(),
        })
    }

    pub(crate) fn embed(&self, model: &str, text: &str) -> Result<Vec<f64>, String> {
        // The vector the engine returns for one input.
        let v = self.call("POST", "/embeddings", Some(json!({"model": model, "input": text})))?;
        Ok(v["data"][0]["embedding"].as_array().into_iter().flatten().filter_map(Value::as_f64).collect())
    }

    pub(crate) fn transcribe(&self, model: &str, path: &std::path::Path, chat: bool) -> Result<String, String> {
        // goethe-engine: OpenAI's multipart upload (the model field and the file); llama.cpp: a chat turn with the audio as input_audio.
        let audio = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        if chat {
            let format = path.extension().map(|e| e.to_string_lossy().to_ascii_lowercase()).unwrap_or_default();
            let part = json!([{"type": "text", "text": "Transcribe this audio."}, {"type": "input_audio", "input_audio": {"data": base64(&audio), "format": format}}]);
            let body = json!({"model": model, "messages": [{"role": "user", "content": part}], "stream": false});
            return Ok(self.call("POST", "/chat/completions", Some(body))?["choices"][0]["message"]["content"].as_str().unwrap_or_default().trim().to_string());
        }
        let (b, name) = ("goethe-run-boundary", path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default());
        let mut body = format!("--{b}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{model}\r\n--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{name}\"\r\nContent-Type: application/octet-stream\r\n\r\n").into_bytes();
        body.extend(audio);
        body.extend(format!("\r\n--{b}--\r\n").as_bytes());
        let req = self.req("POST", "/audio/transcriptions").set("Content-Type", &format!("multipart/form-data; boundary={b}"));
        Ok(reply(req.send_bytes(&body))?["text"].as_str().unwrap_or_default().to_string())
    }

    pub(crate) fn chat(&self, model: &str, msgs: &[Value], sink: &mut dyn FnMut(&str) -> bool) -> Result<String, String> {
        // Streams the reply piece by piece into the sink (which says false to stop) and returns it whole.
        let body = json!({"model": model, "messages": msgs, "stream": true}).to_string();
        let r = self.req("POST", "/chat/completions").set("Content-Type", "application/json").send_string(&body).map_err(fail)?;
        let mut out = String::new();
        for line in BufReader::new(r.into_reader()).lines() {
            let line = line.map_err(|e| e.to_string())?;
            let Some(data) = line.strip_prefix("data:").map(str::trim).filter(|d| *d != "[DONE]") else { continue };
            let piece = serde_json::from_str::<Value>(data).ok().and_then(|v| v["choices"][0]["delta"]["content"].as_str().map(String::from)).unwrap_or_default();
            out.push_str(&piece);
            if !sink(&piece) {
                break;
            }
        }
        Ok(out)
    }
}

pub(crate) fn embed_line(e: &[f64]) -> String {
    // One line: dimensions, L2 norm and the first 4 values.
    let head: Vec<String> = e.iter().take(4).map(|x| format!("{x:.4}")).collect();
    format!("dims={} norm={:.4} [{}, ...]", e.len(), e.iter().map(|x| x * x).sum::<f64>().sqrt(), head.join(", "))
}

fn base64(bytes: &[u8]) -> String {
    // Standard base64 with padding.
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for c in bytes.chunks(3) {
        let n = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
        for i in 0..4 {
            out.push(if i <= c.len() { T[(n >> (18 - 6 * i) & 63) as usize] as char } else { '=' });
        }
    }
    out
}

pub(crate) fn record() -> Result<std::path::PathBuf, String> {
    // 16 kHz mono PCM16 WAV from the first recorder on PATH, to a temp file, until Enter or Ctrl-D (SIGINT: the recorder closes the WAV).
    let wav = crate::paths::private_temp_dir("goethe-run")?.join("in.wav");
    let f = wav.to_string_lossy().into_owned();
    let f = f.as_str();
    let recorders: [(&str, &[&str]); 3] = [
        ("arecord", &["-q", "-f", "S16_LE", "-r", "16000", "-c", "1", f]),
        ("parecord", &["--rate=16000", "--channels=1", "--format=s16le", "--file-format=wav", f]),
        ("ffmpeg", &["-loglevel", "error", "-y", "-f", "pulse", "-i", "default", "-ar", "16000", "-ac", "1", f]),
    ];
    let mut child = recorders
        .iter()
        .find_map(|(bin, args)| std::process::Command::new(bin).args(*args).stdin(std::process::Stdio::null()).spawn().ok())
        .ok_or("no audio recorder found: install alsa-utils (arecord)")?;
    println!("recording: press Enter to stop");
    let _ = std::io::stdin().read_line(&mut String::new());
    unsafe { libc::kill(child.id() as i32, libc::SIGINT) };
    let _ = child.wait();
    Ok(wav)
}

fn reply(r: Result<ureq::Response, ureq::Error>) -> Result<Value, String> {
    let text = r.map_err(fail)?.into_string().map_err(|e| e.to_string())?;
    serde_json::from_str(&text).map_err(|e| e.to_string())
}

fn fail(e: ureq::Error) -> String {
    // The API's own error message in one line.
    match e {
        ureq::Error::Status(code, r) => {
            let text = r.into_string().unwrap_or_default();
            let v: Value = serde_json::from_str(&text).unwrap_or_default();
            v["error"]["message"].as_str().or(v["detail"].as_str()).map(String::from).unwrap_or_else(|| format!("HTTP {code}: {}", text.trim()))
        }
        e => e.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_signal_unloads_what_this_run_loaded_and_leaves_with_the_shell_code() {
        let mut fds = [0; 2];
        assert_eq!(unsafe { libc::pipe(fds.as_mut_ptr()) }, 0);
        let models = Mutex::new(vec!["org/a".to_string(), "org/b".to_string()]);
        let (unloaded, left) = (Mutex::new(Vec::new()), AtomicI32::new(0));
        let byte = libc::SIGTERM as u8;
        unsafe { libc::write(fds[1], (&byte as *const u8).cast(), 1) };
        signal_worker(fds[0], &models, &|m| unloaded.lock().unwrap().push(m.to_string()), &|c| left.store(c, Ordering::SeqCst));
        assert_eq!(*unloaded.lock().unwrap(), ["org/a", "org/b"]);
        assert_eq!(left.load(Ordering::SeqCst), 143);
        assert!(models.lock().unwrap().is_empty(), "nothing is unloaded twice");
    }

    #[test]
    fn words_route_to_a_model_and_an_input() {
        let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        let one = ["org/m"];
        let off = ["org/z"];
        assert_eq!(route(&s(&[]), &one, &off), Ok(("org/m".into(), None)));
        assert_eq!(route(&s(&["org/m"]), &one, &off), Ok(("org/m".into(), None)));
        assert_eq!(route(&s(&["org/z"]), &one, &off), Ok(("org/z".into(), None)), "an installed model that is not loaded is the model");
        assert_eq!(route(&s(&["hello there"]), &one, &off), Ok(("org/m".into(), Some("hello there".into()))));
        assert_eq!(route(&s(&["org/x", "hi"]), &one, &off), Ok(("org/x".into(), Some("hi".into()))));
        assert_eq!(route(&s(&["hi"]), &["org/a", "org/b"], &off).unwrap_err(), "several models are loaded (org/a, org/b): goethe run MODEL [INPUT]");
        assert_eq!(route(&s(&[]), &[], &off).unwrap_err(), "no model is loaded: goethe load MODEL");
    }

    #[test]
    fn the_input_modality_must_fit_the_model_kind() {
        let d = tempfile::tempdir().unwrap();
        let wav = d.path().join("a.WAV");
        std::fs::write(&wav, b"RIFF").unwrap();
        let wav = wav.to_str().unwrap();
        assert_eq!(mismatch("m", "text", None), None);
        assert_eq!(mismatch("m", "text", Some("hi")), None);
        assert_eq!(mismatch("m", "text", Some(wav)).unwrap(), "m is a text model: give it text");
        assert_eq!(mismatch("m", "embedding", Some("hi")), None);
        assert_eq!(mismatch("m", "embedding", None).unwrap(), "m is an embedding model: give it text");
        assert_eq!(mismatch("m", "speech", Some(wav)), None);
        assert_eq!(mismatch("m", "speech", Some("missing.wav")).unwrap(), "m is a speech model: give it an audio file");
        assert_eq!(mismatch("m", "speech", None), None, "no INPUT records from the microphone");
    }

    #[test]
    fn a_llama_cpp_embedding_card_runs_as_an_embedding_model() {
        // A fake llama-server: /v1/models names no kind, /v1/embeddings answers one vector; the card's task embed decides the kind.
        let srv = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = srv.local_addr().unwrap().port();
        std::thread::spawn(move || {
            for (stream, body) in srv.incoming().zip([r#"{"data": [{"id": "org/e"}]}"#, r#"{"data": [{"embedding": [0.6, 0.8, 0.0]}]}"#]) {
                let mut stream = stream.unwrap();
                let mut req = BufReader::new(stream.try_clone().unwrap());
                let mut len = 0;
                for line in (&mut req).lines().map_while(Result::ok).take_while(|l| !l.is_empty()) {
                    len = line.to_ascii_lowercase().strip_prefix("content-length:").map_or(len, |n| n.trim().parse().unwrap_or(0));
                }
                let _ = std::io::Read::read_exact(&mut req, &mut vec![0u8; len]);
                let _ = write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
            }
        });
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(card_task(dir.path()), ("text", false));
        std::fs::write(dir.path().join(crate::install::CARD_FILE), "runtime = \"llama.cpp\"\ntask = \"embed\"\n").unwrap();
        let api = Api { base: format!("http://127.0.0.1:{port}/v1"), key: Some("k".into()) };
        assert_eq!(kind_of(&api.call("GET", "/models", None).unwrap(), "org/e", card_task(dir.path()).0), "embedding");
        assert_eq!(embed_line(&api.embed("org/e", "hello world").unwrap()), "dims=3 norm=1.0000 [0.6000, 0.8000, 0.0000, ...]");
    }

    #[test]
    fn base64_matches_the_standard_vectors() {
        for (raw, enc) in [("", ""), ("f", "Zg=="), ("fo", "Zm8="), ("foo", "Zm9v"), ("foob", "Zm9vYg=="), ("fooba", "Zm9vYmE="), ("foobar", "Zm9vYmFy")] {
            assert_eq!(base64(raw.as_bytes()), enc);
        }
    }

    #[test]
    fn a_llama_cpp_transcribe_card_is_a_speech_model_that_takes_audio_as_a_chat_turn() {
        // A fake llama-server: the request must be a non-streaming chat turn holding the clip as base64 input_audio.
        let srv = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = srv.local_addr().unwrap().port();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let mut stream = srv.accept().unwrap().0;
            let mut req = BufReader::new(stream.try_clone().unwrap());
            let mut len = 0;
            for line in (&mut req).lines().map_while(Result::ok).take_while(|l| !l.is_empty()) {
                len = line.to_ascii_lowercase().strip_prefix("content-length:").map_or(len, |n| n.trim().parse().unwrap_or(0));
            }
            let mut body = vec![0u8; len];
            std::io::Read::read_exact(&mut req, &mut body).unwrap();
            tx.send(String::from_utf8(body).unwrap()).unwrap();
            let out = r#"{"choices": [{"message": {"content": " and so my fellow Americans "}}]}"#;
            let _ = write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{out}", out.len());
        });
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(crate::install::CARD_FILE), "runtime = \"llama.cpp\"\ntask = \"transcribe\"\n").unwrap();
        assert_eq!(card_task(dir.path()), ("speech", true));
        let wav = dir.path().join("a.wav");
        std::fs::write(&wav, b"RIFFfoo").unwrap();
        assert_eq!(mismatch("m", "speech", Some("plain text")).unwrap(), "m is a speech model: give it an audio file");
        let api = Api { base: format!("http://127.0.0.1:{port}/v1"), key: Some("k".into()) };
        assert_eq!(api.transcribe("org/t", &wav, true).unwrap(), "and so my fellow Americans");
        let sent: Value = serde_json::from_str(&rx.recv().unwrap()).unwrap();
        let audio = &sent["messages"][0]["content"][1];
        assert_eq!((sent["stream"].as_bool(), audio["type"].as_str(), audio["input_audio"]["format"].as_str()), (Some(false), Some("input_audio"), Some("wav")));
        assert_eq!(audio["input_audio"]["data"], base64(b"RIFFfoo"));
    }

    #[test]
    fn run_loads_only_an_installed_model_that_is_not_serving_and_never_installs() {
        assert_eq!(load_need(true, true, "org/m"), Ok(false), "serving: left alone");
        assert_eq!(load_need(false, true, "org/m"), Ok(true), "installed, not loaded: run loads it");
        assert_eq!(load_need(false, false, "org/m").unwrap_err(), "org/m is not installed: goethe install org/m");
    }

    #[test]
    fn run_loads_with_a_100k_context_or_the_model_maximum_when_smaller() {
        assert_eq!(run_ctx_for(262_144), Some(100_000));
        assert_eq!(run_ctx_for(32_768), Some(32_768));
        assert_eq!(run_ctx_for(0), None);
    }

    #[test]
    fn a_model_run_loaded_is_unloaded_at_the_end_unless_kept() {
        let seen = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let log = |keep: bool, models: &[&str]| {
            let sink = seen.clone();
            let mut g = Loaded { models: Vec::new(), keep, unload: Box::new(move |m| sink.borrow_mut().push(m.to_string())) };
            models.iter().for_each(|m| g.add(m));
        };
        log(false, &[]);
        assert!(seen.borrow().is_empty(), "nothing loaded by run: nothing unloaded");
        log(true, &["org/m"]);
        assert!(seen.borrow().is_empty(), "--keep leaves it loaded");
        log(false, &["org/m", "org/n"]);
        assert_eq!(*seen.borrow(), ["org/m", "org/n"]);
    }

    #[test]
    fn the_kind_comes_from_the_engine_model_list() {
        let v = json!({"data": [{"id": "a", "kind": "embedding"}, {"id": "b", "kind": "speech"}]});
        assert_eq!(kind_of(&v, "b", "text"), "speech");
        assert_eq!(kind_of(&v, "other", "text"), "embedding", "an engine serving one model under another name");
        assert_eq!(kind_of(&json!({"data": [{"id": "a"}]}), "a", "text"), "text");
        assert_eq!(kind_of(&json!({}), "a", "text"), "text");
        assert_eq!(kind_of(&json!({"data": [{"id": "a"}]}), "a", "embedding"), "embedding", "llama.cpp names no kind: the card's task decides");
    }
}
