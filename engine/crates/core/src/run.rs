// goethe run: try a loaded model by hand, by input modality: text (a chat, one reply, an embedding summary) or an audio file (a transcript).

use crate::gateway_cache::GatewayCache;
use crate::paths::goethe_home;
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};

pub fn run(args: &[String]) -> Result<(), String> {
    // The model's own engine on 127.0.0.1 with its engine key from state.db: no API key is created or shown.
    let snap = GatewayCache::open(&goethe_home()).exact();
    let mut args = args.to_vec();
    if let Some(first) = args.first().map(|f| snap.resolve_model(f).to_string()) {
        args[0] = first;
    }
    let (model, input) = route(&args, &snap.serving_models())?;
    let r = snap.routes(&model).iter().find(|r| !r.draining).ok_or_else(|| format!("{model} is not loaded: goethe load {model}"))?;
    let api = Api { base: format!("http://127.0.0.1:{}/v1", r.port), key: r.engine_key.clone() };
    let card = std::fs::read_to_string(crate::paths::model_dir(&goethe_home(), &model).join(crate::install::CARD_FILE)).unwrap_or_default();
    let task = card.parse::<toml::Table>().ok().and_then(|c| c.get("task").and_then(|t| t.as_str().map(String::from)));
    let kind = kind_of(&api.call("GET", "/models", None)?, &model, if task.as_deref() == Some("embed") { "embedding" } else { "text" });
    if let Some(e) = mismatch(&model, &kind, input.as_deref()) {
        return Err(e);
    }
    match (kind.as_str(), input) {
        ("embedding", Some(text)) => {
            let v = api.call("POST", "/embeddings", Some(json!({"model": model, "input": text})))?;
            let e: Vec<f64> = v["data"][0]["embedding"].as_array().into_iter().flatten().filter_map(Value::as_f64).collect();
            let head: Vec<String> = e.iter().take(4).map(|x| format!("{x:.4}")).collect();
            println!("dims={} norm={:.4} [{}, ...]", e.len(), e.iter().map(|x| x * x).sum::<f64>().sqrt(), head.join(", "));
            Ok(())
        }
        ("speech", Some(path)) => api.transcribe(&model, std::path::Path::new(&path)),
        ("speech", None) => {
            let wav = record()?;
            let done = api.transcribe(&model, &wav);
            let _ = std::fs::remove_file(&wav);
            done
        }
        (_, Some(text)) => api.chat(&model, &[json!({"role": "user", "content": text})]).map(|_| ()),
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
                match api.chat(&model, &msgs) {
                    Ok(reply) => msgs.push(json!({"role": "assistant", "content": reply})),
                    Err(e) => {
                        msgs.pop();
                        eprintln!("{e}");
                    }
                }
            }
        }
    }
}

pub fn route(args: &[String], loaded: &[&str]) -> Result<(String, Option<String>), String> {
    // MODEL and INPUT from up to two words: a first word that is a loaded model is the model; with one loaded model it may be left out.
    let only = || match loaded {
        [m] => Ok(m.to_string()),
        [] => Err("no model is loaded: goethe load MODEL".to_string()),
        many => Err(format!("several models are loaded ({}): goethe run MODEL [INPUT]", many.join(", "))),
    };
    match args {
        [m, i] => Ok((m.clone(), Some(i.clone()))),
        [w] if loaded.contains(&w.as_str()) => Ok((w.clone(), None)),
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

pub fn kind_of(models: &Value, model: &str, card: &str) -> String {
    // The engine's /v1/models kind (goethe-engine: text, embedding or speech); other engines name none: the card's (llama.cpp task embed, else text).
    let data = models["data"].as_array().cloned().unwrap_or_default();
    let entry = data.iter().find(|m| m["id"] == model).or(data.first());
    entry.and_then(|m| m["kind"].as_str()).unwrap_or(card).to_string()
}

struct Api {
    base: String,
    key: Option<String>,
}

impl Api {
    fn req(&self, method: &str, path: &str) -> ureq::Request {
        let req = crate::net::agent().request(method, &format!("{}{path}", self.base));
        match &self.key {
            Some(k) => req.set("Authorization", &format!("Bearer {k}")),
            None => req,
        }
    }

    fn call(&self, method: &str, path: &str, body: Option<Value>) -> Result<Value, String> {
        let req = self.req(method, path);
        reply(match body {
            Some(b) => req.set("Content-Type", "application/json").send_string(&b.to_string()),
            None => req.call(),
        })
    }

    fn transcribe(&self, model: &str, path: &std::path::Path) -> Result<(), String> {
        // OpenAI's multipart upload: the model field and the file.
        let audio = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let (b, name) = ("goethe-run-boundary", path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default());
        let mut body = format!("--{b}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{model}\r\n--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{name}\"\r\nContent-Type: application/octet-stream\r\n\r\n").into_bytes();
        body.extend(audio);
        body.extend(format!("\r\n--{b}--\r\n").as_bytes());
        let req = self.req("POST", "/audio/transcriptions").set("Content-Type", &format!("multipart/form-data; boundary={b}"));
        println!("{}", reply(req.send_bytes(&body))?["text"].as_str().unwrap_or_default());
        Ok(())
    }

    fn chat(&self, model: &str, msgs: &[Value]) -> Result<String, String> {
        // Streams the reply to stdout as it arrives and returns it whole.
        let body = json!({"model": model, "messages": msgs, "stream": true}).to_string();
        let r = self.req("POST", "/chat/completions").set("Content-Type", "application/json").send_string(&body).map_err(fail)?;
        let mut out = String::new();
        for line in BufReader::new(r.into_reader()).lines() {
            let line = line.map_err(|e| e.to_string())?;
            let Some(data) = line.strip_prefix("data:").map(str::trim).filter(|d| *d != "[DONE]") else { continue };
            let piece = serde_json::from_str::<Value>(data).ok().and_then(|v| v["choices"][0]["delta"]["content"].as_str().map(String::from)).unwrap_or_default();
            print!("{piece}");
            let _ = std::io::stdout().flush();
            out.push_str(&piece);
        }
        println!();
        Ok(out)
    }
}

fn record() -> Result<std::path::PathBuf, String> {
    // 16 kHz mono PCM16 WAV from the first recorder on PATH, to a temp file, until Enter or Ctrl-D (SIGINT: the recorder closes the WAV).
    let wav = std::env::temp_dir().join(format!("goethe-run-{}.wav", std::process::id()));
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
    fn words_route_to_a_model_and_an_input() {
        let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        let one = ["org/m"];
        assert_eq!(route(&s(&[]), &one), Ok(("org/m".into(), None)));
        assert_eq!(route(&s(&["org/m"]), &one), Ok(("org/m".into(), None)));
        assert_eq!(route(&s(&["hello there"]), &one), Ok(("org/m".into(), Some("hello there".into()))));
        assert_eq!(route(&s(&["org/x", "hi"]), &one), Ok(("org/x".into(), Some("hi".into()))));
        assert_eq!(route(&s(&["hi"]), &["org/a", "org/b"]).unwrap_err(), "several models are loaded (org/a, org/b): goethe run MODEL [INPUT]");
        assert_eq!(route(&s(&[]), &[]).unwrap_err(), "no model is loaded: goethe load MODEL");
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
    fn the_kind_comes_from_the_engine_model_list() {
        let v = json!({"data": [{"id": "a", "kind": "embedding"}, {"id": "b", "kind": "speech"}]});
        assert_eq!(kind_of(&v, "b", "text"), "speech");
        assert_eq!(kind_of(&v, "other", "text"), "embedding", "an engine serving one model under another name");
        assert_eq!(kind_of(&json!({"data": [{"id": "a"}]}), "a", "text"), "text");
        assert_eq!(kind_of(&json!({}), "a", "text"), "text");
        assert_eq!(kind_of(&json!({"data": [{"id": "a"}]}), "a", "embedding"), "embedding", "llama.cpp names no kind: the card's task decides");
    }
}
