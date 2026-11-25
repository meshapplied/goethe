// Refusals before ranking (shell characters, requests Goethe has nothing for) and the parsers for typed values in a request.

fn quotes(s: &str) -> Vec<(usize, usize)> {
    // Quoted spans as byte ranges of their two quote marks: a quote opens at the start of a word and closes at the end of one, so "let's" is no quote and an opener that never closes is none either.
    let ends_word = |i: usize| s[i + 1..].chars().next().is_none_or(|c| c.is_whitespace() || ",.;:!?)".contains(c));
    let mut spans = Vec::new();
    let mut open: Option<(usize, char)> = None;
    let mut prev = ' ';
    for (i, ch) in s.char_indices() {
        match open {
            Some((at, q)) if ch == q && ends_word(i) => {
                spans.push((at, i));
                open = None;
            }
            None if (ch == '"' || ch == '\'') && prev.is_whitespace() => open = Some((i, ch)),
            _ => {}
        }
        prev = ch;
    }
    spans
}

fn quoted(spans: &[(usize, usize)], i: usize) -> bool {
    spans.iter().any(|(a, b)| (*a..=*b).contains(&i))
}

pub fn find_shell_metachar(s: &str) -> Option<&'static str> {
    // Reject shell chaining, piping, command substitution, and redirection outside quotes (and $( or ` anywhere).
    for always in ["$(", "`", "\n", "\r"] {
        if s.contains(always) {
            return Some(always);
        }
    }
    let spans = quotes(s);
    s.char_indices().filter(|(i, _)| !quoted(&spans, *i)).find_map(|(i, _)| ["&&", "||", ";", "|", ">", "<"].into_iter().find(|p| s[i..].starts_with(p)))
}

pub fn is_obvious_ood(query: &str) -> bool {
    // Prompt injection and other tools' commands are refused here; small talk is refused later, when no command ranks high enough.
    let l = query.to_ascii_lowercase();
    const REJECT: &[&str] = &[
        "ignore previous",
        "ignore all instructions",
        "disregard previous",
        "system prompt",
        "rm -rf",
        "sudo ",
        "git commit",
        "git push",
        "git rebase",
        "docker run",
        "docker build",
        "kubectl ",
        "apt install",
        "brew install",
        "pip install",
        "cargo build",
        "npm install",
    ];
    REJECT.iter().any(|p| l.contains(p))
}

pub fn split_parts(s: &str) -> Vec<String> {
    // A request in parts: cut at the words "and" / "then" / "also" / "but" and after a word ending in a comma, all outside quotes ("X, Y and then Z").
    let spans = quotes(s);
    let mut parts = Vec::new();
    let mut cur = String::new();
    for word in s.split_whitespace() {
        let outside = !quoted(&spans, word.as_ptr() as usize - s.as_ptr() as usize);
        if outside && matches!(word.to_ascii_lowercase().as_str(), "and" | "then" | "also" | "but") {
            parts.push(std::mem::take(&mut cur));
            continue;
        }
        let end = word.as_ptr() as usize - s.as_ptr() as usize + word.len() - 1;
        match word.strip_suffix(',') {
            Some(head) if !quoted(&spans, end) => {
                cur.push_str(head);
                parts.push(std::mem::take(&mut cur));
            }
            _ => {
                cur.push_str(word);
                cur.push(' ');
            }
        }
    }
    parts.push(cur);
    parts.into_iter().map(|p| p.trim().to_string()).filter(|p| !p.is_empty()).collect()
}

pub fn without_wish(s: &str) -> String {
    // "I want you to", "can you please": a wish before the verb says nothing about the command and drowns the verb.
    const WISH: [&str; 14] = ["i", "i'd", "want", "would", "like", "need", "you", "to", "can", "could", "will", "please", "pls", "kindly"];
    let mut rest = s.trim_start();
    while let Some((word, tail)) = rest.split_once(char::is_whitespace) {
        if !WISH.contains(&word.trim_end_matches(',').to_ascii_lowercase().as_str()) {
            break;
        }
        rest = tail.trim_start();
    }
    if rest.is_empty() { s.to_string() } else { rest.to_string() }
}

pub fn split_words(s: &str) -> Vec<String> {
    // Shell-like quote-preserving word splitter without running a shell.
    let spans = quotes(s);
    let mut out = Vec::new();
    let mut cur = String::new();
    for (i, ch) in s.char_indices() {
        if spans.iter().any(|(a, b)| i == *a || i == *b) {
            continue;
        }
        if ch.is_whitespace() && !quoted(&spans, i) {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
        } else {
            cur.push(ch);
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

pub fn find_quoted(text: &str) -> Option<String> {
    quotes(text).into_iter().map(|(a, b)| text[a + 1..b].trim()).find(|inner| !inner.is_empty()).map(str::to_string)
}

pub fn is_local_path(t: &str) -> bool {
    t.starts_with("./") || t.starts_with("../") || t.starts_with("~/") || (t.starts_with('/') && t.len() > 1)
}

pub fn is_url(t: &str) -> bool {
    t.starts_with("https://") || t.starts_with("http://")
}

pub fn is_auth_key(t: &str) -> bool {
    t.starts_with("msh_ak_")
}

pub fn negation(text: &str) -> Option<String> {
    // A word that says not to do something; ranking ignores it, so "don't stop goethe" would read as "stop goethe".
    text.split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '\'' && c != '\u{2019}').to_ascii_lowercase().replace('\u{2019}', "'"))
        .find(|w| matches!(w.as_str(), "not" | "never" | "dont" | "doesnt" | "didnt") || w.ends_with("n't"))
}

pub fn find_hf_or_path(tokens: &[String]) -> Option<String> {
    for t in tokens {
        let clean = t.trim_matches(|c: char| matches!(c, ',' | ';' | ')' | '('));
        if is_local_path(clean) {
            return Some(clean.to_string());
        }
        if let Some((org, name)) = clean.split_once('/') {
            if !org.is_empty()
                && !name.is_empty()
                && !org.contains(':')
                && !clean.starts_with("http")
                && !is_cidr(clean)
                && org.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
                && name.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | ':'))
            {
                return Some(clean.to_string());
            }
        }
    }
    None
}

pub fn find_path_token(tokens: &[String]) -> Option<String> {
    tokens
        .iter()
        .find(|t| is_local_path(t))
        .cloned()
}

pub fn find_gpus(text: &str) -> Vec<String> {
    let toks = split_words(&text.to_ascii_lowercase());
    let mut out = Vec::new();
    let mut i = 0;
    while i < toks.len() {
        let t = toks[i].trim_matches(|c: char| matches!(c, ',' | '.'));
        if is_gpu_tok(t) {
            out.push(t.to_string());
        } else if t == "gpu" && i + 1 < toks.len() && toks[i + 1].chars().all(|c| c.is_ascii_digit()) {
            out.push(format!("gpu{}", toks[i + 1]));
            i += 1;
        }
        i += 1;
    }
    out.sort();
    out.dedup();
    out
}

fn is_gpu_tok(t: &str) -> bool {
    t.strip_prefix("gpu").is_some_and(|rest| !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit()))
}

pub fn find_cidrs(tokens: &[String]) -> Vec<String> {
    // Addresses and ranges, also one that ends a sentence ("allow 203.0.113.5.").
    let mut out = Vec::new();
    for t in tokens {
        let c = t.trim_matches(|ch: char| matches!(ch, ',' | ';' | ')' | '(' | '.'));
        if is_cidr(c) || c.parse::<std::net::IpAddr>().is_ok() {
            out.push(c.to_string());
        }
    }
    out
}

pub fn is_cidr(s: &str) -> bool {
    let Some((ip, prefix)) = s.split_once('/') else {
        return false;
    };
    ip.parse::<std::net::IpAddr>().is_ok() && prefix.parse::<u8>().is_ok_and(|p| p <= 128)
}

pub fn find_duration(tokens: &[String]) -> Option<String> {
    for (i, t) in tokens.iter().enumerate() {
        let l = t.trim_matches(|c: char| matches!(c, ',' | '.')).to_ascii_lowercase();
        if is_duration(&l) {
            return Some(l);
        }
        if l.chars().all(|c| c.is_ascii_digit()) && !l.is_empty() && i + 1 < tokens.len() {
            let unit = tokens[i + 1].trim_matches(',').to_ascii_lowercase();
            let suffix = match unit.as_str() {
                "m" | "min" | "mins" | "minute" | "minutes" => Some("m"),
                "h" | "hr" | "hrs" | "hour" | "hours" => Some("h"),
                "d" | "day" | "days" => Some("d"),
                "w" | "wk" | "week" | "weeks" => Some("w"),
                _ => None,
            };
            if let Some(sfx) = suffix {
                return Some(format!("{l}{sfx}"));
            }
        }
    }
    None
}

pub fn is_duration(s: &str) -> bool {
    let mut chars = s.chars();
    let Some(last) = chars.next_back() else {
        return false;
    };
    if !matches!(last, 'm' | 'h' | 'd' | 'w') {
        return false;
    }
    let prefix = chars.as_str();
    !prefix.is_empty() && prefix.bytes().all(|b| b.is_ascii_digit())
}

pub fn find_domain(tokens: &[String]) -> Option<String> {
    for t in tokens {
        let c = t.trim_matches(|ch: char| matches!(ch, ',' | '.'));
        if c.contains('.')
            && !c.contains('/')
            && c.parse::<std::net::IpAddr>().is_err()
            && c.split('.').all(|p| !p.is_empty() && p.chars().all(|ch| ch.is_ascii_alphanumeric() || ch == '-'))
            && c.split('.').next_back().is_some_and(|tld| tld.len() >= 2 && tld.chars().all(|ch| ch.is_ascii_alphabetic()))
        {
            return Some(c.to_string());
        }
    }
    None
}

pub fn number(token: &str) -> Option<u64> {
    // 8192, 32k (32768), or a small count in words.
    const WORDS: [&str; 10] = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"];
    let t = token.to_ascii_lowercase();
    let (digits, scale) = t.strip_suffix('k').map_or((t.as_str(), 1), |d| (d, 1024));
    if !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()) {
        return digits.parse::<u64>().ok().and_then(|n| n.checked_mul(scale));
    }
    WORDS.iter().position(|w| *w == t).map(|i| i as u64 + 1)
}

pub fn find_named_token(text: &str, tokens: &[String], cues: &[&str]) -> Option<String> {
    if let Some(q) = find_quoted(text) {
        return Some(q);
    }
    for (i, t) in tokens.iter().enumerate() {
        let l = t.to_ascii_lowercase();
        if cues.contains(&l.as_str()) && i + 1 < tokens.len() {
            let next = tokens[i + 1].trim_matches(|c: char| matches!(c, ',' | '.'));
            if !next.starts_with('-')
                && number(next).is_none()
                && !cues.contains(&next.to_ascii_lowercase().as_str())
                && !matches!(
                    next.to_ascii_lowercase().as_str(),
                    "key" | "keys" | "token" | "tokens" | "api" | "a" | "an" | "at" | "to" | "of" | "on" | "new" | "my" | "the" | "for" | "with" | "alias" | "model"
                )
            {
                return Some(next.to_string());
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metachar_and_duration_survive_unicode() {
        let ellipsis = "\u{2026}";
        let grin = "\u{1F600}";
        assert!(find_shell_metachar(ellipsis).is_none());
        assert!(find_shell_metachar(&format!("show status {ellipsis} please")).is_none());
        assert!(find_shell_metachar("a && b").is_some());
        assert!(find_shell_metachar("echo \"a | b\"").is_none());
        assert!(find_shell_metachar("echo 'a;b'").is_none());
        assert!(!is_duration(&format!("{grin}m")));
        assert!(!is_duration("m"));
        assert!(is_duration("2h"));
        assert!(is_duration("15m"));
        let _ = split_words("load \u{201C}fancy\u{201D} model");
        let _ = find_quoted("say \"hello \u{4E16}\u{754C}\"");
        let _ = find_gpus("on gpu0 and GPU 1 with \u{65E5}\u{672C}\u{8A9E}");
        let _ = find_duration(&split_words(&format!("share for 2 hours {ellipsis}")));
        let _ = find_hf_or_path(&split_words("load ./\u{6A21}\u{578B}/foo"));
    }

    #[test]
    fn a_quote_opens_at_a_word_start_and_closes_at_a_word_end() {
        // An apostrophe inside a word ("let's") is no quote, and an opener that never closes is none either.
        assert_eq!(split_parts("let's load qwen and share it"), ["let's load qwen", "share it"]);
        assert_eq!(split_words("name it 'demo key', don't wait"), ["name", "it", "demo key,", "don't", "wait"]);
        assert_eq!(find_quoted("let's call it 'demo key'").as_deref(), Some("demo key"));
        assert_eq!(find_shell_metachar("let's go; rm x"), Some(";"));
        assert_eq!(find_shell_metachar("call it 'a; b'"), None);
        assert_eq!(split_words("'open never closes"), ["'open", "never", "closes"]);
        assert_eq!(split_words("say 'it's fine' ok"), ["say", "it's fine", "ok"]);
        assert_eq!(split_parts("load qwen but don't share it"), ["load qwen", "don't share it"]);
    }

    #[test]
    fn a_request_splits_at_and_or_then_outside_quotes() {
        assert_eq!(split_parts("install x, and then load it"), ["install x", "load it"]);
        assert_eq!(split_parts("run m \"salt and pepper\" then show status"), ["run m \"salt and pepper\"", "show status"]);
        assert_eq!(split_parts("allow addresses or ranges"), ["allow addresses or ranges"]);
        assert_eq!(split_parts("and"), Vec::<String>::new());
        assert_eq!(split_parts("show status, show gpus and also list models"), ["show status", "show gpus", "list models"]);
        assert_eq!(split_parts("allow 1.2.3.4,5.6.7.8 then show status"), ["allow 1.2.3.4,5.6.7.8", "show status"]);
        assert_eq!(split_parts("run m \"a, b\" and stop"), ["run m \"a, b\"", "stop"]);
    }

    #[test]
    fn odd_inputs_do_not_panic() {
        let long = "a".repeat(10_000);
        let cases = [
            "".to_string(),
            "\u{2026}".to_string(),
            "\u{2014}".to_string(),
            "\u{1F525}\u{1F525}\u{1F525}".to_string(),
            long,
            "load mistral | rm -rf /".into(),
            "status && down".into(),
            "\"unclosed".into(),
            "'still open".into(),
            format!("share {}\u{1F600}h", ""),
            "\u{0000}null".into(),
            "gpu\u{0301}0".into(),
        ];
        for s in &cases {
            let _ = find_shell_metachar(s);
            let _ = is_obvious_ood(s);
            let toks = split_words(s);
            let _ = find_quoted(s);
            let _ = find_hf_or_path(&toks);
            let _ = find_path_token(&toks);
            let _ = find_gpus(s);
            let _ = find_cidrs(&toks);
            let _ = find_duration(&toks);
            let _ = find_domain(&toks);
            let _ = find_named_token(s, &toks, &["named", "called"]);
            for t in &toks {
                let _ = is_duration(t);
                let _ = is_gpu_tok(t);
                let _ = is_cidr(t);
            }
        }
    }
}
