// Ranking for goethe ask: a WordPiece tokenizer and int8 static embeddings (no runtime dependencies), BM25 on words, and both fused.

use super::schema::CliCatalog;
use std::collections::{HashMap, HashSet};
use std::ops::{AddAssign, MulAssign};
use std::path::{Path, PathBuf};

pub const MODEL_FILE: &str = "ask-model-v1.bin";
const MAGIC: &[u8; 4] = b"GASK";
const FORMAT_VERSION: u32 = 1;

#[derive(Clone, Debug)]
pub struct ScoredIntent {
    pub id: String,
    pub score: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RetrievalMode {
    Hybrid,
    #[cfg(test)]
    LexicalOnly,
    #[cfg(test)]
    DenseOnly,
}

pub trait ShortlistBackend {
    fn rank(&self, query: &str, top_k: usize) -> Vec<ScoredIntent>;
}

pub struct StaticModel {
    vocab: HashMap<String, u32>,
    dim: usize,
    unk_id: u32,
    scales: Vec<f32>,
    weights: Vec<i8>,
}

impl StaticModel {
    pub fn default_path() -> PathBuf {
        if let Ok(p) = std::env::var("GOETHE_ASK_MODEL") {
            if !p.is_empty() {
                return PathBuf::from(p);
            }
        }
        goethe_core::paths::goethe_home().join("ask").join(MODEL_FILE)
    }

    pub fn load_default() -> Option<Self> {
        let path = Self::default_path();
        if !path.is_file() {
            return None;
        }
        Self::from_file(&path).ok()
    }

    pub fn from_file(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::from_bytes(&bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        // GASK v1, little-endian: header, vocabulary, one f32 scale and int8 values per row; a file with an adapter matrix is refused.
        if bytes.len() < 24 || &bytes[0..4] != MAGIC {
            return Err("invalid ask model header".into());
        }
        let ver = read_u32(bytes, 4)?;
        if ver != FORMAT_VERSION {
            return Err(format!("unsupported ask model version {ver}"));
        }
        let vocab_size = read_u32(bytes, 8)? as usize;
        let dim = read_u32(bytes, 12)? as usize;
        let unk_id = read_u32(bytes, 16)?;
        if read_u32(bytes, 20)? != 0 {
            return Err("this ask model carries an adapter, which goethe does not read; run goethe update".into());
        }
        if vocab_size == 0 || dim == 0 || dim > 4096 {
            return Err("invalid ask model dimensions".into());
        }
        let mut pos = 24usize;
        let mut vocab = HashMap::with_capacity(vocab_size);
        for id in 0..vocab_size {
            if pos + 2 > bytes.len() {
                return Err("truncated vocab length".into());
            }
            let len = u16::from_le_bytes([bytes[pos], bytes[pos + 1]]) as usize;
            pos += 2;
            if pos + len > bytes.len() {
                return Err("truncated vocab token".into());
            }
            let tok = std::str::from_utf8(&bytes[pos..pos + len]).map_err(|_| "invalid utf8 in vocab")?;
            vocab.insert(tok.to_string(), id as u32);
            pos += len;
        }
        let scale_bytes = vocab_size.checked_mul(4).ok_or("scale size overflow")?;
        if pos + scale_bytes > bytes.len() {
            return Err("truncated scale table".into());
        }
        let mut scales = Vec::with_capacity(vocab_size);
        for i in 0..vocab_size {
            scales.push(f32::from_le_bytes([
                bytes[pos + i * 4],
                bytes[pos + i * 4 + 1],
                bytes[pos + i * 4 + 2],
                bytes[pos + i * 4 + 3],
            ]));
        }
        pos += scale_bytes;
        let weight_bytes = vocab_size.checked_mul(dim).ok_or("weight size overflow")?;
        if pos + weight_bytes > bytes.len() {
            return Err("truncated embedding matrix".into());
        }
        let weights: Vec<i8> = bytes[pos..pos + weight_bytes].iter().map(|&b| b as i8).collect();
        pos += weight_bytes;
        if pos != bytes.len() {
            return Err("ask model has trailing bytes".into());
        }
        Ok(StaticModel { vocab, dim, unk_id, scales, weights })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        // Basic lowercase WordPiece without [CLS]/[SEP] special tokens (avoids model2vec#335 stale-ID bug).
        let mut ids = Vec::new();
        for word in basic_words(text) {
            if word.len() > 100 {
                ids.push(self.unk_id);
                continue;
            }
            let chars: Vec<char> = word.chars().collect();
            let mut start = 0usize;
            let mut word_ids = Vec::new();
            let mut bad = false;
            while start < chars.len() {
                let mut end = chars.len();
                let mut hit = None;
                while start < end {
                    let slice: String = chars[start..end].iter().collect();
                    let cand = if start > 0 { format!("##{slice}") } else { slice };
                    if let Some(&tid) = self.vocab.get(&cand) {
                        hit = Some(tid);
                        break;
                    }
                    end -= 1;
                }
                if let Some(tid) = hit {
                    word_ids.push(tid);
                    start = end;
                } else {
                    bad = true;
                    break;
                }
            }
            if bad {
                ids.push(self.unk_id);
            } else {
                ids.extend(word_ids);
            }
        }
        ids
    }

    pub fn embed(&self, text: &str) -> Vec<f32> {
        // The mean of the token vectors, unit length.
        let ids = self.tokenize(text);
        let mut acc = vec![0.0f32; self.dim];
        let mut count = 0usize;
        for id in ids {
            let row = id as usize;
            if row >= self.scales.len() {
                continue;
            }
            let scale = self.scales[row];
            let base = row * self.dim;
            let slice = &self.weights[base..base + self.dim];
            for (k, &q) in slice.iter().enumerate() {
                acc[k] += f32::from(q) * scale;
            }
            count += 1;
        }
        if count > 0 {
            let inv = 1.0 / count as f32;
            acc.iter_mut().for_each(|v| v.mul_assign(inv));
        }
        l2_normalize(&mut acc);
        acc
    }
}

pub struct HybridRetriever {
    ids: Vec<String>,
    cmd_tokens: Vec<Vec<String>>,
    exact_phrases: Vec<Vec<String>>,
    doc_tf: Vec<HashMap<String, f32>>,
    doc_len: Vec<f32>,
    avg_len: f32,
    idf: HashMap<String, f32>,
    model: Option<StaticModel>,
    card_vecs: Vec<Vec<f32>>,
    mode: RetrievalMode,
}

impl HybridRetriever {
    pub fn has_dense(&self) -> bool {
        self.model.is_some()
    }

    pub fn dense_dim(&self) -> usize {
        self.model.as_ref().map(StaticModel::dim).unwrap_or(0)
    }

    pub fn build(catalog: &CliCatalog, model: Option<StaticModel>) -> Self {
        Self::build_with_mode(catalog, model, RetrievalMode::Hybrid)
    }

    pub fn build_with_mode(catalog: &CliCatalog, model: Option<StaticModel>, mode: RetrievalMode) -> Self {
        // Index all live clap intent cards with BM25 statistics and optional static embedding vectors.
        let n = catalog.intents.len().max(1) as f32;
        let mut df: HashMap<String, usize> = HashMap::new();
        let mut doc_tf = Vec::with_capacity(catalog.intents.len());
        let mut doc_len = Vec::with_capacity(catalog.intents.len());
        let mut ids = Vec::with_capacity(catalog.intents.len());
        let mut cmd_tokens = Vec::with_capacity(catalog.intents.len());
        let mut exact_phrases = Vec::with_capacity(catalog.intents.len());
        for card in &catalog.intents {
            ids.push(card.id.clone());
            cmd_tokens.push(card.id.split('.').map(|s| s.to_ascii_lowercase()).collect());
            let mut phrases = vec![card.id.replace('.', " ")];
            if card.argv_prefix.len() > 1 {
                phrases.push(card.argv_prefix[1..].join(" "));
            }
            exact_phrases.push(phrases);
            let toks = lex_tokens(&card.search_text);
            doc_len.push(toks.len().max(1) as f32);
            let mut tf: HashMap<String, f32> = HashMap::new();
            let mut seen = HashSet::new();
            for t in toks {
                tf.entry(t.clone()).or_insert(0.0).add_assign(1.0);
                if seen.insert(t.clone()) {
                    df.entry(t).or_insert(0).add_assign(1);
                }
            }
            doc_tf.push(tf);
        }
        let avg_len = (doc_len.iter().sum::<f32>() / n).max(1.0);
        let idf: HashMap<String, f32> = df
            .into_iter()
            .map(|(k, c)| (k, ((n - c as f32 + 0.5) / (c as f32 + 0.5) + 1.0).ln()))
            .collect();
        let card_vecs = match &model {
            Some(m) => catalog.intents.iter().map(|c| m.embed(&c.embed_text)).collect(),
            None => Vec::new(),
        };
        HybridRetriever { ids, cmd_tokens, exact_phrases, doc_tf, doc_len, avg_len, idf, model, card_vecs, mode }
    }

    fn bm25_score(&self, q_lower: &str, q_toks: &[String], idx: usize) -> f32 {
        // BM25 scaled to 0..1; naming all of a command's words, or its whole name, lifts the score to a floor.
        if q_toks.is_empty() {
            return 0.0;
        }
        let tf_map = &self.doc_tf[idx];
        let dl = self.doc_len[idx];
        let k1 = 1.2f32;
        let b = 0.65f32;
        let mut raw = 0.0f32;
        let mut max_possible = 0.0f32;
        for qt in q_toks {
            let idf = self.idf.get(qt).copied().unwrap_or(1.5);
            max_possible += idf * (k1 + 1.0);
            if let Some(&tf) = tf_map.get(qt) {
                raw += idf * (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * dl / self.avg_len));
            } else if qt.len() >= 4 && !qt.chars().any(|c| c.is_ascii_digit()) {
                let partial = tf_map.iter().filter(|(k, _)| k.starts_with(qt.as_str()) || qt.starts_with(k.as_str()));
                if let Some(tf) = partial.map(|(_, &tf)| tf).reduce(f32::max) {
                    raw += 0.65 * idf * (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * dl / self.avg_len));
                }
            }
        }
        let mut norm = if max_possible > 0.0 { (raw / max_possible).clamp(0.0, 1.0) } else { 0.0 };
        let cmd = &self.cmd_tokens[idx];
        let hits = cmd.iter().filter(|ct| q_toks.iter().any(|qt| qt == *ct)).count();
        if hits == cmd.len() && hits > 0 {
            let cov = (hits as f32 / q_toks.len().max(1) as f32).clamp(0.15, 1.0);
            if hits >= 2 {
                norm = norm.max((0.68 + 0.22 * cov + 0.04 * (hits.min(3) as f32)).min(0.98));
            } else if cmd.first().is_some_and(|ct| q_toks.first() == Some(ct)) {
                norm = norm.max(0.76 + 0.16 * cov);
            }
        }
        for p in &self.exact_phrases[idx] {
            if p == q_lower {
                return 1.0;
            }
            if p.len() >= 5 && contains_phrase(q_lower, p) {
                let coverage = (p.len() as f32 / q_lower.len().max(1) as f32).clamp(0.2, 1.0);
                norm = norm.max(0.72 + 0.24 * coverage);
            }
        }
        norm.clamp(0.0, 1.0)
    }
}

impl ShortlistBackend for HybridRetriever {
    fn rank(&self, query: &str, top_k: usize) -> Vec<ScoredIntent> {
        // BM25 and embedding similarity, fused by reciprocal rank and blended with both scores.
        let head = query.split_once(" -- ").map(|(l, _)| l).unwrap_or(query);
        let q_lower = head.trim().to_ascii_lowercase();
        let q_toks = lex_tokens(&q_lower);
        let n = self.ids.len();
        let lex_scores: Vec<f32> = (0..n).map(|i| self.bm25_score(&q_lower, &q_toks, i)).collect();
        let dense_scores: Vec<f32> = match &self.model {
            Some(m) => {
                let qv = m.embed(&q_lower);
                self.card_vecs.iter().map(|cv| dot(&qv, cv).clamp(0.0, 1.0)).collect()
            }
            None => vec![0.0; n],
        };
        let lex_rank = rank_order(&lex_scores);
        let dense_rank = if self.model.is_some() { rank_order(&dense_scores) } else { vec![n; n] };
        let mut out: Vec<ScoredIntent> = (0..n)
            .map(|i| {
                let lex = lex_scores[i];
                let den = dense_scores[i];
                let score = match self.mode {
                    #[cfg(test)]
                    RetrievalMode::LexicalOnly => lex,
                    #[cfg(test)]
                    RetrievalMode::DenseOnly => den,
                    RetrievalMode::Hybrid if self.model.is_some() => {
                        let rrf_k = 12.0f32;
                        let rrf_raw = 1.0 / (rrf_k + lex_rank[i] as f32) + 1.0 / (rrf_k + dense_rank[i] as f32);
                        let rrf_norm = (rrf_raw / (2.0 / (rrf_k + 1.0))).clamp(0.0, 1.0);
                        let blend = 0.55 * lex + 0.45 * den;
                        (0.78 * blend + 0.22 * rrf_norm * (lex.max(den))).clamp(0.0, 1.0)
                    }
                    RetrievalMode::Hybrid => lex,
                };
                ScoredIntent { id: self.ids[i].clone(), score }
            })
            .collect();
        out.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.id.cmp(&b.id)));
        out.truncate(top_k);
        out
    }

}

fn rank_order(scores: &[f32]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
    let mut ranks = vec![scores.len(); scores.len()];
    for (r, idx) in order.into_iter().enumerate() {
        ranks[idx] = r + 1;
    }
    ranks
}
fn basic_words(text: &str) -> Vec<String> {
    // Split on whitespace and ASCII punctuation, keeping each punctuation mark separate like BERT BasicTokenizer.
    let mut out = Vec::new();
    let mut cur = String::new();
    let flush = |cur: &mut String, out: &mut Vec<String>| {
        if !cur.is_empty() {
            out.push(std::mem::take(cur));
        }
    };
    for ch in text.to_lowercase().chars() {
        if ch.is_whitespace() {
            flush(&mut cur, &mut out);
        } else if ch.is_ascii_punctuation() {
            flush(&mut cur, &mut out);
            out.push(ch.to_string());
        } else {
            cur.push(ch);
        }
    }
    flush(&mut cur, &mut out);
    out
}
pub fn lex_tokens(text: &str) -> Vec<String> {
    // Normalized lexical tokens for BM25, dropping tiny stop words and paths/URLs.
    const STOP: &[&str] = &["the", "and", "for", "with", "that", "this", "from", "into", "your", "please", "can", "you"];
    text.split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .map(|s| s.to_ascii_lowercase())
        .filter(|s| s.len() >= 2 && !STOP.contains(&s.as_str()))
        .collect()
}
fn contains_phrase(hay: &str, needle: &str) -> bool {
    // Whole-word substring match so short phrases never fire inside unrelated tokens.
    if needle.is_empty() {
        return false;
    }
    let mut start = 0;
    while start <= hay.len() {
        let Some(pos) = hay[start..].find(needle) else {
            break;
        };
        let abs = start + pos;
        if !hay.is_char_boundary(abs) || !hay.is_char_boundary(abs + needle.len()) {
            break;
        }
        let before = hay[..abs].chars().next_back().is_none_or(|c| !c.is_ascii_alphanumeric());
        let after = hay[abs + needle.len()..].chars().next().is_none_or(|c| !c.is_ascii_alphanumeric());
        if before && after {
            return true;
        }
        start = abs + needle.len().max(1);
    }
    false
}
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}
fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 {
        let inv = 1.0 / norm;
        v.iter_mut().for_each(|x| x.mul_assign(inv));
    }
}
fn read_u32(bytes: &[u8], off: usize) -> Result<u32, String> {
    if off + 4 > bytes.len() {
        return Err("truncated header".into());
    }
    Ok(u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]))
}
