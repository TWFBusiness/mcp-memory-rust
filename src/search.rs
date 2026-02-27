use rusqlite::Connection;

use crate::embedding::bytes_to_f32;

/// Resultado de busca
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: String,
    pub mem_type: String,
    pub content: String,
    pub tags: String,
    pub created_at: String,
    pub relevance: f64,
    pub method: String,
}

/// Cosine similarity entre dois vetores
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for i in 0..a.len() {
        let ai = a[i] as f64;
        let bi = b[i] as f64;
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-8 {
        return 0.0;
    }
    dot / denom
}

/// Temporal decay: 1/(1+log1p(days)) com strength 0.15
/// score * (1.0 - DECAY_STRENGTH + DECAY_STRENGTH * recency)
pub fn apply_temporal_decay(score: f64, created_at: &str) -> f64 {
    const DECAY_STRENGTH: f64 = 0.15;

    let days_old = parse_days_old(created_at);
    let recency = 1.0 / (1.0 + (days_old as f64).ln_1p());
    score * (1.0 - DECAY_STRENGTH + DECAY_STRENGTH * recency)
}

fn parse_days_old(created_at: &str) -> i64 {
    // Tenta parsear datetime ISO
    if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(created_at, "%Y-%m-%d %H:%M:%S") {
        let now = chrono::Utc::now().naive_utc();
        return (now - dt).num_days().max(0);
    }
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(created_at) {
        let now = chrono::Utc::now();
        return (now - dt.with_timezone(&chrono::Utc))
            .num_days()
            .max(0);
    }
    0
}

/// Busca FTS5 com scores BM25 normalizados via sigmoid
pub fn search_fts(conn: &Connection, query: &str, limit: usize) -> Vec<SearchResult> {
    let tokens: Vec<&str> = query.split_whitespace().filter(|t| !t.is_empty()).collect();
    if tokens.is_empty() {
        return vec![];
    }

    let fts_query = tokens
        .iter()
        .map(|t| format!("\"{}\"", t))
        .collect::<Vec<_>>()
        .join(" OR ");

    let sql = "SELECT m.id, m.type, m.content, m.tags, m.created_at, \
               bm25(memories_fts) as bm25_score \
               FROM memories_fts f \
               JOIN memories m ON f.rowid = m.rowid \
               WHERE memories_fts MATCH ?1 \
               ORDER BY bm25_score \
               LIMIT ?2";

    let mut stmt = match conn.prepare(sql) {
        Ok(s) => s,
        Err(_) => return vec![],
    };

    let fetch_limit = (limit * 3) as i64;
    let rows = match stmt.query_map(rusqlite::params![fts_query, fetch_limit], |row| {
        let bm25_raw: f64 = row.get::<_, f64>(5)?.abs();
        let bm25_normalized = bm25_raw / (bm25_raw + 1.0);
        let created_at: String = row.get::<_, Option<String>>(4)?.unwrap_or_default();
        let score = apply_temporal_decay(bm25_normalized, &created_at);

        Ok(SearchResult {
            id: row.get(0)?,
            mem_type: row.get(1)?,
            content: row.get(2)?,
            tags: row.get::<_, Option<String>>(3)?.unwrap_or_default(),
            created_at,
            relevance: score,
            method: "fts".into(),
        })
    }) {
        Ok(r) => r,
        Err(_) => return vec![],
    };

    rows.flatten().collect()
}

/// Busca por embedding: scan linear (memórias + chunks)
pub fn search_embedding(
    conn: &Connection,
    query_embedding: &[f32],
    limit: usize,
) -> Vec<SearchResult> {
    const MIN_SIM: f64 = 0.3;

    let mut results_map: std::collections::HashMap<String, SearchResult> =
        std::collections::HashMap::new();

    // Busca nos embeddings principais
    if let Ok(mut stmt) = conn.prepare(
        "SELECT id, type, content, tags, created_at, embedding FROM memories WHERE embedding IS NOT NULL",
    ) {
        if let Ok(rows) = stmt.query_map([], |row| {
            let id: String = row.get(0)?;
            let mem_type: String = row.get(1)?;
            let content: String = row.get(2)?;
            let tags: String = row.get::<_, Option<String>>(3)?.unwrap_or_default();
            let created_at: String = row.get::<_, Option<String>>(4)?.unwrap_or_default();
            let blob: Vec<u8> = row.get(5)?;
            Ok((id, mem_type, content, tags, created_at, blob))
        }) {
            for r in rows.flatten() {
                let stored = bytes_to_f32(&r.5);
                let sim = cosine_similarity(query_embedding, &stored);
                if sim > MIN_SIM {
                    let score = apply_temporal_decay(sim, &r.4);
                    let entry = results_map.entry(r.0.clone()).or_insert(SearchResult {
                        id: r.0,
                        mem_type: r.1,
                        content: r.2,
                        tags: r.3,
                        created_at: r.4,
                        relevance: score,
                        method: "embedding".into(),
                    });
                    if score > entry.relevance {
                        entry.relevance = score;
                    }
                }
            }
        }
    }

    // Busca nos chunks
    if let Ok(mut stmt) = conn.prepare(
        "SELECT c.memory_id, c.embedding, m.type, m.content, m.tags, m.created_at \
         FROM memory_chunks c JOIN memories m ON c.memory_id = m.id \
         WHERE c.embedding IS NOT NULL",
    ) {
        if let Ok(rows) = stmt.query_map([], |row| {
            let mem_id: String = row.get(0)?;
            let blob: Vec<u8> = row.get(1)?;
            let mem_type: String = row.get(2)?;
            let content: String = row.get(3)?;
            let tags: String = row.get::<_, Option<String>>(4)?.unwrap_or_default();
            let created_at: String = row.get::<_, Option<String>>(5)?.unwrap_or_default();
            Ok((mem_id, blob, mem_type, content, tags, created_at))
        }) {
            for r in rows.flatten() {
                let stored = bytes_to_f32(&r.1);
                let sim = cosine_similarity(query_embedding, &stored);
                if sim > MIN_SIM {
                    let score = apply_temporal_decay(sim, &r.5);
                    let entry = results_map.entry(r.0.clone()).or_insert(SearchResult {
                        id: r.0,
                        mem_type: r.2,
                        content: r.3,
                        tags: r.4,
                        created_at: r.5,
                        relevance: score,
                        method: "embedding-chunk".into(),
                    });
                    if score > entry.relevance {
                        entry.relevance = score;
                    }
                }
            }
        }
    }

    let mut results: Vec<SearchResult> = results_map.into_values().collect();
    results.sort_by(|a, b| b.relevance.partial_cmp(&a.relevance).unwrap());
    results.truncate(limit);
    results
}

/// Busca híbrida: 0.7 embedding + 0.3 BM25
pub fn search_hybrid(
    conn: &Connection,
    query: &str,
    query_embedding: Option<&[f32]>,
    limit: usize,
) -> Vec<SearchResult> {
    const VECTOR_WEIGHT: f64 = 0.7;
    const TEXT_WEIGHT: f64 = 0.3;

    let fts_results = search_fts(conn, query, limit);
    let emb_results = if let Some(emb) = query_embedding {
        search_embedding(conn, emb, limit)
    } else {
        vec![]
    };

    // Merge scores
    let mut score_map: std::collections::HashMap<String, (f64, f64, SearchResult)> =
        std::collections::HashMap::new();

    for r in &fts_results {
        let entry = score_map
            .entry(r.id.clone())
            .or_insert((0.0, 0.0, r.clone()));
        entry.0 = entry.0.max(r.relevance); // fts score
    }

    for r in &emb_results {
        let entry = score_map
            .entry(r.id.clone())
            .or_insert((0.0, 0.0, r.clone()));
        entry.1 = entry.1.max(r.relevance); // emb score
        entry.2 = r.clone(); // prefer embedding data
    }

    let mut merged: Vec<SearchResult> = score_map
        .into_values()
        .map(|(fts_score, emb_score, mut data)| {
            let raw = VECTOR_WEIGHT * emb_score + TEXT_WEIGHT * fts_score;
            let final_score = apply_temporal_decay(raw, &data.created_at);
            data.relevance = (final_score * 10000.0).round() / 10000.0;
            if emb_score > 0.0 && fts_score > 0.0 {
                data.method = "hybrid".into();
            }
            data
        })
        .collect();

    merged.sort_by(|a, b| b.relevance.partial_cmp(&a.relevance).unwrap());
    merged.truncate(limit);
    merged
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 0.001);
    }

    #[test]
    fn test_temporal_decay_recent() {
        // Data recente deve ter score próximo do original
        let now = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S").to_string();
        let decayed = apply_temporal_decay(1.0, &now);
        assert!(decayed > 0.99);
    }

    #[test]
    fn test_temporal_decay_old() {
        let decayed = apply_temporal_decay(1.0, "2020-01-01 00:00:00");
        assert!(decayed < 1.0);
        assert!(decayed > 0.85); // DECAY_STRENGTH=0.15
    }
}
