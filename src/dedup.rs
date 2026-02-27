use std::collections::HashSet;
use rusqlite::Connection;

/// Similaridade Jaccard por palavras
pub fn jaccard_sim(text_a: &str, text_b: &str) -> f64 {
    let a_lower = text_a.to_lowercase();
    let b_lower = text_b.to_lowercase();
    let words_a: HashSet<&str> = a_lower.split_whitespace().collect();
    let words_b: HashSet<&str> = b_lower.split_whitespace().collect();
    if words_a.is_empty() || words_b.is_empty() {
        return 0.0;
    }
    let intersection = words_a.intersection(&words_b).count();
    let union = words_a.union(&words_b).count();
    intersection as f64 / union as f64
}

/// Verifica se memória similar já existe. Retorna ID existente ou None.
/// Passo 1: exact match por content+type
/// Passo 2: FTS rough match + Jaccard refinement
pub fn find_duplicate(
    conn: &Connection,
    content: &str,
    mem_type: &str,
    threshold: f64,
) -> Option<String> {
    // Passo 1: exact match
    let mut stmt = conn
        .prepare("SELECT id FROM memories WHERE type = ? AND content = ?")
        .ok()?;
    if let Ok(id) = stmt.query_row(rusqlite::params![mem_type, content], |row| {
        row.get::<_, String>(0)
    }) {
        return Some(id);
    }

    // Passo 2: FTS rough + Jaccard
    let tokens: Vec<&str> = content.split_whitespace().take(20).collect();
    let fts_terms: Vec<&str> = tokens.into_iter().filter(|t| t.len() > 2).collect();
    if fts_terms.is_empty() {
        return None;
    }

    let fts_query = fts_terms
        .iter()
        .map(|t| format!("\"{}\"", t))
        .collect::<Vec<_>>()
        .join(" OR ");

    let sql = "SELECT m.id, m.content FROM memories_fts f \
               JOIN memories m ON f.rowid = m.rowid \
               WHERE m.type = ? AND memories_fts MATCH ? LIMIT 10";

    let mut stmt = conn.prepare(sql).ok()?;
    let rows: Vec<(String, String)> = stmt
        .query_map(rusqlite::params![mem_type, fts_query], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })
        .ok()?
        .filter_map(|r| r.ok())
        .collect();

    for (id, existing_content) in rows {
        if jaccard_sim(content, &existing_content) >= threshold {
            return Some(id);
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jaccard_identical() {
        assert!((jaccard_sim("hello world", "hello world") - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_jaccard_different() {
        assert!(jaccard_sim("hello world", "foo bar") < 0.01);
    }

    #[test]
    fn test_jaccard_partial() {
        let sim = jaccard_sim("hello world foo", "hello world bar");
        // intersection=2 (hello,world), union=4 (hello,world,foo,bar) => 0.5
        assert!((sim - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_jaccard_empty() {
        assert_eq!(jaccard_sim("", "hello"), 0.0);
        assert_eq!(jaccard_sim("hello", ""), 0.0);
    }
}
