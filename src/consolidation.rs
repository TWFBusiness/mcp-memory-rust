/// Consolidação automática de memórias:
/// - Agrupa sessões conversation do mesmo projeto em resumos
/// - Merge memórias similares não-conversation
/// - Arquiva sessões originais após consolidação
use std::collections::HashMap;
use rusqlite::Connection;
use crate::dedup::jaccard_sim;
use crate::storage;

#[derive(Debug, Default)]
pub struct ConsolidationResult {
    pub conversations_consolidated: usize,
    pub similar_merged: usize,
    pub archived: usize,
}

/// Consolida memórias de conversation por projeto.
/// Se >= 5 sessões do mesmo projeto, gera resumo consolidado.
pub fn consolidate_conversations(conn: &Connection) -> usize {
    let mut consolidated = 0usize;

    // Buscar projetos com >= 5 conversations não-arquivadas
    let mut stmt = match conn.prepare(
        "SELECT tags, COUNT(*) as cnt FROM memories \
         WHERE type = 'conversation' AND archived = 0 \
         GROUP BY tags HAVING cnt >= 5"
    ) {
        Ok(s) => s,
        Err(_) => return 0,
    };

    let projects: Vec<String> = stmt
        .query_map([], |row| row.get::<_, String>(0))
        .ok()
        .map(|r| r.flatten().collect())
        .unwrap_or_default();

    for project_tags in projects {
        // Extrair project name das tags
        let project_name = project_tags
            .split(',')
            .find(|t| !["conversation", "claude-code", "auto-saved", "consolidated"].contains(t))
            .unwrap_or("unknown")
            .trim();

        // Buscar todas as sessões deste projeto
        let mut fetch_stmt = match conn.prepare(
            "SELECT id, content, created_at FROM memories \
             WHERE type = 'conversation' AND archived = 0 AND tags = ? \
             ORDER BY created_at ASC"
        ) {
            Ok(s) => s,
            Err(_) => continue,
        };

        let sessions: Vec<(String, String, String)> = fetch_stmt
            .query_map(rusqlite::params![project_tags], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get::<_, Option<String>>(2)?.unwrap_or_default()))
            })
            .ok()
            .map(|r| r.flatten().collect())
            .unwrap_or_default();

        if sessions.len() < 5 {
            continue;
        }

        // Gerar resumo consolidado
        let mut all_topics = Vec::new();
        let mut all_tools: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut all_files: std::collections::HashSet<String> = std::collections::HashSet::new();
        let session_ids: Vec<String> = sessions.iter().map(|s| s.0.clone()).collect();

        for (_, content, _) in &sessions {
            for line in content.lines() {
                if line.starts_with("Tools: ") {
                    for tool in line[7..].split(", ") {
                        all_tools.insert(tool.trim().to_string());
                    }
                } else if line.starts_with("Files: ") {
                    for file in line[7..].split(", ") {
                        all_files.insert(file.trim().to_string());
                    }
                } else if line.starts_with("  - ") {
                    let topic = line[4..].trim();
                    if !topic.is_empty() && all_topics.len() < 30 {
                        all_topics.push(topic.to_string());
                    }
                }
            }
        }

        // Deduplicar tópicos por similaridade
        let unique_topics = dedup_topics(&all_topics, 0.7);

        let summary = format!(
            "[{}] Consolidated ({} sessions)\nTools: {}\nFiles: {}\nTopics:\n{}",
            project_name,
            sessions.len(),
            all_tools.into_iter().take(30).collect::<Vec<_>>().join(", "),
            all_files.into_iter().take(20).collect::<Vec<_>>().join(", "),
            unique_topics.iter().take(20).map(|t| format!("  - {}", t)).collect::<Vec<_>>().join("\n"),
        );

        let auto_tags = crate::autotag::extract_tags(&summary);
        let base_tags = format!("consolidated,conversation,{}", project_name);
        let final_tags = crate::autotag::merge_tags(&base_tags, &auto_tags);

        // Salvar resumo consolidado
        let consolidated_id = storage::generate_id(&summary, "consolidated");
        let _ = conn.execute(
            "INSERT OR REPLACE INTO memories (id, type, content, tags, updated_at, importance) \
             VALUES (?, 'consolidated', ?, ?, datetime('now'), 0.7)",
            rusqlite::params![consolidated_id, summary, final_tags],
        );

        // Arquivar sessões originais e criar edges supersedes
        for session_id in &session_ids {
            let _ = conn.execute(
                "UPDATE memories SET archived = 1 WHERE id = ?",
                rusqlite::params![session_id],
            );
            storage::create_edge(conn, &consolidated_id, session_id, "supersedes");
        }

        consolidated += 1;
    }

    consolidated
}

/// Consolida memórias similares (não-conversation) com Jaccard 0.6-0.84.
/// Mantém a mais recente, cria edge supersedes.
pub fn consolidate_similar(conn: &Connection) -> usize {
    let mut merged = 0usize;

    // Buscar memórias não-conversation, não-archived, não-consolidated
    let mut stmt = match conn.prepare(
        "SELECT id, type, content, updated_at FROM memories \
         WHERE type NOT IN ('conversation', 'consolidated') AND archived = 0 \
         ORDER BY updated_at DESC"
    ) {
        Ok(s) => s,
        Err(_) => return 0,
    };

    let memories: Vec<(String, String, String, String)> = stmt
        .query_map([], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get::<_, Option<String>>(3)?.unwrap_or_default()))
        })
        .ok()
        .map(|r| r.flatten().collect())
        .unwrap_or_default();

    // Agrupar por tipo
    let mut by_type: HashMap<String, Vec<(String, String)>> = HashMap::new();
    for (id, mem_type, content, _) in &memories {
        by_type.entry(mem_type.clone()).or_default().push((id.clone(), content.clone()));
    }

    let mut archived_ids: std::collections::HashSet<String> = std::collections::HashSet::new();

    for (_mem_type, items) in &by_type {
        for i in 0..items.len() {
            if archived_ids.contains(&items[i].0) {
                continue;
            }
            for j in (i + 1)..items.len() {
                if archived_ids.contains(&items[j].0) {
                    continue;
                }
                let sim = jaccard_sim(&items[i].1, &items[j].1);
                if sim >= 0.6 && sim < 0.85 {
                    // i é mais recente (sorted DESC), criar edge supersedes
                    storage::create_edge(conn, &items[i].0, &items[j].0, "supersedes");

                    // Merge content: append unique info do mais antigo ao mais recente
                    let merged_content = format!("{}\n\n[Merged from {}]: {}",
                        items[i].1,
                        items[j].0,
                        &items[j].1.chars().take(300).collect::<String>()
                    );
                    let _ = conn.execute(
                        "UPDATE memories SET content = ?, updated_at = datetime('now') WHERE id = ?",
                        rusqlite::params![merged_content, items[i].0],
                    );

                    // Arquivar o mais antigo
                    let _ = conn.execute(
                        "UPDATE memories SET archived = 1 WHERE id = ?",
                        rusqlite::params![items[j].0],
                    );
                    archived_ids.insert(items[j].0.clone());
                    merged += 1;
                }
            }
        }
    }

    merged
}

/// Executa consolidação completa
pub fn run_consolidation(conn: &Connection) -> ConsolidationResult {
    let conversations_consolidated = consolidate_conversations(conn);
    let similar_merged = consolidate_similar(conn);

    ConsolidationResult {
        conversations_consolidated,
        similar_merged,
        archived: conversations_consolidated + similar_merged,
    }
}

/// Deduplicar tópicos por similaridade Jaccard
fn dedup_topics(topics: &[String], threshold: f64) -> Vec<String> {
    let mut unique = Vec::new();
    for topic in topics {
        let is_dup = unique.iter().any(|existing: &String| {
            jaccard_sim(topic, existing) >= threshold
        });
        if !is_dup {
            unique.push(topic.clone());
        }
    }
    unique
}
