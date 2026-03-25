use std::path::{Path, PathBuf};
use anyhow::Result;
use rusqlite::Connection;
use sha2::{Sha256, Digest};

/// Diretórios e paths dos DBs
pub struct MemoryPaths {
    pub global_db: PathBuf,
    pub personality_db: PathBuf,
    pub data_dir: PathBuf,
}

impl MemoryPaths {
    pub fn new() -> Result<Self> {
        let home = dirs::home_dir().ok_or_else(|| anyhow::anyhow!("home dir not found"))?;
        let data_dir = home.join(".mcp-memoria").join("data");
        Ok(Self {
            global_db: data_dir.join("global.db"),
            personality_db: data_dir.join("personality.db"),
            data_dir,
        })
    }

    pub fn project_db_path() -> Option<PathBuf> {
        let cwd = std::env::var("MCP_PROJECT_DIR")
            .or_else(|_| std::env::var("CLAUDE_CWD"))
            .ok()
            .or_else(|| {
                std::env::current_dir()
                    .ok()
                    .map(|p| p.to_string_lossy().to_string())
            })?;
        Some(Path::new(&cwd).join(".mcp-memoria").join("project.db"))
    }
}

/// Resolve scope para lista de (nome, path)
pub fn resolve_scope_dbs(scope: &str, paths: &MemoryPaths) -> Vec<(String, PathBuf)> {
    match scope {
        "global" => vec![("global".into(), paths.global_db.clone())],
        "project" => {
            if let Some(p) = MemoryPaths::project_db_path() {
                vec![("project".into(), p)]
            } else {
                vec![]
            }
        }
        "personality" => vec![("personality".into(), paths.personality_db.clone())],
        "both" => {
            let mut dbs = vec![("global".into(), paths.global_db.clone())];
            if let Some(p) = MemoryPaths::project_db_path() {
                dbs.push(("project".into(), p));
            }
            dbs
        }
        "all" => {
            let mut dbs = vec![
                ("global".into(), paths.global_db.clone()),
                ("personality".into(), paths.personality_db.clone()),
            ];
            if let Some(p) = MemoryPaths::project_db_path() {
                dbs.push(("project".into(), p));
            }
            dbs
        }
        _ => vec![],
    }
}

/// Inicializa SQLite com schema v2 (inclui access_count, importance, archived, memory_edges)
pub fn init_db(db_path: &Path) -> Result<Connection> {
    if let Some(parent) = db_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let conn = Connection::open(db_path)?;
    conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")?;

    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            content TEXT NOT NULL,
            tags TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            embedding BLOB,
            access_count INTEGER DEFAULT 0,
            importance FLOAT DEFAULT 0.5,
            archived INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS memory_chunks (
            id TEXT PRIMARY KEY,
            memory_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            embedding BLOB,
            FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS embedding_cache (
            text_hash TEXT NOT NULL,
            model TEXT NOT NULL,
            embedding BLOB NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (text_hash, model)
        );

        CREATE TABLE IF NOT EXISTS memory_edges (
            from_id TEXT NOT NULL,
            to_id TEXT NOT NULL,
            relation TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (from_id, to_id, relation),
            FOREIGN KEY (from_id) REFERENCES memories(id) ON DELETE CASCADE,
            FOREIGN KEY (to_id) REFERENCES memories(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_type ON memories(type);
        CREATE INDEX IF NOT EXISTS idx_created ON memories(created_at);
        CREATE INDEX IF NOT EXISTS idx_chunks_memory ON memory_chunks(memory_id);
        CREATE INDEX IF NOT EXISTS idx_edges_from ON memory_edges(from_id);
        CREATE INDEX IF NOT EXISTS idx_edges_to ON memory_edges(to_id);",
    )?;

    // Migrate existing DBs: add columns if missing
    migrate_add_column(&conn, "memories", "access_count", "INTEGER DEFAULT 0");
    migrate_add_column(&conn, "memories", "importance", "FLOAT DEFAULT 0.5");
    migrate_add_column(&conn, "memories", "archived", "INTEGER DEFAULT 0");

    // Index on archived (after migration ensures column exists)
    let _ = conn.execute_batch("CREATE INDEX IF NOT EXISTS idx_archived ON memories(archived);");

    // Backfill importance by type (only for default 0.5 values from migration)
    backfill_importance(&conn);

    // FTS5
    conn.execute_batch(
        "CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
            content, tags, content='memories', content_rowid='rowid'
        );

        CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
            INSERT INTO memories_fts(rowid, content, tags)
            VALUES (NEW.rowid, NEW.content, NEW.tags);
        END;

        CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, content, tags)
            VALUES('delete', OLD.rowid, OLD.content, OLD.tags);
        END;

        CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, content, tags)
            VALUES('delete', OLD.rowid, OLD.content, OLD.tags);
            INSERT INTO memories_fts(rowid, content, tags)
            VALUES (NEW.rowid, NEW.content, NEW.tags);
        END;",
    )?;

    Ok(conn)
}

/// Backfill importance para memórias que ficaram com default 0.5
fn backfill_importance(conn: &Connection) {
    // Só executa se há memórias com importance=0.5 que deveriam ter outro valor
    let needs_backfill: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM memories WHERE importance = 0.5 AND type IN ('pattern','decision','architecture','conversation','note','todo')",
            [],
            |r| r.get(0),
        )
        .unwrap_or(0);

    if needs_backfill == 0 {
        return;
    }

    // Desabilitar triggers temporariamente pra evitar FTS rebuild a cada UPDATE
    let _ = conn.execute_batch("DROP TRIGGER IF EXISTS memories_au;");

    let updates: &[(&str, &[&str])] = &[
        ("0.9", &["pattern"]),
        ("0.8", &["decision", "architecture"]),
        ("0.7", &["solution", "consolidated"]),
        ("0.6", &["implementation", "preference"]),
        ("0.4", &["note", "todo"]),
        ("0.3", &["conversation"]),
    ];

    for (importance, types) in updates {
        let placeholders: Vec<String> = types.iter().map(|_| "?".to_string()).collect();
        let sql = format!(
            "UPDATE memories SET importance = {} WHERE importance = 0.5 AND type IN ({})",
            importance,
            placeholders.join(",")
        );
        let mut stmt = match conn.prepare(&sql) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let params: Vec<Box<dyn rusqlite::types::ToSql>> =
            types.iter().map(|t| Box::new(t.to_string()) as Box<dyn rusqlite::types::ToSql>).collect();
        let param_refs: Vec<&dyn rusqlite::types::ToSql> = params.iter().map(|p| p.as_ref()).collect();
        let _ = stmt.execute(param_refs.as_slice());
    }

    // Recriar trigger
    let _ = conn.execute_batch(
        "CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, content, tags)
            VALUES('delete', OLD.rowid, OLD.content, OLD.tags);
            INSERT INTO memories_fts(rowid, content, tags)
            VALUES (NEW.rowid, NEW.content, NEW.tags);
        END;"
    );
}

/// Migração segura: adiciona coluna se não existir
fn migrate_add_column(conn: &Connection, table: &str, column: &str, col_type: &str) {
    let sql = format!("ALTER TABLE {} ADD COLUMN {} {}", table, column, col_type);
    let _ = conn.execute_batch(&sql); // ignora erro se já existe
}

/// Gera ID único (sha256[:16] de type:content:timestamp)
pub fn generate_id(content: &str, mem_type: &str) -> String {
    let now = chrono::Utc::now().to_rfc3339();
    let input = format!("{}:{}:{}", mem_type, content, now);
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    format!("{:x}", hasher.finalize())[..16].to_string()
}

/// Calcula importance base por tipo
pub fn base_importance(mem_type: &str) -> f64 {
    match mem_type {
        "pattern" => 0.9,
        "decision" | "architecture" => 0.8,
        "solution" => 0.7,
        "implementation" => 0.6,
        "preference" => 0.6,
        "note" | "todo" => 0.4,
        "conversation" => 0.3,
        "consolidated" => 0.7,
        _ => 0.5,
    }
}

/// Salva memória com dedup check, auto-tags e importance
pub fn save_memory(
    conn: &Connection,
    mem_type: &str,
    content: &str,
    tags: &str,
) -> Result<SaveResult> {
    // Auto-tag
    let auto_tags = crate::autotag::extract_tags(content);
    let final_tags = crate::autotag::merge_tags(tags, &auto_tags);
    let importance = base_importance(mem_type);

    // Dedup check
    if mem_type != "conversation" {
        if let Some(existing_id) =
            crate::dedup::find_duplicate(conn, content, mem_type, 0.85)
        {
            conn.execute(
                "UPDATE memories SET content = ?, tags = ?, updated_at = datetime('now'), \
                 importance = MAX(importance, ?) WHERE id = ?",
                rusqlite::params![content, final_tags, importance, existing_id],
            )?;
            return Ok(SaveResult {
                id: existing_id,
                dedup: "updated".into(),
            });
        }

        // Se há similar com Jaccard 0.5-0.84, criar edge relates_to
        if let Some(related_id) =
            crate::dedup::find_duplicate(conn, content, mem_type, 0.5)
        {
            // Será linkado depois do insert
            let mem_id = generate_id(content, mem_type);
            conn.execute(
                "INSERT OR REPLACE INTO memories (id, type, content, tags, updated_at, importance) \
                 VALUES (?, ?, ?, ?, datetime('now'), ?)",
                rusqlite::params![mem_id, mem_type, content, final_tags, importance],
            )?;
            let _ = create_edge(conn, &mem_id, &related_id, "relates_to");
            return Ok(SaveResult {
                id: mem_id,
                dedup: "new".into(),
            });
        }
    }

    let mem_id = generate_id(content, mem_type);
    conn.execute(
        "INSERT OR REPLACE INTO memories (id, type, content, tags, updated_at, importance) \
         VALUES (?, ?, ?, ?, datetime('now'), ?)",
        rusqlite::params![mem_id, mem_type, content, final_tags, importance],
    )?;

    Ok(SaveResult {
        id: mem_id,
        dedup: "new".into(),
    })
}

pub struct SaveResult {
    pub id: String,
    pub dedup: String,
}

/// Cria edge entre duas memórias
pub fn create_edge(conn: &Connection, from_id: &str, to_id: &str, relation: &str) -> Result<bool> {
    let inserted = conn.execute(
        "INSERT OR IGNORE INTO memory_edges (from_id, to_id, relation) VALUES (?, ?, ?)",
        rusqlite::params![from_id, to_id, relation],
    )?;
    Ok(inserted > 0)
}

/// Incrementa access_count e atualiza importance
pub fn update_access_count(conn: &Connection, id: &str) {
    let _ = conn.execute(
        "UPDATE memories SET access_count = access_count + 1, \
         importance = MIN(1.0, importance + 0.02) \
         WHERE id = ?",
        rusqlite::params![id],
    );
}

/// Busca 1-hop neighbors via edges
pub fn get_edge_neighbors(conn: &Connection, ids: &[String]) -> Vec<String> {
    if ids.is_empty() {
        return vec![];
    }
    let placeholders: Vec<String> = ids.iter().map(|_| "?".to_string()).collect();
    let ph = placeholders.join(",");

    let sql = format!(
        "SELECT DISTINCT CASE WHEN from_id IN ({ph}) THEN to_id ELSE from_id END as neighbor \
         FROM memory_edges \
         WHERE from_id IN ({ph}) OR to_id IN ({ph})"
    );

    let mut stmt = match conn.prepare(&sql) {
        Ok(s) => s,
        Err(_) => return vec![],
    };

    // Bind all params (3x the ids)
    let mut all_params: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
    for _ in 0..3 {
        for id in ids {
            all_params.push(Box::new(id.clone()));
        }
    }
    let param_refs: Vec<&dyn rusqlite::types::ToSql> = all_params.iter().map(|p| p.as_ref()).collect();

    let rows = match stmt.query_map(param_refs.as_slice(), |row| row.get::<_, String>(0)) {
        Ok(r) => r,
        Err(_) => return vec![],
    };

    let existing: std::collections::HashSet<&String> = ids.iter().collect();
    rows.flatten()
        .filter(|n| !existing.contains(n))
        .collect()
}

/// Lista memórias recentes (exclui archived por padrão)
pub fn list_memories(
    conn: &Connection,
    mem_type: Option<&str>,
    limit: i64,
) -> Result<Vec<MemoryRecord>> {
    let mut results = Vec::new();

    if let Some(t) = mem_type {
        let mut stmt = conn.prepare(
            "SELECT id, type, content, tags, created_at FROM memories \
             WHERE type = ? AND archived = 0 ORDER BY updated_at DESC LIMIT ?",
        )?;
        let rows = stmt.query_map(rusqlite::params![t, limit], map_memory_row)?;
        for r in rows {
            results.push(r?);
        }
    } else {
        let mut stmt = conn.prepare(
            "SELECT id, type, content, tags, created_at FROM memories \
             WHERE archived = 0 ORDER BY updated_at DESC LIMIT ?",
        )?;
        let rows = stmt.query_map(rusqlite::params![limit], map_memory_row)?;
        for r in rows {
            results.push(r?);
        }
    }

    Ok(results)
}

fn map_memory_row(row: &rusqlite::Row) -> rusqlite::Result<MemoryRecord> {
    Ok(MemoryRecord {
        id: row.get(0)?,
        mem_type: row.get(1)?,
        content: row.get(2)?,
        tags: row.get::<_, Option<String>>(3)?.unwrap_or_default(),
        created_at: row.get::<_, Option<String>>(4)?.unwrap_or_default(),
    })
}

#[derive(Debug, Clone)]
pub struct MemoryRecord {
    pub id: String,
    pub mem_type: String,
    pub content: String,
    pub tags: String,
    pub created_at: String,
}

/// Estatísticas do DB
pub fn get_stats(conn: &Connection) -> DbStats {
    let total: i64 = conn
        .query_row("SELECT COUNT(*) FROM memories WHERE archived = 0", [], |r| r.get(0))
        .unwrap_or(0);
    let archived: i64 = conn
        .query_row("SELECT COUNT(*) FROM memories WHERE archived = 1", [], |r| r.get(0))
        .unwrap_or(0);
    let indexed: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM memories WHERE embedding IS NOT NULL AND archived = 0",
            [],
            |r| r.get(0),
        )
        .unwrap_or(0);
    let chunks: i64 = conn
        .query_row("SELECT COUNT(*) FROM memory_chunks", [], |r| r.get(0))
        .unwrap_or(0);
    let cache: i64 = conn
        .query_row("SELECT COUNT(*) FROM embedding_cache", [], |r| r.get(0))
        .unwrap_or(0);
    let edges: i64 = conn
        .query_row("SELECT COUNT(*) FROM memory_edges", [], |r| r.get(0))
        .unwrap_or(0);

    let mut by_type = Vec::new();
    if let Ok(mut stmt) = conn.prepare(
        "SELECT type, COUNT(*) FROM memories WHERE archived = 0 GROUP BY type"
    ) {
        if let Ok(mapped) = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
        }) {
            for r in mapped.flatten() {
                by_type.push(r);
            }
        }
    }

    DbStats {
        total,
        archived,
        indexed,
        chunks,
        cache_entries: cache,
        edges,
        by_type,
    }
}

#[derive(Debug)]
pub struct DbStats {
    pub total: i64,
    pub archived: i64,
    pub indexed: i64,
    pub chunks: i64,
    pub cache_entries: i64,
    pub edges: i64,
    pub by_type: Vec<(String, i64)>,
}

/// Reindex: enfileira memórias sem embedding
pub fn get_unindexed_memories(conn: &Connection) -> Result<Vec<(String, String)>> {
    let mut stmt =
        conn.prepare("SELECT id, content FROM memories WHERE embedding IS NULL AND archived = 0")?;
    let rows = stmt.query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?;
    Ok(rows.flatten().collect())
}

/// Compact: VACUUM + rebuild FTS + apply TTL
pub fn compact_db(conn: &Connection, scope: &str) -> Result<CompactResult> {
    let result = CompactResult {
        ttl_applied: apply_ttl(conn, scope),
        decayed: apply_importance_decay(conn),
    };

    // Rebuild FTS
    let _ = conn.execute_batch("INSERT INTO memories_fts(memories_fts) VALUES('rebuild');");
    conn.execute_batch("VACUUM;")?;

    Ok(result)
}

#[derive(Debug, Default)]
pub struct CompactResult {
    pub ttl_applied: i64,
    pub decayed: i64,
}

/// Aplica TTL baseado no scope
pub fn apply_ttl(conn: &Connection, scope: &str) -> i64 {
    let personality_days: i64 = std::env::var("MEMORY_TTL_PERSONALITY_DAYS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(90);

    let project_days: i64 = std::env::var("MEMORY_TTL_PROJECT_DAYS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(30);

    match scope {
        "global" => 0, // sem expiração
        "personality" => {
            // Arquivar conversations antigas (> N dias), manter decisions/patterns
            conn.execute(
                "UPDATE memories SET archived = 1 \
                 WHERE type = 'conversation' AND archived = 0 \
                 AND julianday('now') - julianday(updated_at) > ? \
                 AND access_count = 0",
                rusqlite::params![personality_days],
            ).unwrap_or(0) as i64
        }
        "project" => {
            // Reduzir importance de memórias stale (> N dias sem acesso)
            conn.execute(
                "UPDATE memories SET importance = importance * 0.5 \
                 WHERE archived = 0 AND access_count = 0 \
                 AND julianday('now') - julianday(updated_at) > ? \
                 AND importance > 0.1",
                rusqlite::params![project_days],
            ).unwrap_or(0) as i64
        }
        _ => 0,
    }
}

/// Decay importance para memórias não acessadas > 90 dias
fn apply_importance_decay(conn: &Connection) -> i64 {
    conn.execute(
        "UPDATE memories SET importance = importance * 0.5 \
         WHERE access_count = 0 AND archived = 0 \
         AND julianday('now') - julianday(updated_at) > 90 \
         AND importance > 0.1 \
         AND type NOT IN ('pattern', 'decision', 'architecture')",
        [],
    ).unwrap_or(0) as i64
}
