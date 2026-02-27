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
    pub fn new() -> Self {
        let home = dirs::home_dir().expect("home dir not found");
        let data_dir = home.join(".mcp-memoria").join("data");
        Self {
            global_db: data_dir.join("global.db"),
            personality_db: data_dir.join("personality.db"),
            data_dir,
        }
    }

    pub fn project_db_path() -> Option<PathBuf> {
        let cwd = std::env::var("MCP_PROJECT_DIR")
            .or_else(|_| std::env::var("CLAUDE_CWD"))
            .unwrap_or_else(|_| std::env::current_dir().unwrap().to_string_lossy().to_string());
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

/// Inicializa SQLite com schema idêntico ao Python
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
            embedding BLOB
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

        CREATE INDEX IF NOT EXISTS idx_type ON memories(type);
        CREATE INDEX IF NOT EXISTS idx_created ON memories(created_at);
        CREATE INDEX IF NOT EXISTS idx_chunks_memory ON memory_chunks(memory_id);",
    )?;

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

/// Gera ID único (sha256[:16] de type:content:timestamp)
pub fn generate_id(content: &str, mem_type: &str) -> String {
    let now = chrono::Utc::now().to_rfc3339();
    let input = format!("{}:{}:{}", mem_type, content, now);
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    format!("{:x}", hasher.finalize())[..16].to_string()
}

/// Salva memória com dedup check
pub fn save_memory(
    conn: &Connection,
    mem_type: &str,
    content: &str,
    tags: &str,
) -> Result<SaveResult> {
    // Dedup check
    if mem_type != "conversation" {
        if let Some(existing_id) =
            crate::dedup::find_duplicate(conn, content, mem_type, 0.85)
        {
            conn.execute(
                "UPDATE memories SET content = ?, tags = ?, updated_at = datetime('now') WHERE id = ?",
                rusqlite::params![content, tags, existing_id],
            )?;
            return Ok(SaveResult {
                id: existing_id,
                dedup: "updated".into(),
            });
        }
    }

    let mem_id = generate_id(content, mem_type);
    conn.execute(
        "INSERT OR REPLACE INTO memories (id, type, content, tags, updated_at) \
         VALUES (?, ?, ?, ?, datetime('now'))",
        rusqlite::params![mem_id, mem_type, content, tags],
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

/// Lista memórias recentes
pub fn list_memories(
    conn: &Connection,
    mem_type: Option<&str>,
    limit: i64,
) -> Result<Vec<MemoryRecord>> {
    let mut results = Vec::new();

    if let Some(t) = mem_type {
        let mut stmt = conn.prepare(
            "SELECT id, type, content, tags, created_at FROM memories \
             WHERE type = ? ORDER BY updated_at DESC LIMIT ?",
        )?;
        let rows = stmt.query_map(rusqlite::params![t, limit], map_memory_row)?;
        for r in rows {
            results.push(r?);
        }
    } else {
        let mut stmt = conn.prepare(
            "SELECT id, type, content, tags, created_at FROM memories \
             ORDER BY updated_at DESC LIMIT ?",
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
        .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
        .unwrap_or(0);
    let indexed: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM memories WHERE embedding IS NOT NULL",
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

    let mut by_type = Vec::new();
    if let Ok(mut stmt) = conn.prepare("SELECT type, COUNT(*) FROM memories GROUP BY type") {
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
        indexed,
        chunks,
        cache_entries: cache,
        by_type,
    }
}

#[derive(Debug)]
pub struct DbStats {
    pub total: i64,
    pub indexed: i64,
    pub chunks: i64,
    pub cache_entries: i64,
    pub by_type: Vec<(String, i64)>,
}

/// Reindex: enfileira memórias sem embedding
pub fn get_unindexed_memories(conn: &Connection) -> Result<Vec<(String, String)>> {
    let mut stmt =
        conn.prepare("SELECT id, content FROM memories WHERE embedding IS NULL")?;
    let rows = stmt.query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?;
    Ok(rows.flatten().collect())
}

/// Compact: VACUUM + rebuild FTS
pub fn compact_db(conn: &Connection) -> Result<()> {
    // Rebuild FTS
    let _ = conn.execute_batch("INSERT INTO memories_fts(memories_fts) VALUES('rebuild');");
    conn.execute_batch("VACUUM;")?;
    Ok(())
}
