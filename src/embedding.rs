use std::sync::Arc;
use anyhow::Result;
use fastembed::{TextEmbedding, InitOptions, EmbeddingModel};
use rusqlite::Connection;
use sha2::{Sha256, Digest};
use tokio::sync::mpsc;
use tracing::{info, warn};

/// Wrapper para fastembed TextEmbedding (thread-safe via Mutex)
pub struct EmbeddingEngine {
    model: std::sync::Mutex<TextEmbedding>,
}

impl EmbeddingEngine {
    pub fn new() -> Result<Self> {
        Self::with_model(EmbeddingModel::AllMiniLML6V2)
    }

    pub fn with_model(model_type: EmbeddingModel) -> Result<Self> {
        info!("Carregando modelo de embedding ({:?})...", model_type);
        let model = TextEmbedding::try_new(
            InitOptions::new(model_type).with_show_download_progress(true),
        )?;
        info!("Modelo de embedding carregado");
        Ok(Self { model: std::sync::Mutex::new(model) })
    }

    /// Gera embedding para um texto
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let mut model = self.model.lock().map_err(|e| anyhow::anyhow!("lock: {}", e))?;
        let results = model.embed(vec![text.to_string()], None)?;
        Ok(results.into_iter().next().unwrap_or_default())
    }

    /// Gera embeddings em batch
    pub fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut model = self.model.lock().map_err(|e| anyhow::anyhow!("lock: {}", e))?;
        let results = model.embed(texts.to_vec(), None)?;
        Ok(results)
    }
}

/// Cache de embeddings em SQLite (text_hash + model → embedding blob)
pub fn get_cached_embedding(conn: &Connection, text: &str, model: &str) -> Option<Vec<f32>> {
    let text_hash = compute_text_hash(text, model);
    let mut stmt = conn
        .prepare("SELECT embedding FROM embedding_cache WHERE text_hash = ? AND model = ?")
        .ok()?;
    let blob: Vec<u8> = stmt
        .query_row(rusqlite::params![text_hash, model], |row| row.get(0))
        .ok()?;
    Some(bytes_to_f32(&blob))
}

pub fn store_cached_embedding(conn: &Connection, text: &str, model: &str, embedding: &[f32]) {
    let text_hash = compute_text_hash(text, model);
    let blob = f32_to_bytes(embedding);
    let _ = conn.execute(
        "INSERT OR REPLACE INTO embedding_cache (text_hash, model, embedding, created_at) \
         VALUES (?, ?, ?, datetime('now'))",
        rusqlite::params![text_hash, model, blob],
    );
}

fn compute_text_hash(text: &str, model: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(format!("{}:{}", model, text).as_bytes());
    format!("{:x}", hasher.finalize())
}

pub fn f32_to_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

pub fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Item para o background worker processar
pub struct EmbeddingJob {
    pub db_path: String,
    pub record_id: String,
    pub content: String,
}

/// Inicia background worker que processa embedding jobs
pub fn start_background_worker(
    engine: Arc<EmbeddingEngine>,
) -> mpsc::Sender<EmbeddingJob> {
    let (tx, mut rx) = mpsc::channel::<EmbeddingJob>(256);

    tokio::spawn(async move {
        info!("Background embedding worker started");
        while let Some(job) = rx.recv().await {
            // Processa em blocking thread por causa do fastembed
            let engine = engine.clone();
            tokio::task::spawn_blocking(move || {
                if let Err(e) = process_embedding_job(&engine, &job) {
                    warn!("Embedding job error for {}: {}", job.record_id, e);
                }
            })
            .await
            .ok();
        }
    });

    tx
}

fn process_embedding_job(engine: &EmbeddingEngine, job: &EmbeddingJob) -> Result<()> {
    use crate::chunking::chunk_text;

    let conn = Connection::open(&job.db_path)?;
    conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")?;

    let model_name = "all-MiniLM-L6-v2";

    // Check cache
    let embedding = if let Some(cached) = get_cached_embedding(&conn, &job.content, model_name) {
        cached
    } else {
        let emb = engine.embed(&job.content)?;
        store_cached_embedding(&conn, &job.content, model_name, &emb);
        emb
    };

    // Atualiza embedding da memória principal
    let blob = f32_to_bytes(&embedding);
    conn.execute(
        "UPDATE memories SET embedding = ? WHERE id = ?",
        rusqlite::params![blob, job.record_id],
    )?;

    // Chunk conteúdos longos
    let chunks = chunk_text(&job.content, 400, 80);
    if chunks.len() > 1 {
        conn.execute(
            "DELETE FROM memory_chunks WHERE memory_id = ?",
            rusqlite::params![job.record_id],
        )?;

        for (idx, chunk) in chunks.iter().enumerate() {
            let chunk_id = format!("{}_c{}", job.record_id, idx);
            let chunk_emb =
                if let Some(cached) = get_cached_embedding(&conn, chunk, model_name) {
                    cached
                } else {
                    let emb = engine.embed(chunk)?;
                    store_cached_embedding(&conn, chunk, model_name, &emb);
                    emb
                };
            let chunk_blob = f32_to_bytes(&chunk_emb);
            conn.execute(
                "INSERT OR REPLACE INTO memory_chunks \
                 (id, memory_id, chunk_index, chunk_text, embedding) \
                 VALUES (?, ?, ?, ?, ?)",
                rusqlite::params![chunk_id, job.record_id, idx as i64, chunk, chunk_blob],
            )?;
        }
    }

    Ok(())
}
