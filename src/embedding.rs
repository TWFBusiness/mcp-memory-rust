use std::sync::Arc;
use anyhow::Result;
use fastembed::{TextEmbedding, InitOptions, EmbeddingModel};
use half::f16;
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

    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let mut model = self.model.lock().map_err(|e| anyhow::anyhow!("lock: {}", e))?;
        let results = model.embed(vec![text.to_string()], None)?;
        Ok(results.into_iter().next().unwrap_or_default())
    }

    pub fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut model = self.model.lock().map_err(|e| anyhow::anyhow!("lock: {}", e))?;
        let results = model.embed(texts.to_vec(), None)?;
        Ok(results)
    }
}

// ---- Embedding compression (f16) ----

/// Comprime Vec<f32> para bytes f16 (50% menos espaço)
pub fn compress_embedding(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|&f| f16::from_f32(f).to_le_bytes())
        .collect()
}

/// Descomprime bytes f16 para Vec<f32>
pub fn decompress_embedding(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect()
}

/// Converte f32 para bytes (formato legado, 4 bytes por float)
pub fn f32_to_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// Lê bytes e detecta automaticamente formato (f16 ou f32) baseado no tamanho
/// 384 dims: f16 = 768 bytes, f32 = 1536 bytes
pub fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    // Heurística: se divisível por 4 e não por 2-only, assume f32
    // Se tamanho == 768 (384*2), é f16
    // Se tamanho == 1536 (384*4), é f32
    // Genérico: se bytes.len() % 4 == 0 e bytes.len() / 4 é um dim conhecido, f32
    let possible_f32_dims = bytes.len() / 4;
    let possible_f16_dims = bytes.len() / 2;

    // Se o número de dimensões f16 é 384 (nosso modelo), priorizar f16
    if possible_f16_dims == 384 && bytes.len() == 768 {
        return decompress_embedding(bytes);
    }

    // Fallback: assume f32 se divisível por 4
    if bytes.len() % 4 == 0 && possible_f32_dims > 0 {
        return bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
    }

    // Tenta f16 como último recurso
    if bytes.len() % 2 == 0 {
        return decompress_embedding(bytes);
    }

    vec![]
}

// ---- Cache ----

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
    let blob = compress_embedding(embedding); // Salva como f16
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

// ---- Background Worker ----

pub struct EmbeddingJob {
    pub db_path: String,
    pub record_id: String,
    pub content: String,
}

pub fn start_background_worker(
    engine: Arc<EmbeddingEngine>,
) -> mpsc::Sender<EmbeddingJob> {
    let (tx, mut rx) = mpsc::channel::<EmbeddingJob>(256);

    tokio::spawn(async move {
        info!("Background embedding worker started");
        while let Some(job) = rx.recv().await {
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

    // Salva como f16 comprimido
    let blob = compress_embedding(&embedding);
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
            let chunk_blob = compress_embedding(&chunk_emb);
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

/// Migra embeddings legados (f32) para f16 em background
pub fn migrate_embeddings_to_f16(conn: &Connection) -> usize {
    let mut count = 0usize;
    // Detecta embeddings f32 (1536 bytes para 384 dims)
    if let Ok(mut stmt) = conn.prepare(
        "SELECT id, embedding FROM memories WHERE embedding IS NOT NULL AND length(embedding) = 1536"
    ) {
        let rows: Vec<(String, Vec<u8>)> = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
            .ok()
            .map(|r| r.flatten().collect())
            .unwrap_or_default();

        for (id, blob) in rows {
            let f32_vec: Vec<f32> = blob.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let compressed = compress_embedding(&f32_vec);
            let _ = conn.execute(
                "UPDATE memories SET embedding = ? WHERE id = ?",
                rusqlite::params![compressed, id],
            );
            count += 1;
        }
    }

    // Mesma migração para chunks
    if let Ok(mut stmt) = conn.prepare(
        "SELECT id, embedding FROM memory_chunks WHERE embedding IS NOT NULL AND length(embedding) = 1536"
    ) {
        let rows: Vec<(String, Vec<u8>)> = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
            .ok()
            .map(|r| r.flatten().collect())
            .unwrap_or_default();

        for (id, blob) in rows {
            let f32_vec: Vec<f32> = blob.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let compressed = compress_embedding(&f32_vec);
            let _ = conn.execute(
                "UPDATE memory_chunks SET embedding = ? WHERE id = ?",
                rusqlite::params![compressed, id],
            );
            count += 1;
        }
    }

    count
}
