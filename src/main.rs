mod autotag;
mod chunking;
mod consolidation;
mod dedup;
mod embedding;
mod search;
mod storage;

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use rmcp::{
    ErrorData as McpError, ServerHandler, ServiceExt,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{CallToolResult, Content, ServerCapabilities, ServerInfo},
    schemars, tool, tool_handler, tool_router,
    transport::stdio,
};
use serde::Deserialize;
use tokio::sync::mpsc;
use tracing::info;

use embedding::{EmbeddingEngine, EmbeddingJob};
use storage::MemoryPaths;

// ---- Tool Parameter Structs ----

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct SaveParams {
    #[schemars(description = "What to save")]
    pub content: String,
    #[schemars(
        description = "Type: decision, pattern, preference, architecture, implementation, solution, todo, note"
    )]
    #[serde(default = "default_type")]
    pub r#type: String,
    #[schemars(description = "Scope: global, project, personality")]
    #[serde(default = "default_scope_project")]
    pub scope: String,
    #[schemars(description = "Comma-separated tags")]
    #[serde(default)]
    pub tags: String,
    #[schemars(description = "Project name (auto-detected if not provided)")]
    #[serde(default)]
    pub project_name: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct SearchParams {
    #[schemars(description = "Search term")]
    pub query: String,
    #[schemars(description = "Scope: global, project, personality, both, all")]
    #[serde(default = "default_scope_both")]
    pub scope: String,
    #[schemars(description = "Max results")]
    #[serde(default = "default_limit_5")]
    pub limit: usize,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ContextParams {
    #[schemars(description = "Current context or user question")]
    pub query: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ListParams {
    #[schemars(description = "Filter by type (optional)")]
    #[serde(default)]
    pub r#type: Option<String>,
    #[schemars(description = "Scope: global, project, personality, both, all")]
    #[serde(default = "default_scope_both")]
    pub scope: String,
    #[schemars(description = "Max results")]
    #[serde(default = "default_limit_10")]
    pub limit: usize,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct StatsParams {}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct DeleteParams {
    #[schemars(description = "Memory ID to delete")]
    pub id: String,
    #[schemars(description = "Scope: global, project, personality")]
    #[serde(default = "default_scope_project")]
    pub scope: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ReindexParams {
    #[schemars(description = "Scope: global, project, personality, all")]
    #[serde(default = "default_scope_all")]
    pub scope: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct CompactParams {
    #[schemars(description = "Scope: personality, project, global")]
    #[serde(default = "default_scope_personality")]
    pub scope: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ConsolidateParams {
    #[schemars(description = "Scope: personality, project, global, all")]
    #[serde(default = "default_scope_personality")]
    pub scope: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct LinkParams {
    #[schemars(description = "Source memory ID")]
    pub from_id: String,
    #[schemars(description = "Target memory ID")]
    pub to_id: String,
    #[schemars(description = "Relation: relates_to, supersedes, derived_from")]
    #[serde(default = "default_relation")]
    pub relation: String,
    #[schemars(description = "Scope of the memories")]
    #[serde(default = "default_scope_personality")]
    pub scope: String,
}

// ---- Defaults ----
fn default_type() -> String { "note".into() }
fn default_scope_project() -> String { "project".into() }
fn default_scope_both() -> String { "both".into() }
fn default_scope_all() -> String { "all".into() }
fn default_scope_personality() -> String { "personality".into() }
fn default_limit_5() -> usize { 5 }
fn default_limit_10() -> usize { 10 }
fn default_relation() -> String { "relates_to".into() }

// ---- Scope weights for cross-scope merge ----
fn scope_weight(scope: &str) -> f64 {
    match scope {
        "project" => 1.0,
        "personality" => 0.85,
        "global" => 0.7,
        _ => 0.8,
    }
}

// ---- MCP Server ----

#[derive(Clone)]
pub struct MemoryServer {
    paths: Arc<MemoryPaths>,
    embedding_engine: Arc<EmbeddingEngine>,
    job_sender: mpsc::Sender<EmbeddingJob>,
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl MemoryServer {
    pub fn new(
        paths: MemoryPaths,
        engine: Arc<EmbeddingEngine>,
        job_sender: mpsc::Sender<EmbeddingJob>,
    ) -> Self {
        Self {
            paths: Arc::new(paths),
            embedding_engine: engine,
            job_sender,
            tool_router: Self::tool_router(),
        }
    }

    /// Cross-scope parallel search com tokio::join! e scope weights
    async fn do_search_parallel(
        &self,
        query: String,
        scope: String,
        limit: usize,
    ) -> Vec<(String, search::SearchResult)> {
        let dbs = storage::resolve_scope_dbs(&scope, &self.paths);
        let engine = self.embedding_engine.clone();

        // Compute embedding once (blocking)
        let query_clone = query.clone();
        let query_emb = tokio::task::spawn_blocking(move || {
            engine.embed(&query_clone).ok()
        })
        .await
        .ok()
        .flatten();

        // Parallelizar buscas por scope
        let mut handles = Vec::new();
        for (scope_name, db_path) in dbs {
            if !db_path.exists() && scope_name == "project" {
                continue;
            }
            let query = query.clone();
            let query_emb = query_emb.clone();
            let scope_name = scope_name.clone();

            handles.push(tokio::task::spawn_blocking(move || {
                let conn = match storage::init_db(&db_path) {
                    Ok(c) => c,
                    Err(_) => return vec![],
                };
                let results = search::search_hybrid(
                    &conn,
                    &query,
                    query_emb.as_deref(),
                    limit,
                );
                let weight = scope_weight(&scope_name);
                results
                    .into_iter()
                    .map(|mut r| {
                        r.relevance *= weight;
                        r.relevance = (r.relevance * 10000.0).round() / 10000.0;
                        (scope_name.clone(), r)
                    })
                    .collect::<Vec<_>>()
            }));
        }

        let mut all_results = Vec::new();
        for handle in handles {
            if let Ok(results) = handle.await {
                all_results.extend(results);
            }
        }

        all_results.sort_by(|a, b| b.1.relevance.partial_cmp(&a.1.relevance).unwrap());
        all_results.truncate(limit);
        all_results
    }

    /// Sync version for tools that need it
    fn do_search(
        &self,
        query: &str,
        scope: &str,
        limit: usize,
    ) -> Vec<(String, search::SearchResult)> {
        let dbs = storage::resolve_scope_dbs(scope, &self.paths);
        let mut all_results = Vec::new();

        let query_emb = self.embedding_engine.embed(query).ok();

        for (scope_name, db_path) in dbs {
            if !db_path.exists() && scope_name == "project" {
                continue;
            }
            let conn = match storage::init_db(&db_path) {
                Ok(c) => c,
                Err(_) => continue,
            };
            let results = search::search_hybrid(
                &conn,
                query,
                query_emb.as_deref(),
                limit,
            );
            let weight = scope_weight(&scope_name);
            for mut r in results {
                r.relevance *= weight;
                r.relevance = (r.relevance * 10000.0).round() / 10000.0;
                all_results.push((scope_name.clone(), r));
            }
        }

        all_results.sort_by(|a, b| b.1.relevance.partial_cmp(&a.1.relevance).unwrap());
        all_results.truncate(limit);
        all_results
    }

    fn queue_embedding(&self, db_path: &PathBuf, record_id: &str, content: &str) {
        let _ = self.job_sender.try_send(EmbeddingJob {
            db_path: db_path.to_string_lossy().to_string(),
            record_id: record_id.to_string(),
            content: content.to_string(),
        });
    }

    fn resolve_save_db(&self, scope: &str) -> Option<PathBuf> {
        match scope {
            "global" => Some(self.paths.global_db.clone()),
            "personality" => Some(self.paths.personality_db.clone()),
            "project" => MemoryPaths::project_db_path(),
            _ => Some(self.paths.personality_db.clone()),
        }
    }

    // ---- Tools ----

    #[tool(description = "USE AUTOMATICALLY at the start of each conversation. Returns relevant memories for the current context (project + global). Works as an automatic 'recall'.")]
    async fn memory_context(
        &self,
        Parameters(params): Parameters<ContextParams>,
    ) -> Result<CallToolResult, McpError> {
        let results = self.do_search_parallel(params.query, "all".into(), 8).await;

        if results.is_empty() {
            return Ok(CallToolResult::success(vec![Content::text(
                "[Memory] No relevant context found.",
            )]));
        }

        let mut output = "## Memory Context\n\n".to_string();
        for (scope, r) in &results {
            output.push_str(&format!(
                "**[{}:{}]** {}\n",
                scope, r.mem_type, r.content
            ));
        }
        output.push_str("\n---\n_Use this context to inform your responses._");

        Ok(CallToolResult::success(vec![Content::text(output)]))
    }

    #[tool(description = "Search specific memories when you need detailed information about past decisions, patterns, or preferences. Use 'personality' scope to find similar implementations from other projects.")]
    async fn memory_search(
        &self,
        Parameters(params): Parameters<SearchParams>,
    ) -> Result<CallToolResult, McpError> {
        let results = self.do_search_parallel(params.query, params.scope, params.limit).await;

        if results.is_empty() {
            return Ok(CallToolResult::success(vec![Content::text(
                "No memories found.",
            )]));
        }

        let mut output = format!("## Memories ({})\n\n", results.len());
        for (scope, r) in &results {
            output.push_str(&format!(
                "**[{}] {}** (relevance: {}, method: {})\n{}\n",
                scope.to_uppercase(),
                r.mem_type,
                r.relevance,
                r.method,
                r.content
            ));
            if !r.tags.is_empty() {
                output.push_str(&format!("_Tags: {}_\n", r.tags));
            }
            output.push('\n');
        }

        Ok(CallToolResult::success(vec![Content::text(output)]))
    }

    #[tool(description = "Save important decision, pattern, or implementation. Auto-tags are extracted automatically. Use after: (1) making architecture decisions, (2) defining code patterns, (3) learning user preferences, (4) implementing new features.")]
    fn memory_save(
        &self,
        Parameters(params): Parameters<SaveParams>,
    ) -> Result<CallToolResult, McpError> {
        if params.content.is_empty() {
            return Ok(CallToolResult::success(vec![Content::text(
                "Error: empty content.",
            )]));
        }

        let db_path = match self.resolve_save_db(&params.scope) {
            Some(p) => p,
            None => {
                return Ok(CallToolResult::success(vec![Content::text(
                    "Error: project not detected. Use scope='personality' or 'global'.",
                )]));
            }
        };

        let mut tags = params.tags.clone();

        // Para personality scope, adiciona project name nas tags
        if params.scope == "personality" {
            let project_name = if params.project_name.is_empty() {
                std::env::var("MCP_PROJECT_DIR")
                    .or_else(|_| std::env::var("CLAUDE_CWD"))
                    .ok()
                    .and_then(|p| {
                        std::path::Path::new(&p)
                            .file_name()
                            .map(|n| n.to_string_lossy().to_string())
                    })
                    .unwrap_or_else(|| "no-project".to_string())
            } else {
                params.project_name.clone()
            };
            if !project_name.is_empty() && !tags.contains(&project_name) {
                tags = if tags.is_empty() {
                    project_name
                } else {
                    format!("{},{}", tags, project_name)
                };
            }
        }

        let conn = match storage::init_db(&db_path) {
            Ok(c) => c,
            Err(e) => {
                return Ok(CallToolResult::success(vec![Content::text(format!(
                    "Error: {}",
                    e
                ))]));
            }
        };

        match storage::save_memory(&conn, &params.r#type, &params.content, &tags) {
            Ok(result) => {
                self.queue_embedding(&db_path, &result.id, &params.content);
                let dedup_info = if result.dedup == "updated" {
                    "\n- Dedup: updated existing (similar found)"
                } else {
                    ""
                };
                Ok(CallToolResult::success(vec![Content::text(format!(
                    "Memory saved ({})\n- Type: {}\n- ID: {}\n- Tags: auto-enriched\n- Embedding: queued (f16 compressed){}",
                    params.scope,
                    params.r#type,
                    result.id,
                    dedup_info
                ))]))
            }
            Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Error: {}",
                e
            ))])),
        }
    }

    #[tool(description = "List recent memories. Useful to review decision history or find past implementations.")]
    fn memory_list(
        &self,
        Parameters(params): Parameters<ListParams>,
    ) -> Result<CallToolResult, McpError> {
        let dbs = storage::resolve_scope_dbs(&params.scope, &self.paths);
        let mut all_results = Vec::new();

        for (scope_name, db_path) in dbs {
            if !db_path.exists() && scope_name == "project" {
                continue;
            }
            let conn = match storage::init_db(&db_path) {
                Ok(c) => c,
                Err(_) => continue,
            };
            let mems = storage::list_memories(
                &conn,
                params.r#type.as_deref(),
                params.limit as i64,
            )
            .unwrap_or_default();
            for m in mems {
                all_results.push((scope_name.clone(), m));
            }
        }

        if all_results.is_empty() {
            return Ok(CallToolResult::success(vec![Content::text(
                "No memories found.",
            )]));
        }

        let mut output = format!("## Memories ({})\n\n", all_results.len());
        for (scope, r) in &all_results {
            let truncated: String = r.content.chars().take(80).collect();
            let ellipsis = if r.content.len() > 80 { "..." } else { "" };
            output.push_str(&format!(
                "- **[{}] {}**: {}{}\n",
                scope, r.mem_type, truncated, ellipsis
            ));
            if !r.tags.is_empty() {
                output.push_str(&format!("  _Tags: {}_\n", r.tags));
            }
            output.push_str(&format!("  `{}` | {}\n\n", r.id, r.created_at));
        }

        Ok(CallToolResult::success(vec![Content::text(output)]))
    }

    #[tool(description = "Show memory statistics (total, indexed, edges, archived, by type).")]
    fn memory_stats(
        &self,
        Parameters(_params): Parameters<StatsParams>,
    ) -> Result<CallToolResult, McpError> {
        let mut output = "## Memory Statistics\n\n".to_string();

        for (label, db_path) in [
            ("Global", &self.paths.global_db),
            ("Personality", &self.paths.personality_db),
        ] {
            let conn = match storage::init_db(db_path) {
                Ok(c) => c,
                Err(_) => continue,
            };
            let stats = storage::get_stats(&conn);
            output.push_str(&format!(
                "**{}** ({}):\n- Total: {}\n- Archived: {}\n- Indexed: {}\n- Chunks: {}\n- Edges: {}\n- Cache: {}\n- By type: {:?}\n\n",
                label,
                db_path.display(),
                stats.total,
                stats.archived,
                stats.indexed,
                stats.chunks,
                stats.edges,
                stats.cache_entries,
                stats.by_type,
            ));
        }

        if let Some(project_db) = MemoryPaths::project_db_path() {
            if project_db.exists() {
                if let Ok(conn) = storage::init_db(&project_db) {
                    let stats = storage::get_stats(&conn);
                    output.push_str(&format!(
                        "**Project** ({}):\n- Total: {}\n- Archived: {}\n- Indexed: {}\n- Chunks: {}\n- Edges: {}\n- Cache: {}\n- By type: {:?}\n\n",
                        project_db.display(), stats.total, stats.archived, stats.indexed,
                        stats.chunks, stats.edges, stats.cache_entries, stats.by_type,
                    ));
                }
            }
        }

        output.push_str("**Config v0.2**:\n");
        output.push_str("- Embeddings: f16 compressed (50% less storage)\n");
        output.push_str("- Model: all-MiniLM-L6-v2\n");
        output.push_str("- Search: hybrid (vector=0.7, text=0.3) + importance boost + graph 1-hop\n");
        output.push_str("- Scope weights: project=1.0, personality=0.85, global=0.7\n");
        output.push_str("- Temporal decay: 0.15\n");
        output.push_str("- Dedup threshold: 0.85\n");
        output.push_str("- Auto-tagging: enabled (~100 tech keywords)\n");
        output.push_str("- Consolidation: available (memory_consolidate)\n");

        Ok(CallToolResult::success(vec![Content::text(output)]))
    }

    #[tool(description = "Remove a memory by ID.")]
    fn memory_delete(
        &self,
        Parameters(params): Parameters<DeleteParams>,
    ) -> Result<CallToolResult, McpError> {
        if params.id.is_empty() {
            return Ok(CallToolResult::success(vec![Content::text(
                "Error: ID required.",
            )]));
        }

        let db_path = match self.resolve_save_db(&params.scope) {
            Some(p) => p,
            None => {
                return Ok(CallToolResult::success(vec![Content::text(
                    "Error: project not detected.",
                )]));
            }
        };

        let conn = match storage::init_db(&db_path) {
            Ok(c) => c,
            Err(e) => {
                return Ok(CallToolResult::success(vec![Content::text(format!(
                    "Error: {}",
                    e
                ))]));
            }
        };

        let deleted = conn
            .execute(
                "DELETE FROM memories WHERE id = ?",
                rusqlite::params![params.id],
            )
            .unwrap_or(0);

        if deleted > 0 {
            Ok(CallToolResult::success(vec![Content::text(format!(
                "Memory {} deleted.",
                params.id
            ))]))
        } else {
            Ok(CallToolResult::success(vec![Content::text(format!(
                "Memory {} not found.",
                params.id
            ))]))
        }
    }

    #[tool(description = "Reindex all memories that don't have embeddings yet.")]
    fn memory_reindex(
        &self,
        Parameters(params): Parameters<ReindexParams>,
    ) -> Result<CallToolResult, McpError> {
        let dbs = storage::resolve_scope_dbs(&params.scope, &self.paths);
        let mut total = 0usize;
        let mut details = Vec::new();

        for (scope_name, db_path) in dbs {
            if !db_path.exists() && scope_name == "project" {
                continue;
            }
            let conn = match storage::init_db(&db_path) {
                Ok(c) => c,
                Err(_) => continue,
            };
            let unindexed = storage::get_unindexed_memories(&conn).unwrap_or_default();
            let count = unindexed.len();
            for (id, content) in unindexed {
                self.queue_embedding(&db_path, &id, &content);
            }
            total += count;
            details.push(format!("- {}: {} queued", scope_name, count));
        }

        Ok(CallToolResult::success(vec![Content::text(format!(
            "## Reindex Started\n\nQueued {} memories for embedding (f16).\n{}\n\nWorker processing in background.",
            total,
            details.join("\n")
        ))]))
    }

    #[tool(description = "Compact database: VACUUM + FTS rebuild + TTL cleanup + importance decay.")]
    fn memory_compact(
        &self,
        Parameters(params): Parameters<CompactParams>,
    ) -> Result<CallToolResult, McpError> {
        let db_path = match self.resolve_save_db(&params.scope) {
            Some(p) => p,
            None => {
                return Ok(CallToolResult::success(vec![Content::text(
                    "Error: project not detected.",
                )]));
            }
        };

        let conn = match storage::init_db(&db_path) {
            Ok(c) => c,
            Err(e) => {
                return Ok(CallToolResult::success(vec![Content::text(format!(
                    "Error: {}",
                    e
                ))]));
            }
        };

        match storage::compact_db(&conn, &params.scope) {
            Ok(result) => {
                Ok(CallToolResult::success(vec![Content::text(format!(
                    "## Compaction Complete\n\n- TTL applied: {} memories\n- Importance decayed: {}\n- VACUUM + FTS rebuild done.",
                    result.ttl_applied, result.decayed
                ))]))
            }
            Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Error: {}", e
            ))])),
        }
    }

    #[tool(description = "Consolidate memories: merge similar entries, summarize conversation sessions by project, archive old duplicates. Reduces noise and improves search quality.")]
    fn memory_consolidate(
        &self,
        Parameters(params): Parameters<ConsolidateParams>,
    ) -> Result<CallToolResult, McpError> {
        let dbs = storage::resolve_scope_dbs(&params.scope, &self.paths);
        let mut total_result = consolidation::ConsolidationResult::default();

        for (scope_name, db_path) in dbs {
            if !db_path.exists() && scope_name == "project" {
                continue;
            }
            let conn = match storage::init_db(&db_path) {
                Ok(c) => c,
                Err(_) => continue,
            };
            let result = consolidation::run_consolidation(&conn);
            total_result.conversations_consolidated += result.conversations_consolidated;
            total_result.similar_merged += result.similar_merged;
            total_result.archived += result.archived;
        }

        Ok(CallToolResult::success(vec![Content::text(format!(
            "## Consolidation Complete\n\n- Conversation groups consolidated: {}\n- Similar memories merged: {}\n- Total archived: {}",
            total_result.conversations_consolidated,
            total_result.similar_merged,
            total_result.archived,
        ))]))
    }

    #[tool(description = "Create a manual link between two memories. Relations: relates_to, supersedes, derived_from.")]
    fn memory_link(
        &self,
        Parameters(params): Parameters<LinkParams>,
    ) -> Result<CallToolResult, McpError> {
        if params.from_id.is_empty() || params.to_id.is_empty() {
            return Ok(CallToolResult::success(vec![Content::text(
                "Error: both from_id and to_id required.",
            )]));
        }

        let valid_relations = ["relates_to", "supersedes", "derived_from"];
        if !valid_relations.contains(&params.relation.as_str()) {
            return Ok(CallToolResult::success(vec![Content::text(
                "Error: relation must be: relates_to, supersedes, or derived_from",
            )]));
        }

        let db_path = match self.resolve_save_db(&params.scope) {
            Some(p) => p,
            None => {
                return Ok(CallToolResult::success(vec![Content::text(
                    "Error: project not detected.",
                )]));
            }
        };

        let conn = match storage::init_db(&db_path) {
            Ok(c) => c,
            Err(e) => {
                return Ok(CallToolResult::success(vec![Content::text(format!(
                    "Error: {}",
                    e
                ))]));
            }
        };

        storage::create_edge(&conn, &params.from_id, &params.to_id, &params.relation);

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Link created: {} --[{}]--> {}",
            params.from_id, params.relation, params.to_id
        ))]))
    }
}

#[tool_handler]
impl ServerHandler for MemoryServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            instructions: Some(
                "MCP Memory Server (Rust) v0.2 — Persistent memory for AI assistants. \
                 Hybrid search (0.7 embedding + 0.3 BM25), temporal decay, \
                 Jaccard deduplication, chunking. 3 scopes: global, personality, project. \
                 v0.2: auto-tagging, importance scoring, graph relations, f16 embeddings, \
                 consolidation, TTL, cross-scope parallel search."
                    .into(),
            ),
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            ..Default::default()
        }
    }
}

// ---- Main ----

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .init();

    info!("MCP Memory Server (Rust) v0.2 Starting...");

    let paths = MemoryPaths::new();
    info!("Global DB: {}", paths.global_db.display());
    info!("Personality DB: {}", paths.personality_db.display());

    // Inicializa DBs com novo schema
    let conn_global = storage::init_db(&paths.global_db)?;
    let conn_personality = storage::init_db(&paths.personality_db)?;

    // Migrar embeddings f32 → f16 em background
    let migrated_global = embedding::migrate_embeddings_to_f16(&conn_global);
    let migrated_personality = embedding::migrate_embeddings_to_f16(&conn_personality);
    if migrated_global + migrated_personality > 0 {
        info!(
            "Migrated {} embeddings to f16 (global: {}, personality: {})",
            migrated_global + migrated_personality,
            migrated_global,
            migrated_personality
        );
    }

    // Drop connections antes de iniciar server
    drop(conn_global);
    drop(conn_personality);

    // Carrega embedding engine
    let engine = Arc::new(EmbeddingEngine::new()?);

    // Background worker
    let job_sender = embedding::start_background_worker(engine.clone());

    let server = MemoryServer::new(paths, engine, job_sender);

    info!("Search: hybrid (vector=0.7, text=0.3) + importance + graph 1-hop");
    info!("Embeddings: f16 compressed (50% less storage)");
    info!("Auto-tagging: ~100 tech keywords");
    info!("Dedup: Jaccard threshold=0.85");
    info!("Scope weights: project=1.0, personality=0.85, global=0.7");

    let service = server
        .serve(stdio())
        .await
        .inspect_err(|e| tracing::error!("Erro ao iniciar server: {:?}", e))
        .map_err(|e| anyhow::anyhow!("{:?}", e))?;

    info!("MCP server v0.2 rodando via stdio");
    service.waiting().await?;

    Ok(())
}
