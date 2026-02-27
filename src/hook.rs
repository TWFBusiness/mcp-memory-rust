/// Hook para Claude Code: salva conversa automaticamente no personality.db
///
/// Captura eventos via stdin (JSON):
/// - UserPromptSubmit: acumula pergunta do usuário
/// - Stop: atualiza memória da sessão com resumo + transcript do assistente
///
/// Uma memória por sessão (UPSERT com ID determinístico).
/// Formato estruturado: extrai tools, arquivos, tópicos, auto-tags.
use std::collections::HashSet;
use std::io::Read;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

mod autotag;
mod chunking;
mod dedup;
mod embedding;
mod search;
mod storage;

const MAX_TURNS: usize = 20;

// ---- Structs ----

#[derive(Debug, Deserialize)]
struct HookInput {
    hook_event_name: Option<String>,
    session_id: Option<String>,
    cwd: Option<String>,
    prompt: Option<String>,
    stop_hook_active_tools: Option<Vec<ToolInfo>>,
    transcript: Option<Vec<TranscriptMessage>>,
}

#[derive(Debug, Deserialize)]
struct ToolInfo {
    name: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TranscriptMessage {
    role: Option<String>,
    content: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize, Default)]
struct SessionData {
    turns: Vec<Turn>,
    session_id: String,
    project: String,
    tools: Vec<String>,
    files: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Turn {
    role: String,
    content: String,
    timestamp: String,
}

// ---- Paths ----

fn session_file_path() -> PathBuf {
    let home = dirs::home_dir().expect("home dir");
    home.join(".claude")
        .join("mcp-memoria")
        .join("hooks")
        .join(".current_session.json")
}

fn personality_db_path() -> PathBuf {
    let home = dirs::home_dir().expect("home dir");
    home.join(".mcp-memoria")
        .join("data")
        .join("personality.db")
}

fn session_memory_id(session_id: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(format!("session:{}", session_id).as_bytes());
    format!("{:x}", hasher.finalize())[..16].to_string()
}

fn now_iso() -> String {
    chrono::Utc::now().format("%Y-%m-%dT%H:%M:%S%.6f").to_string()
}

// ---- Session persistence ----

fn load_session() -> SessionData {
    let path = session_file_path();
    if path.exists() {
        if let Ok(data) = std::fs::read_to_string(&path) {
            if let Ok(session) = serde_json::from_str(&data) {
                return session;
            }
        }
    }
    SessionData::default()
}

fn save_session(session: &SessionData) {
    let path = session_file_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(json) = serde_json::to_string(session) {
        let _ = std::fs::write(&path, json);
    }
}

// ---- Extract helpers ----

fn extract_files(text: &str) -> Vec<String> {
    let mut files = HashSet::new();
    for word in text.split_whitespace() {
        let w = word.trim_matches(|c: char| !c.is_alphanumeric() && c != '/' && c != '.' && c != '_' && c != '-');
        if w.starts_with('/') && w.contains('.') && w.len() > 3 {
            files.insert(w.to_string());
        }
    }
    let mut result: Vec<String> = files.into_iter().collect();
    result.sort();
    result.truncate(20);
    result
}

/// Extrai texto do último assistant message no transcript
fn extract_assistant_response(transcript: &[TranscriptMessage]) -> Option<String> {
    // Percorre de trás pra frente buscando último assistant
    for msg in transcript.iter().rev() {
        if msg.role.as_deref() == Some("assistant") {
            if let Some(content) = &msg.content {
                let text = match content {
                    serde_json::Value::String(s) => s.clone(),
                    serde_json::Value::Array(arr) => {
                        // Array de content blocks — extrair texto
                        arr.iter()
                            .filter_map(|item| {
                                if item.get("type")?.as_str()? == "text" {
                                    item.get("text")?.as_str().map(|s| s.to_string())
                                } else {
                                    None
                                }
                            })
                            .collect::<Vec<_>>()
                            .join(" ")
                    }
                    _ => continue,
                };
                if !text.is_empty() {
                    // Truncar a 1000 chars
                    let truncated: String = text.chars().take(1000).collect();
                    return Some(truncated);
                }
            }
        }
    }
    None
}

// ---- Build content ----

fn build_session_content(session: &SessionData) -> String {
    let mut lines = vec![format!(
        "[{}] Session ({} turns)",
        session.project,
        session.turns.len()
    )];

    if !session.tools.is_empty() {
        let tools: Vec<&str> = session.tools.iter().take(20).map(|s| s.as_str()).collect();
        lines.push(format!("Tools: {}", tools.join(", ")));
    }

    if !session.files.is_empty() {
        let files: Vec<&str> = session.files.iter().take(15).map(|s| s.as_str()).collect();
        lines.push(format!("Files: {}", files.join(", ")));
    }

    // User prompts (deduplicados)
    let mut seen = HashSet::new();
    let mut topics = Vec::new();
    for turn in &session.turns {
        if turn.role == "user" && turn.content.len() > 5 {
            let key: String = turn.content.chars().take(50).collect::<String>().to_lowercase();
            if seen.insert(key) {
                let truncated: String = turn.content.chars().take(300).collect();
                topics.push(truncated);
            }
        }
    }

    if !topics.is_empty() {
        lines.push("Topics:".to_string());
        for t in topics.iter().take(10) {
            lines.push(format!("  - {}", t));
        }
    }

    // Incluir última resposta do assistente se disponível
    let assistant_turns: Vec<&Turn> = session.turns.iter()
        .filter(|t| t.role == "assistant" && t.content.len() > 20)
        .collect();
    if let Some(last) = assistant_turns.last() {
        let truncated: String = last.content.chars().take(500).collect();
        lines.push(format!("Last response: {}", truncated));
    }

    lines.join("\n")
}

// ---- DB save ----

fn save_to_db(session: &SessionData) -> Option<String> {
    if session.session_id.is_empty() {
        return None;
    }

    let mem_id = session_memory_id(&session.session_id);
    let content = build_session_content(session);

    // Auto-tag do conteúdo da sessão
    let auto_tags = autotag::extract_tags(&content);
    let base_tags = format!("conversation,claude-code,{},auto-saved", session.project);
    let tags = autotag::merge_tags(&base_tags, &auto_tags);

    let db_path = personality_db_path();
    let conn = storage::init_db(&db_path).ok()?;

    // Check se existe
    let exists: bool = conn
        .query_row(
            "SELECT 1 FROM memories WHERE id = ?",
            rusqlite::params![mem_id],
            |_| Ok(true),
        )
        .unwrap_or(false);

    if exists {
        conn.execute(
            "UPDATE memories SET content = ?, tags = ?, \
             updated_at = datetime('now'), embedding = NULL WHERE id = ?",
            rusqlite::params![content, tags, mem_id],
        )
        .ok()?;
    } else {
        conn.execute(
            "INSERT INTO memories (id, type, content, tags, importance) \
             VALUES (?, 'conversation', ?, ?, 0.3)",
            rusqlite::params![mem_id, content, tags],
        )
        .ok()?;
    }

    Some(mem_id)
}

// ---- Event handlers ----

fn handle_user_prompt(input: &HookInput) {
    let session_id = input.session_id.as_deref().unwrap_or("unknown");
    let cwd = input.cwd.as_deref().unwrap_or("");
    let project = std::path::Path::new(cwd)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "no-project".to_string());
    let prompt = match &input.prompt {
        Some(p) if !p.is_empty() => p,
        _ => return,
    };

    let mut session = load_session();

    // Reset se nova sessão
    if session.session_id != session_id {
        session = SessionData {
            session_id: session_id.to_string(),
            project: project.clone(),
            ..Default::default()
        };
    }

    // Extrai files
    for f in extract_files(prompt) {
        if !session.files.contains(&f) {
            session.files.push(f);
        }
    }

    // Adiciona turno
    let truncated: String = prompt.chars().take(500).collect();
    session.turns.push(Turn {
        role: "user".to_string(),
        content: truncated,
        timestamp: now_iso(),
    });

    // Limita turnos
    if session.turns.len() > MAX_TURNS {
        let start = session.turns.len() - MAX_TURNS;
        session.turns = session.turns[start..].to_vec();
    }

    save_session(&session);
    eprintln!(
        "[Memory Hook] Captured user prompt ({} chars)",
        prompt.len()
    );
}

fn handle_stop(input: &HookInput) {
    let mut session = load_session();

    // Tools usadas
    if let Some(tools) = &input.stop_hook_active_tools {
        for t in tools {
            if let Some(name) = &t.name {
                if !session.tools.contains(name) {
                    session.tools.push(name.clone());
                }
            }
        }
    }

    // Extrair resposta do assistente do transcript
    let assistant_content = if let Some(transcript) = &input.transcript {
        extract_assistant_response(transcript)
    } else {
        None
    };

    // Turno do assistente — agora com conteúdo real se disponível
    let content = if let Some(response) = &assistant_content {
        response.clone()
    } else {
        let tool_names: Vec<String> = input
            .stop_hook_active_tools
            .as_ref()
            .map(|tools| {
                tools.iter().filter_map(|t| t.name.clone()).collect()
            })
            .unwrap_or_default();

        if tool_names.is_empty() {
            "Responded".to_string()
        } else {
            format!("Tools: {}", tool_names.join(", "))
        }
    };

    session.turns.push(Turn {
        role: "assistant".to_string(),
        content,
        timestamp: now_iso(),
    });

    if session.turns.len() > MAX_TURNS {
        let start = session.turns.len() - MAX_TURNS;
        session.turns = session.turns[start..].to_vec();
    }

    let mem_id = save_to_db(&session);
    save_session(&session);

    eprintln!(
        "[Memory Hook] Updated session memory {} ({} turns, transcript: {})",
        mem_id.unwrap_or_else(|| "none".into()),
        session.turns.len(),
        if input.transcript.is_some() { "yes" } else { "no" }
    );
}

// ---- Main ----

fn main() {
    let mut input = String::new();
    if std::io::stdin().read_to_string(&mut input).is_err() {
        return;
    }

    let input = input.trim();
    if input.is_empty() {
        return;
    }

    let hook_data: HookInput = match serde_json::from_str(input) {
        Ok(d) => d,
        Err(_) => return,
    };

    match hook_data.hook_event_name.as_deref() {
        Some("UserPromptSubmit") => handle_user_prompt(&hook_data),
        Some("Stop") => handle_stop(&hook_data),
        _ => {}
    }
}
