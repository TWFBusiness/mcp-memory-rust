# MCP Memory Server (Rust)

High-performance persistent memory server for AI assistants, implementing the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/).

Rewrite of MCP Memory Python in pure Rust for 9x faster search, 80x faster cold start, and 50% less RAM.

---

**[Português](#português)** | **[English](#english)**

---

## Português

### O que é

Servidor MCP que dá memória persistente ao seu assistente AI. Salva decisões, padrões, implementações e soluções entre sessões. Busca híbrida inteligente combina semântica (embeddings) com texto (BM25).

### Performance vs Python

| Operação | Python | Rust | Ganho |
|----------|--------|------|-------|
| Search (warm) | 97ms | 11ms | **9x** |
| Save (com dedup) | 43ms | 5ms | **9x** |
| Cold start | 12s | 0.15s | **80x** |
| RAM | 398 MB | 198 MB | **2x menos** |

### Funcionalidades

- **Busca híbrida**: 70% embedding (cosine similarity) + 30% BM25 (FTS5)
- **Deduplicação**: Jaccard similarity ≥ 0.85 antes de salvar
- **Temporal decay**: memórias recentes recebem boost automático
- **Chunking**: textos longos divididos em chunks de 400 palavras com 80 de overlap
- **3 escopos**: `global` (padrões permanentes), `personality` (cross-project), `project` (específico)
- **8 tools MCP**: save, search, context, list, stats, delete, reindex, compact
- **Embedding local**: all-MiniLM-L6-v2 via ONNX (sem API externa, sem custo)
- **Background worker**: embeddings processados em background sem bloquear
- **Hook de conversas**: binário standalone que salva conversas automaticamente (Claude Code)

### Instalação

#### Binários pré-compilados

Baixe da [página de releases](https://github.com/TWFBusiness/mcp-memory-rust/releases):

- `mcp-memory-rust-x86_64-apple-darwin.tar.gz` — macOS Intel
- `mcp-memory-rust-aarch64-apple-darwin.tar.gz` — macOS Apple Silicon
- `mcp-memory-rust-x86_64-unknown-linux-gnu.tar.gz` — Linux x86_64
- `mcp-memory-rust-x86_64-pc-windows-msvc.zip` — Windows x86_64

Cada release inclui dois binários:
- `mcp-memory-rust` — servidor MCP (funciona em qualquer IDE com suporte MCP)
- `mcp-memory-hook` — hook de conversas (exclusivo para Claude Code)

```bash
# macOS/Linux — extrair e tornar executável
tar xzf mcp-memory-rust-*.tar.gz
chmod +x mcp-memory-rust mcp-memory-hook
```

#### Compilar do fonte

```bash
git clone https://github.com/TWFBusiness/mcp-memory-rust.git
cd mcp-memory-rust
cargo build --release
# Binários em: target/release/mcp-memory-rust e target/release/mcp-memory-hook
```

### Configuração por IDE

#### Claude Code (CLI)

**1. Servidor MCP** (memória persistente):

```bash
claude mcp add -s user memory-rust /caminho/para/mcp-memory-rust
```

Ou edite `~/.claude.json`:

```json
{
  "mcpServers": {
    "memory-rust": {
      "command": "/caminho/para/mcp-memory-rust"
    }
  }
}
```

**2. Hook de conversas** (salva automaticamente toda conversa):

Edite `~/.claude/settings.json`:

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "/caminho/para/mcp-memory-hook",
            "timeout": 5
          }
        ]
      }
    ],
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "/caminho/para/mcp-memory-hook",
            "timeout": 5
          }
        ]
      }
    ]
  }
}
```

O hook captura automaticamente:
- **UserPromptSubmit**: acumula a pergunta do usuário na sessão
- **Stop**: salva a sessão completa no `personality.db` (UPSERT — uma memória por sessão)
- Extrai: tools usadas, arquivos mencionados, tópicos da conversa
- Executa em **<10ms** e usa **6 MB de RAM** (sem carregar modelo de embedding)
- Os embeddings são gerados depois pelo background worker do servidor MCP

> **Nota**: O hook é exclusivo para Claude Code. Cursor e Codex não possuem sistema de hooks — nesses IDEs, use apenas o servidor MCP e instrua o assistente a chamar `memory_save` nas instruções do projeto.

#### Cursor

Vá em **Settings → MCP Servers → Add Server**:

```json
{
  "mcpServers": {
    "memory-rust": {
      "command": "/caminho/para/mcp-memory-rust"
    }
  }
}
```

Ou edite `.cursor/mcp.json` no projeto:

```json
{
  "mcpServers": {
    "memory-rust": {
      "command": "/caminho/para/mcp-memory-rust"
    }
  }
}
```

Para salvar conversas automaticamente no Cursor, adicione nas instruções do projeto (`.cursorrules`):

```
Após cada resposta substantiva, chame memory_save com scope="personality" para salvar decisões, implementações e soluções relevantes.
```

#### OpenAI Codex CLI

```bash
codex mcp add memory-rust /caminho/para/mcp-memory-rust
```

Ou configure via variável de ambiente:

```bash
export MCP_SERVERS='{"memory-rust":{"command":"/caminho/para/mcp-memory-rust"}}'
```

Para salvar conversas automaticamente no Codex, adicione nas instruções do projeto (`AGENTS.md` ou `codex.md`):

```
Após cada resposta substantiva, chame memory_save com scope="personality" para salvar decisões, implementações e soluções relevantes.
```

### Tools disponíveis

| Tool | Descrição |
|------|-----------|
| `memory_context` | Recall automático — use no início de cada conversa |
| `memory_search` | Busca híbrida por memórias específicas |
| `memory_save` | Salva decisão, padrão, implementação, solução |
| `memory_list` | Lista memórias recentes com filtros |
| `memory_stats` | Estatísticas dos databases |
| `memory_delete` | Remove memória por ID |
| `memory_reindex` | Reprocessa embeddings pendentes |
| `memory_compact` | VACUUM + rebuild FTS5 |

### Onde ficam os dados

```
~/.mcp-memoria/data/
├── global.db        # Padrões permanentes
├── personality.db   # Memórias cross-project (conversas salvas aqui)
└── <project>/.mcp-memoria/project.db  # Específico do projeto
```

---

## English

### What is it

MCP server that gives persistent memory to your AI assistant. Saves decisions, patterns, implementations and solutions across sessions. Smart hybrid search combines semantic (embeddings) with text (BM25).

### Performance vs Python

| Operation | Python | Rust | Speedup |
|-----------|--------|------|---------|
| Search (warm) | 97ms | 11ms | **9x** |
| Save (with dedup) | 43ms | 5ms | **9x** |
| Cold start | 12s | 0.15s | **80x** |
| RAM | 398 MB | 198 MB | **2x less** |

### Features

- **Hybrid search**: 70% embedding (cosine similarity) + 30% BM25 (FTS5)
- **Deduplication**: Jaccard similarity ≥ 0.85 before saving
- **Temporal decay**: recent memories get automatic score boost
- **Chunking**: long texts split into 400-word chunks with 80-word overlap
- **3 scopes**: `global` (permanent patterns), `personality` (cross-project), `project` (project-specific)
- **8 MCP tools**: save, search, context, list, stats, delete, reindex, compact
- **Local embedding**: all-MiniLM-L6-v2 via ONNX (no external API, no cost)
- **Background worker**: embeddings processed in background without blocking
- **Conversation hook**: standalone binary that auto-saves conversations (Claude Code only)

### Installation

#### Pre-built binaries

Download from the [releases page](https://github.com/TWFBusiness/mcp-memory-rust/releases):

- `mcp-memory-rust-x86_64-apple-darwin.tar.gz` — macOS Intel
- `mcp-memory-rust-aarch64-apple-darwin.tar.gz` — macOS Apple Silicon
- `mcp-memory-rust-x86_64-unknown-linux-gnu.tar.gz` — Linux x86_64
- `mcp-memory-rust-x86_64-pc-windows-msvc.zip` — Windows x86_64

Each release includes two binaries:
- `mcp-memory-rust` — MCP server (works with any IDE that supports MCP)
- `mcp-memory-hook` — conversation hook (Claude Code only)

```bash
# macOS/Linux — extract and make executable
tar xzf mcp-memory-rust-*.tar.gz
chmod +x mcp-memory-rust mcp-memory-hook
```

#### Build from source

```bash
git clone https://github.com/TWFBusiness/mcp-memory-rust.git
cd mcp-memory-rust
cargo build --release
# Binaries at: target/release/mcp-memory-rust and target/release/mcp-memory-hook
```

### IDE Configuration

#### Claude Code (CLI)

**1. MCP Server** (persistent memory):

```bash
claude mcp add -s user memory-rust /path/to/mcp-memory-rust
```

Or edit `~/.claude.json`:

```json
{
  "mcpServers": {
    "memory-rust": {
      "command": "/path/to/mcp-memory-rust"
    }
  }
}
```

**2. Conversation hook** (auto-saves every conversation):

Edit `~/.claude/settings.json`:

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "/path/to/mcp-memory-hook",
            "timeout": 5
          }
        ]
      }
    ],
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "/path/to/mcp-memory-hook",
            "timeout": 5
          }
        ]
      }
    ]
  }
}
```

The hook automatically captures:
- **UserPromptSubmit**: accumulates the user's prompt in the session
- **Stop**: saves the full session to `personality.db` (UPSERT — one memory per session)
- Extracts: tools used, files mentioned, conversation topics
- Runs in **<10ms** using **6 MB RAM** (no embedding model loaded)
- Embeddings are generated later by the MCP server's background worker

> **Note**: The hook is exclusive to Claude Code. Cursor and Codex don't have a hooks system — for those IDEs, use only the MCP server and instruct the assistant to call `memory_save` in your project instructions.

#### Cursor

Go to **Settings → MCP Servers → Add Server**:

```json
{
  "mcpServers": {
    "memory-rust": {
      "command": "/path/to/mcp-memory-rust"
    }
  }
}
```

Or edit `.cursor/mcp.json` in your project:

```json
{
  "mcpServers": {
    "memory-rust": {
      "command": "/path/to/mcp-memory-rust"
    }
  }
}
```

To auto-save conversations in Cursor, add to your project instructions (`.cursorrules`):

```
After every substantive response, call memory_save with scope="personality" to save relevant decisions, implementations, and solutions.
```

#### OpenAI Codex CLI

```bash
codex mcp add memory-rust /path/to/mcp-memory-rust
```

Or configure via environment variable:

```bash
export MCP_SERVERS='{"memory-rust":{"command":"/path/to/mcp-memory-rust"}}'
```

To auto-save conversations in Codex, add to your project instructions (`AGENTS.md` or `codex.md`):

```
After every substantive response, call memory_save with scope="personality" to save relevant decisions, implementations, and solutions.
```

### Available tools

| Tool | Description |
|------|-------------|
| `memory_context` | Auto-recall — use at the start of each conversation |
| `memory_search` | Hybrid search for specific memories |
| `memory_save` | Save decision, pattern, implementation, solution |
| `memory_list` | List recent memories with filters |
| `memory_stats` | Database statistics |
| `memory_delete` | Remove memory by ID |
| `memory_reindex` | Reprocess pending embeddings |
| `memory_compact` | VACUUM + FTS5 rebuild |

### Data location

```
~/.mcp-memoria/data/
├── global.db        # Permanent patterns
├── personality.db   # Cross-project memories (conversations saved here)
└── <project>/.mcp-memoria/project.db  # Project-specific
```

### Architecture

```
src/
├── main.rs        # MCP server (rmcp), 8 tool handlers
├── hook.rs        # Conversation hook for Claude Code (standalone binary)
├── storage.rs     # SQLite: schema, CRUD, FTS5, scopes
├── search.rs      # Hybrid search, BM25, cosine, temporal decay
├── embedding.rs   # fastembed wrapper, cache, background worker
├── chunking.rs    # Text chunking (400 words, 80 overlap)
└── dedup.rs       # Jaccard deduplication
```

### License

MIT
