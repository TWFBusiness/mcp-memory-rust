/// Auto-tagging inteligente: extrai tags do conteúdo automaticamente.
/// - Tecnologias: regex match contra ~100 termos conhecidos
/// - Tipo de ação: detect keywords (fix/bug → bugfix, implement → feature, etc)
/// - Paths/arquivos: extrai extensões de arquivo

use std::collections::HashSet;

// Tecnologias conhecidas (case-insensitive match)
const TECH_KEYWORDS: &[(&str, &str)] = &[
    // Languages
    ("rust", "rust"), ("python", "python"), ("javascript", "javascript"),
    ("typescript", "typescript"), ("golang", "golang"), ("java", "java"),
    ("kotlin", "kotlin"), ("swift", "swift"), ("ruby", "ruby"),
    ("php", "php"), ("c++", "cpp"), ("c#", "csharp"), ("lua", "lua"),
    ("elixir", "elixir"), ("scala", "scala"), ("haskell", "haskell"),
    ("zig", "zig"), ("dart", "dart"),
    // Frameworks
    ("fastapi", "fastapi"), ("django", "django"), ("flask", "flask"),
    ("express", "express"), ("nextjs", "nextjs"), ("next.js", "nextjs"),
    ("nuxt", "nuxt"), ("react", "react"), ("vue", "vue"), ("angular", "angular"),
    ("svelte", "svelte"), ("solid", "solidjs"), ("solidjs", "solidjs"),
    ("remix", "remix"), ("astro", "astro"), ("actix", "actix"),
    ("axum", "axum"), ("rocket", "rocket"), ("gin", "gin"),
    ("spring", "spring"), ("rails", "rails"), ("laravel", "laravel"),
    ("tauri", "tauri"), ("electron", "electron"),
    // Databases
    ("sqlite", "sqlite"), ("postgres", "postgres"), ("postgresql", "postgres"),
    ("mysql", "mysql"), ("mongodb", "mongodb"), ("redis", "redis"),
    ("dynamodb", "dynamodb"), ("supabase", "supabase"), ("firebase", "firebase"),
    ("turso", "turso"), ("libsql", "libsql"),
    // ORM / DB tools
    ("tortoise", "tortoise-orm"), ("sqlalchemy", "sqlalchemy"),
    ("prisma", "prisma"), ("drizzle", "drizzle"), ("diesel", "diesel"),
    ("sea-orm", "sea-orm"), ("rusqlite", "rusqlite"),
    // Infra / DevOps
    ("docker", "docker"), ("kubernetes", "kubernetes"), ("k8s", "kubernetes"),
    ("nginx", "nginx"), ("terraform", "terraform"), ("ansible", "ansible"),
    ("github actions", "github-actions"), ("ci/cd", "cicd"),
    ("vercel", "vercel"), ("netlify", "netlify"), ("aws", "aws"),
    ("gcp", "gcp"), ("azure", "azure"), ("cloudflare", "cloudflare"),
    // Auth / Security
    ("jwt", "jwt"), ("oauth", "oauth"), ("oauth2", "oauth"), ("auth0", "auth0"),
    ("bcrypt", "bcrypt"), ("cors", "cors"), ("csrf", "csrf"),
    // Protocols / APIs
    ("graphql", "graphql"), ("grpc", "grpc"), ("rest", "rest"),
    ("websocket", "websocket"), ("websockets", "websocket"),
    ("sse", "sse"), ("mqtt", "mqtt"),
    // Tools / Libs
    ("tokio", "tokio"), ("serde", "serde"), ("pydantic", "pydantic"),
    ("pytest", "pytest"), ("jest", "jest"), ("vitest", "vitest"),
    ("webpack", "webpack"), ("vite", "vite"), ("esbuild", "esbuild"),
    ("tailwind", "tailwind"), ("sass", "sass"),
    // AI / ML
    ("openai", "openai"), ("anthropic", "anthropic"), ("claude", "claude"),
    ("langchain", "langchain"), ("llm", "llm"), ("embedding", "embedding"),
    ("fastembed", "fastembed"), ("onnx", "onnx"), ("pytorch", "pytorch"),
    ("tensorflow", "tensorflow"), ("huggingface", "huggingface"),
    // MCP
    ("mcp", "mcp"), ("rmcp", "rmcp"),
    // Package managers
    ("npm", "npm"), ("yarn", "yarn"), ("pnpm", "pnpm"), ("bun", "bun"),
    ("cargo", "cargo"), ("pip", "pip"), ("poetry", "poetry"),
    // Other
    ("wasm", "wasm"), ("webassembly", "wasm"),
    ("linux", "linux"), ("macos", "macos"), ("windows", "windows"),
    ("git", "git"), ("ssh", "ssh"), ("ssl", "ssl"), ("tls", "tls"),
    ("http", "http"), ("https", "https"),
];

// Keywords de ação → tag de tipo
const ACTION_PATTERNS: &[(&[&str], &str)] = &[
    (&["fix", "fixed", "bug", "bugfix", "hotfix", "patch", "corrigir", "corrigido", "erro"], "bugfix"),
    (&["implement", "implemented", "add", "added", "create", "created", "novo", "implementar", "implementado", "adicionar"], "feature"),
    (&["refactor", "refactored", "cleanup", "clean up", "reorganize", "refatorar"], "refactor"),
    (&["config", "configure", "configured", "setup", "configurar", "configuração"], "config"),
    (&["deploy", "deployed", "release", "publish", "publicar"], "deploy"),
    (&["test", "testing", "tested", "testar", "teste"], "testing"),
    (&["performance", "optimize", "optimized", "benchmark", "otimizar", "performance"], "performance"),
    (&["migrate", "migration", "migrar", "migração"], "migration"),
    (&["docs", "documentation", "documented", "documentação", "documentar"], "docs"),
    (&["security", "vulnerability", "segurança", "vulnerabilidade"], "security"),
];

// Extensões de arquivo → linguagem
const EXT_MAP: &[(&str, &str)] = &[
    (".rs", "rust"), (".py", "python"), (".js", "javascript"),
    (".ts", "typescript"), (".tsx", "typescript"), (".jsx", "javascript"),
    (".go", "golang"), (".java", "java"), (".kt", "kotlin"),
    (".rb", "ruby"), (".php", "php"), (".swift", "swift"),
    (".sql", "sql"), (".sh", "shell"), (".bash", "shell"),
    (".yml", "yaml"), (".yaml", "yaml"), (".toml", "toml"),
    (".json", "json"), (".html", "html"), (".css", "css"),
    (".vue", "vue"), (".svelte", "svelte"),
    (".dockerfile", "docker"), (".lua", "lua"), (".zig", "zig"),
    (".ex", "elixir"), (".exs", "elixir"), (".dart", "dart"),
];

/// Checa se keyword aparece como palavra inteira (word boundary)
fn has_word(words: &HashSet<&str>, keyword: &str) -> bool {
    // Keyword simples (sem espaço): check direto no set de palavras
    if !keyword.contains(' ') {
        return words.contains(keyword);
    }
    // Keyword composta (ex: "github actions"): check se contém no texto via contains
    // Neste caso, false positivos são raros, então mantemos contains
    false
}

/// Checa keyword composta no texto original (para termos com espaço)
fn has_compound_keyword(lower: &str, keyword: &str) -> bool {
    if !keyword.contains(' ') {
        return false;
    }
    lower.contains(keyword)
}

/// Extrai tags automaticamente do conteúdo.
/// Retorna Vec<String> com tags únicas, ordenadas, máximo 15.
/// Usa word boundary para evitar falsos positivos (ex: "rest" não matcha "interest").
pub fn extract_tags(content: &str) -> Vec<String> {
    let mut tags = HashSet::new();
    let lower = content.to_lowercase();

    // Tokenizar em palavras para word boundary matching
    let words: HashSet<&str> = lower
        .split(|c: char| !c.is_alphanumeric() && c != '+' && c != '#' && c != '-' && c != '.')
        .filter(|w| !w.is_empty())
        .collect();

    // 1. Tecnologias (word boundary)
    for (keyword, tag) in TECH_KEYWORDS {
        if has_word(&words, keyword) || has_compound_keyword(&lower, keyword) {
            tags.insert(tag.to_string());
        }
    }

    // 2. Ações (word boundary)
    for (keywords, tag) in ACTION_PATTERNS {
        for kw in *keywords {
            if has_word(&words, kw) {
                tags.insert(tag.to_string());
                break;
            }
        }
    }

    // 3. Extensões de arquivo (paths mencionados)
    for word in content.split_whitespace() {
        let w = word.trim_matches(|c: char| !c.is_alphanumeric() && c != '/' && c != '.' && c != '_' && c != '-');
        for (ext, lang) in EXT_MAP {
            if w.ends_with(ext) && w.len() > ext.len() {
                tags.insert(lang.to_string());
                break;
            }
        }
    }

    let mut result: Vec<String> = tags.into_iter().collect();
    result.sort();
    result.truncate(15);
    result
}

/// Merge auto-tags com tags manuais (comma-separated string).
/// Retorna string comma-separated sem duplicatas.
pub fn merge_tags(manual_tags: &str, auto_tags: &[String]) -> String {
    let mut all = HashSet::new();
    for t in manual_tags.split(',') {
        let trimmed = t.trim();
        if !trimmed.is_empty() {
            all.insert(trimmed.to_string());
        }
    }
    for t in auto_tags {
        all.insert(t.clone());
    }
    let mut result: Vec<String> = all.into_iter().collect();
    result.sort();
    result.join(",")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_tech() {
        let tags = extract_tags("Implemented FastAPI endpoint with Redis caching");
        assert!(tags.contains(&"fastapi".to_string()));
        assert!(tags.contains(&"redis".to_string()));
        assert!(tags.contains(&"feature".to_string()));
    }

    #[test]
    fn test_extract_files() {
        let tags = extract_tags("Modified src/main.rs and config.py");
        assert!(tags.contains(&"rust".to_string()));
        assert!(tags.contains(&"python".to_string()));
    }

    #[test]
    fn test_extract_actions() {
        let tags = extract_tags("Fixed bug in authentication flow");
        assert!(tags.contains(&"bugfix".to_string()));
    }

    #[test]
    fn test_merge_tags() {
        let auto = extract_tags("Using Docker and Redis");
        let merged = merge_tags("custom,project-x", &auto);
        assert!(merged.contains("custom"));
        assert!(merged.contains("docker"));
        assert!(merged.contains("redis"));
    }

    #[test]
    fn test_empty_content() {
        let tags = extract_tags("");
        assert!(tags.is_empty());
    }
}
