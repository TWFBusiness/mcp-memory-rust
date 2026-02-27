/// Divide texto em chunks com overlap por contagem de palavras.
/// Idêntico ao Python: chunk_text(text, 400, 80)
pub fn chunk_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() <= chunk_size {
        return vec![text.to_string()];
    }

    let mut chunks = Vec::new();
    let mut start = 0;
    while start < words.len() {
        let end = (start + chunk_size).min(words.len());
        let chunk = words[start..end].join(" ");
        chunks.push(chunk);
        if end >= words.len() {
            break;
        }
        start += chunk_size - overlap;
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_short_text_no_chunking() {
        let text = "hello world foo bar";
        let chunks = chunk_text(text, 400, 80);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], text);
    }

    #[test]
    fn test_chunking_with_overlap() {
        // 10 words, chunk_size=4, overlap=2
        let text = "a b c d e f g h i j";
        let chunks = chunk_text(text, 4, 2);
        assert_eq!(chunks[0], "a b c d");
        assert_eq!(chunks[1], "c d e f");
        assert_eq!(chunks[2], "e f g h");
        assert_eq!(chunks[3], "g h i j");
    }

    #[test]
    fn test_exact_chunk_size() {
        let text = "a b c d";
        let chunks = chunk_text(text, 4, 2);
        assert_eq!(chunks.len(), 1);
    }
}
