use std::fs::{self, File};
use std::io::{self, Read};
use std::panic::{self, AssertUnwindSafe};
use std::path::Path;

use pdf_oxide::PdfDocument;
use pdf_oxide::converters::ConversionOptions;

/// Bytes sampled before committing to a full read; enough for ZIP/PDF headers and a NUL check.
const PEEK: u64 = 8 * 1024;

/// Extensions that are never useful UTF-8 source. Checked before any read.
const BINARY_EXT: &[&str] = &[
    "7z", "a", "avi", "bin", "bmp", "bz2", "class", "db", "dll", "dmg", "dylib", "ear", "eot",
    "exe", "flac", "gif", "gz", "heic", "ico", "iso", "jar", "jpeg", "jpg", "lock", "mkv", "mov",
    "mp3", "mp4", "o", "ogg", "otf", "png", "pyc", "pyo", "rar", "so", "solc", "sqlite", "sqlite3",
    "tar", "tgz", "tif", "tiff", "ttf", "war", "wasm", "wav", "webm", "webp", "woff", "woff2",
    "xz", "zip", "zst",
];

/// Lockfiles and Finder junk whose names do not use a binary extension.
const SKIP_NAMES: &[&str] = &[
    ".DS_Store",
    "Cargo.lock",
    "Gemfile.lock",
    "bun.lock",
    "composer.lock",
    "package-lock.json",
    "pnpm-lock.yaml",
    "poetry.lock",
    "uv.lock",
    "yarn.lock",
];

pub fn is_pdf(file: &Path) -> bool {
    file.extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("pdf"))
}

pub fn is_binary_name(file: &Path) -> bool {
    if file
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| BINARY_EXT.iter().any(|b| ext.eq_ignore_ascii_case(b)))
    {
        return true;
    }
    let Some(name) = file.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    // Editor backups (`configure~`, `file.py~`) and known lock/junk names.
    name.ends_with('~') || SKIP_NAMES.iter().any(|n| name.eq_ignore_ascii_case(n))
}

/// True when the sampled prefix is not source text (NUL, or hex bytecode like `0x6060…`).
fn looks_binary(prefix: &[u8]) -> bool {
    if prefix.contains(&0) {
        return true;
    }
    // EVM / hex dumps: optional 0x then a long hex run — not prose to embed.
    if let Some(rest) = prefix.strip_prefix(b"0x") {
        let sample = &rest[..rest.len().min(64)];
        if !sample.is_empty() && sample.iter().all(|b| b.is_ascii_hexdigit()) {
            return true;
        }
    }
    false
}

/// The text indexed for `file`: Markdown converted from a PDF, or the contents of a UTF-8 text
/// file. `None` for other binary files. A PDF that can't be converted is `InvalidData`.
/// Indexing and snippets both read files through this, so line numbers agree.
pub fn text(file: &Path) -> io::Result<Option<String>> {
    if is_pdf(file) {
        let bytes = fs::read(file)?;
        return pdf_to_markdown(bytes).map(Some);
    }
    // shortcut: known non-text names never yield text, upgrade if we add extractors for them
    if is_binary_name(file) {
        return Ok(None);
    }

    let mut file = File::open(file)?;
    let mut bytes = Vec::new();
    (&mut file).take(PEEK).read_to_end(&mut bytes)?;
    if looks_binary(&bytes) {
        return Ok(None);
    }
    file.read_to_end(&mut bytes)?;
    if bytes.len() > PEEK as usize && bytes[PEEK as usize..].contains(&0) {
        return Ok(None);
    }
    Ok(String::from_utf8(bytes).ok())
}

fn pdf_to_markdown(bytes: Vec<u8>) -> io::Result<String> {
    let options = ConversionOptions {
        strip_running_headers_footers: true,
        ..ConversionOptions::default()
    };
    // Malformed PDFs can panic the parser; that must not abort indexing the other files.
    panic::catch_unwind(AssertUnwindSafe(|| {
        PdfDocument::from_bytes(bytes)?.to_markdown_all(&options)
    }))
    .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "PDF parser crashed"))?
    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn tmp(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "heracles-extract-{}-{}",
            name,
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn zip_extension_is_binary_without_reading() {
        assert!(is_binary_name(Path::new("polymarket-ai.zip")));
        let dir = tmp("zip");
        let path = dir.join("x.zip");
        fs::write(&path, "hello without nul").unwrap();
        assert_eq!(text(&path).unwrap(), None);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn solc_and_lockfiles_are_skipped() {
        assert!(is_binary_name(Path::new("contract.solc")));
        assert!(is_binary_name(Path::new("uv.lock")));
        assert!(is_binary_name(Path::new("package-lock.json")));
        assert!(is_binary_name(Path::new(".DS_Store")));
        assert!(is_binary_name(Path::new("configure~")));
    }

    #[test]
    fn hex_bytecode_prefix_is_binary() {
        let dir = tmp("hex");
        let path = dir.join("blob.dat");
        fs::write(&path, "0x60606040526004361061007f5763ffffffff").unwrap();
        assert_eq!(text(&path).unwrap(), None);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn nul_in_prefix_skips_rest() {
        let dir = tmp("nul");
        let path = dir.join("blob.dat");
        let mut f = fs::File::create(&path).unwrap();
        f.write_all(b"head\0").unwrap();
        let big = vec![b'x'; 100_000];
        f.write_all(&big).unwrap();
        assert_eq!(text(&path).unwrap(), None);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn utf8_text_still_loads() {
        let dir = tmp("txt");
        let path = dir.join("a.txt");
        fs::write(&path, "one\ntwo\n").unwrap();
        assert_eq!(text(&path).unwrap().as_deref(), Some("one\ntwo\n"));
        let _ = fs::remove_dir_all(&dir);
    }
}
