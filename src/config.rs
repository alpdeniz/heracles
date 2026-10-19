use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};

/// The `~/.heracles` directory. Overridable with `HERACLES_HOME`.
pub struct Home {
    root: PathBuf,
}

impl Home {
    pub fn resolve() -> Result<Self> {
        let root = match std::env::var_os("HERACLES_HOME") {
            Some(dir) => absolute(PathBuf::from(dir))?,
            None => dirs::home_dir()
                .context("could not determine home directory")?
                .join(".heracles"),
        };
        Ok(Self { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn config_file(&self) -> PathBuf {
        self.root.join("config.toml")
    }

    /// Where the vector memory (embeddings + chunk metadata) lives.
    pub fn memory_dir(&self) -> PathBuf {
        self.root.join("memory")
    }

    #[cfg(test)]
    pub(crate) fn at(root: PathBuf) -> Self {
        Self { root }
    }

    pub fn init(&self, force: bool) -> Result<()> {
        fs::create_dir_all(self.memory_dir())
            .with_context(|| format!("creating {}", self.memory_dir().display()))?;

        let config_file = self.config_file();
        if force || !config_file.exists() {
            Config::default().save(&config_file)?;
        }
        Ok(())
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub index: IndexConfig,
    pub search: SearchConfig,
    pub embedding: EmbeddingConfig,
}

/// Defaults for `search` (overridable per invocation only via `--mode`).
#[derive(Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct SearchConfig {
    /// Hybrid blend weight for an exact ripgrep hit: `(1 - w) * semantic + w * grep`.
    /// Must be in `0.0..=1.0`.
    pub grep_weight: f32,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self { grep_weight: 0.35 }
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct IndexConfig {
    /// Files larger than this are skipped.
    pub max_file_size_kb: u64,
    /// The same for PDFs, which are mostly fonts and images rather than text.
    pub max_pdf_size_kb: u64,
    /// Skip files matched by `.gitignore`, `.git/info/exclude` and the global gitignore.
    pub respect_gitignore: bool,
    /// Chunk budget, estimated as 4 UTF-8 bytes per token. Must fit the embedding server's
    /// physical batch (llama-server `-ub`), since inputs longer than that are rejected.
    pub chunk_tokens: usize,
    pub chunk_overlap_tokens: usize,
    /// Threads that read and chunk files while indexing. Embedding stays on
    /// `embedding.concurrent_requests`. Values below 1 count as 1. Default 4 fits
    /// 2023+ notebooks (NVMe + ≥8 logical cores) without queuing far ahead of embeds.
    pub read_workers: usize,
}

impl IndexConfig {
    pub fn chunking(&self) -> Chunking {
        Chunking {
            tokens: self.chunk_tokens,
            overlap_tokens: self.chunk_overlap_tokens,
        }
    }
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            max_file_size_kb: 1024,
            max_pdf_size_kb: 51200,
            respect_gitignore: false,
            chunk_tokens: 512,
            chunk_overlap_tokens: 64,
            read_workers: 4,
        }
    }
}

/// The settings that determine what text each embedding covers.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Chunking {
    pub tokens: usize,
    pub overlap_tokens: usize,
}

impl Chunking {
    pub const BYTES_PER_TOKEN: usize = 4;

    /// Chunk size and overlap in bytes. Fails when overlap is not smaller than a non-zero chunk.
    pub fn budget(&self) -> Result<(usize, usize)> {
        ensure!(
            self.tokens > 0 && self.overlap_tokens < self.tokens,
            "index.chunk_overlap_tokens must be smaller than a non-zero index.chunk_tokens"
        );
        Ok((
            self.tokens * Self::BYTES_PER_TOKEN,
            self.overlap_tokens * Self::BYTES_PER_TOKEN,
        ))
    }
}

/// Takes precedence over `embedding.api_key` in config.toml.
const API_KEY_ENV: &str = "HERACLES_EMBEDDING_API_KEY";

#[derive(Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct EmbeddingConfig {
    /// OpenAI-compatible embeddings endpoint, e.g. llama.cpp's `llama-server`.
    pub url: String,
    /// Sent as `Authorization: Bearer <key>`; matches llama-server's `--api-key`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    pub model: String,
    /// Stored length. Longer embeddings from the server are truncated, then re-normalized.
    pub dimensions: usize,
    /// Embedding requests in flight while indexing, so the server's slots (llama-server `-np`)
    /// stay busy between requests. Set it to the server's slot count.
    pub concurrent_requests: usize,
    /// Aborts an embedding request that runs longer than this. Values below 1 count as 1.
    pub timeout_secs: u64,
    /// Prepended to search queries (not to indexed text). Instruction-tuned models such as
    /// Qwen3-Embedding expect one; set to "" for models that don't.
    pub query_prefix: String,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            url: "http://localhost:8080/v1/embeddings".into(),
            api_key: None,
            model: "Qwen/Qwen3-embedding-0.6B".into(),
            dimensions: 1024,
            concurrent_requests: 2,
            timeout_secs: 60,
            query_prefix: "Instruct: Given a search query, retrieve relevant passages that answer the query\nQuery:".into(),
        }
    }
}

impl Config {
    /// Loads the config, falling back to defaults if `heracles init` hasn't run.
    pub fn load(home: &Home) -> Result<Self> {
        let path = home.config_file();
        let mut config: Self = match fs::read_to_string(&path) {
            Ok(text) => {
                toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Self::default(),
            Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
        };
        if let Some(key) = std::env::var(API_KEY_ENV).ok().filter(|k| !k.is_empty()) {
            config.embedding.api_key = Some(key);
        }
        config
            .validate()
            .with_context(|| format!("invalid config in {}", path.display()))?;
        Ok(config)
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            !self.embedding.url.trim().is_empty(),
            "embedding.url must not be empty"
        );
        ensure!(
            !self.embedding.model.trim().is_empty(),
            "embedding.model must not be empty"
        );
        ensure!(
            self.embedding.dimensions > 0,
            "embedding.dimensions must be greater than 0"
        );
        ensure!(
            (0.0..=1.0).contains(&self.search.grep_weight),
            "search.grep_weight must be between 0.0 and 1.0"
        );
        self.index.chunking().budget()?;
        Ok(())
    }

    /// A copy safe to print: secrets are masked.
    pub fn redacted(mut self) -> Self {
        if self.embedding.api_key.is_some() {
            self.embedding.api_key = Some("<redacted>".into());
        }
        self
    }

    fn save(&self, path: &Path) -> Result<()> {
        fs::write(path, toml::to_string_pretty(self)?)
            .with_context(|| format!("writing {}", path.display()))
    }
}

/// Makes `path` absolute without requiring it to exist (unlike `canonicalize`).
fn absolute(path: PathBuf) -> Result<PathBuf> {
    if path.is_absolute() {
        Ok(path)
    } else {
        Ok(std::env::current_dir()
            .context("could not determine the current directory")?
            .join(path))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_is_bytes_per_token() {
        let chunking = Chunking {
            tokens: 8,
            overlap_tokens: 2,
        };
        assert_eq!(chunking.budget().unwrap(), (32, 8));
    }

    #[test]
    fn budget_rejects_overlap_covering_a_whole_chunk() {
        assert!(
            Chunking {
                tokens: 8,
                overlap_tokens: 8,
            }
            .budget()
            .is_err()
        );
        assert!(
            Chunking {
                tokens: 0,
                overlap_tokens: 0,
            }
            .budget()
            .is_err()
        );
    }

    #[test]
    fn read_workers_default_for_modern_notebooks() {
        assert_eq!(IndexConfig::default().read_workers, 4);
    }

    #[test]
    fn validate_rejects_empty_url_and_zero_dimensions() {
        let mut config = Config::default();
        config.embedding.url.clear();
        assert!(config.validate().is_err());
        config = Config::default();
        config.embedding.dimensions = 0;
        assert!(config.validate().is_err());
        config = Config::default();
        config.embedding.model.clear();
        assert!(config.validate().is_err());
        assert!(Config::default().validate().is_ok());
    }

    #[test]
    fn validate_rejects_grep_weight_outside_unit_interval() {
        let mut config = Config::default();
        config.search.grep_weight = -0.1;
        assert!(config.validate().is_err());
        config.search.grep_weight = 1.1;
        assert!(config.validate().is_err());
        config.search.grep_weight = 0.0;
        assert!(config.validate().is_ok());
        config.search.grep_weight = 1.0;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn absolute_joins_relative_paths() {
        let abs = absolute(PathBuf::from("rel-home")).unwrap();
        assert!(abs.is_absolute());
        assert!(abs.ends_with("rel-home"));
        let already = PathBuf::from("/tmp/heracles-home");
        assert_eq!(absolute(already.clone()).unwrap(), already);
    }
}
