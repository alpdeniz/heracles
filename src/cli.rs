use std::num::NonZeroUsize;
use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};

use crate::filter::Filter;

/// Heracles — retrieves what you are looking for from your folders.
#[derive(Debug, Parser)]
#[command(name = "heracles", version, about)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Create ~/.heracles with a default config and an empty vector memory.
    Init {
        /// Overwrite an existing config with defaults.
        #[arg(long)]
        force: bool,
    },
    /// Print the active configuration.
    Config,
    /// Embed a folder's contents into the vector memory.
    ///
    /// Hidden files and directories are included. With --include/--exclude, only matching files
    /// are (re-)embedded or forgotten; other
    /// indexed files under the folder are left as they are. With --include or a limited --depth,
    /// the folder does not count as indexed, so a later search there indexes the rest.
    Index {
        #[arg(default_value = ".", value_parser = folder)]
        path: PathBuf,

        #[command(flatten)]
        filter: FilterArgs,

        #[command(flatten)]
        depth: DepthArg,
    },
    /// Search folders by meaning, by ripgrep, or both (`--mode`).
    ///
    /// Semantic/hybrid index or refresh the searched folders (the current directory unless
    /// --path is given). Covered folders are walked again so new files are embedded. --include,
    /// --exclude, or a limited --depth narrow that pass, and then the folder does not count as
    /// covered. Ranking still uses every indexed file unless --path is set; --depth only limits
    /// the index/ripgrep walk. Grep mode skips the vector memory; hybrid greps before locking it.
    Search(SearchArgs),
    /// List indexed files with their chunk counts, followed by memory statistics.
    List(Scope),
}

/// How `search` ranks hits.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, ValueEnum)]
pub enum SearchMode {
    /// Blend of semantic score and ripgrep matches; includes exact hits not in the index.
    #[default]
    Hybrid,
    /// Embedding similarity over indexed chunks.
    Semantic,
    /// Literal/regex lines via the bundled ripgrep engine.
    Grep,
}

#[derive(Debug, Args)]
pub struct SearchArgs {
    pub query: String,

    #[command(flatten)]
    pub scope: Scope,

    #[command(flatten)]
    pub depth: DepthArg,

    /// Ranking mode: hybrid (default), semantic, or grep.
    #[arg(long, value_enum, default_value_t = SearchMode::Hybrid)]
    pub mode: SearchMode,

    /// Maximum number of results.
    #[arg(short = 'n', long, default_value = "10")]
    pub limit: NonZeroUsize,
}

/// The indexed files a command works on.
#[derive(Debug, Args)]
pub struct Scope {
    /// Only files within this folder; repeat for several.
    #[arg(short, long = "path", value_parser = folder)]
    pub paths: Vec<PathBuf>,

    #[command(flatten)]
    pub filter: FilterArgs,
}

/// Default walk depth for `index` and for folders `search` indexes. 0 is unlimited.
pub const DEFAULT_DEPTH: usize = 0;

#[derive(Debug, Args)]
pub struct DepthArg {
    /// Walk at most this many levels into a folder: 1 is the files directly in it. 0 is unlimited
    /// (the default). A limited depth does not mark the folder covered.
    #[arg(short = 'd', long, default_value_t = DEFAULT_DEPTH, value_name = "LEVELS")]
    depth: usize,
}

impl DepthArg {
    /// Levels to walk, or `None` when `--depth 0` (unlimited).
    pub fn get(&self) -> Option<usize> {
        NonZeroUsize::new(self.depth).map(NonZeroUsize::get)
    }
}

/// Patterns use `.gitignore` syntax: a name or glob without a slash (`README.md`, `*.md`,
/// `node_modules`) matches files or directories at any depth. Quote globs so the shell
/// doesn't expand them.
#[derive(Debug, Args)]
pub struct FilterArgs {
    /// Only files matching this name or glob, e.g. '*.md'; repeat for several.
    #[arg(long, value_name = "PATTERN")]
    pub include: Vec<String>,

    /// Skip files or directories matching this name or glob; repeat for several.
    #[arg(long, value_name = "PATTERN")]
    pub exclude: Vec<String>,
}

impl FilterArgs {
    pub fn build(&self) -> anyhow::Result<Filter> {
        Filter::new(&self.include, &self.exclude)
    }
}

/// Canonical form is required: indexed chunks are keyed by absolute path, and
/// `--path` scoping is a prefix match against those keys.
fn folder(s: &str) -> Result<PathBuf, String> {
    let path = std::fs::canonicalize(s).map_err(|e| format!("{s}: {e}"))?;
    if !path.is_dir() {
        return Err(format!("{s}: not a directory"));
    }
    Ok(path)
}
