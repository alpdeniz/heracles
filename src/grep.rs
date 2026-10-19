use std::collections::{HashMap, HashSet};
use std::io;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use grep::regex::RegexMatcherBuilder;
use grep::searcher::{BinaryDetection, SearcherBuilder, sinks};

use crate::filter::{self, Filter};

/// One matching line from the in-process ripgrep searcher.
#[derive(Debug, Clone)]
pub struct Hit {
    pub file: PathBuf,
    pub line: usize,
    pub text: String,
}

/// Searches `paths` (the current directory when empty) and returns up to `limit` hits.
pub fn search(
    query: &str,
    paths: &[PathBuf],
    filter: &Filter,
    depth: Option<usize>,
    limit: usize,
    respect_gitignore: bool,
) -> Result<Vec<Hit>> {
    ensure!(!query.is_empty(), "grep query must not be empty");
    if limit == 0 {
        return Ok(Vec::new());
    }
    let roots = if paths.is_empty() {
        vec![std::fs::canonicalize(".").context("could not resolve the current directory")?]
    } else {
        paths.to_vec()
    };

    let matcher = RegexMatcherBuilder::new()
        .case_smart(true)
        .line_terminator(Some(b'\n'))
        .build(query)
        .context("invalid grep pattern")?;
    let mut searcher = SearcherBuilder::new()
        .binary_detection(BinaryDetection::quit(b'\x00'))
        .line_number(true)
        .build();

    let mut hits = Vec::new();
    'roots: for root in &roots {
        let mut walk = filter::walker(root, depth, respect_gitignore);
        filter.filter_walk(&mut walk);

        for entry in walk.build() {
            if hits.len() >= limit {
                break 'roots;
            }
            let entry = match entry {
                Ok(entry) => entry,
                Err(e) => {
                    eprintln!("warning: {e}");
                    continue;
                }
            };
            if !entry.file_type().is_some_and(|t| t.is_file()) {
                continue;
            }
            let path = entry.path();
            let result = searcher.search_path(
                &matcher,
                path,
                sinks::Lossy(|line, text| {
                    if hits.len() >= limit {
                        return Ok(false);
                    }
                    hits.push(Hit {
                        file: path.to_path_buf(),
                        line: line as usize,
                        text: text.trim_end_matches(['\r', '\n']).to_owned(),
                    });
                    Ok(hits.len() < limit)
                }),
            );
            if let Err(e) = result {
                // Binary quit and ordinary I/O errors: skip the file, keep searching.
                if e.kind() != io::ErrorKind::Other && e.kind() != io::ErrorKind::InvalidData {
                    eprintln!("warning: {}: {e}", path.display());
                }
            }
        }
    }
    Ok(hits)
}

/// Line numbers per file that matched, for hybrid boosting.
pub fn line_index(hits: &[Hit]) -> HashMap<&Path, HashSet<usize>> {
    let mut index: HashMap<&Path, HashSet<usize>> = HashMap::new();
    for hit in hits {
        index.entry(&hit.file).or_default().insert(hit.line);
    }
    index
}

#[cfg(test)]
mod tests {
    use super::*;

    // Binary unit tests don't get `env!("CARGO_TARGET_TMPDIR")`.
    fn temp(name: &str) -> PathBuf {
        let root = std::env::temp_dir().join("heracles-grep-tests").join(name);
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        root
    }

    #[test]
    fn finds_matching_lines() {
        let root = temp("unit");
        std::fs::write(root.join("a.txt"), "alpha\nfindme here\nomega\n").unwrap();
        std::fs::write(root.join("b.txt"), "nothing\n").unwrap();

        let hits = search(
            "findme",
            std::slice::from_ref(&root),
            &Filter::default(),
            None,
            10,
            false,
        )
        .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].line, 2);
        assert!(hits[0].text.contains("findme"));
        assert!(hits[0].file.ends_with("a.txt"));
    }

    #[test]
    fn respects_limit_and_exclude() {
        let root = temp("limit");
        std::fs::create_dir_all(root.join("keep")).unwrap();
        std::fs::create_dir_all(root.join("skip")).unwrap();
        std::fs::write(root.join("keep/a.txt"), "hit one\nhit two\n").unwrap();
        std::fs::write(root.join("skip/b.txt"), "hit three\n").unwrap();

        let filter = Filter::new(&[], &["skip".into()]).unwrap();
        let hits = search("hit", &[root], &filter, None, 1, false).unwrap();
        assert_eq!(hits.len(), 1);
        assert!(hits[0].file.to_string_lossy().contains("keep"));
    }
}
