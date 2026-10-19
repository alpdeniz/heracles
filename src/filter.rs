use std::path::Path;

use anyhow::{Result, ensure};
use ignore::WalkBuilder;
use ignore::gitignore::{Gitignore, GitignoreBuilder};

/// WalkBuilder shared by index and grep: include hidden paths, optional gitignore, depth limit.
pub fn walker(root: &Path, depth: Option<usize>, respect_gitignore: bool) -> WalkBuilder {
    let mut walk = WalkBuilder::new(root);
    walk.max_depth(depth)
        .hidden(false)
        .git_ignore(respect_gitignore)
        .git_global(respect_gitignore)
        .git_exclude(respect_gitignore)
        .require_git(false);
    walk
}

/// `--include` / `--exclude` patterns in `.gitignore` syntax, matched against absolute paths.
/// A pattern without a slash (`README.md`, `*.md`, `node_modules`) matches a file or
/// directory name at any depth; one starting with `/` is an absolute path; any other is
/// matched at any depth (`docs/*.md` matches `/a/docs/x.md`).
#[derive(Clone)]
pub struct Filter {
    include: Gitignore,
    exclude: Gitignore,
}

impl Default for Filter {
    fn default() -> Self {
        Self {
            include: Gitignore::empty(),
            exclude: Gitignore::empty(),
        }
    }
}

impl Filter {
    pub fn new(include: &[String], exclude: &[String]) -> Result<Self> {
        Ok(Self {
            include: matcher(include)?,
            exclude: matcher(exclude)?,
        })
    }

    pub fn is_empty(&self) -> bool {
        self.include.is_empty() && self.exclude.is_empty()
    }

    /// Whether only some files are admitted by name, as opposed to all but excluded ones.
    pub fn has_include(&self) -> bool {
        !self.include.is_empty()
    }

    /// Whether an explicit `heracles index` should record the folder as covered.
    /// `--exclude` still covers (that is the index the user asked for); `--include` or a
    /// limited `--depth` indexes a slice, so a later search fills in the rest.
    pub fn covers_index(&self, depth: Option<usize>) -> bool {
        !self.has_include() && depth.is_none()
    }

    /// Whether a search-driven auto-index should record coverage.
    /// Any filter or limited depth means the pass was narrowed for this search only.
    pub fn covers_auto_index(&self, depth: Option<usize>) -> bool {
        self.is_empty() && depth.is_none()
    }

    /// `file` passes when neither it nor a parent directory is excluded and, if there are
    /// includes, it or a parent directory is included.
    pub fn admits(&self, file: &Path) -> bool {
        let hits = |m: &Gitignore| m.matched_path_or_any_parents(file, false).is_ignore();
        !hits(&self.exclude) && (self.include.is_empty() || hits(&self.include))
    }

    /// Whether a walk should descend into `dir`, given that its parents were admitted.
    pub fn admits_dir(&self, dir: &Path) -> bool {
        !self.exclude.matched(dir, true).is_ignore()
    }

    /// Skips entries this filter rejects. A directory is still entered unless it is excluded,
    /// so an included file inside it can be reached.
    pub fn filter_walk(&self, walk: &mut WalkBuilder) {
        if self.is_empty() {
            return;
        }
        let filter = self.clone();
        walk.filter_entry(move |e| match e.file_type() {
            Some(t) if t.is_dir() => filter.admits_dir(e.path()),
            _ => filter.admits(e.path()),
        });
    }
}

fn matcher(patterns: &[String]) -> Result<Gitignore> {
    let mut builder = GitignoreBuilder::new("/");
    for pattern in patterns {
        ensure!(
            !pattern.starts_with('!'),
            "{pattern}: negated patterns are not supported; use --exclude"
        );
        // `.gitignore` anchors patterns with an inner slash to the root, here `/`.
        let anchored = pattern.starts_with('/') || !pattern.trim_end_matches('/').contains('/');
        let line = if anchored {
            pattern.clone()
        } else {
            format!("**/{pattern}")
        };
        builder.add_line(None, &line)?;
    }
    Ok(builder.build()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn filter(include: &[&str], exclude: &[&str]) -> Filter {
        let owned = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        Filter::new(&owned(include), &owned(exclude)).unwrap()
    }

    fn admits(f: &Filter, file: &str) -> bool {
        f.admits(Path::new(file))
    }

    #[test]
    fn empty_admits_everything() {
        assert!(admits(&Filter::default(), "/a/b.rs"));
    }

    #[test]
    fn plain_names_and_globs_match_at_any_depth() {
        let f = filter(&["README.md", "*.rs"], &[]);
        assert!(admits(&f, "/a/README.md"));
        assert!(admits(&f, "/a/b/c/main.rs"));
        assert!(!admits(&f, "/a/notes.md"));
    }

    #[test]
    fn exclude_wins_and_covers_directories() {
        let f = filter(&["*.md"], &["drafts", "CHANGELOG.md"]);
        assert!(admits(&f, "/a/notes.md"));
        assert!(!admits(&f, "/a/drafts/x/notes.md"));
        assert!(!admits(&f, "/a/CHANGELOG.md"));
        assert!(!f.admits_dir(Path::new("/a/drafts")));
        assert!(f.admits_dir(Path::new("/a/src")));
    }

    #[test]
    fn paths_with_slashes() {
        let f = filter(&["docs/*.md", "/abs/only"], &[]);
        assert!(admits(&f, "/a/docs/x.md"));
        assert!(!admits(&f, "/a/src/x.md"));
        assert!(admits(&f, "/abs/only/x.rs"));
        assert!(!admits(&f, "/a/abs/only/x.rs"));
    }

    #[test]
    fn rejects_negation() {
        assert!(Filter::new(&["!x".into()], &[]).is_err());
    }

    #[test]
    fn cover_rules_differ_for_index_and_auto_index() {
        let empty = Filter::default();
        assert!(empty.covers_index(None));
        assert!(empty.covers_auto_index(None));
        assert!(!empty.covers_index(Some(1)));
        assert!(!empty.covers_auto_index(Some(1)));

        let exclude_only = filter(&[], &["*.lock"]);
        assert!(exclude_only.covers_index(None));
        assert!(!exclude_only.covers_auto_index(None));

        let include = filter(&["*.md"], &[]);
        assert!(!include.covers_index(None));
        assert!(!include.covers_auto_index(None));
    }
}
