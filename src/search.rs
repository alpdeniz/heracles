use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};

use crate::cli::{Scope, SearchArgs, SearchMode};
use crate::config::{Config, Home};
use crate::embed::Embedder;
use crate::filter::Filter;
use crate::grep;
use crate::lock::MemoryLock;
use crate::memory::{Chunk, FileEntry, Memory, Stamp, Vector};

/// Lines of each hit shown under its location.
const SNIPPET_LINES: usize = 3;
const SNIPPET_WIDTH: usize = 120;

pub fn run(home: &Home, config: &Config, args: &SearchArgs) -> Result<()> {
    let Scope { paths, filter } = &args.scope;
    let filter = filter.build()?;
    let depth = args.depth.get();
    announce(args, paths, depth);
    match args.mode {
        SearchMode::Grep => run_grep(config, args, paths, &filter, depth),
        SearchMode::Semantic => {
            let _lock = MemoryLock::acquire(home)?;
            run_semantic(home, config, args, paths, &filter, depth)
        }
        SearchMode::Hybrid => {
            // Grep the live tree before taking the memory lock so an index can proceed.
            let grep_hits = grep::search(
                &args.query,
                paths,
                &filter,
                depth,
                usize::MAX,
                config.index.respect_gitignore,
            )?;
            let _lock = MemoryLock::acquire(home)?;
            run_hybrid(home, config, args, paths, &filter, depth, &grep_hits)
        }
    }
}

fn run_grep(
    config: &Config,
    args: &SearchArgs,
    paths: &[PathBuf],
    filter: &Filter,
    depth: Option<usize>,
) -> Result<()> {
    let hits = grep::search(
        &args.query,
        paths,
        filter,
        depth,
        args.limit.get(),
        config.index.respect_gitignore,
    )?;
    ensure!(!hits.is_empty(), "no matches");
    for hit in &hits {
        print_grep_hit(hit, 1.0);
    }
    Ok(())
}

fn run_hybrid(
    home: &Home,
    config: &Config,
    args: &SearchArgs,
    paths: &[PathBuf],
    filter: &Filter,
    depth: Option<usize>,
    grep_hits: &[grep::Hit],
) -> Result<()> {
    let weight = config.search.grep_weight;
    let lines = grep::line_index(grep_hits);
    let mut memory = Memory::load(home, config)?;
    refresh_index(home, config, &mut memory, paths, filter, depth)?;
    let files: Vec<_> = memory.select(paths, filter).collect();
    let limit = args.limit.get();

    let mut ranked = if files.iter().any(|(_, entry)| !entry.chunks.is_empty()) {
        let query = embed_query(config, &args.query)?;
        let mapped = memory.map(home)?;
        rank(&files, limit, |file, chunk| {
            hybrid_score(
                mapped.score(chunk.slot, &query),
                chunk_overlaps_grep(file, chunk, &lines),
                weight,
            )
        })
    } else {
        Vec::new()
    };

    let results = merge_hybrid(&mut ranked, grep_hits, limit, weight);
    ensure!(!results.is_empty(), "no matches");

    let mut texts: HashMap<&Path, Option<String>> = HashMap::new();
    for result in &results {
        match result {
            HybridHit::Chunk(hit) => {
                let text = texts
                    .entry(hit.file)
                    .or_insert_with(|| crate::extract::text(hit.file).ok().flatten());
                print_hit(hit, text.as_deref());
            }
            HybridHit::Line { hit, score } => print_grep_hit(hit, *score),
        }
    }
    Ok(())
}

fn run_semantic(
    home: &Home,
    config: &Config,
    args: &SearchArgs,
    paths: &[PathBuf],
    filter: &Filter,
    depth: Option<usize>,
) -> Result<()> {
    let mut memory = Memory::load(home, config)?;
    refresh_index(home, config, &mut memory, paths, filter, depth)?;
    let files: Vec<_> = memory.select(paths, filter).collect();
    ensure!(
        files.iter().any(|(_, entry)| !entry.chunks.is_empty()),
        "nothing indexed{}; run `heracles index <folder>` first",
        if paths.is_empty() && filter.is_empty() {
            ""
        } else {
            " matching the given paths and patterns"
        }
    );

    let query = embed_query(config, &args.query)?;
    let mapped = memory.map(home)?;
    let hits = rank(&files, args.limit.get(), |_, chunk| {
        mapped.score(chunk.slot, &query)
    });
    // One read per file: several hits in a PDF would otherwise parse it again each time.
    let mut texts: HashMap<&Path, Option<String>> = HashMap::new();
    for hit in &hits {
        let text = texts
            .entry(hit.file)
            .or_insert_with(|| crate::extract::text(hit.file).ok().flatten());
        print_hit(hit, text.as_deref());
    }
    Ok(())
}

fn embed_query(config: &Config, query: &str) -> Result<Vector> {
    let text = format!("{}{query}", config.embedding.query_prefix);
    let mut embeddings = Embedder::new(&config.embedding).embed(&[text])?;
    ensure!(
        embeddings.len() == 1,
        "embedding server returned {} embeddings for 1 query",
        embeddings.len()
    );
    Ok(Vector::from_unit(&embeddings.pop().unwrap()))
}

/// Weighted blend of cosine similarity and a 0/1 exact-match flag.
fn hybrid_score(semantic: f32, grep_overlap: bool, weight: f32) -> f32 {
    let g = if grep_overlap { 1.0 } else { 0.0 };
    (1.0 - weight) * semantic + weight * g
}

fn chunk_overlaps_grep(file: &Path, chunk: &Chunk, lines: &HashMap<&Path, HashSet<usize>>) -> bool {
    lines
        .get(file)
        .is_some_and(|ls| (chunk.start_line..=chunk.end_line).any(|l| ls.contains(&l)))
}

/// Prints ranking scope and auto-index / ripgrep walk parameters.
/// Ranking uses every indexed file unless `--path` is set; `--depth` only limits the walk
/// (auto-index for semantic/hybrid, ripgrep for grep/hybrid), not which indexed files rank.
fn announce(args: &SearchArgs, paths: &[PathBuf], depth: Option<usize>) {
    let (search, auto_index) = if paths.is_empty() {
        match args.mode {
            SearchMode::Grep => (".".to_owned(), ".".to_owned()),
            _ => ("all indexed".to_owned(), ".".to_owned()),
        }
    } else {
        let joined = paths
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(", ");
        (joined.clone(), joined)
    };
    let depth = depth.map_or("unlimited".to_owned(), |d| d.to_string());
    let mode = match args.mode {
        SearchMode::Semantic => "semantic",
        SearchMode::Grep => "grep",
        SearchMode::Hybrid => "hybrid",
    };
    println!("query:  {}", args.query);
    println!("mode:   {mode}");
    println!("search: {search}");
    if !matches!(args.mode, SearchMode::Grep) {
        println!("index:  {auto_index}");
    }
    println!("walk:   {depth}");
    println!("limit:  {}", args.limit);
    let include = &args.scope.filter.include;
    if !include.is_empty() {
        println!("include: {}", include.join(", "));
    }
    let exclude = &args.scope.filter.exclude;
    if !exclude.is_empty() {
        println!("exclude: {}", exclude.join(", "));
    }
    println!();
}

/// Indexes or refreshes `paths` — the current directory when none were given.
/// Covered folders are walked again so new and changed files enter the index; stamp checks
/// keep that cheap when nothing changed. A filtered or depth-limited pass does not record a
/// root.
fn refresh_index(
    home: &Home,
    config: &Config,
    memory: &mut Memory,
    paths: &[PathBuf],
    filter: &Filter,
    depth: Option<usize>,
) -> Result<()> {
    let folders = if paths.is_empty() {
        vec![fs::canonicalize(".").context("could not resolve the current directory")?]
    } else {
        paths.to_vec()
    };
    for folder in &folders {
        let cover = filter.covers_auto_index(depth);
        let flags = crate::index::Flags::search(cover, memory.is_indexed(folder));
        crate::index::run(home, config, memory, folder, filter, depth, flags)?;
    }
    Ok(())
}

struct Hit<'a> {
    file: &'a Path,
    entry: &'a FileEntry,
    chunk: &'a Chunk,
    score: f32,
}

enum HybridHit<'a> {
    Chunk(Hit<'a>),
    Line { hit: &'a grep::Hit, score: f32 },
}

/// Ranked chunks plus grep hits whose lines are not covered by any selected chunk, best first.
fn merge_hybrid<'a>(
    ranked: &mut Vec<Hit<'a>>,
    grep_hits: &'a [grep::Hit],
    limit: usize,
    weight: f32,
) -> Vec<HybridHit<'a>> {
    let mut out: Vec<HybridHit<'a>> = ranked.drain(..).map(HybridHit::Chunk).collect();
    for hit in grep_hits {
        if out.len() >= limit {
            break;
        }
        let covered = out.iter().any(|r| match r {
            HybridHit::Chunk(c) => {
                c.file == hit.file.as_path()
                    && (c.chunk.start_line..=c.chunk.end_line).contains(&hit.line)
            }
            HybridHit::Line { .. } => false,
        });
        if !covered {
            // No semantic score for this line: pure grep weight.
            out.push(HybridHit::Line { hit, score: weight });
        }
    }
    out.sort_unstable_by(|a, b| b.score().total_cmp(&a.score()));
    if out.len() > limit {
        out.truncate(limit);
    }
    out
}

impl HybridHit<'_> {
    fn score(&self) -> f32 {
        match self {
            HybridHit::Chunk(h) => h.score,
            HybridHit::Line { score, .. } => *score,
        }
    }
}

/// The `limit` chunks most similar to `query`, best first. A chunk overlapping a
/// better-scoring chunk of the same file is dropped, since it shows mostly the same lines.
fn rank<'a>(
    files: &[(&'a PathBuf, &'a FileEntry)],
    limit: usize,
    mut score: impl FnMut(&'a Path, &Chunk) -> f32,
) -> Vec<Hit<'a>> {
    let mut candidates = Vec::new();
    for &(file, entry) in files {
        for chunk in &entry.chunks {
            candidates.push(Hit {
                file,
                entry,
                chunk,
                score: score(file, chunk),
            });
        }
    }
    candidates.sort_unstable_by(|a, b| b.score.total_cmp(&a.score));

    let mut hits: Vec<Hit> = Vec::with_capacity(limit);
    for candidate in candidates {
        if hits.len() == limit {
            break;
        }
        let covered = hits
            .iter()
            .any(|hit| hit.file == candidate.file && overlaps(hit.chunk, candidate.chunk));
        if !covered {
            hits.push(candidate);
        }
    }
    hits
}

fn overlaps(a: &Chunk, b: &Chunk) -> bool {
    a.start_line <= b.end_line && b.start_line <= a.end_line
}

fn print_grep_hit(hit: &grep::Hit, score: f32) {
    println!("{score:.3}  {}:{}", hit.file.display(), hit.line);
    let line = hit.text.trim();
    if !line.is_empty() {
        println!("    {}", truncate(line, SNIPPET_WIDTH));
    }
    println!();
}

fn print_hit(hit: &Hit, text: Option<&str>) {
    let &Chunk {
        start_line,
        end_line,
        ..
    } = hit.chunk;
    let note = match fs::metadata(hit.file) {
        Err(_) => " (missing)",
        Ok(m) if Stamp::of(&m).ok() != hit.entry.stamp => " (changed since indexing)",
        Ok(_) => "",
    };
    println!(
        "{:.3}  {}:{start_line}-{end_line}{note}",
        hit.score,
        hit.file.display()
    );

    if let Some(text) = text {
        text.lines()
            .skip(start_line - 1)
            .take(end_line + 1 - start_line)
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .take(SNIPPET_LINES)
            .for_each(|line| println!("    {}", truncate(line, SNIPPET_WIDTH)));
    }
    println!();
}

fn truncate(line: &str, width: usize) -> String {
    match line.char_indices().nth(width) {
        Some((i, _)) => format!("{}…", &line[..i]),
        None => line.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(vectors: &mut Vec<Vector>, chunks: &[(usize, usize, &[f32])]) -> FileEntry {
        FileEntry {
            stamp: None,
            chunks: chunks
                .iter()
                .map(|&(start_line, end_line, vector)| {
                    let slot = vectors.len() as u32;
                    vectors.push(Vector::from_unit(vector));
                    Chunk {
                        offset: start_line,
                        start_line,
                        end_line,
                        hash: 0,
                        slot,
                    }
                })
                .collect(),
        }
    }

    #[test]
    fn ranks_by_similarity_and_drops_overlaps() {
        let (a, b) = (PathBuf::from("/a"), PathBuf::from("/b"));
        // /a: two overlapping chunks, both close to the query; /b: one distant chunk.
        let mut vectors = Vec::new();
        let ea = entry(&mut vectors, &[(1, 10, &[1.0, 0.0]), (5, 15, &[0.8, 0.6])]);
        let eb = entry(&mut vectors, &[(1, 10, &[0.0, 1.0])]);
        let files = [(&a, &ea), (&b, &eb)];
        let query = Vector::from_unit(&[1.0, 0.0]);

        let hits = rank(&files, 10, |_, chunk| {
            vectors[chunk.slot as usize].cosine(&query)
        });
        let got: Vec<_> = hits
            .iter()
            .map(|h| (h.file.to_str().unwrap(), h.chunk.start_line))
            .collect();
        assert_eq!(got, vec![("/a", 1), ("/b", 1)]);
    }

    #[test]
    fn respects_limit() {
        let a = PathBuf::from("/a");
        let mut vectors = Vec::new();
        let ea = entry(
            &mut vectors,
            &[(1, 2, &[1.0]), (3, 4, &[0.9]), (5, 6, &[0.8])],
        );
        let query = Vector::from_unit(&[1.0]);
        assert_eq!(
            rank(&[(&a, &ea)], 2, |_, chunk| vectors[chunk.slot as usize]
                .cosine(&query))
            .len(),
            2
        );
    }

    #[test]
    fn hybrid_score_blends_without_exceeding_one() {
        let w = 0.35;
        assert!((hybrid_score(1.0, true, w) - 1.0).abs() < 1e-6);
        assert!((hybrid_score(1.0, false, w) - (1.0 - w)).abs() < 1e-6);
        assert!((hybrid_score(0.0, true, w) - w).abs() < 1e-6);
        // Strong semantic without a grep hit beats a weak semantic with one.
        assert!(hybrid_score(0.9, false, w) > hybrid_score(0.2, true, w));
        // Moderate semantic with a hit beats strong semantic without.
        assert!(hybrid_score(0.5, true, w) > hybrid_score(0.9, false, w));
    }

    #[test]
    fn hybrid_boost_prefers_grep_overlap() {
        let (a, b) = (PathBuf::from("/a"), PathBuf::from("/b"));
        let mut vectors = Vec::new();
        // /b is a better semantic match; /a overlaps a grep hit and should win after blend.
        let ea = entry(&mut vectors, &[(1, 5, &[0.5, 0.0])]);
        let eb = entry(&mut vectors, &[(1, 5, &[1.0, 0.0])]);
        let files = [(&a, &ea), (&b, &eb)];
        let query = Vector::from_unit(&[1.0, 0.0]);
        let mut lines = HashMap::new();
        lines.insert(a.as_path(), HashSet::from([3]));
        let w = 0.35;

        let hits = rank(&files, 10, |file, chunk| {
            hybrid_score(
                vectors[chunk.slot as usize].cosine(&query),
                chunk_overlaps_grep(file, chunk, &lines),
                w,
            )
        });
        assert_eq!(hits[0].file, a.as_path());
        assert_eq!(hits[1].file, b.as_path());
        assert!(hits[0].score <= 1.0);
    }

    #[test]
    fn merge_hybrid_fills_unindexed_grep_hits() {
        let a = PathBuf::from("/a");
        let mut vectors = Vec::new();
        let ea = entry(&mut vectors, &[(1, 5, &[1.0])]);
        let files = [(&a, &ea)];
        let query = Vector::from_unit(&[1.0]);
        let mut ranked = rank(&files, 10, |_, chunk| {
            vectors[chunk.slot as usize].cosine(&query)
        });
        let grep_hits = vec![
            grep::Hit {
                file: PathBuf::from("/a"),
                line: 3, // covered by the chunk
                text: "covered".into(),
            },
            grep::Hit {
                file: PathBuf::from("/b"),
                line: 1,
                text: "only in grep".into(),
            },
        ];
        let w = 0.35;
        let merged = merge_hybrid(&mut ranked, &grep_hits, 10, w);
        assert_eq!(merged.len(), 2);
        assert!(matches!(&merged[0], HybridHit::Chunk(_)));
        assert!(matches!(
            &merged[1],
            HybridHit::Line { hit, score } if hit.file.ends_with("b") && (*score - w).abs() < 1e-6
        ));
    }

    #[test]
    fn truncates_on_char_boundary() {
        assert_eq!(truncate("héllo", 2), "hé…");
        assert_eq!(truncate("hi", 2), "hi");
    }
}
