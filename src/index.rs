use std::cell::Cell;
use std::collections::{BTreeMap, VecDeque};
use std::fmt;
use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use crate::chunk::{Window, windows};
use crate::config::{Config, Home, IndexConfig};
use crate::embed::Embedder;
use crate::extract;
use crate::filter::Filter;
use crate::interrupt;
use crate::memory::{Chunk, FileEntry, Memory, Stamp, Vector, Vectors, subtree_end};
use anyhow::{Error, Result, anyhow, bail, ensure};

/// How often the status line may redraw; denser embed completions are coalesced.
const PROGRESS_INTERVAL: Duration = Duration::from_millis(100);

/// Chunks sent to the embedding server per request.
const BATCH_SIZE: usize = 32;

/// Chunks from accepted files not yet batched. Caps how far readers run ahead of embeds.
fn max_pending_chunks(workers: usize) -> usize {
    workers
        .saturating_mul(BATCH_SIZE)
        .saturating_mul(4)
        .max(BATCH_SIZE * 4)
}

/// Completed batches waiting for a free embed worker.
fn max_waiting_batches(workers: usize) -> usize {
    workers.saturating_mul(4).max(4)
}

/// Walks `path` (a canonical directory), chunks text files, embeds the chunks and persists them.
/// Chunks are keyed by file and byte offset: unchanged files are not read, and chunks of changed
/// files whose text is unchanged keep their embeddings. Files under `path` that no longer exist
/// are dropped. A file or directory that cannot be read is left as it was. Files `filter`
/// rejects, or more than `depth` levels below `path` (1 being the files directly in it), are
/// neither walked nor dropped.
/// The catalog is rewritten when this pass changed it, including when embedding fails after
/// some chunks were stored. New vectors are written into free slots or appended. An unchanged
/// folder is left on disk as it was. When `flags.cover` is set
/// and nothing was skipped because it could not be read, `path` is recorded as an indexed root.
pub fn run(
    home: &Home,
    config: &Config,
    memory: &mut Memory,
    path: &Path,
    filter: &Filter,
    depth: Option<usize>,
    flags: Flags,
) -> Result<()> {
    interrupt::install();
    let (chunk_bytes, overlap_bytes) = config.index.chunking().budget()?;
    let embedder = Embedder::new(&config.embedding);
    // Headers name files from the indexed folder itself, e.g. `notes/2024/retries.md`.
    let label_base = path.parent().unwrap_or(path);

    let changes = classify(memory, path, &config.index, filter, depth);
    let removed = changes.removed;
    // Walk stopped early: put untouched indexed files back and save any removals.
    if interrupt::requested() {
        for pending in changes.pending {
            if pending.indexed {
                memory.files.insert(pending.file, pending.previous);
            }
        }
        if removed > 0 {
            memory.save(home)?;
        }
        bail!("interrupted");
    }
    let progress = Progress::new();
    let pass = embed_pending(
        home,
        memory,
        changes.pending,
        label_base,
        chunk_bytes,
        overlap_bytes,
        &embedder,
        config.index.read_workers,
        config.embedding.concurrent_requests,
        &progress,
    )?;
    drop(progress);

    let reused = pass.reused;
    let changed_files = pass.changed.len();
    let chunks_done = pass.chunks_done;
    let incomplete = changes.incomplete || pass.incomplete;
    let unchanged = changes.unchanged + pass.kept.len();
    memory.files.extend(pass.kept);
    memory
        .files
        .extend(pass.changed.into_iter().map(|c| (c.file, c.entry)));
    let recorded = if pass.error.is_none() && flags.cover && !incomplete {
        memory.record_root(path)
    } else {
        false
    };
    // Embeddings already on disk stay there when this pass changed nothing.
    if changes.removed > 0 || changed_files > 0 || recorded {
        memory.save(home)?;
    }
    if let Some(error) = pass.error {
        return Err(match chunks_done {
            0 => error,
            n => error.context(format!(
                "{n} chunks embedded before the failure are saved; index again to resume"
            )),
        });
    }

    let changed = changed_files > 0 || removed > 0 || recorded || incomplete;
    if changed || !flags.quiet {
        println!(
            "Indexed {}: {changed_files} changed files ({chunks_done} chunks embedded, {reused} \
             reused), {unchanged} unchanged, {removed} removed{}{}",
            path.display(),
            if chunks_done > 0 {
                format!(" · {:.1} chunks/s", rate(chunks_done, pass.started))
            } else {
                String::new()
            },
            if incomplete {
                "; some files could not be read and were left as they were"
            } else {
                ""
            }
        );
    }
    Ok(())
}

/// Whether to record coverage and whether to print a no-op summary.
#[derive(Clone, Copy)]
pub struct Flags {
    pub cover: bool,
    /// When set, skip the summary line if this pass changed nothing (search refresh).
    pub quiet: bool,
}

impl Flags {
    pub fn index(cover: bool) -> Self {
        Self {
            cover,
            quiet: false,
        }
    }

    pub fn search(cover: bool, already_covered: bool) -> Self {
        Self {
            cover,
            quiet: already_covered,
        }
    }
}

/// What a walk changes relative to the files already in memory.
struct Changes {
    pending: Vec<PendingFile>,
    unchanged: usize,
    removed: usize,
    /// A path in scope could not be read, so this pass may have missed a file.
    incomplete: bool,
}

/// A file on disk whose stamp differs from memory, or that memory has not seen.
struct PendingFile {
    file: PathBuf,
    stamp: Stamp,
    previous: FileEntry,
    /// Memory already had this file. A failed re-read puts `previous` back.
    indexed: bool,
}

/// A changed file whose chunks are being embedded.
struct Changed {
    file: PathBuf,
    /// The file as named in chunk headers and progress.
    label: String,
    stamp: Stamp,
    /// Chunks not embedded yet.
    remaining: usize,
    /// This file was not fully chunked, so a finished batch must not stamp it.
    abort: bool,
    /// Its chunks embedded so far; `stamp` is set once none remain.
    entry: FileEntry,
}

struct Place {
    file: usize,
    chunk: Chunk,
}

struct Batch {
    texts: Vec<String>,
    places: Vec<Place>,
}

struct Done {
    places: Vec<Place>,
    vectors: Vec<Vector>,
}

struct Pass {
    changed: Vec<Changed>,
    reused: usize,
    kept: Vec<(PathBuf, FileEntry)>,
    incomplete: bool,
    chunks_done: usize,
    started: Instant,
    error: Option<Error>,
}

/// Removes the entries this index may refresh or forget: those under `path`, no deeper than
/// `depth`, that `filter` admits. Everything else stays in `files`; the map is not rebuilt.
fn detach_in_scope(
    files: &mut BTreeMap<PathBuf, FileEntry>,
    path: &Path,
    filter: &Filter,
    depth: Option<usize>,
) -> BTreeMap<PathBuf, FileEntry> {
    let admit = |file: &PathBuf, _: &mut FileEntry| in_scope(file, path, filter, depth);
    match subtree_end(path) {
        Some(end) => files.extract_if(path.to_path_buf()..end, admit).collect(),
        None => files.extract_if(.., admit).collect(),
    }
}

fn in_scope(file: &Path, root: &Path, filter: &Filter, depth: Option<usize>) -> bool {
    file.strip_prefix(root)
        .is_ok_and(|rel| depth.is_none_or(|levels| rel.components().count() <= levels))
        && filter.admits(file)
}

/// Walks `path`, putting unchanged files back and returning the ones to read.
///
/// A path whose metadata cannot be read stays indexed. Leftovers are deleted only when a later stat
/// confirms they are gone; an unreadable directory therefore does not forget the files under it.
fn classify(
    memory: &mut Memory,
    path: &Path,
    index: &IndexConfig,
    filter: &Filter,
    depth: Option<usize>,
) -> Changes {
    let mut previous = detach_in_scope(&mut memory.files, path, filter, depth);
    let mut pending = Vec::new();
    let mut unchanged = 0;
    let mut removed = 0;
    let mut incomplete = false;

    let mut walk = crate::filter::walker(path, depth, index.respect_gitignore);
    filter.filter_walk(&mut walk);

    for entry in walk.build() {
        if interrupt::requested() {
            // Not a full walk: keep indexed files we have not seen yet.
            incomplete = true;
            break;
        }
        let entry = match entry {
            Ok(entry) => entry,
            Err(e) => {
                eprintln!("warning: {e}");
                incomplete = true;
                continue;
            }
        };
        let file = entry.path().to_path_buf();
        let metadata = match entry.metadata() {
            Ok(metadata) => metadata,
            Err(e) => {
                eprintln!("warning: {e}");
                incomplete = true;
                continue;
            }
        };
        let limit = size_limit(&file, index);
        if !metadata.is_file() || metadata.len() > limit {
            if let Some(entry) = previous.remove(&file) {
                memory.release_entry(entry);
                removed += 1;
            }
            continue;
        }
        let stamp = match Stamp::of(&metadata) {
            Ok(stamp) => stamp,
            Err(e) => {
                eprintln!("warning: {}: {e}", file.display());
                incomplete = true;
                continue;
            }
        };
        match previous.remove(&file) {
            Some(entry) if entry.stamp == Some(stamp) => {
                memory.files.insert(file, entry);
                unchanged += 1;
            }
            entry => pending.push(PendingFile {
                indexed: entry.is_some(),
                file,
                stamp,
                previous: entry.unwrap_or_default(),
            }),
        }
    }

    if incomplete {
        for (file, entry) in previous {
            if gone(&file, index) {
                memory.release_entry(entry);
                removed += 1;
            } else {
                memory.files.insert(file, entry);
                unchanged += 1;
            }
        }
    } else {
        removed += previous.len();
        for entry in previous.into_values() {
            memory.release_entry(entry);
        }
    }

    Changes {
        pending,
        unchanged,
        removed,
        incomplete,
    }
}

fn size_limit(file: &Path, index: &IndexConfig) -> u64 {
    let kb = if extract::is_pdf(file) {
        index.max_pdf_size_kb
    } else {
        index.max_file_size_kb
    };
    kb * 1024
}

/// Whether `file` is no longer an indexable file. A stat that fails for any other reason
/// (for example a directory we are not allowed to search) keeps the indexed copy.
fn gone(file: &Path, index: &IndexConfig) -> bool {
    match std::fs::metadata(file) {
        Err(e) if e.kind() == io::ErrorKind::NotFound => true,
        Err(e) => {
            eprintln!("warning: {}: {e}", file.display());
            false
        }
        Ok(metadata) => !metadata.is_file() || metadata.len() > size_limit(file, index),
    }
}

/// Reads pending files on `read_workers` threads, then embeds them a few batches at a time.
/// One queued batch per embed worker, so chunk text in flight stays near
/// `concurrency` times [`BATCH_SIZE`]. Prepared files are bounded by `read_workers`.
#[allow(clippy::too_many_arguments)]
fn embed_pending(
    home: &Home,
    memory: &mut Memory,
    pending: Vec<PendingFile>,
    label_base: &Path,
    chunk_bytes: usize,
    overlap_bytes: usize,
    embedder: &Embedder<'_>,
    read_workers: usize,
    concurrency: usize,
    progress: &Progress,
) -> Result<Pass> {
    let started = Instant::now();
    if pending.is_empty() {
        return Ok(Pass {
            changed: Vec::new(),
            reused: 0,
            kept: Vec::new(),
            incomplete: false,
            chunks_done: 0,
            started,
            error: None,
        });
    }

    let readers = read_workers.max(1);
    let workers = concurrency.max(1);
    thread::scope(|scope| {
        let (res_tx, res_rx) = mpsc::channel();
        let mut txs = Vec::with_capacity(workers);
        for _ in 0..workers {
            let (tx, rx) = mpsc::sync_channel(1);
            let res_tx = res_tx.clone();
            scope.spawn(move || embed_worker(rx, res_tx, embedder));
            txs.push(tx);
        }
        drop(res_tx);

        let (prep_tx, prep_rx) = mpsc::sync_channel(readers);
        let total = pending.len();
        let reading = Arc::new(Mutex::new(String::new()));
        let mut shards: Vec<Vec<PendingFile>> = (0..readers).map(|_| Vec::new()).collect();
        for (i, file) in pending.into_iter().enumerate() {
            shards[i % readers].push(file);
        }
        for shard in shards {
            let prep_tx = prep_tx.clone();
            let reading = Arc::clone(&reading);
            scope.spawn(move || {
                for file in shard {
                    let prepared = prepare(file, label_base, chunk_bytes, overlap_bytes, &reading);
                    if prep_tx.send(prepared).is_err() {
                        break;
                    }
                }
            });
        }
        drop(prep_tx);

        let mut feed = Feed {
            home,
            memory,
            progress,
            reading: &reading,
            total,
            files_seen: 0,
            label: String::new(),
            changed: Vec::new(),
            kept: Vec::new(),
            reused: 0,
            incomplete: false,
            chunks_done: 0,
            started,
            error: None,
            vectors: None,
            texts: Vec::new(),
            places: Vec::new(),
            pending_chunks: VecDeque::new(),
            waiting: VecDeque::new(),
            in_flight: 0,
            turn: 0,
            slots: workers,
            txs,
            res_rx,
        };
        // Accept files and embed on one thread: never block accept on the server, so readers
        // and embed workers keep moving and the status line advances.
        let mut prep_done = false;
        let pending_cap = max_pending_chunks(workers);
        let waiting_cap = max_waiting_batches(workers);
        loop {
            feed.check_interrupt();
            feed.pump()?;
            feed.fill_batches()?;

            let can_accept = !prep_done
                && feed.error.is_none()
                && feed.pending_chunks.len() < pending_cap
                && feed.waiting.len() < waiting_cap;

            if can_accept {
                match prep_rx.recv_timeout(PROGRESS_INTERVAL) {
                    Ok(prepared) => {
                        if feed.error.is_some() {
                            if prepared.indexed {
                                feed.kept.push((prepared.file, prepared.previous));
                            }
                        } else {
                            feed.files_seen += 1;
                            feed.label.clone_from(&prepared.label);
                            feed.accept(prepared)?;
                            feed.refresh();
                        }
                    }
                    Err(mpsc::RecvTimeoutError::Timeout) => feed.refresh(),
                    Err(mpsc::RecvTimeoutError::Disconnected) => prep_done = true,
                }
                continue;
            }

            if feed.in_flight > 0 || !feed.waiting.is_empty() || !feed.pending_chunks.is_empty() {
                // Back-pressured or draining: wait for an embed (with progress ticks).
                if feed.in_flight > 0 {
                    feed.recv_one()?;
                } else {
                    feed.pump()?;
                    feed.fill_batches()?;
                    feed.refresh();
                }
                continue;
            }

            if prep_done || feed.error.is_some() {
                break;
            }
            feed.refresh();
        }
        feed.finish()?;
        Ok(Pass {
            changed: feed.changed,
            reused: feed.reused,
            kept: feed.kept,
            incomplete: feed.incomplete,
            chunks_done: feed.chunks_done,
            started: feed.started,
            error: feed.error,
        })
    })
}

fn embed_worker(
    rx: mpsc::Receiver<Batch>,
    tx: mpsc::Sender<std::result::Result<Done, Error>>,
    embedder: &Embedder<'_>,
) {
    while let Ok(batch) = rx.recv() {
        let msg = match embedder.embed(&batch.texts) {
            Ok(raw) => Ok(Done {
                places: batch.places,
                vectors: raw.into_iter().map(|v| Vector::from_unit(&v)).collect(),
            }),
            Err(e) => Err(e),
        };
        let failed = msg.is_err();
        if tx.send(msg).is_err() {
            break;
        }
        if failed {
            break;
        }
    }
}

/// A file read and chunked off the embed thread. `chunks` is `None` when a transient read
/// error should keep the previous entry; `Some` (possibly empty) means the file was obtained.
/// Warnings are printed on the feed thread so they do not race the progress line.
struct Prepared {
    file: PathBuf,
    stamp: Stamp,
    previous: FileEntry,
    indexed: bool,
    label: String,
    chunks: Option<Vec<ReadyChunk>>,
    warning: Option<String>,
}

struct ReadyChunk {
    offset: usize,
    start_line: usize,
    end_line: usize,
    hash: u64,
    input: String,
}

/// Reads and chunks one pending file. Does not touch memory, the embedder, or stderr.
fn prepare(
    pending: PendingFile,
    label_base: &Path,
    chunk_bytes: usize,
    overlap_bytes: usize,
    reading: &Mutex<String>,
) -> Prepared {
    let PendingFile {
        file,
        stamp,
        previous,
        indexed,
    } = pending;
    let label = file
        .strip_prefix(label_base)
        .unwrap_or(&file)
        .display()
        .to_string();
    if let Ok(mut current) = reading.lock() {
        current.clone_from(&label);
    }
    // Binary files and PDFs that fail to convert are remembered without chunks, so they
    // aren't read again until changed. Any other read error keeps the embeddings already
    // stored; forgetting them would drop a good index because of a transient failure.
    let (text, warning) = match extract::text(&file) {
        Ok(text) => (text.unwrap_or_default(), None),
        Err(e) => {
            let warning = Some(format!("{}: {e}", file.display()));
            if e.kind() != io::ErrorKind::InvalidData {
                return Prepared {
                    file,
                    stamp,
                    previous,
                    indexed,
                    label,
                    chunks: None,
                    warning,
                };
            }
            (String::new(), warning)
        }
    };
    let chunks = windows(&text, chunk_bytes, overlap_bytes)
        .into_iter()
        .map(
            |Window {
                 offset,
                 start_line,
                 end_line,
                 text,
             }| {
                let input = format!("{label}:{start_line}-{end_line}\n{text}");
                let hash = fnv1a(input.as_bytes());
                ReadyChunk {
                    offset,
                    start_line,
                    end_line,
                    hash,
                    input,
                }
            },
        )
        .collect();
    Prepared {
        file,
        stamp,
        previous,
        indexed,
        label,
        chunks: Some(chunks),
        warning,
    }
}

/// Applies prepared files and owns memory / embed submission. Embed workers borrow `embedder` only.
struct Feed<'a> {
    home: &'a Home,
    memory: &'a mut Memory,
    progress: &'a Progress,
    reading: &'a Mutex<String>,
    total: usize,
    files_seen: usize,
    label: String,
    changed: Vec<Changed>,
    kept: Vec<(PathBuf, FileEntry)>,
    reused: usize,
    incomplete: bool,
    chunks_done: usize,
    started: Instant,
    error: Option<Error>,
    vectors: Option<Vectors>,
    texts: Vec<String>,
    places: Vec<Place>,
    /// Chunks accepted but not yet in `texts` / `waiting` / flight.
    pending_chunks: VecDeque<(usize, ReadyChunk)>,
    /// Full batches waiting for a free embed worker (never blocks accept).
    waiting: VecDeque<Batch>,
    in_flight: usize,
    turn: usize,
    slots: usize,
    txs: Vec<mpsc::SyncSender<Batch>>,
    res_rx: mpsc::Receiver<std::result::Result<Done, Error>>,
}

impl Feed<'_> {
    fn refresh(&self) {
        let reading = self.reading.lock().map(|g| g.clone()).unwrap_or_default();
        let (verb, name) = if self.in_flight > 0 || !self.waiting.is_empty() {
            ("embedding", self.label.as_str())
        } else if !self.pending_chunks.is_empty() || !self.texts.is_empty() {
            ("", self.label.as_str())
        } else if !reading.is_empty() {
            ("reading", reading.as_str())
        } else {
            ("", self.label.as_str())
        };
        let file = if verb.is_empty() {
            name.to_string()
        } else if name.is_empty() {
            verb.to_string()
        } else {
            format!("{verb} {name}")
        };
        let inflight = if self.in_flight > 0 {
            format!(" · {} in flight", self.in_flight)
        } else {
            String::new()
        };
        if self.chunks_done > 0 {
            self.progress.show(format_args!(
                "Indexing {}/{} files · {} chunks · {:.1}/s{} · {}",
                self.files_seen,
                self.total,
                self.chunks_done,
                rate(self.chunks_done, self.started),
                inflight,
                file,
            ));
        } else {
            self.progress.show(format_args!(
                "Indexing {}/{} files · {} chunks{} · {}",
                self.files_seen, self.total, self.chunks_done, inflight, file,
            ));
        }
    }

    fn check_interrupt(&mut self) {
        if interrupt::requested() && self.error.is_none() {
            self.progress.clear();
            eprintln!("interrupted; saving progress…");
            self.error = Some(anyhow!("interrupted"));
        }
    }

    /// Register a prepared file. Chunks needing embeds go on `pending_chunks` — no server wait.
    fn accept(&mut self, prepared: Prepared) -> Result<()> {
        self.check_interrupt();
        if self.error.is_some() {
            if prepared.indexed {
                self.kept.push((prepared.file, prepared.previous));
            }
            return Ok(());
        }
        let Prepared {
            file,
            stamp,
            mut previous,
            indexed,
            label,
            chunks,
            warning,
        } = prepared;
        if let Some(warning) = warning {
            self.progress.clear();
            eprintln!("warning: {warning}");
        }
        let Some(chunks) = chunks else {
            if indexed {
                self.kept.push((file, previous));
            }
            self.incomplete = true;
            return Ok(());
        };
        self.changed.push(Changed {
            file,
            label,
            stamp,
            remaining: 0,
            abort: false,
            entry: FileEntry::default(),
        });
        let file_index = self.changed.len() - 1;
        for chunk in chunks {
            match previous.take_offset(chunk.offset) {
                Some(old) if old.hash == chunk.hash => {
                    self.changed[file_index].entry.upsert(old);
                    self.reused += 1;
                }
                Some(old) => {
                    self.memory.release(old.slot);
                    self.changed[file_index].remaining += 1;
                    self.pending_chunks.push_back((file_index, chunk));
                }
                None => {
                    self.changed[file_index].remaining += 1;
                    self.pending_chunks.push_back((file_index, chunk));
                }
            }
        }
        for chunk in previous.chunks {
            self.memory.release(chunk.slot);
        }
        if self.changed[file_index].remaining == 0 {
            self.changed[file_index].entry.stamp = Some(stamp);
        }
        Ok(())
    }

    /// Move pending chunks into batches and dispatch to free workers. Never blocks.
    fn fill_batches(&mut self) -> Result<()> {
        if self.error.is_some() {
            return Ok(());
        }
        let waiting_cap = max_waiting_batches(self.slots);
        while self.waiting.len() < waiting_cap {
            if self.texts.len() == BATCH_SIZE {
                self.queue_batch();
                continue;
            }
            let Some((file, chunk)) = self.pending_chunks.pop_front() else {
                break;
            };
            self.texts.push(chunk.input);
            self.places.push(Place {
                file,
                chunk: Chunk {
                    offset: chunk.offset,
                    start_line: chunk.start_line,
                    end_line: chunk.end_line,
                    hash: chunk.hash,
                    slot: 0,
                },
            });
        }
        if self.texts.len() == BATCH_SIZE {
            self.queue_batch();
        }
        self.dispatch_waiting()
    }

    fn queue_batch(&mut self) {
        if self.texts.is_empty() {
            return;
        }
        self.waiting.push_back(Batch {
            texts: std::mem::take(&mut self.texts),
            places: std::mem::take(&mut self.places),
        });
    }

    /// Non-blocking: apply finished embeds and send waiting batches into free slots.
    fn pump(&mut self) -> Result<()> {
        while self.error.is_none() {
            match self.res_rx.try_recv() {
                Ok(Ok(done)) => {
                    self.in_flight -= 1;
                    self.apply(done)?;
                }
                Ok(Err(error)) => {
                    self.in_flight -= 1;
                    self.error = Some(error);
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    if self.in_flight > 0 {
                        self.in_flight = 0;
                        if self.error.is_none() {
                            self.error = Some(anyhow!("embedding workers stopped"));
                        }
                    }
                    break;
                }
            }
        }
        self.dispatch_waiting()
    }

    fn dispatch_waiting(&mut self) -> Result<()> {
        while self.error.is_none()
            && self.in_flight < self.slots
            && !self.waiting.is_empty()
            && !self.txs.is_empty()
        {
            let batch = self.waiting.pop_front().unwrap();
            let turn = self.turn % self.txs.len();
            self.turn += 1;
            if self.txs[turn].send(batch).is_err() {
                self.error = Some(anyhow!("embedding workers stopped"));
                return Ok(());
            }
            self.in_flight += 1;
        }
        Ok(())
    }

    fn recv_one(&mut self) -> Result<()> {
        loop {
            match self.res_rx.recv_timeout(PROGRESS_INTERVAL) {
                Ok(Ok(done)) => {
                    self.in_flight -= 1;
                    self.apply(done)?;
                    return self.dispatch_waiting();
                }
                Ok(Err(error)) => {
                    self.in_flight -= 1;
                    if self.error.is_none() {
                        self.error = Some(error);
                    }
                    return Ok(());
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    self.dispatch_waiting()?;
                    self.refresh();
                    self.check_interrupt();
                    if self.error.is_some() {
                        return Ok(());
                    }
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    self.in_flight = 0;
                    if self.error.is_none() {
                        self.error = Some(anyhow!("embedding workers stopped"));
                    }
                    return Ok(());
                }
            }
        }
    }

    fn apply(&mut self, done: Done) -> Result<()> {
        let Done { places, vectors } = done;
        ensure!(
            places.len() == vectors.len(),
            "embedding server returned {} embeddings for {} chunks",
            vectors.len(),
            places.len()
        );
        let label_at = places.last().map(|place| place.file);
        for (place, vector) in places.into_iter().zip(vectors) {
            // Open before alloc so a brand-new file is created while `slots` is still the
            // length already on disk.
            self.writer()?;
            let slot = self.memory.alloc()?;
            self.vectors.as_ref().unwrap().write(slot, &vector)?;
            let Place { file, mut chunk } = place;
            chunk.slot = slot;
            let changed = &mut self.changed[file];
            changed.entry.upsert(chunk);
            changed.remaining -= 1;
            if changed.remaining == 0 && !changed.abort {
                changed.entry.stamp = Some(changed.stamp);
            }
            self.chunks_done += 1;
        }
        if let Some(file) = label_at {
            self.label.clone_from(&self.changed[file].label);
        }
        self.refresh();
        Ok(())
    }

    fn writer(&mut self) -> Result<()> {
        if self.vectors.is_none() {
            self.vectors = Some(Vectors::open(self.home, self.memory)?);
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        // Drain everything left into workers, then join.
        while self.error.is_none()
            && (!self.pending_chunks.is_empty()
                || !self.texts.is_empty()
                || !self.waiting.is_empty()
                || self.in_flight > 0)
        {
            self.fill_batches()?;
            if !self.pending_chunks.is_empty() || !self.texts.is_empty() || !self.waiting.is_empty()
            {
                // Partial last batch once nothing else is coming.
                if self.pending_chunks.is_empty() && !self.texts.is_empty() {
                    self.queue_batch();
                    self.dispatch_waiting()?;
                }
            }
            if self.in_flight > 0 {
                self.recv_one()?;
            } else if self.error.is_some()
                || (self.pending_chunks.is_empty()
                    && self.texts.is_empty()
                    && self.waiting.is_empty())
            {
                break;
            } else {
                // Waiting to dispatch but no in-flight — should not stick; refresh and retry.
                self.dispatch_waiting()?;
                self.refresh();
            }
        }
        let txs = std::mem::take(&mut self.txs);
        drop(txs);
        while self.in_flight > 0 {
            self.recv_one()?;
        }
        if let Some(store) = &self.vectors {
            store.sync()?;
        }
        Ok(())
    }
}

fn rate(count: usize, since: Instant) -> f64 {
    count as f64 / since.elapsed().as_secs_f64()
}

/// FNV-1a: unlike `std`'s hashers, stable across Rust versions, so stored hashes stay valid.
fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, &b| {
        (hash ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// A status line on stderr, redrawn in place; drawn only when stderr is a terminal and
/// cleared on drop so later output, including errors, starts on a clean line.
struct Progress {
    enabled: bool,
    last: Cell<Option<Instant>>,
}

impl Progress {
    fn new() -> Self {
        Self {
            enabled: io::stderr().is_terminal(),
            last: Cell::new(None),
        }
    }

    /// Autowrap is off while drawing, so a line wider than the terminal is clipped instead of
    /// wrapping onto lines the next redraw would not clear. Redraws are throttled so embed
    /// completions do not thrash the terminal.
    fn show(&self, line: fmt::Arguments) {
        if !self.enabled {
            return;
        }
        let now = Instant::now();
        if let Some(last) = self.last.get()
            && now.duration_since(last) < PROGRESS_INTERVAL
        {
            return;
        }
        self.last.set(Some(now));
        eprint!("\r\x1b[2K\x1b[?7l{line}\x1b[?7h");
        let _ = io::stderr().flush();
    }

    fn clear(&self) {
        if self.enabled {
            eprint!("\r\x1b[2K");
            let _ = io::stderr().flush();
            self.last.set(None);
        }
    }
}

impl Drop for Progress {
    fn drop(&mut self) {
        self.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::fs;
    use std::time::SystemTime;

    use crate::filter::Filter;

    fn paths(map: &BTreeMap<PathBuf, FileEntry>) -> Vec<&str> {
        map.keys().map(|path| path.to_str().unwrap()).collect()
    }

    #[test]
    fn detach_takes_only_the_folder() {
        let mut files = BTreeMap::new();
        for path in ["/notes/a", "/notes/sub/b", "/notes-old/c", "/other/d"] {
            files.insert(PathBuf::from(path), FileEntry::default());
        }
        let taken = detach_in_scope(&mut files, Path::new("/notes"), &Filter::default(), None);
        assert_eq!(paths(&taken), vec!["/notes/a", "/notes/sub/b"]);
        assert_eq!(paths(&files), vec!["/notes-old/c", "/other/d"]);
    }

    #[test]
    fn detach_leaves_files_below_depth() {
        let mut files = BTreeMap::new();
        for path in ["/notes/a", "/notes/sub/b"] {
            files.insert(PathBuf::from(path), FileEntry::default());
        }
        let taken = detach_in_scope(&mut files, Path::new("/notes"), &Filter::default(), Some(1));
        assert_eq!(paths(&taken), vec!["/notes/a"]);
        assert_eq!(paths(&files), vec!["/notes/sub/b"]);
    }

    #[test]
    fn classify_interrupt_keeps_unseen_files() {
        interrupt::clear_for_test();
        let dir = std::env::temp_dir().join(format!(
            "heracles-interrupt-{}",
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        let file = dir.join("a.txt");
        fs::write(&file, "hi").unwrap();
        let stamp = Stamp::of(&fs::metadata(&file).unwrap()).unwrap();
        let mut memory = Memory::default();
        memory.files.insert(
            file.clone(),
            FileEntry {
                stamp: Some(stamp),
                chunks: Vec::new(),
            },
        );
        interrupt::request_for_test();
        let changes = classify(
            &mut memory,
            &dir,
            &IndexConfig::default(),
            &Filter::default(),
            None,
        );
        interrupt::clear_for_test();
        let _ = fs::remove_dir_all(&dir);
        assert!(changes.incomplete);
        assert!(changes.pending.is_empty());
        assert!(memory.files.contains_key(&file));
    }

    #[test]
    fn prepare_chunks_a_text_file() {
        let dir = std::env::temp_dir().join(format!(
            "heracles-prepare-{}",
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        let file = dir.join("notes/a.txt");
        fs::create_dir_all(file.parent().unwrap()).unwrap();
        fs::write(&file, "one\ntwo\nthree\n").unwrap();
        let stamp = Stamp::of(&fs::metadata(&file).unwrap()).unwrap();
        let reading = Mutex::new(String::new());
        let prepared = prepare(
            PendingFile {
                file: file.clone(),
                stamp,
                previous: FileEntry::default(),
                indexed: false,
            },
            dir.as_path(),
            32,
            0,
            &reading,
        );
        assert!(prepared.warning.is_none());
        let _ = fs::remove_dir_all(&dir);
        let chunks = prepared.chunks.expect("text file");
        assert_eq!(prepared.label, "notes/a.txt");
        assert!(!chunks.is_empty());
        assert!(chunks[0].input.starts_with("notes/a.txt:"));
        assert_eq!(chunks[0].hash, fnv1a(chunks[0].input.as_bytes()));
    }
}
