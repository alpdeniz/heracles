use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail, ensure};
use memmap2::Mmap;

use crate::config::{Chunking, Config, Home};
use crate::filter::Filter;

/// Catalog layout. Older on-disk layouts (`catalog.json`, `index.bin`) are discarded.
const FORMAT: u32 = 6;
const MAGIC: &[u8; 4] = b"HCLS";

/// Symmetric int8 range. `-128` is unused so a component and its negation quantize evenly.
const MAGNITUDE: f32 = 127.0;

fn default_vector_file() -> String {
    "vectors.bin".into()
}

fn slot_width(dimensions: usize) -> usize {
    4 + dimensions
}

/// The vector memory. `catalog.bin` holds paths, stamps and a slot per chunk.
/// The vector file is a fixed-width array of `[f32 LE scale][i8 codes]`.
///
/// A save rewrites the catalog and writes only slots that changed. Slots freed
/// in this process stay untouched until that catalog is renamed into place, so
/// a crash keeps the previous vectors. When free slots outnumber live chunks,
/// they are packed into a new vector file and the catalog is published after
/// that file is flushed.
#[derive(Debug, Default)]
pub struct Memory {
    pub model: String,
    pub dimensions: usize,
    /// Keyed by canonical absolute path.
    pub files: BTreeMap<PathBuf, FileEntry>,
    /// Canonical directories indexed in full. A directory is covered when it is one of these
    /// or lies inside one. A pass limited by `--include` or a search `--exclude`
    /// does not record a root.
    pub roots: BTreeSet<PathBuf>,
    /// The walk included dotnames.
    pub indexed_hidden: bool,
    /// What the embeddings were built with.
    pub chunking: Chunking,
    /// Slots the last catalog does not reference. Safe to overwrite before the next rename.
    free: Vec<u32>,
    /// Slots in the vector file, live and free.
    slots: u32,
    vector_file: String,
    /// High-water mark of the catalog on disk. Slots below this are still named by that
    /// catalog until [`save`](Self::save) renames a new one.
    committed: u32,
    /// Freed this process, still named by the catalog on disk. Not reused until save.
    retired: Vec<u32>,
}

#[derive(Debug, Default)]
pub struct FileEntry {
    /// The file version `chunks` were embedded from; `None` while some of its chunks are
    /// missing, e.g. after an interrupted index, so the next index picks the file up again.
    pub stamp: Option<Stamp>,
    /// Sorted by byte offset in the file. Empty for binary and blank files.
    pub chunks: Vec<Chunk>,
}

impl FileEntry {
    pub fn take_offset(&mut self, offset: usize) -> Option<Chunk> {
        match self.chunks.binary_search_by_key(&offset, |c| c.offset) {
            Ok(i) => Some(self.chunks.remove(i)),
            Err(_) => None,
        }
    }

    pub fn upsert(&mut self, chunk: Chunk) {
        match self
            .chunks
            .binary_search_by_key(&chunk.offset, |c| c.offset)
        {
            Ok(i) => self.chunks[i] = chunk,
            Err(i) => self.chunks.insert(i, chunk),
        }
    }
}

/// Detects changed files on re-index without reading them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stamp {
    pub size: u64,
    pub modified: SystemTime,
}

impl Stamp {
    pub fn of(metadata: &fs::Metadata) -> std::io::Result<Self> {
        Ok(Self {
            size: metadata.len(),
            modified: metadata.modified()?,
        })
    }
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct Chunk {
    /// Byte offset in the file.
    pub offset: usize,
    /// 1-based, inclusive.
    pub start_line: usize,
    pub end_line: usize,
    /// Hash of the embedded text, header included; a chunk whose text is unchanged keeps
    /// its vector when its file changes.
    pub hash: u64,
    /// Index into the vector file. Assigned when the chunk is embedded.
    pub slot: u32,
}

/// A unit-length embedding stored as int8. `scale` is its largest absolute component
/// (zero when every component is), and each code is that component divided by `scale`,
/// rounded into `-127..=127`.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct Vector {
    pub scale: f32,
    pub codes: Vec<i8>,
}

impl Vector {
    pub fn from_unit(components: &[f32]) -> Self {
        let scale = components.iter().map(|x| x.abs()).fold(0.0, f32::max);
        let codes = if scale == 0.0 {
            vec![0; components.len()]
        } else {
            components
                .iter()
                .map(|x| (x / scale * MAGNITUDE).round().clamp(-MAGNITUDE, MAGNITUDE) as i8)
                .collect()
        };
        Self { scale, codes }
    }

    /// Approximate cosine similarity of the unit-length embeddings these were quantized from.
    #[cfg(test)]
    pub fn cosine(&self, other: &Self) -> f32 {
        let codes: Vec<u8> = other.codes.iter().map(|&c| c as u8).collect();
        self.cosine_codes(other.scale, &codes)
    }

    fn cosine_codes(&self, scale: f32, codes: &[u8]) -> f32 {
        let dot = dot_i8(&self.codes, codes);
        self.scale * scale * dot as f32 / (MAGNITUDE * MAGNITUDE)
    }
}

fn dot_i8(query: &[i8], codes: &[u8]) -> i32 {
    debug_assert_eq!(query.len(), codes.len());
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { dot_neon(query, codes) };
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { dot_avx2(query, codes) };
        }
    }
    #[allow(unreachable_code)]
    dot_scalar(query, codes)
}

fn dot_scalar(query: &[i8], codes: &[u8]) -> i32 {
    query
        .iter()
        .zip(codes)
        .map(|(&a, &b)| i32::from(a) * i32::from(b as i8))
        .sum()
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_neon(query: &[i8], codes: &[u8]) -> i32 {
    use std::arch::aarch64::*;
    // Edition 2024: unsafe fn body is safe by default; NEON ops need a block.
    unsafe {
        let mut sum = vdupq_n_s32(0);
        let mut i = 0;
        while i + 16 <= query.len() {
            let va = vld1q_s8(query.as_ptr().add(i));
            let vb = vld1q_s8(codes.as_ptr().add(i) as *const i8);
            sum = vpadalq_s16(sum, vmull_s8(vget_low_s8(va), vget_low_s8(vb)));
            sum = vpadalq_s16(sum, vmull_high_s8(va, vb));
            i += 16;
        }
        let mut total = vaddvq_s32(sum);
        while i < query.len() {
            total += i32::from(query[i]) * i32::from(codes[i] as i8);
            i += 1;
        }
        total
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_avx2(query: &[i8], codes: &[u8]) -> i32 {
    use std::arch::x86_64::*;
    unsafe {
        let mut sum = _mm256_setzero_si256();
        let mut i = 0;
        while i + 32 <= query.len() {
            // 16×i8 → i16, multiply-add pairs into i32, twice per 32-byte block.
            let a0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(query.as_ptr().add(i) as *const _));
            let b0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(codes.as_ptr().add(i) as *const _));
            let a1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(query.as_ptr().add(i + 16) as *const _));
            let b1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(codes.as_ptr().add(i + 16) as *const _));
            sum = _mm256_add_epi32(sum, _mm256_madd_epi16(a0, b0));
            sum = _mm256_add_epi32(sum, _mm256_madd_epi16(a1, b1));
            i += 32;
        }
        let hi = _mm256_extracti128_si256::<1>(sum);
        let lo = _mm256_castsi256_si128(sum);
        let sum128 = _mm_add_epi32(lo, hi);
        let shuf = _mm_shuffle_epi32::<0b11_10_11_10>(sum128);
        let sum64 = _mm_add_epi32(sum128, shuf);
        let shuf2 = _mm_shuffle_epi32::<0b01_01_01_01>(sum64);
        let mut total = _mm_cvtsi128_si32(_mm_add_epi32(sum64, shuf2));
        while i < query.len() {
            total += i32::from(query[i]) * i32::from(codes[i] as i8);
            i += 1;
        }
        total
    }
}

/// Read-only view of the vector file. Scoring touches the slots it reads.
pub struct Mapped {
    bytes: Option<Mmap>,
    width: usize,
}

impl Mapped {
    pub fn score(&self, slot: u32, query: &Vector) -> f32 {
        let Some(bytes) = &self.bytes else {
            return 0.0;
        };
        let start = slot as usize * self.width;
        let end = start + self.width;
        if end > bytes.len() || self.width < 4 {
            return 0.0;
        }
        let scale = f32::from_le_bytes([
            bytes[start],
            bytes[start + 1],
            bytes[start + 2],
            bytes[start + 3],
        ]);
        query.cosine_codes(scale, &bytes[start + 4..end])
    }
}

/// Writer for the vector file.
pub struct Vectors {
    file: File,
    width: usize,
}

impl Vectors {
    pub fn open(home: &Home, memory: &Memory) -> Result<Self> {
        let dir = home.memory_dir();
        fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
        let path = dir.join(memory.vector_name());
        let width = slot_width(memory.dimensions);
        let want = memory.slots as u64 * width as u64;
        if !path.exists() && memory.slots > 0 {
            bail!("{} is corrupt; delete it and re-index", path.display());
        }
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .with_context(|| format!("opening {}", path.display()))?;
        let len = file.metadata()?.len();
        if len < want {
            bail!("{} is corrupt; delete it and re-index", path.display());
        }
        if len > want {
            file.set_len(want)
                .with_context(|| format!("truncating {}", path.display()))?;
        }
        Ok(Self { file, width })
    }

    pub fn write(&self, slot: u32, vector: &Vector) -> Result<()> {
        ensure!(
            vector.codes.len() + 4 == self.width,
            "vector has {} components, the memory has {}",
            vector.codes.len(),
            self.width - 4
        );
        let mut buf = Vec::with_capacity(self.width);
        buf.extend(vector.scale.to_le_bytes());
        buf.extend(vector.codes.iter().map(|code| *code as u8));
        write_all_at(&self.file, &buf, slot as u64 * self.width as u64)
    }

    pub fn sync(&self) -> Result<()> {
        self.file.sync_all()?;
        Ok(())
    }
}

impl Memory {
    fn catalog_path(home: &Home) -> PathBuf {
        home.memory_dir().join("catalog.bin")
    }

    fn vector_name(&self) -> &str {
        if self.vector_file.is_empty() {
            "vectors.bin"
        } else {
            &self.vector_file
        }
    }

    pub fn stored_bytes(&self, home: &Home) -> u64 {
        let dir = home.memory_dir();
        file_len(&dir.join("catalog.bin")) + file_len(&dir.join(self.vector_name()))
    }

    fn empty(config: &Config) -> Self {
        Self {
            model: config.embedding.model.clone(),
            dimensions: config.embedding.dimensions,
            chunking: config.index.chunking(),
            indexed_hidden: true,
            vector_file: default_vector_file(),
            ..Self::default()
        }
    }

    /// Loads the catalog, or an empty one if nothing has been indexed yet.
    /// Embeddings stay in the vector file; search maps that file.
    /// A previous `catalog.json` or `index.bin` is discarded.
    pub fn load(home: &Home, config: &Config) -> Result<Self> {
        let dir = home.memory_dir();
        discard_legacy(&dir);
        if Self::catalog_path(home).is_file() {
            let memory = Self::read_catalog(home, config)?;
            sweep(&dir, memory.vector_name());
            Ok(memory)
        } else {
            Ok(Self::empty(config))
        }
    }

    pub fn map(&self, home: &Home) -> Result<Mapped> {
        let width = slot_width(self.dimensions);
        if self.slots == 0 {
            return Ok(Mapped { bytes: None, width });
        }
        let path = home.memory_dir().join(self.vector_name());
        let file = File::open(&path).with_context(|| format!("opening {}", path.display()))?;
        let want = self.slots as usize * width;
        ensure!(
            file.metadata()?.len() as usize >= want,
            "{} is corrupt; delete it and re-index",
            path.display()
        );
        // Read-only for the life of the map. Index writes happen in another run.
        let bytes =
            unsafe { Mmap::map(&file) }.with_context(|| format!("mapping {}", path.display()))?;
        Ok(Mapped {
            bytes: Some(bytes),
            width,
        })
    }

    /// A slot the catalog on disk does not name, or a new slot past its end.
    pub fn alloc(&mut self) -> Result<u32> {
        if let Some(slot) = self.free.pop() {
            return Ok(slot);
        }
        let slot = self.slots;
        self.slots = self
            .slots
            .checked_add(1)
            .context("too many chunks to store")?;
        Ok(slot)
    }

    /// Returns `slot` to the pool. A slot the catalog on disk still names is not
    /// handed out again until [`save`](Self::save).
    pub fn release(&mut self, slot: u32) {
        if slot >= self.committed {
            self.free.push(slot);
        } else {
            self.retired.push(slot);
        }
    }

    pub fn release_entry(&mut self, entry: FileEntry) {
        for chunk in entry.chunks {
            self.release(chunk.slot);
        }
    }

    pub fn save(&mut self, home: &Home) -> Result<()> {
        if self.vector_file.is_empty() {
            self.vector_file = default_vector_file();
        }
        self.free.append(&mut self.retired);
        let previous = if self.free.len() > self.live() {
            let previous = self.vector_file.clone();
            self.compact(home)?;
            Some(previous)
        } else {
            None
        };
        self.fit(home)?;
        self.write_catalog(home)?;
        if let Some(previous) = previous
            && previous != self.vector_file
        {
            let _ = fs::remove_file(home.memory_dir().join(previous));
        }
        self.committed = self.slots;
        Ok(())
    }

    fn live(&self) -> usize {
        self.files.values().map(|file| file.chunks.len()).sum()
    }

    fn read_catalog(home: &Home, config: &Config) -> Result<Self> {
        let path = Self::catalog_path(home);
        let bytes = fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
        let corrupt = || format!("{} is corrupt; delete it and re-index", path.display());
        let mut memory = decode_catalog(&bytes).with_context(corrupt)?;
        let embedding = &config.embedding;
        if memory.model != embedding.model || memory.dimensions != embedding.dimensions {
            bail!(
                "{} was built with {} ({} dimensions) but the config uses {} ({} dimensions); \
                 delete it and re-index",
                path.display(),
                memory.model,
                memory.dimensions,
                embedding.model,
                embedding.dimensions
            );
        }
        if memory.chunking != config.index.chunking() {
            eprintln!(
                "note: {} was chunked with other settings; discarding it, so folders are indexed \
                 again when indexed or searched",
                path.display()
            );
            wipe(&home.memory_dir());
            return Ok(Self::empty(config));
        }
        if !memory.indexed_hidden {
            memory.roots.clear();
            memory.indexed_hidden = true;
        }
        memory.committed = memory.slots;
        if memory.vector_file.is_empty() {
            memory.vector_file = default_vector_file();
        }
        Ok(memory)
    }

    fn compact(&mut self, home: &Home) -> Result<()> {
        let dir = home.memory_dir();
        let new_name = next_vector_name(&self.vector_file);
        let new_path = dir.join(&new_name);
        let width = slot_width(self.dimensions);
        let written = (|| -> Result<()> {
            let out = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&new_path)
                .with_context(|| format!("writing {}", new_path.display()))?;
            let old_path = dir.join(self.vector_name());
            let input = if self.live() == 0 {
                None
            } else {
                Some(
                    File::open(&old_path)
                        .with_context(|| format!("opening {}", old_path.display()))?,
                )
            };
            let mut buf = vec![0u8; width];
            let mut slot = 0u32;
            for file in self.files.values_mut() {
                for chunk in &mut file.chunks {
                    let Some(input) = &input else { break };
                    read_exact_at(input, &mut buf, chunk.slot as u64 * width as u64)
                        .with_context(|| format!("reading {}", old_path.display()))?;
                    write_all_at(&out, &buf, slot as u64 * width as u64)
                        .with_context(|| format!("writing {}", new_path.display()))?;
                    chunk.slot = slot;
                    slot += 1;
                }
            }
            out.sync_all()
                .with_context(|| format!("writing {}", new_path.display()))?;
            self.free.clear();
            self.slots = slot;
            self.vector_file = new_name;
            Ok(())
        })();
        if written.is_err() {
            let _ = fs::remove_file(&new_path);
        }
        written
    }

    fn fit(&self, home: &Home) -> Result<()> {
        let path = home.memory_dir().join(self.vector_name());
        if self.slots == 0 && !path.exists() {
            return Ok(());
        }
        let file = OpenOptions::new()
            .write(true)
            .open(&path)
            .with_context(|| format!("opening {}", path.display()))?;
        file.set_len(self.slots as u64 * slot_width(self.dimensions) as u64)
            .with_context(|| format!("sizing {}", path.display()))?;
        file.sync_all()
            .with_context(|| format!("writing {}", path.display()))?;
        Ok(())
    }

    fn write_catalog(&self, home: &Home) -> Result<()> {
        let dir = home.memory_dir();
        fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
        let path = Self::catalog_path(home);
        let tmp = dir.join("catalog.bin.tmp");
        let bytes = encode_catalog(self);
        let mut file = File::create(&tmp).with_context(|| format!("writing {}", tmp.display()))?;
        file.write_all(&bytes)
            .with_context(|| format!("writing {}", tmp.display()))?;
        file.sync_all()
            .with_context(|| format!("writing {}", tmp.display()))?;
        fs::rename(&tmp, &path).with_context(|| format!("writing {}", path.display()))
    }

    /// Files inside any of `paths` (all files if it is empty) that `filter` admits.
    ///
    /// Each folder is a bounded walk of the path-ordered map. A sibling such as
    /// `/notes-old` sorts next to `/notes` and is skipped by `starts_with`.
    pub fn select<'a>(
        &'a self,
        paths: &'a [PathBuf],
        filter: &'a Filter,
    ) -> impl Iterator<Item = (&'a PathBuf, &'a FileEntry)> + 'a {
        let sources: Vec<Box<dyn Iterator<Item = (&PathBuf, &FileEntry)> + 'a>> =
            if paths.is_empty() {
                vec![Box::new(self.files.iter())]
            } else {
                outermost(paths)
                    .into_iter()
                    .map(|root| -> Box<dyn Iterator<Item = _> + 'a> {
                        let range: Box<dyn Iterator<Item = _> + 'a> = match subtree_end(&root) {
                            Some(end) => Box::new(self.files.range(root.clone()..end)),
                            None => Box::new(self.files.range(root.clone()..)),
                        };
                        Box::new(range.filter(move |(file, _)| file.starts_with(&root)))
                    })
                    .collect()
            };
        sources
            .into_iter()
            .flatten()
            .filter(move |(file, _)| filter.admits(file))
    }

    /// Whether `path` is an indexed root or a directory inside one.
    pub fn is_indexed(&self, path: &Path) -> bool {
        self.roots.iter().any(|root| path.starts_with(root))
    }

    /// Remembers `path` as indexed. A root that already covers `path` is left in place;
    /// roots inside `path` are dropped, since `path` now covers them. Returns whether the
    /// set changed, so an index of an already covered folder can skip rewriting the file.
    pub fn record_root(&mut self, path: &Path) -> bool {
        if self.is_indexed(path) {
            return false;
        }
        self.roots.retain(|root| !root.starts_with(path));
        self.roots.insert(path.to_owned());
        true
    }
}

fn encode_catalog(memory: &Memory) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend(MAGIC);
    put_u32(&mut buf, FORMAT);
    put_u32(&mut buf, memory.dimensions as u32);
    put_u32(&mut buf, memory.slots);
    put_u32(&mut buf, memory.chunking.tokens as u32);
    put_u32(&mut buf, memory.chunking.overlap_tokens as u32);
    buf.push(u8::from(memory.indexed_hidden));
    put_str(&mut buf, &memory.model);
    put_str(&mut buf, memory.vector_name());
    put_u32(&mut buf, memory.roots.len() as u32);
    for root in &memory.roots {
        put_path(&mut buf, root);
    }
    put_u32(&mut buf, memory.free.len() as u32);
    for slot in &memory.free {
        put_u32(&mut buf, *slot);
    }
    put_u32(&mut buf, memory.files.len() as u32);
    for (path, entry) in &memory.files {
        put_path(&mut buf, path);
        match entry.stamp {
            None => buf.push(0),
            Some(stamp) => {
                buf.push(1);
                put_u64(&mut buf, stamp.size);
                let (secs, nsecs) = stamp_parts(stamp.modified);
                put_u64(&mut buf, secs);
                put_u32(&mut buf, nsecs);
            }
        }
        put_u32(&mut buf, entry.chunks.len() as u32);
        for chunk in &entry.chunks {
            put_u64(&mut buf, chunk.offset as u64);
            put_u32(&mut buf, chunk.start_line as u32);
            put_u32(&mut buf, chunk.end_line as u32);
            put_u64(&mut buf, chunk.hash);
            put_u32(&mut buf, chunk.slot);
        }
    }
    buf
}

fn decode_catalog(bytes: &[u8]) -> Result<Memory> {
    let mut r = Reader { bytes, pos: 0 };
    let magic = r.take(4)?;
    ensure!(magic == MAGIC, "bad magic");
    let format = r.u32()?;
    ensure!(format == FORMAT, "unsupported format {format}");
    let dimensions = r.u32()? as usize;
    let slots = r.u32()?;
    let tokens = r.u32()? as usize;
    let overlap_tokens = r.u32()? as usize;
    let indexed_hidden = r.u8()? != 0;
    let model = r.string()?;
    let vector_file = r.string()?;
    let n_roots = r.u32()? as usize;
    let mut roots = BTreeSet::new();
    for _ in 0..n_roots {
        roots.insert(r.path()?);
    }
    let n_free = r.u32()? as usize;
    let mut free = Vec::with_capacity(n_free);
    for _ in 0..n_free {
        free.push(r.u32()?);
    }
    let n_files = r.u32()? as usize;
    let mut files = BTreeMap::new();
    for _ in 0..n_files {
        let path = r.path()?;
        let stamp = match r.u8()? {
            0 => None,
            1 => {
                let size = r.u64()?;
                let secs = r.u64()?;
                let nsecs = r.u32()?;
                Some(Stamp {
                    size,
                    modified: UNIX_EPOCH + Duration::new(secs, nsecs),
                })
            }
            _ => bail!("bad stamp flag"),
        };
        let n_chunks = r.u32()? as usize;
        let mut chunks = Vec::with_capacity(n_chunks);
        for _ in 0..n_chunks {
            chunks.push(Chunk {
                offset: r.u64()? as usize,
                start_line: r.u32()? as usize,
                end_line: r.u32()? as usize,
                hash: r.u64()?,
                slot: r.u32()?,
            });
        }
        files.insert(path, FileEntry { stamp, chunks });
    }
    ensure!(r.pos == bytes.len(), "trailing catalog bytes");
    Ok(Memory {
        model,
        dimensions,
        files,
        roots,
        indexed_hidden,
        chunking: Chunking {
            tokens,
            overlap_tokens,
        },
        free,
        slots,
        vector_file,
        committed: 0,
        retired: Vec::new(),
    })
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8]> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|e| *e <= self.bytes.len())
            .context("truncated catalog")?;
        let slice = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&mut self) -> Result<u64> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn string(&mut self) -> Result<String> {
        let n = self.u32()? as usize;
        let bytes = self.take(n)?;
        String::from_utf8(bytes.to_vec()).context("catalog string is not UTF-8")
    }

    fn path(&mut self) -> Result<PathBuf> {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;
        let n = self.u32()? as usize;
        let bytes = self.take(n)?;
        Ok(PathBuf::from(OsStr::from_bytes(bytes)))
    }
}

fn put_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend(v.to_le_bytes());
}

fn put_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend(v.to_le_bytes());
}

fn put_str(buf: &mut Vec<u8>, s: &str) {
    put_u32(buf, s.len() as u32);
    buf.extend(s.as_bytes());
}

fn put_path(buf: &mut Vec<u8>, path: &Path) {
    use std::os::unix::ffi::OsStrExt;
    let bytes = path.as_os_str().as_bytes();
    put_u32(buf, bytes.len() as u32);
    buf.extend(bytes);
}

fn stamp_parts(time: SystemTime) -> (u64, u32) {
    let d = time.duration_since(UNIX_EPOCH).unwrap_or_default();
    (d.as_secs(), d.subsec_nanos())
}

fn next_vector_name(current: &str) -> String {
    let n = current
        .strip_prefix("vectors.")
        .and_then(|rest| rest.strip_suffix(".bin"))
        .and_then(|n| n.parse::<u32>().ok())
        .unwrap_or(0);
    format!("vectors.{}.bin", n + 1)
}

fn file_len(path: &Path) -> u64 {
    fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

fn discard_legacy(dir: &Path) {
    let legacy = ["catalog.json", "catalog.json.tmp", "index.bin"];
    if !legacy.iter().any(|name| dir.join(name).is_file()) {
        return;
    }
    eprintln!(
        "note: {} has an older heracles layout; discarding it, so folders are indexed again when \
         indexed or searched",
        dir.display()
    );
    wipe(dir);
}

fn wipe(dir: &Path) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        if name == "lock" {
            continue;
        }
        if name == "catalog.bin"
            || name == "catalog.bin.tmp"
            || name == "catalog.json"
            || name == "catalog.json.tmp"
            || name == "index.bin"
            || (name.starts_with("vectors") && name.ends_with(".bin"))
        {
            let _ = fs::remove_file(entry.path());
        }
    }
}

fn sweep(dir: &Path, keep: &str) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        if name == "index.bin"
            || name == "catalog.json"
            || name == "catalog.json.tmp"
            || (name.starts_with("vectors") && name.ends_with(".bin") && name != keep)
        {
            let _ = fs::remove_file(entry.path());
        }
    }
}

fn write_all_at(file: &File, mut buf: &[u8], mut offset: u64) -> Result<()> {
    use std::os::unix::fs::FileExt;
    while !buf.is_empty() {
        let n = file.write_at(buf, offset)?;
        ensure!(n > 0, "write ended early");
        buf = &buf[n..];
        offset += n as u64;
    }
    Ok(())
}

fn read_exact_at(file: &File, buf: &mut [u8], mut offset: u64) -> io::Result<()> {
    use std::os::unix::fs::FileExt;
    let mut start = 0;
    while start < buf.len() {
        let n = file.read_at(&mut buf[start..], offset)?;
        if n == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "truncated vectors",
            ));
        }
        start += n;
        offset += n as u64;
    }
    Ok(())
}

/// Drops a folder that lies inside another, so overlapping `--path`s yield each file once.
fn outermost(paths: &[PathBuf]) -> Vec<PathBuf> {
    let mut kept: Vec<PathBuf> = Vec::new();
    for path in paths {
        if kept.iter().any(|other| path.starts_with(other)) {
            continue;
        }
        kept.retain(|other| !other.starts_with(path));
        kept.push(path.clone());
    }
    kept.sort();
    kept
}

/// Exclusive end of the map range that contains every path under `path`.
///
/// The bound is the next sibling name (`/notes` ends at `/notet`), which also covers a few
/// names that merely sort between the two, such as `/notes-old`. Callers still check
/// `starts_with`. `None` when `path` has no name to increment (`/`), meaning the whole map.
pub(crate) fn subtree_end(path: &Path) -> Option<PathBuf> {
    use std::os::unix::ffi::OsStrExt;
    let name = path.file_name()?.as_bytes();
    if name.is_empty() {
        return None;
    }
    let mut bytes = name.to_vec();
    let i = (0..bytes.len()).rev().find(|&i| bytes[i] < 0xff)?;
    bytes[i] += 1;
    bytes.truncate(i + 1);
    Some(path.parent()?.join(std::ffi::OsStr::from_bytes(&bytes)))
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::config::{EmbeddingConfig, IndexConfig};
    use crate::filter::Filter;

    fn config() -> Config {
        Config {
            index: IndexConfig {
                chunk_tokens: 8,
                chunk_overlap_tokens: 2,
                ..IndexConfig::default()
            },
            embedding: EmbeddingConfig {
                model: "m".into(),
                dimensions: 2,
                ..EmbeddingConfig::default()
            },
            ..Config::default()
        }
    }

    fn scratch(name: &str) -> (PathBuf, Home) {
        let root = std::env::temp_dir().join(format!("heracles-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        (root.clone(), Home::at(root))
    }

    fn blank() -> Memory {
        Memory {
            model: "m".into(),
            dimensions: 2,
            indexed_hidden: true,
            vector_file: default_vector_file(),
            chunking: Chunking {
                tokens: 8,
                overlap_tokens: 2,
            },
            ..Memory::default()
        }
    }

    fn put(memory: &mut Memory, store: &Vectors, vector: &[f32]) -> Chunk {
        let vector = Vector::from_unit(vector);
        let slot = memory.alloc().unwrap();
        store.write(slot, &vector).unwrap();
        Chunk {
            offset: 0,
            start_line: 1,
            end_line: 1,
            hash: slot as u64,
            slot,
        }
    }

    #[test]
    fn save_load_scores_from_the_slot_file() {
        let (root, home) = scratch("roundtrip");
        let mut memory = blank();
        let store = Vectors::open(&home, &memory).unwrap();
        let mut chunk = put(&mut memory, &store, &[1.0, 0.0]);
        chunk.offset = 0;
        memory.files.insert(
            PathBuf::from("/a.txt"),
            FileEntry {
                stamp: Some(Stamp {
                    size: 3,
                    modified: SystemTime::UNIX_EPOCH,
                }),
                chunks: vec![chunk],
            },
        );
        memory.roots.insert(PathBuf::from("/"));
        store.sync().unwrap();
        drop(store);
        memory.save(&home).unwrap();

        let loaded = Memory::load(&home, &config()).unwrap();
        let slot = loaded.files[Path::new("/a.txt")].chunks[0].slot;
        let score = loaded
            .map(&home)
            .unwrap()
            .score(slot, &Vector::from_unit(&[1.0, 0.0]));
        assert!((score - 1.0).abs() < 1e-2);
        assert_eq!(loaded.roots, memory.roots);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn append_leaves_existing_slot_bytes() {
        let (root, home) = scratch("append");
        let mut memory = blank();
        let store = Vectors::open(&home, &memory).unwrap();
        let mut first = put(&mut memory, &store, &[1.0, 0.0]);
        first.offset = 0;
        memory.files.insert(
            PathBuf::from("/a.txt"),
            FileEntry {
                stamp: None,
                chunks: vec![first],
            },
        );
        store.sync().unwrap();
        drop(store);
        memory.save(&home).unwrap();
        let path = home.memory_dir().join("vectors.bin");
        let before = fs::read(&path).unwrap();

        let mut memory = Memory::load(&home, &config()).unwrap();
        let store = Vectors::open(&home, &memory).unwrap();
        let mut second = put(&mut memory, &store, &[0.0, 1.0]);
        second.offset = 4;
        memory
            .files
            .get_mut(Path::new("/a.txt"))
            .unwrap()
            .upsert(second);
        store.sync().unwrap();
        drop(store);
        memory.save(&home).unwrap();

        let after = fs::read(&path).unwrap();
        assert_eq!(&after[..before.len()], &before);
        assert_eq!(after.len(), before.len() + slot_width(2));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn retired_slot_is_reused_on_the_next_load_without_growing() {
        let (root, home) = scratch("reuse");
        let mut memory = blank();
        let store = Vectors::open(&home, &memory).unwrap();
        let mut kept = put(&mut memory, &store, &[1.0, 0.0]);
        kept.offset = 0;
        let mut dropped = put(&mut memory, &store, &[0.0, 1.0]);
        dropped.offset = 4;
        let dropped_slot = dropped.slot;
        memory.files.insert(
            PathBuf::from("/a.txt"),
            FileEntry {
                stamp: None,
                chunks: vec![kept, dropped],
            },
        );
        store.sync().unwrap();
        drop(store);
        memory.save(&home).unwrap();

        let mut memory = Memory::load(&home, &config()).unwrap();
        let dropped = memory
            .files
            .get_mut(Path::new("/a.txt"))
            .unwrap()
            .take_offset(4)
            .unwrap();
        memory.release(dropped.slot);
        assert_ne!(memory.alloc().unwrap(), dropped_slot);
        let mut memory = Memory::load(&home, &config()).unwrap();
        let dropped = memory
            .files
            .get_mut(Path::new("/a.txt"))
            .unwrap()
            .take_offset(4)
            .unwrap();
        memory.release(dropped.slot);
        memory.save(&home).unwrap();

        let path = home.memory_dir().join("vectors.bin");
        let len = fs::metadata(&path).unwrap().len();
        let mut memory = Memory::load(&home, &config()).unwrap();
        let store = Vectors::open(&home, &memory).unwrap();
        let slot = memory.alloc().unwrap();
        assert_eq!(slot, dropped_slot);
        store.write(slot, &Vector::from_unit(&[0.0, 1.0])).unwrap();
        assert_eq!(fs::metadata(&path).unwrap().len(), len);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn compact_packs_when_free_slots_outnumber_live_chunks() {
        let (root, home) = scratch("compact");
        let mut memory = blank();
        let store = Vectors::open(&home, &memory).unwrap();
        let mut chunks = Vec::new();
        for (offset, v) in [
            (0, &[1.0, 0.0][..]),
            (1, &[0.0, 1.0][..]),
            (2, &[1.0, 1.0][..]),
        ] {
            let mut chunk = put(&mut memory, &store, v);
            chunk.offset = offset;
            chunks.push(chunk);
        }
        memory.files.insert(
            PathBuf::from("/a.txt"),
            FileEntry {
                stamp: None,
                chunks,
            },
        );
        store.sync().unwrap();
        drop(store);
        memory.save(&home).unwrap();

        let mut memory = Memory::load(&home, &config()).unwrap();
        let gone: Vec<_> = {
            let file = memory.files.get_mut(Path::new("/a.txt")).unwrap();
            [1, 2]
                .into_iter()
                .map(|offset| file.take_offset(offset).unwrap().slot)
                .collect()
        };
        for slot in gone {
            memory.release(slot);
        }
        memory.save(&home).unwrap();

        let loaded = Memory::load(&home, &config()).unwrap();
        assert_eq!(loaded.slots, 1);
        assert!(loaded.free.is_empty());
        let len = fs::metadata(home.memory_dir().join(loaded.vector_name()))
            .unwrap()
            .len();
        assert_eq!(len, slot_width(2) as u64);
        let slot = loaded.files[Path::new("/a.txt")].chunks[0].slot;
        let score = loaded
            .map(&home)
            .unwrap()
            .score(slot, &Vector::from_unit(&[1.0, 0.0]));
        assert!((score - 1.0).abs() < 1e-2);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn int8_cosine_matches_unit_vectors() {
        let a = Vector::from_unit(&[0.6, -0.8]);
        assert_eq!(a.scale, 0.8);
        assert_eq!(a.codes, vec![95, -127]);
        let parallel = Vector::from_unit(&[0.6, -0.8]);
        assert!((a.cosine(&parallel) - 1.0).abs() < 1e-2);
        let angled = Vector::from_unit(&[1.0, 0.0]);
        let true_cosine = 0.6;
        assert!((a.cosine(&angled) - true_cosine).abs() < 1e-2);
        assert_eq!(a.cosine(&Vector::from_unit(&[0.0, 0.0])), 0.0);
        assert_eq!(
            Vector::from_unit(&[1.0, 0.0]).cosine(&Vector::from_unit(&[-1.0, 0.0])),
            -1.0
        );
    }

    #[test]
    fn simd_dot_matches_scalar_on_wide_vectors() {
        let n = 1024;
        let query: Vec<i8> = (0..n).map(|i| (i % 255) as i8).collect();
        let codes: Vec<u8> = (0..n).map(|i| ((i * 7) % 255) as u8).collect();
        assert_eq!(dot_i8(&query, &codes), dot_scalar(&query, &codes));
    }

    #[test]
    fn select_keeps_path_order_and_skips_siblings() {
        let mut files = BTreeMap::new();
        for path in ["/notes/a", "/notes/sub/b", "/notes-old/c", "/other/d"] {
            files.insert(PathBuf::from(path), FileEntry::default());
        }
        let memory = Memory {
            files,
            ..Memory::default()
        };
        let paths = vec![
            PathBuf::from("/notes/sub"),
            PathBuf::from("/notes"),
            PathBuf::from("/other"),
        ];
        let got: Vec<_> = memory
            .select(&paths, &Filter::default())
            .map(|(path, _)| path.clone())
            .collect();
        assert_eq!(
            got,
            vec![
                PathBuf::from("/notes/a"),
                PathBuf::from("/notes/sub/b"),
                PathBuf::from("/other/d"),
            ]
        );
    }

    #[test]
    fn root_covers_itself_and_descendants() {
        let mut memory = Memory::default();
        memory.record_root(Path::new("/notes"));
        assert!(memory.is_indexed(Path::new("/notes")));
        assert!(memory.is_indexed(Path::new("/notes/sub")));
        assert!(!memory.is_indexed(Path::new("/other")));
        assert!(!memory.is_indexed(Path::new("/notes-old")));
    }

    #[test]
    fn indexing_parent_replaces_child_roots() {
        let mut memory = Memory::default();
        memory.record_root(Path::new("/notes/sub"));
        memory.record_root(Path::new("/notes/other"));
        memory.record_root(Path::new("/notes"));
        assert_eq!(
            memory
                .roots
                .iter()
                .map(PathBuf::as_path)
                .collect::<Vec<_>>(),
            vec![Path::new("/notes")]
        );
    }

    #[test]
    fn indexing_child_keeps_parent_root() {
        let mut memory = Memory::default();
        memory.record_root(Path::new("/notes"));
        memory.record_root(Path::new("/notes/sub"));
        assert_eq!(memory.roots.len(), 1);
    }

    #[test]
    fn legacy_json_catalog_is_discarded() {
        let (root, home) = scratch("legacy-json");
        let dir = home.memory_dir();
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("catalog.json"), b"{}").unwrap();
        let loaded = Memory::load(&home, &config()).unwrap();
        assert!(!dir.join("catalog.json").exists());
        assert!(loaded.files.is_empty());
        let _ = fs::remove_dir_all(root);
    }
}
