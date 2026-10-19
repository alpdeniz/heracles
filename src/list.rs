use anyhow::Result;

use crate::cli::Scope;
use crate::config::{Config, Home};
use crate::memory::Memory;

/// Prints the indexed files in `scope`, one per line with its chunk count and size, then
/// totals for them and facts about the memory as a whole.
pub fn run(home: &Home, config: &Config, scope: &Scope) -> Result<()> {
    let filter = scope.filter.build()?;
    let memory = Memory::load(home, config)?;

    let (mut files, mut chunks, mut bytes, mut textless, mut partial) = (0, 0, 0, 0, 0);
    for (file, entry) in memory.select(&scope.paths, &filter) {
        let n = entry.chunks.len();
        let (size, note) = match entry.stamp {
            None => (
                "-".to_owned(),
                "  (partially embedded; index again to finish)",
            ),
            Some(stamp) => {
                bytes += stamp.size;
                (size(stamp.size), if n == 0 { "  (no text)" } else { "" })
            }
        };
        println!("{n:>7}  {size:>10}  {}{note}", file.display());
        files += 1;
        chunks += n;
        textless += usize::from(entry.stamp.is_some() && n == 0);
        partial += usize::from(entry.stamp.is_none());
    }
    if files > 0 {
        println!();
    }

    let notes: Vec<_> = [(textless, "without text"), (partial, "partially embedded")]
        .into_iter()
        .filter(|&(n, _)| n > 0)
        .map(|(n, what)| format!("{n} {what}"))
        .collect();
    let notes = if notes.is_empty() {
        String::new()
    } else {
        format!(" ({})", notes.join(", "))
    };
    println!("Files:     {files}{notes}");
    println!("Chunks:    {chunks}");
    println!("Size:      {}", size(bytes));
    let roots: Vec<_> = memory
        .roots
        .iter()
        .map(|r| r.display().to_string())
        .collect();
    println!(
        "Roots:     {}",
        if roots.is_empty() {
            "none".to_owned()
        } else {
            roots.join(", ")
        }
    );
    let dir = home.memory_dir();
    println!(
        "Memory:    {} ({})",
        dir.display(),
        size(memory.stored_bytes(home))
    );
    println!(
        "Model:     {} ({} dimensions)",
        memory.model, memory.dimensions
    );
    println!(
        "Chunking:  {} tokens, {} overlapping",
        memory.chunking.tokens, memory.chunking.overlap_tokens
    );
    Ok(())
}

fn size(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["KiB", "MiB", "GiB", "TiB"];
    if bytes < 1024 {
        return format!("{bytes} B");
    }
    let mut value = bytes as f64 / 1024.0;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    format!("{value:.1} {}", UNITS[unit])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_are_human_readable() {
        assert_eq!(size(0), "0 B");
        assert_eq!(size(1023), "1023 B");
        assert_eq!(size(1536), "1.5 KiB");
        assert_eq!(size(5 * 1024 * 1024), "5.0 MiB");
    }
}
