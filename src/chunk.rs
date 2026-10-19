/// A chunk of a text, before embedding.
pub struct Window<'a> {
    /// Byte offset in the text.
    pub offset: usize,
    /// 1-based, inclusive.
    pub start_line: usize,
    pub end_line: usize,
    pub text: &'a str,
}

/// A 1-based line, or part of one, as a byte range of the text.
struct Piece {
    line: usize,
    start: usize,
    end: usize,
}

/// `text`'s lines (each with its newline), lines longer than `size` bytes split at char
/// boundaries. `size` must be at least 4, the longest UTF-8 char.
fn pieces(text: &str, size: usize) -> Vec<Piece> {
    let mut pieces = Vec::new();
    let mut start = 0;
    for (i, line) in text.split_inclusive('\n').enumerate() {
        let end = start + line.len();
        let mut from = start;
        while end - from > size {
            let mut to = from + size;
            while !text.is_char_boundary(to) {
                to -= 1;
            }
            pieces.push(Piece {
                line: i + 1,
                start: from,
                end: to,
            });
            from = to;
        }
        pieces.push(Piece {
            line: i + 1,
            start: from,
            end,
        });
        start = end;
    }
    pieces
}

/// Splits `text` into chunks of at most `size` bytes that end at line breaks (unless a single
/// line exceeds `size`), consecutive chunks sharing up to `overlap` bytes of whole lines.
/// Blank chunks are skipped.
pub fn windows(text: &str, size: usize, overlap: usize) -> Vec<Window<'_>> {
    let pieces = pieces(text, size);
    let mut chunks = Vec::new();
    let mut first = 0;
    while first < pieces.len() {
        let start = pieces[first].start;
        let mut last = first;
        while last + 1 < pieces.len() && pieces[last + 1].end - start <= size {
            last += 1;
        }
        let end = pieces[last].end;
        let slice = &text[start..end];
        if !slice.trim().is_empty() {
            chunks.push(Window {
                offset: start,
                start_line: pieces[first].line,
                end_line: pieces[last].line,
                text: slice,
            });
        }
        if last + 1 == pieces.len() {
            break;
        }
        first = (first + 1..=last)
            .find(|&i| end - pieces[i].start <= overlap)
            .unwrap_or(last + 1);
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spans(text: &str, size: usize, overlap: usize) -> Vec<(usize, usize, &str)> {
        windows(text, size, overlap)
            .into_iter()
            .map(|w| (w.start_line, w.end_line, w.text))
            .collect()
    }

    #[test]
    fn overlapping_windows() {
        assert_eq!(
            spans("a\nb\nc\nd\ne\n", 6, 2),
            vec![(1, 3, "a\nb\nc\n"), (3, 5, "c\nd\ne\n")]
        );
    }

    #[test]
    fn offsets_are_byte_positions() {
        let text = "a\n\n\nbb\ncc\n";
        let offsets: Vec<_> = windows(text, 3, 0).iter().map(|w| w.offset).collect();
        assert_eq!(offsets, vec![0, 4, 7]);
    }

    #[test]
    fn appending_keeps_full_chunks() {
        let before = windows("aaa\nbbb\nccc\n", 8, 0);
        let after = windows("aaa\nbbb\nccc\nddd\n", 8, 0);
        let key = |w: &Window| (w.offset, w.text.to_owned());
        assert_eq!(key(&before[0]), key(&after[0]));
        assert_ne!(key(&before[1]), key(&after[1]), "the last chunk grows");
    }

    #[test]
    fn overlap_keeps_only_whole_lines_that_fit() {
        assert_eq!(
            spans("aaa\nbbb\nccc\n", 8, 3),
            vec![(1, 2, "aaa\nbbb\n"), (3, 3, "ccc\n")]
        );
    }

    #[test]
    fn last_line_without_newline() {
        assert_eq!(spans("a\nb", 5, 0), vec![(1, 2, "a\nb")]);
    }

    #[test]
    fn skips_empty_and_blank() {
        assert!(spans("", 4, 1).is_empty());
        assert_eq!(spans("\n\n\nx\n", 4, 0), vec![(4, 4, "x\n")]);
    }

    #[test]
    fn splits_long_lines_on_char_boundaries() {
        assert_eq!(
            spans("ab\néééé\n", 5, 0),
            vec![(1, 1, "ab\n"), (2, 2, "éé"), (2, 2, "éé\n")]
        );
    }
}
