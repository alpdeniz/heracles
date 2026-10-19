use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

#[path = "common/mod.rs"]
mod common;

use common::{run, stdout, temp};

/// Serves `/v1/embeddings` with the same 2-dimensional vector for every input, failing every
/// request from the `fail_from`th on.
fn embedding_server(fail_from: usize) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}/v1/embeddings", listener.local_addr().unwrap());
    std::thread::spawn(move || {
        for (n, stream) in listener.incoming().enumerate() {
            let mut stream = stream.unwrap();
            let mut reader = BufReader::new(&stream);
            let mut length = 0;
            loop {
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                if let Some((name, value)) = line.split_once(':')
                    && name.eq_ignore_ascii_case("content-length")
                {
                    length = value.trim().parse().unwrap();
                }
                if line == "\r\n" {
                    break;
                }
            }
            let mut body = vec![0; length];
            reader.read_exact(&mut body).unwrap();
            let request: serde_json::Value = serde_json::from_slice(&body).unwrap();
            let data: Vec<_> = (0..request["input"].as_array().unwrap().len())
                .map(|index| serde_json::json!({ "index": index, "embedding": [1.0, 0.0] }))
                .collect();
            let (status, body) = if n + 1 >= fail_from {
                ("500 Internal Server Error", "down".to_owned())
            } else {
                ("200 OK", serde_json::json!({ "data": data }).to_string())
            };
            write!(
                stream,
                "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\
                 Connection: close\r\n\r\n{body}",
                body.len()
            )
            .unwrap();
        }
    });
    url
}

fn configure(home: &Path, fail_from: usize) {
    std::fs::write(
        home.join("config.toml"),
        format!(
            "[index]\nchunk_tokens = 4\nchunk_overlap_tokens = 0\n\n\
             [embedding]\nurl = \"{}\"\ndimensions = 2\nconcurrent_requests = 1\n",
            embedding_server(fail_from)
        ),
    )
    .unwrap();
}

/// A home whose config embeds through a working `embedding_server`, two 8-byte lines per chunk.
fn home(name: &str) -> PathBuf {
    let home = temp(name);
    configure(&home, usize::MAX);
    home
}

fn lines(range: std::ops::RangeInclusive<u32>) -> String {
    range.map(|i| format!("line {i:02}\n")).collect()
}

#[test]
fn failed_index_keeps_embedded_chunks() {
    let home = temp("resume-home");
    let dir = temp("resume-dir");
    // 40 chunks: a full batch of 32, then a batch of 8 the server rejects.
    std::fs::write(dir.join("big.txt"), lines(1..=80)).unwrap();
    configure(&home, 2);

    let failed = run(&home, &dir, &["index"]);
    let stderr = String::from_utf8_lossy(&failed.stderr);
    assert!(!failed.status.success());
    assert!(
        stderr.contains("32 chunks embedded before the failure"),
        "{stderr}"
    );
    let list = stdout(&home, &dir, &["list"]);
    assert!(
        list.contains("Files:     1 (1 partially embedded)"),
        "{list}"
    );
    assert!(list.contains("Roots:     none"), "{list}");

    configure(&home, usize::MAX);
    let resumed = stdout(&home, &dir, &["index"]);
    assert!(
        resumed.contains("(8 chunks embedded, 32 reused)"),
        "{resumed}"
    );
}

#[test]
fn reindex_embeds_only_changed_chunks() {
    let home = home("reindex-home");
    let dir = temp("reindex-dir");
    let file = dir.join("log.txt");
    std::fs::write(&file, lines(1..=6)).unwrap();
    std::fs::write(dir.join("blob.bin"), b"\0\x01\x02").unwrap();

    let first = stdout(&home, &dir, &["index"]);
    assert!(first.contains("(3 chunks embedded, 0 reused)"), "{first}");

    let again = stdout(&home, &dir, &["index"]);
    assert!(
        again.contains("0 changed files") && again.contains("2 unchanged"),
        "{again}"
    );

    std::fs::write(&file, lines(1..=8)).unwrap();
    let appended = stdout(&home, &dir, &["index"]);
    assert!(
        appended.contains("(1 chunks embedded, 3 reused)"),
        "{appended}"
    );

    let list = stdout(&home, &dir, &["list"]);
    assert!(list.contains("log.txt"), "{list}");
    assert!(list.contains("blob.bin  (no text)"), "{list}");
    assert!(list.contains("Files:     2 (1 without text)"), "{list}");
    assert!(list.contains("Chunks:    4"), "{list}");
}

/// A one-page PDF showing `text` in Helvetica.
fn pdf(text: &str) -> Vec<u8> {
    let stream = format!("BT /F1 12 Tf 72 720 Td ({text}) Tj ET");
    let objects = [
        "<< /Type /Catalog /Pages 2 0 R >>".to_owned(),
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_owned(),
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R \
         /Resources << /Font << /F1 5 0 R >> >> >>"
            .to_owned(),
        format!(
            "<< /Length {} >>\nstream\n{stream}\nendstream",
            stream.len()
        ),
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_owned(),
    ];
    let mut pdf = b"%PDF-1.4\n".to_vec();
    let mut offsets = Vec::new();
    for (i, object) in objects.iter().enumerate() {
        offsets.push(pdf.len());
        pdf.extend(format!("{} 0 obj\n{object}\nendobj\n", i + 1).bytes());
    }
    let xref = pdf.len();
    pdf.extend(format!("xref\n0 {}\n0000000000 65535 f \n", objects.len() + 1).bytes());
    for offset in offsets {
        pdf.extend(format!("{offset:010} 00000 n \n").bytes());
    }
    pdf.extend(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n",
            objects.len() + 1
        )
        .bytes(),
    );
    pdf
}

#[test]
fn indexes_pdfs_as_markdown() {
    let home = home("pdf-home");
    let dir = temp("pdf-dir");
    std::fs::write(
        dir.join("paper.pdf"),
        pdf("Retries use exponential backoff"),
    )
    .unwrap();
    std::fs::write(dir.join("broken.pdf"), b"%PDF-1.4\nnot really a pdf").unwrap();

    let first = run(&home, &dir, &["index"]);
    let stderr = String::from_utf8_lossy(&first.stderr);
    assert!(first.status.success(), "{stderr}");
    assert!(stderr.contains("broken.pdf"), "{stderr}");

    let search = stdout(&home, &dir, &["search", "backoff"]);
    assert!(search.contains("paper.pdf:"), "{search}");
    assert!(search.contains("exponential backoff"), "{search}");

    let again = run(&home, &dir, &["index"]);
    assert!(
        String::from_utf8_lossy(&again.stdout).contains("2 unchanged"),
        "a PDF that failed to convert is not read again until it changes"
    );
    assert!(again.stderr.is_empty(), "{:?}", again.stderr);

    let list = stdout(&home, &dir, &["list"]);
    assert!(list.contains("broken.pdf  (no text)"), "{list}");
}

#[test]
fn depth_limits_walk_and_keeps_deeper_files() {
    let home = home("depth-home");
    let dir = temp("depth-dir");
    std::fs::create_dir_all(dir.join("sub")).unwrap();
    std::fs::write(dir.join("top.txt"), lines(1..=2)).unwrap();
    std::fs::write(dir.join("sub/deep.txt"), lines(1..=2)).unwrap();

    let sub = dir.join("sub");
    stdout(&home, &dir, &["index", sub.to_str().unwrap()]);
    let shallow = stdout(&home, &dir, &["index", "--depth", "1"]);
    assert!(
        shallow.contains("1 changed files") && shallow.contains("0 removed"),
        "{shallow}"
    );

    let list = stdout(&home, &dir, &["list"]);
    assert!(
        list.contains("top.txt") && list.contains("deep.txt"),
        "{list}"
    );
    // Depth limits the walk and leaves deeper files from earlier; it does not cover the parent
    // (a parent root would replace the earlier child root of `sub`).
    assert!(
        list.contains(&format!("Roots:     {}\n", sub.display())),
        "{list}"
    );
}

#[test]
fn depth_limited_index_does_not_cover_the_folder() {
    let home = home("depth-cover-home");
    let dir = temp("depth-cover-dir");
    std::fs::create_dir_all(dir.join("sub")).unwrap();
    std::fs::write(dir.join("top.txt"), lines(1..=2)).unwrap();
    std::fs::write(dir.join("sub/deep.txt"), lines(1..=2)).unwrap();

    let shallow = stdout(&home, &dir, &["index", "--depth", "1"]);
    assert!(shallow.contains("1 changed files"), "{shallow}");

    let list = stdout(&home, &dir, &["list"]);
    assert!(list.contains("top.txt"), "{list}");
    assert!(!list.contains("deep.txt"), "{list}");
    assert!(list.contains("Roots:     none"), "{list}");

    // A later unlimited index fills in what the depth limit skipped.
    let full = stdout(&home, &dir, &["index"]);
    assert!(
        full.contains("1 changed files") && full.contains("1 unchanged"),
        "{full}"
    );
    let list = stdout(&home, &dir, &["list"]);
    assert!(list.contains("deep.txt"), "{list}");
    assert!(
        list.contains(&format!("Roots:     {}\n", dir.display())),
        "{list}"
    );
}

#[test]
fn search_include_indexes_only_matching_files() {
    let home = home("include-home");
    let dir = temp("include-dir");
    std::fs::write(dir.join("notes.md"), lines(1..=2)).unwrap();
    std::fs::write(dir.join("code.rs"), lines(1..=2)).unwrap();

    let search = stdout(&home, &dir, &["search", "line", "--include", "*.md"]);
    assert!(search.contains("(1 chunks embedded"), "{search}");
    assert!(search.contains("notes.md"), "{search}");
    assert!(!search.contains("code.rs"), "{search}");

    let list = stdout(&home, &dir, &["list"]);
    assert!(!list.contains("code.rs"), "{list}");
    assert!(list.contains("Roots:     none"), "{list}");

    // The filtered index didn't cover the folder, so an unfiltered search indexes the rest.
    let search = stdout(&home, &dir, &["search", "line"]);
    assert!(
        search.contains("(1 chunks embedded") && search.contains("1 unchanged"),
        "{search}"
    );
    assert!(search.contains("code.rs"), "{search}");
}

#[test]
fn search_exclude_does_not_cover_the_folder() {
    let home = home("exclude-home");
    let dir = temp("exclude-dir");
    std::fs::write(dir.join("notes.md"), lines(1..=2)).unwrap();
    std::fs::write(dir.join("code.rs"), lines(1..=2)).unwrap();

    let search = stdout(&home, &dir, &["search", "line", "--exclude", "*.rs"]);
    assert!(search.contains("notes.md"), "{search}");
    assert!(!search.contains("code.rs"), "{search}");

    let list = stdout(&home, &dir, &["list"]);
    assert!(!list.contains("code.rs"), "{list}");
    assert!(list.contains("Roots:     none"), "{list}");

    let search = stdout(&home, &dir, &["search", "line"]);
    assert!(search.contains("code.rs"), "{search}");
    let list = stdout(&home, &dir, &["list"]);
    assert!(
        list.contains(&format!("Roots:     {}\n", dir.display())),
        "{list}"
    );
}

#[test]
fn search_refreshes_covered_folder() {
    let home = home("refresh-home");
    let dir = temp("refresh-dir");
    std::fs::write(dir.join("a.txt"), lines(1..=2)).unwrap();
    stdout(&home, &dir, &["index"]);

    std::fs::write(dir.join("b.txt"), lines(1..=2)).unwrap();
    let search = stdout(&home, &dir, &["search", "line"]);
    assert!(
        search.contains("1 changed files") && search.contains("b.txt"),
        "covered folder was not refreshed:\n{search}"
    );
}

#[test]
fn index_exclude_covers_the_folder() {
    let home = home("index-exclude-home");
    let dir = temp("index-exclude-dir");
    std::fs::write(dir.join("notes.md"), lines(1..=2)).unwrap();
    std::fs::write(dir.join("code.rs"), lines(1..=2)).unwrap();

    stdout(&home, &dir, &["index", "--exclude", "*.rs"]);
    let list = stdout(&home, &dir, &["list"]);
    assert!(list.contains("notes.md"), "{list}");
    assert!(!list.contains("code.rs"), "{list}");
    assert!(
        list.contains(&format!("Roots:     {}\n", dir.display())),
        "{list}"
    );
}

#[test]
fn unreadable_directory_is_not_forgotten() {
    let home = home("unreadable-home");
    let dir = temp("unreadable-dir");
    let sub = dir.join("sub");
    std::fs::create_dir(&sub).unwrap();
    std::fs::write(sub.join("secret.txt"), lines(1..=2)).unwrap();
    std::fs::write(dir.join("top.txt"), lines(1..=2)).unwrap();
    stdout(&home, &dir, &["index"]);
    std::fs::remove_file(dir.join("top.txt")).unwrap();

    let _guard = ModeGuard::lock(&sub, 0o000);
    let again = stdout(&home, &dir, &["index"]);
    assert!(again.contains("some files could not be read"), "{again}");
    let list = stdout(&home, &dir, &["list"]);
    assert!(list.contains("secret.txt"), "{list}");
    assert!(!list.contains("top.txt"), "{list}");
}

#[test]
fn incomplete_index_does_not_cover_the_folder() {
    let home = home("incomplete-cover-home");
    let dir = temp("incomplete-cover-dir");
    let sub = dir.join("sub");
    std::fs::create_dir(&sub).unwrap();
    std::fs::write(dir.join("top.txt"), lines(1..=2)).unwrap();
    std::fs::write(sub.join("secret.txt"), lines(1..=2)).unwrap();

    let guard = ModeGuard::lock(&sub, 0o000);
    stdout(&home, &dir, &["index"]);
    let list = stdout(&home, &dir, &["list"]);
    assert!(list.contains("top.txt"), "{list}");
    assert!(!list.contains("secret.txt"), "{list}");
    assert!(list.contains("Roots:     none"), "{list}");
    drop(guard);

    stdout(&home, &dir, &["index"]);
    let list = stdout(&home, &dir, &["list"]);
    assert!(list.contains("secret.txt"), "{list}");
    assert!(list.contains("top.txt"), "{list}");
    assert!(
        list.contains(&format!("Roots:     {}\n", dir.display())),
        "{list}"
    );
}

#[test]
fn failed_reread_keeps_embeddings() {
    let home = home("reread-home");
    let dir = temp("reread-dir");
    let file = dir.join("log.txt");
    std::fs::write(&file, lines(1..=4)).unwrap();
    stdout(&home, &dir, &["index"]);
    let before = stdout(&home, &dir, &["list"]);

    std::fs::OpenOptions::new()
        .write(true)
        .open(&file)
        .unwrap()
        .set_modified(SystemTime::now() + Duration::from_secs(30))
        .unwrap();
    let _guard = ModeGuard::lock(&file, 0o000);
    let again = stdout(&home, &dir, &["index"]);
    assert!(again.contains("0 changed files"), "{again}");
    assert!(again.contains("0 removed"), "{again}");
    assert!(again.contains("some files could not be read"), "{again}");

    let after = stdout(&home, &dir, &["list"]);
    let chunk_line = |list: &str| {
        list.lines()
            .find(|line| line.contains("log.txt"))
            .unwrap()
            .to_owned()
    };
    assert_eq!(chunk_line(&before), chunk_line(&after), "{after}");
}

#[test]
fn unchanged_index_does_not_rewrite_memory() {
    let home = home("rewrite-home");
    let dir = temp("rewrite-dir");
    std::fs::write(dir.join("log.txt"), lines(1..=2)).unwrap();
    stdout(&home, &dir, &["index"]);
    let path = home.join("memory/catalog.bin");
    let inode = std::fs::metadata(&path).unwrap().ino();
    stdout(&home, &dir, &["index"]);
    assert_eq!(std::fs::metadata(&path).unwrap().ino(), inode);
}

#[test]
fn embedding_request_times_out() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}/v1/embeddings", listener.local_addr().unwrap());
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            std::thread::sleep(Duration::from_secs(30));
            drop(stream);
        }
    });

    let home = temp("timeout-home");
    let dir = temp("timeout-dir");
    std::fs::write(dir.join("log.txt"), lines(1..=2)).unwrap();
    std::fs::write(
        home.join("config.toml"),
        format!(
            "[index]\nchunk_tokens = 4\nchunk_overlap_tokens = 0\n\n\
             [embedding]\nurl = \"{url}\"\ndimensions = 2\ntimeout_secs = 1\n"
        ),
    )
    .unwrap();

    let started = std::time::Instant::now();
    let failed = run(&home, &dir, &["index"]);
    assert!(!failed.status.success());
    assert!(
        started.elapsed() < Duration::from_secs(10),
        "hung for {:?}",
        started.elapsed()
    );
    let stderr = String::from_utf8_lossy(&failed.stderr);
    assert!(stderr.contains("gave up after 1s"), "{stderr}");
}

#[test]
fn hidden_files_and_directories_are_indexed() {
    let home = home("hidden-home");
    let dir = temp("hidden-dir");
    std::fs::write(dir.join(".secret.txt"), lines(1..=2)).unwrap();
    let nested = dir.join(".notes");
    std::fs::create_dir(&nested).unwrap();
    std::fs::write(nested.join("inside.txt"), lines(1..=2)).unwrap();

    stdout(&home, &dir, &["index"]);
    let list = stdout(&home, &dir, &["list"]);
    assert!(list.contains(".secret.txt"), "{list}");
    assert!(list.contains("inside.txt"), "{list}");
}

/// Restores `mode` when dropped, including after a panic, so a locked file can be deleted.
struct ModeGuard<'a> {
    path: &'a Path,
    mode: u32,
}

impl<'a> ModeGuard<'a> {
    fn lock(path: &'a Path, mode: u32) -> Self {
        let current = std::fs::metadata(path).unwrap().permissions().mode() & 0o777;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode)).unwrap();
        Self {
            path,
            mode: current,
        }
    }
}

impl Drop for ModeGuard<'_> {
    fn drop(&mut self) {
        let _ = std::fs::set_permissions(self.path, std::fs::Permissions::from_mode(self.mode));
    }
}
