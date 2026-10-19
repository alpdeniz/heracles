use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::path::PathBuf;

#[path = "common/mod.rs"]
mod common;

use common::{stdout, temp};

/// Serves `/v1/embeddings` with the same 2-d vector for every input.
fn embedding_server() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}/v1/embeddings", listener.local_addr().unwrap());
    std::thread::spawn(move || {
        for stream in listener.incoming() {
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
            let body = serde_json::json!({ "data": data }).to_string();
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\
                 Connection: close\r\n\r\n{body}",
                body.len()
            )
            .unwrap();
        }
    });
    url
}

fn write_config(home: &std::path::Path, grep_weight: f32, extra_index: &str) {
    std::fs::write(
        home.join("config.toml"),
        format!(
            "[index]\nchunk_tokens = 4\nchunk_overlap_tokens = 0\n{extra_index}\n\
             [search]\ngrep_weight = {grep_weight}\n\n\
             [embedding]\nurl = \"{}\"\ndimensions = 2\nconcurrent_requests = 1\nquery_prefix = \"\"\n",
            embedding_server()
        ),
    )
    .unwrap();
}

fn home_with_weight(name: &str, grep_weight: f32) -> PathBuf {
    let home = temp(name);
    write_config(&home, grep_weight, "");
    home
}

#[test]
fn default_mode_is_hybrid() {
    let home = home_with_weight("hybrid-default-home", 0.35);
    let dir = temp("hybrid-default-dir");
    std::fs::write(dir.join("a.txt"), "line 01\nline 02\n").unwrap();

    let out = stdout(&home, &dir, &["search", "line"]);
    assert!(
        out.contains("mode:   hybrid"),
        "expected hybrid default:\n{out}"
    );
}

#[test]
fn hybrid_prefers_exact_grep_match() {
    let home = home_with_weight("hybrid-boost-home", 0.35);
    let dir = temp("hybrid-boost-dir");
    // Identical embeddings from the mock, so ranking must come from the grep blend.
    std::fs::write(dir.join("exact.txt"), "unique_token lives here\n").unwrap();
    std::fs::write(dir.join("other.txt"), "unrelated content only\n").unwrap();

    let out = stdout(&home, &dir, &["search", "unique_token", "-n", "2"]);
    let exact = out
        .find("exact.txt")
        .unwrap_or_else(|| panic!("exact.txt missing:\n{out}"));
    let other = out
        .find("other.txt")
        .unwrap_or_else(|| panic!("other.txt missing:\n{out}"));
    assert!(exact < other, "exact match should rank first:\n{out}");
}

#[test]
fn grep_weight_one_scores_exact_hit_at_one() {
    let home = home_with_weight("hybrid-weight-home", 1.0);
    let dir = temp("hybrid-weight-dir");
    std::fs::write(dir.join("exact.txt"), "unique_token lives here\n").unwrap();
    std::fs::write(dir.join("other.txt"), "unrelated content only\n").unwrap();

    let out = stdout(&home, &dir, &["search", "unique_token", "-n", "2"]);
    let exact_line = out
        .lines()
        .find(|l| l.contains("exact.txt"))
        .unwrap_or_else(|| panic!("exact.txt missing:\n{out}"));
    assert!(
        exact_line.starts_with("1.000"),
        "grep_weight=1 should score exact hit 1.000:\n{exact_line}\n{out}"
    );
}

#[test]
fn hybrid_fills_grep_hit_when_nothing_is_embedded() {
    let home = temp("hybrid-fill-home");
    // Oversized-everything: index stores no chunks; hybrid still returns live grep hits.
    write_config(&home, 0.35, "max_file_size_kb = 0\n");
    let dir = temp("hybrid-fill-dir");
    std::fs::write(dir.join("live.txt"), "unique_token only in live tree\n").unwrap();

    let out = stdout(&home, &dir, &["search", "unique_token"]);
    assert!(
        out.contains("live.txt") && out.contains("0.350"),
        "expected grep-only fill at weight 0.35:\n{out}"
    );
}
