use std::path::{Path, PathBuf};

#[path = "common/mod.rs"]
mod common;

fn corpus() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/corpus")
}

#[test]
fn grep_mode_finds_literal_without_index() {
    let home = common::temp("heracles-grep");
    let dir = corpus();
    let path = dir.to_str().unwrap();

    common::stdout(&home, &dir, &["init"]);
    let output = common::stdout(
        &home,
        &dir,
        &[
            "search",
            "idempotency",
            "--mode",
            "grep",
            "-p",
            path,
            "-n",
            "5",
        ],
    );
    assert!(
        output.contains("networking.txt"),
        "expected networking.txt in:\n{output}"
    );
    assert!(
        output.contains("idempotency"),
        "expected matching line in:\n{output}"
    );
}

#[test]
fn grep_mode_respects_include() {
    let home = common::temp("heracles-grep-include");
    let dir = corpus();
    let path = dir.to_str().unwrap();

    common::stdout(&home, &dir, &["init"]);
    let output = common::run(
        &home,
        &dir,
        &[
            "search",
            "idempotency",
            "--mode",
            "grep",
            "-p",
            path,
            "--include",
            "*.md",
        ],
    );
    assert!(
        !output.status.success(),
        "include *.md should exclude .txt matches"
    );
}
