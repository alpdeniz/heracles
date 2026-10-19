#[path = "common/mod.rs"]
mod common;

use common::{run, temp};

#[test]
fn search_indexes_unindexed_cwd_once() {
    let home = temp("auto-index-home");
    let cwd = temp("auto-index-cwd");

    let first = run(&home, &cwd, &["search", "anything", "--mode", "semantic"]);
    let stdout = String::from_utf8(first.stdout).unwrap();
    let stderr = String::from_utf8(first.stderr).unwrap();
    assert!(!first.status.success(), "{stderr}");
    assert!(
        stdout.contains("Indexed "),
        "expected the current directory to be indexed:\n{stdout}"
    );
    assert!(
        stderr.contains("nothing indexed"),
        "expected an empty index after indexing an empty directory:\n{stderr}"
    );

    let second = run(&home, &cwd, &["search", "anything", "--mode", "semantic"]);
    let stdout = String::from_utf8(second.stdout).unwrap();
    let stderr = String::from_utf8(second.stderr).unwrap();
    assert!(
        !stdout.contains("Indexed "),
        "indexed the current directory again:\n{stdout}"
    );
    assert!(stderr.contains("nothing indexed"), "{stderr}");
}

#[test]
fn changed_chunking_discards_memory() {
    let home = temp("auto-index-rechunk-home");
    let cwd = temp("auto-index-rechunk-cwd");

    run(&home, &cwd, &["search", "anything", "--mode", "semantic"]);
    std::fs::write(home.join("config.toml"), "[index]\nchunk_tokens = 256\n").unwrap();

    let second = run(&home, &cwd, &["search", "anything", "--mode", "semantic"]);
    let stdout = String::from_utf8(second.stdout).unwrap();
    let stderr = String::from_utf8(second.stderr).unwrap();
    assert!(stderr.contains("chunked with other settings"), "{stderr}");
    assert!(
        stdout.contains("Indexed "),
        "did not index again after discarding:\n{stdout}"
    );
}

#[test]
fn search_skips_cwd_inside_indexed_root() {
    let home = temp("auto-index-nested-home");
    let parent = temp("auto-index-nested-cwd");
    let child = parent.join("child");
    std::fs::create_dir(&child).unwrap();

    let first = run(
        &home,
        &parent,
        &["search", "anything", "--mode", "semantic"],
    );
    assert!(
        String::from_utf8(first.stdout)
            .unwrap()
            .contains("Indexed "),
        "parent was not indexed"
    );

    let second = run(&home, &child, &["search", "anything", "--mode", "semantic"]);
    let stdout = String::from_utf8(second.stdout).unwrap();
    assert!(
        !stdout.contains("Indexed "),
        "indexed a directory inside an indexed root:\n{stdout}"
    );
}
