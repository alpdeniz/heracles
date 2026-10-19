use std::path::{Path, PathBuf};

#[path = "common/mod.rs"]
mod common;

const FIXTURES: [&str; 5] = [
    "astronomy.txt",
    "baking.txt",
    "finance.txt",
    "gardening.txt",
    "networking.txt",
];

fn corpus() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/corpus")
}

fn heracles(home: &Path, args: &[&str]) -> String {
    common::stdout(home, &corpus(), args)
}

fn top_fixture(output: &str) -> Option<&'static str> {
    FIXTURES
        .iter()
        .filter_map(|name| output.find(name).map(|pos| (pos, *name)))
        .min()
        .map(|(_, name)| name)
}

/// The user's config, so the test reaches their embedding server while keeping its memory isolated.
fn user_config() -> Option<PathBuf> {
    let home = std::env::var_os("HERACLES_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| Path::new(&h).join(".heracles")))?;
    Some(home.join("config.toml")).filter(|c| c.is_file())
}

#[test]
#[ignore = "requires the embedding server configured in config.toml"]
fn finds_files_by_meaning() {
    let home = common::temp("heracles-home");
    if let Some(config) = user_config() {
        std::fs::copy(config, home.join("config.toml")).unwrap();
    }
    let corpus = corpus();
    let corpus = corpus.to_str().unwrap();

    heracles(&home, &["init"]);
    heracles(&home, &["index", corpus]);
    let again = heracles(&home, &["index", corpus]);
    assert!(again.contains("0 changed files"), "re-embedded: {again}");

    let cases = [
        ("how do I make a loaf with natural leavening", "baking.txt"),
        ("what remains after a giant star dies", "astronomy.txt"),
        ("growing vegetables in the backyard", "gardening.txt"),
        ("saving money for retirement", "finance.txt"),
        ("how did we handle retries", "networking.txt"),
    ];

    for (query, expected) in cases {
        let output = heracles(&home, &["search", query, "-p", corpus]);
        assert_eq!(
            top_fixture(&output),
            Some(expected),
            "query {query:?} ranked the wrong file first:\n{output}"
        );
    }
}
