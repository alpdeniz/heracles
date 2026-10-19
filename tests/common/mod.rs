#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

pub fn temp(name: &str) -> PathBuf {
    let path = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&path);
    std::fs::create_dir_all(&path).unwrap();
    path.canonicalize().unwrap()
}

pub fn run(home: &Path, cwd: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_heracles"))
        .env("HERACLES_HOME", home)
        .current_dir(cwd)
        .args(args)
        .output()
        .expect("failed to run heracles")
}

/// Runs heracles and returns stdout. Fails the test if the command fails.
pub fn stdout(home: &Path, cwd: &Path, args: &[&str]) -> String {
    let output = run(home, cwd, args);
    assert!(
        output.status.success(),
        "`heracles {}` failed:\n{}",
        args.join(" "),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("stdout is not UTF-8")
}
