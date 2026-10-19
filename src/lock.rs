use std::fs::{File, OpenOptions};
use std::io;
use std::io::Write;
use std::os::fd::AsRawFd;
use std::path::Path;

use anyhow::{Context, Result};

use crate::config::Home;

/// Exclusive flock on `memory/lock`, held for the life of a command that reads or
/// writes the vector memory. Another heracles process blocks until this drops.
pub struct MemoryLock {
    _file: File,
}

impl MemoryLock {
    pub fn acquire(home: &Home) -> Result<Self> {
        let dir = home.memory_dir();
        fs_create(&dir)?;
        let path = dir.join("lock");
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&path)
            .with_context(|| format!("opening {}", path.display()))?;
        flock_exclusive(&file).with_context(|| {
            format!(
                "waiting for another heracles process to finish using {}",
                dir.display()
            )
        })?;
        Ok(Self { _file: file })
    }
}

fn fs_create(dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))
}

fn flock_exclusive(file: &File) -> io::Result<()> {
    // LOCK_EX blocks until held; LOCK_NB fails immediately when another process holds it.
    const LOCK_EX: i32 = 2;
    const LOCK_NB: i32 = 4;
    let fd = file.as_raw_fd();
    let rc = unsafe { flock(fd, LOCK_EX | LOCK_NB) };
    if rc == 0 {
        return Ok(());
    }
    let err = io::Error::last_os_error();
    if err.kind() != io::ErrorKind::WouldBlock {
        return Err(err);
    }
    // Print before blocking so a held lock is not silent.
    let _ = writeln!(
        io::stderr(),
        "waiting for another heracles process to finish…"
    );
    let rc = unsafe { flock(fd, LOCK_EX) };
    if rc == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

unsafe extern "C" {
    fn flock(fd: i32, op: i32) -> i32;
}
