use std::sync::atomic::{AtomicBool, Ordering};

static INTERRUPTED: AtomicBool = AtomicBool::new(false);

const SIGINT: i32 = 2;
const SIG_DFL: usize = 0;

/// Catch Ctrl+C so the index pass can finish in-flight work and save. A second
/// Ctrl+C restores the default and kills the process.
pub fn install() {
    unsafe extern "C" fn handle(_: i32) {
        INTERRUPTED.store(true, Ordering::Relaxed);
        unsafe {
            signal(SIGINT, SIG_DFL);
        }
    }
    unsafe {
        signal(SIGINT, handle as *const () as usize);
    }
}

pub fn requested() -> bool {
    INTERRUPTED.load(Ordering::Relaxed)
}

#[cfg(test)]
pub(crate) fn request_for_test() {
    INTERRUPTED.store(true, Ordering::Relaxed);
}

#[cfg(test)]
pub(crate) fn clear_for_test() {
    INTERRUPTED.store(false, Ordering::Relaxed);
}

unsafe extern "C" {
    fn signal(sig: i32, handler: usize) -> usize;
}
