use std::time::Instant;

pub struct Profiler {
    start: Instant,
}

impl Profiler {
    pub fn new() -> *mut Profiler {
        Box::into_raw(Box::new(Profiler { start: Instant::now() }))
    }

    pub fn elapsed(&self) -> std::time::Duration {
        self.start.elapsed()
    }
}

pub fn start_profiling() -> *mut Profiler {
    Profiler::new()
}

pub fn stop_profiling(profiler: *mut Profiler) {
    unsafe {
        let _ = Box::from_raw(profiler);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiling() {
        let profiler = start_profiling();
        std::thread::sleep(std::time::Duration::from_millis(100));
        stop_profiling(profiler);
    }
}