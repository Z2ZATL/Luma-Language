use std::time::Instant;

pub struct Profiler {
    start: Instant,
}

impl Profiler {
    pub fn new() -> Self {
        Profiler {
            start: Instant::now(),
        }
    }

    pub fn elapsed_ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1000.0
    }
}

#[no_mangle]
pub extern "C" fn luma_start_profiling() -> *mut Profiler {
    let profiler = Box::new(Profiler::new());
    Box::into_raw(profiler)
}

#[no_mangle]
pub extern "C" fn luma_get_elapsed(profiler: *mut Profiler) -> f64 {
    if profiler.is_null() {
        return -1.0;
    }
    let profiler = unsafe { &*profiler };
    profiler.elapsed_ms()
}

#[no_mangle]
pub extern "C" fn luma_free_profiler(profiler: *mut Profiler) {
    if !profiler.is_null() {
        unsafe { Box::from_raw(profiler); }
    }
}