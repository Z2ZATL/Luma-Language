//! Luma - A Domain-Specific Language for AI Development
//!
//! This library provides the core functionality for Luma, including AI pipelines,
//! integrations, utilities, and plugins.

pub mod ai;
pub mod core;
pub mod integrations;
pub mod utilities;

#[path = "../plugins/mod.rs"]
pub mod plugins;

pub mod repl;

pub use ai::{self, data, deployment, engine, evaluation, models, training};
pub use core::{self, compiler, interpreter, parser, stdlib};
pub use integrations::{self, tensorflow, pytorch, huggingface, web};
pub use utilities::{self, profiling, debugging, logging, visualization};
pub use plugins::{self, community, registry};
pub use repl::{self, repl};

// [cfg(test)] ยังคงเหมือนเดิม
#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}