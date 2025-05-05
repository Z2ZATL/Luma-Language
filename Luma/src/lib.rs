//! Luma - A Domain-Specific Language for AI Development
//!
//! This library provides the core functionality for Luma, including AI pipelines,
//! integrations, utilities, and plugins.

pub mod ai;
pub mod core;
pub mod integrations;
pub mod utilities;
pub mod plugins;
pub mod repl;

pub use ai::*;
pub use core::*;
pub use integrations::*;
pub use utilities::*;
pub use plugins::*;
pub use repl::*;

// Re-export submodules for convenience
pub use ai::data;
pub use ai::deployment;
pub use ai::engine;
pub use ai::evaluation;
pub use ai::models;
pub use ai::training;
pub use core::compiler;
pub use core::interpreter;
pub use core::parser;
pub use core::stdlib;
pub use integrations::tensorflow;
pub use integrations::pytorch;
pub use integrations::huggingface;
pub use integrations::web;
pub use utilities::profiling;
pub use utilities::debugging;
pub use utilities::logging;
pub use utilities::visualization;
pub use plugins::community;
pub use plugins::registry;
pub use repl::repl;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}