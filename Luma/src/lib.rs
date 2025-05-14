//! Luma - A Domain-Specific Language for AI Development
//!
//! This library provides the core functionality for Luma, including AI pipelines,
//! integrations, utilities, and plugins.

pub mod ai;
pub mod core;
pub mod compiler;
pub mod integrations;
pub mod utilities;

#[path = "../plugins/mod.rs"]
pub mod plugins;

pub mod repl;

pub use ai::data as ai_data;
pub use ai::deployment as ai_deployment;
pub use ai::engine as ai_engine;
pub use ai::evaluation as ai_evaluation;
pub use ai::models as ai_models;
pub use ai::training as ai_training;

pub use core::compiler as core_compiler;
pub use core::interpreter as core_interpreter;
pub use core::parser as core_parser;
pub use core::stdlib as core_stdlib;

pub use integrations::tensorflow as int_tensorflow;
pub use integrations::pytorch as int_pytorch;
pub use integrations::huggingface as int_huggingface;
pub use integrations::web as int_web;

pub use utilities::profiling as util_profiling;
pub use utilities::debugging as util_debugging;
pub use utilities::logging as util_logging;
pub use utilities::visualization as util_visualization;

// Community plugins support has been temporarily removed
// pub use plugins::community as plg_community;
pub use plugins::registry as plg_registry;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}