//! [![github]](https://github.com/tamaskis/numint)&ensp;[![crates-io]](https://crates.io/crates/numint)&ensp;[![docs-rs]](https://docs.rs/numint)
//!
//! [github]: https://img.shields.io/badge/github-8da0cb?style=for-the-badge&labelColor=555555&logo=github
//! [crates-io]: https://img.shields.io/badge/crates.io-fc8d62?style=for-the-badge&labelColor=555555&logo=rust
//! [docs-rs]: https://img.shields.io/badge/docs.rs-66c2a5?style=for-the-badge&labelColor=555555&logo=docs.rs
//!
//! ODE solvers and numerical integration in Rust.

// Linter setup.
#![warn(missing_docs)]

// Linking project modules.
pub(crate) mod module;

// Re-exports.
pub use crate::module::example_function;
