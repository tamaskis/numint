//! [![github]](https://github.com/tamaskis/numint)&ensp;[![crates-io]](https://crates.io/crates/numint)&ensp;[![docs-rs]](https://docs.rs/numint)
//!
//! [github]: https://img.shields.io/badge/github-8da0cb?style=for-the-badge&labelColor=555555&logo=github
//! [crates-io]: https://img.shields.io/badge/crates.io-fc8d62?style=for-the-badge&labelColor=555555&logo=rust
//! [docs-rs]: https://img.shields.io/badge/docs.rs-66c2a5?style=for-the-badge&labelColor=555555&logo=docs.rs
//!
//! ODE solvers and numerical integration in Rust.

// Linter setup.
#![warn(missing_docs)]

// Module declarations.
pub(crate) mod integration_methods;
pub(crate) mod ode_state;
pub(crate) mod solution;
pub(crate) mod solve_ivp;

// Re-exports.
pub use crate::integration_methods::integration_method_trait::IntegrationMethod;
pub use crate::integration_methods::runge_kutta::Euler;
pub use crate::ode_state::ode_state_trait::{OdeState, StateIndex};
pub use crate::solution::Solution;
pub use crate::solve_ivp::solve_ivp;
