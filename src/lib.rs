//! [![github]](https://github.com/tamaskis/numint)&ensp;[![crates-io]](https://crates.io/crates/numint)&ensp;[![docs-rs]](https://docs.rs/numint)
//!
//! [github]: https://img.shields.io/badge/github-8da0cb?style=for-the-badge&labelColor=555555&logo=github
//! [crates-io]: https://img.shields.io/badge/crates.io-fc8d62?style=for-the-badge&labelColor=555555&logo=rust
//! [docs-rs]: https://img.shields.io/badge/docs.rs-66c2a5?style=for-the-badge&labelColor=555555&logo=docs.rs
//!
//! ODE solvers and numerical integration in Rust.
//!
//! # Overview
//!
//! At its core, this crate is designed to work with scalar-valued, vector-valued, and matrix-valued
//! ordinary differential equations.
//!
//! | ODE Type | Function Signature |
//! | ------------ | ------------------------------ |
//! | scalar-valued | $$\frac{dy}{dt}=f(t,y)\quad\quad\left\(f:\mathbb{R}\times\mathbb{R}\to\mathbb{R}\right\)$$ |
//! | vector-valued | $$\frac{d\mathbf{y}}{dt}=\mathbf{f}(t,\mathbf{y})\quad\quad\left\(\mathbf{f}:\mathbb{R}\times\mathbb{R}^{p}\to\mathbb{R}^{p}\right\)$$ |
//! | matrix-valued | $$\frac{d\mathbf{Y}}{dt}=\mathbf{F}(t,\mathbf{Y})\quad\quad\left\(\mathbf{F}:\mathbb{R}\times\mathbb{R}^{p\times r}\to\mathbb{R}^{p\times r}\right\)$$ |
//!
//! # Initial Value Problem (IVP) Solver
//!
//! The [`solve_ivp()`] function is a general purpose IVP solver with the following features:
//!
//! * it accepts a generic parameter defining the integration method
//! * it can be used for scalar-valued, vector-valued, and matrix-valued problems (this is
//!   accomplished by accepting another generic parameter defining the type of the ODE state)
//!     * any types of vectors or matrices can be used with this function as long as they implement
//!       the [`OdeState`] trait (see [The `OdeState` trait](#the-odestate-trait) section below).
//!
//! # Integration Methods
//!
//! ## Runge-Kutta Methods
//!
//! | Integration Method | Order | Implementation |
//! | ------------------ | ----- | -------------- |
//! | Euler Method | 2 | [`Euler`] |
//! | Midpoint Method | 2 | [`RK2`] |
//! | Heun's Second-Order Method | 2 | [`RK2Heun`] |
//! | Ralston's Second-Order Method | 2 | [`RK2Ralston`] |
//! | Classic (Kutta's) Third-Order method | 2 | [`RK3`] |
//! | Heun's Third-Order Method | 3 | [`RK3Heun`] |
//! | Ralston's Third-Order Method | 3 | [`RK3Ralston`] |
//! | Strong Stability Preserving Runge-Kutta Third-Order Method | 3 | [`SSPRK3`] |
//! | (Classic) Runge-Kutta Fourth-Order Method | 4 | [`RK4`] |
//! | Ralston's Fourth-Order Method | 4 | [`RK4Ralston`] |
//! | 3/8 Rule Fourth Order Method | 4 | [`RK438`] |
//!
//! # The [`OdeState`] trait
//!
//! To allow users to use their favorite vector and matrix representations, this crate defines the
//! [`OdeState`] trait. As long as the ODE state trait is implemented for a type, you can use that
//! type to define an ODE to be solved using this crate.
//!
//! This crate already defines the [`OdeState`] for a variety of different types from the standard
//! library, `nalgebra`, `ndarray`, `faer`, and `linalg-traits`. For a full list, refer to the
//! [`OdeState`] documentation. This crate also provides macros to automate the implementation of
//! the [`OdeState`] trait for types that implement `linalg_traits::Scalar`,
//! `linalg_traits::Vector`, or `linalg_traits::Matrix` traits.
//!
//! | Macro | Description |
//! | ----- | ----------- |
//! | [`impl_ode_state_for_scalar!`] | Implement [`OdeState`] for a scalar type already implementing the [`linalg_traits::Scalar`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.Scalar.html) trait. |
//! | [`impl_ode_state_for_dvector!`] | Implement [`OdeState`] for a dynamically-sized vector type already implementing the [`linalg_traits::Vector`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.Vector.html) trait. |
//! | [`impl_ode_state_for_svector!`] | Implement [`OdeState`] for a statically-sized vector type already implementing the [`linalg_traits::Vector`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.Vector.html) trait. |
//! | [`impl_ode_state_for_dmatrix!`] | Implement [`OdeState`] for a dynamically-sized matrix type already implementing the [`linalg_traits::Matrix`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.Matrix.html) trait. |
//! | [`impl_ode_state_for_smatrix!`] | Implement [`OdeState`] for a statically-sized matrix type already implementing the [`linalg_traits::Matrix`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.Matrix.html) trait. |
//!
//! # Bouncing Ball Example
//!
//! TODO

// Linter setup.
#![warn(missing_docs)]

// Module declarations.
pub(crate) mod events;
pub(crate) mod integrators;
pub(crate) mod ode_state;
pub(crate) mod solution;
pub(crate) mod solve_ivp;

// Re-exports.
pub use crate::events::event::{
    ConditionFunction, Direction, Event, EventFunction, StateResetFunction, Termination,
};
pub use crate::events::event_manager::EventManager;
pub use crate::events::methods::EventDetectionMethod;
pub use crate::integrators::integrator_trait::Integrator;
pub use crate::integrators::runge_kutta::{
    Euler, RK2Heun, RK2Ralston, RK3Heun, RK3Ralston, RK4Ralston, RK2, RK3, RK4, RK438, SSPRK3,
};
pub use crate::ode_state::ode_state_trait::{OdeState, StateIndex};
pub use crate::solution::Solution;
pub use crate::solve_ivp::solve_ivp;
