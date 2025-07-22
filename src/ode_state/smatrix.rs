use crate::ode_state::ode_state_trait::{OdeState, StateIndex};
use linalg_traits::Matrix;

#[cfg(feature = "nalgebra")]
use nalgebra::SMatrix;

/// Macro to implement the [`OdeState`] trait for statically-sized matrix types.
///
/// A type must implement the
/// [`linalg_traits::Matrix`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.Matrix.html)
/// trait to be used with this macro.
///
/// # Arguments
///
/// * `$type:ident` - Type to implement the [`OdeState`] trait for. Do not include any type
///   parameters (e.g. use `SMatrix` instead of `SMatrix<f64, R, C>`).
///
/// # Warning
///
/// We use this macro to implement [`OdeState`] for `nalgebra::SMatrix<f64, R, C>`. However, this
/// also ends up implementing it for `nalgebra::SVector<f64, N>`, since both of them are type
/// aliases for `nalgebra::Matrix`.
///
/// # Example
///
/// This example demonstrates how we use the `impl_ode_state_for_smatrix` macro internally within
/// the `numint` crate to implement the [`crate::OdeState`] trait for `SMatrix<f64, R, C>`.
///
/// ```ignore
/// use linalg_traits::Matrix;
/// use nalgebra::SMatrix;
///
/// use numint::{impl_ode_state_for_smatrix, OdeState};
///
/// impl_ode_state_for_smatrix!(SMatrix);
/// ```
#[macro_export]
macro_rules! impl_ode_state_for_smatrix {
    ($($type:ident),*) => {
        $(
            impl<const R: usize, const C: usize> OdeState for $type<f64, R, C> {
                fn add(&self, other: &Self) -> Self {
                    <Self as Matrix<f64>>::add(self, other)
                }
                fn add_assign(&mut self, other: &Self) {
                    <Self as Matrix<f64>>::add_assign(self, other);
                }
                fn sub(&self, other: &Self) -> Self {
                    <Self as Matrix<f64>>::sub(self, other)
                }
                fn sub_assign(&mut self, other: &Self) {
                    <Self as Matrix<f64>>::sub_assign(self, other);
                }
                fn mul(&self, scalar: f64) -> Self {
                    <Self as Matrix<f64>>::mul(self, scalar)
                }
                fn mul_assign(&mut self, scalar: f64) {
                    <Self as Matrix<f64>>::mul_assign(self, scalar);
                }
                fn get_state_variable(&self, index: StateIndex) -> f64 {
                    match index {
                        StateIndex::Scalar() => panic!("Cannot index a matrix ODE state with a StateIndex::Scalar."),
                        StateIndex::Vector(i) => self[i],   // to support nalgebra::SVector
                        StateIndex::Matrix(i, j) => self[(i, j)]
                    }
                }
            }
        )*
    };
}

// Implementation of OdeState for nalgebra::SVector<f64, N> and nalgebra::SMatrix<f64, R, C>.
#[cfg(feature = "nalgebra")]
impl_ode_state_for_smatrix!(SMatrix);
