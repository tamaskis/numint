use crate::ode_state::ode_state_trait::{OdeState, StateIndex};
use linalg_traits::{Mat, Matrix};

#[cfg(feature = "nalgebra")]
use nalgebra::DMatrix;

#[cfg(feature = "ndarray")]
use ndarray::Array2;

#[cfg(feature = "faer")]
use faer::Mat as FMat;

/// Macro to implement the [`OdeState`] trait for dynamically-sized matrix types.
///
/// A type must implement the
/// [`linalg_traits::Matrix`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.Matrix.html)
/// trait to be used with this macro.
///
/// # Arguments
///
/// * `$type:ident` - Type to implement the [`OdeState`] trait for. Do not include any type
///   parameters (e.g. use `Mat` instead of `Mat<f64>`).
///
/// # Example
///
/// This example demonstrates how we use the `impl_ode_state_for_dmatrix` macro internally within
/// the `numint` crate to implement the [`OdeState`] trait for `Mat<f64>`.
///
/// ```ignore
/// use linalg_traits::{Mat, Matrix};
///
/// use numint::{impl_ode_state_for_dmatrix, OdeState};
///
/// impl_ode_state_for_dmatrix!(Mat);
/// ```
#[macro_export]
macro_rules! impl_ode_state_for_dmatrix {
    ($($type:ident),*) => {
        $(
            impl OdeState for $type<f64> {
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
                        StateIndex::Vector(_) => panic!("Cannot index a matrix ODE state with a StateIndex::Vector."),
                        StateIndex::Matrix(i, j) => self[(i, j)]
                    }
                }
            }
        )*
    };
}

// Implementation of OdeState for Mat<f64>.
impl_ode_state_for_dmatrix!(Mat);

// Implementation of OdeState for nalgebra::DMatrix<f64>.
#[cfg(feature = "nalgebra")]
impl_ode_state_for_dmatrix!(DMatrix);

// Implementation of OdeState for ndarray::Array2<f64>.
#[cfg(feature = "ndarray")]
impl_ode_state_for_dmatrix!(Array2);

// Implementation of OdeState for faer::Mat<f64>.
#[cfg(feature = "faer")]
impl_ode_state_for_dmatrix!(FMat);

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper function for testing the implementation of `OdeState` for matrix types.
    ///
    /// # Type Parameters
    ///
    /// * `T` - The type to test.
    fn ode_state_matrix_test_helper<T: OdeState + Matrix<f64>>() {
        // Test values.
        let a = <T as Matrix<f64>>::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let b = <T as Matrix<f64>>::from_row_slice(2, 2, &[10.0, 8.0, 4.0, 2.0]);
        let c = 5.0;
        let a_plus_b = <T as Matrix<f64>>::from_row_slice(2, 2, &[11.0, 10.0, 7.0, 6.0]);
        let a_minus_b = <T as Matrix<f64>>::from_row_slice(2, 2, &[-9.0, -6.0, -1.0, 2.0]);
        let a_times_c = <T as Matrix<f64>>::from_row_slice(2, 2, &[5.0, 10.0, 15.0, 20.0]);

        // Check addition.
        assert_eq!(OdeState::add(&a, &b), a_plus_b);

        // Check addition-assignment.
        let mut d = a.clone();
        OdeState::add_assign(&mut d, &b);
        assert_eq!(d, a_plus_b);

        // Check subtraction.
        assert_eq!(OdeState::sub(&a, &b), a_minus_b);

        // Check subtraction-assignment.
        d = a.clone();
        OdeState::sub_assign(&mut d, &b);
        assert_eq!(d, a_minus_b);

        // Check multiplication.
        assert_eq!(OdeState::mul(&a, c), a_times_c);

        // Check multiplication-assignment.
        d = a.clone();
        OdeState::mul_assign(&mut d, c);
        assert_eq!(d, a_times_c);

        // Check indexing.
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(a.get_state_variable(StateIndex::Matrix(i, j)), a[(i, j)]);
            }
        }
    }

    #[test]
    fn test_ode_state_linalg_traits_mat() {
        ode_state_matrix_test_helper::<Mat<f64>>();
    }

    #[test]
    #[cfg(feature = "nalgebra")]
    fn test_ode_state_nalgebra_dmatrix() {
        ode_state_matrix_test_helper::<DMatrix<f64>>();
    }

    #[test]
    #[cfg(feature = "ndarray")]
    fn test_ode_state_ndarray_array2() {
        ode_state_matrix_test_helper::<Array2<f64>>();
    }

    #[test]
    #[cfg(feature = "faer")]
    fn test_ode_state_faer_mat() {
        ode_state_matrix_test_helper::<Mat<f64>>();
    }
}
