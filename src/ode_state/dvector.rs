use crate::ode_state::ode_state_trait::{OdeState, StateIndex};
use linalg_traits::Vector;

#[cfg(feature = "nalgebra")]
use nalgebra::DVector;

#[cfg(feature = "ndarray")]
use ndarray::Array1;

/// Macro to implement the [`OdeState`] trait for dynamically-sized vector types.
///
/// A type must implement the
/// [`linalg_traits::Vector`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.Vector.html)
/// trait to be used with this macro.
///
/// # Arguments
///
/// * `$type:ident` - Type to implement the [`OdeState`] trait for. Do not include any type
///   parameters (e.g. use `Vec` instead of `Vec<f64>`).
///
/// # Example
///
/// This example demonstrates how we use the `impl_ode_state_for_dvector` macro internally within
/// the `numint` crate to implement the [`OdeState`] trait for [`Vec<f64>`].
///
/// ```ignore
/// use linalg_traits::Vector;
///
/// use numint::{impl_ode_state_for_dvector, OdeState};
///
/// impl_ode_state_for_dvector!(Vec);
/// ```
#[macro_export]
macro_rules! impl_ode_state_for_dvector {
    ($($type:ident),*) => {
        $(
            impl OdeState for $type<f64> {
                fn add(&self, other: &Self) -> Self {
                    <Self as Vector<f64>>::add(self, other)
                }
                fn add_assign(&mut self, other: &Self) {
                    <Self as Vector<f64>>::add_assign(self, other);
                }
                fn sub(&self, other: &Self) -> Self {
                    <Self as Vector<f64>>::sub(self, other)
                }
                fn sub_assign(&mut self, other: &Self) {
                    <Self as Vector<f64>>::sub_assign(self, other);
                }
                fn mul(&self, scalar: f64) -> Self {
                    <Self as Vector<f64>>::mul(self, scalar)
                }
                fn mul_assign(&mut self, scalar: f64) {
                    <Self as Vector<f64>>::mul_assign(self, scalar);
                }
                fn get_state_variable(&self, index: StateIndex) -> f64 {
                    match index {
                        StateIndex::Scalar() => panic!("Cannot index a vector ODE state with a StateIndex::Scalar."),
                        StateIndex::Vector(i) => self[i],
                        StateIndex::Matrix(_, _) => panic!("Cannot index a vector ODE state with a StateIndex::Matrix.")
                    }
                }
            }
        )*
    };
}

// Implementation of OdeState for Vec<f64>.
impl_ode_state_for_dvector!(Vec);

// Implementation of OdeState for nalgebra::DVector<f64>.
#[cfg(feature = "nalgebra")]
impl_ode_state_for_dvector!(DVector);

// Implementation of OdeState for ndarray::Array1<f64>.
#[cfg(feature = "ndarray")]
impl_ode_state_for_dvector!(Array1);

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper function for testing the implementation of `OdeState` for vector types.
    ///
    /// # Type Parameters
    ///
    /// * `T` - The type to test.
    fn ode_state_vector_test_helper<T: OdeState + Vector<f64>>() {
        // Test values.
        let a = <T as Vector<f64>>::from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let b = <T as Vector<f64>>::from_slice(&[10.0, 8.0, 4.0, 2.0]);
        let c = 5.0;
        let a_plus_b = <T as Vector<f64>>::from_slice(&[11.0, 10.0, 7.0, 6.0]);
        let a_minus_b = <T as Vector<f64>>::from_slice(&[-9.0, -6.0, -1.0, 2.0]);
        let a_times_c = <T as Vector<f64>>::from_slice(&[5.0, 10.0, 15.0, 20.0]);

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
            assert_eq!(a.get_state_variable(StateIndex::Vector(i)), a.vget(i));
        }
    }

    #[test]
    fn test_ode_state_linalg_traits_mat() {
        ode_state_vector_test_helper::<Vec<f64>>();
    }

    #[test]
    #[cfg(feature = "nalgebra")]
    fn test_ode_state_nalgebra_dmatrix() {
        ode_state_vector_test_helper::<DVector<f64>>();
    }

    #[test]
    #[cfg(feature = "ndarray")]
    fn test_ode_state_ndarray_array2() {
        ode_state_vector_test_helper::<Array1<f64>>();
    }
}
