use crate::ode_state::ode_state_trait::{OdeState, StateIndex};
use num_traits::ToPrimitive;

/// Macro to implement the [`OdeState`] trait for scalar types.
///
/// To be compatible with this macro, a type must implement the following traits:
///
/// * `std::ops::Add<T, Output=T>`
/// * `std::ops::AddAssign<T>`
/// * `std::ops::Sub<T, Output=T>`
/// * `std::ops::SubAssign<T`
/// * `std::ops::Mul<f64, Output=T>`
/// * `std::ops::MulAssign<f64>`
///
/// Note that any type implementing the
/// [`linalg_traits::Scalar`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.Scalar.html)
/// trait can be used with this macro.
///
/// # Arguments
///
/// * `$($type:ty),*` - A comma-separated list of scalar types for which the [`OdeState`] trait
///   implementation will be generated.
///
/// # Example
///
/// This example demonstrates how we use the `impl_ode_state_for_scalar` macro internally within the
/// `numint` crate to implement the [`OdeState`] trait for [`f64`].
///
/// ```ignore
/// use numint::{impl_ode_state_for_scalar, OdeState};
///
/// impl_ode_state_for_scalar!(f64);
/// ```
#[macro_export]
macro_rules! impl_ode_state_for_scalar {
    ($($type:ty),*) => {
        $(
            impl OdeState for $type {
                fn add(&self, other: &Self) -> Self {
                    self + other
                }
                fn add_assign(&mut self, other: &Self) {
                    *self += other;
                }
                fn sub(&self, other: &Self) -> Self {
                    self - other
                }
                fn sub_assign(&mut self, other: &Self) {
                    *self -= other
                }
                fn mul(&self, scalar: f64) -> Self {
                    self * scalar
                }
                fn mul_assign(&mut self, scalar: f64) {
                    *self *= scalar
                }
                fn get_state_variable(&self, index: StateIndex) -> f64 {
                    match index {
                        StateIndex::Scalar() => <Self as ToPrimitive>::to_f64(self).unwrap(),
                        StateIndex::Vector(_) => panic!("Cannot index a scalar ODE state with a StateIndex::Vector."),
                        StateIndex::Matrix(_, _) => panic!("Cannot index a scalar ODE state with a StateIndex::Matrix.")
                    }
                }
            }
        )*
    };
}

// Implementation of OdeState for f64.
impl_ode_state_for_scalar!(f64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ode_state_f64() {
        // Test values.
        let a = 2.0;
        let b = 6.0;
        let c = 5.0;
        let a_plus_b = 8.0;
        let a_minus_b = -4.0;
        let a_times_c = 10.0;

        // Check addition.
        assert_eq!(OdeState::add(&a, &b), a_plus_b);

        // Check addition-assignment.
        let mut d = a;
        OdeState::add_assign(&mut d, &b);
        assert_eq!(d, a_plus_b);

        // Check subtraction.
        assert_eq!(OdeState::sub(&a, &b), a_minus_b);

        // Check subtraction-assignment.
        d = a;
        OdeState::sub_assign(&mut d, &b);
        assert_eq!(d, a_minus_b);

        // Check multiplication.
        assert_eq!(OdeState::mul(&a, c), a_times_c);

        // Check multiplication-assignment.
        d = a;
        OdeState::mul_assign(&mut d, c);
        assert_eq!(d, a_times_c);

        // Check indexing.
        assert_eq!(a.get_state_variable(StateIndex::Scalar()), a);
    }
}
