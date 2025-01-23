/// Macro to implement the [`crate::OdeState`] trait for statically-sized vector types.
///
/// A type must implement the
/// [`linalg_traits::Vector`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.Vector.html)
/// trait to be used with this macro.
///
/// # Arguments
///
/// * `$type:ident` - Type to implement the [`crate::OdeState`] trait for. Do not include any type
///   parameters (e.g. use `SVector` instead of `SVector<f64, N>`).
///
/// # Warning
///
/// This macro is not currently used anywhere within the `numint` crate. This is because the
/// [`crate::OdeState`] trait is already implemented for `nalgebra::SVector<f64, N>` by the
/// `impl_ode_state_for_smatrix` macro.
#[macro_export]
macro_rules! impl_ode_state_for_svector {
    ($($type:ident),*) => {
        $(
            impl<const N: usize> OdeState for $type<f64, N> {
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
