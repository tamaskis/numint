pub use crate::integration_methods::integration_method_trait::IntegrationMethod;
pub use crate::ode_state::ode_state_trait::OdeState;

/// Euler (1st-order Runge-Kutta) method.
#[allow(dead_code)]
pub struct Euler;

impl<T: OdeState> IntegrationMethod<T> for Euler {
    fn propagate(f: &impl Fn(f64, &T) -> T, t: f64, h: f64, y: &mut T) {
        // yₙ₊₁ = yₙ + hf(tₙ, yₙ)
        y.add_assign(&f(t, y).mul(h));
    }
}

/// (Classic) Runge-Kutta fourth-order method.
pub struct RK4;

impl<T: OdeState> IntegrationMethod<T> for RK4 {
    fn propagate(f: &impl Fn(f64, &T) -> T, t: f64, h: f64, y: &mut T) {
        // k₁ = f(tₙ, yₙ)
        let k1 = f(t, y);

        // k₂ = f(tₙ + (h/2), yₙ + (hk₁/2))
        let k2 = f(t + (h / 2.0), &y.add(&k1.mul(h / 2.0)));

        // k₃ = f(tₙ + (h/2), yₙ + (hk₂/2))
        let k3 = f(t + (h / 2.0), &y.add(&k2.mul(h / 2.0)));

        // k₄ = f(tₙ + h, yₙ + hk₃)
        let k4 = f(t + h, &y.add(&k3.mul(h)));

        // yₙ₊₁ = yₙ = (h/6)(k₁ + 2k₂ + 2k₃ + k₄)
        y.add_assign(&(k1.add(&k2.mul(2.0)).add(&k3.mul(2.0)).add(&k4)).mul(h / 6.0));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper function for testing Runge-Kutta integration methods.
    ///
    /// # Type Parameters
    ///
    /// * `T` - Integration method type.
    ///
    /// # Arguments
    ///
    /// * `y_exp` - Expected value of the ODE state after one propagation.
    ///
    /// # Panics
    ///
    /// * If the ODE state after one propagation does not equal `y_exp`.
    fn rkx_test_helper<T: IntegrationMethod<f64>>(y_exp: f64) {
        // Function defining the ODE.
        let f = |t: f64, x: &f64| -2.0 * x + t;

        // Current state.
        let mut y = 1.0;

        // Current sample time.
        let t = 0.0;

        // Time step.
        let h = 0.1;

        // Propagate using the Euler method.
        T::propagate(&f, t, h, &mut y);

        // Check value after one propagation.
        assert_eq!(y, y_exp);
    }

    #[test]
    fn test_euler() {
        rkx_test_helper::<Euler>(0.8);
    }

    #[test]
    fn test_rk4() {
        rkx_test_helper::<RK4>(0.8234166666666667);
    }
}
