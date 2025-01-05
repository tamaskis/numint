pub use crate::integration_methods::integration_method_trait::IntegrationMethod;
pub use crate::ode_state::ode_state_trait::OdeState;

/// Euler (1st-order Runge-Kutta) method.
#[allow(dead_code)]
pub struct Euler;

impl<T: OdeState> IntegrationMethod<T> for Euler {
    fn propagate(f: &impl Fn(f64, &T) -> T, t: f64, h: f64, y: &mut T) {
        // y(t + h) = y(t) + hf(t, y)
        let mut f_eval = f(t, y);
        f_eval.mul_assign(h);
        y.add_assign(&f_eval);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euler_scalar() {
        // Function defining the ODE.
        let f = |_t: f64, x: &f64| -2.0 * x;

        // Current state.
        let mut y = 1.0;

        // Current sample time.
        let t = 0.0;

        // Time step.
        let h = 0.1;

        // Propagate using the Euler method.
        Euler::propagate(&f, t, h, &mut y);

        // Check value after one propagation.
        assert_eq!(y, 0.8);
    }
}
