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
        let f = |_t: f64, x: &f64| -2.0 * x;
        // Initial state: y(0) = 1.0
        let mut y = 1.0;
        let t = 0.0; // Initial time
        let h = 0.1; // Time step

        // Propagate using Euler method
        Euler::propagate(&f, t, h, &mut y);

        // After one step, the value should be updated based on dy/dt = -2y
        // For Euler: y(0.1) = y(0) + h * f(t, y) = 1.0 + 0.1 * (-2.0 * 1.0) = 1.0 - 0.2 = 0.8
        assert!((y - 0.8).abs() < 1e-6);
    }
}
