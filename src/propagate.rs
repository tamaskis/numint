use crate::state::State;

/// | Dependent Variable Type | Function Signature | ODE Form |
///     | ----------------------- | ------------------ | -------- |
///     | scalar | $f:\mathbb{R}\times\mathbb{R}\to\mathbb{R}$ | $\dfrac{dy}{dt}=f(t,y)$ |
///     | vector | $\mathbf{f}:\mathbb{R}\times\mathbb{R}^{p}\to\mathbb{R}^{p}$ | $\dfrac{d\mathbf{y}}{dt}=\mathbf{f}(t,\mathbf{y})$ |
///     | matrix | $\mathbf{F}:\mathbb{R}\times\mathbb{R}^{p\times r}\to\mathbb{R}^{p\times r}$ | $\dfrac{d\mathbf{Y}}{dt}=\mathbf{F}(t,\mathbf{Y})$ |
///
/// # Note
///
/// This method modifies the state in-place.
///
/// * Before this method called, the state corresponds to the current sample time, `t`.
/// * After this method is called, the state corresponds to the next sample time, `t + h`.
pub trait IntegrationMethod<T: State> {
    /// Propagate the state vector forward one time step using the Euler (first-order) method.
    ///
    /// # Arguments
    ///
    /// * `f` - Function defining the ordinary differential equation, `dy/dt = f(t,y)`.
    /// * `t` - Current sample time.
    /// * `h` - Step size.
    /// * `y` - Current state (i.e. solution at the current sample time).
    ///
    /// # Note
    ///
    /// This method modifies the state, `y`, in-place.
    ///
    /// * Before this method called, `y` corresponds to the state at the current sample time, `t`.
    /// * After this method is called, `y` corresponds to the state at the next sample time,
    ///   `t + h`.
    fn propagate(f: &impl Fn(f64, &T) -> T, t: f64, h: f64, y: &mut T);
}

/// Euler (1st-order Runge-Kutta) method.
#[allow(dead_code)]
pub struct Euler;

impl<T: State> IntegrationMethod<T> for Euler {
    fn propagate(f: &impl Fn(f64, &T) -> T, t: f64, h: f64, y: &mut T) {
        let mut f_eval = f(t, y);
        f_eval.mul_assign(h);
        y.add_assign(&f_eval);
    }
}

fn propagate<T: State, M: IntegrationMethod<T>>(
    f: &impl Fn(f64, &T) -> T,
    y: &mut T,
    t: f64,
    h: f64,
) {
    M::propagate(f, t, h, y);
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
        propagate::<f64, Euler>(&f, &mut y, t, h);

        // After one step, the value should be updated based on dy/dt = -2y
        // For Euler: y(0.1) = y(0) + h * f(t, y) = 1.0 + 0.1 * (-2.0 * 1.0) = 1.0 - 0.2 = 0.8
        assert!((y - 0.8).abs() < 1e-6);
    }
}
