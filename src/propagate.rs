use crate::state::State;

// TODO: this should implement all integrators
pub trait Propagate<T: State> {
    /// Propagate the state vector forward one time step using the Euler (first-order) method.
    ///
    /// # Arguments
    ///
    /// * `f` - Multivariate function defining the ordinary differential equation, `dy/dt = f(t,y)`.
    ///         The function can be one of three options:
    ///
    ///     | Dependent Variable Type | Function Signature | ODE Form |
    ///     | ----------------------- | ------------------ | -------- |
    ///     | scalar | $f:\mathbb{R}\times\mathbb{R}\to\mathbb{R}$ | $\dfrac{dy}{dt}=f(t,y)$ |
    ///     | vector | $\mathbf{f}:\mathbb{R}\times\mathbb{R}^{p}\to\mathbb{R}^{p}$ | $\dfrac{d\mathbf{y}}{dt}=\mathbf{f}(t,\mathbf{y})$ |
    ///     | matrix | $\mathbf{F}:\mathbb{R}\times\mathbb{R}^{p\times r}\to\mathbb{R}^{p\times r}$ | $\dfrac{d\mathbf{Y}}{dt}=\mathbf{F}(t,\mathbf{Y})$ |
    fn rk1_euler(&mut self, f: &impl Fn(f64, &Self) -> Self, t: f64, h: f64);
}

impl<T: State> Propagate<T> for T {
    fn rk1_euler(&mut self, f: &impl Fn(f64, &T) -> T, t: f64, h: f64) {
        let mut f_eval = f(t, self);
        f_eval.mul_assign(h);
        self.add_assign(&f_eval);
    }
}
