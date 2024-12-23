use crate::state::State;

/// Propagates the state vector forward one time step using the Euler (first-order) method.
///
/// # Arguments
///
/// * `f` - Multivariate, vector-valued function defining the vector-valued ODE
///         $\mathbf{f}:\mathbb{R}\times\mathbb{R}^{p}\to\mathbb{R}^{p}$. TODO could be scalar valued, vector-valued, or matrix-valued.
///         $$\frac{d\mathbf{y}}{dt}=\mathbf{f}(t,\mathbf{y})$$
pub fn rk1_euler<T: State>(f: &impl Fn(f64, &T) -> T, t: f64, h: f64, y: &mut T) {
    let mut f_eval = f(t, y);
    f_eval.mul_assign(h);
    y.add_assign(&f_eval);
}
