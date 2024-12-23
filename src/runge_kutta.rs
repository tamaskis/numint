use linalg_traits::{Scalar, Vector};

/// Propagates the state vector forward one time step using the Euler (first-order) method.
///
/// # Arguments
///
/// * `f` - Multivariate, vector-valued function defining the vector-valued ODE
///         $\mathbf{f}:\mathbb{R}\times\mathbb{R}^{p}\to\mathbb{R}^{p}$.
///         $$\frac{d\mathbf{y}}{dt}=\mathbf{f}(t,\mathbf{y})$$
pub fn rk1_euler<S: Scalar, V: Vector<S>>(f: &impl Fn(S, &V) -> V, t: S, h: S, y: &mut V) {
    let mut f_eval = f(t, y);
    f_eval.mul_assign(h);
    y.add_assign(&f_eval);
}
