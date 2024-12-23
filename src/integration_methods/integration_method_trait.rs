use crate::ode_state::ode_state_trait::OdeState;

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
pub trait IntegrationMethod<T: OdeState> {
    /// Propagate the state vector forward one time step.
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
