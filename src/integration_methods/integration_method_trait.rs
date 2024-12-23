use crate::ode_state::ode_state_trait::OdeState;

/// Trait defining an integration method.
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
    ///
    /// # Note
    ///
    /// The ordinary differential equation `f` can be specified in one of the following three ways:
    ///
    /// | Type | Ordinary Differential Equation |
    /// | ------------ | ------------------------------ |
    /// | scalar-valued | $$\frac{dy}{dt}=f(t,y)\quad\quad\left\(f:\mathbb{R}\times\mathbb{R}\to\mathbb{R}\right\)$$ |
    /// | vector-valued | $$\frac{d\mathbf{y}}{dt}=\mathbf{f}(t,\mathbf{y})\quad\quad\left\(\mathbf{f}:\mathbb{R}\times\mathbb{R}^{p}\to\mathbb{R}^{p}\right\)$$ |
    /// | matrix-valued | $$\frac{d\mathbf{Y}}{dt}=\mathbf{F}(t,\mathbf{Y})\quad\quad\left\(\mathbf{F}:\mathbb{R}\times\mathbb{R}^{p\times r}\to\mathbb{R}^{p\times r}\right\)$$ |
    fn propagate(f: &impl Fn(f64, &T) -> T, t: f64, h: f64, y: &mut T);
}
