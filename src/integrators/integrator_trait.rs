use crate::ode_state::ode_state_trait::OdeState;

/// Trait defining an integration method.
pub trait Integrator<T: OdeState> {
    /// Propagate the state vector forward one time step.
    ///
    /// # Arguments
    ///
    /// * `f` - Function defining the ordinary differential equation, `dy/dt = f(t,y)`. See the
    ///   [Overview](crate#overview) section in the documentation for more information.
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
