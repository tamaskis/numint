use crate::events::event::Direction;
use crate::events::event::Event;
use crate::integrators::integrator_trait::Integrator;
use crate::ode_state::ode_state_trait::OdeState;
use rootfinder::{Interval, root_bisection};

/// Event detection method.
///
/// # Theory
///
/// An event is defined to occur when the event function, $g(t,\mathbf{y})$, has a zero crossing.
/// However, when taking finite steps to integrate an ODE, the event function will (almost) never be
/// exactly 0 at a specific sample time. Instead, what we will usually notice is the event function
/// changing sign from one time step to the next.
///
/// When detecting an event between two sample times, we deal with the following times:
///
/// * $t_{\mathrm{prev}}=$ previous time
/// * $t_{\mathrm{curr}}=$ current time
/// * $t_{\mathrm{event}}=$ event time
///
/// The corresponding states are
///
/// * $\mathbf{y}_{\mathrm{prev}}=\mathbf{y}(t\_{\mathrm{prev}})$ ODE state at previous time
/// * $\mathbf{y}_{\mathrm{curr}}=\mathbf{y}(t\_{\mathrm{curr}})$ ODE state at current time
/// * $\mathbf{y}_{\mathrm{event}}=\mathbf{y}(t\_{\mathrm{event}})$ ODE state at event
///
/// The step size between the current and previous times, $h$, is just
///
/// $$h=t_{\mathrm{curr}}=t_{\mathrm{prev}}$$
///
/// The step size between the event time and the previous time, $h_{\mathrm{event}}$, is similarly
///
/// $$h_{\mathrm{event}}=t_{\mathrm{event}}-t_{\mathrm{prev}}$$
///
/// The goal of event detection is to identify the step size, $h_{\mathrm{event}}$, that we need to
/// take to get from the previous time, $t_{\mathrm{prev}}$, to the event time,
/// $t_{\mathrm{event}}$.
pub enum EventDetectionMethod {
    /// Identify the _exact_ time of an event (to within machine precision).
    ///
    /// # Theory
    ///
    /// Evaluating the event function at the event,
    ///
    /// $$g(t_{\mathrm{event}})=g(t_{\mathrm{event}},\mathbf{y}(t_{\mathrm{event}}))$$
    ///
    /// By definition, $g(t_{\mathrm{event}})=0$, so we have
    ///
    /// $$0=g(t_{\mathrm{event}},\mathbf{y}(t_{\mathrm{event}}))$$
    ///
    /// Since $t_{\mathrm{event}}=t_{\mathrm{prev}}+h_{\mathrm{event}}$,
    ///
    /// $$0=g(t_{\mathrm{prev}}+h_{\mathrm{event}},\mathbf{y}(t_{\mathrm{prev}}+h_{\mathrm{event}}))$$
    ///
    /// We don't explicitly know $\mathbf{y}(t})$; in the context of ODE solvers, that is the very
    /// quantity we are trying to solve for! However, we _do_ have integration methods that let us
    /// approximate $\mathbf{y}(t)$. Specifically, event detection is a subroutine called within an
    /// ODE solver which is using some selected integration method, so we should just use that
    /// same integration method.
    ///
    /// Mathematically, we can describe an integration method as a function
    /// $\mathbf{m}(t,\mathbf{y},h)$ that approximates the ODE state at time $t+h$ given the ODE
    /// state at time $t$.
    ///
    /// $$
    /// \begin{aligned}
    /// \mathbf{y}(t+h)&=\mathbf{y}(t)+\int_{t}^{t+h}\mathbf{f}(t,\mathbf{y}(t))\,dt \\\\
    /// &\approx\mathbf{m}(t,\mathbf{y}(t),h)
    /// \end{aligned}
    /// $$
    ///
    /// If we assume $\mathbf{y}(t+h)=\mathbf{m}(t,\mathbf{y},h)$, then we can write
    ///
    /// $$
    /// \begin{aligned}
    /// \mathbf{y}(t_{\mathrm{prev}}+h_{\mathrm{event}})&=\mathbf{m}(t_{\mathrm{prev}},\mathbf{y}(t_{\mathrm{prev}}),h_{\mathrm{event}}) \\\\
    /// &=\mathbf{m}(t_{\mathrm{prev}},\mathbf{y}\_{\mathrm{prev}},h_{\mathrm{event}})
    /// \end{aligned}
    /// $$
    ///
    /// Substituting this into our equation from earlier,
    ///
    /// $$0=g(t_{\mathrm{prev}}+h_{\mathrm{event}},\mathbf{m}(t_{\mathrm{prev}},\mathbf{y}\_{\mathrm{prev}},h_{\mathrm{event}}))$$
    ///
    /// Finally, let's define the auxiliary event function, $\tilde{g}(h_{\mathrm{event}})$, noting
    /// that $t_{\mathrm{prev}}$ and $y_{\mathrm{prev}}$ are constants.
    ///
    /// $$\tilde{g}(h_{\mathrm{event}})=g(t_{\mathrm{prev}}+h_{\mathrm{event}},\mathbf{m}(t_{\mathrm{prev}},\mathbf{y}\_{\mathrm{prev}},h_{\mathrm{event}}))$$
    ///
    /// Thus, we are left with
    ///
    /// $$0=\tilde{g}(h_{\mathrm{event}})$$
    ///
    /// so solving for $h_{\mathrm{event}}$ amounts to solving for the root of
    /// $\tilde{g}(h_{\mathrm{event}})$. We can do this using any root-finding technique, but it is
    /// advantageous to use a bracketing method (such as the bisection method) since
    ///
    /// 1. We know a lower and upper bound for $h_{\mathrm{event}}$ (it is just between $0$ and $h$,
    ///    where $h$ is the step size used by the integrator).
    /// 2. They are guaranteed to converge to within machine precision.
    ///
    /// ## Note on zero-crossing definition
    ///
    /// Since the event function must _cross_ 0 (it is not enough to "touch" 0, such as the function
    /// `g(t,y) = (t - 5)²`), using a bracketing method guarantees that a root will be found. See
    /// [`crate::EventFunction`] for more information.
    Exact,

    /// Approximate the time of an event using linear interpolation.
    ///
    /// # Theory
    ///
    /// When there is an event between between times $t_{\mathrm{prev}}$ and $t_{\mathrm{curr}}$,
    /// then the corresponding values of the event function, $g_{\mathrm{prev}}$ and
    /// $g_{\mathrm{curr}}$, respectively, will have different signs.
    /// [`EventDetectionMethod::Exact`] is used to configure an ODE solver to use a root-finding
    /// method to find the exact time where $g(t,\mathbf{y})$ crosses 0. Instead of solving for the
    /// exact time, we can approximate it using linear interpolation.
    ///
    /// Imagine a straight line connecting the points $(t_{\mathrm{prev}},g_{\mathrm{prev}})$ and
    /// $(t_{\mathrm{curr}},g_{\mathrm{curr}})$. Let's assume that $g(t,\mathbf{y})$ is a linear
    /// function between these points. Then the point $(t_{\mathrm{event}},0)$ will also lie on this
    /// line. Thus, the slope between $(t_{\mathrm{prev}},g_{\mathrm{prev}})$ and
    /// $(t_{\mathrm{event}},0)$ is equal to the slope between
    /// $(t_{\mathrm{prev}},g_{\mathrm{prev}})$ and $(t_{\mathrm{curr}},g_{\mathrm{curr}})$.
    ///
    /// $$\frac{0-g_{\mathrm{prev}}}{t_{\mathrm{event}}-t_{\mathrm{prev}}}=\frac{g_{\mathrm{curr}}-g_{\mathrm{prev}}}{t_{\mathrm{curr}}-t_{\mathrm{prev}}}$$
    ///
    /// Since $t_{\mathrm{event}}-t_{\mathrm{prev}}=h_{\mathrm{event}}$ and
    /// $t_{\mathrm{curr}}-t_{\mathrm{prev}}=h$,
    ///
    /// $$\frac{-g_{\mathrm{prev}}}{h_{\mathrm{event}}}=\frac{g_{\mathrm{curr}}-g_{\mathrm{prev}}}{h}$$
    ///
    /// Solving for $h_{\mathrm{event}}$,
    ///
    /// $$h_{\mathrm{event}}=\frac{-hg_{\mathrm{prev}}}{g_{\mathrm{curr}}-g_{\mathrm{prev}}}$$
    LinearInterpolation,

    /// Approximate the time of an event as the last sample time before the event occurs (i.e.
    /// left-interpolation).
    ///
    /// For this method, we simply assume that $t_{\mathrm{event}}=t_{\mathrm{prev}}$, such that
    ///
    /// $$h_{\mathrm{event}}=0$$
    LeftInterpolation,

    /// Approximate the time of an event as the last sample time before the event occurs (i.e.
    /// right-interpolation).
    ///
    /// For this method, we simply assume that $t_{\mathrm{event}}=t_{\mathrm{curr}}$, such that
    ///
    /// $$h_{\mathrm{event}}=h$$
    ///
    /// where $h$ is the step size used by the integrator.
    RightInterpolation,
}

/// Evaluate the event function at the previous and current times and states and determine if the
/// event is triggered (i.e. if the event function changes sign).
///
/// # Arguments
///
/// * `event` - Event.
/// * `t_prev` - Previous sample time.
/// * `y_prev` - Previous state (i.e. solution at the previous sample time).
/// * `y_curr` - Current state (i.e. solution at the current sample time).
/// * `h` - Step size (to get from the previous time to the current time).
///
/// # Returns
///
/// An option where:
///
/// * `Some` contains a tuple with the values of the event function at the previous and current
///   times and states.
/// * `None` indicates that the event wasn't detected.
fn event_detection_helper<T: OdeState>(
    event: &Event<T>,
    t_prev: f64,
    y_prev: &T,
    y_curr: &T,
    h: f64,
) -> Option<(f64, f64)> {
    // Evaluate the event function at the previous time and state.
    let g_prev = (event.g)(t_prev, y_prev);

    // Evaluate the event function at the current time and state.
    let g_curr = (event.g)(t_prev + h, y_curr);

    // Check the direction of g(t,y) if the event is only triggered in one direction.
    if !matches!(event.direction, Direction::Either) {
        // The event is not triggered if:
        //  1) The event function does not change in value over the time step.
        //  2) The event function increases over the time step, but the event is configured to only
        //     trigger when the event function is decreasing.
        //  3) The event function decreases over the time step, but the event is configured to only
        //     trigger when the event function is increasing.
        if (g_curr == g_prev)
            || ((g_curr > g_prev) && matches!(event.direction, Direction::Decreasing))
            || ((g_curr < g_prev) && matches!(event.direction, Direction::Increasing))
        {
            return None;
        }
    }

    Some((g_prev, g_curr))
}

/// Perform exact event detection (i.e. uses a root-solver to find the exact time of the event).
///
/// See [`EventDetectionMethod::Exact`] for more information.
///
/// # Arguments
///
/// * `f` - Function defining the ordinary differential equation, `dy/dt = f(t,y)`. See the
///   [Overview](crate#overview) section in the documentation for more information.
/// * `event` - Event.
/// * `t_prev` - Previous sample time.
/// * `y_prev` - Previous state (i.e. solution at the previous sample time).
/// * `y_curr` - Current state (i.e. solution at the current sample time).
/// * `h` - Step size (to get from the previous time to the current time).
///
/// # Returns
///
/// An option where:
///
/// * `Some` contains the exact step size required to advance from the current sample time to the
///   event.
/// * `None` indicates that the event wasn't found.
pub(crate) fn exact_event_detection<T: OdeState, I: Integrator<T>>(
    f: &impl Fn(f64, &T) -> T,
    event: &Event<T>,
    t_prev: f64,
    y_prev: &T,
    y_curr: &T,
    h: f64,
) -> Option<f64> {
    if event_detection_helper(event, t_prev, y_prev, y_curr, h).is_some() {
        // Redefine the event function as a function of the time step, h.
        //  --> g(t,y) becomes gₕ(h)
        let gh = |h: f64| {
            // Make a copy of the previous state.
            let mut y_copy = y_prev.clone();

            // Propagate the state from the previous time (t_prev) to our current estimate for the event
            // time (t_prev + h).
            I::propagate(&f, t_prev, h, &mut y_copy);

            // Call the event function at the propagated state.
            (event.g)(t_prev + h, &y_copy)
        };

        // Solve for the step size required to advance from the current time to the event.
        //  --> If root_bisection returns a solver error, it is because there is no zero crossing,
        //      so we cannot detect the event.
        //  --> If root_bisection returns some value, it is guaranteed to have found a zero crossing
        //      (this guarantee comes from the fact that we are not rebracketing).
        root_bisection(&gh, Interval::new(0.0, h), None, None).ok()
    } else {
        None
    }
}

/// Perform linear event detection.
///
/// See [`EventDetectionMethod::Linear`] for more information.
///
/// # Arguments
///
/// * `event` - Event.
/// * `t_prev` - Previous sample time.
/// * `y_prev` - Previous state (i.e. solution at the previous sample time).
/// * `y_curr` - Current state (i.e. solution at the current sample time).
/// * `h` - Step size (to get from the previous time to the current time).
///
/// # Returns
///
/// An option where:
///
/// * `Some` contains the approximate step size required to advance from the current sample time to
///   the event.
/// * `None` indicates that the event wasn't found.
pub(crate) fn linear_event_detection<T: OdeState>(
    event: &Event<T>,
    t_prev: f64,
    y_prev: &T,
    y_curr: &T,
    h: f64,
) -> Option<f64> {
    if let Some((g_prev, g_curr)) = event_detection_helper(event, t_prev, y_prev, y_curr, h) {
        // Compute `h_new` such that the step to where `g(t_prev + h_new) == 0`, assuming g(t) is linear
        // between `g_prev` and `g_curr`.
        Some(-h * g_prev / (g_curr - g_prev))
    } else {
        None
    }
}

/// Perform left event detection.
///
/// See [`EventDetectionMethod::Left`] for more information.
///
/// # Arguments
///
/// * `event` - Event.
/// * `t_prev` - Previous sample time.
/// * `y_prev` - Previous state (i.e. solution at the previous sample time).
/// * `y_curr` - Current state (i.e. solution at the current sample time).
/// * `h` - Step size (to get from the previous time to the current time).
///
/// # Returns
///
/// An option where:
///
/// * `Some` contains the approximate step size required to advance from the current sample time to
///   the event.
///     
///     * If the event occurs exactly at the previous time or exactly at the current time, this
///       function will return `Some(0.0)` (corresponding to the previous time) or `Some(h)`
///       (corresponding to the current time), respectively.
///     * If the event occurs between the previous and current times, `Some(0.0)` is returned.
///
/// * `None` indicates that the event wasn't found.
pub(crate) fn left_event_detection<T: OdeState>(
    event: &Event<T>,
    t_prev: f64,
    y_prev: &T,
    y_curr: &T,
    h: f64,
) -> Option<f64> {
    if let Some((_, g_curr)) = event_detection_helper(event, t_prev, y_prev, y_curr, h) {
        // Check if the event is at the current time, and if so, return the step size to get from
        // the previous time to the current time to identify it as the event. Otherwise, return 0 to
        // identify the previous time as the event.
        if g_curr == 0.0 { Some(h) } else { Some(0.0) }
    } else {
        None
    }
}

/// Perform right event detection.
///
/// See [`EventDetectionMethod::Right`] for more information.
///
/// # Arguments
///
/// * `event` - Event.
/// * `t_prev` - Previous sample time.
/// * `y_prev` - Previous state (i.e. solution at the previous sample time).
/// * `y_curr` - Current state (i.e. solution at the current sample time).
/// * `h` - Step size (to get from the previous time to the current time).
///
/// # Returns
///
/// An option where:
///
/// * `Some` contains the approximate step size required to advance from the current sample time to
///   the event.
///     
///     * If the event occurs exactly at the previous time or exactly at the current time, this
///       function will return `Some(0.0)` (corresponding to the previous time) or `Some(h)`
///       (corresponding to the current time), respectively.
///     * If the event occurs between the previous and current times, `Some(h)` is returned.
///
/// * `None` indicates that the event wasn't found.
pub(crate) fn right_event_detection<T: OdeState>(
    event: &Event<T>,
    t_prev: f64,
    y_prev: &T,
    y_curr: &T,
    h: f64,
) -> Option<f64> {
    if let Some((g_prev, _)) = event_detection_helper(event, t_prev, y_prev, y_curr, h) {
        // Check if the event is at the previous time, and if so, return 0 to identify it as the
        // event. Otherwise, return the step size to get from the previous time to the current time
        // to identify the current time as the event.
        if g_prev == 0.0 { Some(0.0) } else { Some(h) }
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Euler, RK4};
    use numtest::*;

    /// Check the value of the event function at the event.
    ///
    /// Note that this is primarily useful for exact event detection. The value of the event
    /// function at event times determined by the other event detection methods will be considerably
    /// further from 0.
    ///
    /// # Arguments
    ///
    /// * `f` - Function defining the ordinary differential equation, `dy/dt = f(t,y)`. See the
    ///   [Overview](crate#overview) section in the documentation for more information.
    /// * `event` - Event.
    /// * `t_prev` - Previous sample time.
    /// * `y_prev` - Previous state (i.e. solution at the previous sample time).
    /// * `h_event` - The step size required to advance from the current sample time to the event.
    ///
    /// # Panics
    ///
    /// * If the value of the event function is not 0 to within two times the machine epsilon at the
    ///   located event.
    fn check_event_function_value<I: Integrator<f64>>(
        f: &impl Fn(f64, &f64) -> f64,
        event: &Event<f64>,
        t_prev: f64,
        y_prev: &f64,
        h_event: f64,
    ) {
        // Initialize the state at the event to be the state at the previous sample time (will be
        // propagated forward).
        let mut y_event = *y_prev;

        // Propagate the state to the event.
        I::propagate(f, t_prev, h_event, &mut y_event);

        // Get the time of the event.
        let t_event = t_prev + h_event;

        // Evaluate the event function at the event.
        let g_event = (event.g)(t_event, &y_event);

        // Verify that the value of the event function at the event is 0 (to within 2 times the
        // machine epsilon).
        assert_equal_to_atol!(g_event, 0.0, 2.0 * f64::EPSILON);
    }

    #[test]
    fn test_event_detection_on_time_basic() {
        // Define the event with event function g(t,y) = √(t) - 0.5.
        let event = Event::new(|t: f64, _y: &f64| t.sqrt() - 0.5);

        // Define the ODE dy/dt = f(t,y) = y.
        let f = |_t: f64, y: &f64| *y;

        // Define the previous sample time and the corresponding state.
        let t_prev = 0.0;
        let y_prev = 1.0;

        // Define the step size.
        let h = 1.0;

        // Define the current state using numerical integration instead of analytical solution.
        let mut y_curr = y_prev;
        Euler::propagate(&f, t_prev, h, &mut y_curr);

        // Check with exact event detection.
        let h_event_exact =
            exact_event_detection::<f64, Euler>(&f, &event, t_prev, &y_prev, &y_curr, h).unwrap();
        assert_equal_to_decimal!(h_event_exact, 0.25, 15);
        check_event_function_value::<Euler>(&f, &event, t_prev, &y_prev, h_event_exact);

        // Check with linear event detection.
        let h_event_linear =
            linear_event_detection::<f64>(&event, t_prev, &y_prev, &y_curr, h).unwrap();
        assert_eq!(h_event_linear, 0.5);

        // Check with left event detection.
        let h_event_left =
            left_event_detection::<f64>(&event, t_prev, &y_prev, &y_curr, h).unwrap();
        assert_eq!(h_event_left, 0.0);

        // Check with right event detection.
        let h_event_right =
            right_event_detection::<f64>(&event, t_prev, &y_prev, &y_curr, h).unwrap();
        assert_eq!(h_event_right, h);
    }

    #[test]
    fn test_event_detection_helper_1() {
        // Define the event with event function g(t,y) = y - t - 0.5.
        let event = Event::new(|t: f64, y: &f64| y - t - 0.5);

        // Define the previous and current sample times and states.
        let t_prev = 0.0;
        let y_prev = 1.0; // g_prev = 1.0 - 0.0 - 0.5 = 0.5
        let y_curr = 1.5; // g_curr = 1.5 - 1.0 - 0.5 = 0.0
        let h = 1.0;

        // Check that the event is detected and verify the event function values.
        let result = event_detection_helper(&event, t_prev, &y_prev, &y_curr, h);
        assert!(result.is_some());
        let (g_prev, g_curr) = result.unwrap();
        assert_equal_to_decimal!(g_prev, 0.5, 15);
        assert_equal_to_decimal!(g_curr, 0.0, 15);
    }

    #[test]
    fn test_event_detection_helper_2() {
        // Define the event with event function g(t,y) = y - t - 0.5, configured to only detect
        // decreasing events.
        let event = Event::new(|t: f64, y: &f64| y - t - 0.5).direction(Direction::Decreasing);

        // Define the previous and current sample times and states.
        let t_prev = 0.0;
        let y_prev = 1.0; // g_prev = 1.0 - 0.0 - 0.5 = 0.5
        let y_curr = 1.5; // g_curr = 1.5 - 1.0 - 0.5 = 0.0
        let h = 1.0;

        // Check that the event is detected since g(t,y) decreases from 0.5 to 0.0.
        let result = event_detection_helper(&event, t_prev, &y_prev, &y_curr, h);
        assert!(result.is_some());
        let (g_prev, g_curr) = result.unwrap();
        assert_equal_to_decimal!(g_prev, 0.5, 15);
        assert_equal_to_decimal!(g_curr, 0.0, 15);
    }

    #[test]
    fn test_event_detection_helper_3() {
        // Define the event with event function g(t,y) = y - t - 0.5, configured to only detect
        // decreasing events.
        let event = Event::new(|t: f64, y: &f64| y - t - 0.5).direction(Direction::Decreasing);

        // Define the previous and current sample times and states.
        let t_prev = 0.0;
        let y_prev = 0.5; // g_prev = 0.5 - 0.0 - 0.5 = 0.0
        let y_curr = 1.5; // g_curr = 1.5 - 1.0 - 0.5 = 0.0
        let h = 1.0;

        // Check that no event is detected since g(t,y) does not change (stays at 0.0).
        let result = event_detection_helper(&event, t_prev, &y_prev, &y_curr, h);
        assert!(result.is_none());
    }

    #[test]
    fn test_event_detection_helper_4() {
        // Define the event with event function g(t,y) = y - t - 0.5, configured to only detect
        // increasing events.
        let event = Event::new(|t: f64, y: &f64| y - t - 0.5).direction(Direction::Increasing);

        // Define the previous and current sample times and states.
        let t_prev = 0.0;
        let y_prev = 0.0; // g_prev = 0.0 - 0.0 - 0.5 = -0.5
        let y_curr = 1.5; // g_curr = 1.5 - 1.0 - 0.5 = 0.0
        let h = 1.0;

        // Check that the event is detected since g(t,y) increases from -0.5 to 0.0.
        let result = event_detection_helper(&event, t_prev, &y_prev, &y_curr, h);
        assert!(result.is_some());
        let (g_prev, g_curr) = result.unwrap();
        assert_equal_to_decimal!(g_prev, -0.5, 15);
        assert_equal_to_decimal!(g_curr, 0.0, 15);
    }

    #[test]
    fn test_event_detection_helper_5() {
        // Define the event with event function g(t,y) = y - t - 0.5, configured to only detect
        // increasing events.
        let event = Event::new(|t: f64, y: &f64| y - t - 0.5).direction(Direction::Increasing);

        // Define the previous and current sample times and states.
        let t_prev = 0.0;
        let y_prev = 1.0; // g_prev = 1.0 - 0.0 - 0.5 = 0.5
        let y_curr = 1.5; // g_curr = 1.5 - 1.0 - 0.5 = 0.0
        let h = 1.0;

        // Check that no event is detected since g(t,y) decreases from 0.5 to 0.0, but the event
        // is configured to only trigger when g(t,y) increases.
        let result = event_detection_helper(&event, t_prev, &y_prev, &y_curr, h);
        assert!(result.is_none());
    }

    #[test]
    fn test_event_detection_on_time_lower_bound() {
        // Define the event with event function g(t,y) = t.
        let event = Event::new(|t: f64, _y: &f64| t);

        // Define the ODE dy/dt = f(t,y) = y.
        let f = |_t: f64, y: &f64| *y;

        // Define the previous sample time and the corresponding state.
        let t_prev = 0.0;
        let y_prev = 1.0;

        // Define the step size.
        let h = 1.0;

        // Get the current state by integrating using the Euler method.
        let mut y_curr = y_prev;
        Euler::propagate(&f, t_prev, h, &mut y_curr);

        // Check with exact event detection.
        let h_event_exact =
            exact_event_detection::<f64, Euler>(&f, &event, t_prev, &y_prev, &y_curr, h).unwrap();
        assert_eq!(h_event_exact, 0.0);
        check_event_function_value::<Euler>(&f, &event, t_prev, &y_prev, h_event_exact);

        // Check with linear event detection.
        let h_event_linear =
            linear_event_detection::<f64>(&event, t_prev, &y_prev, &y_curr, h).unwrap();
        assert_eq!(h_event_linear, 0.0);

        // Check with left event detection.
        let h_event_left =
            left_event_detection::<f64>(&event, t_prev, &y_prev, &y_curr, h).unwrap();
        assert_eq!(h_event_left, 0.0);

        // Check with right event detection.
        let h_event_right =
            right_event_detection::<f64>(&event, t_prev, &y_prev, &y_curr, h).unwrap();
        assert_eq!(h_event_right, 0.0);
    }

    #[test]
    fn test_event_detection_on_time_upper_bound() {
        // Define the event with event function g(t,y) = t - 1.
        let event = Event::new(|t: f64, _y: &f64| t - 1.0);

        // Define the ODE dy/dt = f(t,y) = y.
        let f = |_t: f64, y: &f64| *y;

        // Define the previous sample time and the corresponding state.
        let t_prev = 0.0;
        let y_prev = 1.0;

        // Define the step size.
        let h = 1.0;

        // Get the current state by integrating using the Euler method.
        let mut y_curr = y_prev;
        Euler::propagate(&f, t_prev, h, &mut y_curr);

        // Check with exact event detection.
        let h_event_exact =
            exact_event_detection::<f64, Euler>(&f, &event, t_prev, &y_prev, &y_curr, h).unwrap();
        assert_eq!(h_event_exact, 1.0);
        check_event_function_value::<Euler>(&f, &event, t_prev, &y_prev, h_event_exact);

        // Check with linear event detection.
        let h_event_linear =
            linear_event_detection::<f64>(&event, t_prev, &y_prev, &y_curr, h).unwrap();
        assert_eq!(h_event_linear, 1.0);

        // Check with left event detection.
        let h_event_left =
            left_event_detection::<f64>(&event, t_prev, &y_prev, &y_curr, h).unwrap();
        assert_eq!(h_event_left, h);

        // Check with right event detection.
        let h_event_right =
            right_event_detection::<f64>(&event, t_prev, &y_prev, &y_curr, h).unwrap();
        assert_eq!(h_event_right, h);
    }

    #[test]
    fn test_exact_event_detection_on_state_basic() {
        // Define the event with event function g(t,y) = y - 1.5.
        let event = Event::new(|_t: f64, y: &f64| *y - 1.5);

        // Define the ODE dy/dt = f(t,y) = y.
        let f = |_t: f64, y: &f64| *y;

        // Define the previous sample time and the corresponding state.
        let t_prev = 0.0;
        let y_prev = 1.0;

        // Define the step size.
        let h = 1.0;

        // Get the current state by integrating using the Euler method.
        let mut y_curr = y_prev;
        Euler::propagate(&f, t_prev, h, &mut y_curr);

        // Check with exact event detection.
        let h_event =
            exact_event_detection::<f64, Euler>(&f, &event, t_prev, &y_prev, &y_curr, h).unwrap();
        assert_equal_to_decimal!(h_event, 0.5, 15);
        check_event_function_value::<Euler>(&f, &event, t_prev, &y_prev, h_event);

        // Check with linear event detection.
        //  --> Note that in this case it ends up being identical because we integrated using the
        //      Euler method.
        let h_event_linear =
            linear_event_detection::<f64>(&event, t_prev, &y_prev, &y_curr, h).unwrap();
        assert_eq!(h_event_linear, 0.5);

        // Check with left event detection.
        let h_event_left =
            left_event_detection::<f64>(&event, t_prev, &y_prev, &y_curr, h).unwrap();
        assert_eq!(h_event_left, 0.0);

        // Check with right event detection.
        let h_event_right =
            right_event_detection::<f64>(&event, t_prev, &y_prev, &y_curr, h).unwrap();
        assert_eq!(h_event_right, h);
    }

    #[test]
    fn test_exact_event_detection_on_state_different_integrators_different_directions() {
        // Define the event with event function g(t,y) = y - 1.5.
        let event = Event::new(|_t: f64, y: &f64| *y - 1.5);

        // Define the ODE dy/dt = f(t,y) = y.
        let f = |_t: f64, y: &f64| *y;

        // Define the previous sample time and the corresponding state.
        let t_prev = 0.0;
        let y_prev = 1.0;

        // Define the current state.
        //  --> Note that this is the true solution, but it is ok because this value is only used
        //      for a sign check.
        let y_curr = 1.0_f64.exp();

        // Define the step size.
        let h = 1.0;

        // Solve for and check the event where y = 1.5 using the Euler method for propagation.
        //  --> Note that here the event is set to trigger both when g(t,y) is increasing AND when
        //      it is decreasing.
        assert!(matches!(event.direction, Direction::Either));
        let h_event =
            exact_event_detection::<f64, Euler>(&f, &event, t_prev, &y_prev, &y_curr, h).unwrap();
        assert_equal_to_decimal!(h_event, 0.5, 15);
        check_event_function_value::<Euler>(&f, &event, t_prev, &y_prev, h_event);

        // Perform the same check as above but using RK4 for propagation.
        let h_event =
            exact_event_detection::<f64, RK4>(&f, &event, t_prev, &y_prev, &y_curr, h).unwrap();
        assert_eq!(h_event, 0.40553040739646273);
        check_event_function_value::<RK4>(&f, &event, t_prev, &y_prev, h_event);

        // Now, update the event to only trigger when g(t,y) is decreasing, and verify that solving
        // for the event will now fail since g(t,y) is always increasing.
        let event = event.direction(Direction::Decreasing);
        assert!(
            exact_event_detection::<f64, Euler>(&f, &event, t_prev, &y_prev, &y_curr, h).is_none()
        );

        // Finally, update the event to ONLY trigger when g(t,y) is increasing, and verify that
        // solving for the event will get the same result as before.
        let event = event.direction(Direction::Increasing);
        let h_event =
            exact_event_detection::<f64, RK4>(&f, &event, t_prev, &y_prev, &y_curr, h).unwrap();
        assert_eq!(h_event, 0.40553040739646273);
    }
}
