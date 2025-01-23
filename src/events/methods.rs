use crate::events::event::Direction;
use crate::events::event::Event;
use crate::integration_methods::integration_method_trait::IntegrationMethod;
use crate::ode_state::ode_state_trait::OdeState;
use rootfinder::{root_bisection, Interval};

/// Event detection method.
pub enum EventDetectionMethod {
    /// Use the bisection method (via the [`rootfinder::root_bisection`] function) to find the exact
    /// time of the event (within machine precision).
    Exact,

    /// TODO explain how linear interpolation is used instead of the propagation method.
    LinearInterpolation,

    /// Identify an event as the last sample time before the event occurs (i.e. left-interpolation).
    LeftInterpolation,

    /// Identify an event as the last sample time before the event occurs (i.e.
    /// right-interpolation).
    RightInterpolation,
}

/// TODO.
pub(crate) fn exact_event_detection<T: OdeState, M: IntegrationMethod<T>>(
    f: &impl Fn(f64, &T) -> T,
    event: &Event<T>,
    t_prev: f64,
    y_prev: &T,
    y_curr: &T,
    h: f64,
) -> Option<f64> {
    // Check the direction of g(t,y) if the event is only triggered in one direction.
    if !matches!(event.direction, Direction::Either) {
        // Evaluate the event function at the previous time and state.
        let g_prev = (event.g)(t_prev, y_prev);

        // Evaluate the event function at the current time and state.
        let g_curr = (event.g)(t_prev + h, y_curr);

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

    // Evaluate the event function at the previous time and state.
    let g_prev = (event.g)(t_prev, y_prev);

    // Evaluate the event function at the current time and state.
    let g_curr = (event.g)(t_prev + h, y_curr);

    // Redefine the event function as a function of the time step, h.
    //  --> g(t,y) becomes gₕ(h)
    let gh = |h: f64| {
        // Make a copy of the previous state.
        let mut y_copy = y_prev.clone();

        // Propagate the state from the previous time (t_prev) to our current estimate for the event
        // time (t_prev + h).
        M::propagate(&f, t_prev, h, &mut y_copy);

        // Call the event function at the propagated state.
        (event.g)(t_prev + h, &y_copy)
    };

    // Solve for the step size required to advance from the current time to the event.
    //  --> If root_bisection returns a solver error, it is because there is no zero crossing,
    //      so we cannot detect the event.
    //  --> If root_bisection returns some value, it is guaranteed to have found a zero crossing
    //      (this guarantee comes from the fact that we are not rebracketing).
    match root_bisection(&gh, Interval::new(0.0, h), None, None) {
        Ok(value) => Some(value),
        Err(_) => None,
    }
}

/// TODO.
pub(crate) fn linear_event_detection<T: OdeState>(
    event: &Event<T>,
    t_prev: f64,
    y_prev: &T,
    y_curr: &T,
    h: f64,
) -> Option<f64> {
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

    // Compute `h_new` such that the step to where `g(t_prev + h_new) == 0`, assuming g(t) is linear
    // between `g_prev` and `g_curr`.
    Some(-h * g_prev / (g_curr - g_prev))
}

/// TODO.
pub(crate) fn left_event_detection<T: OdeState>(
    event: &Event<T>,
    t_prev: f64,
    y_prev: &T,
    y_curr: &T,
    h: f64,
) -> Option<f64> {
    // Evaluate the event function at the previous time and state.
    let g_prev = (event.g)(t_prev, y_prev);

    // Check the direction of g(t,y) if the event is only triggered in one direction.
    if !matches!(event.direction, Direction::Either) {
        // Evaluate the event function at the current time and state.
        let g_curr = (event.g)(t_prev + h, y_curr);

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

    // Return `0.0` since we want to TODO.
    Some(0.0)
}

/// TODO.
pub(crate) fn right_event_detection<T: OdeState>(
    event: &Event<T>,
    t_prev: f64,
    y_prev: &T,
    y_curr: &T,
    h: f64,
) -> Option<f64> {
    // Evaluate the event function at the current time and state.
    let g_curr = (event.g)(t_prev + h, y_curr);

    // Check the direction of g(t,y) if the event is only triggered in one direction.
    if !matches!(event.direction, Direction::Either) {
        // Evaluate the event function at the previous time and state.
        let g_prev = (event.g)(t_prev, y_prev);

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

    // Return `h` since we want to TODO.
    Some(h)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Euler, RK4};
    use numtest::*;

    /// Check the value of the event function at the event.
    ///
    /// # Arguments
    ///
    /// * `f` - Function defining the ordinary differential equation, `dy/dt = f(t,y)`. See the
    ///         [Overview](crate#overview) section in the documentation for more information.
    /// * `event` - Event.
    /// * `t_prev` - Previous sample time.
    /// * `y_prev` - Previous state (i.e. solution at the previous sample time).
    /// * `h_event` - The step size required to advance from the current sample time to the event.
    fn check_event_function_value<M: IntegrationMethod<f64>>(
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
        M::propagate(f, t_prev, h_event, &mut y_event);

        // Get the time of the event.
        let t_event = t_prev + h_event;

        // Evaluate the event function at the event.
        let g_event = (event.g)(t_event, &y_event);

        // Verify that the value of the event function at the event is 0 (to within 2 times the
        // machine epsilon).
        assert_equal_to_atol!(g_event, 0.0, 2.0 * f64::EPSILON);
    }

    // TODO test more edge cases
    #[test]
    fn test_event_detection_on_time() {
        // Define the event with event function g(t,y) = 0.7.
        let event = Event::new(|t: f64, _y: &f64| t.sqrt() - 0.5);

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

    // TODO rename, add checks with all event detection methods
    #[test]
    fn test_exact_event_detection_state() {
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

    // TODO rename
    #[test]
    fn test_exact_event_detection_state_2() {
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
    }
}
