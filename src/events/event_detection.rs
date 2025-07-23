use crate::events::event::Event;
use crate::events::methods::EventDetectionMethod;
use crate::events::methods::{
    exact_event_detection, left_event_detection, linear_event_detection, right_event_detection,
};
use crate::integrators::integrator_trait::Integrator;
use crate::ode_state::ode_state_trait::OdeState;

/// Detect an event within a time step.
///
/// # Type Parameters
///
/// * `T` - ODE state type (any type implementing the [`OdeState`] trait).
/// * `I` - Integrator type (any type implementing the [`Integrator`] trait).
///
/// # Arguments
///
/// * `f` - Function defining the ordinary differential equation, `dy/dt = f(t,y)`. See the
///   [Overview](crate#overview) section in the documentation for more information.
/// * `event` - Event to detect.
/// * `t_prev` - Previous sample time.
/// * `y_prev` - Previous state (i.e. solution at the previous sample time).
/// * `y_curr` - Current state (i.e. solution at the current sample time).
/// * `h` - Step size.
///
/// # Returns
///
/// An option where where:
///
/// * `Some` - The step size required to advance from the current sample time to the event.
/// * `None` - Indicates that the event wasn't found.
pub(crate) fn detect_event<T: OdeState, I: Integrator<T>>(
    f: &impl Fn(f64, &T) -> T,
    event: &Event<T>,
    t_prev: f64,
    y_prev: &T,
    y_curr: &T,
    h: f64,
) -> Option<f64> {
    // Immediately return None if this event was not active at the beginning of this time step.
    if let Some(c) = &event.c {
        if !(c)(t_prev, y_curr) {
            return None;
        }
    }

    // Perform the requested method of event detection.
    match event.method {
        EventDetectionMethod::Exact => {
            exact_event_detection::<T, I>(&f, event, t_prev, y_prev, y_curr, h)
        }
        EventDetectionMethod::LinearInterpolation => {
            linear_event_detection::<T>(event, t_prev, y_prev, y_curr, h)
        }
        EventDetectionMethod::LeftInterpolation => {
            left_event_detection::<T>(event, t_prev, y_prev, y_curr, h)
        }
        EventDetectionMethod::RightInterpolation => {
            right_event_detection::<T>(event, t_prev, y_prev, y_curr, h)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RK4;
    use numtest::*;

    #[test]
    fn test_detect_event_active() {
        // Define the event with event function g(t,y) = t - 0.5.
        let event = Event::new(|t: f64, _y: &f64| t - 0.5);

        // Define the ODE dy/dt = f(t,y) = y.
        let f = |_t: f64, y: &f64| *y;

        // Define the previous sample time and the corresponding state.
        let t_prev = 0.0;
        let y_prev = 0.0;

        // Define the current state.
        let y_curr: f64 = 1.0;

        // Define the step size.
        let h = 1.0;

        // Check that the event was correctly detected.
        let h_event = detect_event::<f64, RK4>(&f, &event, t_prev, &y_prev, &y_curr, h).unwrap();
        assert_equal_to_decimal!(h_event, 0.5, 15);
    }

    #[test]
    fn test_detect_event_inactive() {
        // Define the event function g(t,y) = t - 0.5.
        let g = |t: f64, _y: &f64| t - 0.5;

        // Define the condition function c(t,y) = y > 10.
        let c = |_t: f64, y: &f64| *y > 10.0;

        // Define the event with event function g(t,y) = t - 0.5.
        let event = Event::new(g).c(c);

        // Define the ODE dy/dt = f(t,y) = y.
        let f = |_t: f64, y: &f64| *y;

        // Define the previous sample time and the corresponding state.
        let t_prev = 0.0;
        let y_prev = 0.0;

        // Define the current state.
        let y_curr: f64 = 1.0;

        // Define the step size.
        let h = 1.0;

        // Check that no event was detected since the event was not active.
        let h_event = detect_event::<f64, RK4>(&f, &event, t_prev, &y_prev, &y_curr, h);
        assert!(h_event.is_none());
    }
}
