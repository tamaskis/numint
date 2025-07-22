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
fn detect_event<T: OdeState, I: Integrator<T>>(
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

/// Detect the first occurence of an event within a time step.
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
/// * `events` - Events to detect.
/// * `t_prev` - Previous sample time.
/// * `y_prev` - Previous state (i.e. solution at the previous sample time).
/// * `y_curr` - Current state (i.e. solution at the current sample time).
/// * `h` - Step size.
///
/// # Returns
///
/// * `idx_event` - An option where:
///
///     * `Some` - The index of `events` corresponding to the first event that was detected.
///     * `None` - Indicates that no events were detected.
///
/// * `h_event` - An option where:
///
///     * `Some` - The step size required to advance from the current sample time to the first
///       detected event.
///     * `None` - Indicates that no events were detected.
pub(crate) fn detect_events<T: OdeState, I: Integrator<T>>(
    f: &impl Fn(f64, &T) -> T,
    events: &mut [Event<T>],
    t_prev: f64,
    y_prev: &T,
    y_curr: &T,
    h: f64,
) -> (Option<usize>, Option<f64>) {
    // Initialize a vector to store the step sizes required to reach each of the events, or None
    // for events that aren't detected.
    let mut h_events: Vec<Option<f64>> = vec![];

    // Get a fresh reborrow of "events" (into_iter will perform a move since Event does not
    // implement copy, and we will need to modify an event in a subsequent step).
    let events_fresh = &mut *events;

    // Try detecting each event.
    for event in events_fresh {
        h_events.push(detect_event::<T, I>(f, event, t_prev, y_prev, y_curr, h));
    }

    // Identify the first event that was detected.
    let (idx_event, h_event) = identify_first_event(&h_events);

    // Update the number of detections.
    if let Some(idx_event) = idx_event {
        events[idx_event].num_detections += 1;
    }

    (idx_event, h_event)
}

/// Helper function to identify the first detected event.
///
/// # Arguments
///
/// * `h_events` - A slice of options where:
///
///     * `Some` - The step size required to advance from the current sample time to the detected
///       event.
///     * `None` - Indicates that the event was not detected.
///
/// # Returns
///
/// * `idx_event` - An option where:
///
///     * `Some` - The index of `h_events` corresponding to the first event that was detected.
///     * `None` - Indicates that no events were detected.
///
/// * `h_event` - An option where:
///
///     * `Some` - The step size required to advance from the current sample time to the first
///       detected event.
///     * `None` - Indicates that no events were detected.
fn identify_first_event(h_events: &[Option<f64>]) -> (Option<usize>, Option<f64>) {
    // Initialize both the index and the corresponding step size to None.
    let mut idx_min: Option<usize> = None;
    let mut h_min = None;

    // Iterate over each event.
    for (idx, h) in h_events.iter().enumerate() {
        // Skip to the next iteration if the current event wasn't detected.
        if h.is_none() {
            continue;
        }

        // Extract the step size to reach this event.
        let h = *h.as_ref().unwrap();

        // If the step size to reach this event is shorter than the step size to reach the currently
        // tracked first event, then we need to update the first event to be this event.
        if h_min.is_none() || h < h_min.unwrap() {
            idx_min = Some(idx);
            h_min = Some(h);
        }
    }

    (idx_min, h_min)
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

    #[test]
    fn test_detect_events_both_events_detected() {
        // Define an event with the event function g(t,y) = t - 0.5.
        let event_1 = Event::new(|t: f64, _y: &f64| t - 0.5);

        // Define a second event with event function g(t,y) = t - 0.25.
        let event_2 = Event::new(|t: f64, _y: &f64| t - 0.25);

        // Define the ODE dy/dt = f(t,y) = y.
        let f = |_t: f64, y: &f64| *y;

        // Define the previous sample time and the corresponding state.
        let t_prev = 0.0;
        let y_prev = 0.0;

        // Define the current state.
        let y_curr: f64 = 1.0;

        // Define the step size.
        let h = 1.0;

        let mut events = vec![event_1, event_2];

        // Check that the event was correctly detected.
        let (idx_event, h_event) =
            detect_events::<f64, RK4>(&f, &mut events, t_prev, &y_prev, &y_curr, h);
        let idx_event = idx_event.unwrap();
        assert_eq!(idx_event, 1);
        assert_equal_to_decimal!(h_event.unwrap(), 0.25, 15);
        assert_eq!(events[idx_event].num_detections, 1);
    }

    #[test]
    fn test_identify_first_event_1() {
        let (idx_event, h_event) = identify_first_event(&[None]);
        assert!(idx_event.is_none());
        assert!(h_event.is_none());
    }

    #[test]
    fn test_identify_first_event_2() {
        let (idx_event, h_event) = identify_first_event(&[Some(1.0)]);
        assert_eq!(idx_event, Some(0));
        assert_eq!(h_event, Some(1.0));
    }

    #[test]
    fn test_identify_first_event_3() {
        let (idx_event, h_event) = identify_first_event(&[None, Some(1.0)]);
        assert_eq!(idx_event, Some(1));
        assert_eq!(h_event, Some(1.0));
    }

    #[test]
    fn test_identify_first_event_4() {
        let (idx_event, h_event) = identify_first_event(&[Some(1.0), None]);
        assert_eq!(idx_event, Some(0));
        assert_eq!(h_event, Some(1.0));
    }

    #[test]
    fn test_identify_first_event_5() {
        let (idx_event, h_event) = identify_first_event(&[Some(1.0), None, Some(0.5)]);
        assert_eq!(idx_event, Some(2));
        assert_eq!(h_event, Some(0.5));
    }
}
