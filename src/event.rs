use crate::integration_methods::integration_method_trait::IntegrationMethod;
use crate::ode_state::ode_state_trait::OdeState;
use rootfinder::{root_bisection, Interval};
use std::marker::PhantomData;

/// Direction of a zero crossing to trigger the event.
pub enum Direction {
    /// Event will only trigger when the event function, `g(t,y)`, goes from negative to positive.
    Increasing,

    /// Event will only trigger when the event function, `g(t,y)`, goes from positive to negative.
    Decreasing,

    /// Event will trigger both when the event function, `g(t,y)`, goes from negative to positive
    /// and when it goes from positive to negative.
    Either,
}

/// Event detection method.
pub enum EventDetectionMethod {
    /// Use the bisection method (via the [`rootfinder::root_bisection`] function) to find the exact
    /// time of the event (within machine precision).
    Exact,
    // /// TODO explain how linear interpolation is used instead of the propagation method.
    // LinearInterpolation,

    // /// Identify an event as the last sample time before the event occurs (i.e. left-interpolation).
    // LeftInterpolation,

    // /// Identify an event as the last sample time before the event occurs (i.e.
    // /// right-interpolation).
    // RightInterpolation,
}

/// Event.
///
/// TODO detailed description
pub struct Event<T: OdeState> {
    /// Event function, `g(t,y)`. TODO.
    pub(crate) g: Box<dyn Fn(f64, &T) -> f64>,

    /// Condition function `c(t,y)`. TODO.
    pub(crate) c: Box<dyn Fn(f64, &T) -> bool>,

    /// State reset function, `r(y)`. TODO.
    pub(crate) r: Option<Box<dyn Fn(&T) -> T>>,

    /// Direction of a zero crossing to trigger the event. See [`Direction`] for more information.
    pub(crate) direction: Direction,

    /// TODO.
    ///
    /// * If `0`, integration will not terminate, but integration will be restarted from this event.
    ///   TODO improve this description?
    /// * If `1`, integration will be terminated at the first instance of this event.
    /// * If `n` where `n` is an integer greater than 1, then integration will be terminated on the
    ///   `n`th occurence of this event.
    pub(crate) terminal: usize,

    /// Event detection method. See [`EventDetectionMethod`] for more information.
    pub(crate) method: EventDetectionMethod,

    /// Name of the event.
    pub(crate) name: String,

    /// Number of times this event has been detected.
    pub(crate) num_detections: usize,

    /// Times at which the event was located.
    pub(crate) t_located: Vec<f64>,

    /// Values of the ODE state at the times at which the event was located.
    pub(crate) y_located: Vec<T>,

    /// Tracks unused generic parameters.
    _phantom: PhantomData<T>,
}

/// TODO: just switch to option instead of using this?
fn default_condition_function<T: OdeState>(_: f64, _: &T) -> bool {
    true
}

impl<T: OdeState + 'static> Event<T> {
    /// Constructor.
    ///
    /// # Arguments
    ///
    /// * `g` - Event function.
    ///
    /// # Note
    ///
    /// This constructor defaults the following TODO cleanup:
    ///
    /// * `direction` to [`Direction::Either`]
    /// * `terminal` to `1`
    /// * `method` to [`EventDetectionMethod::Exact`]
    pub fn new(g: impl Fn(f64, &T) -> f64 + 'static) -> Event<T> {
        Event {
            g: Box::new(g),
            c: Box::new(default_condition_function),
            r: None,
            direction: Direction::Either,
            terminal: 1,
            method: EventDetectionMethod::Exact,
            name: "".to_string(),
            num_detections: 0,
            t_located: vec![],
            y_located: vec![],
            _phantom: PhantomData,
        }
    }

    pub fn c(&mut self, c: impl Fn(f64, &T) -> bool + 'static) {
        self.c = Box::new(c);
    }
    pub fn direction(mut self, direction: Direction) -> Self {
        self.direction = direction;
        self
    }

    pub fn terminal(mut self, terminal: usize) -> Self {
        self.terminal = terminal;
        self
    }

    pub fn method(mut self, method: EventDetectionMethod) -> Self {
        self.method = method;
        self
    }

    // TODO: this follows builder pattern
    pub fn name(mut self, name: String) -> Self {
        self.name = name;
        self
    }

    /// TODO actually use this method
    pub(crate) fn store(&mut self, t: f64, y: &T) {
        self.t_located.push(t);
        self.y_located.push(y.clone());
        self.num_detections += 1;
    }
}

/// Detect an event within a time step.
///
/// # Arguments
///
/// * `f` - Function defining the ordinary differential equation, `dy/dt = f(t,y)`. See the
///         [Overview](crate#overview) section in the documentation for more information.
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
fn detect_event<T: OdeState, M: IntegrationMethod<T>>(
    f: &impl Fn(f64, &T) -> T,
    event: &Event<T>,
    t_prev: f64,
    y_prev: &T,
    y_curr: &T,
    h: f64,
) -> Option<f64> {
    // Immediately return None if this event was not active at the beginning of this time step.
    if !(event.c)(t_prev, y_curr) {
        return None;
    }

    // TODO.
    match event.method {
        EventDetectionMethod::Exact => {
            exact_event_detection::<T, M>(&f, event, t_prev, y_prev, y_curr, h)
        }
    }
}

/// Detect the first occurence of an event within a time step.
///
/// # Type Parameters
///
/// TODO
///
/// # Arguments
///
/// * `f` - Function defining the ordinary differential equation, `dy/dt = f(t,y)`. See the
///         [Overview](crate#overview) section in the documentation for more information.
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
///                detected event.
///     * `None` - Indicates that no events were detected.
pub(crate) fn detect_events<T: OdeState, M: IntegrationMethod<T>>(
    f: &impl Fn(f64, &T) -> T,
    events: &mut Vec<Event<T>>,
    t_prev: f64,
    y_prev: &T,
    y_curr: &T,
    h: f64,
) -> (Option<usize>, Option<f64>) {
    // Initialize a vector to store the step sizes required to reach each of the events, or None
    // for events that aren't detected.
    let mut h_events: Vec<Option<f64>> = vec![];

    // Try detecting each event.
    for i in 0..events.len() {
        h_events.push(detect_event::<T, M>(
            f, &events[i], t_prev, y_prev, y_curr, h,
        ));
    }

    // Identify the first event that was detected.
    let (idx_event, h_event) = identify_first_event(&h_events);

    // Update the number of detections.
    if let Some(idx_event) = idx_event {
        events[idx_event].num_detections += 1;
    }

    (idx_event, h_event)
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

/// Helper function to identify the first detected event.
///
/// # Arguments
///
/// * `h_events` - A slice of options where:
///
///     * `Some` - The step size required to advance from the current sample time to the detected
///                event.
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
///                detected event.
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

    #[test]
    fn test_new() {
        let g = |t: f64, y: &f64| y * t;
        let event = Event::new(g);
        assert!(matches!(event.direction, Direction::Either));
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
}
