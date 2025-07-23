use crate::events::event::Event;
use crate::events::event_detection::detect_event;
use crate::integrators::integrator_trait::Integrator;
use crate::ode_state::ode_state_trait::OdeState;
use std::ops::Index;

/// Event manager.
///
/// This struct is responsible for managing events, detecting them during the integration process,
/// and storing the times and values of the ODE state at which these events were detected.
///
/// # Type Parameters
///
/// * `T` - ODE state type (any type implementing the [`OdeState`] trait).
pub struct EventManager<'a, T: OdeState> {
    /// Events.
    events: Vec<&'a Event<T>>,

    /// Number of times each event was detected. The `i`th element corresponds to the number of
    /// times the `i`th event was detected.
    pub(crate) num_detections: Vec<usize>,

    /// Times at which each event was located.
    ///
    /// The `i`th element is a vector storing the times at which the `i`th event was located.
    t_located: Vec<Vec<f64>>,

    /// Values of the state variable at the times each event was located.
    ///
    /// The `i`th element is a vector storing the values of the state variable at the times the
    /// `i`th event was located.
    y_located: Vec<Vec<T>>,
}

impl<'a, T: OdeState> Index<usize> for EventManager<'a, T> {
    type Output = Event<T>;
    fn index(&self, index: usize) -> &Self::Output {
        self.events[index]
    }
}

impl<'a, T: OdeState + 'static> EventManager<'a, T> {
    /// Constructor.
    ///
    /// # Arguments
    ///
    /// * `events` - Vector of events.
    ///
    /// # Returns
    ///
    /// Event manager.
    ///
    /// # Note
    ///
    /// If an [`Event`] was not given a name, then the [`EventManager`] constructor will name it
    /// `"Event idx"` where `idx` is the index (0-based indexing) of the event in the `events`
    /// vector passed to this constructor.
    ///
    /// # Example
    ///
    /// ```
    /// use numint::{Event, EventManager};
    ///
    /// let event_1 = Event::new(|t: f64, y: &f64| y * t);
    /// let event_2 = Event::new(|t: f64, y: &f64| y / t);
    ///
    /// let event_manager = EventManager::new(vec![&event_1, &event_2]);
    /// ```
    pub fn new(events: Vec<&'a Event<T>>) -> EventManager<'a, T> {
        let num_events = events.len();
        EventManager {
            events,
            num_detections: vec![0; num_events],
            t_located: vec![vec![]; num_events],
            y_located: vec![vec![]; num_events],
        }
    }
}

impl<T: OdeState + 'static> EventManager<'_, T> {
    /// Detect the first occurence of an event within a time step.
    ///
    /// # Type Parameters
    ///
    /// * `I` - Integrator type (any type implementing the [`Integrator`] trait).
    ///
    /// # Arguments
    ///
    /// * `f` - Function defining the ordinary differential equation, `dy/dt = f(t,y)`. See the
    ///   [Overview](crate#overview) section in the documentation for more information.
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
    pub(crate) fn detect_events<I: Integrator<T>>(
        &mut self,
        f: &impl Fn(f64, &T) -> T,
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
        let events_fresh = &mut *self.events;

        // Try detecting each event.
        for event in events_fresh {
            h_events.push(detect_event::<T, I>(f, event, t_prev, y_prev, y_curr, h));
        }

        // Identify the first event that was detected.
        let (idx_event, h_event) = identify_first_event(&h_events);

        // Update the number of detections.
        if let Some(idx_event) = idx_event {
            self.num_detections[idx_event] += 1;
        }

        (idx_event, h_event)
    }

    /// Store the time and the value of the ODE state at an occurence of this event.
    ///
    /// # Arguments
    ///
    /// * `t` - Time at the occurence of this event.
    /// * `y` - Value of the ODE state at the occurence of this event.
    /// * `idx_event` - Index of the event in the `events` vector.
    ///
    /// # Note
    ///
    /// This method is only responsible for storing the time of an event and the corresponding value
    /// of the ODE state. [`crate::events::event_detection::detect_events`] is responsible for
    /// updating the number of time this event was detected.
    pub(crate) fn store(&mut self, t: f64, y: &T, idx_event: usize) {
        self.t_located[idx_event].push(t);
        self.y_located[idx_event].push(y.clone());
    }
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
    fn test_new() {
        // Create events without names.
        let event_1 = Event::new(|t: f64, y: &f64| y * t);
        let event_2 = Event::new(|t: f64, y: &f64| y / t);

        // Construct the event manager.
        let event_manager = EventManager::new(vec![&event_1, &event_2]);

        // Verify the number of events.
        assert_eq!(event_manager.events.len(), 2);

        // Verify that the events were automatically named.
        assert_eq!(event_manager[0].name, "");
        assert_eq!(event_manager[1].name, "");
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

        let mut event_manager = EventManager::new(vec![&event_1, &event_2]);

        // Check that the event was correctly detected.
        let (idx_event, h_event) =
            event_manager.detect_events::<RK4>(&f, t_prev, &y_prev, &y_curr, h);
        let idx_event = idx_event.unwrap();
        assert_eq!(idx_event, 1);
        assert_equal_to_decimal!(h_event.unwrap(), 0.25, 15);
        assert_eq!(event_manager.num_detections[idx_event], 1);
    }

    #[test]
    fn test_store() {
        let g = |t: f64, y: &f64| y * t;
        let event = Event::new(g);
        let mut event_manager = EventManager::new(vec![&event]);
        assert_eq!(event_manager.t_located[0], vec![]);
        assert_eq!(event_manager.y_located[0], vec![]);
        event_manager.store(0.5, &1.5, 0);
        assert_eq!(event_manager.t_located[0], vec![0.5]);
        assert_eq!(event_manager.y_located[0], vec![1.5]);
        event_manager.store(1.0, &5.0, 0);
        assert_eq!(event_manager.t_located[0], vec![0.5, 1.0]);
        assert_eq!(event_manager.y_located[0], vec![1.5, 5.0]);
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
