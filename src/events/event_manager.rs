use crate::events::event::Event;
use crate::ode_state::ode_state_trait::OdeState;
use std::ops::Index;

/// TODO: this struct as it exists is a glorified vector of Events only ever used to rename events.
/// We should instead have a simple function to do this that is called during solve_ivp, OR flesh
/// out the functionality of this a little more (e.g. number of detections, etc. are tracked here).
/// I'm leaning towards the latter since that way we can share events between different functions
///
/// TODO: before anything else is done, we need to implement Index, IndexMut, and IntoIterator for
/// this struct
pub struct EventManager<'a, T: OdeState> {
    /// Events.
    events: Vec<&'a Event<T>>,

    /// Number of times each event was detected. The `i`th element corresponds to the number of
    /// times the `i`th event was detected.
    #[allow(dead_code)] // TODO remove when used
    num_detections: Vec<usize>,

    /// Times at which each event was located.
    ///
    /// The `i`th element is a vector storing the times at which the `i`th event was located.
    #[allow(dead_code)] // TODO remove when used
    t_located: Vec<Vec<f64>>,

    /// Values of the state variable at the times each event was located.
    ///
    /// The `i`th element is a vector storing the values of the state variable at the times the
    /// `i`th event was located.
    #[allow(dead_code)] // TODO remove when used
    y_located: Vec<Vec<T>>,
}

/// TODO.
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
    ///
    /// assert_eq!(event_manager[0].get_name(), "");
    /// assert_eq!(event_manager[1].get_name(), "");
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
