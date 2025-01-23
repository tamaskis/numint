use crate::events::event::Event;
use crate::ode_state::ode_state_trait::OdeState;

/// TODO: this struct as it exists is a glorified vector of Events only ever used to rename events.
/// We should instead have a simple function to do this that is called during solve_ivp, OR flesh
/// out the functionality of this a little more (e.g. number of detections, etc. are tracked here).
/// I'm leaning towards the latter since that way we can share events between different functions
///
/// TODO: before anything else is done, we need to implement Index, IndexMut, and IntoIterator for
/// this struct
#[derive(Default)]
pub struct EventManager<T: OdeState> {
    /// Events.
    pub(crate) events: Vec<Event<T>>,
}

impl<T: OdeState + 'static> EventManager<T> {
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
    /// let event_manager = EventManager::new(vec![event_1, event_2]);
    ///
    /// assert_eq!(event_manager.as_slice()[0].get_name(), "Event 0");
    /// assert_eq!(event_manager.as_slice()[1].get_name(), "Event 1");
    /// ```
    pub fn new(events: Vec<Event<T>>) -> EventManager<T> {
        // Rename events without a name.
        let events = events
            .into_iter()
            .enumerate()
            .map(|(idx, event)| {
                if event.name.is_empty() {
                    event.name(format!("Event {idx}"))
                } else {
                    event
                }
            })
            .collect();

        // Construct the event manager.
        EventManager { events }
    }

    /// Get a slice view of the events in this event manager.
    ///
    /// # Returns
    ///
    /// Events managed by this event manager.
    ///
    /// # Example
    ///
    /// ```
    /// use numint::{Event, EventManager};
    ///
    /// // Create an event manager.
    /// let mut event_manager = EventManager::default();
    ///
    /// // Define two events.
    /// let event_1 = Event::new(|t: f64, y: &f64| y * t);
    /// let event_2 = Event::new(|t: f64, y: &f64| y / t);
    ///
    /// // Add the events to the event manager.
    /// event_manager.add(event_1);
    /// event_manager.add(event_2);
    ///
    /// // Get the events from the event manager.
    /// let events: &[Event<f64>] = event_manager.as_slice();
    ///
    /// // Check the number of events and their names.
    /// assert_eq!(events.len(), 2);
    /// assert_eq!(events[0].get_name(), "Event 0");
    /// assert_eq!(events[1].get_name(), "Event 1");
    /// ```
    pub fn as_slice(&self) -> &[Event<T>] {
        self.events.as_slice()
    }

    /// Add an event to this set of events.
    ///
    /// # Arguments
    ///
    /// * `event` - Event to add to this set of events.
    ///
    /// # Example
    ///
    /// ```
    /// use numint::{Event, EventManager};
    ///
    /// let mut event_manager = EventManager::default();
    ///
    /// let event = Event::new(|t: f64, y: &f64| y * t);
    ///
    /// event_manager.add(event);
    /// ```
    ///
    /// TODO: I don't think this is actually needed. If so, we can get rid of the update_name logic
    /// as well
    pub fn add(&mut self, mut event: Event<T>) {
        // If the event was not given a name, then name it "Event idx", where idx is the index of
        // the event in the events vector.
        if event.name.is_empty() {
            // Note that the event has not been added yet, so the index is the current length of the
            // vector.
            let idx = self.events.len();

            // Name the event.
            event.update_name(&format!("Event {idx}"));
        }

        // Add the event to the events vector.
        self.events.push(event);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default() {
        let event_manager = EventManager::<f64>::default();
        assert!(event_manager.events.is_empty());
    }

    #[test]
    fn test_new() {
        // Create events without names.
        let event_1 = Event::new(|t: f64, y: &f64| y * t);
        let event_2 = Event::new(|t: f64, y: &f64| y / t);

        // Construct the event manager.
        let event_manager = EventManager::new(vec![event_1, event_2]);

        // Verify the number of events.
        assert_eq!(event_manager.events.len(), 2);

        // Verify that the events were automatically named.
        assert_eq!(event_manager.events[0].name, "Event 0");
        assert_eq!(event_manager.events[1].name, "Event 1");
    }
}
