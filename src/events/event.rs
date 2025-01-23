use crate::events::methods::EventDetectionMethod;
use crate::ode_state::ode_state_trait::OdeState;
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

/// Event function, `g(t,y)`.
///
/// The event function is used to mathematically define some event that is a function of either the
/// current time ($t$), the current state ($y\in\mathbb{R}$, $\mathbf{y}\in\mathbb{R}^{p}$, or
/// $\mathbf{Y}\in\mathbb{R}^{p\times r}$), or both. An event occurs when the value of the event
/// function crosses 0.
///
/// # Note on zero-crossing definition
///
/// A root of the event function alone is insufficient to trigger the event; the event function must
/// also change sign at the root (i.e. it must cross 0). For example, the function
/// `g(t,y) = (t - 5)²` has a root at `t = 5`, but it only "touches" 0 there, and does not have a
/// sign change across the root.
///
/// # ODE State Type
///
/// The ODE state, `y`, can be either a scalar, vector, or matrix, as outlined in the table below.
/// See [`OdeState`] for more details.
///
/// | State Type | Event Function Signature |
/// | ---------- | ------------------------ |
/// | Scalar ($y\in\mathbb{R}$) | $g:\mathbb{R}\times\mathbb{R}\to\mathbb{R}$ |
/// | Vector ($\mathbf{y}\in\mathbb{R}^{p}$) | $g:\mathbb{R}\times\mathbb{R}^{p}\to\mathbb{R}$ |
/// | Matrix ($\mathbf{Y}\in\mathbb{R}^{p\times r}$) | $g:\mathbb{R}\times\mathbb{R}^{p\times r}\to\mathbb{R}$ |
pub type EventFunction<T> = Box<dyn Fn(f64, &T) -> f64>;

/// Condition function, `c(t,y)`.
///
/// The condition function is used to define when the ODE solver should be trying to detect an
/// event. Given the current time (`t`) and the current state (`y`), the condition function should
/// return `true` if we should try to detect this [`Event`], or `false` if we should _not_ try to
/// detect this [`Event`].
///
/// The most precise way to perform ODE event detection is to use a root solver to find where the
/// event function (see [`EventFunction`]) crosses 0. Since this requires updating the value of the
/// ODE state, `y`, this requires a large amount of intermediate propagations (using some
/// [`crate::Integrator`]), which in turn results in many additional evaluations of the ODE
/// `dy/dt = f(t,y)`. In large-scale practical applications (i.e. not toy problems), the evaluation
/// of `f(t,y)` is typically the most costly operation.
///
/// The purpose of the condition function is to only try to detect an event when the current time
/// and/or state satisfies some condition. For example, if we know some event will only feasibly
/// occur after time `t = 20`, it makes no sense to try to perform a costly event detection at times
/// before `t = 20`.
///
/// # ODE State Type
///
/// The ODE state, `y`, can be either a scalar, vector, or matrix, as outlined in the table below.
/// See [`OdeState`] for more details.
///
/// | State Type | Event Function Signature |
/// | ---------- | ------------------------ |
/// | Scalar ($y\in\mathbb{R}$) | $c:\mathbb{R}\times\mathbb{R}\to\mathbb{B}$ |
/// | Vector ($\mathbf{y}\in\mathbb{R}^{p}$) | $c:\mathbb{R}\times\mathbb{R}^{p}\to\mathbb{B}$ |
/// | Matrix ($\mathbf{Y}\in\mathbb{R}^{p\times r}$) | $c:\mathbb{R}\times\mathbb{R}^{p\times r}\to\mathbb{B}$ |
pub type ConditionFunction<T> = Box<dyn Fn(f64, &T) -> bool>;

/// State reset function, `s(t,y)`.
///
/// The state reset function is used to produce a new value of the state, `y`, that should be used.
/// When an event is detected, the state reset function is called to produce a new state from which
/// integration can be restarted.
///
/// For example, one could define a state reset function to swap flip the sign of some components of
/// the state when and event is detected (see the
/// [bouncing ball example](crate#bouncing-ball-example)).
///
/// Note that no state reset is performed at the instance of an event where the ODE solver is
/// terminated.
///
/// # ODE State Type
///
/// The ODE state, `y`, can be either a scalar, vector, or matrix, as outlined in the table below.
/// See [`OdeState`] for more details.
///
/// | State Type | Event Function Signature |
/// | ---------- | ------------------------ |
/// | Scalar ($y\in\mathbb{R}$) | $s:\mathbb{R}\times\mathbb{R}\to\mathbb{R}$ |
/// | Vector ($\mathbf{y}\in\mathbb{R}^{p}$) | $s:\mathbb{R}\times\mathbb{R}^{p}\to\mathbb{R}^{p}$ |
/// | Matrix ($\mathbf{Y}\in\mathbb{R}^{p\times r}$) | $s:\mathbb{R}\times\mathbb{R}^{p\times r}\to\mathbb{R}^{p\times r}$ |
pub type StateResetFunction<T> = Box<dyn Fn(f64, &T) -> T>;

/// ODE solver termination condition.
///
/// Defines the number of times an event must be detected to terminate an ODE solver.
pub struct Termination {
    /// Number of detections of an event required to terminate the ODE solver.
    ///
    /// * If `0`, integration will not terminate, but integration will be restarted from the event.
    /// * If `1`, integration will be terminated at the first instance of the event.
    /// * If `n` where `n` is an integer greater than `1`, then integration will be terminated on
    ///   the `n`th occurence of the event.
    pub(crate) num_detections: usize,
}

impl Termination {
    /// Constructor.
    ///
    /// # Arguments
    ///
    /// * `num_detections` - Number of detections of this event required to terminate the ODE
    ///   solver.
    ///
    ///     * If `0`, integration will not terminate, but integration will be restarted from this
    ///       event.
    ///     * If `1`, integration will be terminated at the first instance of this event.
    ///     * If `n` where `n` is an integer greater than `1`, then integration will be terminated
    ///       on the `n`th occurence of this event.
    ///
    /// # Returns
    ///
    /// ODE solver termination condition.
    pub fn new(num_detections: usize) -> Termination {
        Termination { num_detections }
    }
}

impl Default for Termination {
    /// Default constructor.
    ///
    /// Sets the number of event detections required to terminate the ODE solver to 1.
    fn default() -> Termination {
        Termination { num_detections: 1 }
    }
}

/// Event.
///
/// # Overview
///
/// An event is a condition that is monitored during numerical integration of an ODE. When the
/// condition is met, the ODE solver detects it and takes appropriate action. Events are commonly
/// used for handling discontinuities, stopping conditions, or state-dependent modifications of the
/// ODE being solved.
///
/// # Defining Events
///
/// At the core of an event is the event function, `g(t,y)`. The event corresponding to an event
/// function occurs when the event function evaluates to 0 (i.e. an event is the root/zero of an
/// event function). See [`EventFunction`] for more details.
///
/// In addition to storing an event function `g(t,y)` (something common to most ODE solvers), the
/// [`Event`] struct also supports two additional types of functions that aim to improve efficiency
/// while also making it easier to express more nuanced problems:
///
/// * [`ConditionFunction`] - Defines when the solver should even bother checking for the existence
///   of an event.
/// * [`StateResetFunction`] - Can be used to reset the value of the ODE state at the occurence of
///   an event.
///
/// There are also many options for controlling the behavior of the ODE solver at an event:
///
/// * [`Direction`] - Defines whether the event function must be decreasing or increasing (or
///   either) for a root of the event function to be considered an event.
/// * [`Termination`] - Defines how many times this event must be detected to terminate an ODE
///   solver.
/// * [`EventDetectionMethod`] - Event detection method, see [`EventDetectionMethod`]  for more
///   details.
///
/// To actually define an event, you can use the [`Event::new`] method. There are dedicated methods
/// for setting additional options, such as a condition function ([`Event::c`]), a state reset
/// function ([`Event::s`]), the event detection method ([`Event::method`]), the direction of the
/// event function ([`Event::direction`]), the termination condition ([`Event::termination`]), and
/// the name of the event ([`Event::name`]).
pub struct Event<T: OdeState> {
    /// Event function, `g(t,y)`. See [`EventFunction`] for more details.
    pub(crate) g: EventFunction<T>,

    /// Condition function, `c(t,y)`. See [`ConditionFunction`] for more details.
    pub(crate) c: Option<ConditionFunction<T>>,

    /// State reset function, `s(t,y)`. See [`StateResetFunction`] for more details.
    pub(crate) s: Option<StateResetFunction<T>>,

    /// Direction of a zero crossing to trigger the event. See [`Direction`] for more information.
    pub(crate) direction: Direction,

    /// ODE solver termination condition. See [`Termination`] for more information.
    pub(crate) termination: Termination,

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

impl<T: OdeState + 'static> Event<T> {
    /// Constructor.
    ///
    /// # Arguments
    ///
    /// * `g` - Event function.
    ///
    /// # Returns
    ///
    /// Event.
    ///
    /// # Note
    ///
    /// This constructor defaults the following:
    ///
    /// * `direction` to [`Direction::Either`]
    /// * `termination` to [`Termination::default()`]
    /// * `method` to [`EventDetectionMethod::Exact`]
    /// * `name` to an empty string
    ///
    /// # Example
    ///
    /// ```
    /// use numint::Event;
    ///
    /// let event = Event::new(|t: f64, y: &f64| y * t);
    /// ```
    pub fn new(g: impl Fn(f64, &T) -> f64 + 'static) -> Self {
        Event {
            g: Box::new(g),
            c: None,
            s: None,
            direction: Direction::Either,
            termination: Termination::default(),
            method: EventDetectionMethod::Exact,
            name: "".to_string(),
            num_detections: 0,
            t_located: vec![],
            y_located: vec![],
            _phantom: PhantomData,
        }
    }

    /// Set the condition function for this event.
    ///
    /// # Arguments
    ///
    /// * `c` - Condition function. See [`ConditionFunction`] for more information.
    ///
    /// # Example
    ///
    /// ```
    /// use numint::Event;
    ///
    /// // Define an event.
    /// let event = Event::new(|t: f64, y: &f64| y * t);
    ///
    /// // Set the condition function.
    /// let event = event.c(|t: f64, _y: &f64| t > 20.0);
    /// ```
    pub fn c(mut self, c: impl Fn(f64, &T) -> bool + 'static) -> Self {
        self.c = Some(Box::new(c));
        self
    }

    /// Set the state reset function for this event.
    ///
    /// # Arguments
    ///
    /// * `s` - State reset function. See [`StateResetFunction`] for more information.
    ///
    /// # Example
    ///
    /// ```
    /// use numint::Event;
    ///
    /// // Define an event.
    /// let event = Event::new(|t: f64, y: &f64| y * t);
    ///
    /// // Set the state reset function.
    /// let event = event.s(|_t: f64, y: &f64| -*y);
    /// ```
    pub fn s(mut self, s: impl Fn(f64, &T) -> T + 'static) -> Self {
        self.s = Some(Box::new(s));
        self
    }

    /// Set the direction of a zero crossing needed to trigger this event.
    ///
    /// # Arguments
    ///
    /// * `direction` - Direction of a zero crossing to trigger the event. See [`Direction`] for
    ///   more information.
    ///
    /// # Returns
    ///
    /// The event with the the updated `direction`.
    ///
    /// # Example
    ///
    /// ```
    /// use numint::{Direction, Event};
    ///
    /// // Define an event.
    /// let event = Event::new(|t: f64, y: &f64| y * t);
    ///
    /// // Update the event so it is only triggered when the event function is decreasing.
    /// let event = event.direction(Direction::Decreasing);
    /// ```
    pub fn direction(mut self, direction: Direction) -> Self {
        self.direction = direction;
        self
    }

    /// Set the ODE solver termination condition.
    ///
    /// # Arguments
    ///
    /// * `termination` - ODE solver termination condition. See [`Termination`].
    ///
    /// # Returns
    ///
    /// The event with the the updated ODE solver termination condition.
    ///
    /// # Example
    ///
    /// ```
    /// use numint::{Event, Termination};
    ///
    /// // Define an event.
    /// let event = Event::new(|t: f64, y: &f64| y * t);
    ///
    /// // Update the event so it only terminates the ODE solver the 5th time that it is detected.
    /// let event = event.termination(Termination::new(5));
    /// ```
    pub fn termination(mut self, termination: Termination) -> Self {
        self.termination = termination;
        self
    }

    /// Set the event detection method.
    ///
    /// # Arguments
    ///
    /// * `method` - Event detection method. See [`EventDetectionMethod`] for options.
    ///
    /// # Returns
    ///
    /// The event with the updated event detection method.
    ///
    /// # Example
    ///
    /// ```
    /// use numint::{Event, EventDetectionMethod};
    ///
    /// // Define an event.
    /// let event = Event::new(|t: f64, y: &f64| y * t);
    ///
    /// // Set the event detection method.
    /// let event = event.method(EventDetectionMethod::Exact);
    /// ```
    pub fn method(mut self, method: EventDetectionMethod) -> Self {
        self.method = method;
        self
    }

    /// Set the name of the event.
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the event.
    ///
    /// # Returns
    ///
    /// The event with the updated name.
    ///
    /// # Example
    ///
    /// ```
    /// use numint::{Event, EventDetectionMethod};
    ///
    /// // Define an event.
    /// let event = Event::new(|t: f64, y: &f64| y * t);
    ///
    /// // Set the name of the event.
    /// let event = event.name(String::from("My Event"));
    /// ```
    pub fn name(mut self, name: String) -> Self {
        self.name = name;
        self
    }

    /// Update the name of the event.
    ///
    /// # Arguments
    ///
    /// * `name` - New name of the event.
    ///
    /// # Note
    ///
    /// This method is used to update the name of the event after it has been created in the case
    /// that it was not set during construction. This is useful for renaming events after they have
    /// been added to an event manager (we cannot just use [`Event::name`] because it consumes the
    /// event, and we cannot implement [`Clone`] for `Event` because it contains a [`Box`]).
    pub(crate) fn update_name(&mut self, name: &str) {
        self.name = name.to_string();
    }

    /// Get the name of the event.
    ///
    /// # Returns
    ///
    /// Name of the event.
    ///
    /// # Example
    ///
    /// ```
    /// use numint::Event;
    ///
    /// // Define an event.
    /// let event = Event::new(|t: f64, y: &f64| y * t)
    ///     .name(String::from("My Event"));
    ///
    /// // Get the name of the event.
    /// assert_eq!(event.get_name(), "My Event");
    /// ```
    pub fn get_name(&self) -> &str {
        &self.name
    }

    /// Store the time and the value of the ODE state at an occurence of this event.
    ///
    /// # Arguments
    ///
    /// * `t` - Time at the occurence of this event.
    /// * `y` - Value of the ODE state at the occurence of this event.
    ///
    /// # Note
    ///
    /// This method is only responsible for storing the time of an event and the corresponding value
    /// of the ODE state. [`crate::events::event_detection::detect_events`] is responsible for
    /// updating the number of time this event was detected.
    pub(crate) fn store(&mut self, t: f64, y: &T) {
        self.t_located.push(t);
        self.y_located.push(y.clone());
    }
}

#[cfg(test)]
mod termination_tests {
    use super::*;

    #[test]
    fn test_new() {
        let termination = Termination::new(2);
        assert_eq!(termination.num_detections, 2);
    }

    #[test]
    fn test_default() {
        let termination = Termination::default();
        assert_eq!(termination.num_detections, 1);
    }
}

#[cfg(test)]
mod event_tests {
    use super::*;

    #[test]
    fn test_new() {
        let g = |t: f64, y: &f64| y * t;
        let event = Event::new(g);
        assert_eq!((event.g)(2.0, &3.0), 6.0);
        assert!(event.c.is_none());
        assert!(event.s.is_none());
        assert!(matches!(event.direction, Direction::Either));
        assert_eq!(event.termination.num_detections, 1);
        assert!(matches!(event.method, EventDetectionMethod::Exact));
        assert_eq!(event.name, "");
        assert_eq!(event.num_detections, 0);
        assert_eq!(event.t_located, vec![]);
        assert_eq!(event.y_located, vec![]);
    }

    #[test]
    fn test_c() {
        let g = |t: f64, y: &f64| y * t;
        let c = |t: f64, _y: &f64| t > 20.0;
        let event = Event::new(g).c(c);
        assert!((event.c.as_deref().unwrap())(21.0, &0.0));
        assert!(!(event.c.unwrap())(19.0, &0.0));
    }

    #[test]
    fn test_s() {
        let g = |t: f64, y: &f64| y * t;
        let s = |_t: f64, y: &f64| -y;
        let event = Event::new(g).s(s);
        assert_eq!((event.s.unwrap())(5.0, &3.0), -3.0);
    }

    #[test]
    fn test_direction() {
        let g = |t: f64, y: &f64| y * t;
        let direction = Direction::Decreasing;
        let event = Event::new(g).direction(direction);
        assert!(matches!(event.direction, Direction::Decreasing));
    }

    #[test]
    fn test_termination() {
        let g = |t: f64, y: &f64| y * t;
        let termination = Termination::new(5);
        let event = Event::new(g).termination(termination);
        assert_eq!(event.termination.num_detections, 5);
    }

    #[test]
    fn test_method() {
        let g = |t: f64, y: &f64| y * t;
        let event = Event::new(g).method(EventDetectionMethod::LinearInterpolation);
        assert!(matches!(
            event.method,
            EventDetectionMethod::LinearInterpolation
        ));
    }

    #[test]
    fn test_name() {
        let g = |t: f64, y: &f64| y * t;
        let name = String::from("My Event");
        let event = Event::new(g).name(name);
        assert_eq!(event.name, "My Event");
    }

    #[test]
    fn test_get_name() {
        let g = |t: f64, y: &f64| y * t;
        let name = String::from("My Event");
        let event = Event::new(g).name(name);
        assert_eq!(event.get_name(), "My Event");
    }

    #[test]
    fn test_store() {
        let g = |t: f64, y: &f64| y * t;
        let mut event = Event::new(g);
        assert_eq!(event.t_located, vec![]);
        assert_eq!(event.y_located, vec![]);
        event.store(0.5, &1.5);
        assert_eq!(event.t_located, vec![0.5]);
        assert_eq!(event.y_located, vec![1.5]);
        event.store(1.0, &5.0);
        assert_eq!(event.t_located, vec![0.5, 1.0]);
        assert_eq!(event.y_located, vec![1.5, 5.0]);
    }
}
