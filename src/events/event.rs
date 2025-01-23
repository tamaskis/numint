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
/// function becomes 0.
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
/// [`IntegrationMethod`]), which in turn results in many additional evaluations of the ODE
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
/// TODO more details.
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

/// Event.
///
/// TODO detailed description
///
/// In addition to storing an event function $g(t,\mathbf{y})$ (something common to most ODE
/// solvers), the `Event` struct also supports two additional types of functions that aim to improve
/// efficiency and also make it easier to express more nuanced initial value problems.
pub struct Event<T: OdeState> {
    /// Event function, `g(t,y)`. See [`EventFunction`] for more details.
    pub(crate) g: EventFunction<T>,

    /// Condition function, `c(t,y)`. See [`ConditionFunction`] for more details.
    pub(crate) c: Option<ConditionFunction<T>>,

    /// State reset function, `s(t,y)`. See [`StateResetFunction`] for more details.
    pub(crate) s: Option<StateResetFunction<T>>,

    /// Direction of a zero crossing to trigger the event. See [`Direction`] for more information.
    pub(crate) direction: Direction,

    /// Terminal state of the event TODO this is not a great name.
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
    ///
    /// # Example
    ///
    /// TODO
    pub fn new(g: impl Fn(f64, &T) -> f64 + 'static) -> Self {
        Event {
            g: Box::new(g),
            c: None,
            s: None,
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
    ///                 more information.
    ///
    /// # Returns
    ///
    /// The event with the the updated `direction`.
    pub fn direction(mut self, direction: Direction) -> Self {
        self.direction = direction;
        self
    }

    /// TODO.
    ///
    /// # Arguments
    ///
    /// * `terminal` - TODO.
    ///
    /// # Returns
    ///
    /// TODO.
    pub fn terminal(mut self, terminal: usize) -> Self {
        self.terminal = terminal;
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

    /// Store the time and the value of the ODE state at an occurence of this event.
    ///
    /// # Arguments
    ///
    /// * `t` - Time at the occurence of this event.
    /// * `y` - Value of the ODE state at the occurence of this event.
    pub(crate) fn store(&mut self, t: f64, y: &T) {
        self.t_located.push(t);
        self.y_located.push(y.clone());
    }
}

#[cfg(test)]
mod event_tests {
    use super::*;

    #[test]
    fn test_new() {
        let g = |t: f64, y: &f64| y * t;
        let event = Event::new(g);
        assert!(matches!(event.direction, Direction::Either));
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
    fn test_method() {
        let g = |t: f64, y: &f64| y * t;
        let event = Event::new(g).method(EventDetectionMethod::Exact);
        assert!(matches!(event.method, EventDetectionMethod::Exact)); // TODO test with a non-default method instead
    }

    #[test]
    fn test_name() {
        let g = |t: f64, y: &f64| y * t;
        let event = Event::new(g).name(String::from("My Event"));
        assert_eq!(event.name, "My Event");
    }
}
