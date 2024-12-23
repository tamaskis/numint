use linalg_traits::Vector;

use crate::ode_state::ode_state_trait::{OdeState, StateIndex};

// TODO: something to indicate that the solution has been written to, e.g. that the vector actually
// wrote to something

/// Solution of an ordinary differential equation `dy/dt = f(t,y)`.
///
/// TODO: note on possible ODE types
pub struct Solution<T: OdeState> {
    /// Time vector (length-`N`).
    ///
    /// This vector stores each sample time.
    pub t: Vec<f64>,

    /// State history vector (length-`N`).
    ///
    /// This vector stores the ODE solution (i.e. the state vector) corresponding to each sample
    /// time in `t`.
    pub y: Vec<T>,

    /// TODO.
    pub solution_stored: bool,
}

impl<T: OdeState> Solution<T> {
    /// TODO: docs
    /// TODO: unit test
    fn new_with_capacity(capacity: usize) -> Solution<T> {
        Solution {
            t: Vec::<f64>::with_capacity(capacity),
            y: Vec::<T>::with_capacity(capacity),
            solution_stored: false,
        }
    }

    /// Construct a [`Solution`] to store the solution to an initial value problem.
    ///
    /// # Arguments
    ///
    /// * `y0` - Initial condition.
    /// * `t0` - Initial time.
    /// * `tf` - Final time.
    /// * `h` - Time step.
    ///
    /// # Returns
    ///
    /// A [`Solution`] object to store the result of an initial value problem.
    ///
    /// # Note
    ///
    /// This method does the following:
    ///
    /// * Allocates memory for the time vector based on the initial time, final time, and the
    ///   time step.
    /// * Assigns the initial time ot the first (i.e. index `0`) element of the time vector.
    /// * Assigns the initial condition to the first (i.e. index `0`) element of the state history
    ///   vector.
    pub fn new_for_ivp(y0: &T, t0: f64, tf: f64, h: f64) -> Solution<T> {
        // Time vector length (based on the initial time, final time, and time step).
        let length = (((tf - t0) / h).ceil() as usize) + 1;

        // Creates a Solution object, preallocating the time vector and state history vectors.
        let mut sol = Solution::new_with_capacity(length);

        // Store the initial time in the first (i.e. index 0) element of the time vector.
        sol.t.push(t0);

        // Store the initial condition in the first (i.e. index 0) element of the state history
        // vector.
        sol.y.push(y0.clone());

        sol
    }

    /// Length of the solution.
    ///
    /// # Returns
    ///
    /// Length of the solution (equal to the number of sample times).
    ///
    /// # Note
    ///
    /// This method returns `0` if an ODE solution hasn't been stored.
    ///
    /// TODO: unit test
    pub fn len(&self) -> usize {
        if self.solution_stored {
            self.t.len()
        } else {
            0
        }
    }

    /// Determine if this [`Solution`] is empty (i.e. no solution has yet been stored).
    ///
    /// # Returns
    ///
    /// `true` if this [`Solution`] is empty, `false` if a solution has been stored.
    ///
    /// TODO: unit test
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    // TODO: with capacity method for vector trait?

    pub fn get_state_variable<V: Vector<f64>>(&self, index: &StateIndex) -> V {
        let mut x = V::new_with_length(self.len());
        for (i, y) in self.y.iter().enumerate() {
            x[i] = y.get_state_variable(*index);
        }
        x
    }
}
