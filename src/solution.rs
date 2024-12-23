use crate::state::State;

pub struct InitialConditions<T: State> {
    pub(crate) t0: f64,
    pub(crate) y0: T,
}

pub struct Solution<T: State> {
    // Time vector (length-`N`).
    pub(crate) t: Vec<f64>,

    // Solution at each time (length-`N`).
    pub(crate) y: Vec<T>,
}

impl<T: State> Solution<T> {
    pub fn new(y0: &T, t0: f64, tf: f64, h: f64) -> Solution<T> {
        let length = ((tf - t0) / h).ceil() as usize;
        let mut sol = Solution {
            t: Vec::with_capacity(length),
            y: Vec::with_capacity(length),
        };

        // Store the initial conditions.
        sol.t[0] = t0;
        sol.y[0] = y0.clone();

        sol
    }
}
