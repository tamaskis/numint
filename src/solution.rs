use crate::state::State;

pub struct Solution<T: State> {
    // Time vector (length-`N`).
    t: Vec<f64>,

    // Solution at each time (length-`N`).
    y: Vec<T>,
}

impl<T: State> Solution<T> {
    pub fn new(t0: f64, tf: f64, h: f64) -> Solution<T> {
        let length = ((tf - t0) / h).ceil() as usize;
        Solution {
            t: Vec::with_capacity(length),
            y: Vec::with_capacity(length),
        }
    }
}
