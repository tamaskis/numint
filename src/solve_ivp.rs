use crate::propagate::IntegrationMethod;
use crate::solution::Solution;
use crate::state::State;

pub fn solve_ivp<T: State, M: IntegrationMethod<T>>(
    f: &impl Fn(f64, &T) -> T,
    y: &mut T,
    t0: f64,
    tf: f64,
    h: f64,
) -> Solution<T> {
    // Initialize the vector to store the solution at each sample time.
    M::propagate(f, y, t, h);
}
