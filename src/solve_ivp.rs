use crate::propagate::IntegrationMethod;
use crate::solution::Solution;
use crate::state::State;

pub fn solve_ivp<T: State, M: IntegrationMethod<T>>(
    f: &impl Fn(f64, &T) -> T,
    y0: &T,
    t0: f64,
    tf: f64,
    h: f64,
) -> Solution<T> {
    // Note this also stores initial condition TODO cleanup
    let mut sol = Solution::new(y0, t0, tf, h);

    let mut y = y0.clone();

    // TODO
    for i in 1..sol.t.len() {
        // Update current sample time.
        sol.t[i] = t0 + (i as f64) * h;

        // Propagate state to current sample time.
        M::propagate(f, sol.t[i - 1], h, &mut y);
    }

    sol

    // Initialize the vector to store the solution at each sample time.
}
