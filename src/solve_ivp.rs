use crate::propagate::IntegrationMethod;
use crate::solution::Solution;
use crate::state::State;

/// TODO.
pub fn solve_ivp<T: State, M: IntegrationMethod<T>>(
    f: &impl Fn(f64, &T) -> T,
    y0: &T,
    t0: f64,
    tf: f64,
    mut h: f64,
) -> Solution<T> {
    // Note this also stores initial condition TODO cleanup
    let mut sol = Solution::new(y0, t0, tf, h);

    // Solution at the current sample time.
    let mut y = y0.clone();

    // Solve the initial value problem.
    for i in 1..sol.t.len() {
        // Update the current sample time.
        sol.t[i] = t0 + (i as f64) * h;

        // Adjust the time step for the last step.
        if i == sol.t.len() - 1 {
            h = tf - sol.t[i - 1];
        }

        // Propagate the state to the  current sample time.
        M::propagate(f, sol.t[i - 1], h, &mut y);

        // Store the solution at the current sample time.
        sol.y[i] = y.clone();
    }

    sol
}
