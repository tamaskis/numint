use crate::events::event_manager::EventManager;
use crate::integrators::integrator_trait::Integrator;
use crate::ode_state::ode_state_trait::OdeState;
use crate::solution::Solution;

/// Solve an initial value problem.
///
/// # Type Parameters
///
/// * `T` - ODE state type (any type implementing the [`OdeState`] trait).
/// * `I` - Integrator type (any type implementing the [`Integrator`] trait).
///
/// # Arguments
///
/// * `f` - Function defining the ordinary differential equation, `dy/dt = f(t,y)`.
/// * `t0` - Initial time.
/// * `y0` - Initial condition.
/// * `tf` - Final time.
/// * `h` - Time step.
/// * `events` - Events.
///
/// # Returns
///
/// Solution of the initial value problem.
///
/// # Note
///
/// The initial value problem can be specified in one of the following three ways:
///
/// | Problem Type | Ordinary Differential Equation | Initial Condition |
/// | ------------ | ------------------------------ | ------------------ |
/// | scalar-valued | $$\frac{dy}{dt}=f(t,y)\quad\quad\left\(f:\mathbb{R}\times\mathbb{R}\to\mathbb{R}\right\)$$ | $$y(t_{0})=y_{0}\quad\quad\left(y_{0}\in\mathbb{R}\right)$$ |
/// | vector-valued | $$\frac{d\mathbf{y}}{dt}=\mathbf{f}(t,\mathbf{y})\quad\quad\left\(\mathbf{f}:\mathbb{R}\times\mathbb{R}^{p}\to\mathbb{R}^{p}\right\)$$ | $$\mathbf{y}(t_{0})=\mathbf{y}\_{0}\quad\quad\left(\mathbf{y}_{0}\in\mathbb{R}^{p}\right)$$ |
/// | matrix-valued | $$\frac{d\mathbf{Y}}{dt}=\mathbf{F}(t,\mathbf{Y})\quad\quad\left\(\mathbf{F}:\mathbb{R}\times\mathbb{R}^{p\times r}\to\mathbb{R}^{p\times r}\right\)$$ | $$\mathbf{Y}(t_{0})=\mathbf{Y}\_{0}\quad\quad\left(\mathbf{Y}_{0}\in\mathbb{R}^{p\times r}\right)$$ |
///
/// # Examples
///
/// ## Solving a scalar-valued initial value problem
///
/// Let's solve the IVP
///
/// $$
/// \frac{dy}{dt}=y,\quad y(0)=1
/// $$
///
/// over the time interval $t\in\[0,3\]$.
///
/// We need to write the ODE in the form
///
/// $$
/// \frac{dy}{dt}=f(t,y)
/// $$
///
/// In this case, we have
///
/// $$
/// f(t,y)=y
/// $$
///
/// Note that this ODE is actually independent of time ($t$). The initial time ($t_{0}$) and
/// corresponding initial condition ($y_{0}=y(t_{0})$) are
///
/// $$
/// \begin{aligned}
///     t_{0}&=0 \\\\
///     y_{0}&=0
/// \end{aligned}
/// $$
///
/// while the final time is just $t_{f}=3$. Choosing a time step of $h=1$, we are ready to solve
/// this IVP using `solve_ivp`.
///
/// ```
/// use numint::{solve_ivp, Euler};
///
/// // Function defining the ODE.
/// let f = |_t: f64, y: &f64| *y;
///
/// // Initial condition.
/// let y0 = 1.0;
///
/// // Initial and final time.
/// let t0 = 0.0;
/// let tf = 3.0;
///
/// // Time step.
/// let h = 1.0;
///
/// // Solve the initial value problem.
/// let sol = solve_ivp::<f64, Euler>(&f, t0, &y0, tf, h, None);
///
/// // Check the results.
/// assert_eq!(sol.t, [0.0, 1.0, 2.0, 3.0]);
/// assert_eq!(sol.y, [1.0, 2.0, 4.0, 8.0]);
/// ```
pub fn solve_ivp<T: OdeState + 'static, I: Integrator<T>>(
    f: &impl Fn(f64, &T) -> T,
    t0: f64,
    y0: &T,
    tf: f64,
    mut h: f64,
    mut event_manager: Option<&mut EventManager<T>>,
) -> Solution<T> {
    // Initialize the struct to store the solution. This:
    //  --> Preallocates memory for the time and solution vectors.
    //  --> Stores the initial conditions in the solution.
    let mut sol = Solution::new_for_ivp(y0, t0, tf, h);

    // Current sample time.
    let mut t;

    // Solution at the current sample time.
    let mut y = y0.clone();

    // Solve the initial value problem.
    for i in 1..sol.t.capacity() {
        // Update and store the current sample time.
        t = t0 + (i as f64) * h;
        sol.t.push(t);

        // Adjust the time step for the last step.
        if i == sol.t.capacity() - 1 {
            h = tf - sol.t[i - 1];
            sol.t[i] = tf;
        }

        // Propagate the state to the current sample time.
        I::propagate(f, sol.t[i - 1], h, &mut y);

        // Store the solution at the current sample time.
        sol.y.push(y.clone());

        // Perform event detection. TODO this should be done in the event manager.
        if let Some(event_manager) = event_manager.as_deref_mut() {
            // Get the step size to reach the first detected event (if one was detected) and the
            // corresponding index of the event in the vector of events.
            let (idx_event, h_event) =
                event_manager.detect_events::<I>(f, sol.t[i - 1], &sol.y[i - 1], &y, h);

            // If an event was detected, propagate to the event, store the event information, and
            // terminate integration if necessary.
            //  --> TODO: probably best to break some of this stuff out into helper functions to
            //            make unit testing way easier
            if let (Some(idx_event), Some(h_event)) = (idx_event, h_event) {
                // Event time.
                let t_event = sol.t[i - 1] + h_event;

                // Propagate the state to the event.
                //  --> If the event is exactly at the previous time or the current time, don't
                //      perform any propagation (since we already know the corresponding states at
                //      those times).
                let mut y_event;
                if h_event == 0.0 {
                    y_event = sol.y[i - 1].clone();
                } else if h_event == h {
                    y_event = sol.y[i].clone();
                } else {
                    y_event = sol.y[i - 1].clone();
                    I::propagate(f, sol.t[i - 1], h_event, &mut y_event);
                }

                // Store the solution at the event.
                sol.t[i] = t_event;
                sol.y[i] = y_event.clone();

                // Store the time and the value of the state when the event was detected.
                //  --> Note that if a state reset is done, this still stores the value at the event
                //      before the state reset.
                event_manager.store(t_event, &y_event, idx_event);

                // Break the integration loop if the number of detections has reached the number of
                // detections requiring termination.
                //  --> Note that no state reset is done in this case.
                // TODO: num_detections could be stored in EventManager
                if event_manager.num_detections[idx_event]
                    == event_manager[idx_event].termination.num_detections
                {
                    break;
                }

                // Reset the state.
                if let Some(s) = &event_manager[idx_event].s {
                    sol.y[i] = s(t_event, &y);
                }
            }
        }
    }

    // Free up any unused memory.
    sol.shrink_to_fit();

    sol
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Euler;
    use crate::StateIndex;
    use crate::events::event::Event;
    use numtest::*;

    #[cfg(feature = "nalgebra")]
    use nalgebra::{DVector, SMatrix, dvector};

    // TODO: test with the state being reset

    /// https://en.wikipedia.org/wiki/Euler_method#Using_step_size_equal_to_1_(h_=_1)s
    #[test]
    fn test_solve_ivp_scalar() {
        // Function defining the ODE.
        let f = |_t: f64, y: &f64| *y;

        // Initial condition.
        let y0 = 1.0;

        // Initial and final time.
        let t0 = 0.0;
        let tf = 3.0;

        // Time step.
        let h = 1.0;

        // Solve the initial value problem.
        let sol = solve_ivp::<f64, Euler>(&f, t0, &y0, tf, h, None);

        // Check the results.
        assert_eq!(sol.t, [0.0, 1.0, 2.0, 3.0]);
        assert_eq!(sol.y, [1.0, 2.0, 4.0, 8.0]);
    }

    #[test]
    fn test_solve_ivp_event_detection_on_state() {
        // Function defining the ODE.
        let f = |_t: f64, y: &f64| *y;

        // Initial condition.
        let y0 = 1.0;

        // Initial and final time.
        let t0 = 0.0;
        let tf = 3.0;

        // Time step.
        let h = 1.0;

        // Event.
        let event = Event::new(|_t: f64, y: &f64| y - 3.5);

        // Event manager.
        let mut event_manager = EventManager::new(vec![&event]);

        // Solve the initial value problem. // TODO: don't modify event stuff, rather store in the solution struct
        let sol = solve_ivp::<f64, Euler>(&f, t0, &y0, tf, h, Some(&mut event_manager));

        // Check the results.
        assert_eq!(sol.t, [0.0, 1.0, 1.7499999999999998]);
        assert_eq!(sol.y, [1.0, 2.0, 3.4999999999999996]);
    }

    #[test]
    #[cfg(feature = "nalgebra")]
    fn test_solve_ivp_vector() {
        // Spring-mass-damper parameters.
        let b = 5.0; // damping constant [N.s/m]
        let k = 1.0; // spring constant [N/m]
        let m = 2.0; // mass [kg]

        // Initial conditions.
        let x0 = 1.0; // initial position [m]
        let xdot0 = 0.0; // initial velocity [m/s]

        // Function defining the ODE.
        let f = |t: f64, y: &DVector<f64>| {
            DVector::<f64>::from_row_slice(&[
                y[1],
                -(b / m) * y[1] - (k / m) * y[0] + (1.0 / m) * t.sin(),
            ])
        };

        // Initial condition.
        let y0 = dvector![x0, xdot0];

        // Initial and final time.
        let t0 = 0.0;
        let tf = 1.0;

        // Time step.
        let h = 0.1;

        // Solve the initial value problem.
        let sol = solve_ivp::<DVector<f64>, Euler>(&f, t0, &y0, tf, h, None);

        // Check the results.
        assert_arrays_equal_to_decimal!(
            sol.t,
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            15
        );
        assert_arrays_equal!(
            sol.get_state_variable::<DVector<f64>>(&StateIndex::Vector(0)),
            [
                1.0,
                1.0,
                0.995,
                0.9867491670832341,
                0.976579389049635,
                0.9654959107223262,
                0.9542474967431397,
                0.9433808343981592,
                0.9332828125226833,
                0.9242134803802741,
                0.9163318476653514
            ]
        );
        assert_arrays_equal!(
            sol.get_state_variable::<DVector<f64>>(&StateIndex::Vector(1)),
            [
                0.0,
                -0.05,
                -0.0825083291676586,
                -0.10169778033599089,
                -0.11083478327308789,
                -0.11248413979186514,
                -0.10866662344980502,
                -0.10098021875475897,
                -0.09069332142409263,
                -0.0788163271492275,
                -0.06615657389956016
            ]
        );
    }

    #[test]
    #[cfg(feature = "nalgebra")]
    fn test_solve_ivp_matrix() {
        // Function defining the ODE.
        let f = |t: f64, y: &SMatrix<f64, 2, 2>| {
            SMatrix::<f64, 2, 2>::from_row_slice(&[
                y[(0, 1)],
                -2.5 * y[(0, 1)] - 0.5 * y[(0, 0)] + 0.5 * t.sin(),
                y[(1, 0)],
                0.5 * y[(1, 1)],
            ])
        };
        // Initial condition.
        let y0 = SMatrix::<f64, 2, 2>::from_row_slice(&[1.0, 0.0, 1.0, 1.0]);

        // Initial and final time.
        let t0 = 0.0;
        let tf = 1.0;

        // Time step.
        let h = 0.1;

        // Solve the initial value problem.
        let sol = solve_ivp::<SMatrix<f64, 2, 2>, Euler>(&f, t0, &y0, tf, h, None);

        // Check the results.
        assert_arrays_equal_to_decimal!(
            sol.t,
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            15
        );
        assert_eq!(
            sol.get_state_variable::<Vec<f64>>(&StateIndex::Matrix(0, 0)),
            [
                1.0,
                1.0,
                0.995,
                0.9867491670832341,
                0.976579389049635,
                0.9654959107223262,
                0.9542474967431397,
                0.9433808343981592,
                0.9332828125226833,
                0.9242134803802741,
                0.9163318476653514
            ]
        );
        assert_eq!(
            sol.get_state_variable::<Vec<f64>>(&StateIndex::Matrix(0, 1)),
            [
                0.0,
                -0.05,
                -0.0825083291676586,
                -0.10169778033599089,
                -0.11083478327308789,
                -0.11248413979186514,
                -0.10866662344980502,
                -0.10098021875475897,
                -0.09069332142409263,
                -0.0788163271492275,
                -0.06615657389956016
            ]
        );
        assert_eq!(
            sol.get_state_variable::<Vec<f64>>(&StateIndex::Matrix(1, 0)),
            [
                1.0,
                1.1,
                1.2100000000000002,
                1.3310000000000002,
                1.4641000000000002,
                1.61051,
                1.7715610000000002,
                1.9487171,
                2.1435888100000002,
                2.357947691,
                2.5937424601
            ]
        );
        assert_eq!(
            sol.get_state_variable::<Vec<f64>>(&StateIndex::Matrix(1, 1)),
            [
                1.0,
                1.05,
                1.1025,
                1.1576250000000001,
                1.2155062500000002,
                1.2762815625000004,
                1.3400956406250004,
                1.4071004226562505,
                1.477455443789063,
                1.5513282159785162,
                1.628894626777442
            ]
        );
    }

    #[test]
    fn test_solve_ivp_stress_time_termination() {
        // Function defining the ODE.
        let f = |_t: f64, _y: &f64| 1.0;

        // Initial condition.
        let y0 = 0.0;

        // Time step.
        let h = 1.0;

        // Initial and final time.
        let t0 = 0.0;
        let tf = 4.5;

        // Solve the initial value problem.
        let sol = solve_ivp::<f64, Euler>(&f, t0, &y0, tf, h, None);

        // Check the results.
        assert_eq!(sol.t, [0.0, 1.0, 2.0, 3.0, 4.0, 4.5]);
        assert_arrays_equal!(sol.y, [0.0, 1.0, 2.0, 3.0, 4.0, 4.5]);
    }
}
