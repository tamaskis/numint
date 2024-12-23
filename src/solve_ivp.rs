use crate::integration_methods::integration_method_trait::IntegrationMethod;
use crate::ode_state::ode_state_trait::OdeState;
use crate::solution::Solution;

/// Solve an initial value problem.
///
/// # Type Parameters
///
/// * `T` - The type of the state.
/// * `M` - The type of the integration method.
///
/// # Arguments
///
/// * `f` - Function defining the ordinary differential equation, `dy/dt = f(t,y)`.
/// * `t0` - Initial time.
/// * `y0` - Initial condition.
/// * `tf` - Final time.
/// * `h` - Step size.
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
pub fn solve_ivp<T: OdeState, M: IntegrationMethod<T>>(
    f: &impl Fn(f64, &T) -> T,
    t0: f64,
    y0: &T,
    tf: f64,
    mut h: f64,
) -> Solution<T> {
    // Initialize the struct to store the solution. This:
    //  --> Preallocates memory for the time and solution vectors.
    //  --> Stores the initial conditions in the solution.
    let mut sol = Solution::new_for_ivp(y0, t0, tf, h);

    // Solution at the current sample time.
    let mut y = y0.clone();

    // Solve the initial value problem.
    for i in 1..sol.t.capacity() {
        // Update the current sample time.
        sol.t.push(t0 + (i as f64) * h);

        // Adjust the time step for the last step.
        if i == sol.t.capacity() - 1 {
            h = tf - sol.t[i - 1];
            sol.t[i] = tf;
        }

        // Propagate the state to the current sample time.
        M::propagate(f, sol.t[i - 1], h, &mut y);

        // Store the solution at the current sample time.
        sol.y.push(y.clone());
    }

    // Flag that the solution is fully stored.
    sol.solution_stored = true;

    sol
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Euler;
    use crate::StateIndex;
    use numtest::*;

    #[cfg(feature = "nalgebra")]
    use nalgebra::{dvector, DVector};

    /// # Reference
    ///
    /// https://en.wikipedia.org/wiki/Euler_method#Using_step_size_equal_to_1_(h_=_1)s
    #[test]
    fn test_solve_ivp_scalar() {
        // Function defining the ODE.
        let f = |_t: f64, y: &f64| *y;

        // Initial condition.
        let y0 = 1.0;

        // Time step.
        let h = 1.0;

        // Initial and final time.
        let t0 = 0.0;
        let tf = 3.0;

        // Solve the initial value problem.
        let sol = solve_ivp::<f64, Euler>(&f, t0, &y0, tf, h);

        // Check the results.
        assert_arrays_equal!(sol.t, [0.0, 1.0, 2.0, 3.0]);
        assert_arrays_equal!(sol.y, [1.0, 2.0, 4.0, 8.0]);
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

        // Time step.
        let h = 0.1;

        // Initial and final time.
        let t0 = 0.0;
        let tf = 1.0;

        // Solve the initial value problem.
        let sol = solve_ivp::<DVector<f64>, Euler>(&f, t0, &y0, tf, h);

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
        let sol = solve_ivp::<f64, Euler>(&f, t0, &y0, tf, h);

        // Check the results.
        assert_arrays_equal!(sol.t, [0.0, 1.0, 2.0, 3.0, 4.0, 4.5]);
        assert_arrays_equal!(sol.y, [0.0, 1.0, 2.0, 3.0, 4.0, 4.5]);
    }
}
