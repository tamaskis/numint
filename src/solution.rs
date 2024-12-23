use linalg_traits::Vector;

use crate::ode_state::ode_state_trait::{OdeState, StateIndex};

/// Solution of an ordinary differential equation `dy/dt = f(t,y)`.
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
}

impl<T: OdeState> Solution<T> {
    /// Constructs a new, empty `Solution<T>` with at least the specified capacity.
    ///
    /// # Arguments
    ///
    /// * `capacity` - (Minimum) capacity to allocate.
    ///
    /// # Returns
    ///
    /// A new, empty `Solution<T>` with at least the specified capacity.
    fn with_capacity(capacity: usize) -> Solution<T> {
        Solution {
            t: Vec::<f64>::with_capacity(capacity),
            y: Vec::<T>::with_capacity(capacity),
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
    pub(crate) fn new_for_ivp(y0: &T, t0: f64, tf: f64, h: f64) -> Solution<T> {
        // Time vector length (based on the initial time, final time, and time step).
        let length = (((tf - t0) / h).ceil() as usize) + 1;

        // Creates a Solution object, preallocating the time vector and state history vectors.
        let mut sol = Solution::with_capacity(length);

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
    fn len(&self) -> usize {
        self.t.len()
    }

    /// Shrinks the capacity of the `Solution<T>` as much as possible.
    pub(crate) fn shrink_to_fit(&mut self) {
        self.t.shrink_to_fit();
        self.y.shrink_to_fit();
    }

    /// Get the time history of the state variable at the specified index.
    ///
    /// # Type Parameters
    ///
    /// * `V` - The type of vector to use to store the time history of the requested state variable.
    ///         This type must implement the [`Vector`] trait.
    ///
    /// # Arguments
    ///
    /// * `index` - Index of the state variable (using 0-based indexing).
    ///
    /// # Returns
    ///
    /// Time history of the state variable.
    ///
    /// # Examples
    ///
    /// ## Vector-valued
    ///
    /// ```
    /// use numint::{solve_ivp, Euler, StateIndex};
    ///
    /// let f = |t: f64, y: &Vec<f64>| { vec![y[1], -2.5 * y[1] - 0.5 * y[0] + 0.5 * t.sin()] };
    /// let y0 = vec![1.0, 0.0];
    /// let t0 = 0.0;
    /// let tf = 1.0;
    /// let h = 0.1;
    /// let sol = solve_ivp::<Vec<f64>, Euler>(&f, t0, &y0, tf, h);
    ///
    /// // Get the time history of the y₁, where y = (y₀,y₁)ᵀ.
    /// let idx = StateIndex::Vector(1);
    /// let y1 = sol.get_state_variable::<Vec<f64>>(&idx);
    /// ```
    ///
    /// ## Matrix-valued
    ///
    /// ```
    /// # #[cfg(feature = "nalgebra")]
    /// # {
    /// use nalgebra::SMatrix;
    /// use numint::{solve_ivp, Euler, StateIndex};
    ///
    /// // Solve a simple initial value problem.
    /// let f = |t: f64, y: &SMatrix<f64, 2, 2>| {
    ///     SMatrix::<f64, 2, 2>::from_row_slice(&[
    ///         y[(0, 1)],
    ///         -2.5 * y[(0, 1)] - 0.5 * y[(0, 0)] + 0.5 * t.sin(),
    ///         y[(1, 0)],
    ///         0.5 * y[(1, 1)],
    ///     ])
    /// };
    /// let y0 = SMatrix::<f64, 2, 2>::from_row_slice(&[1.0, 0.0, 1.0, 1.0]);
    /// let t0 = 0.0;
    /// let tf = 1.0;
    /// let h = 0.1;
    /// let sol = solve_ivp::<SMatrix<f64, 2, 2>, Euler>(&f, t0, &y0, tf, h);
    ///
    /// // Get the time history of y₁₀, where y = ((y₀₀,y₀₁), (y₁₀,y₁₁)).
    /// let idx = StateIndex::Matrix(1, 0);
    /// let y10 = sol.get_state_variable::<Vec<f64>>(&idx);
    /// # }
    /// ```
    ///
    /// ## Scalar-valued
    ///
    /// ```
    /// use numint::{solve_ivp, Euler, StateIndex};
    ///
    /// // Solve a simple initial value problem.
    /// let f = |_t: f64, y: &f64| *y;
    /// let y0 = 1.0;
    /// let t0 = 0.0;
    /// let tf = 3.0;
    /// let h = 1.0;
    /// let sol = solve_ivp::<f64, Euler>(&f, t0, &y0, tf, h);
    ///
    /// // Get the time history of the only state variable.
    /// let idx = StateIndex::Scalar();
    /// let y = sol.get_state_variable::<Vec<f64>>(&idx);
    /// ```
    pub fn get_state_variable<V: Vector<f64>>(&self, index: &StateIndex) -> V {
        let mut x = V::new_with_length(self.len());
        for (i, y) in self.y.iter().enumerate() {
            x[i] = y.get_state_variable(*index);
        }
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use linalg_traits::{Mat, Matrix};

    #[cfg(feature = "nalgebra")]
    use nalgebra::{dvector, DMatrix, DVector, SMatrix, SVector};

    #[cfg(feature = "ndarray")]
    use ndarray::{array, Array1, Array2};

    #[test]
    fn test_with_capacity() {
        let sol = Solution::<f64>::with_capacity(100);
        assert!(sol.t.capacity() >= 100);
        assert!(sol.y.capacity() >= 100);
    }

    #[test]
    fn test_new_for_ivp_f64() {
        // Initial condition.
        let y0 = 2.0;

        // Initial and final times.
        let t0 = 1.0;
        let tf = 3.0;

        // Time step.
        let h = 0.5;

        // Construct the object to store the solution for an IVP where the dependent variable is
        // an f64.
        let sol = Solution::<f64>::new_for_ivp(&y0, t0, tf, h);

        // Verify that at least 5 elements have been preallocated (for the solution at times 1.0
        // (the initial condition), 1.5, 2.0, 2.5, and 3.0).
        assert!(sol.t.capacity() >= 5);
        assert!(sol.y.capacity() >= 5);

        // Verify that the initial time and initial condition were stored.
        assert_eq!(sol.t[0], 1.0);
        assert_eq!(sol.y[0], 2.0);

        // Verify that nothing has been stored besides the initial time and initial condition.
        assert_eq!(sol.len(), 1);
    }

    fn new_for_ivp_vec_helper<V: Vector<f64> + OdeState>() {
        // Initial condition.
        let y0 = [1.0, 2.0];

        // Initial and final times.
        let t0 = 1.0;
        let tf = 3.0;

        // Time step.
        let h = 0.5;

        // Construct the object to store the solution for an IVP where the dependent variable is
        // a vector of f64's.
        let sol = Solution::<V>::new_for_ivp(&V::from_slice(&y0), t0, tf, h);

        // Verify that at least 5 elements have been preallocated (for the solution at times 1.0
        // (the initial condition), 1.5, 2.0, 2.5, and 3.0).
        assert!(sol.t.capacity() >= 5);
        assert!(sol.y.capacity() >= 5);

        // Verify that the initial time and initial condition were stored.
        assert_eq!(sol.t[0], 1.0);
        assert_eq!(sol.y[0].as_slice(), &[1.0, 2.0]);

        // Verify that nothing has been stored besides the initial time and initial condition.
        assert_eq!(sol.len(), 1);
    }

    #[test]
    fn test_new_for_ivp_vec() {
        new_for_ivp_vec_helper::<Vec<f64>>();
    }

    #[test]
    #[cfg(feature = "nalgebra")]
    fn test_new_for_ivp_nalgebra_dvector() {
        new_for_ivp_vec_helper::<DVector<f64>>();
    }

    #[test]
    #[cfg(feature = "nalgebra")]
    fn test_new_for_ivp_nalgebra_svector() {
        new_for_ivp_vec_helper::<SVector<f64, 2>>();
    }

    #[test]
    #[cfg(feature = "ndarray")]
    fn test_new_for_ivp_ndarray_array1() {
        new_for_ivp_vec_helper::<Array1<f64>>();
    }

    fn new_for_ivp_mat_helper<M: Matrix<f64> + OdeState>() {
        // Initial condition.
        let y0 = [1.0, 2.0, 3.0, 4.0];

        // Initial and final times.
        let t0 = 1.0;
        let tf = 3.0;

        // Time step.
        let h = 0.5;

        // Construct the object to store the solution for an IVP where the dependent variable is
        // a vector of f64's.
        let sol = Solution::<M>::new_for_ivp(&M::from_row_slice(2, 2, &y0), t0, tf, h);

        // Verify that at least 5 elements have been preallocated (for the solution at times 1.0
        // (the initial condition), 1.5, 2.0, 2.5, and 3.0).
        assert!(sol.t.capacity() >= 5);
        assert!(sol.y.capacity() >= 5);

        // Verify that the initial time and initial condition were stored.
        assert_eq!(sol.t[0], 1.0);
        if M::is_row_major() {
            assert_eq!(sol.y[0].as_slice(), &[1.0, 2.0, 3.0, 4.0]);
        } else {
            assert_eq!(sol.y[0].as_slice(), &[1.0, 3.0, 2.0, 4.0]);
        }

        // Verify that nothing has been stored besides the initial time and initial condition.
        assert_eq!(sol.len(), 1);
    }

    #[test]
    fn test_new_for_ivp_mat() {
        new_for_ivp_mat_helper::<Mat<f64>>();
    }

    #[test]
    #[cfg(feature = "nalgebra")]
    fn test_new_for_ivp_nalgebra_dmatrix() {
        new_for_ivp_mat_helper::<DMatrix<f64>>();
    }

    #[test]
    #[cfg(feature = "nalgebra")]
    fn test_new_for_ivp_nalgebra_smatrix() {
        new_for_ivp_mat_helper::<SMatrix<f64, 2, 2>>();
    }

    #[test]
    #[cfg(feature = "ndarray")]
    fn test_new_for_ivp_ndarray_array2() {
        new_for_ivp_mat_helper::<Array2<f64>>();
    }

    #[test]
    fn test_new_for_ivp_non_integer_sample_times() {
        // Initial condition.
        let y0 = 2.0;

        // Initial and final times.
        let t0 = 1.0;
        let tf = 3.1;

        // Time step.
        let h = 0.5;

        // Construct the object to store the solution for an IVP where the dependent variable is
        // an f64.
        let sol = Solution::<f64>::new_for_ivp(&y0, t0, tf, h);

        // Verify that at least 6 elements have been preallocated (for the solution at times 1.0
        // (the initial condition), 1.5, 2.0, 2.5, 3.0, and 3.1).
        assert!(sol.t.capacity() >= 6);
        assert!(sol.y.capacity() >= 6);

        // Verify that the initial time and initial condition were stored.
        assert_eq!(sol.t[0], 1.0);
        assert_eq!(sol.y[0], 2.0);

        // Verify that nothing has been stored besides the initial time and initial condition.
        assert_eq!(sol.len(), 1);
    }

    #[test]
    fn test_len() {
        // Construct a solution with capacity for a length-2 solution.
        let mut sol = Solution::<f64>::with_capacity(2);

        // Verify that the solution is empty (even though we have allocated memory for it).
        assert_eq!(sol.len(), 0);

        // Store a dummy length-2 solution.
        sol.t.push(0.0);
        sol.y.push(1.0);
        sol.t.push(0.1);
        sol.y.push(2.0);

        // Verify that the length of the solution is now 2.
        assert_eq!(sol.len(), 2);
    }

    #[test]
    fn test_shrink_to_fit() {
        // Construct a solution with capacity for a length-3 solution.
        let mut sol = Solution::<f64>::with_capacity(3);

        // Store a dummy length-2 solution.
        sol.t.push(0.0);
        sol.y.push(1.0);
        sol.t.push(0.1);
        sol.y.push(2.0);

        // Verify that the capacity is still greater than or equal to 3.
        assert!(sol.t.capacity() >= 3);
        assert!(sol.y.capacity() >= 3);

        // Now, shrink the capacity to fit the solution that has been stored.
        sol.shrink_to_fit();

        // Verify that the capacity is greater than or equal to 2.
        assert!(sol.t.capacity() >= 2);
        assert!(sol.y.capacity() >= 2);
    }

    #[test]
    fn test_get_state_variable_f64() {
        // Construct a solution object and store a dummy solution.
        let mut sol = Solution::<f64>::with_capacity(2);
        sol.t.push(0.0);
        sol.t.push(0.1);
        sol.y.push(1.0);
        sol.y.push(2.0);

        // Extract the state variable.
        let y = sol.get_state_variable::<Vec<f64>>(&StateIndex::Scalar());

        // Verify that the state variable time history was correctly extracted.
        assert_eq!(y, vec![1.0, 2.0]);
    }

    fn get_state_variable_vec_helper<V: Vector<f64> + OdeState>() {
        // Construct a solution object and store a dummy solution.
        let mut sol = Solution::<V>::with_capacity(2);
        sol.t.push(0.0);
        sol.t.push(0.1);
        sol.y.push(V::from_slice(&[1.0, 3.0]));
        sol.y.push(V::from_slice(&[2.0, 6.0]));

        // Extract the state variables.
        let y0 = sol.get_state_variable::<Vec<f64>>(&StateIndex::Vector(0));
        let y1 = sol.get_state_variable::<Vec<f64>>(&StateIndex::Vector(1));

        // Verify that the state variable time histories were correctly extracted.
        assert_eq!(y0, vec![1.0, 2.0]);
        assert_eq!(y1, vec![3.0, 6.0]);
    }

    #[test]
    fn test_get_state_variable_vec() {
        get_state_variable_vec_helper::<Vec<f64>>();
    }

    #[test]
    #[cfg(feature = "nalgebra")]
    fn test_get_state_variable_nalgebra_dvector() {
        get_state_variable_vec_helper::<DVector<f64>>();
    }

    #[test]
    #[cfg(feature = "nalgebra")]
    fn test_get_state_variable_nalgebra_svector() {
        get_state_variable_vec_helper::<SVector<f64, 2>>();
    }

    #[test]
    #[cfg(feature = "ndarray")]
    fn test_get_state_variable_ndarray_array1() {
        get_state_variable_vec_helper::<Array1<f64>>();
    }

    fn get_state_variable_mat_helper<M: Matrix<f64> + OdeState>() {
        // Construct a solution object and store a dummy solution.
        let mut sol = Solution::<M>::with_capacity(2);
        sol.t.push(0.0);
        sol.t.push(0.1);
        sol.y.push(M::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]));
        sol.y.push(M::from_row_slice(2, 2, &[2.0, 4.0, 6.0, 8.0]));

        // Extract the state variables.
        let y00 = sol.get_state_variable::<Vec<f64>>(&StateIndex::Matrix(0, 0));
        let y01 = sol.get_state_variable::<Vec<f64>>(&StateIndex::Matrix(0, 1));
        let y10 = sol.get_state_variable::<Vec<f64>>(&StateIndex::Matrix(1, 0));
        let y11 = sol.get_state_variable::<Vec<f64>>(&StateIndex::Matrix(1, 1));

        // Verify that the state variable time histories were correctly extracted.
        assert_eq!(y00, vec![1.0, 2.0]);
        assert_eq!(y01, vec![2.0, 4.0]);
        assert_eq!(y10, vec![3.0, 6.0]);
        assert_eq!(y11, vec![4.0, 8.0]);
    }

    #[test]
    fn test_get_state_variable_mat() {
        get_state_variable_mat_helper::<Mat<f64>>();
    }

    #[test]
    #[cfg(feature = "nalgebra")]
    fn test_get_state_variable_nalgebra_dmatrix() {
        get_state_variable_mat_helper::<DMatrix<f64>>();
    }

    #[test]
    #[cfg(feature = "nalgebra")]
    fn test_get_state_variable_nalgebra_smatrix() {
        get_state_variable_mat_helper::<SMatrix<f64, 2, 2>>();
    }

    #[test]
    #[cfg(feature = "ndarray")]
    fn test_get_state_variable_ndarray_array2() {
        get_state_variable_mat_helper::<Array2<f64>>();
    }

    #[test]
    #[cfg(feature = "nalgebra")]
    #[cfg(feature = "ndarray")]
    fn test_get_state_variable_different_array_type() {
        // Construct a solution object and store a dummy solution.
        let mut sol = Solution::<Mat<f64>>::with_capacity(2);
        sol.t.push(0.0);
        sol.t.push(0.1);
        sol.y.push(Mat::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]));
        sol.y.push(Mat::from_row_slice(2, 2, &[2.0, 4.0, 6.0, 8.0]));

        // Extract the state variables.
        let y00: Vec<f64> = sol.get_state_variable::<Vec<f64>>(&StateIndex::Matrix(0, 0));
        let y01: DVector<f64> = sol.get_state_variable::<DVector<f64>>(&StateIndex::Matrix(0, 1));
        let y10: SVector<f64, 2> =
            sol.get_state_variable::<SVector<f64, 2>>(&StateIndex::Matrix(1, 0));
        let y11: Array1<f64> = sol.get_state_variable::<Array1<f64>>(&StateIndex::Matrix(1, 1));

        // Verify that the state variable time histories were correctly extracted.
        assert_eq!(y00, vec![1.0, 2.0]);
        assert_eq!(y01, dvector![2.0, 4.0]);
        assert_eq!(y10, SVector::<f64, 2>::from_slice(&[3.0, 6.0]));
        assert_eq!(y11, array![4.0, 8.0]);
    }
}
