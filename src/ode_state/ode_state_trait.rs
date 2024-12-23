/// Index for indexing into ODE state.
#[derive(Copy, Clone)]
pub enum StateIndex {
    /// Represents a scalar index.
    ///
    /// This variant indicates that the ODE state is a single scalar value.
    Scalar(),

    /// Represents a vector index.
    ///
    /// This variant is used when indexing into a vector. The associated value specifies the index
    /// of the desired element.
    Vector(usize),

    /// Represents a matrix index.
    ///
    /// This variant is used when indexing into a matrix. The associated values specify the row and
    /// column indices of the desired element.
    Matrix(usize, usize),
}

/// Trait defining an ODE solver state (i.e. the dependent variable in an ODE).
///
/// Any type implementing this trait can be used as the dependent variable in an ODE.
///
/// This crate implements this trait for the following types:
///
/// * `f64`
/// * `Vec<f64>`
/// * `linalg_traits::Mat<f64>`
/// * `nalgebra::DVector<f64>`
/// * `nalgebra::SVector<f64, N>`
/// * `nalgebra::DMatrix<f64>`
/// * `nalgebra::SMatrix<f64, R, C>`
/// * `ndarray::Array1<f64>`
/// * `ndarray::Array2<f64>`
///
/// # Note on the lack of blanket implementations
///
/// We cannot perform blanket implementations using
/// [`linalg_traits::Scalar`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.Scalar.html),
/// [`linalg_traits::Vector`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.Vector.html),
/// and
/// [`linalg_traits::Matrix`](https://docs.rs/linalg-traits/latest/linalg_traits/trait.Matrix.html).
/// There is nothing restricting types from implementing a combination of these three traits, so the
/// compiler will not allow blanket implementations binding to these traits to avoid any potential
/// conflicting implementations.
pub trait OdeState: Clone {
    /// Addition (`self + other`).
    ///
    /// # Arguments
    ///
    /// * `other` - The other state to add to this state.
    ///
    /// # Returns
    ///
    /// Sum of this state with the other state (i.e. `self + other`).
    ///
    /// # Panics
    ///
    /// * If `self` and `other` are dynamically-sized types and do not have the same length (for
    ///   vectors) or shape (for matrices).
    fn add(&self, other: &Self) -> Self;

    /// In-place addition (`self += other`).
    ///
    /// # Arguments
    ///
    /// * `other` - The other state to add to this state.
    ///
    /// # Panics
    ///
    /// * If `self` and `other` are dynamically-sized types and do not have the same length (for
    ///   vectors) or shape (for matrices).
    fn add_assign(&mut self, other: &Self);

    /// Subtraction (`self + other`).
    ///
    /// # Arguments
    ///
    /// * `other` - The other state to subtract from this state.
    ///
    /// # Returns
    ///
    /// Difference of this state with the other state (i.e. `self - other`).
    ///
    /// # Panics
    ///
    /// * If `self` and `other` are dynamically-sized types and do not have the same length (for
    ///   vectors) or shape (for matrices).
    fn sub(&self, other: &Self) -> Self;

    /// In-place subtraction (`self -= other`).
    ///
    /// # Arguments
    ///
    /// * `other` - The other state to subtract from this state.
    ///
    /// # Panics
    ///
    /// * If `self` and `other` are dynamically-sized types and do not have the same length (for
    ///   vectors) or shape (for matrices).
    fn sub_assign(&mut self, other: &Self);

    /// Multiplication (`self * other`).
    ///
    /// # Arguments
    ///
    /// * `scalar` - The scalar to multiply this state by.
    ///
    /// # Returns
    ///
    /// Product of this state with the scalar (i.e. `self * scalar`).
    fn mul(&self, scalar: f64) -> Self;

    /// In-place multiplication (`self * other`).
    ///
    /// # Arguments
    ///
    /// * `scalar` - The scalar to multiply this state by.
    fn mul_assign(&mut self, scalar: f64);

    /// Get the value of the state variable at the specified index.
    ///
    /// # Arguments
    ///
    /// * `index` - Index of the state variable to retrieve.
    ///
    /// # Returns
    ///
    /// Value of the state variabe at the specified index.
    fn get_state_variable(&self, index: StateIndex) -> f64;
}
