use linalg_traits::{Mat, Matrix, Vector};

pub trait State {
    fn add(&self, other: &Self) -> Self;
    fn add_assign(&mut self, other: &Self);
    fn sub(&self, other: &Self) -> Self;
    fn sub_assign(&mut self, other: &Self);
    fn mul(&self, scalar: f64) -> Self;
    fn mul_assign(&mut self, scalar: f64);
}

// TODO: we do it this way to avoid conflicting trait implementations (technically, a type that
// implements Vector could also implement Matrix)

// TODO: make a macro to automate these implementations

impl State for f64 {
    fn add(&self, other: &Self) -> Self {
        self + other
    }
    fn add_assign(&mut self, other: &Self) {
        *self += other;
    }
    fn sub(&self, other: &Self) -> Self {
        self - other
    }
    fn sub_assign(&mut self, other: &Self) {
        *self -= other
    }
    fn mul(&self, scalar: f64) -> Self {
        self * scalar
    }
    fn mul_assign(&mut self, scalar: f64) {
        *self *= scalar
    }
}

impl State for Vec<f64> {
    fn add(&self, other: &Self) -> Self {
        <Self as Vector<f64>>::add(self, other)
    }
    fn add_assign(&mut self, other: &Self) {
        <Self as Vector<f64>>::add_assign(self, other);
    }
    fn sub(&self, other: &Self) -> Self {
        <Self as Vector<f64>>::sub(self, other)
    }
    fn sub_assign(&mut self, other: &Self) {
        <Self as Vector<f64>>::sub_assign(self, other);
    }
    fn mul(&self, scalar: f64) -> Self {
        <Self as Vector<f64>>::mul(self, scalar)
    }
    fn mul_assign(&mut self, scalar: f64) {
        <Self as Vector<f64>>::mul_assign(self, scalar);
    }
}

impl State for Mat<f64> {
    fn add(&self, other: &Self) -> Self {
        <Self as Matrix<f64>>::add(self, other)
    }
    fn add_assign(&mut self, other: &Self) {
        <Self as Matrix<f64>>::add_assign(self, other);
    }
    fn sub(&self, other: &Self) -> Self {
        <Self as Matrix<f64>>::sub(self, other)
    }
    fn sub_assign(&mut self, other: &Self) {
        <Self as Matrix<f64>>::sub_assign(self, other);
    }
    fn mul(&self, scalar: f64) -> Self {
        <Self as Matrix<f64>>::mul(self, scalar)
    }
    fn mul_assign(&mut self, scalar: f64) {
        <Self as Matrix<f64>>::mul_assign(self, scalar);
    }
}
