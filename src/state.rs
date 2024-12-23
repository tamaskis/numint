use linalg_traits::{Scalar, Vector};

pub trait State {
    fn add(&self, other: &Self) -> Self;
    fn add_assign(&mut self, other: &Self);
    fn sub(&self, other: &Self) -> Self;
    fn sub_assign(&mut self, other: &Self);
    fn mul(&self, scalar: f64) -> Self;
    fn mul_assign(&mut self, scalar: f64);
}

impl<V: Vector<f64>> State for V {
    fn add(&self, other: &Self) -> Self {
        self.add(other)
    }
    fn add_assign(&mut self, other: &Self) {
        self.add_assign(other);
    }
    fn sub(&self, other: &Self) -> Self {
        self.sub(other)
    }
    fn sub_assign(&mut self, other: &Self) {
        self.sub_assign(other);
    }
    fn mul(&self, scalar: f64) -> Self {
        self.mul(scalar)
    }
    fn mul_assign(&mut self, scalar: f64) {
        self.mul_assign(scalar);
    }
}
