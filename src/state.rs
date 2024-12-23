use linalg_traits::{Scalar, Vector};

pub trait State<S: Scalar> {
    fn add(&self, other: &Self) -> Self;
    fn sub(&self, other: &Self) -> Self;
    fn mul(&self, scalar: S) -> Self;
}

impl<S: Scalar, T: Scalar> State for S {
    fn add(&self, other: &Self) -> Self {
        *self + *other
    }
    fn sub(&self, other: &Self) -> Self {
        *self - *other
    }
    fn mul(&self, scalar: &T) -> Self {
        *self * *other
    }
}
