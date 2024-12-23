use crate::state::State;

// TODO: this should implement all integrators
pub trait Propagate<T: State> {
    fn rk1_euler(&mut self, f: &impl Fn(f64, &Self) -> Self, t: f64, h: f64);
}

impl<T: State> Propagate<T> for T {
    fn rk1_euler(&mut self, f: &impl Fn(f64, &T) -> T, t: f64, h: f64) {
        let mut f_eval = f(t, self);
        f_eval.mul_assign(h);
        self.add_assign(&f_eval);
    }
}
