use linalg_traits::{Scalar, Vector};

pub fn rk1_euler<S, V>(f: &impl Fn(S, &V) -> V, t: S, h: S, y: &mut V)
where
    S: Scalar,
    V: Vector<S>,
{
    let mut f_eval = f(t, &y);
    f_eval.mul_assign(h);
    y.add_assign(&f_eval);
}
