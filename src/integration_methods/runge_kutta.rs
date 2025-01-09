pub use crate::integration_methods::integration_method_trait::IntegrationMethod;
pub use crate::ode_state::ode_state_trait::OdeState;

/// Euler (first-order) method.
pub struct Euler;

impl<T: OdeState> IntegrationMethod<T> for Euler {
    fn propagate(f: &impl Fn(f64, &T) -> T, t: f64, h: f64, y: &mut T) {
        // yₙ₊₁ = yₙ + hf(tₙ, yₙ)
        y.add_assign(&f(t, y).mul(h));
    }
}

/// Midpoint (second-order) method.
pub struct RK2;

impl<T: OdeState> IntegrationMethod<T> for RK2 {
    fn propagate(f: &impl Fn(f64, &T) -> T, t: f64, h: f64, y: &mut T) {
        // k₁ = f(tₙ, yₙ)
        let k1 = f(t, y);

        // k₂ = f(tₙ + (h/2), yₙ + (hk₁/2))
        let k2 = f(t + (h / 2.0), &y.add(&k1.mul(h / 2.0)));

        // yₙ₊₁ = yₙ + hk₂
        y.add_assign(&k2.mul(h));
    }
}

/// Heun's second-order method.
pub struct RK2Heun;

impl<T: OdeState> IntegrationMethod<T> for RK2Heun {
    fn propagate(f: &impl Fn(f64, &T) -> T, t: f64, h: f64, y: &mut T) {
        // k₁ = f(tₙ, yₙ)
        let k1 = f(t, y);

        // k₂ = f(tₙ + h, yₙ + (hk₁/2))
        let k2 = f(t + h, &y.add(&k1.mul(h)));

        // yₙ₊₁ = yₙ + (h/2)(k₁ + k₂)
        y.add_assign(&k1.add(&k2).mul(h / 2.0));
    }
}

/// Ralston's second-order method.
pub struct RK2Ralston;

impl<T: OdeState> IntegrationMethod<T> for RK2Ralston {
    fn propagate(f: &impl Fn(f64, &T) -> T, t: f64, h: f64, y: &mut T) {
        // k₁ = f(tₙ, yₙ)
        let k1 = f(t, y);

        // k₂ = f(tₙ + (2h/3), yₙ + (2hk₁/3))
        let k2 = f(t + (2.0 * h / 3.0), &y.add(&k1.mul(2.0 * h / 3.0)));

        // yₙ₊₁ = yₙ + (h/4)(k₁ + 3k₂)
        y.add_assign(&k1.add(&k2.mul(3.0)).mul(h / 4.0));
    }
}

/// Classic (Kutta's) third-order method.
pub struct RK3;

impl<T: OdeState> IntegrationMethod<T> for RK3 {
    fn propagate(f: &impl Fn(f64, &T) -> T, t: f64, h: f64, y: &mut T) {
        // k₁ = f(tₙ, yₙ)
        let k1 = f(t, y);

        // k₂ = f(tₙ + (h/2), yₙ + (hk₁/2))
        let k2 = f(t + (h / 2.0), &y.add(&k1.mul(h / 2.0)));

        // k₃ = f(tₙ + h, yₙ - hk₁ + 2hk₂)
        let k3 = f(t + (h / 2.0), &y.sub(&k1.mul(h)).add(&k2.mul(2.0 * h)));

        // yₙ₊₁ = yₙ = (h/6)(k₁ + 4k₂ + k₃)
        y.add_assign(&(k1.add(&k2.mul(4.0)).add(&k3)).mul(h / 6.0));
    }
}

/// Heun's third-order method.
pub struct RK3Heun;

impl<T: OdeState> IntegrationMethod<T> for RK3Heun {
    fn propagate(f: &impl Fn(f64, &T) -> T, t: f64, h: f64, y: &mut T) {
        // k₁ = f(tₙ, yₙ)
        let k1 = f(t, y);

        // k₂ = f(tₙ + (h/3), yₙ + (hk₁/3))
        let k2 = f(t + (h / 3.0), &y.add(&k1.mul(h / 3.0)));

        // k₃ = f(tₙ + (2h/3), yₙ + (2hk₂/3))
        let k3 = f(t + (2.0 * h / 3.0), &y.add(&k2.mul(2.0 * h / 3.0)));

        // yₙ₊₁ = yₙ + (h/4)(k₁ + 3k₃)
        y.add_assign(&k1.add(&k3.mul(3.0)).mul(h / 4.0));
    }
}

/// Ralston's third-order method.
pub struct RK3Ralston;

impl<T: OdeState> IntegrationMethod<T> for RK3Ralston {
    fn propagate(f: &impl Fn(f64, &T) -> T, t: f64, h: f64, y: &mut T) {
        // k₁ = f(tₙ, yₙ)
        let k1 = f(t, y);

        // k₂ = f(tₙ + (h/2), yₙ + (hk₁/2))
        let k2 = f(t + (h / 2.0), &y.add(&k1.mul(h / 2.0)));

        // k₃ = f(tₙ + (3h/4), yₙ + (3hk₂/4))
        let k3 = f(t + (3.0 * h / 4.0), &y.add(&k2.mul(3.0 * h / 4.0)));

        // yₙ₊₁ = yₙ + (h/9)(2k₁ + 3k₂ + 4k₃)
        y.add_assign(&(k1.mul(2.0).add(&k2.mul(3.0)).add(&k3.mul(4.0))).mul(h / 9.0));
    }
}

/// Strong stability preserving Runge-Kutta third-order method.
pub struct SSPRK3;

impl<T: OdeState> IntegrationMethod<T> for SSPRK3 {
    fn propagate(f: &impl Fn(f64, &T) -> T, t: f64, h: f64, y: &mut T) {
        // k₁ = f(tₙ, yₙ)
        let k1 = f(t, y);

        // k₂ = f(tₙ + h, yₙ + hk₁)
        let k2 = f(t + h, &y.add(&k1.mul(h)));

        // k₃ = f(tₙ + (h/2), yₙ + (hk₁/4) + (hk₂/4))
        let k3 = f(
            t + (h / 2.0),
            &y.add(&k1.mul(h / 4.0).add(&k2.mul(h / 4.0))),
        );

        // yₙ₊₁ = yₙ + (h/6)(k₁ + k₂ + 4k₃)
        y.add_assign(&(k1.add(&k2).add(&k3.mul(4.0))).mul(h / 6.0));
    }
}

/// Classic Runge-Kutta fourth-order method.
pub struct RK4;

impl<T: OdeState> IntegrationMethod<T> for RK4 {
    fn propagate(f: &impl Fn(f64, &T) -> T, t: f64, h: f64, y: &mut T) {
        // k₁ = f(tₙ, yₙ)
        let k1 = f(t, y);

        // k₂ = f(tₙ + (h/2), yₙ + (hk₁/2))
        let k2 = f(t + (h / 2.0), &y.add(&k1.mul(h / 2.0)));

        // k₃ = f(tₙ + (h/2), yₙ + (hk₂/2))
        let k3 = f(t + (h / 2.0), &y.add(&k2.mul(h / 2.0)));

        // k₄ = f(tₙ + h, yₙ + hk₃)
        let k4 = f(t + h, &y.add(&k3.mul(h)));

        // yₙ₊₁ = yₙ = (h/6)(k₁ + 2k₂ + 2k₃ + k₄)
        y.add_assign(&(k1.add(&k2.mul(2.0)).add(&k3.mul(2.0)).add(&k4)).mul(h / 6.0));
    }
}

/// Ralston's fourth-order method.
pub struct RK4Ralston;

impl<T: OdeState> IntegrationMethod<T> for RK4Ralston {
    fn propagate(f: &impl Fn(f64, &T) -> T, t: f64, h: f64, y: &mut T) {
        // k₁ = f(tₙ, yₙ)
        let k1 = f(t, y);

        // k₂ = f(tₙ + 0.4h, yₙ + 0.4hk₁)
        let k2 = f(t + 0.4 * h, &y.add(&k1.mul(0.4 * h)));

        // k₃ = f(tₙ + (0.45573725)h, yₙ + (0.29697761)hk₁ + (0.15875964)hk₂)
        let k3 = f(
            t + 0.45573725 * h,
            &y.add(&k1.mul(0.29697761 * h).add(&k2.mul(0.15875964 * h))),
        );

        // k₄ = f(tₙ + h, yₙ + (0.21810040)hk₁ - (3.05096516)hk₂ + (3.83286476)hk₃)
        let k4 = f(
            t + h,
            &y.add(&k1.mul(0.21810040 * h))
                .sub(&k2.mul(3.05096516 * h))
                .add(&k3.mul(3.83286476 * h)),
        );

        // yₙ₊₁ = yₙ + h((0.17476028)k₁ - (0.55148066)k₂ + (1.20553560)k₃ + (0.17118478)k₄)
        y.add_assign(
            &(k1.mul(0.17476028)
                .sub(&k2.mul(0.55148066))
                .add(&k3.mul(1.20553560))
                .add(&k4.mul(0.17118478)))
            .mul(h),
        );
    }
}

/// 3/8 rule fourth-order method.
pub struct RK438;

impl<T: OdeState> IntegrationMethod<T> for RK438 {
    fn propagate(f: &impl Fn(f64, &T) -> T, t: f64, h: f64, y: &mut T) {
        // k₁ = f(tₙ, yₙ)
        let k1 = f(t, y);

        // k₂ = f(tₙ + (h/3), yₙ + (hk₁/3))
        let k2 = f(t + (h / 3.0), &y.add(&k1.mul(h / 3.0)));

        // k₃ = f(tₙ + (2h/3), yₙ - (hk₁/3) + hk₂)
        let k3 = f(
            t + (2.0 * h / 3.0),
            &y.sub(&k1.mul(h / 3.0)).add(&k2.mul(h)),
        );

        // k₄ = f(tₙ + h, yₙ + hk₁ - hk₂ + hk₃)
        let k4 = f(t + h, &y.add(&k1.mul(h)).sub(&k2.mul(h)).add(&k3.mul(h)));

        // yₙ₊₁ = yₙ = (h/8)(k₁ + 3k₂ + 3k₃ + k₄)
        y.add_assign(&(k1.add(&k2.mul(3.0)).add(&k3.mul(3.0)).add(&k4)).mul(h / 8.0));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper function for testing Runge-Kutta integration methods.
    ///
    /// # Type Parameters
    ///
    /// * `T` - Integration method type.
    ///
    /// # Arguments
    ///
    /// * `y_exp` - Expected value of the ODE state after one propagation.
    ///
    /// # Panics
    ///
    /// * If the ODE state after one propagation does not equal `y_exp`.
    fn rkx_test_helper<T: IntegrationMethod<f64>>(y_exp: f64) {
        // Function defining the ODE.
        let f = |t: f64, x: &f64| -2.0 * x + t.powi(2);

        // Current state.
        let mut y = 1.0;

        // Current sample time.
        let t = 0.0;

        // Time step.
        let h = 0.1;

        // Propagate using the Euler method.
        T::propagate(&f, t, h, &mut y);

        // Check value after one propagation.
        assert_eq!(y, y_exp);
    }

    #[test]
    fn test_euler() {
        rkx_test_helper::<Euler>(0.8);
    }

    #[test]
    fn test_rk2() {
        rkx_test_helper::<RK2>(0.8202499999999999);
    }

    #[test]
    fn test_rk2_heun() {
        rkx_test_helper::<RK2Heun>(0.8205);
    }

    #[test]
    fn test_rk2_ralston() {
        rkx_test_helper::<RK2Ralston>(0.8203333333333334);
    }

    #[test]
    fn test_rk3() {
        rkx_test_helper::<RK3>(0.8188583333333334);
    }

    #[test]
    fn test_rk3_heun() {
        rkx_test_helper::<RK3Heun>(0.8189888888888889);
    }

    #[test]
    fn test_rk3_ralston() {
        rkx_test_helper::<RK3Ralston>(0.8189833333333334);
    }

    #[test]
    fn test_rk3_ssprk3() {
        rkx_test_helper::<SSPRK3>(0.8189666666666666);
    }

    #[test]
    fn test_rk4() {
        rkx_test_helper::<RK4>(0.8190508333333333);
    }

    #[test]
    fn test_rk4_ralston() {
        rkx_test_helper::<RK4Ralston>(0.8190506665171147);
    }

    #[test]
    fn test_rk4_38() {
        rkx_test_helper::<RK438>(0.8190505555555555);
    }
}
