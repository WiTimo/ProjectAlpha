/// Division helper that guards against tiny denominators.
pub fn safe_div(numerator: f64, denominator: f64, epsilon: f64) -> f64 {
    numerator / (denominator.abs() + epsilon)
}

/// Signed variant of log1p used for skewed heavy-tailed features.
pub fn signed_log1p(value: f64) -> f64 {
    let sign = value.signum();
    let magnitude = value.abs();
    sign * (1.0 + magnitude).ln()
}
