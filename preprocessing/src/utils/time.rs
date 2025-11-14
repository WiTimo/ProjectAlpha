use time::{Duration, OffsetDateTime};

use crate::domain::Resolution;

/// Align timestamp to the start of the bar bucket for a given resolution.
pub fn align_to_resolution(ts: OffsetDateTime, resolution: Resolution) -> OffsetDateTime {
    align_to_period(ts, resolution.bar_duration())
}

/// Floor the timestamp to multiples of `period` since UNIX epoch.
pub fn align_to_period(ts: OffsetDateTime, period: Duration) -> OffsetDateTime {
    let nanos = period.whole_nanoseconds();
    if nanos <= 0 {
        return ts;
    }

    let ts_nanos = ts.unix_timestamp_nanos();
    let floored = ts_nanos - (ts_nanos % nanos);
    OffsetDateTime::from_unix_timestamp_nanos(floored).unwrap_or(ts)
}
