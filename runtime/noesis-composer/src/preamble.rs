//! Preamble template rendering — `{date}`, `{time}`, `{hostname}`.

use std::time::{SystemTime, UNIX_EPOCH};

/// Render a preamble template. Unknown `{…}` placeholders are left intact.
pub fn render_preamble(template: &str) -> String {
    if template.is_empty() {
        return String::new();
    }
    let (date, time) = utc_date_time();
    let hostname = hostname();
    template
        .replace("{date}", &date)
        .replace("{time}", &time)
        .replace("{hostname}", &hostname)
}

fn utc_date_time() -> (String, String) {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let sod = (secs % 86_400) as u32;
    let h = sod / 3600;
    let m = (sod / 60) % 60;
    let s = sod % 60;
    let days = (secs / 86_400) as i64;
    let (y, mo, d) = civil_from_days(days);
    (format!("{y:04}-{mo:02}-{d:02}"), format!("{h:02}:{m:02}:{s:02}"))
}

fn civil_from_days(days: i64) -> (i32, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i32 + era as i32 * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mo = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if mo <= 2 { y + 1 } else { y };
    (y, mo, d)
}

fn hostname() -> String {
    std::fs::read_to_string("/etc/hostname")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn substitutes_known_vars() {
        let out = render_preamble("date={date} time={time} host={hostname}");
        assert!(!out.contains("{date}"));
        assert!(!out.contains("{time}"));
        assert!(!out.contains("{hostname}"));
        // date looks like YYYY-MM-DD
        assert!(out.contains('-'));
    }

    #[test]
    fn unknown_placeholders_preserved() {
        let out = render_preamble("x={unknown}");
        assert_eq!(out, "x={unknown}");
    }

    #[test]
    fn empty_template_returns_empty() {
        assert_eq!(render_preamble(""), "");
    }
}
