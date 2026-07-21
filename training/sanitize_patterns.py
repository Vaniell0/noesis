"""Regex families for corpus sanitization.

Each pattern is (name, compiled_regex). Names surface in redaction markers
`<REDACTED:{name}>` and in audit logs so we can trace which family fired.

Coverage (per cosmic-purring-cocke.md §Step 4):
 - Vendor API keys: Anthropic, OpenAI, xAI, Slack, GitHub, Stripe, HuggingFace
 - AWS access key + secret co-location
 - GCP service-account JSON (private_key_id / private_key blob)
 - `.env`-shaped KV assignments with high-entropy value
 - SSH private key blocks (RSA/ED25519/OPENSSH/PGP)
 - Private IPv4 addresses (10./172.16-31./192.168.) + localhost variants
"""

from __future__ import annotations

import re
from typing import Iterable

PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("anthropic_key", re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}")),
    ("openai_key", re.compile(r"sk-(?:proj-)?[A-Za-z0-9_\-]{20,}")),
    ("xai_key", re.compile(r"xai-[A-Za-z0-9]{20,}")),
    ("slack_bot_token", re.compile(r"xox[baprs]-[A-Za-z0-9\-]{10,}")),
    ("github_token", re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}")),
    ("stripe_key", re.compile(r"sk_(?:live|test)_[A-Za-z0-9]{20,}")),
    ("hf_token", re.compile(r"hf_[A-Za-z0-9]{30,}")),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    (
        "aws_secret",
        re.compile(
            r"(?i)aws(.{0,20})?(secret|key)(.{0,20})?[=:]\s*['\"]?([A-Za-z0-9/+=]{40})"
        ),
    ),
    ("gcp_private_key_id", re.compile(r'"private_key_id"\s*:\s*"[a-f0-9]{32,}"')),
    ("gcp_private_key", re.compile(r'"private_key"\s*:\s*"-----BEGIN [A-Z ]*PRIVATE KEY-----[\\A-Za-z0-9+/=\n\r]+-----END [A-Z ]*PRIVATE KEY-----[\\A-Za-z0-9+/=\n\r]*"')),
    (
        "env_kv_highentropy",
        re.compile(
            r"^\s*(?:export\s+)?([A-Z][A-Z0-9_]{3,})\s*=\s*['\"]?([A-Za-z0-9+/=_\-]{20,})['\"]?\s*$",
            re.MULTILINE,
        ),
    ),
    (
        "ssh_private_key",
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]{20,}?-----END [A-Z ]*PRIVATE KEY-----"
        ),
    ),
    (
        "pgp_private_key",
        re.compile(
            r"-----BEGIN PGP PRIVATE KEY BLOCK-----[\s\S]{20,}?-----END PGP PRIVATE KEY BLOCK-----"
        ),
    ),
    (
        "private_ipv4",
        re.compile(
            r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|"
            r"172\.(?:1[6-9]|2\d|3[0-1])\.\d{1,3}\.\d{1,3}|"
            r"192\.168\.\d{1,3}\.\d{1,3})\b"
        ),
    ),
]

# ENV_KV_ALLOWLIST — names that look like KV assignments but are safe to keep.
ENV_KV_ALLOWLIST: set[str] = {
    "PATH",
    "PYTHONPATH",
    "LD_LIBRARY_PATH",
    "HOME",
    "USER",
    "SHELL",
    "TERM",
    "LANG",
    "LC_ALL",
    "PWD",
    "OLDPWD",
    "DISPLAY",
    "TZ",
    "EDITOR",
    "VISUAL",
    "PAGER",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "NO_PROXY",
}


# Pattern classes:
#   * "critical" — real credentials. If they appear in a target (tool_use),
#     the rollout is dropped so the model doesn't learn to emit REDACTED.
#   * "info" — sensitive metadata (private IPs, internal hostnames). Worth
#     redacting in context but not worth throwing away a whole rollout
#     over — the model isn't going to leak infrastructure by memorising a
#     LAN IP that appears in a curl command.
PATTERN_CLASS: dict[str, str] = {
    "private_ipv4": "info",
}


def is_critical(name: str) -> bool:
    return PATTERN_CLASS.get(name, "critical") == "critical"


def find_matches(text: str) -> list[tuple[str, int, int, str]]:
    """Return list of (pattern_name, start, end, match_text) hits in text.

    `env_kv_highentropy` matches are filtered against ENV_KV_ALLOWLIST on the
    captured variable name to avoid false positives on PATH-style entries.
    """
    hits: list[tuple[str, int, int, str]] = []
    for name, rx in PATTERNS:
        for m in rx.finditer(text):
            if name == "env_kv_highentropy":
                var = m.group(1)
                if var in ENV_KV_ALLOWLIST:
                    continue
            hits.append((name, m.start(), m.end(), m.group(0)))
    return hits


def redact(text: str) -> tuple[str, list[dict]]:
    """Replace every hit with `<REDACTED:{name}>` marker.

    Returns (redacted_text, audit_records). Overlapping matches are resolved
    by processing hits sorted by start position, longest-first at ties.
    """
    hits = find_matches(text)
    if not hits:
        return text, []
    hits.sort(key=lambda h: (h[1], -(h[2] - h[1])))
    audit: list[dict] = []
    out: list[str] = []
    cursor = 0
    last_end = 0
    for name, start, end, match_text in hits:
        if start < last_end:
            continue
        out.append(text[cursor:start])
        out.append(f"<REDACTED:{name}>")
        audit.append({
            "pattern": name,
            "start": start,
            "end": end,
            "match_len": end - start,
            "sample": match_text[:24],
        })
        cursor = end
        last_end = end
    out.append(text[cursor:])
    return "".join(out), audit


def any_match(text: str) -> bool:
    for _name, rx in PATTERNS:
        if rx.search(text):
            return True
    return False


def match_names(text: str) -> list[str]:
    return sorted({name for name, *_ in find_matches(text)})


def iter_pattern_names() -> Iterable[str]:
    yield from (name for name, _ in PATTERNS)
