"""
restructure_rfc.py — Slice binary-protocol QA tasks from IETF RFCs.

Downloads RFC text from rfc-editor.org, detects bit-field diagrams
(the +-+-+ ASCII art), and generates reasoning tasks in plain-CoT format:

  system: "You are a precise protocol analyst."
  user:   "<RFC excerpt>\n\nQuestion: <question>"
  think:  <step-by-step reasoning>
  answer: <plain-text answer, no DSL>

Output: JSONL with fields: id, system, user, think, answer, source_rfc, task_type

Usage:
  source training/.venv/bin/activate
  python training/scripts/restructure_rfc.py \
      --out training/corpus_open/rfc_qa.jsonl \
      --min-tasks 300
"""

import re, json, time, random, argparse, hashlib
from typing import Optional
import requests

# ── Target RFCs (binary wire formats, bit fields, encoding rules) ────────────

TARGET_RFCS = [
    (791,  "IP"),
    (793,  "TCP"),
    (768,  "UDP"),
    (1035, "DNS"),
    (2460, "IPv6"),
    (826,  "ARP"),
    (792,  "ICMP"),
    (2131, "DHCP"),
    (3550, "RTP"),
    (4303, "ESP"),
    (4302, "AH"),
    (8200, "IPv6 bis"),
    # Additional binary-format RFCs
    (4960, "SCTP"),
    (5905, "NTP"),
    (7540, "HTTP/2"),
    (2328, "OSPF"),
    (4271, "BGP"),
    (4443, "ICMPv6"),
    (8415, "DHCPv6"),
    (7252, "CoAP"),
    (4340, "DCCP"),
    (6347, "DTLS"),
    (6455, "WebSocket"),
    (4291, "IPv6-Addr"),
    (2784, "GRE"),
    (2205, "RSVP"),
    (4601, "PIM-SM"),
    (3588, "Diameter"),
    (1812, "IPv4-Router"),
    (7323, "TCP-Opt"),
    (4034, "DNSSEC"),
    (3376, "IGMPv3"),
]

HEADERS = {"User-Agent": "noesis-research/1.0 (academic; contact vaniello544@gmail.com)"}

# ── RFC fetch ────────────────────────────────────────────────────────────────

def fetch_rfc(num: int) -> Optional[str]:
    url = f"https://www.rfc-editor.org/rfc/rfc{num}.txt"
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code == 200:
            return r.text
    except Exception as e:
        print(f"  [warn] RFC {num}: {e}")
    return None

# ── Bit-field diagram parser ─────────────────────────────────────────────────

BIT_BORDER = re.compile(r'^\s*\+[-+]{10,}\+\s*$')   # +-+-+-+…+ (≥10 chars)
BIT_FIELDS = re.compile(r'^\s*\|[^|]+\|\s*$')
SECTION_HDR = re.compile(r'^\d+(\.\d+)*\.?\s+\S')

def parse_diagrams(text: str) -> list[dict]:
    """Return list of {section, fields: [{name, bits}], raw_lines}.
    Uses the first +-+-+ border line as the diagram anchor."""
    lines = text.splitlines()
    diagrams = []
    current_section = "Introduction"
    i = 0
    while i < len(lines):
        ln = lines[i]
        if SECTION_HDR.match(ln.strip()):
            current_section = ln.strip()
        if BIT_BORDER.match(ln):
            # Collect ruler (up to 2 lines before border) + border + fields
            start = max(0, i - 2)
            raw = list(lines[start:i]) + [ln]
            i += 1
            blank_count = 0
            while i < len(lines):
                l = lines[i]
                if BIT_BORDER.match(l) or BIT_FIELDS.match(l):
                    raw.append(l)
                    blank_count = 0
                    i += 1
                elif l.strip() == "":
                    blank_count += 1
                    if blank_count >= 2:
                        break
                    i += 1
                else:
                    break
            fields = _extract_fields(raw)
            if fields:
                diagrams.append({
                    "section": current_section,
                    "fields": fields,
                    "raw": "\n".join(raw),
                })
        else:
            i += 1
    return diagrams

def _extract_fields(raw_lines: list[str]) -> list[dict]:
    """Parse field names and bit widths from diagram lines.

    In standard RFC bit-field ASCII art each bit occupies 2 characters
    (content char + separator '+'), except the trailing border.  We
    preserve the raw column width (including padding) before stripping
    the name, so the bit-width estimate is based on column width not
    name length.
    """
    fields = []
    field_lines = [l for l in raw_lines if BIT_FIELDS.match(l)]
    for fl in field_lines:
        inner = fl.strip().lstrip("|").rstrip("|")
        raw_parts = inner.split("|")
        for rp in raw_parts:
            stripped = rp.strip()
            if not stripped:
                continue
            col_width = len(rp)  # raw width including spaces
            # Each bit ≈ 2 chars in standard RFC ASCII art (content + '-')
            bit_est = max(1, round(col_width / 2))
            # Snap to standard powers-of-2
            if bit_est <= 2:
                bits = 1
            elif bit_est <= 5:
                bits = 4
            elif bit_est <= 10:
                bits = 8
            elif bit_est <= 20:
                bits = 16
            else:
                bits = 32
            name = re.sub(r'\s+', ' ', stripped)
            fields.append({"name": name, "bits": bits})
    return fields

# ── Context extractor ────────────────────────────────────────────────────────

def get_section_text(text: str, section_header: str, max_chars: int = 1200) -> str:
    """Return up to max_chars of text starting from section_header."""
    idx = text.find(section_header[:40])
    if idx < 0:
        return section_header
    snippet = text[idx: idx + max_chars]
    # trim at last complete sentence or paragraph
    last_para = snippet.rfind("\n\n")
    if last_para > 200:
        snippet = snippet[:last_para]
    return snippet.strip()

# ── Task generators ──────────────────────────────────────────────────────────

rng = random.Random(42)

def make_field_extraction_task(rfc_num: int, proto: str, diagram: dict, rng=rng) -> Optional[dict]:
    """Given a header byte sequence, extract a specific field value."""
    fields = [f for f in diagram["fields"] if f["bits"] in (4, 8, 16, 32)]
    if not fields:
        return None
    field = rng.choice(fields)

    # generate a synthetic byte value for the field
    max_val = (1 << field["bits"]) - 1
    val = rng.randint(0, max_val)
    hex_val = f"0x{val:0{field['bits']//4}X}"
    bin_val = f"{val:0{field['bits']}b}"

    ctx = diagram["raw"][:600]
    question = (
        f"In the {proto} header format shown above, "
        f"the '{field['name']}' field is {field['bits']} bits wide. "
        f"If the raw value of this field is {hex_val}, "
        f"what is it in decimal and binary?"
    )
    think = (
        f"The '{field['name']}' field occupies {field['bits']} bits.\n"
        f"Raw hex value: {hex_val}\n"
        f"Convert to decimal: {val}\n"
        f"Convert to binary ({field['bits']} bits): {bin_val}"
    )
    answer = f"Decimal: {val}. Binary: {bin_val}."

    return _make_item(rfc_num, proto, "field_extraction", ctx, question, think, answer)

def make_flag_decode_task(rfc_num: int, proto: str, diagram: dict, rng=rng) -> Optional[dict]:
    """Given a flag byte value, identify which flags are set."""
    # look for fields named like "Flags" or single-bit fields
    flag_fields = [f for f in diagram["fields"] if f["bits"] == 1 or "flag" in f["name"].lower()]
    if len(flag_fields) < 2:
        return None

    # pick 3-8 flag fields, generate a random byte
    flags = flag_fields[:min(8, len(flag_fields))]
    n_bits = len(flags)
    val = rng.randint(0, (1 << n_bits) - 1)
    bits = [(val >> (n_bits - 1 - i)) & 1 for i in range(n_bits)]
    set_flags = [flags[i]["name"] for i, b in enumerate(bits) if b == 1]

    ctx = diagram["raw"][:500]
    question = (
        f"In the {proto} header, the flag byte (or control bits) has value "
        f"0x{val:02X} (binary: {val:0{n_bits}b}). "
        f"The bit fields in order are: {', '.join(f['name'] for f in flags)}. "
        f"Which flags are set (bit=1)?"
    )
    if set_flags:
        think = (
            f"Binary value: {val:0{n_bits}b}\n"
            + "\n".join(f"  bit[{i}] ({flags[i]['name']}): {bits[i]}" for i in range(n_bits))
            + f"\nFlags with bit=1: {', '.join(set_flags)}"
        )
        answer = f"Set flags: {', '.join(set_flags)}."
    else:
        think = f"Binary value: {val:0{n_bits}b}\nAll bits are 0 — no flags set."
        answer = "No flags are set (all bits are 0)."

    return _make_item(rfc_num, proto, "flag_decode", ctx, question, think, answer)

def make_encoding_task(rfc_num: int, proto: str, diagram: dict, rng=rng) -> Optional[dict]:
    """Given field values, encode them into the correct bit pattern."""
    fields = [f for f in diagram["fields"] if f["bits"] in (8, 16)]
    if len(fields) < 2:
        return None
    chosen = rng.sample(fields, min(3, len(fields)))

    vals = {f["name"]: rng.randint(0, (1 << f["bits"]) - 1) for f in chosen}
    total_bits = sum(f["bits"] for f in chosen)
    packed = 0
    for f in chosen:
        packed = (packed << f["bits"]) | vals[f["name"]]
    hex_out = f"0x{packed:0{total_bits//4}X}"

    ctx = diagram["raw"][:500]
    field_str = ", ".join(f"'{f['name']}'={vals[f['name']]} ({f['bits']} bits)" for f in chosen)
    question = (
        f"In the {proto} format, encode the following field values into a contiguous bit string "
        f"in the order they appear in the header:\n{field_str}\n"
        f"Give the result as hex."
    )
    think_lines = ["Pack fields in order:"]
    shift = total_bits
    for f in chosen:
        shift -= f["bits"]
        think_lines.append(
            f"  '{f['name']}' = {vals[f['name']]} → binary {vals[f['name']]:0{f['bits']}b} at bit offset {total_bits - shift - f['bits']}"
        )
    think_lines.append(f"Concatenated: {packed:0{total_bits}b}")
    think_lines.append(f"As hex: {hex_out}")
    think = "\n".join(think_lines)
    answer = f"Encoded value: {hex_out}."

    return _make_item(rfc_num, proto, "encoding", ctx, question, think, answer)

def make_offset_task(rfc_num: int, proto: str, diagram: dict, rng=rng) -> Optional[dict]:
    """Calculate byte offset of a field within the header."""
    fields = diagram["fields"]
    if len(fields) < 3:
        return None
    target_idx = rng.randint(1, len(fields) - 1)
    target = fields[target_idx]
    offset_bits = sum(f["bits"] for f in fields[:target_idx])
    offset_bytes = offset_bits // 8
    offset_bits_rem = offset_bits % 8

    ctx = diagram["raw"][:500]
    question = (
        f"In the {proto} header format above, at what bit offset (counting from 0) "
        f"does the '{target['name']}' field start?"
    )
    think = (
        f"Sum the bits of all preceding fields:\n"
        + "\n".join(f"  '{fields[i]['name']}': {fields[i]['bits']} bits" for i in range(target_idx))
        + f"\nTotal: {offset_bits} bits"
        + (f" = {offset_bytes} bytes + {offset_bits_rem} bits" if offset_bits_rem else f" = {offset_bytes} bytes")
    )
    answer = f"The '{target['name']}' field starts at bit offset {offset_bits} ({offset_bytes} bytes{'+ ' + str(offset_bits_rem) + ' bits' if offset_bits_rem else ''})."

    return _make_item(rfc_num, proto, "bit_offset", ctx, question, think, answer)

def _make_item(rfc_num, proto, task_type, ctx, question, think, answer) -> dict:
    uid = hashlib.md5(f"{rfc_num}:{task_type}:{question[:80]}".encode()).hexdigest()[:12]
    return {
        "id": f"rfc{rfc_num}_{task_type}_{uid}",
        "source_rfc": rfc_num,
        "protocol": proto,
        "task_type": task_type,
        "system": "You are a precise network protocol analyst. Work step by step.",
        "user": f"RFC {rfc_num} ({proto}) excerpt:\n\n{ctx}\n\nQuestion: {question}",
        "think": think,
        "answer": answer,
    }

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="training/corpus_open/rfc_qa.jsonl")
    ap.add_argument("--min-tasks", type=int, default=300)
    ap.add_argument("--tasks-per-diagram", type=int, default=12)
    ap.add_argument("--rfcs", nargs="*", type=int, help="Override RFC list")
    args = ap.parse_args()

    rfc_list = [(n, p) for n, p in TARGET_RFCS] if not args.rfcs else [(n, "RFC") for n in args.rfcs]
    generators = [make_field_extraction_task, make_flag_decode_task,
                  make_encoding_task, make_offset_task]

    all_tasks = []
    for rfc_num, proto in rfc_list:
        print(f"  RFC {rfc_num} ({proto}) ...", end=" ", flush=True)
        text = fetch_rfc(rfc_num)
        if not text:
            print("SKIP")
            continue
        diagrams = parse_diagrams(text)
        print(f"{len(diagrams)} diagrams", end=" ")
        n_before = len(all_tasks)
        for diag in diagrams:
            for gen in generators:
                for _ in range(args.tasks_per_diagram // len(generators) + 1):
                    item = gen(rfc_num, proto, diag)
                    if item:
                        all_tasks.append(item)
        print(f"→ {len(all_tasks) - n_before} tasks")
        time.sleep(0.5)  # be polite to rfc-editor.org

    rng.shuffle(all_tasks)
    # deduplicate by id
    seen = set()
    unique = []
    for t in all_tasks:
        if t["id"] not in seen:
            seen.add(t["id"])
            unique.append(t)

    print(f"\nTotal: {len(unique)} tasks")
    with open(args.out, "w") as f:
        for t in unique:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"Written → {args.out}")

    if len(unique) < args.min_tasks:
        print(f"[warn] Only {len(unique)} tasks, target was {args.min_tasks}. Add more RFCs.")

if __name__ == "__main__":
    main()
