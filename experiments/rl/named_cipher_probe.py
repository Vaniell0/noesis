#!/usr/bin/env python3
"""named_cipher_probe.py - does a pretrained RWKV-7 decode a cipher it is only told the name (or the rule) of?

    python experiments/rl/named_cipher_probe.py --model MODEL [--weights W.pth] --out OUT.json

Knowledge as it is in the literature, applied: no table in the prompt. Ciphers: ROT13, Atbash, Caesar +3, the text
reversed, English typed on the Russian keyboard layout ("руддщ" for "hello") and Russian typed on the English layout
("ghbdtn" for "привет"). Two prompts per cipher: the name only, and the name with the rule in words. Greedy decoding
after M = 0, after `--ticks` own-latent ticks, and with the own stop (the first tick whose next-token max p >=
`--tau`, at most `--ticks`; no stop head); exact match and the share of letters right. Plaintexts: the first
words of the c1d eval texts (English), a fixed list of common Russian words. Inference only (plan
noesis-closed-loop-toy, note 11-20 evening: the Platonic / context-knowledge test).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from experiments.rl.loader import load_rwkv7, load_weights_into
from experiments.rl.script_alignment_probe import texts

EN = "qwertyuiop[]asdfghjkl;'zxcvbnm,."
RU = "йцукенгшщзхъфывапролджэячсмитьбю"
EN2RU, RU2EN = dict(zip(EN, RU)), dict(zip(RU, EN))
RU_WORDS = ("привет спасибо хорошо сегодня завтра работа погода деньги машина дорога человек вопрос ответ письмо "
            "книга школа город улица окно дверь вечер утро ночь собака кошка солнце море лес река гора друг семья "
            "время место слово жизнь рука голова сила правда работать говорить").split()


def rot(t: str, k: int) -> str:
    return "".join(chr((ord(c) - 97 + k) % 26 + 97) if "a" <= c <= "z" else c for c in t)


CIPHERS = {
    "rot13": (lambda t: rot(t, 13), "ROT13", "each letter is shifted 13 places in the alphabet"),
    "atbash": (lambda t: "".join(chr(219 - ord(c)) if "a" <= c <= "z" else c for c in t), "Atbash",
               "the alphabet is reversed: a is written as z, b as y, c as x, and so on"),
    "caesar3": (lambda t: rot(t, 3), "a Caesar cipher with shift 3",
                "each letter is replaced by the letter three places later: a is written as d, b as e"),
    "reverse": (lambda t: t[::-1], "the text written backwards", "the whole text is written from the last letter to the first"),
    "en_on_ru": (lambda t: "".join(EN2RU.get(c, c) for c in t), "English typed with the keyboard in the Russian layout",
                 "each key was pressed as on an English keyboard but the Russian layout was on, so q gives й, w gives ц"),
    "ru_on_en": (lambda t: "".join(RU2EN.get(c, c) for c in t), "Russian typed with the keyboard in the English layout",
                 "each key was pressed as on a Russian keyboard but the English layout was on, so й gives q, ц gives w"),
}


HEAD = {"en": "Decode the message.", "ru": "Расшифруй сообщение."}   # only the first line changes (owner 11-20)


def prompt(cipher: str, msg: str, with_rule: bool, head: str = "en") -> str:
    _, name, rule = CIPHERS[cipher]
    lang = "Russian" if cipher == "ru_on_en" else "English"
    how = f" Rule: {rule}." if with_rule else ""
    return f"{HEAD[head]}\nCipher: {name}.{how}\nMessage: {msg}\nWrite only the decoded {lang} text."


@torch.no_grad()
def decode(loaded, q: str, ticks: int, max_new: int, tau: float = 0.0) -> tuple[str, int]:
    tok, E = loaded.tokenizer, loaded.embedding_weight
    ids = tok.encode(loaded.silent_prompt(q))
    lg, st, h = loaded.forward_stateful_embeds(E[torch.tensor([ids])], loaded.new_state(batch=1), return_hidden=True)
    lg, h = lg[:, -1:], h[:, -1:]
    used = 0
    for _ in range(ticks):
        if tau and float(lg[0, -1].softmax(-1).max()) >= tau:
            break
        lg, st, h = loaded.forward_stateful_embeds(h, st, return_hidden=True)
        used += 1
    out = []
    for _ in range(max_new):
        t = int(lg[0, -1].argmax())
        if t == 0:
            break
        out.append(t)
        if "\n" in tok.decode(out[-1:]) and len(out) > 1:
            break
        lg, st = loaded.forward_stateful_embeds(E[torch.tensor([[t]])], st)
    return tok.decode(out).strip().split("\n")[0].strip().lower(), used


def letters_right(got: str, gold: str) -> float:
    return sum(a == b for a, b in zip(got, gold)) / max(1, len(gold))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--weights", type=Path)
    ap.add_argument("--eval", default="/home/vaniello/tmp/c1d/eval.jsonl")
    ap.add_argument("--n", type=int, default=40, help="messages per cipher")
    ap.add_argument("--words", type=int, default=3, help="English plaintext: the first N words of a c1d text")
    ap.add_argument("--ticks", type=int, default=8)
    ap.add_argument("--head", choices=list(HEAD), default="en", help="the instruction line: English or Russian")
    ap.add_argument("--tau", type=float, default=0.9, help="own stop: confidence that ends the ticks")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    en = [" ".join(t.lower().split()[:a.words]) for t in texts(a.eval)][:a.n]
    loaded = load_rwkv7(a.model, device="cpu", dtype=torch.float32, backend="peft", ctx_len=2048)
    if a.weights:
        load_weights_into(loaded, a.weights)
    res, rows, t0 = {}, [], time.time()
    for c, (enc, _, _) in CIPHERS.items():
        plain = RU_WORDS[:a.n] if c == "ru_on_en" else en
        for rule in (False, True):
            for M, tau in ((0, 0.0), (a.ticks, 0.0), ("stop", a.tau)):
                ex, lr, tk = 0, 0.0, 0
                for p in plain:
                    got, used = decode(loaded, prompt(c, enc(p), rule, a.head), a.ticks if M == "stop" else M, 2 * len(p) + 8, tau)
                    ex += got == p
                    lr += letters_right(got, p)
                    tk += used
                    rows.append({"cipher": c, "rule": rule, "M": M, "plain": p, "got": got, "ticks": used})
                key = f"{c}_{'rule' if rule else 'name'}_M{M}"
                res[key] = {"exact": round(ex / len(plain), 3), "letters": round(lr / len(plain), 3),
                            "ticks": round(tk / len(plain), 2)}
                print(f"[named] {key}: exact {res[key]['exact']:.3f} letters {res[key]['letters']:.3f} "
                      f"ticks {res[key]['ticks']:.1f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"model": a.model, "weights": str(a.weights) if a.weights else None, "n": a.n,
                                 "ticks": a.ticks, "head": a.head, "acc": res, "rows": rows}, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
