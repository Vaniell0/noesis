#!/usr/bin/env bash
# Resume the two G1L downloads of rwkv_chain_2026_11_07.sh; the first attempt died on TLS EOF, which plain --retry skips.
cd ~/Desktop/projects/noesis
for n in rwkv7-g1l-1.5b-20261031-ctx34304.pth rwkv7-g1l-2.9b-20261031-ctx34304.pth; do
  f=models/$n
  [ -s $f ] && continue
  curl -sS -L -x socks5h://127.0.0.1:2080 --retry 500 --retry-delay 30 --retry-all-errors -C - -o $f.part \
    "https://huggingface.co/BlinkDL/rwkv7-g1/resolve/main/$n" && mv $f.part $f
  echo "[$(date +%F' '%T)] $n exit $? size $(stat -c %s $f 2>/dev/null)"
done
