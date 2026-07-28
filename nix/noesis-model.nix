# noesis-model — converts an RWKV-7 PyTorch checkpoint to rwkv.cpp format.
#
# Generic across model sizes and sources. Call it with a `model` record:
#
#   model = {
#     name    = "rwkv7-world-0.4b";         # used in pname + output filenames
#     version = "v2.9-20250107";             # used in derivation version
#     pth     = <nix-path-to-.pth>;          # the PyTorch checkpoint
#   };
#
# The `pth` field accepts anything that produces a nix store path:
#   - `pkgs.fetchurl { url = ...; hash = ...; }`  for public HF checkpoints
#   - `builtins.path { path = /abs/path; name = ...; }`  for local trained weights
#
# Two- or three-stage derivation:
#   1. FP16  — Python `convert_pytorch_to_ggml.py` (ships with rwkv.cpp source)
#   2. quant — `rwkv-quantize` CLI for Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0
#
# FP16 is the default. Q5_1 is recommended for CPU-only inference (good
# perplexity, ~2.3× size reduction from FP16 on 2.9B).
#
# Bumping a model:
#   - For fetchurl sources: update `hash` (get from `nix-prefetch-url <url>`).
#   - For local sources (builtins.path): bump the path in flake.nix.

{ lib
, stdenv
, python3
, rwkv-cpp
, model   # { name : str, version : str, pth : path }
, dtype ? "FP16"
}:

let
  quantizedDtypes = [ "Q4_0" "Q4_1" "Q5_0" "Q5_1" "Q8_0" ];
  isQuantized = builtins.elem dtype quantizedDtypes;

  dtypeLower  = lib.toLower dtype;
  intermediate = "${model.name}-fp16.bin";
  output       = "${model.name}-${dtypeLower}.bin";

  pyEnv = python3.withPackages (ps: with ps; [ torch numpy ]);
in
stdenv.mkDerivation {
  pname   = "noesis-model-${model.name}";
  version = model.version;

  dontUnpack = true;

  nativeBuildInputs = [ pyEnv ] ++ lib.optional isQuantized rwkv-cpp;

  buildPhase = ''
    runHook preBuild
  '' + (if isQuantized then ''
    echo "Stage 1/2: ${model.name} .pth → ${intermediate} (FP16)"
    python ${rwkv-cpp.src}/python/convert_pytorch_to_ggml.py \
      ${model.pth} ${intermediate} FP16
    echo "Stage 2/2: ${intermediate} → ${output} (${dtype})"
    ${rwkv-cpp}/bin/rwkv-quantize ${intermediate} ${output} ${dtype}
    rm ${intermediate}
  '' else ''
    echo "Converting ${model.name} .pth → ${output} (${dtype})"
    python ${rwkv-cpp.src}/python/convert_pytorch_to_ggml.py \
      ${model.pth} ${output} ${dtype}
  '') + ''
    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall
    install -Dm644 ${output} $out/${output}
    ln -s ${output} $out/model.bin
    runHook postInstall
  '';

  meta = {
    description = "RWKV model ${model.name} (${dtype}), in rwkv.cpp format";
    homepage    = "https://huggingface.co/BlinkDL";
    license     = lib.licenses.asl20;
    platforms   = [ "x86_64-linux" ];
  };
}
