# noesis-model — RWKV-7 World 0.4B, converted to rwkv.cpp format.
#
# Two- or three-stage derivation:
#
#   1. `weights`  = fetchurl of the upstream PyTorch checkpoint from
#                   HuggingFace (BlinkDL/rwkv-7-world). Immutable pin
#                   by sha256, so bit-identical across all builds.
#   2. FP16 build = torch-only Python env runs the upstream
#                   `convert_pytorch_to_ggml.py` (shipped inside the
#                   pinned rwkv.cpp source) → intermediate .bin.
#   3. quantize   = for Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 dtypes, run the
#                   `rwkv-quantize` CLI shipped by `rwkv-cpp` on the
#                   FP16 intermediate. Skipped when dtype is FP16/FP32.
#
# Default is FP16 (matches upstream convention; the 0.4B checkpoint is
# ~861 MB after FP16 conversion; ~591 MB after Q8_0 — 1.46× compression
# ratio, most of the tail is the FP16 head + FP32 layernorms). Pass
# `dtype = "Q8_0"` for the recommended CPU quantization — negligible
# perplexity cost vs. FP16 in upstream measurements.
#
# Bumping the model:
#   - Update `checkpoint.filename` + `checkpoint.sha256`.
#   - `nix-prefetch-url` gives the sha256 in nix32 form.
#   - Model paths on HF follow the BlinkDL/rwkv-7-world convention:
#     RWKV-x070-World-<size>-<version>-<date>-ctx<len>.pth

{ lib
, stdenv
, fetchurl
, python3
, rwkv-cpp
, dtype ? "FP16"
}:

let
  # Pinned checkpoint. See the module-level docstring for how to update.
  checkpoint = {
    filename = "RWKV-x070-World-0.4B-v2.9-20250107-ctx4096.pth";
    sha256 = "0jnsh2hmqz7ll9x4fvhncwf0ri84gypsy34ggdqsan9ync1gd360";
  };

  weights = fetchurl {
    url = "https://huggingface.co/BlinkDL/rwkv-7-world/resolve/main/${checkpoint.filename}";
    sha256 = checkpoint.sha256;
  };

  # torch on CPU is enough — the conversion script uses map_location='cpu'
  # and does no forward pass. numpy is a transitive dep torch ships.
  pyEnv = python3.withPackages (ps: with ps; [ torch numpy ]);

  # The upstream `convert_pytorch_to_ggml.py` only emits FP16 / FP32. Anything
  # smaller (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0) must be produced by a second pass
  # through `rwkv-quantize` on the FP16 intermediate.
  quantizedDtypes = [ "Q4_0" "Q4_1" "Q5_0" "Q5_1" "Q8_0" ];
  isQuantized = builtins.elem dtype quantizedDtypes;

  intermediateBasename = "rwkv7-world-0.4b-fp16.bin";
  outputBasename = "rwkv7-world-0.4b-${lib.toLower dtype}.bin";
in
stdenv.mkDerivation {
  pname = "noesis-model-rwkv7-world-0.4b";
  version = "v2.9-20250107";

  # No src of our own — we consume `weights` + `rwkv-cpp.src`.
  dontUnpack = true;

  # rwkv-cpp is only needed when we quantize; for FP16/FP32 the python
  # converter alone produces the final artifact and we save a build closure.
  nativeBuildInputs = [ pyEnv ] ++ lib.optional isQuantized rwkv-cpp;

  # The converter always writes to CWD; run from $TMPDIR so we don't leak
  # into the source directory (there isn't one — dontUnpack is set).
  buildPhase = ''
    runHook preBuild
  '' + (if isQuantized then ''
    echo "Stage 1/2: ${checkpoint.filename} → ${intermediateBasename} (FP16)"
    python ${rwkv-cpp.src}/python/convert_pytorch_to_ggml.py \
      ${weights} \
      ${intermediateBasename} \
      FP16
    echo "Stage 2/2: ${intermediateBasename} → ${outputBasename} (${dtype})"
    ${rwkv-cpp}/bin/rwkv-quantize \
      ${intermediateBasename} \
      ${outputBasename} \
      ${dtype}
    rm ${intermediateBasename}
  '' else ''
    echo "Converting ${checkpoint.filename} → ${outputBasename} (${dtype})"
    python ${rwkv-cpp.src}/python/convert_pytorch_to_ggml.py \
      ${weights} \
      ${outputBasename} \
      ${dtype}
  '') + ''
    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall
    install -Dm644 ${outputBasename} $out/${outputBasename}
    # Also drop a stable-name symlink so downstream configs don't have
    # to know the exact version string.
    ln -s ${outputBasename} $out/model.bin
    runHook postInstall
  '';

  meta = {
    description = "RWKV-7 World 0.4B checkpoint, converted to rwkv.cpp .bin format";
    homepage = "https://huggingface.co/BlinkDL/rwkv-7-world";
    # Model weights: Apache 2.0 per BlinkDL (see the model card).
    license = lib.licenses.asl20;
    platforms = [ "x86_64-linux" ];
  };
}
