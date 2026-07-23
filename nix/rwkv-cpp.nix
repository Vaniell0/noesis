# rwkv.cpp derivation — CPU inference for the RWKV-7 backbone.
#
# Compiles with the host CPU's ISA flags (Alder Lake defaults here, since the
# development machine is an i5-1235U; hybrid P/E-cores share AVX2+AVX-VNNI+FMA
# but lack AVX-512, so `-march=alderlake` is the correct baseline).
#
# Callers can override `march` / `mtune` to retarget for other hardware
# (e.g. `mtune = "znver3"` on a Ryzen box).
#
# NOTE: RWKV-7 support in the upstream rwkv.cpp repo depends on the pinned
# revision. Bump `rev` / `hash` here when a new RWKV-7-capable release lands.
# On first build, set `hash = lib.fakeSha256` and let Nix print the actual one.

{ lib
, stdenv
, fetchFromGitHub
, cmake
, ninja
, pkg-config
, openblas
, march ? "alderlake"
, mtune ? "alderlake"
, useOpenBLAS ? true
, useOpenMP ? true
}:

stdenv.mkDerivation (finalAttrs: {
  pname = "rwkv-cpp";
  version = "unstable-2025-03-23";

  src = fetchFromGitHub {
    owner = "RWKV";
    repo = "rwkv.cpp";
    rev = "14663c83b6aba4885a47c1fba91204efc74a49d3";
    hash = "sha256-GEihyOkWy0ye/vaUZK7VmmdMtPfGxJiUh88rA78vNbk=";
    fetchSubmodules = true;
  };

  nativeBuildInputs = [ cmake ninja pkg-config ];
  buildInputs = lib.optional useOpenBLAS openblas;

  cmakeFlags = [
    "-DCMAKE_BUILD_TYPE=Release"
    "-DRWKV_BUILD_SHARED_LIBRARY=ON"
  ] ++ lib.optional useOpenBLAS "-DGGML_OPENBLAS=ON"
    ++ lib.optional (!useOpenMP) "-DGGML_OPENMP=OFF";

  # rwkv.cpp's CMake only installs the ggml sublibraries — the actual
  # librwkv.so and rwkv.h have no install() target upstream. Copy them
  # by hand so downstream consumers (noesis-rwkv-sys bindgen + linker)
  # can point at a stable prefix. Also grab the `bin/rwkv_quantize`
  # executable that CMake built from `extras/quantize.c` — upstream
  # doesn't install it, but `noesis-model.nix` needs it to produce
  # Q4/Q5/Q8 variants without a Python round-trip.
  postInstall = ''
    install -Dm755 librwkv.so $out/lib/librwkv.so
    install -Dm644 $src/rwkv.h $out/include/rwkv.h
    install -Dm755 bin/rwkv_quantize $out/bin/rwkv-quantize
    # CMake links the extras against build-tree relative paths, so the
    # binary embeds an RPATH pointing at /build/. Rewrite it to point at
    # our final $out/lib so the nix builder's forbidden-refs check passes.
    patchelf --set-rpath "$out/lib" $out/bin/rwkv-quantize
  '';

  # ISA-tuning flags. `-O3 -fomit-frame-pointer` on top of Release for the
  # hot decode loop; ggml's SIMD intrinsics already assume AVX2, so the
  # `-march` here mostly benefits the non-intrinsic scalar wrapper code.
  env.NIX_CFLAGS_COMPILE = lib.concatStringsSep " " [
    "-march=${march}"
    "-mtune=${mtune}"
    "-O3"
    "-fomit-frame-pointer"
  ];

  # Expose the pinned source tree so `noesis-model.nix` (below) can reuse
  # the shipped `python/convert_pytorch_to_ggml.py` without pinning the
  # revision twice.
  passthru = {
    src = finalAttrs.src;
  };

  meta = {
    description = "RWKV inference in C/C++ (CPU + quantised), tuned for host ISA";
    homepage = "https://github.com/RWKV/rwkv.cpp";
    license = lib.licenses.mit;
    platforms = [ "x86_64-linux" ];
  };
})
