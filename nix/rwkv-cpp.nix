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
  version = "unstable-2026-07-22";

  src = fetchFromGitHub {
    owner = "RWKV";
    repo = "rwkv.cpp";
    # TODO(user): pin a rev that supports RWKV-7 and refresh the hash.
    # On first build, `nix build` will report the actual SRI hash.
    rev = "HEAD";
    hash = lib.fakeHash;
    fetchSubmodules = true;
  };

  nativeBuildInputs = [ cmake ninja pkg-config ];
  buildInputs = lib.optional useOpenBLAS openblas;

  cmakeFlags = [
    "-DCMAKE_BUILD_TYPE=Release"
    "-DRWKV_BUILD_SHARED_LIBRARY=ON"
  ] ++ lib.optional useOpenBLAS "-DGGML_OPENBLAS=ON"
    ++ lib.optional (!useOpenMP) "-DGGML_OPENMP=OFF";

  # ISA-tuning flags. `-O3 -fomit-frame-pointer` on top of Release for the
  # hot decode loop; ggml's SIMD intrinsics already assume AVX2, so the
  # `-march` here mostly benefits the non-intrinsic scalar wrapper code.
  env.NIX_CFLAGS_COMPILE = lib.concatStringsSep " " [
    "-march=${march}"
    "-mtune=${mtune}"
    "-O3"
    "-fomit-frame-pointer"
  ];

  meta = {
    description = "RWKV inference in C/C++ (CPU + quantised), tuned for host ISA";
    homepage = "https://github.com/RWKV/rwkv.cpp";
    license = lib.licenses.mit;
    platforms = [ "x86_64-linux" ];
  };
})
