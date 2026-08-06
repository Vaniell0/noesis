{
  description = "noesis — persistent cognitive runtime (RWKV-7 backbone)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };

      rwkv-cpp = pkgs.callPackage ./nix/rwkv-cpp.nix { };

      # ── Model checkpoints ────────────────────────────────────────────────────
      #
      # Each variant is a `noesis-model.nix` call with a different `model`
      # record. Public models use `pkgs.fetchurl`; locally-trained weights use
      # `builtins.path` (available only on the build host that has the file).
      #
      # Naming convention:  noesis-model-<name>[-<dtype>]
      #   default dtype omitted → FP16 (for 0.4B; Q5_1 for 2.9B variants)

      mkModel = model: dtype:
        pkgs.callPackage ./nix/noesis-model.nix { inherit rwkv-cpp model dtype; };

      # ── 0.4B World v2.9 (public, fetchurl) ──────────────────────────────────
      world-04b-pth = pkgs.fetchurl {
        url  = "https://huggingface.co/BlinkDL/rwkv-7-world/resolve/main/RWKV-x070-World-0.4B-v2.9-20250107-ctx4096.pth";
        hash = "sha256-wIz2ArM+WaVxe48Mr69/BMUMHGcWbkd6ovR8XKGA2ko=";
      };
      world-04b = { name = "rwkv7-world-0.4b"; version = "v2.9-20250107"; pth = world-04b-pth; };

      noesis-model         = mkModel world-04b "FP16";
      noesis-model-q8_0    = mkModel world-04b "Q8_0";
      noesis-model-q5_1    = mkModel world-04b "Q5_1";
      noesis-model-q4_0    = mkModel world-04b "Q4_0";

      # ── 2.9B World v3 / 20250211 (public, fetchurl) ─────────────────────────
      #   HF: BlinkDL/rwkv-7-world → RWKV-x070-World-2.9B-v3-20250211-ctx4096.pth
      #   sha256 verified against local copy 2026-07-28.
      world-29b-pth = pkgs.fetchurl {
        url  = "https://huggingface.co/BlinkDL/rwkv-7-world/resolve/main/RWKV-x070-World-2.9B-v3-20250211-ctx4096.pth";
        hash = "sha256-XXkBuR7xZGEV2wPQPRcTcLykVAYa1CahJW/anhxCjPg=";
      };
      world-29b = { name = "rwkv7-world-2.9b-v3"; version = "v3-20250211"; pth = world-29b-pth; };

      noesis-model-world-29b       = mkModel world-29b "Q5_1";
      noesis-model-world-29b-q8_0  = mkModel world-29b "Q8_0";

      # ── G1H 2.9B (locally-trained, builtins.path) ────────────────────────────
      #   Weights live at ~/.libs/models/rwkv7/ on the dev host.
      #   Build will fail on machines that don't have this path — expected.
      g1h-29b-pth = builtins.path {
        path = /home/vaniello/.libs/models/rwkv7/rwkv7-g1h-2.9b-20260710-ctx10240.pth;
        name = "rwkv7-g1h-2.9b-20260710-ctx10240.pth";
      };
      g1h-29b = { name = "rwkv7-g1h-2.9b"; version = "20260710-ctx10240"; pth = g1h-29b-pth; };

      noesis-model-g1h-29b       = mkModel g1h-29b "Q5_1";
      noesis-model-g1h-29b-q8_0  = mkModel g1h-29b "Q8_0";
      noesis-model-g1h-29b-fp16  = mkModel g1h-29b "FP16";

      # ── G1D 0.4B (locally-trained, builtins.path) ────────────────────────────
      g1d-04b-pth = builtins.path {
        path = /home/vaniello/.libs/models/rwkv7/rwkv7-g1d-0.4b-20260210-ctx8192.pth;
        name = "rwkv7-g1d-0.4b-20260210-ctx8192.pth";
      };
      g1d-04b = { name = "rwkv7-g1d-0.4b"; version = "20260210-ctx8192"; pth = g1d-04b-pth; };

      noesis-model-g1d-04b      = mkModel g1d-04b "Q8_0";
      noesis-model-g1d-04b-fp16 = mkModel g1d-04b "FP16";

      # ── G1H 1.5B (locally-trained, builtins.path) ────────────────────────────
      g1h-15b-pth = builtins.path {
        path = /home/vaniello/.libs/models/rwkv7/rwkv7-g1h-1.5b-20260710-ctx10240.pth;
        name = "rwkv7-g1h-1.5b-20260710-ctx10240.pth";
      };
      g1h-15b = { name = "rwkv7-g1h-1.5b"; version = "20260710-ctx10240"; pth = g1h-15b-pth; };

      noesis-model-g1h-15b       = mkModel g1h-15b "Q5_1";
      noesis-model-g1h-15b-q8_0  = mkModel g1h-15b "Q8_0";

      # ── Runtime ──────────────────────────────────────────────────────────────
      noesis-runtime = pkgs.rustPlatform.buildRustPackage {
        pname   = "noesis-runtime";
        version = "0.1.0";
        src     = ./runtime;
        cargoLock.lockFile = ./runtime/Cargo.lock;
        buildAndTestSubdir = "noesis-runtime";
        nativeBuildInputs = [ pkgs.pkg-config pkgs.llvmPackages.libclang ];
        buildInputs       = [ rwkv-cpp ];
        NOESIS_RWKV_CPP_PREFIX = "${rwkv-cpp}";
        LIBCLANG_PATH          = "${pkgs.llvmPackages.libclang.lib}/lib";
        meta = {
          description = "noesis persistent cognitive runtime supervisor";
          license     = pkgs.lib.licenses.mit;
          platforms   = [ "x86_64-linux" ];
        };
      };
    in {
      packages.${system} = {
        inherit rwkv-cpp noesis-runtime;
        # 0.4B world
        inherit noesis-model noesis-model-q8_0 noesis-model-q5_1 noesis-model-q4_0;
        # 2.9B world-v3
        inherit noesis-model-world-29b noesis-model-world-29b-q8_0;
        # 2.9B G1H
        inherit noesis-model-g1h-29b noesis-model-g1h-29b-q8_0 noesis-model-g1h-29b-fp16;
        # 0.4B G1D
        inherit noesis-model-g1d-04b noesis-model-g1d-04b-fp16;
        # 1.5B G1H
        inherit noesis-model-g1h-15b noesis-model-g1h-15b-q8_0;
        default = noesis-runtime;
      };

      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          rustc cargo rust-analyzer clippy rustfmt
          cmake ninja pkg-config openblas
          llvmPackages.libclang
          python312 uv
        ];
        LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
      };

      homeModules.default  = import ./nix/hm-module.nix self;
      nixosModules.default = import ./nix/nixos-module.nix self;
    };
}
