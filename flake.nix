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

      noesis-model = pkgs.callPackage ./nix/noesis-model.nix { inherit rwkv-cpp; };
      noesis-model-q8_0 = noesis-model.override { dtype = "Q8_0"; };
      noesis-model-q5_1 = noesis-model.override { dtype = "Q5_1"; };
      noesis-model-q4_0 = noesis-model.override { dtype = "Q4_0"; };

      noesis-runtime = pkgs.rustPlatform.buildRustPackage {
        pname = "noesis-runtime";
        version = "0.1.0";
        src = ./runtime;
        cargoLock.lockFile = ./runtime/Cargo.lock;
        buildAndTestSubdir = "noesis-runtime";
        nativeBuildInputs = [ pkgs.pkg-config ];
        # libsqlite3-sys is bundled — no external sqlite dep needed.
        meta = {
          description = "noesis persistent cognitive runtime supervisor";
          license = pkgs.lib.licenses.mit;
          platforms = [ "x86_64-linux" ];
        };
      };
    in {
      packages.${system} = {
        inherit rwkv-cpp noesis-model noesis-model-q8_0 noesis-model-q5_1 noesis-model-q4_0 noesis-runtime;
        default = noesis-runtime;
      };

      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          # Rust toolchain for the runtime workspace (once it lands).
          rustc cargo rust-analyzer clippy rustfmt
          # C/C++ toolchain for rwkv.cpp local hacking.
          cmake ninja pkg-config openblas
          # libclang for bindgen in noesis-rwkv-sys.
          llvmPackages.libclang
          # Python side of experiments (A0.6/A0.7/A0.8 sweeps).
          python312 uv
        ];
        # bindgen locates libclang via LIBCLANG_PATH.
        LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
      };

      homeModules.default = import ./nix/hm-module.nix self;
    };
}
