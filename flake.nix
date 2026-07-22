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
        inherit rwkv-cpp noesis-runtime;
        default = noesis-runtime;
      };

      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          # Rust toolchain for the runtime workspace (once it lands).
          rustc cargo rust-analyzer clippy rustfmt
          # C/C++ toolchain for rwkv.cpp local hacking.
          cmake ninja pkg-config openblas
          # Python side of experiments (A0.6/A0.7/A0.8 sweeps).
          python312 uv
        ];
      };

      homeModules.default = import ./nix/hm-module.nix self;
    };
}
