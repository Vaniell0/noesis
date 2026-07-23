//! Build script for `noesis-rwkv-sys`.
//!
//! Discovers rwkv.cpp's `librwkv.so` + `rwkv.h` via one of, in order:
//!
//! 1. `NOESIS_RWKV_CPP_PREFIX` — explicit override (used by CI / nix build).
//! 2. `RWKV_CPP_PREFIX` — same, older name kept for compatibility.
//! 3. `<workspace>/../result` — the default `nix build .#rwkv-cpp` symlink
//!    at the project root. This is the convention for a hacker in the
//!    dev shell who just ran `nix build .#rwkv-cpp` once.
//!
//! Fails loudly if the prefix does not contain `include/rwkv.h` — silent
//! fallback would just produce a link error 30 seconds later.

use std::env;
use std::path::{Path, PathBuf};

fn discover_prefix() -> PathBuf {
    for var in ["NOESIS_RWKV_CPP_PREFIX", "RWKV_CPP_PREFIX"] {
        if let Ok(p) = env::var(var) {
            let path = PathBuf::from(p);
            if path.join("include/rwkv.h").is_file() {
                return path;
            }
            panic!("{var}={} does not contain include/rwkv.h", path.display());
        }
    }
    // Fall back to the standard nix build outputs at the project root.
    // Preferred: `nix build .#rwkv-cpp -o result-rwkv-cpp` (leaves the
    // plain `result` symlink free for other derivations). Legacy: plain
    // `result` from `nix build .#rwkv-cpp`.
    // CARGO_MANIFEST_DIR is runtime/noesis-rwkv-sys, so ../../<name>.
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    for name in ["../../result-rwkv-cpp", "../../result"] {
        if let Ok(p) = manifest.join(name).canonicalize() {
            if p.join("include/rwkv.h").is_file() {
                return p;
            }
        }
    }
    panic!(
        "rwkv.cpp prefix not found. Run `nix build .#rwkv-cpp -o result-rwkv-cpp` at the \
         project root, or set NOESIS_RWKV_CPP_PREFIX to a directory \
         containing include/rwkv.h and lib/librwkv.so."
    );
}

fn main() {
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=NOESIS_RWKV_CPP_PREFIX");
    println!("cargo:rerun-if-env-changed=RWKV_CPP_PREFIX");

    let prefix = discover_prefix();
    let include = prefix.join("include");
    let lib = prefix.join("lib");

    println!("cargo:rustc-link-search=native={}", lib.display());
    println!("cargo:rustc-link-lib=dylib=rwkv");
    // ggml sublibraries are dlopen'd by librwkv.so at load time; ensure
    // the linker knows the search path so the final binary's rpath sees
    // them too.
    println!("cargo:rustc-link-lib=dylib=ggml");
    println!("cargo:rustc-link-lib=dylib=ggml-base");
    println!("cargo:rustc-link-lib=dylib=ggml-cpu");

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}", include.display()))
        // Only surface the rwkv.h symbols — ggml.h drags in hundreds of
        // items we don't need.
        .allowlist_function("rwkv_.*")
        .allowlist_type("rwkv_.*")
        .allowlist_var("RWKV_.*")
        .prepend_enum_name(false)
        .default_enum_style(bindgen::EnumVariation::Consts)
        .generate_comments(true)
        .generate()
        .expect("bindgen failed on rwkv.h");

    let out_path: PathBuf = PathBuf::from(env::var("OUT_DIR").unwrap()).join("bindings.rs");
    bindings
        .write_to_file(&out_path)
        .expect("failed to write bindings.rs");
    // Emit a metadata key so downstream `links = "rwkv"` consumers
    // (`noesis-rwkv`, `noesis-runtime`) see where the .so lives.
    println!("cargo:prefix={}", prefix.display());
    println!("cargo:libdir={}", lib.display());
    println!("cargo:includedir={}", include.display());
    // Runtime search — ensure the produced binary can find librwkv.so
    // without LD_LIBRARY_PATH gymnastics during dev.
    let _ = Path::new(&lib);
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib.display());
}
