# noesis home-manager module.
#
# Wires up:
#   - the rwkv.cpp inference binary (host-tuned) into $HOME
#   - a user-scope systemd service that runs the noesis-runtime supervisor,
#     which loads rwkv.cpp in-process.
#
# Default model layout (can be overridden per-option):
#   substrate  (always-resident reasoning model) → G1H 2.9B Q5_1
#   utility    (emit-gate / classifier / formatter) → World 0.4B Q5_1
#
# Usage from ~/.config/home-manager/flake.nix:
#   inputs.noesis.url = "path:/home/vaniello/Desktop/projects/noesis";
#   inputs.noesis.inputs.nixpkgs.follows = "nixpkgs";
#   ... modules = [ noesis.homeModules.default ... ];
# then in home.nix:
#   services.noesis-runtime.enable = true;

self:
{ config, lib, pkgs, ... }:

let
  cfg    = config.services.noesis-runtime;
  system = pkgs.system;

  runtimePkg       = self.packages.${system}.noesis-runtime;
  rwkvCppPkg       = self.packages.${system}.rwkv-cpp;
  substratePkg     = self.packages.${system}.noesis-model-g1h-29b;       # G1H 2.9B Q5_1
  utilityPkg       = self.packages.${system}.noesis-model-q5_1;           # World 0.4B Q5_1
in
{
  options.services.noesis-runtime = {
    enable = lib.mkEnableOption "noesis persistent cognitive runtime";

    autoStart = lib.mkOption {
      type    = lib.types.bool;
      default = false;
      description = ''
        Start noesis-runtime on graphical-session.target. Disabled by default
        because Phase A / Phase B are still experimental; flip to true only
        after A0.3 sustained-idle verdict lands.
      '';
    };

    # ── Model paths ───────────────────────────────────────────────────────────

    modelPath = lib.mkOption {
      type    = lib.types.str;
      default = "${config.home.homeDirectory}/.cache/huggingface/hub";
      description = ''
        Fallback checkpoint directory. Only used when `modelFile` is null AND
        the runtime is asked to resolve a model from the HF cache. The normal
        path is `modelFile`.
      '';
    };

    modelFile = lib.mkOption {
      type    = lib.types.nullOr lib.types.str;
      default = "${substratePkg}/model.bin";
      description = ''
        Absolute path to the substrate model in rwkv.cpp `.bin` format.
        Default: G1H 2.9B Q5_1 built by this flake.
        Override example (World 0.4B FP16):
          modelFile = "''${self.packages.x86_64-linux.noesis-model}/model.bin";
      '';
    };

    utilityModelFile = lib.mkOption {
      type    = lib.types.nullOr lib.types.str;
      default = "${utilityPkg}/model.bin";
      description = ''
        Absolute path to the utility model binary (0.4B or smaller).
        Used for emit-gate, importance classification, tool-call formatting —
        never for reasoning (plan §8 single-substrate lock).
        Default: World 0.4B Q5_1. Set to null to disable (heuristics only).
      '';
    };

    # ── Threading ─────────────────────────────────────────────────────────────

    threads = lib.mkOption {
      type    = lib.types.int;
      default = 1;
      description = ''
        Thread count for the ambient/heartbeat rwkv-cpp context.
        Drives background drip. Keep low to respect the H1 fan-off budget.
        Use `calibrate --thread-sweep` to find the optimal value for your
        machine and `fan_safe_cpu_percent`.
      '';
    };

    httpThreads = lib.mkOption {
      type    = lib.types.nullOr lib.types.int;
      default = 12;
      description = ''
        Thread count for the interactive/HTTP clone context.
        Defaults to 12 (peak throughput on i5-1235U: ~19.6 tok/s on G1H 2.9B).
        When null, falls back to `threads`.
      '';
    };

    utilityThreads = lib.mkOption {
      type    = lib.types.int;
      default = 2;
      description = "Thread count for the utility model context.";
    };

    utilityKeepAliveSecs = lib.mkOption {
      type    = lib.types.int;
      default = 300;
      description = ''
        Seconds of idle before the utility model is unloaded to free RAM.
        Set 0 for always-resident (not recommended when substrate 2.9B is loaded).
      '';
    };

    # ── HTTP shim ─────────────────────────────────────────────────────────────

    httpBind = lib.mkOption {
      type    = lib.types.nullOr lib.types.str;
      default = "127.0.0.1:11435";
      description = ''
        `host:port` for the HTTP shim (POST /api/generate, /v1/messages,
        /v1/chat/completions). Set to null to disable.
      '';
    };

    # ── State / lens ──────────────────────────────────────────────────────────

    statePath = lib.mkOption {
      type    = lib.types.str;
      default = "${config.home.homeDirectory}/.local/share/noesis";
      description = "Root for noesis persistent state (memory zones, logs).";
    };

    lensRoot = lib.mkOption {
      type    = lib.types.str;
      default = "${config.home.homeDirectory}/.local/share/noesis/lenses";
      description = ''
        Root directory for per-lens WKV snapshots
        (`<lensRoot>/<lens_id>/{wkv.snapshot,meta.json}`). Set to empty
        string to disable /lens/* endpoints.
      '';
    };

    # ── Infrastructure ────────────────────────────────────────────────────────

    inferenceBackend = lib.mkOption {
      type    = lib.types.enum [ "rwkv-cpp" ];
      default = "rwkv-cpp";
      description = "Inference backend. Only rwkv-cpp after the 2026-07-25 Ollama drop.";
    };

    memoryMax = lib.mkOption {
      type    = lib.types.str;
      default = "8G";
      description = "systemd MemoryMax for the runtime process. 8G to fit G1H 2.9B Q5_1 (2.6G) + working set.";
    };

    sourcePath = lib.mkOption {
      type    = lib.types.nullOr lib.types.str;
      default = null;
      description = ''
        Absolute path to the noesis source checkout on this host (where
        `flake.nix` lives). Required when `autoRebuild` is enabled.
      '';
    };

    autoRebuild = lib.mkOption {
      type    = lib.types.bool;
      default = false;
      description = ''
        Enable a systemd .path watcher that rebuilds noesis-runtime from
        `sourcePath` and restarts the service whenever files under
        `runtime/` change. Requires `sourcePath` to be set.
      '';
    };
  };

  config = lib.mkIf cfg.enable {
    home.packages = [ runtimePkg rwkvCppPkg ];

    home.activation.noesisStateDirs = lib.hm.dag.entryAfter [ "writeBoundary" ] ''
      $DRY_RUN_CMD mkdir -p ${cfg.statePath}/input_events
      $DRY_RUN_CMD mkdir -p ${cfg.statePath}/system_obs
      $DRY_RUN_CMD mkdir -p ${cfg.statePath}/personal_vault
      $DRY_RUN_CMD mkdir -p ${cfg.statePath}/session_scratch
      $DRY_RUN_CMD mkdir -p ${cfg.statePath}/logs
      ${lib.optionalString (cfg.lensRoot != "") ''$DRY_RUN_CMD mkdir -p ${cfg.lensRoot}''}
    '';

    xdg.configFile."noesis/runtime.toml".text = ''
      # noesis runtime config. Regenerated by home-manager; edit the module
      # rather than this file.

      state_path         = "${cfg.statePath}"
      model_path         = "${cfg.modelPath}"
      inference_backend  = "${cfg.inferenceBackend}"
      ${lib.optionalString (cfg.lensRoot != "") ''lens_root = "${cfg.lensRoot}"''}

      [rwkv_cpp]
      # Substrate model (reasoning, always-resident). G1H 2.9B Q5_1 by default.
      threads = ${toString cfg.threads}
      ${lib.optionalString (cfg.httpThreads != null) ''http_threads = ${toString cfg.httpThreads}''}
      ${lib.optionalString (cfg.modelFile != null) ''model_path = "${cfg.modelFile}"''}
      ${lib.optionalString (cfg.httpBind != null) ''http_bind = "${cfg.httpBind}"''}
      ${lib.optionalString (cfg.utilityModelFile != null) ''
      # Utility model (0.4B — emit-gate, classifier, formatter).
      utility_model_path    = "${cfg.utilityModelFile}"
      utility_threads       = ${toString cfg.utilityThreads}
      utility_keep_alive_secs = ${toString cfg.utilityKeepAliveSecs}
      ''}
    '';

    assertions = [
      {
        assertion = !cfg.autoRebuild || cfg.sourcePath != null;
        message   = "services.noesis-runtime.autoRebuild requires sourcePath to be set.";
      }
    ];

    systemd.user.services.noesis-runtime = {
      Unit = {
        Description = "noesis persistent cognitive runtime";
        After  = [ "graphical-session.target" ];
        PartOf = [ "graphical-session.target" ];
      };
      Service = {
        ExecStart   = "${runtimePkg}/bin/noesis-runtime";
        Environment = [
          "NOESIS_CONFIG=%h/.config/noesis/runtime.toml"
          "RUST_LOG=info"
        ];
        Restart     = "on-failure";
        RestartSec  = "5s";
        NoNewPrivileges = true;
        ProtectSystem   = "strict";
        ReadWritePaths  = [ cfg.statePath ]
          ++ lib.optional (cfg.lensRoot != "") cfg.lensRoot;
        MemoryMax = cfg.memoryMax;
      };
      Install = lib.mkIf cfg.autoStart {
        WantedBy = [ "graphical-session.target" ];
      };
    };

    systemd.user.services.noesis-runtime-rebuild = lib.mkIf cfg.autoRebuild {
      Unit = {
        Description = "Rebuild noesis-runtime from source and restart the service";
        After = [ "network.target" ];
      };
      Service = {
        Type = "oneshot";
        Environment = [
          "PATH=${lib.makeBinPath [ pkgs.nix pkgs.systemd pkgs.coreutils ]}"
        ];
        WorkingDirectory = cfg.sourcePath;
        ExecStart = pkgs.writeShellScript "noesis-runtime-rebuild" ''
          set -euo pipefail
          cd "${cfg.sourcePath}"
          nix build .#noesis-runtime --no-link --print-out-paths
          systemctl --user try-restart noesis-runtime.service || true
        '';
      };
    };

    systemd.user.paths.noesis-runtime-rebuild = lib.mkIf cfg.autoRebuild {
      Unit = {
        Description = "Watch noesis runtime/ source for changes and trigger rebuild";
      };
      Path = {
        PathChanged = "${cfg.sourcePath}/runtime";
        Unit        = "noesis-runtime-rebuild.service";
      };
      Install = {
        WantedBy = [ "default.target" ];
      };
    };
  };
}
