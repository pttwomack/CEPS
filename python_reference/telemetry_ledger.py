import os
import json
import time
import subprocess
import numpy as np
from typing import Optional, Dict, Any
from mazedemo.telemetry import StateContext, CanonicalBridgeResult
from mazedemo.layout import to_cartesian


class TelemetryLedger:
    """Decoupled passive ledger handling continuous telemetry streaming, git state extraction, and crash snapshot isolation."""

    def __init__(
        self,
        geometry: str,
        run_id: Optional[str | int] = None,
        output_dir: str = "mazetool_results",
    ):
        self.geometry = geometry.lower()
        self.run_id = str(run_id) if run_id is not None else str(int(time.time()))
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.git_state = self._extract_git_state()

    def _extract_git_state(self) -> Dict[str, Any]:
        """Extracts current git commit, branch, and workspace dirty status silently via subprocess."""
        git_info = {
            "commit": "unknown",
            "branch": "unknown",
            "dirty_workspace": False,
        }
        try:
            commit_res = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            git_info["commit"] = commit_res.stdout.strip()
        except Exception:
            pass

        try:
            branch_res = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            git_info["branch"] = branch_res.stdout.strip()
        except Exception:
            pass

        try:
            status_res = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
            )
            git_info["dirty_workspace"] = bool(status_res.stdout.strip())
        except Exception:
            pass

        return git_info

    def stream_iteration(
        self, context: StateContext, exit_message: Optional[str] = None
    ) -> None:
        """Appends a lightweight JSON string (scalars only) to an append-only .jsonl file per iteration pass."""
        if context is None:
            return

        grad_norm = float(np.linalg.norm(context.gradient)) if len(context.gradient) > 0 else 0.0
        max_u = float(np.max(np.abs(context.u))) if len(context.u) > 0 else 0.0

        record = {
            "iteration": int(context.iteration),
            "energy": float(context.energy),
            "gradient_norm": grad_norm,
            "max_scale_factor": max_u,
            "exit_message": exit_message,
            "git_state": self.git_state,
            "timestamp": time.time(),
        }

        stream_filename = f"telemetry_stream_{self.geometry}_{self.run_id}.jsonl"
        stream_path = os.path.join(self.output_dir, stream_filename)

        with open(stream_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def serialize_canonical_bridge(
        self,
        result: CanonicalBridgeResult,
        filename: str = "canonical_bridge_telemetry.json",
    ) -> str:
        """Serializes CanonicalBridgeResult payload to JSON for downstream tools or offline renderers."""
        placed_triangles_cartesian = {}
        for f_id, placement in result.placed_triangles.items():
            f_str = str(f_id)
            placed_triangles_cartesian[f_str] = {}
            for v_id, pos in placement.items():
                v_str = str(v_id)
                cart = to_cartesian(pos)
                placed_triangles_cartesian[f_str][v_str] = cart.tolist()

        dilatation_map_str = {
            str(f_id): float(val) for f_id, val in result.local_dilatation_map.items()
        }

        holonomy_seams_str = {}
        if isinstance(result.holonomy_seams, dict):
            for k, v in result.holonomy_seams.items():
                holonomy_seams_str[str(k)] = str(v)

        arc_samples_cart = to_cartesian(result.poincare_arc_samples)

        payload = {
            "timestamp": time.time(),
            "geometry": "hyperbolic" if result.is_hyperbolic else "euclidean",
            "git_state": self.git_state,
            "placed_triangles": placed_triangles_cartesian,
            "local_dilatation_map": dilatation_map_str,
            "holonomy_seams": holonomy_seams_str,
            "face_sequence": [str(f) for f in result.face_sequence],
            "poincare_arc_samples": arc_samples_cart.tolist(),
            "grid_path_coords": result.grid_path_coords.tolist(),
        }

        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)

        return output_path

    def log_crash(
        self,
        exception: Exception,
        anomaly_info: Optional[Dict[str, Any]],
        context: Optional[StateContext],
        fallback_u: Optional[np.ndarray] = None,
        fallback_grad: Optional[np.ndarray] = None,
    ) -> str:
        """Dumps a heavy isolated crash snapshot JSON file containing full scale factor and gradient arrays."""
        ts = int(time.time())
        u_list = []
        if context is not None and len(context.u) > 0:
            u_list = context.u.tolist()
        elif fallback_u is not None:
            u_list = fallback_u.tolist()

        grad_list = []
        if context is not None and len(context.gradient) > 0:
            grad_list = context.gradient.tolist()
        elif fallback_grad is not None:
            grad_list = fallback_grad.tolist()

        energy_val = (
            float(context.energy)
            if context is not None and hasattr(context, "energy")
            else float("nan")
        )
        iter_num = context.iteration if context is not None else 0

        snapshot_payload = {
            "timestamp": ts,
            "geometry": self.geometry,
            "iteration": iter_num,
            "exception": str(exception),
            "reason": str(exception),
            "anomaly_info": anomaly_info or {},
            "git_state": self.git_state,
            "energy": energy_val,
            "u": u_list,
            "gradient": grad_list,
        }

        crash_dir = os.path.join(self.output_dir, "crash_dumps")
        os.makedirs(crash_dir, exist_ok=True)
        snapshot_path = os.path.join(crash_dir, f"snapshot_{self.geometry}_{ts}.json")

        with open(snapshot_path, "w", encoding="utf-8") as f:
            json.dump(snapshot_payload, f, indent=4)

        if os.path.exists("/content/drive/MyDrive/"):
            gdrive_crash_dir = "/content/drive/MyDrive/mazetool_results/crash_dumps/"
            try:
                os.makedirs(gdrive_crash_dir, exist_ok=True)
                gdrive_snapshot_path = os.path.join(
                    gdrive_crash_dir, f"snapshot_{self.geometry}_{ts}.json"
                )
                with open(gdrive_snapshot_path, "w", encoding="utf-8") as f:
                    json.dump(snapshot_payload, f, indent=4)
            except Exception:
                pass

        return snapshot_path
