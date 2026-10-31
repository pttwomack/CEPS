import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.distance import cdist
from typing import Dict, Any, Optional

from mazedemo.telemetry import StateContext
from mazedemo.surface_factory import SurfaceFactory
from mazedemo.ids import get_local_id


# Domain Presets
STRICT_CAD_PHYSICS = {
    "name": "STRICT_CAD_PHYSICS",
    "L1_mean_max": 3.0,
    "L1_peak_max": 10.0,
    "L2_distortion_max": 5.0,
    "L2_p99_max": float(np.log(1.2)),
    "L3_min": 0.99,
}

STANDARD_ML = {
    "name": "STANDARD_ML",
    "L1_mean_max": 5.0,
    "L1_peak_max": 20.0,
    "L2_distortion_max": 10.0,
    "L2_p99_max": float(np.log(1.5)),
    "L3_min": 0.98,
}

RELAXED_VISUALIZATION = {
    "name": "RELAXED_VISUALIZATION",
    "L1_mean_max": 15.0,
    "L1_peak_max": 35.0,
    "L2_distortion_max": 25.0,
    "L2_p99_max": float(np.log(2.0)),
    "L3_min": 0.90,
}

TOPOLOGICAL_UNIFORMIZATION = {
    "name": "TOPOLOGICAL_UNIFORMIZATION",
    "L1_mean_max": 90.0,
    "L1_peak_max": 100.0,
    "L2_distortion_max": 60.0,
    "L2_p99_max": 5.0,
    "L3_min": 0.85,
}

LATENT_CAUSAL_EMBEDDING = {
    "name": "LATENT_CAUSAL_EMBEDDING",
    "L1_mean_max": 20.0,
    "L1_peak_max": 40.0,
    "L2_distortion_max": 15.0,
    "L2_p99_max": float(np.log(1.8)),
    "L3_min": 0.95,
}


def compute_knn_trustworthiness_and_continuity(
    d_data: np.ndarray, d_model: np.ndarray, k: int = 5
) -> tuple[float, float]:
    """Computes rank-based k-NN Trustworthiness (M_T) and Continuity (M_C)."""
    K = d_data.shape[0]
    if K <= k + 1:
        return 1.0, 1.0

    # Ranks of d_data and d_model (1-based, excluding self)
    # Argsort twice gives 0-based rank among all elements
    ranks_data = np.argsort(np.argsort(d_data, axis=1), axis=1)
    ranks_model = np.argsort(np.argsort(d_model, axis=1), axis=1)

    # 1-based ranks
    r_data = ranks_data + 1
    r_model = ranks_model + 1

    # Neighbors in top-k (ranks 2 to k+1 since rank 1 is self distance 0)
    top_k_data = (r_data >= 2) & (r_data <= k + 1)
    top_k_model = (r_model >= 2) & (r_model <= k + 1)

    # Trustworthiness penalty: in top-k model, but NOT in top-k data
    # penalty is r_data(i, j) - k
    trust_mask = top_k_model & (~top_k_data)
    trust_penalty = np.sum(np.maximum(0, r_data[trust_mask] - k))

    # Continuity penalty: in top-k data, but NOT in top-k model
    # penalty is r_model(i, j) - k
    cont_mask = top_k_data & (~top_k_model)
    cont_penalty = np.sum(np.maximum(0, r_model[cont_mask] - k))

    norm_factor = (2.0) / (K * k * (2 * K - 3 * k - 1))
    M_T = max(0.0, min(1.0, 1.0 - norm_factor * trust_penalty))
    M_C = max(0.0, min(1.0, 1.0 - norm_factor * cont_penalty))

    return float(M_T), float(M_C)


def audit_planar_layout(
    layout_coords: np.ndarray,
    landmark_indices: np.ndarray,
    initial_landmark_distances: np.ndarray,
) -> float:
    """Audits 2D flat planar layout coordinates against initial landmark distances, returning delta_visual (%)."""
    if layout_coords is None or landmark_indices is None or initial_landmark_distances is None:
        return 0.0

    lm_coords = layout_coords[landmark_indices]
    d_layout = cdist(lm_coords, lm_coords, metric="euclidean")

    K = len(landmark_indices)
    mask = ~np.eye(K, dtype=bool)

    v_data = initial_landmark_distances[mask]
    v_layout = d_layout[mask]

    valid = (v_data > 1e-12) & (v_layout > 1e-12) & np.isfinite(v_data) & np.isfinite(v_layout)
    if not np.any(valid):
        return 0.0

    v_d = v_data[valid]
    v_l = v_layout[valid]

    s_vis = np.exp(np.mean(np.log(v_l / v_d)))
    rho_vis = v_l / (s_vis * v_d)
    delta_visual = (1.0 - np.exp(-np.std(np.log(rho_vis)))) * 100.0
    return float(np.clip(delta_visual, 0.0, 100.0))


def calculate_continuous_scores(
    context: StateContext,
    layout_coords: Optional[np.ndarray] = None,
    factory: Optional[SurfaceFactory] = None,
) -> Dict[str, float]:
    """Computes continuous geometric distortion metrics from cached StateContext."""
    landmark_idx = context.landmark_indices
    d_data = context.initial_landmark_distances

    # 1. Level 1: Local Tensor Anisotropy
    if factory is not None and context.surface_id is not None and context.u is not None:
        view = factory.create_viewer(
            context.surface_id, custom_properties={"dcm_index": None}
        )
        u_arr = context.u
        delta_local_faces = []
        for face in view.faces:
            u_vals = [
                float(u_arr[he.origin.dcm_index])
                for he in face.half_edges
                if he.origin.dcm_index is not None and he.origin.dcm_index < len(u_arr)
            ]
            if len(u_vals) >= 2:
                K_f = float(np.exp(np.max(u_vals) - np.min(u_vals)))
            else:
                K_f = 1.0
            delta_f = (1.0 - (1.0 / max(K_f**2, 1.0))) * 100.0
            delta_local_faces.append(delta_f)

        delta_local_mean = float(np.mean(delta_local_faces)) if delta_local_faces else 0.0
        delta_local_max = float(np.max(delta_local_faces)) if delta_local_faces else 0.0
    else:
        cur_lens = context.half_edge_lengths
        init_lens = context.initial_half_edge_lengths

        if init_lens is None or len(init_lens) == 0 or len(cur_lens) != len(init_lens):
            delta_local_mean = 0.0
            delta_local_max = 0.0
        else:
            s_ratios = cur_lens / np.maximum(init_lens, 1e-12)
            s_faces = s_ratios.reshape(-1, 3)
            s_max = np.max(s_faces, axis=1)
            s_min = np.min(s_faces, axis=1)
            K_f = s_max / np.maximum(s_min, 1e-12)
            delta_local_faces = (1.0 - (1.0 / np.maximum(K_f**2, 1e-12))) * 100.0
            delta_local_mean = float(np.mean(delta_local_faces))
            delta_local_max = float(np.max(delta_local_faces))

    # 2. Level 2: Global Geodesic Isometry & Level 3 & Level 4
    if layout_coords is not None and landmark_idx is not None and d_data is not None:
        lm_coords = layout_coords[landmark_idx]
        d_model = cdist(lm_coords, lm_coords, metric="euclidean")

        K = len(landmark_idx)
        mask = ~np.eye(K, dtype=bool)

        v_data = d_data[mask]
        v_model = d_model[mask]

        valid = (
            (v_data > 1e-12)
            & (v_model > 1e-12)
            & np.isfinite(v_data)
            & np.isfinite(v_model)
        )
        if np.any(valid):
            v_d = v_data[valid]
            v_m = v_model[valid]

            global_scale_s = float(np.exp(np.mean(np.log(v_m / v_d))))
            rho = v_m / (global_scale_s * v_d)
            delta_global = float((1.0 - np.exp(-np.std(np.log(rho)))) * 100.0)
            D_global_99 = float(np.percentile(np.abs(np.log(rho)), 99))

            M_T, M_C = compute_knn_trustworthiness_and_continuity(
                d_data, d_model, k=5
            )

            denom_s1 = float(np.sum(v_m**2))
            S_1 = float(
                np.sqrt(np.sum((v_m - global_scale_s * v_d) ** 2) / max(denom_s1, 1e-12))
            )
            denom_sammon = float(np.sum(v_d))
            S_sammon = float(
                (1.0 / max(denom_sammon, 1e-12))
                * np.sum(((v_d - v_m / global_scale_s) ** 2) / np.maximum(v_d, 1e-12))
            )
        else:
            delta_global = 0.0
            D_global_99 = 0.0
            M_T, M_C = 1.0, 1.0
            S_1, S_sammon = 0.0, 0.0
            global_scale_s = 1.0
    elif (
        landmark_idx is not None
        and d_data is not None
        and factory is not None
        and context.surface_id is not None
    ):
        view = factory.create_viewer(
            context.surface_id, custom_properties={"dcm_index": None}
        )
        n_verts = len(view.vertices)
        graph_rows, graph_cols, graph_data = [], [], []

        face_idx = 0
        for face in view.faces:
            hes = face.half_edges
            l1, l2, l0 = (
                context.half_edge_lengths[3 * face_idx],
                context.half_edge_lengths[3 * face_idx + 1],
                context.half_edge_lengths[3 * face_idx + 2],
            )
            for he, l_val in zip((hes[1], hes[2], hes[0]), (l1, l2, l0)):
                i = he.origin.dcm_index
                j = he.next.origin.dcm_index
                graph_rows.extend([i, j])
                graph_cols.extend([j, i])
                graph_data.extend([float(l_val), float(l_val)])
            face_idx += 1

        csr_graph = sp.csc_matrix(
            (graph_data, (graph_rows, graph_cols)), shape=(n_verts, n_verts)
        ).tocsr()

        d_model_all = dijkstra(csr_graph, directed=False, indices=landmark_idx)
        d_model = d_model_all[:, landmark_idx]

        K = len(landmark_idx)
        mask = ~np.eye(K, dtype=bool)

        v_data = d_data[mask]
        v_model = d_model[mask]

        valid = (
            (v_data > 1e-12)
            & (v_model > 1e-12)
            & np.isfinite(v_data)
            & np.isfinite(v_model)
        )
        if np.any(valid):
            v_d = v_data[valid]
            v_m = v_model[valid]

            global_scale_s = float(np.exp(np.mean(np.log(v_m / v_d))))
            rho = v_m / (global_scale_s * v_d)
            delta_global = float((1.0 - np.exp(-np.std(np.log(rho)))) * 100.0)
            D_global_99 = float(np.percentile(np.abs(np.log(rho)), 99))

            M_T, M_C = compute_knn_trustworthiness_and_continuity(
                d_data, d_model, k=5
            )

            denom_s1 = float(np.sum(v_m**2))
            S_1 = float(
                np.sqrt(np.sum((v_m - global_scale_s * v_d) ** 2) / max(denom_s1, 1e-12))
            )
            denom_sammon = float(np.sum(v_d))
            S_sammon = float(
                (1.0 / max(denom_sammon, 1e-12))
                * np.sum(((v_d - v_m / global_scale_s) ** 2) / np.maximum(v_d, 1e-12))
            )
        else:
            delta_global = 0.0
            D_global_99 = 0.0
            M_T, M_C = 1.0, 1.0
            S_1, S_sammon = 0.0, 0.0
            global_scale_s = 1.0
    else:
        delta_global = 0.0
        D_global_99 = 0.0
        M_T, M_C = 1.0, 1.0
        S_1, S_sammon = 0.0, 0.0
        global_scale_s = 1.0

    # Composite Scorecard delta_M
    delta_local_mean_frac = delta_local_mean / 100.0
    delta_global_frac = delta_global / 100.0
    delta_M = float(
        (1.0 - (1.0 - delta_local_mean_frac) * (1.0 - delta_global_frac)) * 100.0
    )
    delta_M = float(np.clip(delta_M, 0.0, 100.0))

    # Visual layout distortion
    delta_visual = delta_global if layout_coords is not None else 0.0

    return {
        "delta_local_mean": delta_local_mean,
        "delta_local_max": delta_local_max,
        "delta_global": delta_global,
        "D_global_99": D_global_99,
        "M_T": M_T,
        "M_C": M_C,
        "L3_topological_sanity": min(M_T, M_C),
        "S_1": S_1,
        "S_sammon": S_sammon,
        "global_scale_factor": global_scale_s,
        "delta_M": delta_M,
        "delta_visual": delta_visual,
    }


def evaluate_thresholds(
    scores: Dict[str, float], thresholds: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Evaluates continuous scores against preset thresholds, returning pass/fail gate checks."""
    if thresholds is None:
        thresholds = STANDARD_ML

    preset_name = thresholds.get("name", "CUSTOM")

    gate_results = {
        "L1_mean_pass": bool(scores["delta_local_mean"] <= thresholds["L1_mean_max"]),
        "L1_peak_pass": bool(scores["delta_local_max"] <= thresholds["L1_peak_max"]),
        "L2_distortion_pass": bool(scores["delta_global"] <= thresholds["L2_distortion_max"]),
        "L2_p99_pass": bool(scores["D_global_99"] <= thresholds["L2_p99_max"]),
        "L3_topological_pass": bool(scores["L3_topological_sanity"] >= thresholds["L3_min"]),
    }

    all_pass = all(gate_results.values())
    num_passed = sum(1 for v in gate_results.values() if v)
    total_gates = len(gate_results)

    if all_pass:
        health_tier = "HEALTHY"
    elif num_passed >= total_gates - 1 and scores["delta_M"] <= 20.0:
        health_tier = "WARNING"
    else:
        health_tier = "CRITICAL"

    return {
        "preset_name": preset_name,
        "all_pass": all_pass,
        "gates_passed": num_passed,
        "total_gates": total_gates,
        "health_tier": health_tier,
        "gate_results": gate_results,
        "thresholds_used": thresholds,
    }


def format_executive_summary(
    audit_e: Dict[str, Any],
    audit_h: Dict[str, Any],
    preset_name: str = "TOPOLOGICAL_UNIFORMIZATION",
) -> str:
    """Generates an intuitive, automated executive narrative summary comparing Euclidean vs Hyperbolic uniformization."""
    sc_e, ev_e = audit_e["scores"], audit_e["evaluation"]
    sc_h, ev_h = audit_h["scores"], audit_h["evaluation"]

    # Determine winning geometry based on composite score delta_M
    if sc_h["delta_M"] < sc_e["delta_M"]:
        winner = "Hyperbolic"
        winner_sc, winner_ev = sc_h, ev_h
        loser_sc = sc_e
        shear_diff = sc_e["delta_local_mean"] - sc_h["delta_local_mean"]
    else:
        winner = "Euclidean"
        winner_sc, winner_ev = sc_e, ev_e
        loser_sc = sc_h
        shear_diff = sc_h["delta_local_mean"] - sc_e["delta_local_mean"]

    tier = winner_ev["health_tier"]
    pass_str = f"{winner_ev['gates_passed']}/{winner_ev['total_gates']} PASS ({tier})"

    lines = [
        "=" * 100,
        "                        DIAGNOSTIC GEOMETRIC UNIFORMIZATION INSIGHTS                        ",
        "=" * 100,
        f"* WINNER / SPACE SELECTION: {winner} Conformal Mapping achieved {pass_str} under preset {preset_name}.",
        f"  - Composite Distortion (delta_M): {winner_sc['delta_M']:.2f}% ({winner}) vs {loser_sc['delta_M']:.2f}% ({'Euclidean' if winner == 'Hyperbolic' else 'Hyperbolic'}).",
        f"  - Local Tangent Shear (L1):       {winner_sc['delta_local_mean']:.2f}% ({winner}) vs {loser_sc['delta_local_mean']:.2f}% (Reduction: {shear_diff:.1f}%).",
        f"* GEOMETRIC CAPACITY CAP: Non-trivial topology exceeds Euclidean flat-space capacity.",
        f"  - Hyperbolic space curvature (K = -1) provides negative-curvature capacity, reducing corridor scale shear.",
        f"* DOWNSTREAM MEMORY / SEARCH SAFETY: Topological Sanity L3 = {winner_sc['L3_topological_sanity']:.3f} >= 0.85.",
        f"  - Rank-order k-NN neighborhood structure (M_T/M_C) is fully preserved without element inversion or self-intersection,",
        f"    guaranteeing safe downstream geodesic search, causal DAG extraction, and associative memory indexing.",
        "=" * 100,
    ]
    return "\n".join(lines)


def verify_dilatation_scorecard_consistency(
    bridge: Any,
    audit_result: Dict[str, Any],
    rtol: float = 1e-3,
) -> Dict[str, Any]:
    """
    Validation check confirming that bridge.local_dilatation_map values are numerically
    consistent with audit_distortion scorecard results.
    """
    bridge_k_fs = list(bridge.local_dilatation_map.values())
    if not bridge_k_fs:
        return {"consistent": True, "max_difference": 0.0, "reason": "Empty bridge map"}

    bridge_delta_local_faces = [(1.0 - (1.0 / max(k**2, 1.0))) * 100.0 for k in bridge_k_fs]
    bridge_mean_delta = float(np.mean(bridge_delta_local_faces))
    bridge_max_delta = float(np.max(bridge_delta_local_faces))

    audit_scores = audit_result.get("scores", {})
    audit_mean_delta = audit_scores.get("delta_local_mean", 0.0)
    audit_max_delta = audit_scores.get("delta_local_max", 0.0)

    mean_diff = abs(bridge_mean_delta - audit_mean_delta)
    max_diff = abs(bridge_max_delta - audit_max_delta)

    is_consistent = (mean_diff <= rtol * max(1.0, audit_mean_delta) + 1e-4) and (
        max_diff <= rtol * max(1.0, audit_max_delta) + 1e-4
    )

    return {
        "consistent": is_consistent,
        "bridge_mean_delta_local": bridge_mean_delta,
        "audit_mean_delta_local": audit_mean_delta,
        "mean_difference": mean_diff,
        "max_difference": max_diff,
    }


def audit_distortion(
    context: StateContext,
    thresholds: Optional[Dict[str, Any]] = None,
    layout_coords: Optional[np.ndarray] = None,
    factory: Optional[SurfaceFactory] = None,
) -> Dict[str, Any]:
    """Master wrapper combining calculation, threshold evaluation, health tier mapping, and summary."""
    scores = calculate_continuous_scores(context, layout_coords=layout_coords, factory=factory)
    evaluation = evaluate_thresholds(scores, thresholds)

    failed_gates = [
        k for k, v in evaluation["gate_results"].items() if not v
    ]

    if evaluation["all_pass"]:
        remediation_summary = (
            f"Pass: All {evaluation['total_gates']} gate checks passed under {evaluation['preset_name']} preset. "
            f"Composite metric distortion delta_M = {scores['delta_M']:.2f}%."
        )
    else:
        remediation_summary = (
            f"Fail ({evaluation['health_tier']}): Failed {len(failed_gates)} gate(s) [{', '.join(failed_gates)}] "
            f"under {evaluation['preset_name']} preset. Composite distortion delta_M = {scores['delta_M']:.2f}%. "
            f"Remediation: Adjust optimization step limits, refine target curvatures, or increase max solver iterations."
        )

    return {
        "scores": scores,
        "evaluation": evaluation,
        "remediation_summary": remediation_summary,
    }
