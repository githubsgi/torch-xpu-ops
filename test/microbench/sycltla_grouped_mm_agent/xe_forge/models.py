"""Stages, issue taxonomy and data models.

Mapping from Xe-Forge's original stages (src/xe_forge/models.py, planner.py):

  original              here                 what it means for the grouped GEMM
  --------------------  -------------------  ------------------------------------------------------
  ANALYSIS              ANALYSIS             LLM + profile digests -> DetectedIssue list (JSON)
  ALGORITHMIC           ALGORITHMIC          work decomposition / scheduling: tile linearization over
                                             groups, tail balancing, split-K / stream-K style
                                             restructuring of the K-grouped kernel (ABI-limited)
  DISCOVERY             DISCOVERY            open_ended issues (novel ideas not covered below)
  DTYPE_FIX             (dropped)            fp32 accumulation + bf16 output is the fixed contract
  FUSION                (dropped)            no epilogue to fuse; Params / ABI are fixed
  MEMORY_ACCESS         MEMORY_ACCESS        prefetch distance, K tile, 2D block copy atoms, VNNI /
                                             transposed loads, redundant reorders
  TENSOR_DESCRIPTORS    (merged into         the CuTe 2D block descriptors are already used; their
                         MEMORY_ACCESS)      selection is part of MEMORY_ACCESS
  PERSISTENT_KERNEL     PERSISTENT_KERNEL    grid size (wgs), raster / order for L3 reuse, last wave
  DEVICE_SPECIFIC       DEVICE_SPECIFIC      sub-group layout, per-SG DPAS tile, GRF 128 vs 256,
                                             barrier scope / placement
  AUTOTUNING            AUTOTUNING           configs.json sweeps per kernel kind, evaluated in parallel
"""

import json
from dataclasses import asdict, dataclass, field
from enum import Enum


class Stage(str, Enum):
    ANALYSIS = "analysis"
    ALGORITHMIC = "algorithmic"
    DISCOVERY = "discovery"
    MEMORY_ACCESS = "memory_access"
    PERSISTENT_KERNEL = "persistent_kernel"
    DEVICE_SPECIFIC = "device_specific"
    AUTOTUNING = "autotuning"


OPTIMIZATION_STAGES = [s for s in Stage if s != Stage.ANALYSIS]

# Planner fallback order (original: algorithmic, discovery, dtype, fusion, memory, tensor_desc,
# persistent, device_specific, autotuning).
DEFAULT_STAGE_ORDER = [Stage.ALGORITHMIC, Stage.DISCOVERY, Stage.MEMORY_ACCESS, Stage.PERSISTENT_KERNEL,
                       Stage.DEVICE_SPECIFIC, Stage.AUTOTUNING]

# (before, after): before must precede after when both are planned. Enforced regardless of the LLM.
HARD_DEPENDENCIES = [
    (Stage.ALGORITHMIC, Stage.MEMORY_ACCESS),       # work decomposition decides what each tile loads
    (Stage.DISCOVERY, Stage.MEMORY_ACCESS),
    (Stage.ALGORITHMIC, Stage.PERSISTENT_KERNEL),   # grid / raster tuning presumes the final tile space
    (Stage.MEMORY_ACCESS, Stage.PERSISTENT_KERNEL),
    (Stage.MEMORY_ACCESS, Stage.DEVICE_SPECIFIC),   # prefetch / copy atoms before sg layout / GRF / barriers
    (Stage.PERSISTENT_KERNEL, Stage.DEVICE_SPECIFIC),
    (Stage.DEVICE_SPECIFIC, Stage.AUTOTUNING),      # sweep configs last, on the final kernel structure
]

KINDS = ("m_rowB", "m_colB", "k_grouped")
KIND_DOC = {"m_rowB": "forward (2D x 3D, B K-contiguous 'R')",
            "m_colB": "grad_input (2D x 3D, B N-contiguous 'C')",
            "k_grouped": "grad_weight (2D x 2D, K-grouped)"}
OPS = ("w13_fwd", "w2_fwd", "w13_dX", "w2_dX", "w13_dW", "w2_dW")
KIND_OPS = {"m_rowB": ("w13_fwd", "w2_fwd"), "m_colB": ("w13_dX", "w2_dX"), "k_grouped": ("w13_dW", "w2_dW")}

# issue_type -> stage. The enum below is the only vocabulary the analysis may use.
ISSUE_STAGE = {
    # ALGORITHMIC: work decomposition and scheduling
    "short_k_mainloop": Stage.ALGORITHMIC,          # K-grouped: ~6 K tiles per expert, prologue/epilogue dominate
    "split_k_missing": Stage.ALGORITHMIC,           # K-grouped output tiles could be split over K / merged
    "tile_padding_waste": Stage.ALGORITHMIC,        # last M tile of a small group mostly padding
    "scheduler_overhead": Stage.ALGORITHMIC,        # per-tile locate()/describe(), 64-bit div/mod, offs reads
    "tile_order_group_unaware": Stage.ALGORITHMIC,  # linearization (m fastest) vs B/A reuse across groups
    "suboptimal_algorithm": Stage.ALGORITHMIC,
    # MEMORY_ACCESS
    "prefetch_distance": Stage.MEMORY_ACCESS,
    "load_latency_stall": Stage.MEMORY_ACCESS,      # SBID stalls on own 2D block loads
    "k_tile_size": Stage.MEMORY_ACCESS,             # kTileK 32 vs 64 / 16
    "transposed_or_vnni_load": Stage.MEMORY_ACCESS, # strided / transposed A or B copy atom choice
    "redundant_reorder": Stage.MEMORY_ACCESS,       # reorder(tArA,tCrA) copies in the k loop
    "copy_atom_selection": Stage.MEMORY_ACCESS,     # explicit XE_LOAD_2D shape vs auto
    "epilogue_store": Stage.MEMORY_ACCESS,          # D reorder + 2D store
    # PERSISTENT_KERNEL
    "grid_size": Stage.PERSISTENT_KERNEL,           # wgs (56) vs tiles
    "last_wave_underfill": Stage.PERSISTENT_KERNEL, # partial last wave, idle XVEs
    "raster_order_l3_reuse": Stage.PERSISTENT_KERNEL,
    "work_distribution": Stage.PERSISTENT_KERNEL,   # round-robin lin += nwg vs chunked / stealing
    # DEVICE_SPECIFIC
    "subgroup_layout": Stage.DEVICE_SPECIFIC,
    "per_sg_dpas_tile": Stage.DEVICE_SPECIFIC,
    "grf_mode": Stage.DEVICE_SPECIFIC,
    "register_spills": Stage.DEVICE_SPECIFIC,
    "barrier_scope_placement": Stage.DEVICE_SPECIFIC,
    "low_xmx_utilization": Stage.DEVICE_SPECIFIC,
    # AUTOTUNING
    "tile_config_per_kind": Stage.AUTOTUNING,
    "missing_per_kind_tuning": Stage.AUTOTUNING,
    # DISCOVERY
    "open_ended": Stage.DISCOVERY,
}
ISSUE_TYPES = tuple(ISSUE_STAGE)

ISSUE_DOC = {
    "short_k_mainloop": "K-grouped kernel: each expert contributes few K tiles (~6 at 190 rows), prologue (prefetch fill) and epilogue (256x256 store) dominate",
    "split_k_missing": "K-grouped output tiles processed by one WG over all K of a group; no split-K or tile-level parallelism over K",
    "tile_padding_waste": "rows of a small group fill the last 256-row M tile only partially (zero-filled loads, masked stores)",
    "scheduler_overhead": "per-tile scheduler cost: linear tile index -> (group, m, n), 64-bit div/mod, offs reads, descriptor rebuilds",
    "tile_order_group_unaware": "tile linearization order (m fastest within a group) is poor for operand reuse in L3/L1",
    "suboptimal_algorithm": "any other work-decomposition problem",
    "prefetch_distance": "kPrefetchDist too short or too long for the kind / K length",
    "load_latency_stall": "SBID (scoreboard) stalls waiting on own 2D block loads",
    "k_tile_size": "K tile (kTileK) too small/large: loads in flight vs registers",
    "transposed_or_vnni_load": "A/B loaded with transposed or non-VNNI path, extra reorder cost",
    "redundant_reorder": "reorder() register shuffles in the mainloop between copy and MMA fragments",
    "copy_atom_selection": "automatic copy atom choice (get_block_2d_copy_*<void>) is not the best 2D block shape",
    "epilogue_store": "accumulator -> bf16 conversion and 2D block store cost",
    "grid_size": "number of persistent work-groups (wgs) not matched to tiles/Xe cores",
    "last_wave_underfill": "last wave of tiles leaves XVEs idle (tail effect over groups)",
    "raster_order_l3_reuse": "tile visiting order (rasterization) wastes L3/L1 reuse of A or B",
    "work_distribution": "static round-robin distribution of tiles over work-groups is imbalanced",
    "subgroup_layout": "sub-group arrangement (sgM x sgN) over the WG tile",
    "per_sg_dpas_tile": "per-sub-group accumulator tile (e.g. 32x64) vs register budget / reuse",
    "grf_mode": "GRF mode 128 vs 256 vs threads per XVE",
    "register_spills": "spills reported in the build/properties",
    "barrier_scope_placement": "barrier_arrive/barrier_wait scope or placement costs sync stalls",
    "low_xmx_utilization": "XMX (DPAS) pipe starved although XVEs are active",
    "tile_config_per_kind": "tile / sg / prefetch config of a kind should differ (configs.json sweep)",
    "missing_per_kind_tuning": "a kind still uses the production config without having been swept",
    "open_ended": "novel idea not covered by any type above; must carry open_ended_proposal",
}

ISSUE_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "description": "3-6 sentences: where the time goes per kind and regime"},
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "issue_type": {"type": "string", "enum": list(ISSUE_TYPES)},
                    "severity": {"type": "integer", "minimum": 1, "maximum": 5},
                    "kinds": {"type": "array", "items": {"type": "string", "enum": list(KINDS)}},
                    "ops": {"type": "array", "items": {"type": "string", "enum": list(OPS)}},
                    "evidence": {"type": "string", "description": "cite measured metrics from the digests"},
                    "suggested_fix": {"type": "string"},
                    "estimated_speedup": {"type": "string"},
                    "open_ended_proposal": {"type": "string"},
                },
                "required": ["issue_type", "severity", "kinds", "evidence", "suggested_fix", "estimated_speedup"],
            },
        },
    },
    "required": ["summary", "issues"],
}


@dataclass
class DetectedIssue:
    issue_type: str
    severity: int
    kinds: list = field(default_factory=list)
    ops: list = field(default_factory=list)
    evidence: str = ""
    suggested_fix: str = ""
    estimated_speedup: str = ""
    open_ended_proposal: str = ""

    @property
    def stage(self):
        return ISSUE_STAGE[self.issue_type]

    def to_dict(self):
        return asdict(self)

    def render(self):
        s = (f"- [{self.severity}/5] {self.issue_type} (kinds {','.join(self.kinds) or '-'}; "
             f"ops {','.join(self.ops) or '-'}): {self.evidence}\n  Fix: {self.suggested_fix}\n"
             f"  Estimated speedup (unmeasured guess): {self.estimated_speedup or 'unknown'}")
        if self.open_ended_proposal:
            s += f"\n  Proposal: {self.open_ended_proposal}"
        return s


def coerce_issue(raw):
    """Validate one issue dict from the LLM. Unknown types are dropped (original: _coerce_issue)."""
    if not isinstance(raw, dict):
        return None
    t = str(raw.get("issue_type", "")).strip().lower()
    if t not in ISSUE_STAGE:
        return None
    try:
        sev = max(1, min(5, int(raw.get("severity", 3))))
    except (TypeError, ValueError):
        sev = 3
    kinds = [k for k in raw.get("kinds", []) if k in KINDS]
    ops = [o for o in raw.get("ops", []) if o in OPS]
    return DetectedIssue(t, sev, kinds, ops, str(raw.get("evidence", ""))[:1500],
                         str(raw.get("suggested_fix", ""))[:1500], str(raw.get("estimated_speedup", ""))[:200],
                         str(raw.get("open_ended_proposal", ""))[:2500])


def stages_needed(issues):
    out = {}
    for i in issues:
        out.setdefault(i.stage, []).append(i.issue_type)
    return out


def dumps(obj):
    return json.dumps(obj, indent=1, default=str)
