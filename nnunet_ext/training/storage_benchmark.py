"""Storage benchmark of the replay buffer: log lines and the per-domain bar chart.

Used by nnUNetTrainerFeatureRehearsalSparseDynamicMemory (and so by PRISM) after each domain.
The ``gz_sparse`` values are projections (gzip of the sparse level-0 feature maps at the image's
floating-point precision); they are the numbers reported in the paper.
"""
import os
from os.path import join

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _buffer_label(report: dict) -> str:
    """'b2nd' for buffers stored with Blosc2 (PRISM), 'pkl' for the original pickle format."""
    return 'b2nd' if report.get('storage_format') == 'blosc2' else 'pkl'


def log_storage_summary(report: dict, storage_benchmarks: list, print_to_log_file) -> None:
    """Log actual on-disk + gz_sparse projected sizes (level 0 only) and cumulative totals.

    ``storage_benchmarks`` holds the reports of all domains so far, including ``report``.
    """
    if not report:
        return

    n_sel     = report.get('n_selected', '?')
    n_total   = report.get('n_total',    '?')
    nifti_mb  = (report.get('nifti_stats') or {}).get('total_nifti_MB', None)
    buffer_mb = report.get('actual_pkl_total_MB', None)
    model_mb  = report.get('model_size_MB',       None)

    dtype_str = (report.get('nifti_stats') or {}).get('nifti_dtype', 'float32')
    if 'float16' in str(dtype_str):
        sparse_mb = report.get('projected', {}).get('level_0', {}).get('gz_sparse_f16_MB', None)
    else:
        sparse_mb = report.get('projected', {}).get('level_0', {}).get('gz_sparse_f32_MB', None)

    parts = [f"kept {n_sel}/{n_total} patches"]
    if model_mb  is not None: parts.append(f"model={model_mb:.1f} MB")
    lbl = _buffer_label(report)
    if buffer_mb is not None: parts.append(f"{lbl}={buffer_mb:.1f} MB")
    if report.get('actual_buffer_total_MB') is not None:
        parts.append(f"buffer_on_disk={report['actual_buffer_total_MB']:.2f} MB")
    if sparse_mb is not None: parts.append(f"gz_sparse_{dtype_str}_l0={sparse_mb:.2f} MB")
    if buffer_mb is not None and sparse_mb is not None and sparse_mb > 0:
        parts.append(f"{lbl}→sparse={buffer_mb / sparse_mb:.0f}×")
    if nifti_mb  is not None and sparse_mb is not None and sparse_mb > 0:
        parts.append(f"NIfTI/sparse={nifti_mb / sparse_mb:.1f}×")

    print_to_log_file("Storage  —  " + "  |  ".join(parts))

    # Running cumulative
    cum_nifti  = sum((r.get('nifti_stats') or {}).get('total_nifti_MB', 0.0)
                     for r in storage_benchmarks)
    cum_pkl    = sum(r.get('actual_pkl_total_MB', 0.0) for r in storage_benchmarks)
    cum_model  = sum(r.get('model_size_MB',       0.0) for r in storage_benchmarks)
    cum_sparse = 0.0
    for r in storage_benchmarks:
        dt = (r.get('nifti_stats') or {}).get('nifti_dtype', 'float32')
        if 'float16' in str(dt):
            cum_sparse += r.get('projected', {}).get('level_0', {}).get('gz_sparse_f16_MB', 0.0)
        else:
            cum_sparse += r.get('projected', {}).get('level_0', {}).get('gz_sparse_f32_MB', 0.0)

    n_domains = len(storage_benchmarks)
    if cum_pkl > 0:
        print_to_log_file(
            f"Cumulative ({n_domains} domain(s))  —  "
            f"Model={cum_model:.1f} MB  |  "
            f"NIfTI_gz={cum_nifti:.1f} MB  |  "
            f"Buffer_{lbl}={cum_pkl:.1f} MB  |  "
            f"Buffer_gz_sparse_l0={cum_sparse:.2f} MB  |  "
            f"{lbl}→sparse={cum_pkl / max(cum_sparse, 1e-9):.0f}×  |  "
            f"NIfTI/sparse={cum_nifti / max(cum_sparse, 1e-9):.1f}×"
        )


def save_storage_plot(storage_benchmarks: list, storage_task_names: list, output_dir: str, print_to_log_file) -> None:
    """Save memory-per-domain chart (actual on-disk sizes + level-0 sparse projection + model size) as
    memory_benchmark.png and .pdf in ``output_dir``."""
    if not storage_benchmarks:
        return

    from matplotlib.transforms import blended_transform_factory

    def _get(report, *keys, default=0.0):
        val = report
        for k in keys:
            if not isinstance(val, dict):
                return default
            val = val.get(k, default)
        return val if val is not None else default

    n = len(storage_benchmarks)
    x = np.arange(n, dtype=float)
    xlabels = [
        t.replace("Task", "T").replace("_", "\n")
        for t in storage_task_names
    ]

    nifti_gz   = np.array([_get(r, 'nifti_stats', 'total_nifti_MB') for r in storage_benchmarks])
    buffer_pkl = np.array([r.get('actual_pkl_total_MB', 0.0) for r in storage_benchmarks])

    feat_sparse = []
    for r in storage_benchmarks:
        dtype = (r.get('nifti_stats') or {}).get('nifti_dtype', 'float32')
        if 'float16' in str(dtype):
            v = _get(r, 'projected', 'level_0', 'gz_sparse_f16_MB')
        else:
            v = _get(r, 'projected', 'level_0', 'gz_sparse_f32_MB')
        feat_sparse.append(v)
    feat_sparse = np.array(feat_sparse)

    has_nifti = np.any(nifti_gz > 0)

    model_mb = storage_benchmarks[-1].get('model_size_MB', None)

    nifti_dtype_str = "unknown"
    for r in reversed(storage_benchmarks):
        dt = (r.get('nifti_stats') or {}).get('nifti_dtype')
        if dt:
            nifti_dtype_str = dt
            break

    bar_w     = 0.22
    inner_gap = 0.04
    step      = bar_w + inner_gap
    offsets   = np.array([-1, 0, 1]) * step
    offsets  -= offsets.mean()

    c_nifti  = "#4878cf"
    c_pkl    = "#ffa040"
    c_sparse = "#d05c00"
    c_model  = "#6abf69"

    fig, ax = plt.subplots(figsize=(max(8, n * 2.5 + 3), 5.5))
    fig.suptitle(
        "On-Disk Storage: Feature Buffer vs. NIfTI imagesTr",
        fontsize=11, fontweight="bold",
    )

    def _bars(offset, data, color, label):
        for i, val in enumerate(data):
            ax.bar(i + offset, val, bar_w, color=color,
                   label=label if i == 0 else "_nolegend_",
                   alpha=0.88, edgecolor="white", linewidth=0.4)
            if val > 0:
                ax.text(i + offset, val * 1.08, f"{val:.1f}",
                        ha="center", va="bottom", fontsize=6.5,
                        color=color, rotation=45)

    if has_nifti:
        _bars(offsets[0], nifti_gz,    c_nifti,  "NIfTI compressed (.nii.gz)")
    buf_lbl = _buffer_label(storage_benchmarks[-1])
    _bars(offsets[1], buffer_pkl,  c_pkl,    f"Buffer: actual .{buf_lbl} on disk")
    _bars(offsets[2], feat_sparse, c_sparse,
          f"Buffer: gz_sparse {nifti_dtype_str} level 0 only (replay cost)")

    if model_mb is not None:
        ax.axhline(
            model_mb, color=c_model, linewidth=1.5, linestyle="--",
            label=f"Model weights (params only) = {model_mb:.1f} MB",
        )
        ax.text(
            n - 0.5, model_mb * 1.06, f"{model_mb:.1f} MB",
            color=c_model, fontsize=7.5, va="bottom", ha="right",
        )

    ax.set_yscale("log")
    ax.set_xticks(list(x))
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_xlabel("Domain (task)", fontsize=9)
    ax.set_ylabel("Size (MB, log scale)", fontsize=9)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9, handlelength=1.4)
    ax.grid(axis="y", which="both", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    trans = blended_transform_factory(ax.transData, ax.transAxes)
    cfg = []
    if has_nifti:
        cfg.append((offsets[0], "NIfTI\ngz"))
    cfg += [(offsets[1], f"Buffer\n.{buf_lbl}"), (offsets[2], "Buffer\ngz_sparse\nl0 only")]
    for off, lbl in cfg:
        for xi in x:
            ax.text(xi + off, -0.16, lbl,
                    transform=trans,
                    ha="center", va="top", fontsize=6,
                    color="#555555", style="italic")

    total_nifti  = float(np.sum(nifti_gz))
    total_pkl    = float(np.sum(buffer_pkl))
    total_sparse = float(np.sum(feat_sparse))

    model_line = (
        f"  Model weights (params only): {model_mb:.1f} MB\n"
        if model_mb is not None else ""
    )
    ann = (
        f"Cumulative across {n} domain(s)\n"
        f"{model_line}"
        f"  NIfTI gz total:              {total_nifti:.1f} MB\n"
        f"  Buffer .{buf_lbl} total:{' ' * (14 - len(buf_lbl))}{total_pkl:.1f} MB  ({total_pkl / max(total_nifti, 1e-9):.2f}× NIfTI)\n"
        f"  Buffer gz_sparse l0 total:   {total_sparse:.1f} MB  ({total_sparse / max(total_nifti, 1e-9):.2f}× NIfTI)\n"
        f"  .{buf_lbl} → gz_sparse l0:{' ' * (12 - len(buf_lbl))}{total_pkl / max(total_sparse, 1e-9):.0f}× reduction"
    )
    fig.text(
        0.99, 0.01, ann,
        ha="right", va="bottom", fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f8f8",
                  edgecolor="#cccccc", alpha=0.92),
    )

    plt.tight_layout(rect=[0, 0.15, 1, 1])

    base_path = output_dir
    os.makedirs(base_path, exist_ok=True)

    for ext in ("png", "pdf"):
        fig.savefig(join(base_path, f"memory_benchmark.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)

    print_to_log_file(
        f"Storage plot saved → {join(base_path, 'memory_benchmark.png')}"
    )
