"""Make every figure, plus ``results/summary.csv``, from the CSVs in ``results/`` only."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results"
GIB = 2**30
CHEM_ACC_MHA = 1.6
VRAM_GIB = 8
BACKENDS = ["nvidia", "nvidia-fp64", "qpp-cpu"]
LABELS = {"nvidia": "GPU fp32 (nvidia)", "nvidia-fp64": "GPU fp64 (nvidia-fp64)", "qpp-cpu": "CPU (qpp-cpu)"}
# categorical slots 1-3 of the validated default palette; markers are the redundant encoding
COLORS = {"nvidia": "#257de0", "nvidia-fp64": "#f47230", "qpp-cpu": "#00b894"}
MARKERS = {"nvidia": "o", "nvidia-fp64": "s", "qpp-cpu": "^"}
INK, INK2, GRID = "#191919", "#666666", "#e6e6e6"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight", "font.size": 10,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 5,
})


def read(name: str) -> list[dict[str, str]]:
    with (RES / name).open(newline="") as f:
        return list(csv.DictReader(f))


def ok_points(rows: list[dict], backend: str, circuit: str, col: str) -> tuple[list[int], list[float]]:
    pts = [(int(r["n_qubits"]), float(r[col])) for r in rows
           if r["backend"] == backend and r["circuit"] == circuit and r["status"] == "OK"]
    return [p[0] for p in pts], [p[1] for p in pts]


def plot_scaling_time(rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, circuit, title in zip(axes, ["hwe", "ghz"],
                                  ["Hardware-efficient circuit (depth 10)", "GHZ circuit"]):
        for b in BACKENDS:
            n, t = ok_points(rows, b, circuit, "median_s")
            if n:
                ax.plot(n, t, color=COLORS[b], marker=MARKERS[b], label=LABELS[b])
                ax.annotate(b, (n[-1], t[-1]), xytext=(6, 0), textcoords="offset points",
                            va="center", fontsize=8, color=INK)
            failed = [(int(r["n_qubits"]), r["status"]) for r in rows
                      if r["backend"] == b and r["circuit"] == circuit and r["status"] != "OK"]
            for i, (nf, status) in enumerate(failed):
                ax.plot(nf, 0.96, transform=ax.get_xaxis_transform(), marker="X", ms=8,
                        color=COLORS[b], mec="white", mew=1, clip_on=False, ls="none",
                        label=f"{LABELS[b]}: {status} (X at top)" if i == 0 else None)
        ax.set_yscale("log")
        ax.set_title(title, loc="left")
        ax.set_xlabel("Number of qubits")
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2))
    axes[0].set_ylabel("Median wall time per run (s)")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Statevector simulation time: GPU vs CPU",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(RES / "scaling_time.png")
    plt.close(fig)


def plot_scaling_memory(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    n = np.arange(10, 31)
    ax.plot(n, 2.0**n * 8 / GIB, color=INK2, ls="--", lw=1.2, label="theory fp32: $2^n$ × 8 B")
    ax.plot(n, 2.0**n * 16 / GIB, color=INK2, ls=":", lw=1.2, label="theory fp64: $2^n$ × 16 B")
    ax.axhline(VRAM_GIB, color=INK, lw=1.2)
    ax.annotate("8 GB VRAM", (21, VRAM_GIB), xytext=(0, 4), textcoords="offset points", fontsize=8)
    for b in ("nvidia", "nvidia-fp64"):
        pts = [(n_, m) for n_, m in zip(*ok_points(rows, b, "hwe", "peak_mem_gb")) if m > 0]
        # a reading of 0 means the poller missed a sub-second run; it cannot sit on a log axis
        ax.plot(*zip(*pts), color=COLORS[b], marker=MARKERS[b], label=f"measured peak, {LABELS[b]}")
    ax.set_yscale("log")
    ax.set_xlabel("Number of qubits")
    ax.set_ylabel("Peak GPU memory (GB, 1 GB = $2^{30}$ B)")
    ax.set_title("Peak GPU memory, hardware-efficient circuit", loc="left")
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2))
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(RES / "scaling_memory.png")
    plt.close(fig)


def plot_h2(rows: list[dict]) -> None:
    r = np.array([float(x["R"]) for x in rows])
    col = {k: np.array([float(x[k]) for x in rows]) for k in ("E_HF", "E_FCI", "E_VQE", "abs_err_mHa")}

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(r, col["E_HF"], color=COLORS["nvidia-fp64"], label="Hartree–Fock")
    ax.plot(r, col["E_FCI"], color=INK2, label="FCI (PySCF)")
    ax.plot(r, col["E_VQE"], ls="none", marker="o", ms=7, color=COLORS["nvidia"],
            mec="white", mew=1.2, label="UCCSD-VQE (nvidia-fp64)")
    ax.annotate("Hartree–Fock", (r[-1], col["E_HF"][-1]), xytext=(-4, 6),
                textcoords="offset points", ha="right", fontsize=8)
    ax.set_xlabel("H–H bond length R (Å)")
    ax.set_ylabel("Energy (Ha)")
    ax.set_title("H$_2$ / STO-3G potential energy curve", loc="left")
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(RES / "h2_pes.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(r, col["abs_err_mHa"], color=COLORS["nvidia"], marker="o", label="|E$_{VQE}$ − E$_{FCI}$|")
    ax.axhline(CHEM_ACC_MHA, color=INK, ls="--", lw=1.2)
    ax.annotate(f"chemical accuracy ({CHEM_ACC_MHA} mHa)", (r[0], CHEM_ACC_MHA), xytext=(0, 4),
                textcoords="offset points", fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("H–H bond length R (Å)")
    ax.set_ylabel("|E$_{VQE}$ − E$_{FCI}$| (mHa)")
    ax.set_title("H$_2$ UCCSD-VQE error vs FCI", loc="left")
    fig.savefig(RES / "h2_error.png")
    plt.close(fig)


def summary(scaling: list[dict], h2: list[dict]) -> list[dict]:
    """Key numbers for the README, derived only from the CSVs."""
    out = []
    for circuit in ("hwe", "ghz"):
        for b in BACKENDS:
            n, t = ok_points(scaling, b, circuit, "median_s")
            out += [{"key": f"max_qubits_{circuit}_{b}", "value": max(n) if n else ""},
                    {"key": f"median_s_at_max_{circuit}_{b}", "value": f"{t[-1]:.4g}" if t else ""}]
    for circuit in ("hwe", "ghz"):
        gpu = dict(zip(*ok_points(scaling, "nvidia", circuit, "median_s")))
        cpu = dict(zip(*ok_points(scaling, "qpp-cpu", circuit, "median_s")))
        n = max(set(gpu) & set(cpu))
        out += [{"key": f"speedup_{circuit}_n", "value": n},
                {"key": f"speedup_{circuit}_gpu_fp32_median_s", "value": f"{gpu[n]:.4g}"},
                {"key": f"speedup_{circuit}_cpu_median_s", "value": f"{cpu[n]:.4g}"},
                {"key": f"speedup_{circuit}_x", "value": f"{cpu[n] / gpu[n]:.0f}"}]
    mem29 = dict(zip(*ok_points(scaling, "nvidia", "hwe", "peak_mem_gb")))
    mem28 = dict(zip(*ok_points(scaling, "nvidia-fp64", "hwe", "peak_mem_gb")))
    out += [{"key": "peak_mem_gb_nvidia_max_n", "value": mem29[max(mem29)]},
            {"key": "peak_mem_gb_nvidia-fp64_max_n", "value": mem28[max(mem28)]}]
    check = read("correctness.csv")
    out += [{"key": "correctness_points", "value": len(check)},
            {"key": "correctness_all_agree", "value": all(c["agree"] == "True" for c in check)},
            {"key": "correctness_max_abs_diff", "value": f"{max(float(c['max_abs_diff']) for c in check):.2e}"}]
    err = np.array([float(x["abs_err_mHa"]) for x in h2])
    out += [{"key": "h2_points", "value": len(err)},
            {"key": "h2_max_err_mHa", "value": f"{err.max():.2e}"},
            {"key": "h2_mean_err_mHa", "value": f"{err.mean():.2e}"},
            {"key": "h2_all_below_chem_acc", "value": bool((err < CHEM_ACC_MHA).all())}]
    return out


def main() -> None:
    scaling, h2 = read("scaling.csv"), read("h2_pes.csv")
    plot_scaling_time(scaling)
    plot_scaling_memory(scaling)
    plot_h2(h2)
    rows = summary(scaling, h2)
    with (RES / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["key", "value"])
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['key']:40s} {r['value']}")


if __name__ == "__main__":
    main()
