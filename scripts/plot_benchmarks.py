"""Renders the README figures from docs/benchmarks/*.jsonl into docs/media/ (light and dark variants).

usage: python scripts/plot_benchmarks.py   (needs matplotlib)
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "docs" / "benchmarks"
OUT = ROOT / "docs" / "media"

THEMES = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a"],
        # renderer identity (quad / tile), kept apart from the stage colors above
        "renderer": ["#6250d6", "#e87ba4"],
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70"],
        "renderer": ["#9085e9", "#d55181"],
    },
}


def load(name):
    with open(DATA / name) as f:
        return {r["label"]: r for r in map(json.loads, f)}


def style(ax, t, grid_axis="y"):
    ax.set_facecolor(t["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["axis"])
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=t["muted"], labelcolor=t["ink2"], labelsize=9, length=0, pad=6)
    ax.grid(axis=grid_axis, color=t["grid"], linewidth=1)
    ax.set_axisbelow(True)


def title(fig, t, text, sub):
    fig.text(0.012, 0.965, text, color=t["ink"], fontsize=13, fontweight="semibold", va="top")
    fig.text(0.012, 0.885, sub, color=t["ink2"], fontsize=9.5, va="top")


def save(fig, name, mode):
    fig.savefig(OUT / f"{name}-{mode}.png", dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


# -------------------------------------------------------------------------------------------------
# 1. per-stage GPU time of every renderer on the default bicycle view
# -------------------------------------------------------------------------------------------------
def stage_breakdown(mode):
    t = THEMES[mode]
    s = load("bicycle_sweep.jsonl")
    rows = [
        ("Point cloud", 0, 0, s["pointcloud"]["render"]),
        ("Quad, reference 3σ bounds", s["quad|tight_off"]["preprocess"], s["quad|tight_off"]["sort"], s["quad|tight_off"]["render"]),
        ("Quad, tight bounds (default)", s["quad|f32|wg256"]["preprocess"], s["quad|f32|wg256"]["sort"], s["quad|f32|wg256"]["render"]),
        ("Tile, reference 3σ bounds", s["tile|tight_off"]["preprocess"], s["tile|tight_off"]["sort"], s["tile|tight_off"]["rasterize"]),
        ("Tile, tight bounds", s["tile|f32|wg256"]["preprocess"], s["tile|f32|wg256"]["sort"], s["tile|f32|wg256"]["rasterize"]),
    ][::-1]
    fig, ax = plt.subplots(figsize=(8.4, 3.9), facecolor=t["surface"])
    fig.subplots_adjust(left=0.25, right=0.93, top=0.74, bottom=0.12)
    style(ax, t, grid_axis="x")
    names = ["Preprocess", "Sort", "Render / rasterize"]
    y = range(len(rows))
    left = [0.0] * len(rows)
    for k in range(3):
        vals = [r[k + 1] for r in rows]
        ax.barh(y, vals, left=left, height=0.5, color=t["series"][k], edgecolor=t["surface"], linewidth=2, label=names[k])
        left = [a + b for a, b in zip(left, vals)]
    for i, total in enumerate(left):
        ax.text(total + 0.25, i, f"{total:.1f} ms", va="center", color=t["ink"], fontsize=9.5)
    ax.set_yticks(list(y), [r[0] for r in rows])
    ax.tick_params(axis="y", labelcolor=t["ink"], labelsize=9.5)
    ax.set_xlim(0, max(left) * 1.13)
    ax.set_xlabel("GPU time per frame (ms)", color=t["ink2"], fontsize=9)
    title(fig, t, "Where the frame time goes", "bicycle (1.06M gaussians), default camera, 1280×720, RTX 4050 Laptop")
    leg = ax.legend(loc="lower left", bbox_to_anchor=(-0.005, 1.0), ncol=3, frameon=False, fontsize=9, handlelength=1.0, columnspacing=1.4)
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])
    save(fig, "stage-breakdown", mode)


# -------------------------------------------------------------------------------------------------
# 2. scaling with the number of gaussians (random subsets of bicycle)
# -------------------------------------------------------------------------------------------------
def gaussian_scaling(mode):
    t = THEMES[mode]
    c = load("gaussian_count.jsonl")
    counts = {"25": 265772, "50": 531545, "75": 797318, "100": 1063091}
    xs = [counts[k] / 1e6 for k in counts]
    quad = [c[f"bicycle_{k}|quad|f32|wg256"] for k in counts]
    tile = [c[f"bicycle_{k}|tile|f32|wg256"] for k in counts]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), facecolor=t["surface"])
    fig.subplots_adjust(left=0.07, right=0.88, top=0.72, bottom=0.15, wspace=0.42)

    ax = axes[0]
    style(ax, t)
    for k, (series, name) in enumerate([(quad, "Quad"), (tile, "Tile")]):
        ys = [r["gpu_total"] for r in series]
        ax.plot(xs, ys, color=t["renderer"][k], linewidth=2, solid_capstyle="round", label=name)
        ax.scatter(xs, ys, s=36, color=t["renderer"][k], edgecolor=t["surface"], linewidth=2, zorder=3)
        ax.text(xs[-1] + 0.04, ys[-1], f"{name} {ys[-1]:.1f} ms", color=t["ink"], fontsize=9, va="center")
    ax.set_ylim(0, 12)
    ax.set_xlim(0.2, 1.1)
    ax.set_title("Total GPU time", color=t["ink"], fontsize=10, loc="left", pad=8)

    ax = axes[1]
    style(ax, t)
    for k, (key, name) in enumerate([("preprocess", "Preprocess"), ("sort", "Sort"), ("render", "Render")]):
        ys = [r[key] for r in quad]
        ax.plot(xs, ys, color=t["series"][k], linewidth=2, solid_capstyle="round", label=name)
        ax.scatter(xs, ys, s=36, color=t["series"][k], edgecolor=t["surface"], linewidth=2, zorder=3)
        ax.text(xs[-1] + 0.04, ys[-1], name, color=t["ink"], fontsize=9, va="center")
    ax.set_ylim(0, 3.5)
    ax.set_xlim(0.2, 1.1)
    ax.set_title("Quad renderer by stage", color=t["ink"], fontsize=10, loc="left", pad=8)

    for ax in axes:
        ax.set_xlabel("Gaussians in the scene (millions)", color=t["ink2"], fontsize=9)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g} ms"))
    title(fig, t, "Cost grows with the number of gaussians", "random 25 / 50 / 75 / 100% subsets of bicycle, default camera, 1280×720")
    save(fig, "gaussian-scaling", mode)


# -------------------------------------------------------------------------------------------------
# 3. frustum culling while dollying into the scene
# -------------------------------------------------------------------------------------------------
def culling(mode):
    t = THEMES[mode]
    c = load("bicycle_f16_and_culling.jsonl")
    steps = range(9)
    xs = [0.5 * i for i in steps]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), facecolor=t["surface"])
    fig.subplots_adjust(left=0.08, right=0.87, top=0.72, bottom=0.15, wspace=0.45)
    panels = [
        ("visible", "Gaussians sent to sort + draw", lambda v, _: f"{v / 1e6:g}M", (0, 1.1e6)),
        ("gpu_total", "Quad renderer GPU time", lambda v, _: f"{v:g} ms", (0, 11)),
    ]
    for ax, (key, name, fmt, ylim) in zip(axes, panels):
        style(ax, t)
        for k, (cull, label) in enumerate([("cull_on", "Culling on"), ("cull_off", "Culling off")]):
            ys = [c[f"quad|zoom{i}|{cull}"][key] for i in steps]
            ax.plot(xs, ys, color=t["series"][k], linewidth=2, solid_capstyle="round", label=label)
            ax.scatter(xs, ys, s=36, color=t["series"][k], edgecolor=t["surface"], linewidth=2, zorder=3)
            ax.text(xs[-1] + 0.12, ys[-1], label, color=t["ink"], fontsize=9, va="center")
        ax.set_title(name, color=t["ink"], fontsize=10, loc="left", pad=8)
        ax.set_xlabel("Camera moved towards the bike (units)", color=t["ink2"], fontsize=9)
        ax.yaxis.set_major_formatter(FuncFormatter(fmt))
        ax.set_ylim(*ylim)
        ax.set_xlim(-0.1, 4.1)
    title(fig, t, "Frustum culling pays off once the scene leaves the screen",
          "bicycle, default camera dollied forward in 0.5 unit steps, quad renderer, 1280×720")
    save(fig, "culling", mode)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for m in THEMES:
        stage_breakdown(m)
        gaussian_scaling(m)
        culling(m)
    print("figures written to", OUT)
