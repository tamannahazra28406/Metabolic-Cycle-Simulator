"""
visualization.py
=================
Plotting utilities for the metabolic pathway simulator.

Three kinds of figures are produced for every pathway:
  1. Concentration-vs-time curves for every chemical intermediate.
  2. Flux-vs-time curves for every enzyme (how fast each reaction runs).
  3. A schematic pathway diagram: intermediates -> enzymes -> intermediates,
     with dashed arrows showing allosteric regulators (activators in green,
     inhibitors in red) feeding into the enzyme they control.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D

from core import Pathway

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.size": 10,
})

PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
           "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#393b79"]


def plot_concentrations(pathway: Pathway, t, sol, save_path=None, ax=None):
    """Line plot of every metabolite's concentration over time."""
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(9, 5.5))
    names = pathway.metabolite_names()
    for i, name in enumerate(names):
        ax.plot(t, sol[:, i], label=name, color=PALETTE[i % len(PALETTE)],
                linewidth=2)
    ax.set_xlabel("Time (a.u.)")
    ax.set_ylabel("Concentration (a.u.)")
    ax.set_title(f"{pathway.name} — chemical intermediates over time")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8,
               frameon=False)
    ax.grid(alpha=0.25)
    if own_fig:
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig
    return ax


def plot_fluxes(pathway: Pathway, t, fluxes, save_path=None, ax=None):
    """Line plot of every enzyme's instantaneous flux (reaction rate)."""
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, (name, arr) in enumerate(fluxes.items()):
        ax.plot(t, arr, label=name, color=PALETTE[i % len(PALETTE)], linewidth=2)
    ax.set_xlabel("Time (a.u.)")
    ax.set_ylabel("Flux (a.u./time)")
    ax.set_title(f"{pathway.name} — enzymatic flux over time")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8,
               frameon=False)
    ax.grid(alpha=0.25)
    if own_fig:
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig
    return ax


def plot_summary_figure(pathway: Pathway, t, sol, fluxes, save_path=None):
    """Combined 2-panel figure: concentrations on top, fluxes below."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    plot_concentrations(pathway, t, sol, ax=axes[0])
    plot_fluxes(pathway, t, fluxes, ax=axes[1])
    fig.suptitle(pathway.name, fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Schematic pathway diagram: substrates -> enzymes -> products, with
# regulation arrows layered on top.
# ---------------------------------------------------------------------------

def _wrap(text, width=16):
    import textwrap
    return "\n".join(textwrap.wrap(text, width=width))


def plot_pathway_diagram(pathway: Pathway, chain, save_path=None,
                          figsize=None):
    """
    Draw a left-to-right schematic: metabolite box -> enzyme box ->
    metabolite box -> enzyme box -> ...

    `chain` is a list of metabolite names in the order they appear along
    the main pathway (must have len(enzymes)+1 entries, matching
    pathway.enzymes in order). Regulators (from Enzyme.activators /
    Enzyme.inhibitors) are drawn as dashed arrows dropping onto the enzyme
    box that they regulate, colored green (activation) or red (inhibition).
    """
    n_steps = len(pathway.enzymes)
    assert len(chain) == n_steps + 1, \
        "chain must contain one more metabolite than there are enzymes"

    if figsize is None:
        figsize = (max(12.0, 1.9 * n_steps + 2.0), 6.0)

    step_w = 2.6
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-0.65, n_steps * step_w + 0.65)
    ax.set_ylim(-2.75, 2.15)
    ax.axis("off")

    met_box_w, met_box_h = 1.05, 0.75
    enz_box_w, enz_box_h = 1.3, 0.9

    met_x = {}
    for i, met in enumerate(chain):
        x = i * step_w
        met_x[i] = x
        box = FancyBboxPatch((x - met_box_w / 2, -met_box_h / 2),
                              met_box_w, met_box_h,
                              boxstyle="round,pad=0.05,rounding_size=0.08",
                              linewidth=1.4, edgecolor="#1f4e79",
                              facecolor="#cfe2f3")
        ax.add_patch(box)
        fs = 8.5 if len(met) <= 14 else 7.2
        ax.text(x, 0, _wrap(met, 14), ha="center", va="center",
                fontsize=fs, fontweight="bold")

    for i, enz in enumerate(pathway.enzymes):
        x0, x1 = met_x[i], met_x[i + 1]
        xe = (x0 + x1) / 2
        ye = 1.4
        # main reaction arrow (substrate -> product), passing behind/under
        arrow = FancyArrowPatch((x0 + met_box_w / 2, 0), (x1 - met_box_w / 2, 0),
                                 arrowstyle="-|>", mutation_scale=16,
                                 linewidth=1.6, color="#444444", zorder=1)
        ax.add_patch(arrow)

        # enzyme box sitting above the arrow, connected by a short stem
        ebox = FancyBboxPatch((xe - enz_box_w / 2, ye - enz_box_h / 2),
                               enz_box_w, enz_box_h,
                               boxstyle="round,pad=0.05,rounding_size=0.08",
                               linewidth=1.4, edgecolor="#7f6000",
                               facecolor="#fff2cc")
        ax.add_patch(ebox)
        label = enz.name
        ax.text(xe, ye, _wrap(label, 16), ha="center", va="center",
                fontsize=7.6, fontweight="bold")
        ax.plot([xe, xe], [ye - enz_box_h / 2, 0.08], color="#7f6000",
                linewidth=1.0, linestyle=":")

        # process label under the main arrow
        if enz.process:
            ax.text(xe, -0.45, _wrap(enz.process, 18), ha="center", va="top",
                    fontsize=6.8, color="#555555", style="italic")

        # regulation arrows dropping from below onto the enzyme box
        n_reg = len(enz.activators) + len(enz.inhibitors)
        if n_reg:
            half_width = min(1.05, 0.34 * n_reg)
            spread = (np.linspace(-half_width, half_width, n_reg)
                      if n_reg > 1 else [0.0])
            idx = 0
            for reg in enz.activators:
                rx = xe + spread[idx]; idx += 1
                ax.annotate(
                    "", xy=(rx, ye - enz_box_h / 2 - 0.02),
                    xytext=(rx, -1.9),
                    arrowprops=dict(arrowstyle="-|>", color="#2ca02c",
                                     linestyle="--", linewidth=1.4))
                ax.text(rx, -2.1, _wrap("+ " + reg, 12), ha="center",
                        va="top", fontsize=6.6, color="#2ca02c", fontweight="bold")
            for reg in enz.inhibitors:
                rx = xe + spread[idx]; idx += 1
                ax.annotate(
                    "", xy=(rx, ye - enz_box_h / 2 - 0.02),
                    xytext=(rx, -1.9),
                    arrowprops=dict(arrowstyle="-|>", color="#d62728",
                                     linestyle="--", linewidth=1.4))
                ax.text(rx, -2.1, _wrap("\u2013 " + reg, 12), ha="center",
                        va="top", fontsize=6.6, color="#d62728", fontweight="bold")

    legend_elems = [
        Line2D([0], [0], color="#444444", lw=1.6, label="Reaction flow"),
        Line2D([0], [0], color="#2ca02c", lw=1.4, ls="--", label="Allosteric activation"),
        Line2D([0], [0], color="#d62728", lw=1.4, ls="--", label="Allosteric inhibition"),
    ]
    ax.legend(handles=legend_elems, loc="upper center",
              bbox_to_anchor=(0.5, 1.16), ncol=3, frameon=False, fontsize=9)
    ax.set_title(pathway.name, fontsize=14, fontweight="bold", y=1.05)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
