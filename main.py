"""
main.py
=======
Metabolic Pathway Simulator — command-line entry point.

Builds ODE models (scipy.integrate.odeint) for five core human metabolic
cycles, prints a human-readable showcase of each pathway's chemical
intermediates / enzymes / regulation, integrates the ODEs, and saves:

  * <pathway>_concentrations.png  - every intermediate's level over time
  * <pathway>_fluxes.png          - every enzyme's reaction rate over time
  * <pathway>_diagram.png         - schematic: intermediates -> enzymes,
                                     with regulation arrows
  * <pathway>_summary.png         - combined concentrations+fluxes panel

...plus two "regulation in action" comparison figures showing the SAME
pathway under two different hormonal/metabolic states (e.g. fed vs.
fasting), so the effect of regulation is directly visible.

Usage
-----
    python main.py                       # run everything, save all figures
    python main.py --pathway glycolysis  # run just one pathway
    python main.py --outdir myplots      # change output directory
    python main.py --list                # list available pathways
"""

import os
import argparse
import matplotlib.pyplot as plt

from pathways import (
    build_glycolysis, build_gluconeogenesis, build_glycogenesis,
    build_glycogenolysis, build_beta_oxidation, build_tca_cycle,
    build_pentose_phosphate, build_ketogenesis, build_urea_cycle,
    PATHWAY_BUILDERS,
)
from visualization import (
    plot_concentrations, plot_fluxes, plot_summary_figure, plot_pathway_diagram,
)

# ---------------------------------------------------------------------------
# Per-pathway settings: simulation horizon + the metabolite "chain" used to
# draw the schematic diagram (must have len(enzymes)+1 entries).
# ---------------------------------------------------------------------------

PATHWAY_CONFIG = {
    "glycolysis": dict(
        builder=build_glycolysis, t_span=(0, 150),
        chain=["Glucose", "G6P", "F6P", "F16BP", "TP", "BPG13", "PG",
               "PEP", "Pyruvate"],
    ),
    "gluconeogenesis": dict(
        builder=build_gluconeogenesis, t_span=(0, 60),
        chain=["Pyruvate", "PEP", "F16BP", "F6P", "G6P", "Glucose"],
    ),
    "glycogenesis": dict(
        builder=build_glycogenesis, t_span=(0, 150),
        chain=["G6P", "G1P", "UDPglucose", "Glycogen"],
    ),
    "glycogenolysis": dict(
        builder=build_glycogenolysis, t_span=(0, 40),
        chain=["Glycogen", "G1P", "G6P", "Glucose"],
    ),
    "beta_oxidation": dict(
        builder=build_beta_oxidation, t_span=(0, 60),
        chain=["FA_cytosol", "FA_mito", "trans-Enoyl-CoA", "3-OH-Acyl-CoA",
               "3-Keto-Acyl-CoA", "AcetylCoA + shortened acyl-CoA"],
    ),
    "tca_cycle": dict(
        builder=build_tca_cycle, t_span=(0, 60),
        chain=["OAA", "Citrate", "Isocitrate", "AlphaKG", "SuccinylCoA",
               "Succinate", "Fumarate", "OAA (regenerated)"],
    ),
    "pentose_phosphate": dict(
        builder=build_pentose_phosphate, t_span=(0, 60),
        chain=["G6P", "6-Phosphogluconate", "Ribulose-5-P", "Ribose-5-P"],
    ),
    "ketogenesis": dict(
        builder=build_ketogenesis, t_span=(0, 60),
        chain=["AcetylCoA (x2)", "AcetoacetylCoA", "HMGCoA", "Acetoacetate", "BHB"],
    ),
    "urea_cycle": dict(
        builder=build_urea_cycle, t_span=(0, 60),
        chain=["Ammonia+CO2", "CarbamoylP", "Citrulline", "Argininosuccinate",
               "Arginine", "Ornithine + Urea"],
    ),
}


def run_pathway(key: str, outdir: str, show_summary: bool = True):
    cfg = PATHWAY_CONFIG[key]
    pathway = cfg["builder"]()
    if show_summary:
        print("\n" + pathway.summary() + "\n")

    t, sol, fluxes = pathway.simulate(t_span=cfg["t_span"], n_points=500)

    base = os.path.join(outdir, key)
    fig1 = plot_concentrations(pathway, t, sol, save_path=f"{base}_concentrations.png")
    plt.close(fig1)
    fig2 = plot_fluxes(pathway, t, fluxes, save_path=f"{base}_fluxes.png")
    plt.close(fig2)
    fig3 = plot_summary_figure(pathway, t, sol, fluxes, save_path=f"{base}_summary.png")
    plt.close(fig3)
    fig4 = plot_pathway_diagram(pathway, cfg["chain"], save_path=f"{base}_diagram.png")
    plt.close(fig4)

    print(f"[{key}] figures saved: "
          f"{base}_concentrations.png, {base}_fluxes.png, "
          f"{base}_summary.png, {base}_diagram.png")
    return pathway, t, sol, fluxes


def run_regulation_contrast(outdir: str):
    """
    Demonstrate regulation by running the SAME pathway under two different
    hormonal/metabolic states and overlaying key trajectories.
    """
    import numpy as np

    # --- Glycolysis: high insulin drive (fed) vs low (fasting) ------------
    pw_fed = build_glycolysis(hormonal_drive=1.8)
    pw_fast = build_glycolysis(hormonal_drive=0.2)
    t1, sol_fed, flux_fed = pw_fed.simulate(t_span=(0, 150), n_points=400)
    t2, sol_fast, flux_fast = pw_fast.simulate(t_span=(0, 150), n_points=400)
    idx_pyr = pw_fed.metabolite_names().index("Pyruvate")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(t1, flux_fed["PFK1"], label="High insulin (fed)", color="#2ca02c")
    axes[0].plot(t2, flux_fast["PFK1"], label="Low insulin (fasting)", color="#d62728")
    axes[0].set_title("PFK-1 flux: insulin activates the committed step")
    axes[0].set_xlabel("Time (a.u.)"); axes[0].set_ylabel("Flux (a.u./time)")
    axes[0].legend(frameon=False); axes[0].grid(alpha=0.25)

    axes[1].plot(t1, sol_fed[:, idx_pyr], label="High insulin (fed)", color="#2ca02c")
    axes[1].plot(t2, sol_fast[:, idx_pyr], label="Low insulin (fasting)", color="#d62728")
    axes[1].set_title("Downstream effect: pyruvate output")
    axes[1].set_xlabel("Time (a.u.)"); axes[1].set_ylabel("Concentration (a.u.)")
    axes[1].legend(frameon=False); axes[1].grid(alpha=0.25)
    fig.suptitle("Regulation in action — Glycolysis under fed vs. fasting hormonal state",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(outdir, "regulation_contrast_glycolysis.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[regulation demo] saved {path}")

    # --- Glycogenolysis: hormonal surge (fasting/exercise) vs rest --------
    pw_rest = build_glycogenolysis(hormonal_signal=0.15)
    pw_surge = build_glycogenolysis(hormonal_signal=1.8)
    t3, sol_rest, flux_rest = pw_rest.simulate(t_span=(0, 40), n_points=400)
    t4, sol_surge, flux_surge = pw_surge.simulate(t_span=(0, 40), n_points=400)
    idx_gly = pw_rest.metabolite_names().index("Glycogen")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(t3, flux_rest["Glycogen_phosphorylase"], label="Resting (low glucagon/epinephrine)",
                 color="#1f77b4")
    axes[0].plot(t4, flux_surge["Glycogen_phosphorylase"], label="Fasting/exercise (hormone surge)",
                 color="#ff7f0e")
    axes[0].set_title("Glycogen phosphorylase flux vs. hormonal signal")
    axes[0].set_xlabel("Time (a.u.)"); axes[0].set_ylabel("Flux (a.u./time)")
    axes[0].legend(frameon=False); axes[0].grid(alpha=0.25)

    axes[1].plot(t3, sol_rest[:, idx_gly], label="Resting", color="#1f77b4")
    axes[1].plot(t4, sol_surge[:, idx_gly], label="Fasting/exercise", color="#ff7f0e")
    axes[1].set_title("Glycogen store depletion")
    axes[1].set_xlabel("Time (a.u.)"); axes[1].set_ylabel("Glycogen (a.u.)")
    axes[1].legend(frameon=False); axes[1].grid(alpha=0.25)
    fig.suptitle("Regulation in action — Glycogenolysis under hormonal surge vs. rest",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(outdir, "regulation_contrast_glycogenolysis.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[regulation demo] saved {path}")

    # --- Beta-oxidation: fed (high malonyl-CoA, CPT1 blocked) vs fasting --
    pw_fedfat = build_beta_oxidation(fed_state=1.0)
    pw_fastfat = build_beta_oxidation(fed_state=0.0)
    t5, sol_fedfat, flux_fedfat = pw_fedfat.simulate(t_span=(0, 60), n_points=400)
    t6, sol_fastfat, flux_fastfat = pw_fastfat.simulate(t_span=(0, 60), n_points=400)
    idx_acoa = pw_fedfat.metabolite_names().index("AcetylCoA")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(t5, flux_fedfat["CPT1"], label="Fed (high malonyl-CoA)", color="#9467bd")
    axes[0].plot(t6, flux_fastfat["CPT1"], label="Fasting (low malonyl-CoA)", color="#17becf")
    axes[0].set_title("CPT-1 flux: malonyl-CoA gatekeeps fat entry to mitochondria")
    axes[0].set_xlabel("Time (a.u.)"); axes[0].set_ylabel("Flux (a.u./time)")
    axes[0].legend(frameon=False); axes[0].grid(alpha=0.25)

    axes[1].plot(t5, sol_fedfat[:, idx_acoa], label="Fed", color="#9467bd")
    axes[1].plot(t6, sol_fastfat[:, idx_acoa], label="Fasting", color="#17becf")
    axes[1].set_title("Downstream effect: acetyl-CoA output")
    axes[1].set_xlabel("Time (a.u.)"); axes[1].set_ylabel("Concentration (a.u.)")
    axes[1].legend(frameon=False); axes[1].grid(alpha=0.25)
    fig.suptitle("Regulation in action — Beta-oxidation gated by malonyl-CoA (fed vs. fasting)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(outdir, "regulation_contrast_beta_oxidation.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[regulation demo] saved {path}")

    # --- TCA cycle: high energy demand (exercise) vs resting ---------------
    from pathways import build_tca_cycle
    pw_rest_tca = build_tca_cycle(energy_demand=0.2)
    pw_ex_tca = build_tca_cycle(energy_demand=1.8)
    t7, sol_rest_tca, flux_rest_tca = pw_rest_tca.simulate(t_span=(0, 60), n_points=400)
    t8, sol_ex_tca, flux_ex_tca = pw_ex_tca.simulate(t_span=(0, 60), n_points=400)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(t7, flux_rest_tca["IDH"], label="Resting (energy-replete)", color="#1f77b4")
    axes[0].plot(t8, flux_ex_tca["IDH"], label="Exercise (high ADP demand)", color="#e377c2")
    axes[0].set_title("Isocitrate dehydrogenase flux vs. energy demand")
    axes[0].set_xlabel("Time (a.u.)"); axes[0].set_ylabel("Flux (a.u./time)")
    axes[0].legend(frameon=False); axes[0].grid(alpha=0.25)

    axes[1].plot(t7, flux_rest_tca["NADH_to_ETC"], label="Resting", color="#1f77b4")
    axes[1].plot(t8, flux_ex_tca["NADH_to_ETC"], label="Exercise", color="#e377c2")
    axes[1].set_title("Downstream effect: NADH delivered to electron transport chain")
    axes[1].set_xlabel("Time (a.u.)"); axes[1].set_ylabel("Flux (a.u./time)")
    axes[1].legend(frameon=False); axes[1].grid(alpha=0.25)
    fig.suptitle("Regulation in action — TCA cycle throttled by ADP/energy charge",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(outdir, "regulation_contrast_tca_cycle.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[regulation demo] saved {path}")

    # --- Ketogenesis: prolonged fasting vs fed -----------------------------
    from pathways import build_ketogenesis
    pw_fed_k = build_ketogenesis(fasting_drive=0.2)
    pw_fast_k = build_ketogenesis(fasting_drive=1.8)
    t9, sol_fed_k, flux_fed_k = pw_fed_k.simulate(t_span=(0, 60), n_points=400)
    t10, sol_fast_k, flux_fast_k = pw_fast_k.simulate(t_span=(0, 60), n_points=400)
    idx_bhb = pw_fed_k.metabolite_names().index("BHB")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(t9, flux_fed_k["HMGCoA_synthase"], label="Fed", color="#8c564b")
    axes[0].plot(t10, flux_fast_k["HMGCoA_synthase"], label="Fasting", color="#bcbd22")
    axes[0].set_title("HMG-CoA synthase flux vs. acetyl-CoA supply")
    axes[0].set_xlabel("Time (a.u.)"); axes[0].set_ylabel("Flux (a.u./time)")
    axes[0].legend(frameon=False); axes[0].grid(alpha=0.25)

    axes[1].plot(t9, sol_fed_k[:, idx_bhb], label="Fed", color="#8c564b")
    axes[1].plot(t10, sol_fast_k[:, idx_bhb], label="Fasting", color="#bcbd22")
    axes[1].set_title("Downstream effect: beta-hydroxybutyrate (ketone body) level")
    axes[1].set_xlabel("Time (a.u.)"); axes[1].set_ylabel("Concentration (a.u.)")
    axes[1].legend(frameon=False); axes[1].grid(alpha=0.25)
    fig.suptitle("Regulation in action — Ketogenesis ramps up with fasting",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(outdir, "regulation_contrast_ketogenesis.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[regulation demo] saved {path}")


def main():
    parser = argparse.ArgumentParser(description="Metabolic Pathway Simulator")
    parser.add_argument("--pathway", choices=list(PATHWAY_CONFIG.keys()) + ["all"],
                         default="all", help="Which pathway to simulate")
    parser.add_argument("--outdir", default="outputs", help="Output directory for figures")
    parser.add_argument("--list", action="store_true", help="List available pathways and exit")
    parser.add_argument("--no-regulation-demo", action="store_true",
                         help="Skip the fed-vs-fasting regulation contrast figures")
    args = parser.parse_args()

    if args.list:
        print("Available pathways:")
        for k in PATHWAY_CONFIG:
            print(f"  - {k}")
        return

    os.makedirs(args.outdir, exist_ok=True)

    if args.pathway == "all":
        for key in PATHWAY_CONFIG:
            run_pathway(key, args.outdir)
        if not args.no_regulation_demo:
            run_regulation_contrast(args.outdir)
    else:
        run_pathway(args.pathway, args.outdir)

    print(f"\nAll figures saved under: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
