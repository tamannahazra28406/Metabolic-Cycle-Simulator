# Metabolic Pathway Simulator

ODE-based simulator (built on `scipy.integrate.odeint`) for nine core human
metabolic cycles, with visualization of chemical intermediates, enzymatic
flux over time, and allosteric/hormonal regulation.

## Pathways included

| Pathway | Direction | Key regulated enzyme(s) |
|---|---|---|
| **Glycolysis** | Glucose → Pyruvate | PFK-1 (ATP/AMP/citrate/F2,6-BP), Pyruvate kinase (F1,6-BP/ATP), Hexokinase (G6P) |
| **Gluconeogenesis** | Pyruvate → Glucose | Pyruvate carboxylase/PEPCK (acetyl-CoA), FBPase-1 (AMP/F2,6-BP/citrate) |
| **Glycogenesis** | Glucose → Glycogen | Glycogen synthase (insulin, G6P, storage capacity) |
| **Glycogenolysis** | Glycogen → Glucose/G6P | Glycogen phosphorylase (glucagon/epinephrine, AMP, ATP, glucose) |
| **Beta-oxidation** | Fatty acyl-CoA → Acetyl-CoA | CPT-1 (malonyl-CoA gatekeeper), oxidation spiral (NADH/FADH2/acetyl-CoA feedback) |
| **TCA (Krebs) cycle** | Acetyl-CoA → CO2 + NADH/FADH2 | Isocitrate DH & alpha-KG DH (ADP activates; ATP/NADH inhibit), citrate synthase (citrate/ATP/NADH) |
| **Pentose phosphate pathway** | G6P → NADPH + ribose-5-P | G6PD & 6PGD (strongly NADPH-inhibited; "pulled" by NADPH demand) |
| **Ketogenesis** | Acetyl-CoA → Ketone bodies | HMG-CoA synthase (driven by acetyl-CoA supply during fasting) |
| **Urea cycle** | NH3 → Urea | CPS-I (requires N-acetylglutamate; fed forward by arginine) |

## Files

- `core.py` — kinetics building blocks (Michaelis-Menten, Hill
  activation/inhibition) and the `Metabolite` / `Enzyme` / `Pathway`
  data structures used to both simulate and describe each pathway.
- `pathways.py` — the nine ODE models themselves, each documented with its
  chemical intermediates, enzymes, reactions, and regulation.
- `visualization.py` — concentration-vs-time plots, flux-vs-time plots, and
  schematic pathway diagrams (intermediates → enzymes → intermediates, with
  dashed arrows showing allosteric regulators).
- `main.py` — command-line driver that runs everything and saves figures.

## Usage

```bash
# Run every pathway, print a text "showcase" of each, save all figures
python main.py

# Just one pathway
python main.py --pathway glycolysis
python main.py --pathway tca_cycle
python main.py --pathway pentose_phosphate
python main.py --pathway ketogenesis
python main.py --pathway urea_cycle

# Change output folder
python main.py --outdir myplots

# List available pathways
python main.py --list
```

For each pathway this produces:
- `<pathway>_concentrations.png` — every intermediate's level over time
- `<pathway>_fluxes.png` — every enzyme's reaction rate over time
- `<pathway>_summary.png` — both stacked in one figure
- `<pathway>_diagram.png` — schematic pathway map with regulation arrows

Plus five **"regulation in action"** comparison figures that run the same
pathway under two different hormonal/metabolic states side by side:
- `regulation_contrast_glycolysis.png` — fed (high insulin) vs. fasting
- `regulation_contrast_glycogenolysis.png` — hormone surge vs. rest
- `regulation_contrast_beta_oxidation.png` — fed (malonyl-CoA blocks CPT-1) vs. fasting
- `regulation_contrast_tca_cycle.png` — exercise (high ADP) vs. resting
- `regulation_contrast_ketogenesis.png` — prolonged fasting vs. fed

## Using it programmatically

```python
from pathways import build_glycolysis, build_tca_cycle
from visualization import plot_pathway_diagram, plot_summary_figure

pathway = build_glycolysis(hormonal_drive=1.5)   # simulate high insulin
print(pathway.summary())                          # text showcase

t, sol, fluxes = pathway.simulate(t_span=(0, 150))
plot_summary_figure(pathway, t, sol, fluxes, save_path="glycolysis.png")

tca = build_tca_cycle(energy_demand=1.8)          # simulate exercise
t2, sol2, fluxes2 = tca.simulate(t_span=(0, 60))
```

Every `build_*()` function in `pathways.py` accepts a parameter that shifts
the hormonal/metabolic state, so you can explore how regulation reshapes the
flux and concentration curves:

| Function | Regulation parameter |
|---|---|
| `build_glycolysis` | `hormonal_drive` (insulin / F2,6-BP level) |
| `build_gluconeogenesis` | `fasting_drive` (acetyl-CoA / glucagon level) |
| `build_glycogenesis` | `insulin_signal` |
| `build_glycogenolysis` | `hormonal_signal` (glucagon/epinephrine), `liver` (bool) |
| `build_beta_oxidation` | `fed_state` (malonyl-CoA level) |
| `build_tca_cycle` | `energy_demand` (ADP/Ca2+ level) |
| `build_pentose_phosphate` | `nadph_demand` |
| `build_ketogenesis` | `fasting_drive` |
| `build_urea_cycle` | `protein_intake` |

## Notes on the modeling approach

- Concentrations/rates are in arbitrary relative units chosen for clear,
  stable, textbook-shaped dynamics rather than literal in-vivo values — the
  goal is to teach the *logic* of each pathway (which step is rate-limiting,
  which regulators speed it up or slow it down, and how the pathways
  reciprocally regulate each other) rather than to reproduce exact
  physiological concentrations.
- Some multi-step segments are deliberately lumped into a single ODE flux
  (e.g. the four inner steps of one β-oxidation spiral turn) while still
  being listed individually as distinct enzymes in the pathway "showcase"
  for educational completeness.
- The TCA cycle and urea cycle are modeled as true cycles: their terminal
  intermediates (oxaloacetate, ornithine) are regenerated each turn rather
  than being consumed, exactly as in the real pathways.
- The framework in `core.py` is intentionally generic — further pathways
  (fatty-acid synthesis, amino-acid catabolism, the HMP shunt's
  non-oxidative branch, etc.) can be added by writing a new `build_*()`
  function following the same pattern.
