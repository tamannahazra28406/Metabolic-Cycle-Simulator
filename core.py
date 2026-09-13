"""
core.py
=======
Core framework for building and simulating metabolic pathway ODE models.

Design philosophy
------------------
Every pathway is represented as a chain of enzyme-catalyzed reactions acting
on a vector of chemical intermediate concentrations. Reaction rates (fluxes)
are computed with standard enzyme-kinetics building blocks (Michaelis-Menten,
reversible MM, Hill activation/inhibition) so that allosteric regulation
(feedback inhibition, allosteric activation, hormonal signals) can be layered
on top of the basic saturation kinetics.

`scipy.integrate.odeint` is used to integrate d[metabolite]/dt = f(state, t)
for every pathway. Concentrations are in arbitrary relative units (a.u.,
loosely "mM-like") chosen for clear, stable, illustrative dynamics rather
than being literally in-vivo accurate -- the point of the simulator is to
*teach the shape and logic* of each pathway (intermediates, enzymes,
direction of flux, and how regulators speed up / slow down specific steps).
"""

from __future__ import annotations
import numpy as np
from scipy.integrate import odeint
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Kinetic building blocks
# ---------------------------------------------------------------------------

def mm(S: float, vmax: float, km: float) -> float:
    """Irreversible Michaelis-Menten rate: v = Vmax*S/(Km+S)."""
    S = max(S, 0.0)
    return vmax * S / (km + S)


def mm_reversible(S: float, P: float, vmax_f: float, km_s: float,
                   vmax_r: float, km_p: float) -> float:
    """Net rate of a reversible MM step (positive = forward S->P)."""
    S, P = max(S, 0.0), max(P, 0.0)
    return vmax_f * S / (km_s + S) - vmax_r * P / (km_p + P)


def hill_activation(x: float, ka: float, n: float = 1.0) -> float:
    """Fractional activation (0->1) by an allosteric activator x."""
    x = max(x, 0.0)
    return x ** n / (ka ** n + x ** n)


def hill_inhibition(x: float, ki: float, n: float = 1.0) -> float:
    """Fractional remaining activity (1->0) under inhibitor x."""
    x = max(x, 0.0)
    return ki ** n / (ki ** n + x ** n)


def clamp_nonneg(v: float) -> float:
    return max(v, 0.0)


def adenylate_pool(atp: float, a_total: float, k_ak: float = 1.0):
    """
    Derive ADP and AMP from a dynamically tracked ATP pool, assuming
    conservation ATP+ADP+AMP = A_total and near-equilibrium adenylate
    kinase: 2 ADP <-> ATP + AMP  =>  AMP*ATP/ADP^2 = k_ak.
    Returns (adp, amp).
    """
    atp = min(max(atp, 1e-6), a_total - 1e-6)
    remaining = a_total - atp
    # Solve AMP*atp = k_ak*(remaining-AMP)^2 approximately via simple split:
    # Use a stable closed-form approximation instead of full quadratic to
    # keep the ODE right-hand-side cheap and smooth.
    amp = remaining * (remaining / a_total) * k_ak
    amp = min(amp, remaining)
    adp = remaining - amp
    return adp, amp


# ---------------------------------------------------------------------------
# Descriptive data classes (used for both simulation bookkeeping AND for
# generating the human-readable "showcase" of each pathway)
# ---------------------------------------------------------------------------

@dataclass
class Metabolite:
    name: str
    initial: float
    unit: str = "a.u."
    description: str = ""


@dataclass
class Enzyme:
    name: str
    reaction: str            # e.g. "Fructose-6-P -> Fructose-1,6-BP"
    ec_number: str = ""
    process: str = ""        # which stage / process this step belongs to
    regulation: str = ""     # human readable regulation summary
    regulators: List[str] = field(default_factory=list)   # metabolite names
    activators: List[str] = field(default_factory=list)
    inhibitors: List[str] = field(default_factory=list)


@dataclass
class Pathway:
    name: str
    metabolites: List[Metabolite]
    enzymes: List[Enzyme]
    ode_func: Callable
    flux_func: Callable
    params: Dict = field(default_factory=dict)
    notes: str = ""

    def metabolite_names(self) -> List[str]:
        return [m.name for m in self.metabolites]

    def initial_state(self) -> List[float]:
        return [m.initial for m in self.metabolites]

    def simulate(self, t_span=(0.0, 100.0), n_points: int = 600):
        """Integrate the ODE system with odeint and evaluate fluxes along
        the resulting trajectory. Returns (t, state_matrix, flux_dict)."""
        t = np.linspace(t_span[0], t_span[1], n_points)
        y0 = self.initial_state()
        sol = odeint(self.ode_func, y0, t, args=(self.params,))

        flux_records = [self.flux_func(state, tt, self.params)
                         for state, tt in zip(sol, t)]
        flux_names = list(flux_records[0].keys())
        fluxes = {name: np.array([rec[name] for rec in flux_records])
                  for name in flux_names}
        return t, sol, fluxes

    def summary(self) -> str:
        lines = [f"PATHWAY: {self.name}", "=" * 70]
        lines.append("\nChemical intermediates:")
        for m in self.metabolites:
            desc = f" - {m.description}" if m.description else ""
            lines.append(f"  * {m.name:<20s} start={m.initial:>7.3f} {m.unit}{desc}")
        lines.append("\nEnzymes / reactions / regulation:")
        for e in self.enzymes:
            ec = f" [{e.ec_number}]" if e.ec_number else ""
            lines.append(f"  * {e.name}{ec}")
            lines.append(f"      reaction : {e.reaction}")
            if e.process:
                lines.append(f"      process  : {e.process}")
            if e.regulation:
                lines.append(f"      regulate : {e.regulation}")
        if self.notes:
            lines.append(f"\nNotes: {self.notes}")
        return "\n".join(lines)
