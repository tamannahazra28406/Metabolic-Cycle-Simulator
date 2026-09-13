"""
pathways.py
===========
Concrete ODE models for five core human metabolic cycles:

    1. Glycolysis            (glucose -> pyruvate, cytosol)
    2. Gluconeogenesis       (pyruvate/lactate -> glucose, liver)
    3. Glycogenesis          (glucose -> glycogen, liver/muscle)
    4. Glycogenolysis        (glycogen -> glucose/G6P, liver/muscle)
    5. Beta-oxidation        (fatty acyl-CoA -> acetyl-CoA, mitochondria)

Each `build_*()` function returns a `core.Pathway` object bundling:
  - the chemical intermediates (state vector) with starting concentrations
  - the enzymes, the reaction each catalyzes, and its regulation
  - the ODE right-hand side used by scipy.integrate.odeint
  - a matching flux function so every enzyme's instantaneous rate can be
    plotted over time, not just the concentrations.

All concentrations/rates are in arbitrary relative units chosen to give
clear, numerically stable, textbook-shaped dynamics.
"""

import numpy as np
from core import Metabolite, Enzyme, Pathway, mm, mm_reversible, \
    hill_activation, hill_inhibition, adenylate_pool


# ===========================================================================
# 1. GLYCOLYSIS
# ===========================================================================
def build_glycolysis(hormonal_drive: float = 1.0) -> Pathway:
    """
    Glucose --HK--> G6P --PGI--> F6P --PFK1--> F1,6BP --ALD/TPI--> 2 TP
      --GAPDH--> 1,3BPG --PGK--> 3PG/2PG(lumped) --ENO--> PEP --PK--> Pyruvate

    Key regulation modeled:
      * Hexokinase        : product-inhibited by G6P
      * PFK-1 (committed, rate-limiting step): allosterically INHIBITED by
        ATP and citrate, ACTIVATED by AMP and fructose-2,6-bisphosphate
        (the master signal of the fed state / insulin action)
      * Pyruvate kinase   : activated by F1,6BP (feed-forward activation),
        inhibited by ATP (energy-charge feedback)
      * ATP is tracked dynamically; ADP/AMP derived from a conserved
        adenine-nucleotide pool so that energy-charge feedback is self
        consistent instead of a fixed constant.

    `hormonal_drive` (0-2, default 1) mimics insulin signalling raising
    fructose-2,6-bisphosphate, which strongly activates PFK-1.
    """
    metabolites = [
        Metabolite("Glucose", 5.0, "a.u.", "extracellular/cytosolic glucose"),
        Metabolite("G6P", 0.5, "a.u.", "glucose-6-phosphate"),
        Metabolite("F6P", 0.3, "a.u.", "fructose-6-phosphate"),
        Metabolite("F16BP", 0.1, "a.u.", "fructose-1,6-bisphosphate"),
        Metabolite("TP", 0.1, "a.u.", "triose phosphates (GAP+DHAP pool)"),
        Metabolite("BPG13", 0.05, "a.u.", "1,3-bisphosphoglycerate"),
        Metabolite("PG", 0.05, "a.u.", "3-PG / 2-PG pool"),
        Metabolite("PEP", 0.05, "a.u.", "phosphoenolpyruvate"),
        Metabolite("Pyruvate", 0.2, "a.u.", "end product -> TCA / lactate"),
        Metabolite("ATP", 3.0, "a.u.", "cytosolic ATP (energy charge)"),
        Metabolite("NADH", 0.3, "a.u.", "cytosolic NADH"),
    ]

    enzymes = [
        Enzyme("Hexokinase", "Glucose + ATP -> G6P + ADP", "EC 2.7.1.1",
               "Committed entry step",
               "Product-inhibited by G6P (feedback)", inhibitors=["G6P"]),
        Enzyme("Phosphoglucose isomerase", "G6P <-> F6P", "EC 5.3.1.9",
               "Isomerization", "Near-equilibrium, no major regulation"),
        Enzyme("Phosphofructokinase-1 (PFK-1)", "F6P + ATP -> F1,6BP + ADP",
               "EC 2.7.1.11", "Committed, rate-limiting step",
               "Inhibited by ATP & citrate; activated by AMP and "
               "fructose-2,6-BP (insulin signal)",
               inhibitors=["ATP", "Citrate"], activators=["AMP", "F26BP"]),
        Enzyme("Aldolase + Triose-P isomerase", "F1,6BP -> 2 TP",
               "EC 4.1.2.13 / 5.3.1.1", "Cleavage & isomerization",
               "Constitutive"),
        Enzyme("GAPDH", "TP + NAD+ + Pi -> 1,3-BPG + NADH", "EC 1.2.1.12",
               "Oxidation / substrate-level phosphorylation prep",
               "Inhibited by high NADH/NAD+ ratio", inhibitors=["NADH"]),
        Enzyme("Phosphoglycerate kinase", "1,3-BPG + ADP -> 3-PG + ATP",
               "EC 2.7.2.3", "Substrate-level phosphorylation",
               "Constitutive"),
        Enzyme("Enolase + PGM (lumped)", "3-PG -> PEP", "EC 4.2.1.11",
               "Dehydration", "Constitutive"),
        Enzyme("Pyruvate kinase", "PEP + ADP -> Pyruvate + ATP",
               "EC 2.7.1.40", "Committed, rate-limiting step",
               "Activated by F1,6BP (feed-forward); inhibited by ATP",
               activators=["F16BP"], inhibitors=["ATP"]),
    ]

    params = dict(
        v_glc_in=0.9,
        vmax_hk=2.5, km_hk=1.5, ki_g6p=1.2,
        vmax_pgi_f=6.0, km_pgi_s=0.5, vmax_pgi_r=5.0, km_pgi_p=0.5,
        vmax_pfk=3.0, km_pfk=0.4, ki_atp_pfk=4.0, n_atp_pfk=2.0,
        ka_amp_pfk=0.4, n_amp_pfk=2.0, ki_citrate=2.0, n_cit=2.0,
        vmax_ald=6.0, km_ald=0.3,
        vmax_gapdh=6.0, km_gapdh=0.3, ki_nadh=1.0, n_nadh=2.0,
        vmax_pgk=6.0, km_pgk=0.2,
        vmax_eno=6.0, km_eno=0.2,
        vmax_pk=7.0, km_pk=0.3, ka_f16bp_pk=0.05, n_f16bp=2.0,
        ki_atp_pk=5.0, n_atp_pk=2.0,
        vmax_pyr_sink=2.0, km_pyr_sink=0.5,
        k_atpase=0.55,   # general cellular ATP-ase load, scales with [ATP]
        vmax_nadh_reox=5.0, km_nadh_reox=0.3,
        a_total=4.0, k_ak=0.8,
        citrate_const=0.6,
        f26bp=0.5 * hormonal_drive,   # insulin-controlled activator level
    )

    def rates(state, p):
        (Glc, G6P, F6P, F16BP, TP, BPG, PG, PEP, Pyr, ATP, NADH) = \
            [max(v, 0.0) for v in state]
        ADP, AMP = adenylate_pool(ATP, p["a_total"], p["k_ak"])
        NAD = max(1.0 - NADH, 0.05)  # small pooled NAD+ availability proxy

        v_hk = mm(Glc, p["vmax_hk"], p["km_hk"]) * \
            hill_inhibition(G6P, p["ki_g6p"]) * (ATP / (ATP + 0.5))
        v_pgi = mm_reversible(G6P, F6P, p["vmax_pgi_f"], p["km_pgi_s"],
                               p["vmax_pgi_r"], p["km_pgi_p"])
        v_pfk = mm(F6P, p["vmax_pfk"], p["km_pfk"]) * \
            hill_inhibition(ATP, p["ki_atp_pfk"], p["n_atp_pfk"]) * \
            (0.2 + 0.8 * hill_activation(AMP + p["f26bp"], p["ka_amp_pfk"],
                                          p["n_amp_pfk"])) * \
            hill_inhibition(p["citrate_const"], p["ki_citrate"], p["n_cit"])
        v_ald = mm(F16BP, p["vmax_ald"], p["km_ald"])
        v_gapdh = mm(TP, p["vmax_gapdh"], p["km_gapdh"]) * (NAD / (NAD + 0.2)) \
            * hill_inhibition(NADH, p["ki_nadh"], p["n_nadh"])
        v_pgk = mm(BPG, p["vmax_pgk"], p["km_pgk"])
        v_eno = mm(PG, p["vmax_eno"], p["km_eno"])
        v_pk = mm(PEP, p["vmax_pk"], p["km_pk"]) * \
            (0.3 + 0.7 * hill_activation(F16BP, p["ka_f16bp_pk"], p["n_f16bp"])) * \
            hill_inhibition(ATP, p["ki_atp_pk"], p["n_atp_pk"])
        v_pyr_sink = mm(Pyr, p["vmax_pyr_sink"], p["km_pyr_sink"])
        v_nadh_reox = mm(NADH, p["vmax_nadh_reox"], p["km_nadh_reox"])

        return dict(HK=v_hk, PGI=v_pgi, PFK1=v_pfk, ALD_TPI=v_ald,
                     GAPDH=v_gapdh, PGK=v_pgk, ENO=v_eno, PK=v_pk,
                     Pyruvate_sink=v_pyr_sink, NADH_reox=v_nadh_reox)

    def ode(state, t, p):
        v = rates(state, p)
        (Glc, G6P, F6P, F16BP, TP, BPG, PG, PEP, Pyr, ATP, NADH) = state
        dGlc = p["v_glc_in"] - v["HK"]
        dG6P = v["HK"] - v["PGI"]
        dF6P = v["PGI"] - v["PFK1"]
        dF16BP = v["PFK1"] - v["ALD_TPI"]
        dTP = 2 * v["ALD_TPI"] - v["GAPDH"]
        dBPG = v["GAPDH"] - v["PGK"]
        dPG = v["PGK"] - v["ENO"]
        dPEP = v["ENO"] - v["PK"]
        dPyr = v["PK"] - v["Pyruvate_sink"]
        dATP = (-v["HK"] - v["PFK1"] + v["PGK"] + v["PK"]
                - p["k_atpase"] * max(ATP, 0.0))
        dNADH = v["GAPDH"] - v["NADH_reox"]
        return [dGlc, dG6P, dF6P, dF16BP, dTP, dBPG, dPG, dPEP, dPyr, dATP, dNADH]

    def flux_func(state, t, p):
        return rates(state, p)

    return Pathway(
        name="Glycolysis",
        metabolites=metabolites, enzymes=enzymes,
        ode_func=ode, flux_func=flux_func, params=params,
        notes="Cytosolic pathway present in essentially all cells; liver and "
              "muscle shown here. PFK-1 is the master regulatory/rate-limiting "
              "enzyme, integrating energy charge (ATP/AMP) and hormonal state "
              "(fructose-2,6-BP driven by insulin/glucagon).",
    )


# ===========================================================================
# 2. GLUCONEOGENESIS
# ===========================================================================
def build_gluconeogenesis(fasting_drive: float = 1.0) -> Pathway:
    """
    Pyruvate --PC+PEPCK--> PEP --(reverse glycolysis enzymes)--> F1,6BP
      --FBPase-1--> F6P --PGI--> G6P --G6Pase--> Glucose  (liver/kidney)

    Key regulation:
      * Pyruvate carboxylase / PEPCK bypass: ACTIVATED by acetyl-CoA
        (signals abundant fat oxidation / fasting), the classic
        "acetyl-CoA activates pyruvate carboxylase" control point.
      * Fructose-1,6-bisphosphatase (FBPase-1): the reciprocal regulatory
        mirror of PFK-1 -- INHIBITED by AMP and by fructose-2,6-BP
        (i.e. insulin/fed state shuts gluconeogenesis down), activated
        when citrate is high.
      * Glucose-6-phosphatase: liver/kidney only, releases free glucose.

    `fasting_drive` (0-2, default 1) raises acetyl-CoA & lowers F2,6BP,
    the hormonal (glucagon-dominant) push toward gluconeogenesis.
    """
    metabolites = [
        Metabolite("Pyruvate", 2.0, "a.u.", "from lactate/alanine (Cori/glucose-alanine cycle)"),
        Metabolite("PEP", 0.2, "a.u.", "phosphoenolpyruvate"),
        Metabolite("F16BP", 0.1, "a.u.", "fructose-1,6-bisphosphate"),
        Metabolite("F6P", 0.1, "a.u.", "fructose-6-phosphate"),
        Metabolite("G6P", 0.1, "a.u.", "glucose-6-phosphate"),
        Metabolite("Glucose", 0.5, "a.u.", "free glucose released to blood"),
    ]

    enzymes = [
        Enzyme("Pyruvate carboxylase + PEPCK (bypass 1)",
               "Pyruvate + CO2 + ATP -> OAA -> PEP + GDP", "EC 6.4.1.1 / 4.1.1.32",
               "Anaplerotic bypass of pyruvate kinase",
               "Activated allosterically by acetyl-CoA (fasting/beta-oxidation signal)",
               activators=["AcetylCoA"]),
        Enzyme("Reverse glycolytic enzymes (ENO, PGM, PGK, GAPDH, ALD - lumped)",
               "PEP -> F1,6BP", "EC 4.2.1.11 etc.", "Reversal of glycolytic steps",
               "Near-equilibrium reactions, driven by mass action / NADH supply"),
        Enzyme("Fructose-1,6-bisphosphatase (FBPase-1)", "F1,6BP -> F6P",
               "EC 3.1.3.11", "Bypass of PFK-1 (committed, rate-limiting)",
               "Inhibited by AMP and fructose-2,6-BP; activated by citrate",
               inhibitors=["AMP", "F26BP"], activators=["Citrate"]),
        Enzyme("Phosphoglucose isomerase", "F6P -> G6P", "EC 5.3.1.9",
               "Isomerization", "Near-equilibrium"),
        Enzyme("Glucose-6-phosphatase", "G6P -> Glucose + Pi", "EC 3.1.3.9",
               "Bypass of hexokinase; liver/kidney only",
               "Only expressed in gluconeogenic/glycogenolytic organs"),
    ]

    params = dict(
        vmax_pcpepck=2.2, km_pcpepck=1.0,
        ka_accoa=0.6, n_accoa=2.0, acetylcoa=0.9 * fasting_drive,
        vmax_rev=5.0, km_rev_s=0.4, vmax_rev_r=1.0, km_rev_p=0.4,
        vmax_fbpase=2.0, km_fbpase=0.3,
        ki_amp=0.5, n_amp=2.0, amp_const=0.35 / max(fasting_drive, 0.3),
        ki_f26bp=0.3, n_f26bp=2.0, f26bp=0.15 / max(fasting_drive, 0.3),
        ka_citrate=0.4, n_cit=2.0, citrate_const=0.7,
        vmax_pgi=5.0, km_pgi=0.3,
        vmax_g6pase=1.8, km_g6pase=0.3,
        pyr_input=0.5,
        vmax_glc_out=1.5, km_glc_out=1.0,   # export of glucose to blood (saturating)
    )

    def rates(state, p):
        Pyr, PEP, F16BP, F6P, G6P, Glc = [max(v, 0.0) for v in state]

        v1 = mm(Pyr, p["vmax_pcpepck"], p["km_pcpepck"]) * \
            (0.2 + 0.8 * hill_activation(p["acetylcoa"], p["ka_accoa"], p["n_accoa"]))
        v2 = mm_reversible(PEP, F16BP, p["vmax_rev"], p["km_rev_s"],
                            p["vmax_rev_r"], p["km_rev_p"])
        v3 = mm(F16BP, p["vmax_fbpase"], p["km_fbpase"]) * \
            hill_inhibition(p["amp_const"], p["ki_amp"], p["n_amp"]) * \
            hill_inhibition(p["f26bp"], p["ki_f26bp"], p["n_f26bp"]) * \
            (0.4 + 0.6 * hill_activation(p["citrate_const"], p["ka_citrate"], p["n_cit"]))
        v4 = mm(F6P, p["vmax_pgi"], p["km_pgi"])
        v5 = mm(G6P, p["vmax_g6pase"], p["km_g6pase"])
        v6 = mm(Glc, p["vmax_glc_out"], p["km_glc_out"])

        return dict(PC_PEPCK=v1, Reverse_glycolysis=v2, FBPase1=v3,
                     PGI=v4, G6Pase=v5, Glucose_export=v6)

    def ode(state, t, p):
        v = rates(state, p)
        dPyr = p["pyr_input"] - v["PC_PEPCK"]
        dPEP = v["PC_PEPCK"] - v["Reverse_glycolysis"]
        dF16BP = v["Reverse_glycolysis"] - v["FBPase1"]
        dF6P = v["FBPase1"] - v["PGI"]
        dG6P = v["PGI"] - v["G6Pase"]
        dGlc = v["G6Pase"] - v["Glucose_export"]
        return [dPyr, dPEP, dF16BP, dF6P, dG6P, dGlc]

    def flux_func(state, t, p):
        return rates(state, p)

    return Pathway(
        name="Gluconeogenesis",
        metabolites=metabolites, enzymes=enzymes,
        ode_func=ode, flux_func=flux_func, params=params,
        notes="Primarily hepatic (and renal cortex) pathway active during "
              "fasting. It is the reciprocal mirror image of glycolysis: "
              "wherever glycolysis has an irreversible, regulated step, "
              "gluconeogenesis uses a distinct bypass enzyme so the two "
              "pathways are never both running at full speed at once "
              "(reciprocal regulation via fructose-2,6-BP).",
    )


# ===========================================================================
# 3. GLYCOGENESIS
# ===========================================================================
def build_glycogenesis(insulin_signal: float = 1.0) -> Pathway:
    """
    G6P <-PGM-> G1P --UDPG-pyrophosphorylase--> UDP-glucose --Glycogen synthase--> Glycogen

    Key regulation:
      * Glycogen synthase (rate-limiting): activated allosterically by G6P
        and, physiologically, by insulin-driven dephosphorylation (its
        active "I-form"). `insulin_signal` (0-2, default 1) directly scales
        the active-enzyme fraction, mimicking the PP1/PKB-AKT dephosphorylation
        cascade triggered by insulin.
    """
    metabolites = [
        Metabolite("G6P", 1.0, "a.u.", "glucose-6-phosphate (glycolysis/glycogenesis hub)"),
        Metabolite("G1P", 0.2, "a.u.", "glucose-1-phosphate"),
        Metabolite("UDPglucose", 0.2, "a.u.", "activated glucosyl donor"),
        Metabolite("Glycogen", 1.0, "a.u.", "storage polymer (relative glucosyl units)"),
    ]

    enzymes = [
        Enzyme("Phosphoglucomutase", "G6P <-> G1P", "EC 5.4.2.2",
               "Isomerization", "Near-equilibrium"),
        Enzyme("UDP-glucose pyrophosphorylase", "G1P + UTP -> UDP-glucose + PPi",
               "EC 2.7.7.9", "Activation of glucosyl donor",
               "Constitutive, pulled forward by PPi hydrolysis"),
        Enzyme("Glycogen synthase", "UDP-glucose + Glycogen(n) -> Glycogen(n+1) + UDP",
               "EC 2.4.1.11", "Committed, rate-limiting step",
               "Allosterically activated by G6P; activity fraction directly "
               "controlled by insulin-driven dephosphorylation (active I-form)",
               activators=["G6P", "Insulin"]),
    ]

    params = dict(
        vmax_pgm_f=4.0, km_pgm_s=0.3, vmax_pgm_r=3.5, km_pgm_p=0.3,
        vmax_ugpase=1.2, km_ugpase=0.3,
        vmax_gs=2.0, km_gs=0.3, ka_g6p=0.15, n_g6p=2.0,
        insulin_signal=insulin_signal,
        g6p_input=0.55,
        vmax_mobilization=0.15, km_mobilization=1.0,   # slow basal turnover (saturating)
        glycogen_capacity=12.0,   # finite hepatic/muscle storage capacity
    )

    def rates(state, p):
        G6P, G1P, UDPG, Glycogen = [max(v, 0.0) for v in state]
        active_gs_fraction = min(1.0, 0.15 + 0.75 * p["insulin_signal"])

        v_pgm = mm_reversible(G6P, G1P, p["vmax_pgm_f"], p["km_pgm_s"],
                               p["vmax_pgm_r"], p["km_pgm_p"])
        space_left = max(0.0, 1.0 - Glycogen / p["glycogen_capacity"])
        # UDP (byproduct of glycogen synthase) backs up and feedback-inhibits
        # UGPase as the glycogen store approaches capacity.
        v_ugpase = mm(G1P, p["vmax_ugpase"], p["km_ugpase"]) * \
            (0.05 + 0.95 * space_left)
        v_gs = mm(UDPG, p["vmax_gs"], p["km_gs"]) * \
            hill_activation(G6P, p["ka_g6p"], p["n_g6p"]) * active_gs_fraction * \
            space_left

        v_mobilize = mm(Glycogen, p["vmax_mobilization"], p["km_mobilization"])

        return dict(PGM=v_pgm, UDPG_pyrophosphorylase=v_ugpase,
                     Glycogen_synthase=v_gs, Basal_mobilization=v_mobilize)

    def ode(state, t, p):
        v = rates(state, p)
        G6P, G1P, UDPG, Glycogen = state
        space_left = max(0.0, 1.0 - Glycogen / p["glycogen_capacity"])
        # Hepatic glucose uptake dedicated to glycogen storage tapers off as
        # the store fills (the rest is shunted to glycolysis/other fates,
        # not tracked in this standalone model).
        dG6P = p["g6p_input"] * (0.1 + 0.9 * space_left) - v["PGM"]
        dG1P = v["PGM"] - v["UDPG_pyrophosphorylase"]
        dUDPG = v["UDPG_pyrophosphorylase"] - v["Glycogen_synthase"]
        dGlycogen = v["Glycogen_synthase"] - v["Basal_mobilization"]
        return [dG6P, dG1P, dUDPG, dGlycogen]

    def flux_func(state, t, p):
        return rates(state, p)

    return Pathway(
        name="Glycogenesis",
        metabolites=metabolites, enzymes=enzymes,
        ode_func=ode, flux_func=flux_func, params=params,
        notes="Anabolic storage pathway in liver and muscle, dominant in the "
              "fed state. Glycogen synthase activity is switched on by "
              "insulin (dephosphorylation) and further tuned by G6P levels, "
              "so glycogen accumulates fastest right after a carbohydrate meal.",
    )


# ===========================================================================
# 4. GLYCOGENOLYSIS
# ===========================================================================
def build_glycogenolysis(hormonal_signal: float = 1.0, liver: bool = True) -> Pathway:
    """
    Glycogen --Glycogen phosphorylase--> G1P <-PGM-> G6P --G6Pase--> Glucose
                                                                (liver only)

    Key regulation:
      * Glycogen phosphorylase (rate-limiting): activated by the
        glucagon/epinephrine -> cAMP -> PKA -> phosphorylase kinase cascade
        (`hormonal_signal`, 0-2, default 1) and allosterically by AMP
        (muscle "energy low" signal); inhibited by ATP and (in liver) by
        free glucose itself (feedback).
      * Glucose-6-phosphatase only present in liver/kidney -- in muscle,
        G6P instead feeds directly into glycolysis (no free glucose output),
        toggle with `liver=False`.
    """
    metabolites = [
        Metabolite("Glycogen", 5.0, "a.u.", "storage polymer (relative glucosyl units)"),
        Metabolite("G1P", 0.1, "a.u.", "glucose-1-phosphate"),
        Metabolite("G6P", 0.1, "a.u.", "glucose-6-phosphate"),
        Metabolite("Glucose", 0.5, "a.u.", "free glucose (liver) / retained (muscle)"),
    ]

    enzymes = [
        Enzyme("Glycogen phosphorylase", "Glycogen(n) + Pi -> Glycogen(n-1) + G1P",
               "EC 2.4.1.1", "Committed, rate-limiting step",
               "Activated by glucagon/epinephrine (via PKA/phosphorylase "
               "kinase) and by AMP; inhibited by ATP and (in liver) glucose",
               activators=["Hormone", "AMP"], inhibitors=["ATP", "Glucose"]),
        Enzyme("Phosphoglucomutase", "G1P <-> G6P", "EC 5.4.2.2",
               "Isomerization", "Near-equilibrium"),
        Enzyme("Glucose-6-phosphatase", "G6P -> Glucose + Pi", "EC 3.1.3.9",
               "Liver/kidney only - releases free glucose to blood",
               "Absent in muscle: muscle G6P instead enters glycolysis directly"),
    ]

    params = dict(
        vmax_gp=3.0, km_gp=1.5,
        ka_hormone=0.5, n_hormone=2.0, hormonal_signal=hormonal_signal,
        ka_amp=0.4, n_amp=2.0, amp_const=0.4,
        ki_atp=3.0, n_atp=2.0, atp_const=2.5,
        ki_glucose=2.0, n_glc=2.0,
        vmax_pgm_f=4.0, km_pgm_s=0.2, vmax_pgm_r=3.5, km_pgm_p=0.2,
        vmax_g6pase=2.2, km_g6pase=0.3,
        vmax_glc_out=1.2, km_glc_out=0.8,
        liver=liver,
        muscle_g6p_drain=1.8,   # G6P feeding straight into glycolysis (muscle)
        km_muscle_drain=0.3,
    )

    def rates(state, p):
        Glycogen, G1P, G6P, Glucose = [max(v, 0.0) for v in state]

        v_gp = mm(Glycogen, p["vmax_gp"], p["km_gp"]) * \
            (0.1 + 0.9 * hill_activation(p["hormonal_signal"], p["ka_hormone"], p["n_hormone"])) * \
            (0.3 + 0.7 * hill_activation(p["amp_const"], p["ka_amp"], p["n_amp"])) * \
            hill_inhibition(p["atp_const"], p["ki_atp"], p["n_atp"]) * \
            (hill_inhibition(Glucose, p["ki_glucose"], p["n_glc"]) if p["liver"] else 1.0)

        v_pgm = mm_reversible(G1P, G6P, p["vmax_pgm_f"], p["km_pgm_s"],
                               p["vmax_pgm_r"], p["km_pgm_p"])

        if p["liver"]:
            v_g6pase = mm(G6P, p["vmax_g6pase"], p["km_g6pase"])
            v_drain = 0.0
        else:
            v_g6pase = 0.0
            v_drain = mm(G6P, p["muscle_g6p_drain"], p["km_muscle_drain"])

        v_glc_out = mm(Glucose, p["vmax_glc_out"], p["km_glc_out"]) if p["liver"] else 0.0

        return dict(Glycogen_phosphorylase=v_gp, PGM=v_pgm,
                     G6Pase=v_g6pase, Glycolysis_drain=v_drain,
                     Glucose_export=v_glc_out)

    def ode(state, t, p):
        v = rates(state, p)
        Glycogen, G1P, G6P, Glucose = state
        dGlycogen = -v["Glycogen_phosphorylase"]
        dG1P = v["Glycogen_phosphorylase"] - v["PGM"]
        dG6P = v["PGM"] - v["G6Pase"] - v["Glycolysis_drain"]
        dGlucose = v["G6Pase"] - v["Glucose_export"]
        return [dGlycogen, dG1P, dG6P, dGlucose]

    def flux_func(state, t, p):
        return rates(state, p)

    tissue = "liver" if liver else "muscle"
    return Pathway(
        name=f"Glycogenolysis ({tissue})",
        metabolites=metabolites, enzymes=enzymes,
        ode_func=ode, flux_func=flux_func, params=params,
        notes=f"Catabolic mobilization of stored glycogen, modeled here for "
              f"{tissue} tissue. Triggered by glucagon (liver) or epinephrine "
              f"(liver & muscle) during fasting/exercise. "
              + ("Liver expresses glucose-6-phosphatase so it can export free "
                 "glucose to the blood."
                 if liver else
                 "Muscle lacks glucose-6-phosphatase, so G6P is retained and "
                 "funneled directly into glycolysis for local ATP production."),
    )


# ===========================================================================
# 5. BETA-OXIDATION OF FATTY ACIDS
# ===========================================================================
def build_beta_oxidation(fed_state: float = 0.3) -> Pathway:
    """
    Fatty acyl-CoA(cytosol) --CPT-1--> Fatty acyl-CoA(mitochondria)
      --[Acyl-CoA DH -> Enoyl-CoA hydratase -> 3-OH-acyl-CoA DH -> Thiolase]
      (one full lumped cycle, repeated, chain shortens by 2C each turn)--> Acetyl-CoA + FADH2 + NADH (per cycle)

    Key regulation:
      * Carnitine palmitoyltransferase I (CPT-1): the master control point.
        Strongly INHIBITED by malonyl-CoA, which is high in the FED state
        (malonyl-CoA is also the first committed intermediate of fatty-acid
        SYNTHESIS, so this single molecule prevents futile simultaneous
        synthesis and oxidation of fat). `fed_state` (0-1, default 0.3)
        controls malonyl-CoA level directly.
      * The oxidation spiral itself is product-inhibited by the
        NADH/NAD+ and FADH2/FAD ratios (respiratory control): if the
        electron transport chain can't keep up, beta-oxidation slows down.
    """
    metabolites = [
        Metabolite("FA_cytosol", 3.0, "a.u.", "cytosolic long-chain fatty acyl-CoA"),
        Metabolite("FA_mito", 0.2, "a.u.", "mitochondrial fatty acyl-CoA pool"),
        Metabolite("AcetylCoA", 0.3, "a.u.", "acetyl-CoA -> TCA cycle / ketogenesis"),
        Metabolite("NADH", 0.2, "a.u.", "mitochondrial NADH"),
        Metabolite("FADH2", 0.2, "a.u.", "mitochondrial FADH2"),
    ]

    enzymes = [
        Enzyme("Carnitine palmitoyltransferase I (CPT-1)",
               "Fatty acyl-CoA(cyto) + carnitine -> Fatty acyl-carnitine(mito)",
               "EC 2.3.1.21", "Committed, rate-limiting transport step",
               "Strongly inhibited by malonyl-CoA (high in the fed state / "
               "active fatty-acid synthesis)", inhibitors=["MalonylCoA"]),
        Enzyme("Acyl-CoA dehydrogenase", "Acyl-CoA -> trans-Delta2-enoyl-CoA + FADH2",
               "EC 1.3.8.7", "Oxidation step 1 of the spiral",
               "Inhibited by high FADH2/FAD ratio (respiratory control)",
               inhibitors=["FADH2"]),
        Enzyme("Enoyl-CoA hydratase", "trans-Delta2-enoyl-CoA + H2O -> 3-OH-acyl-CoA",
               "EC 4.2.1.17", "Hydration step 2 of the spiral", "Near-equilibrium"),
        Enzyme("3-hydroxyacyl-CoA dehydrogenase", "3-OH-acyl-CoA -> 3-ketoacyl-CoA + NADH",
               "EC 1.1.1.35", "Oxidation step 3 of the spiral",
               "Inhibited by high NADH/NAD+ ratio", inhibitors=["NADH"]),
        Enzyme("Thiolase (3-ketoacyl-CoA thiolase)",
               "3-ketoacyl-CoA + CoA-SH -> Acetyl-CoA + Acyl-CoA(n-2)",
               "EC 2.3.1.16", "Thiolytic cleavage, step 4 of the spiral",
               "Inhibited by high acetyl-CoA/CoA-SH ratio", inhibitors=["AcetylCoA"]),
    ]

    params = dict(
        fa_input=0.9,
        vmax_cpt1=2.0, km_cpt1=1.5,
        malonylcoa=1.2 * fed_state, ki_malonylcoa=0.5, n_mal=2.0,
        vmax_cycle=3.0, km_cycle=0.3,
        ki_nadh=1.0, n_nadh=2.0,
        ki_fadh2=1.0, n_fadh2=2.0,
        ki_accoa=1.2, n_accoa=2.0,
        vmax_accoa_sink=2.5, km_accoa_sink=0.3,
        vmax_nadh_reox=4.0, km_nadh_reox=0.3,
        vmax_fadh2_reox=4.0, km_fadh2_reox=0.3,
    )

    def rates(state, p):
        FA_c, FA_m, AcCoA, NADH, FADH2 = [max(v, 0.0) for v in state]

        v_cpt1 = mm(FA_c, p["vmax_cpt1"], p["km_cpt1"]) * \
            hill_inhibition(p["malonylcoa"], p["ki_malonylcoa"], p["n_mal"])

        v_cycle = mm(FA_m, p["vmax_cycle"], p["km_cycle"]) * \
            hill_inhibition(NADH, p["ki_nadh"], p["n_nadh"]) * \
            hill_inhibition(FADH2, p["ki_fadh2"], p["n_fadh2"]) * \
            hill_inhibition(AcCoA, p["ki_accoa"], p["n_accoa"])

        v_accoa_sink = mm(AcCoA, p["vmax_accoa_sink"], p["km_accoa_sink"])
        v_nadh_reox = mm(NADH, p["vmax_nadh_reox"], p["km_nadh_reox"])
        v_fadh2_reox = mm(FADH2, p["vmax_fadh2_reox"], p["km_fadh2_reox"])

        return dict(CPT1=v_cpt1, Oxidation_spiral=v_cycle,
                     AcetylCoA_to_TCA=v_accoa_sink,
                     NADH_to_ETC=v_nadh_reox, FADH2_to_ETC=v_fadh2_reox)

    def ode(state, t, p):
        v = rates(state, p)
        FA_c, FA_m, AcCoA, NADH, FADH2 = state
        dFA_c = p["fa_input"] - v["CPT1"]
        dFA_m = v["CPT1"] - v["Oxidation_spiral"]
        dAcCoA = v["Oxidation_spiral"] - v["AcetylCoA_to_TCA"]
        dNADH = v["Oxidation_spiral"] - v["NADH_to_ETC"]
        dFADH2 = v["Oxidation_spiral"] - v["FADH2_to_ETC"]
        return [dFA_c, dFA_m, dAcCoA, dNADH, dFADH2]

    def flux_func(state, t, p):
        return rates(state, p)

    return Pathway(
        name="Beta-oxidation of fatty acids",
        metabolites=metabolites, enzymes=enzymes,
        ode_func=ode, flux_func=flux_func, params=params,
        notes="Mitochondrial spiral pathway (liver, muscle, heart) that "
              "degrades fatty acyl-CoA two carbons at a time. Modeled here "
              "as a lumped 'one turn of the spiral' flux rather than "
              "tracking every chain length explicitly. CPT-1/malonyl-CoA is "
              "the master switch coordinating fat synthesis vs. fat "
              "oxidation so they are not both maximal at once.",
    )


# ===========================================================================
# 6. CITRIC ACID (TCA / KREBS) CYCLE
# ===========================================================================
def build_tca_cycle(energy_demand: float = 1.0) -> Pathway:
    """
    Acetyl-CoA + OAA --CS--> Citrate --ACO--> Isocitrate --IDH--> AlphaKG
      --KGDH--> Succinyl-CoA --SCS--> Succinate --SDH--> Fumarate
      --FH/MDH(lumped)--> OAA (regenerated, closing the cycle)

    Key regulation:
      * Citrate synthase: product-inhibited by citrate itself (and, more
        broadly, by a high ATP/ADP & NADH/NAD+ ratio signalling ample
        energy already available).
      * Isocitrate dehydrogenase (IDH) and alpha-ketoglutarate
        dehydrogenase (KGDH): the two rate-limiting, tightly regulated
        steps. Both are ACTIVATED by ADP (low energy charge / high
        `energy_demand`, e.g. exercise) and INHIBITED by NADH and ATP
        (feedback from a full energy tank). KGDH is additionally
        product-inhibited by succinyl-CoA.
      * Succinate dehydrogenase (also Complex II of the ETC) is
        competitively inhibited by oxaloacetate.

    `energy_demand` (0-2, default 1) mimics ADP/Ca2+ availability: high
    values (exercising muscle) speed the whole cycle up; low values (resting,
    energy-replete cell) throttle it back.
    """
    metabolites = [
        Metabolite("AcetylCoA", 0.6, "a.u.", "from glycolysis / beta-oxidation"),
        Metabolite("Citrate", 0.3, "a.u.", ""),
        Metabolite("Isocitrate", 0.2, "a.u.", ""),
        Metabolite("AlphaKG", 0.2, "a.u.", "alpha-ketoglutarate"),
        Metabolite("SuccinylCoA", 0.15, "a.u.", ""),
        Metabolite("Succinate", 0.15, "a.u.", ""),
        Metabolite("Fumarate", 0.15, "a.u.", ""),
        Metabolite("OAA", 0.3, "a.u.", "oxaloacetate, regenerated each turn"),
        Metabolite("NADH", 0.3, "a.u.", "mitochondrial NADH"),
        Metabolite("FADH2", 0.2, "a.u.", "mitochondrial FADH2"),
    ]

    enzymes = [
        Enzyme("Citrate synthase", "Acetyl-CoA + OAA -> Citrate", "EC 2.3.3.1",
               "Committed entry step",
               "Inhibited by citrate (product), ATP, and NADH (energy-replete signal)",
               inhibitors=["Citrate", "ATP", "NADH"]),
        Enzyme("Aconitase", "Citrate <-> Isocitrate", "EC 4.2.1.3",
               "Isomerization", "Near-equilibrium"),
        Enzyme("Isocitrate dehydrogenase (IDH)", "Isocitrate -> alpha-KG + NADH + CO2",
               "EC 1.1.1.41", "Committed, rate-limiting step",
               "Activated by ADP; inhibited by ATP and NADH",
               activators=["ADP"], inhibitors=["ATP", "NADH"]),
        Enzyme("alpha-KG dehydrogenase (KGDH)", "alpha-KG -> Succinyl-CoA + NADH + CO2",
               "EC 1.2.4.2", "Rate-limiting step",
               "Inhibited by NADH, succinyl-CoA (product), and ATP",
               inhibitors=["NADH", "SuccinylCoA", "ATP"]),
        Enzyme("Succinyl-CoA synthetase", "Succinyl-CoA -> Succinate + GTP(ATP)",
               "EC 6.2.1.4/5", "Substrate-level phosphorylation", "Constitutive"),
        Enzyme("Succinate dehydrogenase (Complex II)", "Succinate -> Fumarate + FADH2",
               "EC 1.3.5.1", "Also part of the electron transport chain",
               "Competitively inhibited by oxaloacetate", inhibitors=["OAA"]),
        Enzyme("Fumarase + Malate dehydrogenase (lumped)", "Fumarate -> OAA + NADH",
               "EC 4.2.1.2 / 1.1.1.37", "Hydration + oxidation, regenerates OAA",
               "MDH step pulled forward by citrate synthase consuming OAA"),
    ]

    params = dict(
        acetylcoa_input=0.55,
        vmax_cs=2.2, km_cs_accoa=0.3, km_cs_oaa=0.3,
        ki_citrate=1.0, n_cit=2.0, ki_atp_cs=4.0, n_atp_cs=2.0,
        ki_nadh_cs=1.2, n_nadh_cs=2.0,
        vmax_aco=6.0, km_aco_s=0.3, vmax_aco_r=5.0, km_aco_p=0.3,
        vmax_idh=2.2, km_idh=0.3, ka_adp=0.6, n_adp=2.0,
        ki_atp_idh=4.0, n_atp_idh=2.0, ki_nadh_idh=1.2, n_nadh_idh=2.0,
        vmax_kgdh=2.2, km_kgdh=0.3, ki_nadh_kgdh=1.2, n_nadh_kgdh=2.0,
        ki_sucoa=1.0, n_sucoa=2.0, ki_atp_kgdh=4.0, n_atp_kgdh=2.0,
        vmax_scs=5.0, km_scs=0.3,
        vmax_sdh=5.0, km_sdh=0.3, ki_oaa_sdh=1.0, n_oaa=2.0,
        vmax_fhmdh=5.0, km_fhmdh=0.3,
        vmax_nadh_reox=5.5, km_nadh_reox=0.3,
        vmax_fadh2_reox=5.5, km_fadh2_reox=0.3,
        atp_const=3.0, energy_demand=energy_demand,
    )

    def rates(state, p):
        (AcCoA, Cit, Isocit, AKG, SucCoA, Succ, Fum, OAA, NADH, FADH2) = \
            [max(v, 0.0) for v in state]
        adp_proxy = p["energy_demand"]

        v_cs = mm(AcCoA, p["vmax_cs"], p["km_cs_accoa"]) * \
            (OAA / (p["km_cs_oaa"] + OAA)) * \
            hill_inhibition(Cit, p["ki_citrate"], p["n_cit"]) * \
            hill_inhibition(p["atp_const"], p["ki_atp_cs"], p["n_atp_cs"]) * \
            hill_inhibition(NADH, p["ki_nadh_cs"], p["n_nadh_cs"])
        v_aco = mm_reversible(Cit, Isocit, p["vmax_aco"], p["km_aco_s"],
                               p["vmax_aco_r"], p["km_aco_p"])
        v_idh = mm(Isocit, p["vmax_idh"], p["km_idh"]) * \
            (0.2 + 0.8 * hill_activation(adp_proxy, p["ka_adp"], p["n_adp"])) * \
            hill_inhibition(p["atp_const"], p["ki_atp_idh"], p["n_atp_idh"]) * \
            hill_inhibition(NADH, p["ki_nadh_idh"], p["n_nadh_idh"])
        v_kgdh = mm(AKG, p["vmax_kgdh"], p["km_kgdh"]) * \
            hill_inhibition(NADH, p["ki_nadh_kgdh"], p["n_nadh_kgdh"]) * \
            hill_inhibition(SucCoA, p["ki_sucoa"], p["n_sucoa"]) * \
            hill_inhibition(p["atp_const"], p["ki_atp_kgdh"], p["n_atp_kgdh"])
        v_scs = mm(SucCoA, p["vmax_scs"], p["km_scs"])
        v_sdh = mm(Succ, p["vmax_sdh"], p["km_sdh"]) * \
            hill_inhibition(OAA, p["ki_oaa_sdh"], p["n_oaa"])
        v_fhmdh = mm(Fum, p["vmax_fhmdh"], p["km_fhmdh"])
        v_nadh_reox = mm(NADH, p["vmax_nadh_reox"], p["km_nadh_reox"])
        v_fadh2_reox = mm(FADH2, p["vmax_fadh2_reox"], p["km_fadh2_reox"])

        return dict(Citrate_synthase=v_cs, Aconitase=v_aco, IDH=v_idh,
                     KGDH=v_kgdh, SCS=v_scs, SDH=v_sdh, FH_MDH=v_fhmdh,
                     NADH_to_ETC=v_nadh_reox, FADH2_to_ETC=v_fadh2_reox)

    def ode(state, t, p):
        v = rates(state, p)
        (AcCoA, Cit, Isocit, AKG, SucCoA, Succ, Fum, OAA, NADH, FADH2) = state
        dAcCoA = p["acetylcoa_input"] - v["Citrate_synthase"]
        dCit = v["Citrate_synthase"] - v["Aconitase"]
        dIsocit = v["Aconitase"] - v["IDH"]
        dAKG = v["IDH"] - v["KGDH"]
        dSucCoA = v["KGDH"] - v["SCS"]
        dSucc = v["SCS"] - v["SDH"]
        dFum = v["SDH"] - v["FH_MDH"]
        dOAA = v["FH_MDH"] - v["Citrate_synthase"]
        dNADH = v["IDH"] + v["KGDH"] + v["FH_MDH"] - v["NADH_to_ETC"]
        dFADH2 = v["SDH"] - v["FADH2_to_ETC"]
        return [dAcCoA, dCit, dIsocit, dAKG, dSucCoA, dSucc, dFum, dOAA, dNADH, dFADH2]

    def flux_func(state, t, p):
        return rates(state, p)

    return Pathway(
        name="TCA (Krebs) cycle",
        metabolites=metabolites, enzymes=enzymes,
        ode_func=ode, flux_func=flux_func, params=params,
        notes="The mitochondrial hub that oxidizes acetyl-CoA (from "
              "glycolysis, beta-oxidation, and amino-acid catabolism) to "
              "CO2, harvesting electrons as NADH and FADH2 for the electron "
              "transport chain. Truly cyclic: oxaloacetate is regenerated "
              "every turn rather than being a terminal product.",
    )


# ===========================================================================
# 7. PENTOSE PHOSPHATE PATHWAY (oxidative branch)
# ===========================================================================
def build_pentose_phosphate(nadph_demand: float = 1.0) -> Pathway:
    """
    G6P --G6PD--> 6-phosphogluconate --6PGD--> Ribulose-5-P --> Ribose-5-P
                                                       (non-oxidative branch)
    Each oxidative step also produces one NADPH.

    Key regulation:
      * Glucose-6-phosphate dehydrogenase (G6PD): the master rate-limiting
        step of the whole pathway. STRONGLY inhibited by its own product
        NADPH (i.e. by a high NADPH/NADP+ ratio) and, conversely, pulled
        forward whenever NADPH is being consumed quickly (`nadph_demand`,
        representing fatty-acid synthesis or oxidative/glutathione stress).
      * 6-phosphogluconate dehydrogenase (6PGD): similarly NADPH-sensitive.

    `nadph_demand` (0-2, default 1) sets how fast NADPH is drained by
    downstream consumers, which is exactly what relieves G6PD inhibition and
    speeds up the whole pathway -- a clean illustration of "pull" regulation.
    """
    metabolites = [
        Metabolite("G6P", 1.0, "a.u.", "shared hub with glycolysis/glycogenesis"),
        Metabolite("PG6", 0.1, "a.u.", "6-phosphogluconate"),
        Metabolite("Ru5P", 0.1, "a.u.", "ribulose-5-phosphate"),
        Metabolite("R5P", 0.1, "a.u.", "ribose-5-phosphate -> nucleotide synthesis"),
        Metabolite("NADPH", 0.4, "a.u.", "reducing power for biosynthesis / antioxidant defense"),
    ]

    enzymes = [
        Enzyme("Glucose-6-phosphate dehydrogenase (G6PD)",
               "G6P + NADP+ -> 6-phosphogluconolactone + NADPH", "EC 1.1.1.49",
               "Committed, rate-limiting step",
               "Strongly inhibited by NADPH (high NADPH/NADP+ ratio); "
               "activity rises whenever NADPH is being consumed",
               inhibitors=["NADPH"]),
        Enzyme("6-phosphogluconate dehydrogenase (6PGD)",
               "6-PG + NADP+ -> Ribulose-5-P + NADPH + CO2", "EC 1.1.1.44",
               "Oxidative decarboxylation", "Also inhibited by NADPH",
               inhibitors=["NADPH"]),
        Enzyme("Ribulose-5-P isomerase / epimerase (lumped)",
               "Ribulose-5-P -> Ribose-5-P (or Xylulose-5-P)", "EC 5.3.1.6",
               "Non-oxidative branch entry",
               "Constitutive; ribose-5-P feeds nucleotide/nucleic-acid synthesis"),
    ]

    params = dict(
        g6p_input=0.5,
        vmax_g6pd=2.2, km_g6pd=0.5, ki_nadph_g6pd=1.2, n_nadph1=2.0,
        vmax_6pgd=2.5, km_6pgd=0.3, ki_nadph_6pgd=1.4, n_nadph2=2.0,
        vmax_iso=2.0, km_iso=0.3,
        vmax_r5p_sink=1.6, km_r5p_sink=0.3,
        nadph_demand=nadph_demand,
        vmax_nadph_consume=2.0, km_nadph_consume=0.4,
    )

    def rates(state, p):
        G6P, PG6, Ru5P, R5P, NADPH = [max(v, 0.0) for v in state]

        v_g6pd = mm(G6P, p["vmax_g6pd"], p["km_g6pd"]) * \
            hill_inhibition(NADPH, p["ki_nadph_g6pd"], p["n_nadph1"])
        v_6pgd = mm(PG6, p["vmax_6pgd"], p["km_6pgd"]) * \
            hill_inhibition(NADPH, p["ki_nadph_6pgd"], p["n_nadph2"])
        v_iso = mm(Ru5P, p["vmax_iso"], p["km_iso"])
        v_r5p_sink = mm(R5P, p["vmax_r5p_sink"], p["km_r5p_sink"])
        v_nadph_consume = p["nadph_demand"] * \
            mm(NADPH, p["vmax_nadph_consume"], p["km_nadph_consume"])

        return dict(G6PD=v_g6pd, SixPGD=v_6pgd, Isomerase=v_iso,
                     R5P_to_nucleotides=v_r5p_sink,
                     NADPH_consumption=v_nadph_consume)

    def ode(state, t, p):
        v = rates(state, p)
        G6P, PG6, Ru5P, R5P, NADPH = state
        dG6P = p["g6p_input"] - v["G6PD"]
        dPG6 = v["G6PD"] - v["SixPGD"]
        dRu5P = v["SixPGD"] - v["Isomerase"]
        dR5P = v["Isomerase"] - v["R5P_to_nucleotides"]
        dNADPH = v["G6PD"] + v["SixPGD"] - v["NADPH_consumption"]
        return [dG6P, dPG6, dRu5P, dR5P, dNADPH]

    def flux_func(state, t, p):
        return rates(state, p)

    return Pathway(
        name="Pentose phosphate pathway",
        metabolites=metabolites, enzymes=enzymes,
        ode_func=ode, flux_func=flux_func, params=params,
        notes="Cytosolic shunt off glucose-6-phosphate that runs in parallel "
              "with glycolysis. Its two products -- NADPH (for fatty-acid "
              "synthesis, glutathione regeneration, and antioxidant defense) "
              "and ribose-5-phosphate (for nucleotide synthesis) -- mean its "
              "flux is 'pulled' by biosynthetic/antioxidant demand rather "
              "than 'pushed' by substrate supply, unlike most of glycolysis.",
    )


# ===========================================================================
# 8. KETOGENESIS
# ===========================================================================
def build_ketogenesis(fasting_drive: float = 1.0) -> Pathway:
    """
    2 Acetyl-CoA --Thiolase--> Acetoacetyl-CoA --HMG-CoA synthase-->
      HMG-CoA --HMG-CoA lyase--> Acetoacetate --BHB dehydrogenase-->
      beta-hydroxybutyrate (BHB)

    Key regulation:
      * HMG-CoA synthase (mitochondrial isoform): the committed,
        rate-limiting step. Its flux is driven mainly by the sheer supply
        of acetyl-CoA pouring in from beta-oxidation during fasting
        (`fasting_drive`) -- i.e. mostly substrate-push regulation, the
        mirror image of the pentose phosphate pathway's demand-pull.
      * Beta-hydroxybutyrate dehydrogenase: reversible, its direction set
        by the mitochondrial NADH/NAD+ ratio (high NADH from active
        beta-oxidation pushes acetoacetate -> BHB).

    `fasting_drive` (0-2, default 1) scales acetyl-CoA supply, mimicking
    how prolonged fasting ramps up hepatic ketogenesis.
    """
    metabolites = [
        Metabolite("AcetylCoA", 0.6, "a.u.", "surplus from beta-oxidation (liver)"),
        Metabolite("AcetoacetylCoA", 0.1, "a.u.", ""),
        Metabolite("HMGCoA", 0.1, "a.u.", "3-hydroxy-3-methylglutaryl-CoA"),
        Metabolite("Acetoacetate", 0.15, "a.u.", "one of the two ketone bodies"),
        Metabolite("BHB", 0.15, "a.u.", "beta-hydroxybutyrate, the major ketone body"),
    ]

    enzymes = [
        Enzyme("Thiolase", "2 Acetyl-CoA -> Acetoacetyl-CoA + CoA-SH",
               "EC 2.3.1.9", "Condensation", "Constitutive"),
        Enzyme("HMG-CoA synthase (mitochondrial)",
               "Acetoacetyl-CoA + Acetyl-CoA -> HMG-CoA", "EC 2.3.3.10",
               "Committed, rate-limiting step",
               "Flux driven by acetyl-CoA supply from beta-oxidation "
               "(fasting/low-insulin state); induced transcriptionally by "
               "fasting/glucagon", activators=["AcetylCoA"]),
        Enzyme("HMG-CoA lyase", "HMG-CoA -> Acetoacetate + Acetyl-CoA",
               "EC 4.1.3.4", "Cleavage", "Constitutive"),
        Enzyme("beta-hydroxybutyrate dehydrogenase", "Acetoacetate + NADH <-> BHB + NAD+",
               "EC 1.1.1.30", "Reversible reduction",
               "Direction set by the mitochondrial NADH/NAD+ ratio "
               "(favors BHB when beta-oxidation is active)"),
    ]

    params = dict(
        acetylcoa_input=0.5 * fasting_drive,
        vmax_thiolase=2.0, km_thiolase=0.3,
        vmax_hmgs=2.0, km_hmgs=0.3, ka_accoa=0.05, n_accoa=1.0,
        vmax_hmgl=3.0, km_hmgl=0.3,
        vmax_bhbdh_f=3.0, km_bhbdh_s=0.3, vmax_bhbdh_r=1.0, km_bhbdh_p=0.3,
        vmax_bhb_sink=1.5, km_bhb_sink=0.4,
        vmax_acac_sink=0.6, km_acac_sink=0.3,   # minor spontaneous decarboxylation to acetone
    )

    def rates(state, p):
        AcCoA, AcAcCoA, HMGCoA, AcAc, BHB = [max(v, 0.0) for v in state]

        v_thiolase = mm(AcCoA, p["vmax_thiolase"], p["km_thiolase"])
        v_hmgs = mm(AcAcCoA, p["vmax_hmgs"], p["km_hmgs"]) * \
            hill_activation(AcCoA, p["ka_accoa"], p["n_accoa"])
        v_hmgl = mm(HMGCoA, p["vmax_hmgl"], p["km_hmgl"])
        v_bhbdh = mm_reversible(AcAc, BHB, p["vmax_bhbdh_f"], p["km_bhbdh_s"],
                                 p["vmax_bhbdh_r"], p["km_bhbdh_p"])
        v_bhb_sink = mm(BHB, p["vmax_bhb_sink"], p["km_bhb_sink"])
        v_acac_sink = mm(AcAc, p["vmax_acac_sink"], p["km_acac_sink"])

        return dict(Thiolase=v_thiolase, HMGCoA_synthase=v_hmgs,
                     HMGCoA_lyase=v_hmgl, BHB_dehydrogenase=v_bhbdh,
                     BHB_export=v_bhb_sink, Acetone_spontaneous=v_acac_sink)

    def ode(state, t, p):
        v = rates(state, p)
        AcCoA, AcAcCoA, HMGCoA, AcAc, BHB = state
        dAcCoA = p["acetylcoa_input"] - 2 * v["Thiolase"] - v["HMGCoA_synthase"] + v["HMGCoA_lyase"]
        dAcAcCoA = v["Thiolase"] - v["HMGCoA_synthase"]
        dHMGCoA = v["HMGCoA_synthase"] - v["HMGCoA_lyase"]
        dAcAc = v["HMGCoA_lyase"] - v["BHB_dehydrogenase"] - v["Acetone_spontaneous"]
        dBHB = v["BHB_dehydrogenase"] - v["BHB_export"]
        return [dAcCoA, dAcAcCoA, dHMGCoA, dAcAc, dBHB]

    def flux_func(state, t, p):
        return rates(state, p)

    return Pathway(
        name="Ketogenesis",
        metabolites=metabolites, enzymes=enzymes,
        ode_func=ode, flux_func=flux_func, params=params,
        notes="Hepatic mitochondrial pathway that converts surplus "
              "acetyl-CoA (generated when beta-oxidation outpaces the "
              "TCA cycle's capacity, e.g. during fasting or diabetic "
              "ketoacidosis) into water-soluble ketone bodies that other "
              "tissues (brain, heart, muscle) can use as fuel.",
    )


# ===========================================================================
# 9. UREA CYCLE
# ===========================================================================
def build_urea_cycle(protein_intake: float = 1.0) -> Pathway:
    """
    NH3 + CO2 --CPS-I--> Carbamoyl-P + Ornithine --OTC--> Citrulline
      --(+Aspartate) ASS--> Argininosuccinate --ASL--> Arginine + Fumarate
      --Arginase--> Urea + Ornithine (regenerated, closing the cycle)

    Key regulation:
      * Carbamoyl phosphate synthetase I (CPS-I): the committed,
        rate-limiting step. Absolutely requires the allosteric activator
        N-acetylglutamate (NAG); NAG synthesis is itself stimulated by
        arginine, so the cycle has a built-in feed-forward loop that
        upregulates itself when nitrogen (protein) load rises.
      * `protein_intake` (0-2, default 1) sets the ammonia (NH3) influx,
        mimicking dietary protein load / amino-acid catabolism.
    """
    metabolites = [
        Metabolite("CarbamoylP", 0.1, "a.u.", "carbamoyl phosphate"),
        Metabolite("Citrulline", 0.15, "a.u.", ""),
        Metabolite("Argininosuccinate", 0.1, "a.u.", ""),
        Metabolite("Arginine", 0.15, "a.u.", ""),
        Metabolite("Ornithine", 0.5, "a.u.", "regenerated each turn"),
        Metabolite("Urea", 0.2, "a.u.", "excreted via the kidney"),
    ]

    enzymes = [
        Enzyme("Carbamoyl phosphate synthetase I (CPS-I)",
               "NH3 + CO2 + 2 ATP -> Carbamoyl-P", "EC 6.3.4.16",
               "Committed, rate-limiting step",
               "Absolutely requires allosteric activator N-acetylglutamate "
               "(NAG); NAG synthesis is stimulated by arginine (feed-forward)",
               activators=["N-acetylglutamate", "Arginine"]),
        Enzyme("Ornithine transcarbamylase (OTC)",
               "Carbamoyl-P + Ornithine -> Citrulline", "EC 2.1.3.3",
               "Mitochondrial matrix step", "Constitutive"),
        Enzyme("Argininosuccinate synthetase (ASS)",
               "Citrulline + Aspartate + ATP -> Argininosuccinate", "EC 6.3.4.5",
               "Cytosolic, energetically costly step", "Constitutive"),
        Enzyme("Argininosuccinate lyase (ASL)",
               "Argininosuccinate -> Arginine + Fumarate", "EC 4.3.2.1",
               "Cleavage (fumarate feeds back into the TCA cycle)",
               "Constitutive"),
        Enzyme("Arginase", "Arginine + H2O -> Urea + Ornithine", "EC 3.5.3.1",
               "Regenerates ornithine, closing the cycle",
               "Constitutive; product ornithine re-enters mitochondria"),
    ]

    params = dict(
        nh3_input=0.5 * protein_intake,
        vmax_cps1=2.0, km_cps1=0.5, ka_arg=0.5, n_arg=2.0,
        vmax_otc=3.0, km_otc_cp=0.3, km_otc_orn=0.3,
        vmax_ass=2.5, km_ass=0.3,
        vmax_asl=3.0, km_asl=0.3,
        vmax_arginase=2.5, km_arginase=0.3,
        vmax_urea_export=1.2, km_urea_export=0.5,
    )

    def rates(state, p):
        CP, Cit, ASA, Arg, Orn, Urea = [max(v, 0.0) for v in state]

        v_cps1 = mm(p["nh3_input"] * 4.0, p["vmax_cps1"], p["km_cps1"]) * \
            (0.15 + 0.85 * hill_activation(Arg, p["ka_arg"], p["n_arg"]))
        v_otc = p["vmax_otc"] * (CP / (p["km_otc_cp"] + CP)) * \
            (Orn / (p["km_otc_orn"] + Orn))
        v_ass = mm(Cit, p["vmax_ass"], p["km_ass"])
        v_asl = mm(ASA, p["vmax_asl"], p["km_asl"])
        v_arginase = mm(Arg, p["vmax_arginase"], p["km_arginase"])
        v_urea_export = mm(Urea, p["vmax_urea_export"], p["km_urea_export"])

        return dict(CPS1=v_cps1, OTC=v_otc, ASS=v_ass, ASL=v_asl,
                     Arginase=v_arginase, Urea_export=v_urea_export)

    def ode(state, t, p):
        v = rates(state, p)
        CP, Cit, ASA, Arg, Orn, Urea = state
        dCP = v["CPS1"] - v["OTC"]
        dCit = v["OTC"] - v["ASS"]
        dASA = v["ASS"] - v["ASL"]
        dArg = v["ASL"] - v["Arginase"]
        dOrn = v["Arginase"] - v["OTC"]
        dUrea = v["Arginase"] - v["Urea_export"]
        return [dCP, dCit, dASA, dArg, dOrn, dUrea]

    def flux_func(state, t, p):
        return rates(state, p)

    return Pathway(
        name="Urea cycle",
        metabolites=metabolites, enzymes=enzymes,
        ode_func=ode, flux_func=flux_func, params=params,
        notes="Hepatic mitochondrial/cytosolic cycle that disposes of "
              "excess nitrogen from amino-acid catabolism as urea for "
              "renal excretion. Ornithine is regenerated every turn, "
              "making this -- like the TCA cycle -- a true cycle rather "
              "than a linear pathway. Fumarate produced by ASL directly "
              "feeds into the TCA cycle, linking the two.",
    )


# Convenience registry so main.py / visualization can iterate everything.
PATHWAY_BUILDERS = {
    "glycolysis": build_glycolysis,
    "gluconeogenesis": build_gluconeogenesis,
    "glycogenesis": build_glycogenesis,
    "glycogenolysis": build_glycogenolysis,
    "beta_oxidation": build_beta_oxidation,
    "tca_cycle": build_tca_cycle,
    "pentose_phosphate": build_pentose_phosphate,
    "ketogenesis": build_ketogenesis,
    "urea_cycle": build_urea_cycle,
}
