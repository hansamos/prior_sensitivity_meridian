"""The notebook's 15 ROI LogNormal scenarios (cell 11), kept in one place for the Phase 3 scripts."""
BASE_MU, BASE_SIGMA = 0.2, 0.9

# (scenario, mu, sigma, family) -- copied from notebook cell 11
SCENARIOS = [
    ("S01_sigma_0.40", 0.2, 0.40, "scale"),
    ("S02_sigma_0.55", 0.2, 0.55, "scale"),
    ("S03_sigma_0.70", 0.2, 0.70, "scale"),
    ("S04_sigma_1.10", 0.2, 1.10, "scale"),
    ("S05_sigma_1.30", 0.2, 1.30, "scale"),
    ("S06_mu_-0.20", -0.2, 0.90, "location"),
    ("S07_mu_0.00", 0.0, 0.90, "location"),
    ("S08_mu_0.40", 0.4, 0.90, "location"),
    ("S09_mu_0.60", 0.6, 0.90, "location"),
    ("S10_mu_0.80", 0.8, 0.90, "location"),
    ("S11_joint_0.0_0.6", 0.0, 0.60, "joint"),
    ("S12_joint_0.4_0.6", 0.4, 0.60, "joint"),
    ("S13_joint_0.6_0.6", 0.6, 0.60, "joint"),
    ("S14_stress_high", 0.8, 0.35, "stress"),
    ("S15_stress_low", -0.4, 0.35, "stress"),
]

REFIT_SEED_BASE = 1000  # refit i (1-based) uses seed 1000 + i; the reference fit used 124
