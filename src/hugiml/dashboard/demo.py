"""Bundled demo data for no-argument dashboard mode."""

from __future__ import annotations

import numpy as np
import pandas as pd


def load_demo_credit_risk(n: int = 800, random_state: int = 2026) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(random_state)
    income = rng.normal(65000, 22000, n).clip(18000, 180000)
    loan_amount = rng.normal(22000, 9000, n).clip(1000, 75000)
    utilization = rng.beta(2.2, 2.8, n)
    delinquency_count = rng.poisson(0.7, n).clip(0, 6)
    credit_history_months = rng.gamma(4.0, 18.0, n).clip(1, 240)
    employment_status = rng.choice(["permanent", "temporary", "self_employed", "unemployed"], n, p=[0.58, 0.18, 0.18, 0.06])
    region = rng.choice(["R1", "R2", "R3", "R4"], n, p=[0.34, 0.29, 0.22, 0.15])
    age_band = rng.choice(["young", "middle", "senior"], n, p=[0.32, 0.50, 0.18])

    logit = (
        -2.1
        + 2.7 * utilization
        + 0.42 * delinquency_count
        + 0.000018 * loan_amount
        - 0.000012 * income
        - 0.004 * credit_history_months
        + (employment_status == "temporary") * 0.45
        + (employment_status == "unemployed") * 0.95
        + (age_band == "young") * 0.20
        + (region == "R3") * 0.22
    )
    p = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, p)

    X = pd.DataFrame({
        "application_id": [f"APP-{i+1:06d}" for i in range(n)],
        "customer_name": [f"Customer {i+1:04d}" for i in range(n)],
        "income": income.round(2),
        "loan_amount": loan_amount.round(2),
        "loan_to_income": (loan_amount / income).round(4),
        "utilization": utilization.round(4),
        "delinquency_count": delinquency_count,
        "credit_history_months": credit_history_months.round(1),
        "employment_status": employment_status,
        "region": region,
        "age_band": age_band,
    })

    miss = rng.random(n) < 0.08
    X.loc[miss, "credit_history_months"] = np.nan
    miss = rng.random(n) < 0.05
    X.loc[miss, "employment_status"] = np.nan

    return X, y.astype(int)


def demo_roles() -> dict:
    return {
        "target": "default_risk",
        "id_column": "application_id",
        "excluded_columns": ["customer_name"],
        "sensitive_columns": ["age_band", "region"],
    }
