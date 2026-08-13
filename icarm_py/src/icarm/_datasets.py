"""Built-in synthetic datasets — mirrors R icarm datasets."""
from __future__ import annotations
import numpy as np
import pandas as pd


def _rng(seed: int = 2025) -> np.random.Generator:
    return np.random.default_rng(seed)


def load_medical(seed: int = 2025) -> pd.DataFrame:
    """
    Synthetic hospital readmission dataset (n=500).

    Mirrors R's icarm_medical.

    Columns
    -------
    age, gender, insurance, num_prior_visits, length_of_stay,
    glucose_level, bmi, num_medications, has_diabetes,
    has_hypertension, department, readmitted (outcome).

    Returns
    -------
    pd.DataFrame
    """
    rng = _rng(seed)
    n   = 500
    age = rng.integers(18, 90, n)
    return pd.DataFrame({
        "age"              : age,
        "gender"           : rng.choice(["Male", "Female"], n),
        "insurance"        : rng.choice(["Private", "Public",
                                          "None"], n,
                                         p=[0.5, 0.3, 0.2]),
        "num_prior_visits" : rng.integers(0, 15, n),
        "length_of_stay"   : rng.integers(1, 30, n),
        "glucose_level"    : rng.normal(120, 30, n).round(1),
        "bmi"              : rng.normal(27, 5, n).round(1),
        "num_medications"  : rng.integers(0, 20, n),
        "has_diabetes"     : rng.choice(["Yes", "No"], n,
                                         p=[0.3, 0.7]),
        "has_hypertension" : rng.choice(["Yes", "No"], n,
                                         p=[0.35, 0.65]),
        "department"       : rng.choice(
            ["Cardiology", "Oncology", "General", "Neurology"], n),
        "readmitted"       : rng.choice(["Yes", "No"], n,
                                         p=[0.27, 0.73]),
    })


def load_financial(seed: int = 2025) -> pd.DataFrame:
    """
    Synthetic loan default dataset (n=1000).

    Mirrors R's icarm_financial.

    Columns
    -------
    credit_score, income, loan_amount, debt_ratio,
    employment_years, num_accounts, late_payments,
    gender, ethnicity, age, area, default (outcome).

    Returns
    -------
    pd.DataFrame
    """
    rng = _rng(seed)
    n   = 1000
    cs  = rng.integers(300, 850, n)
    default_prob = 1 / (1 + np.exp((cs - 600) / 50))
    return pd.DataFrame({
        "credit_score"    : cs,
        "income"          : rng.integers(20000, 200000, n),
        "loan_amount"     : rng.integers(5000, 100000, n),
        "debt_ratio"      : rng.uniform(0.05, 0.8, n).round(3),
        "employment_years": rng.integers(0, 30, n),
        "num_accounts"    : rng.integers(1, 20, n),
        "late_payments"   : rng.integers(0, 10, n),
        "gender"          : rng.choice(["Male", "Female"], n),
        "ethnicity"       : rng.choice(
            ["GroupA", "GroupB", "GroupC"], n),
        "age"             : rng.integers(22, 75, n),
        "area"            : rng.choice(["Urban", "Rural"], n),
        "default"         : np.where(
            rng.uniform(size=n) < default_prob, "Yes", "No"),
    })


def load_racism_survey(seed: int = 2025) -> pd.DataFrame:
    """
    Synthetic racism impact survey dataset (n=150).

    Mirrors R's icarm_racism_survey.

    Columns
    -------
    age, gender, skin_color, income, education, region,
    social_support, media_exposure, political_orientation,
    employment, religion, language, neighbourhood_diversity,
    num_incidents, years_in_country, racism_impact (outcome, 0-10).

    Returns
    -------
    pd.DataFrame
    """
    rng = _rng(seed)
    n   = 150
    return pd.DataFrame({
        "age"                    : rng.integers(18, 80, n),
        "gender"                 : rng.choice(
            ["Male", "Female", "Other"], n),
        "skin_color"             : rng.choice(
            ["Light", "Medium", "Dark"], n),
        "income"                 : rng.integers(10000, 100000, n),
        "education"              : rng.choice(
            ["Primary", "Secondary", "Bachelor",
             "Postgraduate"], n),
        "region"                 : rng.choice(
            ["Urban", "Suburban", "Rural"], n),
        "social_support"         : rng.integers(1, 10, n),
        "media_exposure"         : rng.integers(1, 10, n),
        "political_orientation"  : rng.integers(0, 10, n),
        "employment"             : rng.choice(
            ["Employed", "Unemployed", "Student"], n),
        "religion"               : rng.choice(
            ["Christian", "Muslim", "Other", "None"], n),
        "language"               : rng.choice(
            ["Fluent", "Intermediate", "Basic"], n),
        "neighbourhood_diversity": rng.integers(1, 10, n),
        "num_incidents"          : rng.integers(0, 20, n),
        "years_in_country"       : rng.integers(0, 50, n),
        "racism_impact"          : rng.uniform(0, 10, n).round(1),
    })
