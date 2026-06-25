from __future__ import annotations

from pathlib import Path
import os
from typing import Dict, List

import joblib
import numpy as np
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"

FEATURE_GROUPS: Dict[str, List[str]] = {
    "Money Going Out": [
        "out_tx_count",
        "out_amt_sum",
        "out_amt_mean",
        "out_amt_std",
        "out_amt_max",
        "out_log_amt_mean",
        "out_amt_zscore_mean",
        "out_amt_zscore_max",
        "out_cross_border_rate",
        "out_curr_mismatch_rate",
        "out_above_1M_rate",
        "out_above_10M_rate",
        "out_velocity_mean",
        "out_velocity_max",
        "out_txcount10_mean",
        "out_txcount30_mean",
        "out_weekend_rate",
        "out_hour_mean",
        "out_hour_std",
        "out_n_counterparties",
        "out_transmode_A_rate",
        "out_transmode_B_rate",
        "out_transmode_E_rate",
        "out_transmode_F_rate",
        "out_transmode_J_rate",
        "out_transmode_P_rate",
        "out_transmode_Z_rate",
    ],
    "Money Coming In": [
        "in_tx_count",
        "in_amt_sum",
        "in_amt_mean",
        "in_amt_std",
        "in_amt_max",
        "in_log_amt_mean",
        "in_amt_zscore_mean",
        "in_amt_zscore_max",
        "in_cross_border_rate",
        "in_curr_mismatch_rate",
        "in_above_1M_rate",
        "in_above_10M_rate",
        "in_velocity_mean",
        "in_velocity_max",
        "in_txcount10_mean",
        "in_txcount30_mean",
        "in_weekend_rate",
        "in_hour_mean",
        "in_hour_std",
        "in_n_counterparties",
        "in_transmode_A_rate",
        "in_transmode_B_rate",
        "in_transmode_E_rate",
        "in_transmode_F_rate",
        "in_transmode_J_rate",
        "in_transmode_P_rate",
        "in_transmode_Z_rate",
    ],
    "Account Background": [
        "account_age_days",
        "country_risk",
        "is_person",
        "pep",
        "sanctions",
        "total_tx",
        "flow_ratio",
        "net_flow",
        "fan_out",
        "fan_in",
        "counterparty_total",
        "passthrough_score",
    ],
}

FEATURES = [feature for group in FEATURE_GROUPS.values() for feature in group]

DEFAULT_VALUES: Dict[str, float] = {
    feature: 0.0 for feature in FEATURES
}
DEFAULT_VALUES.update(
    {
        "account_age_days": 365.0,
        "country_risk": 0.2,
        "is_person": 1.0,
        "total_tx": 1.0,
        "out_hour_mean": 12.0,
        "in_hour_mean": 12.0,
    }
)

GROUP_DESCRIPTIONS: Dict[str, str] = {
    "Money Going Out": "Fill in what the account usually sends out: count, value, timing, and spread across recipients.",
    "Money Coming In": "Fill in what the account receives: count, value, timing, and spread across senders.",
    "Account Background": "Fill in basic customer risk details and overall movement pattern for the account.",
}

FIELD_META: Dict[str, Dict[str, str]] = {
    "out_tx_count": {"label": "Number of outgoing transactions", "hint": "Total outgoing transactions included in this review period."},
    "out_amt_sum": {"label": "Total outgoing amount", "hint": "Total money sent out during the period."},
    "out_amt_mean": {"label": "Average outgoing amount", "hint": "Average value of outgoing transactions."},
    "out_amt_std": {"label": "Variation in outgoing amounts", "hint": "Higher values mean the outgoing amounts vary a lot."},
    "out_amt_max": {"label": "Largest outgoing transaction", "hint": "Biggest single amount sent out."},
    "out_log_amt_mean": {"label": "Scaled average outgoing amount", "hint": "Model-ready version of average outgoing amount. If unknown, use the sample profile."},
    "out_amt_zscore_mean": {"label": "Average outgoing unusualness", "hint": "Higher means outgoing amounts are further from normal behavior."},
    "out_amt_zscore_max": {"label": "Most unusual outgoing transaction", "hint": "Highest unusualness level for a single outgoing transaction."},
    "out_cross_border_rate": {"label": "Share of outgoing cross-border transfers", "hint": "Enter a value from 0 to 1. Example: 0.25 means 25%."},
    "out_curr_mismatch_rate": {"label": "Share of outgoing currency mismatches", "hint": "Enter a value from 0 to 1 for how often transfer currency differs from expected currency."},
    "out_above_1M_rate": {"label": "Share of outgoing transfers above 1M", "hint": "Enter a value from 0 to 1."},
    "out_above_10M_rate": {"label": "Share of outgoing transfers above 10M", "hint": "Enter a value from 0 to 1."},
    "out_velocity_mean": {"label": "Average outgoing speed", "hint": "How much money moves out quickly across recent transactions."},
    "out_velocity_max": {"label": "Peak outgoing speed", "hint": "Highest burst of outgoing activity."},
    "out_txcount10_mean": {"label": "Average outgoing count in short window", "hint": "Average number of outgoing transactions in a recent small window."},
    "out_txcount30_mean": {"label": "Average outgoing count in wider window", "hint": "Average number of outgoing transactions in a larger recent window."},
    "out_weekend_rate": {"label": "Share of outgoing weekend activity", "hint": "Enter a value from 0 to 1."},
    "out_hour_mean": {"label": "Average outgoing hour", "hint": "Typical hour of day for outgoing transactions, using 0 to 23."},
    "out_hour_std": {"label": "Variation in outgoing time of day", "hint": "Higher means outgoing transfers happen at many different times."},
    "out_n_counterparties": {"label": "Number of outgoing counterparties", "hint": "How many different recipients received money."},
    "out_transmode_A_rate": {"label": "Outgoing channel A share", "hint": "Enter a value from 0 to 1 if your source system uses channel A."},
    "out_transmode_B_rate": {"label": "Outgoing channel B share", "hint": "Enter a value from 0 to 1 if your source system uses channel B."},
    "out_transmode_E_rate": {"label": "Outgoing channel E share", "hint": "Enter a value from 0 to 1 if your source system uses channel E."},
    "out_transmode_F_rate": {"label": "Outgoing channel F share", "hint": "Enter a value from 0 to 1 if your source system uses channel F."},
    "out_transmode_J_rate": {"label": "Outgoing channel J share", "hint": "Enter a value from 0 to 1 if your source system uses channel J."},
    "out_transmode_P_rate": {"label": "Outgoing channel P share", "hint": "Enter a value from 0 to 1 if your source system uses channel P."},
    "out_transmode_Z_rate": {"label": "Outgoing channel Z share", "hint": "Enter a value from 0 to 1 if your source system uses channel Z."},
    "in_tx_count": {"label": "Number of incoming transactions", "hint": "Total incoming transactions included in this review period."},
    "in_amt_sum": {"label": "Total incoming amount", "hint": "Total money received during the period."},
    "in_amt_mean": {"label": "Average incoming amount", "hint": "Average value of incoming transactions."},
    "in_amt_std": {"label": "Variation in incoming amounts", "hint": "Higher values mean the incoming amounts vary a lot."},
    "in_amt_max": {"label": "Largest incoming transaction", "hint": "Biggest single amount received."},
    "in_log_amt_mean": {"label": "Scaled average incoming amount", "hint": "Model-ready version of average incoming amount. If unknown, use the sample profile."},
    "in_amt_zscore_mean": {"label": "Average incoming unusualness", "hint": "Higher means incoming amounts are further from normal behavior."},
    "in_amt_zscore_max": {"label": "Most unusual incoming transaction", "hint": "Highest unusualness level for a single incoming transaction."},
    "in_cross_border_rate": {"label": "Share of incoming cross-border transfers", "hint": "Enter a value from 0 to 1. Example: 0.25 means 25%."},
    "in_curr_mismatch_rate": {"label": "Share of incoming currency mismatches", "hint": "Enter a value from 0 to 1 for how often received currency differs from expected currency."},
    "in_above_1M_rate": {"label": "Share of incoming transfers above 1M", "hint": "Enter a value from 0 to 1."},
    "in_above_10M_rate": {"label": "Share of incoming transfers above 10M", "hint": "Enter a value from 0 to 1."},
    "in_velocity_mean": {"label": "Average incoming speed", "hint": "How much money comes in quickly across recent transactions."},
    "in_velocity_max": {"label": "Peak incoming speed", "hint": "Highest burst of incoming activity."},
    "in_txcount10_mean": {"label": "Average incoming count in short window", "hint": "Average number of incoming transactions in a recent small window."},
    "in_txcount30_mean": {"label": "Average incoming count in wider window", "hint": "Average number of incoming transactions in a larger recent window."},
    "in_weekend_rate": {"label": "Share of incoming weekend activity", "hint": "Enter a value from 0 to 1."},
    "in_hour_mean": {"label": "Average incoming hour", "hint": "Typical hour of day for incoming transactions, using 0 to 23."},
    "in_hour_std": {"label": "Variation in incoming time of day", "hint": "Higher means incoming transfers happen at many different times."},
    "in_n_counterparties": {"label": "Number of incoming counterparties", "hint": "How many different senders transferred money in."},
    "in_transmode_A_rate": {"label": "Incoming channel A share", "hint": "Enter a value from 0 to 1 if your source system uses channel A."},
    "in_transmode_B_rate": {"label": "Incoming channel B share", "hint": "Enter a value from 0 to 1 if your source system uses channel B."},
    "in_transmode_E_rate": {"label": "Incoming channel E share", "hint": "Enter a value from 0 to 1 if your source system uses channel E."},
    "in_transmode_F_rate": {"label": "Incoming channel F share", "hint": "Enter a value from 0 to 1 if your source system uses channel F."},
    "in_transmode_J_rate": {"label": "Incoming channel J share", "hint": "Enter a value from 0 to 1 if your source system uses channel J."},
    "in_transmode_P_rate": {"label": "Incoming channel P share", "hint": "Enter a value from 0 to 1 if your source system uses channel P."},
    "in_transmode_Z_rate": {"label": "Incoming channel Z share", "hint": "Enter a value from 0 to 1 if your source system uses channel Z."},
    "account_age_days": {"label": "Account age in days", "hint": "How old the account is."},
    "country_risk": {"label": "Country risk score", "hint": "Use the institution's country risk score, usually from 0 to 1."},
    "is_person": {"label": "Account holder is a person", "hint": "Enter 1 for an individual person, 0 for a business or organization."},
    "pep": {"label": "PEP flag", "hint": "Enter 1 if the account holder is a politically exposed person, otherwise 0."},
    "sanctions": {"label": "Sanctions flag", "hint": "Enter 1 if the account holder appears on a sanctions list, otherwise 0."},
    "total_tx": {"label": "Total number of transactions", "hint": "Combined incoming and outgoing transaction count."},
    "flow_ratio": {"label": "Money flow balance", "hint": "Higher values can suggest strong movement between money in and money out."},
    "net_flow": {"label": "Net money movement", "hint": "Difference between money in and money out."},
    "fan_out": {"label": "Number of outgoing connections", "hint": "How many different recipients the account sends to."},
    "fan_in": {"label": "Number of incoming connections", "hint": "How many different senders transfer money in."},
    "counterparty_total": {"label": "Total counterparties", "hint": "Total unique senders and recipients connected to the account."},
    "passthrough_score": {"label": "Pass-through behavior score", "hint": "Higher values suggest money moves in and out quickly without staying long."},
}

FIELD_HINTS: Dict[str, str] = {field: meta["hint"] for field, meta in FIELD_META.items()}

SAMPLE_PROFILE: Dict[str, float] = {
    "out_tx_count": 24,
    "out_amt_sum": 9500000,
    "out_amt_mean": 395833.33,
    "out_amt_std": 210000,
    "out_amt_max": 1200000,
    "out_log_amt_mean": 12.7,
    "out_amt_zscore_mean": 1.9,
    "out_amt_zscore_max": 3.4,
    "out_cross_border_rate": 0.35,
    "out_curr_mismatch_rate": 0.2,
    "out_above_1M_rate": 0.25,
    "out_above_10M_rate": 0.0,
    "out_velocity_mean": 650000,
    "out_velocity_max": 1800000,
    "out_txcount10_mean": 3.1,
    "out_txcount30_mean": 7.4,
    "out_weekend_rate": 0.2,
    "out_hour_mean": 20,
    "out_hour_std": 5.2,
    "out_n_counterparties": 11,
    "out_transmode_A_rate": 0.05,
    "out_transmode_B_rate": 0.05,
    "out_transmode_E_rate": 0.3,
    "out_transmode_F_rate": 0.05,
    "out_transmode_J_rate": 0.2,
    "out_transmode_P_rate": 0.2,
    "out_transmode_Z_rate": 0.15,
    "in_tx_count": 19,
    "in_amt_sum": 8800000,
    "in_amt_mean": 463157.89,
    "in_amt_std": 170000,
    "in_amt_max": 1400000,
    "in_log_amt_mean": 12.5,
    "in_amt_zscore_mean": 1.5,
    "in_amt_zscore_max": 2.8,
    "in_cross_border_rate": 0.3,
    "in_curr_mismatch_rate": 0.15,
    "in_above_1M_rate": 0.2,
    "in_above_10M_rate": 0.0,
    "in_velocity_mean": 590000,
    "in_velocity_max": 1600000,
    "in_txcount10_mean": 2.8,
    "in_txcount30_mean": 6.9,
    "in_weekend_rate": 0.15,
    "in_hour_mean": 19,
    "in_hour_std": 4.8,
    "in_n_counterparties": 9,
    "in_transmode_A_rate": 0.05,
    "in_transmode_B_rate": 0.05,
    "in_transmode_E_rate": 0.25,
    "in_transmode_F_rate": 0.05,
    "in_transmode_J_rate": 0.25,
    "in_transmode_P_rate": 0.2,
    "in_transmode_Z_rate": 0.15,
    "account_age_days": 842,
    "country_risk": 0.65,
    "is_person": 1,
    "pep": 1,
    "sanctions": 0,
    "total_tx": 43,
    "flow_ratio": 0.93,
    "net_flow": 700000,
    "fan_out": 11,
    "fan_in": 9,
    "counterparty_total": 18,
    "passthrough_score": 0.99,
}


def load_model():
    return joblib.load(MODEL_PATH)


app = FastAPI(title="Risk Scoring Demo")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

MODEL = None
MODEL_ERROR = None

try:
    MODEL = load_model()
except Exception as exc:  # pragma: no cover - useful for local startup diagnostics
    MODEL_ERROR = str(exc)


def build_feature_array(form_values: Dict[str, float]) -> np.ndarray:
    row = [float(form_values.get(feature, DEFAULT_VALUES[feature])) for feature in FEATURES]
    return np.array([row], dtype=float)


def score_label(score: float) -> str:
    if score >= 0.8:
        return "High"
    if score >= 0.4:
        return "Medium"
    return "Low"


def risk_summary(score: float | None) -> str | None:
    if score is None:
        return None
    if score >= 0.8:
        return "This account should be reviewed first. The pattern looks strongly similar to previously flagged risky accounts."
    if score >= 0.4:
        return "This account deserves a closer look. Some behavior looks unusual, but it is not the strongest alert in the queue."
    return "This account currently looks lower risk. It may still need review if you have outside intelligence or case context."


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "feature_groups": FEATURE_GROUPS,
            "group_descriptions": GROUP_DESCRIPTIONS,
            "values": DEFAULT_VALUES,
            "field_hints": FIELD_HINTS,
            "field_meta": FIELD_META,
            "prediction": None,
            "risk_band": None,
            "risk_summary": None,
            "model_error": MODEL_ERROR,
            "sample_profile": SAMPLE_PROFILE,
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=int(os.getenv("PORT", "8080")), reload=True)


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    out_tx_count: float = Form(DEFAULT_VALUES["out_tx_count"]),
    out_amt_sum: float = Form(DEFAULT_VALUES["out_amt_sum"]),
    out_amt_mean: float = Form(DEFAULT_VALUES["out_amt_mean"]),
    out_amt_std: float = Form(DEFAULT_VALUES["out_amt_std"]),
    out_amt_max: float = Form(DEFAULT_VALUES["out_amt_max"]),
    out_log_amt_mean: float = Form(DEFAULT_VALUES["out_log_amt_mean"]),
    out_amt_zscore_mean: float = Form(DEFAULT_VALUES["out_amt_zscore_mean"]),
    out_amt_zscore_max: float = Form(DEFAULT_VALUES["out_amt_zscore_max"]),
    out_cross_border_rate: float = Form(DEFAULT_VALUES["out_cross_border_rate"]),
    out_curr_mismatch_rate: float = Form(DEFAULT_VALUES["out_curr_mismatch_rate"]),
    out_above_1M_rate: float = Form(DEFAULT_VALUES["out_above_1M_rate"]),
    out_above_10M_rate: float = Form(DEFAULT_VALUES["out_above_10M_rate"]),
    out_velocity_mean: float = Form(DEFAULT_VALUES["out_velocity_mean"]),
    out_velocity_max: float = Form(DEFAULT_VALUES["out_velocity_max"]),
    out_txcount10_mean: float = Form(DEFAULT_VALUES["out_txcount10_mean"]),
    out_txcount30_mean: float = Form(DEFAULT_VALUES["out_txcount30_mean"]),
    out_weekend_rate: float = Form(DEFAULT_VALUES["out_weekend_rate"]),
    out_hour_mean: float = Form(DEFAULT_VALUES["out_hour_mean"]),
    out_hour_std: float = Form(DEFAULT_VALUES["out_hour_std"]),
    out_n_counterparties: float = Form(DEFAULT_VALUES["out_n_counterparties"]),
    out_transmode_A_rate: float = Form(DEFAULT_VALUES["out_transmode_A_rate"]),
    out_transmode_B_rate: float = Form(DEFAULT_VALUES["out_transmode_B_rate"]),
    out_transmode_E_rate: float = Form(DEFAULT_VALUES["out_transmode_E_rate"]),
    out_transmode_F_rate: float = Form(DEFAULT_VALUES["out_transmode_F_rate"]),
    out_transmode_J_rate: float = Form(DEFAULT_VALUES["out_transmode_J_rate"]),
    out_transmode_P_rate: float = Form(DEFAULT_VALUES["out_transmode_P_rate"]),
    out_transmode_Z_rate: float = Form(DEFAULT_VALUES["out_transmode_Z_rate"]),
    in_tx_count: float = Form(DEFAULT_VALUES["in_tx_count"]),
    in_amt_sum: float = Form(DEFAULT_VALUES["in_amt_sum"]),
    in_amt_mean: float = Form(DEFAULT_VALUES["in_amt_mean"]),
    in_amt_std: float = Form(DEFAULT_VALUES["in_amt_std"]),
    in_amt_max: float = Form(DEFAULT_VALUES["in_amt_max"]),
    in_log_amt_mean: float = Form(DEFAULT_VALUES["in_log_amt_mean"]),
    in_amt_zscore_mean: float = Form(DEFAULT_VALUES["in_amt_zscore_mean"]),
    in_amt_zscore_max: float = Form(DEFAULT_VALUES["in_amt_zscore_max"]),
    in_cross_border_rate: float = Form(DEFAULT_VALUES["in_cross_border_rate"]),
    in_curr_mismatch_rate: float = Form(DEFAULT_VALUES["in_curr_mismatch_rate"]),
    in_above_1M_rate: float = Form(DEFAULT_VALUES["in_above_1M_rate"]),
    in_above_10M_rate: float = Form(DEFAULT_VALUES["in_above_10M_rate"]),
    in_velocity_mean: float = Form(DEFAULT_VALUES["in_velocity_mean"]),
    in_velocity_max: float = Form(DEFAULT_VALUES["in_velocity_max"]),
    in_txcount10_mean: float = Form(DEFAULT_VALUES["in_txcount10_mean"]),
    in_txcount30_mean: float = Form(DEFAULT_VALUES["in_txcount30_mean"]),
    in_weekend_rate: float = Form(DEFAULT_VALUES["in_weekend_rate"]),
    in_hour_mean: float = Form(DEFAULT_VALUES["in_hour_mean"]),
    in_hour_std: float = Form(DEFAULT_VALUES["in_hour_std"]),
    in_n_counterparties: float = Form(DEFAULT_VALUES["in_n_counterparties"]),
    in_transmode_A_rate: float = Form(DEFAULT_VALUES["in_transmode_A_rate"]),
    in_transmode_B_rate: float = Form(DEFAULT_VALUES["in_transmode_B_rate"]),
    in_transmode_E_rate: float = Form(DEFAULT_VALUES["in_transmode_E_rate"]),
    in_transmode_F_rate: float = Form(DEFAULT_VALUES["in_transmode_F_rate"]),
    in_transmode_J_rate: float = Form(DEFAULT_VALUES["in_transmode_J_rate"]),
    in_transmode_P_rate: float = Form(DEFAULT_VALUES["in_transmode_P_rate"]),
    in_transmode_Z_rate: float = Form(DEFAULT_VALUES["in_transmode_Z_rate"]),
    account_age_days: float = Form(DEFAULT_VALUES["account_age_days"]),
    country_risk: float = Form(DEFAULT_VALUES["country_risk"]),
    is_person: float = Form(DEFAULT_VALUES["is_person"]),
    pep: float = Form(DEFAULT_VALUES["pep"]),
    sanctions: float = Form(DEFAULT_VALUES["sanctions"]),
    total_tx: float = Form(DEFAULT_VALUES["total_tx"]),
    flow_ratio: float = Form(DEFAULT_VALUES["flow_ratio"]),
    net_flow: float = Form(DEFAULT_VALUES["net_flow"]),
    fan_out: float = Form(DEFAULT_VALUES["fan_out"]),
    fan_in: float = Form(DEFAULT_VALUES["fan_in"]),
    counterparty_total: float = Form(DEFAULT_VALUES["counterparty_total"]),
    passthrough_score: float = Form(DEFAULT_VALUES["passthrough_score"]),
):
    values = locals().copy()
    values.pop("request")

    prediction = None
    risk_band = None
    summary = None
    runtime_error = MODEL_ERROR

    if MODEL is not None:
        feature_array = build_feature_array(values)
        prediction = float(MODEL.predict_proba(feature_array)[0][1])
        risk_band = score_label(prediction)
        summary = risk_summary(prediction)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "feature_groups": FEATURE_GROUPS,
            "group_descriptions": GROUP_DESCRIPTIONS,
            "values": values,
            "field_hints": FIELD_HINTS,
            "field_meta": FIELD_META,
            "prediction": prediction,
            "risk_band": risk_band,
            "risk_summary": summary,
            "model_error": runtime_error,
            "sample_profile": SAMPLE_PROFILE,
        },
    )
