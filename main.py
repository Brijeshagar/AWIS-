"""
AWIS - Adaptive Workflow Intervention System  |  FastAPI Backend
================================================================
POST /predict  -> rejection risk score, confidence, top SHAP-based features
GET  /schema   -> exact request schema + valid example JSON
GET  /health   -> liveness check
GET  /features -> feature list + top importances
GET  /         -> API overview

Run:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import json
import logging
import pickle
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator
from sqlalchemy.orm import Session
import uvicorn

from database import AuditLog, Submission, get_db, init_db

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("awis")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR     = Path(__file__).parent
MODEL_PATH   = BASE_DIR / "model.pkl"
FI_JSON_PATH = BASE_DIR / "feature_importance.json"

# ---------------------------------------------------------------------------
# Feature order MUST match training exactly
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    "source_enc", "permit_type_enc", "zone_enc", "city_enc",
    "area_sqm", "construction_cost", "doc_count", "missing_doc_count",
    "has_title_deed", "has_site_plan", "has_noc_fire", "has_noc_env",
    "has_struct_cert", "has_crz_cert", "pan_valid",
    "zone_type_conflict", "area_exceeds_fsi", "risk_score", "year", "month",
]

# ---------------------------------------------------------------------------
# App lifespan: load model once at startup, release at shutdown
# ---------------------------------------------------------------------------
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    if not MODEL_PATH.exists():
        raise RuntimeError(f"model.pkl not found at {MODEL_PATH}. Run train_model.py first.")

    with open(MODEL_PATH, "rb") as f:
        _state["model"] = pickle.load(f)

    if FI_JSON_PATH.exists():
        with open(FI_JSON_PATH, "r") as f:
            _state["feature_importance"] = json.load(f)
    else:
        _state["feature_importance"] = {}

    log.info("[AWIS] Model loaded from %s", MODEL_PATH)
    log.info("[AWIS] Top feature: %s", next(iter(_state["feature_importance"]), "N/A"))

    # Initialise database (creates tables if they don't exist)
    init_db()

    yield  # --- App is running ---

    # --- Shutdown ---
    _state.clear()
    log.info("[AWIS] Shutdown complete.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AWIS - Adaptive Workflow Intervention System",
    description=(
        "Predicts rejection risk for permit/license applications using a tuned "
        "XGBoost model trained on 486k historical records.\n\n"
        "- **risk_score**: 0-100 (higher = higher rejection risk)\n"
        "- **confidence**: probability of the predicted class\n"
        "- **top_contributing_features**: SHAP-based per-prediction drivers"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

# Valid example — copy-paste this into POST /predict body
# NOTE: risk_score is NOT in the request — computed server-side automatically.
EXAMPLE_INPUT = {
    "source_enc": 1,
    "permit_type_enc": 3,
    "zone_enc": 2,
    "city_enc": 1,
    "area_sqm": 850.0,
    "construction_cost": 12500.0,
    "doc_count": 4,
    "missing_doc_count": 2,
    "has_title_deed": 1,
    "has_site_plan": 0,
    "has_noc_fire": 0,
    "has_noc_env": 0,
    "has_struct_cert": 0,
    "has_crz_cert": 0,
    "pan_valid": 1,
    "zone_type_conflict": 1,
    "area_exceeds_fsi": 1,
    "year": 2024,
    "month": 6,
}

# Fields computed server-side — stripped from client requests and DB form_data
_COMPUTED_FIELDS: set[str] = {"risk_score"}


class ApplicationInput(BaseModel):
    """
    19 user-supplied features.  `risk_score` (the 20th training feature)
    is computed server-side via _compute_risk_tier() and must NOT be sent.
    """
    # ── Categorical (encoded integers) ──────────────────────────────────
    source_enc:        int   = Field(..., ge=0, description="Application source (0=Online, 1=District, 2=Municipal, 3=State)")
    permit_type_enc:   int   = Field(..., ge=0, description="Permit type (0=Residential … 5=Agricultural)")
    zone_enc:          int   = Field(..., ge=0, description="Zone (0=Residential … 5=Special)")
    city_enc:          int   = Field(..., ge=0, description="City/district code")

    # ── Numeric ──────────────────────────────────────────────────────────
    area_sqm:          float = Field(..., ge=0, description="Plot area in sq. metres (>= 0)")
    construction_cost: float = Field(..., ge=0, description="Estimated construction cost")
    doc_count:         int   = Field(..., ge=0, description="Total documents submitted")
    missing_doc_count: int   = Field(..., ge=0, description="Documents missing / incomplete")

    # ── Binary document flags (0 or 1) ───────────────────────────────────
    has_title_deed:    int   = Field(..., ge=0, le=1, description="Title deed present")
    has_site_plan:     int   = Field(..., ge=0, le=1, description="Site plan present")
    has_noc_fire:      int   = Field(..., ge=0, le=1, description="Fire NOC obtained")
    has_noc_env:       int   = Field(..., ge=0, le=1, description="Environmental NOC obtained")
    has_struct_cert:   int   = Field(..., ge=0, le=1, description="Structural certificate present")
    has_crz_cert:      int   = Field(..., ge=0, le=1, description="CRZ certificate present")
    pan_valid:         int   = Field(..., ge=0, le=1, description="PAN number valid")

    # ── Compliance flags ──────────────────────────────────────────────────
    zone_type_conflict: int  = Field(..., ge=0, le=1, description="Zone type conflict detected")
    area_exceeds_fsi:   int  = Field(..., ge=0, le=1, description="Plot area exceeds FSI limit")

    # ── Time only (risk_score is server-computed, never from client) ──────
    year:  int = Field(..., ge=1990, le=2100, description="Year of application")
    month: int = Field(..., ge=1,   le=12,   description="Month (1-12)")

    model_config = {"json_schema_extra": {"example": EXAMPLE_INPUT}}

    @model_validator(mode="before")
    @classmethod
    def coerce_and_strip(cls, values):
        """
        1. Drop `risk_score` silently if a client sends it (computed server-side).
        2. Auto-coerce numeric strings from HTML form submissions.
        """
        coerced = {}
        for k, v in values.items():
            if k in _COMPUTED_FIELDS:
                continue          # strip — never trust client-supplied risk_score
            if isinstance(v, str):
                try:
                    v = float(v) if "." in v else int(v)
                except ValueError:
                    pass
            coerced[k] = v
        return coerced


class ContributingFeature(BaseModel):
    feature: str   = Field(..., description="Feature name")
    shap_value: float = Field(..., description="SHAP value (log-odds contribution)")
    impact: str    = Field(..., description="'increases_risk' or 'decreases_risk'")


class PredictionResponse(BaseModel):
    rejection_risk_score:      float = Field(..., description="0-100: likelihood of rejection (higher = riskier)")
    rejection_probability:     float = Field(..., description="Raw probability of rejection [0, 1]")
    confidence:                float = Field(..., description="Probability of the predicted class [0, 1]")
    prediction:                str   = Field(..., description="'Rejected' or 'Approved'")
    top_contributing_features: list[ContributingFeature] = Field(
        ..., description="Top 5 features driving this prediction (SHAP-based)"
    )


# ---------------------------------------------------------------------------
# Custom 422 handler — logs field-level errors clearly
# ---------------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    friendly = []
    for e in errors:
        loc   = " -> ".join(str(l) for l in e["loc"])
        msg   = e["msg"]
        typ   = e["type"]
        input_val = e.get("input", "<missing>")
        friendly.append({"field": loc, "error": msg, "type": typ, "received": str(input_val)})

    log.error("[422] Validation failed for POST %s", request.url.path)
    for f in friendly:
        log.error("  Field: %-30s | Error: %s | Got: %s", f["field"], f["error"], f["received"])

    return JSONResponse(
        status_code=422,
        content={
            "error":   "Request validation failed",
            "detail":  friendly,
            "hint":    "See GET /schema for the exact expected JSON format.",
            "example": EXAMPLE_INPUT,
        },
    )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _get_model():
    model = _state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Server is initializing.")
    return model


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", tags=["Info"], summary="API Overview")
def root():
    """Returns a brief description and list of available endpoints."""
    return {
        "service":     "AWIS Rejection Prediction API",
        "version":     "1.0.0",
        "model":       "XGBoost (tuned, ROC-AUC 0.9715)",
        "trained_on":  "486,090 historical permit applications",
        "endpoints": {
            "POST /predict":  "Predict rejection risk for an application",
            "GET  /features": "List all expected input features + top importances",
            "GET  /health":   "Liveness / readiness check",
            "GET  /docs":     "Swagger UI (interactive API docs)",
            "GET  /redoc":    "ReDoc API documentation",
        },
    }


@app.get("/health", tags=["Info"], summary="Health Check")
def health():
    """Returns model load status. Use for readiness probes."""
    model_loaded = "model" in _state
    return {
        "status":       "ok" if model_loaded else "error",
        "model_loaded": model_loaded,
        "feature_count": len(FEATURE_NAMES),
    }


@app.get("/schema", tags=["Info"], summary="Request Schema + Example")
def get_schema():
    """
    Returns the exact JSON schema for POST /predict,
    a valid example body, and field-level descriptions.
    """
    schema = ApplicationInput.model_json_schema()
    return {
        "endpoint":        "POST /predict",
        "content_type":    "application/json",
        "required_fields": schema.get("required", FEATURE_NAMES),
        "field_count":     len(FEATURE_NAMES),
        "field_order":     FEATURE_NAMES,
        "properties":      schema.get("properties", {}),
        "valid_example":   EXAMPLE_INPUT,
        "notes": [
            "All 20 fields are required.",
            "Binary flags (has_*, zone_type_conflict, area_exceeds_fsi) must be 0 or 1.",
            "area_sqm and construction_cost accept 0 if not yet known.",
            "year must be 1990-2100; month must be 1-12.",
            "Integer fields do not accept null - use 0 as default.",
        ],
    }


@app.get("/features", tags=["Info"], summary="Feature List & Importances")
def get_features():
    """Returns ordered feature list and top-10 importance scores."""
    fi = _state.get("feature_importance", {})
    return {
        "feature_count": len(FEATURE_NAMES),
        "features":      FEATURE_NAMES,
        "top_10_by_gain": dict(list(fi.items())[:10]),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _risk_to_prediction(risk_score: float) -> str:
    """Derive 'Rejected'/'Approved' from a stored risk_score (threshold = 50)."""
    return "Rejected" if risk_score >= 50.0 else "Approved"


def _compute_risk_tier(d: dict) -> int:
    """
    Derive the `risk_score` training feature (0-10) from user-provided inputs.
    This replaces the pre-computed column that existed in the raw training CSV.

    Scoring:
      missing_doc_count  : 1 pt each, capped at 4
      zone_type_conflict : 2 pts
      area_exceeds_fsi   : 2 pts
      no title deed      : 1 pt
      no site plan       : 1 pt  (total max = 10)
    """
    score  = min(int(d.get("missing_doc_count", 0)), 4)
    score += 2 * int(d.get("zone_type_conflict", 0))
    score += 2 * int(d.get("area_exceeds_fsi",   0))
    score += 1 - int(d.get("has_title_deed",      1))
    score += 1 - int(d.get("has_site_plan",        1))
    return min(score, 10)


# Fields to exclude from stored form_data (computed/internal — not user inputs)
_FORM_DATA_EXCLUDE = _COMPUTED_FIELDS  # {"risk_score"}


def _clean_form_data(raw: dict) -> dict:
    """
    Return a copy of the user payload with all server-computed or
    internal fields removed before persisting to the DB.
    Prevents confusion between input `risk_score` (tier 0-10) and
    output `rejection_risk_score` (probability 0-100).
    """
    return {k: v for k, v in raw.items() if k not in _FORM_DATA_EXCLUDE}


# Label maps (mirrors mock_portal.html dropdown values)
_SOURCE_LABELS = {0:"Online Portal",1:"District Office",2:"Municipal Corporation",3:"State Agency"}
_PERMIT_LABELS = {0:"Residential",1:"Commercial",2:"Industrial",3:"Mixed Use",4:"Infrastructure",5:"Agricultural"}
_ZONE_LABELS   = {0:"Residential",1:"Commercial",2:"Industrial",3:"Agricultural",4:"Mixed-Use",5:"Special/Eco"}
_CITY_LABELS   = {0:"Mumbai",1:"Delhi",2:"Bangalore",3:"Chennai",4:"Kolkata",5:"Hyderabad",6:"Pune",7:"Ahmedabad"}


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Predict Rejection Risk",
)
def predict(
    application: ApplicationInput,
    db: Session = Depends(get_db),
):
    """
    Accepts all 20 application features and returns:
    - **rejection_risk_score**: 0-100 scaled rejection likelihood
    - **rejection_probability**: raw model output probability
    - **confidence**: probability of the predicted class
    - **prediction**: "Rejected" or "Approved"
    - **top_contributing_features**: top 5 SHAP drivers (no extra library needed)

    Each call is persisted to the `submissions` table in awis.db.
    **422 errors?** -> GET /schema for exact format + example.
    """
    model = _get_model()

    # user payload (19 fields — risk_score absent)
    input_dict = application.model_dump()

    # Compute the 20th training feature server-side; inject for model only
    input_dict["risk_score"] = _compute_risk_tier(input_dict)

    input_df = pd.DataFrame(
        [[input_dict[f] for f in FEATURE_NAMES]],
        columns=FEATURE_NAMES,
    )
    log.info("[/predict] risk_tier=%s  fields=%s",
             input_dict["risk_score"],
             {k: input_dict[k] for k in list(input_dict)[:5]})

    # ── Probabilities & prediction ──────────────────────────────────────
    proba          = model.predict_proba(input_df)[0]
    rejection_prob = float(proba[1])
    prediction_int = int(model.predict(input_df)[0])
    prediction_str = "Rejected" if prediction_int == 1 else "Approved"
    confidence           = rejection_prob if prediction_int == 1 else float(proba[0])
    rejection_risk_score = round(rejection_prob * 100, 2)

    # ── SHAP via XGBoost built-in ─────────────────────────────────────────
    dmatrix     = xgb.DMatrix(input_df)
    shap_matrix = model.get_booster().predict(dmatrix, pred_contribs=True)
    shap_vals   = shap_matrix[0, :-1]

    ranked = sorted(
        zip(FEATURE_NAMES, shap_vals.tolist()),
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    top_features = [
        ContributingFeature(
            feature=name,
            shap_value=round(float(val), 6),
            impact="increases_risk" if val > 0 else "decreases_risk",
        )
        for name, val in ranked[:5]
    ]

    # ── Persist to DB ───────────────────────────────────────────────────
    try:
        sub = Submission(
            risk_score=rejection_risk_score,
            source=_SOURCE_LABELS.get(input_dict.get("source_enc"), "Unknown"),
            city=_CITY_LABELS.get(input_dict.get("city_enc"), "Unknown"),
            permit_type=_PERMIT_LABELS.get(input_dict.get("permit_type_enc"), "Unknown"),
            zone=_ZONE_LABELS.get(input_dict.get("zone_enc"), "Unknown"),
        )
        sub.set_form_data(_clean_form_data(input_dict))  # strip computed fields
        sub.set_top_reasons([f.model_dump() for f in top_features])
        db.add(sub)
        db.flush()   # get sub.id without committing yet

        audit = AuditLog(
            submission_id=sub.id,
            event_type="submission_created",
            new_value=prediction_str,
            changed_by="api",
        )
        db.add(audit)
        db.commit()
        log.info("[DB] Saved submission id=%s risk=%.1f", sub.id, rejection_risk_score)
    except Exception as db_err:
        db.rollback()
        log.warning("[DB] Could not save submission: %s", db_err)
        # Non-fatal: prediction is still returned to client

    return PredictionResponse(
        rejection_risk_score=rejection_risk_score,
        rejection_probability=round(rejection_prob, 6),
        confidence=round(confidence, 6),
        prediction=prediction_str,
        top_contributing_features=top_features,
    )


# ---------------------------------------------------------------------------
# Response model for a saved submission record
# ---------------------------------------------------------------------------
class SubmissionRecord(BaseModel):
    id:            int
    risk_score:    float
    prediction:    str
    confidence:    float
    top_reasons:   list
    submitted_at:  str          # ISO-8601 UTC string
    source:        str | None
    city:          str | None
    permit_type:   str | None
    zone:          str | None
    form_data:     dict


# ---------------------------------------------------------------------------
# POST /submit  — run prediction + persist, return full saved record
# ---------------------------------------------------------------------------
@app.post(
    "/submit",
    response_model=SubmissionRecord,
    tags=["Submission"],
    summary="Submit Application & Save to DB",
    status_code=201,
)
def submit(
    application: ApplicationInput,
    db: Session = Depends(get_db),
):
    """
    Runs the full prediction pipeline on the supplied form data,
    persists the result to the `submissions` table, and returns
    the saved record including its `id` and `submitted_at` timestamp.

    Differences from `POST /predict`:
    - Returns the **database record** (id, timestamps, serialised JSON fields)
    - Designed for explicit save actions (e.g. form submit button click)
    - HTTP 201 Created on success
    """
    model = _get_model()

    # ── Build feature DataFrame (inject server-computed risk_score) ──────
    input_dict = application.model_dump()
    input_dict["risk_score"] = _compute_risk_tier(input_dict)

    input_df = pd.DataFrame(
        [[input_dict[f] for f in FEATURE_NAMES]],
        columns=FEATURE_NAMES,
    )
    log.info("[/submit] risk_tier=%s  user_fields=%d",
             input_dict["risk_score"], len(application.model_dump()))

    # ── Model inference ──────────────────────────────────────────────────
    proba          = model.predict_proba(input_df)[0]   # [p_approved, p_rejected]
    rejection_prob = float(proba[1])
    prediction_int = int(model.predict(input_df)[0])
    prediction_str = "Rejected" if prediction_int == 1 else "Approved"
    confidence     = rejection_prob if prediction_int == 1 else float(proba[0])
    risk_score     = round(rejection_prob * 100, 2)

    # ── SHAP contributions ───────────────────────────────────────────────
    dmatrix     = xgb.DMatrix(input_df)
    shap_matrix = model.get_booster().predict(dmatrix, pred_contribs=True)
    shap_vals   = shap_matrix[0, :-1]   # drop bias term

    ranked = sorted(
        zip(FEATURE_NAMES, shap_vals.tolist()),
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    top_reasons = [
        {
            "feature":    name,
            "shap_value": round(float(val), 6),
            "impact":     "increases_risk" if val > 0 else "decreases_risk",
        }
        for name, val in ranked[:5]
    ]

    # ── Persist to DB ────────────────────────────────────────────────────
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)

    try:
        sub = Submission(
            risk_score=risk_score,
            submitted_at=now,
            source=_SOURCE_LABELS.get(input_dict.get("source_enc"), "Unknown"),
            city=_CITY_LABELS.get(input_dict.get("city_enc"), "Unknown"),
            permit_type=_PERMIT_LABELS.get(input_dict.get("permit_type_enc"), "Unknown"),
            zone=_ZONE_LABELS.get(input_dict.get("zone_enc"), "Unknown"),
        )
        sub.set_form_data(_clean_form_data(input_dict))  # strip computed fields → JSON text
        sub.set_top_reasons(top_reasons)                 # → JSON text
        db.add(sub)
        db.flush()   # populate sub.id before the audit row

        db.add(AuditLog(
            submission_id=sub.id,
            event_type="submission_created",
            new_value=json.dumps({
                "prediction": prediction_str,
                "risk_score": risk_score,
            }),
            changed_by="api:/submit",
        ))
        db.commit()
        db.refresh(sub)   # reload to get server-generated defaults
        log.info("[/submit] Saved submission id=%s  risk=%.1f  pred=%s",
                 sub.id, risk_score, prediction_str)

    except Exception as exc:
        db.rollback()
        log.error("[/submit] DB save failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction succeeded but DB save failed: {exc}",
        )

    # ── Build response from saved record ─────────────────────────────────
    return SubmissionRecord(
        id=sub.id,
        risk_score=sub.risk_score,
        prediction=prediction_str,
        confidence=round(confidence, 6),
        top_reasons=sub.get_top_reasons(),      # deserialized list
        submitted_at=sub.submitted_at.isoformat(),
        source=sub.source,
        city=sub.city,
        permit_type=sub.permit_type,
        zone=sub.zone,
        form_data=sub.get_form_data(),          # deserialized dict
    )


class SubmissionListResponse(BaseModel):
    total:   int
    limit:   int
    offset:  int
    results: list[SubmissionRecord]


@app.get(
    "/submissions",
    response_model=SubmissionListResponse,
    tags=["Data"],
    summary="List All Submissions",
)
def list_submissions(
    limit:  int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    """
    Returns all submissions sorted by **risk_score descending** (highest risk first).

    - `limit`  : max rows to return (default 50, max 500)
    - `offset` : pagination offset (default 0)

    JSON fields (`form_data`, `top_reasons`) are returned as parsed objects,
    not raw strings.
    """
    limit = min(limit, 500)  # hard cap

    rows  = (
        db.query(Submission)
        .order_by(Submission.risk_score.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    total = db.query(Submission).count()

    results = [
        SubmissionRecord(
            id=r.id,
            risk_score=r.risk_score,
            prediction=_risk_to_prediction(r.risk_score),
            confidence=round(r.risk_score / 100, 6),
            top_reasons=r.get_top_reasons(),       # deserialized list
            submitted_at=(
                r.submitted_at.isoformat() if r.submitted_at else ""
            ),
            source=r.source,
            city=r.city,
            permit_type=r.permit_type,
            zone=r.zone,
            form_data=r.get_form_data(),           # deserialized dict
        )
        for r in rows
    ]

    return SubmissionListResponse(
        total=total,
        limit=limit,
        offset=offset,
        results=results,
    )


@app.get(
    "/submissions/{submission_id}",
    response_model=SubmissionRecord,
    tags=["Data"],
    summary="Get Single Submission",
)
def get_submission(
    submission_id: int,
    db: Session = Depends(get_db),
):
    """
    Returns a single submission by its database `id`.

    - JSON fields (`form_data`, `top_reasons`) are returned as parsed objects.
    - Raises **404** if the id does not exist.
    - Raises **422** if `id` is not a valid integer.
    """
    row = db.query(Submission).filter(Submission.id == submission_id).first()

    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"Submission id={submission_id} not found.",
        )

    return SubmissionRecord(
        id=row.id,
        risk_score=row.risk_score,
        prediction=_risk_to_prediction(row.risk_score),
        confidence=round(row.risk_score / 100, 6),
        top_reasons=row.get_top_reasons(),         # deserialized list
        submitted_at=(
            row.submitted_at.isoformat() if row.submitted_at else ""
        ),
        source=row.source,
        city=row.city,
        permit_type=row.permit_type,
        zone=row.zone,
        form_data=row.get_form_data(),             # deserialized dict
    )


# ===========================================================================
# Rules CRUD  — /rules
# ===========================================================================

# ── Pydantic models ─────────────────────────────────────────────────────────

class RuleRecord(BaseModel):
    """Response model for a single rule row."""
    id:          int
    rule_name:   str
    rule_type:   str
    zone:        str | None
    permit_type: str | None
    condition:   str
    value:       str
    active:      bool
    created_at:  str
    updated_at:  str


class RuleCreate(BaseModel):
    """Request body for POST /rules."""
    rule_name:   str  = Field(..., min_length=1, max_length=200,
                               description="Unique human-readable name for this rule")
    rule_type:   str  = Field(..., min_length=1, max_length=100,
                               description="Category: threshold | flag_check | document_check | zone_permit_mismatch")
    zone:        str | None = Field(None, max_length=100,
                               description="Zone this rule applies to (null = all zones)")
    permit_type: str | None = Field(None, max_length=100,
                               description="Permit type this rule applies to (null = all types)")
    condition:   str  = Field(..., min_length=1, max_length=200,
                               description="Operator/expression: >, ==, match, missing …")
    value:       str  = Field(..., min_length=1, max_length=200,
                               description="Threshold or comparison value")
    active:      bool = Field(True, description="Enable / disable the rule")

    model_config = {
        "json_schema_extra": {
            "example": {
                "rule_name":   "high_construction_cost",
                "rule_type":   "threshold",
                "zone":        None,
                "permit_type": "Industrial",
                "condition":   ">",
                "value":       "5000000",
                "active":      True,
            }
        }
    }


class RuleUpdate(BaseModel):
    """Request body for PUT /rules/{id} — all fields optional (partial update)."""
    rule_name:   str | None = Field(None, min_length=1, max_length=200)
    rule_type:   str | None = Field(None, min_length=1, max_length=100)
    zone:        str | None = None
    permit_type: str | None = None
    condition:   str | None = Field(None, min_length=1, max_length=200)
    value:       str | None = Field(None, min_length=1, max_length=200)
    active:      bool | None = None


# ── Helper ───────────────────────────────────────────────────────────────────

def _rule_to_record(r) -> RuleRecord:
    """Convert a SQLAlchemy Rule row to a RuleRecord Pydantic model."""
    from database import Rule  # local import keeps top-level clean
    return RuleRecord(
        id=r.id,
        rule_name=r.rule_name,
        rule_type=r.rule_type,
        zone=r.zone,
        permit_type=r.permit_type,
        condition=r.condition,
        value=r.value,
        active=r.active,
        created_at=r.created_at.isoformat() if r.created_at else "",
        updated_at=r.updated_at.isoformat() if r.updated_at else "",
    )


# ── GET /rules ───────────────────────────────────────────────────────────────

@app.get(
    "/rules",
    response_model=list[RuleRecord],
    tags=["Rules"],
    summary="List Rules",
)
def list_rules(
    active_only: bool = False,
    rule_type:   str | None = None,
    db: Session = Depends(get_db),
):
    """
    Returns all rules stored in the database, ordered by id.

    Optional query params:
    - `active_only=true`  — return only enabled rules
    - `rule_type=<type>`  — filter by rule category
    """
    from database import Rule
    q = db.query(Rule)
    if active_only:
        q = q.filter(Rule.active == True)
    if rule_type:
        q = q.filter(Rule.rule_type == rule_type)
    return [_rule_to_record(r) for r in q.order_by(Rule.id).all()]


# ── GET /rules/{id} ──────────────────────────────────────────────────────────

@app.get(
    "/rules/{rule_id}",
    response_model=RuleRecord,
    tags=["Rules"],
    summary="Get Single Rule",
)
def get_rule(rule_id: int, db: Session = Depends(get_db)):
    """Returns a single rule by its database `id`. Raises 404 if not found."""
    from database import Rule
    row = db.query(Rule).filter(Rule.id == rule_id).first()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Rule id={rule_id} not found.")
    return _rule_to_record(row)


# ── POST /rules ───────────────────────────────────────────────────────────────

@app.post(
    "/rules",
    response_model=RuleRecord,
    tags=["Rules"],
    summary="Create Rule",
    status_code=201,
)
def create_rule(body: RuleCreate, db: Session = Depends(get_db)):
    """
    Creates a new rule and persists it to the database.

    - `rule_name` must be **unique** across all rules.
    - Returns the created record with its assigned `id` and timestamps.
    """
    from database import Rule
    from datetime import datetime, timezone

    # Uniqueness guard
    existing = db.query(Rule).filter(Rule.rule_name == body.rule_name).first()
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"A rule named '{body.rule_name}' already exists (id={existing.id}).",
        )

    now = datetime.now(timezone.utc)
    rule = Rule(
        rule_name=body.rule_name,
        rule_type=body.rule_type,
        zone=body.zone,
        permit_type=body.permit_type,
        condition=body.condition,
        value=body.value,
        active=body.active,
        created_at=now,
        updated_at=now,
    )
    db.add(rule)
    db.flush()   # get rule.id before audit

    db.add(AuditLog(
        submission_id=None,
        event_type="rule_created",
        new_value=json.dumps(body.model_dump()),
        changed_by="api:/rules POST",
    ))
    db.commit()
    db.refresh(rule)

    log.info("[Rules] Created rule id=%s name=%r", rule.id, rule.rule_name)
    return _rule_to_record(rule)


# ── PUT /rules/{id} ───────────────────────────────────────────────────────────

@app.put(
    "/rules/{rule_id}",
    response_model=RuleRecord,
    tags=["Rules"],
    summary="Update Rule",
)
def update_rule(rule_id: int, body: RuleUpdate, db: Session = Depends(get_db)):
    """
    Partially updates an existing rule (only supplied fields are changed).

    - Toggling `active` is logged as `rule_activated` / `rule_deactivated`.
    - All other field changes are logged as `rule_updated`.
    - Raises **404** if the rule does not exist.
    - Raises **409** if `rule_name` conflicts with another rule.
    """
    from database import Rule
    from datetime import datetime, timezone

    rule = db.query(Rule).filter(Rule.id == rule_id).first()
    if rule is None:
        raise HTTPException(status_code=404, detail=f"Rule id={rule_id} not found.")

    old_snapshot = json.dumps(rule.to_dict())
    changed_fields: dict = {}

    updates = body.model_dump(exclude_unset=True)  # only fields the client sent
    for field, new_val in updates.items():
        if getattr(rule, field) != new_val:
            changed_fields[field] = {"old": getattr(rule, field), "new": new_val}
            setattr(rule, field, new_val)

    if not changed_fields:
        # Nothing actually changed — return current state
        return _rule_to_record(rule)

    # Uniqueness check if rule_name is being renamed
    if "rule_name" in changed_fields:
        clash = (
            db.query(Rule)
            .filter(Rule.rule_name == body.rule_name, Rule.id != rule_id)
            .first()
        )
        if clash:
            raise HTTPException(
                status_code=409,
                detail=f"Another rule named '{body.rule_name}' already exists (id={clash.id}).",
            )

    rule.updated_at = datetime.now(timezone.utc)

    # Determine event_type
    if list(changed_fields.keys()) == ["active"]:
        event_type = "rule_activated" if body.active else "rule_deactivated"
    else:
        event_type = "rule_updated"

    db.add(AuditLog(
        submission_id=None,
        event_type=event_type,
        old_value=old_snapshot,
        new_value=json.dumps(changed_fields),
        changed_by="api:/rules PUT",
    ))
    db.commit()
    db.refresh(rule)

    log.info("[Rules] %s rule id=%s fields=%s", event_type, rule.id, list(changed_fields))
    return _rule_to_record(rule)


# ── DELETE /rules/{id} ────────────────────────────────────────────────────────

@app.delete(
    "/rules/{rule_id}",
    tags=["Rules"],
    summary="Delete Rule",
    status_code=200,
)
def delete_rule(
    rule_id:    int,
    hard:       bool = False,
    db: Session = Depends(get_db),
):
    """
    Removes a rule from the database.

    - **Default (soft delete):** sets `active=False` — rule is preserved in DB.
    - **`?hard=true` (hard delete):** permanently removes the row.

    Raises **404** if the rule does not exist.
    """
    from database import Rule
    from datetime import datetime, timezone

    rule = db.query(Rule).filter(Rule.id == rule_id).first()
    if rule is None:
        raise HTTPException(status_code=404, detail=f"Rule id={rule_id} not found.")

    old_snapshot = json.dumps(rule.to_dict())

    if hard:
        db.delete(rule)
        event_type = "rule_hard_deleted"
        msg = f"Rule id={rule_id} permanently deleted."
    else:
        rule.active = False
        rule.updated_at = datetime.now(timezone.utc)
        event_type = "rule_deactivated"
        msg = f"Rule id={rule_id} deactivated (soft delete). Use ?hard=true to remove permanently."

    db.add(AuditLog(
        submission_id=None,
        event_type=event_type,
        old_value=old_snapshot,
        new_value=None,
        changed_by="api:/rules DELETE",
    ))
    db.commit()


    log.info("[Rules] %s rule id=%s", event_type, rule_id)
    return {"message": msg, "rule_id": rule_id, "hard_deleted": hard}


# ===========================================================================
# Analytics  — GET /analytics
# ===========================================================================

class ZoneRejectionRate(BaseModel):
    zone:             str | None
    total:            int
    rejected:         int
    approved:         int
    rejection_rate:   float   # 0.0 – 1.0


class TopReason(BaseModel):
    feature:          str
    label:            str
    total_impact:     float   # sum of |shap_value| across all submissions
    avg_impact:       float   # mean |shap_value|
    count:            int     # submissions where this feature appeared in top-5
    direction:        str     # "increases_risk" | "decreases_risk" | "mixed"


class TimePoint(BaseModel):
    period:           str     # "YYYY-MM" (monthly)
    submissions:      int
    avg_risk_score:   float
    rejection_count:  int


class AnalyticsResponse(BaseModel):
    generated_at:        str
    total_submissions:   int
    overall_rejection_rate: float
    rejection_by_zone:   list[ZoneRejectionRate]
    top_rejection_reasons: list[TopReason]
    submissions_over_time: list[TimePoint]


@app.get(
    "/analytics",
    response_model=AnalyticsResponse,
    tags=["Analytics"],
    summary="Rejection Analytics Dashboard",
)
def get_analytics(db: Session = Depends(get_db)):
    """
    Aggregates submission data from the `submissions` table and returns:

    - **rejection_by_zone** — per-zone counts and rejection rate
    - **top_rejection_reasons** — top 5 SHAP features ranked by total impact
    - **submissions_over_time** — monthly trend (YYYY-MM buckets)

    All derived from stored `risk_score` (threshold ≥ 50 = Rejected)
    and `top_reasons` JSON field. No model re-inference required.
    """
    from datetime import datetime, timezone
    from collections import defaultdict

    all_rows = db.query(Submission).all()
    total    = len(all_rows)

    if total == 0:
        return AnalyticsResponse(
            generated_at=datetime.now(timezone.utc).isoformat(),
            total_submissions=0,
            overall_rejection_rate=0.0,
            rejection_by_zone=[],
            top_rejection_reasons=[],
            submissions_over_time=[],
        )

    # ── Categorise each row ──────────────────────────────────────────────────
    rejected_total = sum(1 for r in all_rows if r.risk_score >= 50.0)

    # ── 1. Rejection rate by zone ────────────────────────────────────────────
    zone_stats: dict[str, dict] = defaultdict(lambda: {"total": 0, "rejected": 0})
    for r in all_rows:
        z = r.zone or "Unknown"
        zone_stats[z]["total"]    += 1
        zone_stats[z]["rejected"] += 1 if r.risk_score >= 50.0 else 0

    rejection_by_zone = sorted(
        [
            ZoneRejectionRate(
                zone=z,
                total=s["total"],
                rejected=s["rejected"],
                approved=s["total"] - s["rejected"],
                rejection_rate=round(s["rejected"] / s["total"], 4),
            )
            for z, s in zone_stats.items()
        ],
        key=lambda x: x.rejection_rate,
        reverse=True,
    )

    # ── 2. Top rejection reasons (aggregate SHAP from top_reasons JSON) ──────
    feature_stats: dict[str, dict] = defaultdict(
        lambda: {"total_abs": 0.0, "count": 0, "increases": 0, "decreases": 0}
    )
    for r in all_rows:
        reasons = r.get_top_reasons()   # list[{feature, shap_value, impact}]
        for feat in reasons:
            name    = feat.get("feature", "")
            shap    = abs(float(feat.get("shap_value", 0.0)))
            impact  = feat.get("impact", "")
            if not name:
                continue
            feature_stats[name]["total_abs"]  += shap
            feature_stats[name]["count"]      += 1
            if impact == "increases_risk":
                feature_stats[name]["increases"] += 1
            else:
                feature_stats[name]["decreases"] += 1

    def _direction(s: dict) -> str:
        if s["increases"] > s["decreases"]:
            return "increases_risk"
        if s["decreases"] > s["increases"]:
            return "decreases_risk"
        return "mixed"

    top_reasons_sorted = sorted(
        feature_stats.items(),
        key=lambda kv: kv[1]["total_abs"],
        reverse=True,
    )[:5]

    top_rejection_reasons = [
        TopReason(
            feature=name,
            label=FIELD_LABELS.get(name, name.replace("_", " ").title()),
            total_impact=round(s["total_abs"], 4),
            avg_impact=round(s["total_abs"] / s["count"], 4) if s["count"] else 0.0,
            count=s["count"],
            direction=_direction(s),
        )
        for name, s in top_reasons_sorted
    ]

    # ── 3. Submissions over time (monthly buckets) ───────────────────────────
    monthly: dict[str, dict] = defaultdict(
        lambda: {"submissions": 0, "risk_sum": 0.0, "rejected": 0}
    )
    for r in all_rows:
        if r.submitted_at is None:
            continue
        bucket = r.submitted_at.strftime("%Y-%m")
        monthly[bucket]["submissions"] += 1
        monthly[bucket]["risk_sum"]    += r.risk_score
        monthly[bucket]["rejected"]    += 1 if r.risk_score >= 50.0 else 0

    submissions_over_time = [
        TimePoint(
            period=period,
            submissions=s["submissions"],
            avg_risk_score=round(s["risk_sum"] / s["submissions"], 2),
            rejection_count=s["rejected"],
        )
        for period, s in sorted(monthly.items())   # ascending chronological
    ]

    return AnalyticsResponse(
        generated_at=datetime.now(timezone.utc).isoformat(),
        total_submissions=total,
        overall_rejection_rate=round(rejected_total / total, 4),
        rejection_by_zone=rejection_by_zone,
        top_rejection_reasons=top_rejection_reasons,
        submissions_over_time=submissions_over_time,
    )


# ── FIELD_LABELS for analytics feature name → human label mapping ─────────────
FIELD_LABELS: dict[str, str] = {
    "source_enc":         "Application Source",
    "permit_type_enc":    "Permit Type",
    "zone_enc":           "Zone",
    "city_enc":           "City",
    "area_sqm":           "Area (sqm)",
    "construction_cost":  "Construction Cost",
    "doc_count":          "Docs Submitted",
    "missing_doc_count":  "Missing Docs",
    "has_title_deed":     "Title Deed",
    "has_site_plan":      "Site Plan",
    "has_noc_fire":       "Fire NOC",
    "has_noc_env":        "Env NOC",
    "has_struct_cert":    "Structural Cert",
    "has_crz_cert":       "CRZ Certificate",
    "pan_valid":          "PAN Valid",
    "zone_type_conflict": "Zone Conflict",
    "area_exceeds_fsi":   "Area Exceeds FSI",
    "risk_score":         "Risk Score",
    "year":               "Year",
    "month":              "Month",
}


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
