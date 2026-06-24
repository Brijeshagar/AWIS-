"""
AWIS - Adaptive Workflow Intervention System  |  Database Layer
===============================================================
SQLite + SQLAlchemy (sync) for local persistence.

Tables:
  submissions  — every /predict call that is saved
  rules        — admin-managed rejection rule definitions
  audit_log    — change history for submissions

Usage:
    from database import SessionLocal, init_db

    init_db()          # call once on startup (creates tables if absent)

    db = SessionLocal()
    try:
        ...
    finally:
        db.close()

Or use the FastAPI dependency:
    from database import get_db
    def endpoint(db: Session = Depends(get_db)): ...
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    event,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

log = logging.getLogger("awis.db")

# ---------------------------------------------------------------------------
# Engine & session factory
# ---------------------------------------------------------------------------
# DB_PATH can be overridden via environment variable (used by Docker volume mount)
DB_PATH = Path(os.environ.get("DB_PATH", str(Path(__file__).parent / "awis.db")))
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},   # required for SQLite + threading
    echo=False,                                   # set True to log all SQL
)

# Enable WAL mode for better concurrent read performance with SQLite
@event.listens_for(engine, "connect")
def _set_sqlite_pragma(dbapi_conn, _connection_record):
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA foreign_keys=ON;")
    cursor.close()

SessionLocal: sessionmaker = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Helper: UTC-aware now()
# ---------------------------------------------------------------------------
def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ===========================================================================
# Table 1 — submissions
# ===========================================================================
class Submission(Base):
    """
    Records every application that was scored by the /predict endpoint.

    form_data   : complete raw JSON payload sent by the frontend (stored as text)
    risk_score  : rejection probability scaled 0-100
    top_reasons : JSON list of top SHAP contributing features
    source      : human-readable source label (e.g. "Online Portal")
    city        : human-readable city label
    permit_type : human-readable permit type label
    zone        : human-readable zone label
    """
    __tablename__ = "submissions"

    id            = Column(Integer, primary_key=True, index=True, autoincrement=True)
    form_data     = Column(Text,    nullable=False)      # JSON string
    risk_score    = Column(Float,   nullable=False)
    top_reasons   = Column(Text,    nullable=True)       # JSON string
    submitted_at  = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    source        = Column(String(100), nullable=True)
    city          = Column(String(100), nullable=True)
    permit_type   = Column(String(100), nullable=True)
    zone          = Column(String(100), nullable=True)

    def __repr__(self) -> str:
        return (
            f"<Submission id={self.id} risk={self.risk_score:.1f} "
            f"city={self.city!r} at={self.submitted_at}>"
        )

    # ── Convenience helpers ──────────────────────────────────────────────

    def set_form_data(self, data: dict) -> None:
        self.form_data = json.dumps(data, default=str)

    def get_form_data(self) -> dict:
        return json.loads(self.form_data) if self.form_data else {}

    def set_top_reasons(self, reasons: list) -> None:
        self.top_reasons = json.dumps(reasons, default=str)

    def get_top_reasons(self) -> list:
        return json.loads(self.top_reasons) if self.top_reasons else []

    def to_dict(self) -> dict:
        return {
            "id":           self.id,
            "risk_score":   self.risk_score,
            "top_reasons":  self.get_top_reasons(),
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "source":       self.source,
            "city":         self.city,
            "permit_type":  self.permit_type,
            "zone":         self.zone,
            "form_data":    self.get_form_data(),
        }


# ===========================================================================
# Table 2 — rules
# ===========================================================================
class Rule(Base):
    """
    Admin-managed rejection rules.

    rule_type   : e.g. "threshold", "flag_check", "document_check"
    condition   : operator or expression string (e.g. ">", "==", "missing")
    value       : threshold or comparison value as string (e.g. "500", "true")
    active      : soft-delete / enable flag
    """
    __tablename__ = "rules"

    id           = Column(Integer, primary_key=True, index=True, autoincrement=True)
    rule_name    = Column(String(200), nullable=False, unique=True)
    rule_type    = Column(String(100), nullable=False)
    zone         = Column(String(100), nullable=True)   # None = applies to all zones
    permit_type  = Column(String(100), nullable=True)   # None = applies to all types
    condition    = Column(String(200), nullable=False)
    value        = Column(String(200), nullable=False)
    active       = Column(Boolean,     nullable=False, default=True)
    created_at   = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at   = Column(DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow)

    def __repr__(self) -> str:
        status = "ON" if self.active else "OFF"
        return f"<Rule id={self.id} name={self.rule_name!r} [{status}]>"

    def to_dict(self) -> dict:
        return {
            "id":          self.id,
            "rule_name":   self.rule_name,
            "rule_type":   self.rule_type,
            "zone":        self.zone,
            "permit_type": self.permit_type,
            "condition":   self.condition,
            "value":       self.value,
            "active":      self.active,
            "created_at":  self.created_at.isoformat() if self.created_at else None,
            "updated_at":  self.updated_at.isoformat() if self.updated_at else None,
        }


# ===========================================================================
# Table 3 — audit_log
# ===========================================================================
class AuditLog(Base):
    """
    Immutable change history.  Every time a submission is reviewed, flagged,
    overridden, or a rule is toggled, an entry is written here.

    submission_id : FK to submissions.id (nullable — rules changes have no submission)
    event_type    : e.g. "submission_created", "risk_overridden",
                    "rule_activated", "rule_deactivated", "status_changed"
    old_value     : serialised previous state (string / JSON)
    new_value     : serialised new state
    changed_by    : user/system identifier (e.g. "admin", "api", "system")
    """
    __tablename__ = "audit_log"

    id            = Column(Integer, primary_key=True, index=True, autoincrement=True)
    submission_id = Column(Integer, ForeignKey("submissions.id", ondelete="SET NULL"), nullable=True, index=True)
    event_type    = Column(String(100), nullable=False)
    old_value     = Column(Text, nullable=True)
    new_value     = Column(Text, nullable=True)
    changed_by    = Column(String(200), nullable=False, default="system")
    changed_at    = Column(DateTime(timezone=True), nullable=False, default=_utcnow)

    def __repr__(self) -> str:
        return (
            f"<AuditLog id={self.id} event={self.event_type!r} "
            f"sub={self.submission_id} by={self.changed_by!r}>"
        )

    def to_dict(self) -> dict:
        return {
            "id":            self.id,
            "submission_id": self.submission_id,
            "event_type":    self.event_type,
            "old_value":     self.old_value,
            "new_value":     self.new_value,
            "changed_by":    self.changed_by,
            "changed_at":    self.changed_at.isoformat() if self.changed_at else None,
        }


# ===========================================================================
# DB initialisation
# ===========================================================================
def init_db() -> None:
    """
    Create all tables that do not yet exist.
    Safe to call multiple times — existing tables are never dropped.
    Call this once during FastAPI lifespan startup.
    """
    Base.metadata.create_all(bind=engine)
    log.info("[DB] Tables verified / created at %s", DB_PATH)

    # Seed a default set of rules if the table is empty
    with SessionLocal() as db:
        if db.query(Rule).count() == 0:
            _seed_default_rules(db)


def _seed_default_rules(db: Session) -> None:
    """Insert sensible default rules on first run."""
    defaults = [
        Rule(
            rule_name="missing_docs_high",
            rule_type="threshold",
            zone=None,
            permit_type=None,
            condition=">",
            value="3",
            active=True,
        ),
        Rule(
            rule_name="zone_type_conflict_flag",
            rule_type="flag_check",
            zone=None,
            permit_type=None,
            condition="==",
            value="1",
            active=True,
        ),
        Rule(
            rule_name="area_exceeds_fsi_flag",
            rule_type="flag_check",
            zone=None,
            permit_type=None,
            condition="==",
            value="1",
            active=True,
        ),
        Rule(
            rule_name="no_title_deed",
            rule_type="document_check",
            zone=None,
            permit_type="Residential Building",
            condition="==",
            value="0",
            active=True,
        ),
        Rule(
            rule_name="industrial_eco_zone",
            rule_type="zone_permit_mismatch",
            zone="Special / Eco-Sensitive Zone",
            permit_type="Industrial",
            condition="match",
            value="block",
            active=True,
        ),
    ]
    db.add_all(defaults)
    db.commit()
    log.info("[DB] Seeded %d default rules.", len(defaults))


# ===========================================================================
# FastAPI dependency
# ===========================================================================
def get_db() -> Generator[Session, None, None]:
    """
    Yields a SQLAlchemy session and guarantees it is closed after the request.

    Usage in FastAPI:
        from database import get_db
        from sqlalchemy.orm import Session
        from fastapi import Depends

        @app.get("/example")
        def example(db: Session = Depends(get_db)):
            return db.query(Submission).all()
    """
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ===========================================================================
# Standalone test
# ===========================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log.info("Initialising AWIS database...")
    init_db()

    with SessionLocal() as db:
        sub_count  = db.query(Submission).count()
        rule_count = db.query(Rule).count()
        log.count  = db.query(AuditLog).count()

        print(f"\n  DB path       : {DB_PATH}")
        print(f"  submissions   : {sub_count} rows")
        print(f"  rules         : {rule_count} rows")
        print(f"  audit_log     : {db.query(AuditLog).count()} rows")

        print("\n  Active rules:")
        for r in db.query(Rule).filter(Rule.active == True).all():
            print(f"    [{r.id}] {r.rule_name:<30} {r.condition} {r.value}")
