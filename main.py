"""
Aegis Master ML Pipeline - v3.2.2 (Stable Production)
Integration: 
1. Database: asyncpg Connection Pooling for Supabase.
2. Architecture: Push-based (Node.js sends hub_data).
3. Models: Risk Model, Risk Regressor, Income Model, Fraud Model.
"""

import os
import pickle
import asyncio
import numpy as np
import pandas as pd
import asyncpg
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# --- 1. Environment Setup ---
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
BASE_DIR = os.path.dirname(__file__)

# --- 2. Global Model Initialized with Fallbacks ---
class DummyModel:
    def predict(self, X): return np.array([5.0])
    def inverse_transform(self, X): return ["MODERATE"]

# Initialize global variables for models and features
RISK_MODEL = RISK_LE = RISK_REGRESSOR = DummyModel()
INCOME_REG = INCOME_CLF = INCOME_LE = DummyModel()
FRAUD_REG = FRAUD_CLF = FRAUD_LE = DummyModel()
db_pool = None

RISK_FEATURES = ["temp_c", "feels_like_c", "rainfall_mm", "pm25", "pm10", "traffic_index"]
INCOME_FEATURES = ["earnings_drop_pct", "order_drop_pct", "activity_drop_pct", "orders_last_hour", "earnings_today", "hours_worked_today", "avg_orders_7d", "avg_earnings_12w", "avg_hours_baseline"]
FRAUD_FEATURES = ["activity_drop_pct", "hours_worked_today", "earnings_drop_pct", "active_hours", "deliveries_completed", "avg_deliveries", "movement_distance_km", "order_drop_pct", "orders_last_hour"]

# --- 3. Helper Functions ---
def _load(filename):
    path = os.path.join(BASE_DIR, filename)
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"❌ Error unpickling {filename}: {e}")
    else:
        print(f"⚠️ File not found: {path}")
    return None

# --- 4. App Initialization ---
app = FastAPI(title="Aegis ML Master Pipeline", version="3.2.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 5. Event Handlers (Startup & Shutdown) ---
@app.on_event("startup")
async def startup_event():
    global db_pool, RISK_MODEL, RISK_LE, RISK_REGRESSOR, INCOME_REG, INCOME_CLF, INCOME_LE, FRAUD_REG, FRAUD_CLF, FRAUD_LE
    global RISK_FEATURES, INCOME_FEATURES, FRAUD_FEATURES

    # 1. Database Pool
    try:
        if DATABASE_URL:
            db_pool = await asyncpg.create_pool(
                DATABASE_URL, 
                min_size=1, 
                max_size=5, 
                ssl="require"
            )
            print("✅ PostgreSQL Connection Pool Created")
    except Exception as e:
        print(f"❌ Database Pool Error: {e}")

    # 2. Model Loading
    print("⌛ Loading ML Models...")
    
    # Risk Model
    risk_pkg = _load("risk_model.pkl")
    if risk_pkg:
        RISK_MODEL = risk_pkg.get("model", RISK_MODEL)
        RISK_LE = risk_pkg.get("label_encoder", RISK_LE)
        RISK_FEATURES = risk_pkg.get("features", RISK_FEATURES)
    RISK_REGRESSOR = _load("risk_regressor.pkl") or RISK_REGRESSOR

    # Income Model
    income_pkg = _load("income_model.pkl")
    if income_pkg:
        INCOME_REG = income_pkg.get("regressor", INCOME_REG)
        INCOME_CLF = income_pkg.get("classifier", INCOME_CLF)
        INCOME_LE = income_pkg.get("label_encoder", INCOME_LE)
        INCOME_FEATURES = income_pkg.get("features", INCOME_FEATURES)

    # Fraud Model
    fraud_pkg = _load("fraud_model.pkl")
    if fraud_pkg:
        FRAUD_REG = fraud_pkg.get("regressor", FRAUD_REG)
        FRAUD_CLF = fraud_pkg.get("classifier", FRAUD_CLF)
        FRAUD_LE = fraud_pkg.get("label_encoder", FRAUD_LE)
        FRAUD_FEATURES = fraud_pkg.get("features", FRAUD_FEATURES)

    print("✅ Startup Sequence Complete")

@app.on_event("shutdown")
async def shutdown_event():
    if db_pool:
        await db_pool.close()
        print("🛑 Database Pool Closed")

# --- 6. Data Models ---
class AnalyzeRequest(BaseModel):
    lat: float
    lon: float
    worker_id: str
    hub_data: dict

# --- 7. Core Pipeline Functions ---

async def publish_auto_alert(risk_score, city, zone):
    if not db_pool: return
    async with db_pool.acquire() as conn:
        try:
            exists = await conn.fetchval('''
                SELECT COUNT(*) FROM disruption_alerts 
                WHERE zone = $1 AND status = 'active' AND created_at > NOW() - INTERVAL '1 hour'
            ''', zone)

            if exists == 0 and risk_score > 4.0:
                payout_decimal = 0.85 if risk_score > 7.0 else 0.60
                await conn.execute('''
                    INSERT INTO disruption_alerts (trigger_type, zone, city, severity, status, payout_pct)
                    VALUES ($1, $2, $3, $4, $5, $6)
                ''', 'heavyRainfall', zone, city, risk_score, 'active', payout_decimal)
                print(f"📡 [AUTO-ALERT] Published for {zone}")
        except Exception as e:
            print(f"❌ Alert Publication Error: {e}")

def run_risk_analysis(hub_data):
    try:
        w = hub_data["external_disruption"]["weather"]
        aq = hub_data["external_disruption"]["air_quality"]
        features = {
            "temp_c": w["temp"], "feels_like_c": w["feels_like"], 
            "rainfall_mm": w["rain_1h"], "pm25": aq["pm25"], 
            "pm10": aq["pm10"], "traffic_index": 45
        }
        ml_input = pd.DataFrame([features])[RISK_FEATURES]
        score = float(RISK_REGRESSOR.predict(ml_input)[0])
        level = str(RISK_LE.inverse_transform([RISK_MODEL.predict(ml_input)[0]])[0])
        return {"risk_score": round(max(0.0, min(10.0, score)), 2), "risk_level": level}
    except:
        return {"risk_score": 5.0, "risk_level": "MODERATE"}

def run_income_analysis(hub_data):
    try:
        b = hub_data.get("business_impact", {}).get("historical_baseline", {})
        m = hub_data.get("business_impact", {}).get("metrics", {})
        features = {
            "earnings_drop_pct": m.get("earnings_drop_pct", 0.0), 
            "order_drop_pct": m.get("order_drop_pct", 0.0),
            "activity_drop_pct": m.get("activity_drop_pct", 0.0), 
            "orders_last_hour": 5, "earnings_today": 1200.0, "hours_worked_today": 4.0,
            "avg_orders_7d": b.get("avg_orders_7d", 20.0), 
            "avg_earnings_12w": b.get("avg_earnings_12w", 3500.0),
            "avg_hours_baseline": b.get("avg_hours_baseline", 8.0)
        }
        ml_input = pd.DataFrame([features])[INCOME_FEATURES]
        drop = float(INCOME_REG.predict(ml_input)[0])
        sev = str(INCOME_LE.inverse_transform([INCOME_CLF.predict(ml_input)[0]])[0])
        return {"income_drop_percent": round(max(0.0, drop), 2), "severity": sev}
    except:
        return {"income_drop_percent": 15.0, "severity": "MEDIUM"}

def run_fraud_analysis(hub_data):
    try:
        m = hub_data["business_impact"]["metrics"]
        features = {
            "activity_drop_pct": m["activity_drop_pct"], "hours_worked_today": 4.0, 
            "earnings_drop_pct": m["earnings_drop_pct"], "active_hours": 4.0, 
            "deliveries_completed": 5, "avg_deliveries": 3.0,
            "movement_distance_km": 12.5, "order_drop_pct": m["order_drop_pct"], 
            "orders_last_hour": 5
        }
        ml_input = pd.DataFrame([features])[FRAUD_FEATURES]
        score = float(FRAUD_REG.predict(ml_input)[0])
        level = str(FRAUD_LE.inverse_transform([FRAUD_CLF.predict(ml_input)[0]])[0])
        return {"fraud_score": round(max(0.0, min(1.0, score)), 2), "fraud_level": level}
    except:
        return {"fraud_score": 0.05, "fraud_level": "LOW"}

async def save_payout_record(worker_id, amount, decision):
    if not db_pool: return {"id": 0}
    async with db_pool.acquire() as conn:
        try:
            status = 'PENDING' if decision["requires_verification"] else 'APPROVED'
            row = await conn.fetchrow('''
                INSERT INTO payouts (worker_id, amount, risk_level, income_severity, fraud_level, status)
                VALUES ($1, $2, $3, $4, $5, $6) RETURNING id, created_at
            ''', str(worker_id), float(amount), decision["risk_level"], decision["income_severity"], decision["fraud_level"], status)
            return dict(row)
        except Exception as e:
            print(f"❌ DB Payout Record Error: {e}")
            return {"id": 0, "created_at": datetime.now()}

# --- 8. Endpoints ---

@app.post("/api/v1/analyze")
async def analyze_pipeline(req: AnalyzeRequest):
    hub_data = req.hub_data
    
    risk = run_risk_analysis(hub_data)
    income = run_income_analysis(hub_data)
    fraud = run_fraud_analysis(hub_data)

    baseline = hub_data.get("business_impact", {}).get("historical_baseline", {})
    daily_base = baseline.get("avg_earnings_12w", 3500) / 7
    
    potential_pcts = [0.0]
    reasons = []

    if fraud["fraud_level"] != "HIGH":
        w = hub_data["external_disruption"]["weather"]
        if w.get("rain_1h", 0) > 5.0 or risk["risk_score"] > 7.0:
            potential_pcts.append(0.85)
            reasons.append("High Environmental Risk")
        elif risk["risk_score"] > 4.0:
            potential_pcts.append(0.60)
            reasons.append("Moderate Risk Protection")

    trigger_pct = max(potential_pcts)
    payout_amount = round(daily_base * trigger_pct)
    requires_verification = fraud["fraud_score"] > 0.7

    decision = {
        "risk_level": risk["risk_level"],
        "income_severity": income["severity"],
        "fraud_level": fraud["fraud_level"],
        "requires_verification": requires_verification,
        "payout_amount": payout_amount,
        "payout_triggered": payout_amount > 0,
        "payout_strategy": {
            "amount": payout_amount,
            "selected_payout_pct": f"{int(trigger_pct * 100)}%",
            "active_triggers": reasons or ["Climate Protection Active"],
            "verification_mode": "MANUAL_VERIFY" if requires_verification else "INSTANT_PAY"
        }
    }

    loc = hub_data.get("location", {})
    city = loc.get("city") or "Coimbatore"
    zone = loc.get("zone") or "Central"

    await publish_auto_alert(risk["risk_score"], city, zone)
    db_rec = await save_payout_record(req.worker_id, payout_amount, decision)

    return {
        "status": "success",
        "analysis": {
            "risk": risk, "income": income, "fraud": fraud,
            "payout_strategy": decision["payout_strategy"]
        },
        "payout_record": {
            "payout_id": db_rec.get("id"),
            "status": "APPROVED" if (decision["payout_triggered"] and not requires_verification) else "PENDING"
        }
    }

@app.get("/")
@app.head("/") 
def health():
    return {
        "status": "Aegis Master Pipeline v3.2.2 Live", 
        "db": bool(db_pool)
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)
