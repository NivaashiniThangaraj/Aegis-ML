"""
Aegis Master ML Pipeline - v3.0 (Autonomous & Verified)
Updated to include:
1. Autonomous Alert Publication (Supabase Integration)
2. Conditional Fraud Verification (Instant vs. Manual Payouts)
3. Safe Data Extraction (Handles KeyError: 'city')
"""

import os
import pickle
import asyncio
from datetime import datetime

import numpy as np
import pandas as pd
import httpx
import asyncpg
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# --- 1. Environment Setup ---
load_dotenv()

DATA_HUB_URL = os.getenv("DATA_HUB_URL", "http://localhost:3015/api/risk-data")
DATABASE_URL = os.getenv("DATABASE_URL")

# --- 2. App Initialization ---
app = FastAPI(title="Aegis ML Master Pipeline", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. Database & Autonomous Alert Logic ---
async def get_db_conn():
    if not DATABASE_URL:
        raise HTTPException(status_code=500, detail="DATABASE_URL is missing.")
    try:
        return await asyncpg.connect(DATABASE_URL)
    except Exception as e:
        print(f"❌ Database Error: {e}")
        raise HTTPException(status_code=500, detail="Database connection failed")

async def publish_auto_alert(worker_id, risk_score, risk_level, city, zone):
    """
    AUTONOMOUS TRIGGER: Monitors risk levels and publishes alerts.
    Includes a 1-hour 'cooldown' to prevent duplicate alerts in the same zone.
    """
    conn = await get_db_conn()
    try:
        # 1. 🚀 REAL-TIME DE-DUPLICATION:
        # Check if an active alert was already published for this zone recently.
        # This prevents the DB from filling up during continuous real-time polling.
        existing_alert = await conn.fetchval('''
            SELECT COUNT(*) FROM disruption_alerts 
            WHERE zone = $1 
              AND status = 'active' 
              AND created_at > NOW() - INTERVAL '1 hour'
        ''', zone)

        if existing_alert > 0:
            print(f"ℹ️ [REAL-TIME] Alert for {zone} is already active. Skipping duplicate insert.")
            return

        # 2. Threshold Check
        if risk_score > 4.0:
            # Payout Logic: Sync with your Worker Avg / 7 logic
            # Using decimal format (0.85) to ensure Flutter math works
            payout_decimal = 0.85 if risk_score > 7.0 else 0.60
            
            print(f"🚀 [AUTO-PUBLISH] New High Risk ({risk_score}) detected in {city}. Triggering Global Alert.")
            
            await conn.execute('''
                INSERT INTO disruption_alerts (
                    trigger_type, 
                    zone, 
                    city, 
                    severity, 
                    status, 
                    payout_pct
                ) VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT DO NOTHING
            ''', 
            'heavyRainfall', 
            zone, 
            city, 
            risk_score, # Actual dynamic risk score
            'active', 
            payout_decimal
            )
            print(f"✅ ALERT SUCCESSFULLY PUBLISHED TO SUPABASE FOR {zone}")
            
    except Exception as e:
        print(f"❌ Alert Publication Failed: {e}")
    finally:
        await conn.close()
# --- 4. Load ML Models ---
BASE_DIR = os.path.dirname(__file__)

def _load(filename):
    path = os.path.join(BASE_DIR, filename)
    with open(path, "rb") as f:
        return pickle.load(f)

try:
    # Risk Model
    risk_pkg = _load("risk_model.pkl")
    RISK_MODEL = risk_pkg["model"]
    RISK_LE = risk_pkg["label_encoder"]
    RISK_FEATURES = risk_pkg["features"]
    RISK_REGRESSOR = _load("risk_regressor.pkl")

    # Income Model
    income_pkg = _load("income_model.pkl")
    INCOME_REG = income_pkg["regressor"]
    INCOME_CLF = income_pkg["classifier"]
    INCOME_LE = income_pkg["label_encoder"]
    INCOME_FEATURES = income_pkg["features"]

    # Fraud Model
    fraud_pkg = _load("fraud_model.pkl")
    FRAUD_REG = fraud_pkg["regressor"]
    FRAUD_CLF = fraud_pkg["classifier"]
    FRAUD_LE = fraud_pkg["label_encoder"]
    FRAUD_FEATURES = fraud_pkg["features"]

    print("✅ All ML models loaded successfully.")
except Exception as e:
    print(f"⚠️ Warning: Model load error. Using fallbacks. Error: {e}")
    RISK_FEATURES = ["temp_c", "feels_like_c", "rainfall_mm", "pm25", "pm10", "traffic_index"]
    INCOME_FEATURES = ["earnings_drop_pct", "order_drop_pct", "activity_drop_pct", "orders_last_hour", "earnings_today", "hours_worked_today", "avg_orders_7d", "avg_earnings_12w", "avg_hours_baseline"]
    FRAUD_FEATURES = ["activity_drop_pct", "hours_worked_today", "earnings_drop_pct", "active_hours", "deliveries_completed", "avg_deliveries", "movement_distance_km", "order_drop_pct", "orders_last_hour"]

# --- 5. Data Models ---
class AnalyzeRequest(BaseModel):
    lat: float
    lon: float
    worker_id: str

# --- 6. Core Pipeline Functions ---
async def fetch_db_activity(worker_id: str):
    conn = await get_db_conn()
    try:
        query = """
            SELECT COALESCE(SUM(earnings), 0) as total_earnings, 
                   COUNT(*) as total_orders 
            FROM orders 
            WHERE worker_id = $1 AND timestamp::date = CURRENT_DATE
        """
        row = await conn.fetchrow(query, worker_id)
        return dict(row) if row else {"total_earnings": 0.0, "total_orders": 0}
    except Exception:
        return {"total_earnings": 0.0, "total_orders": 0}
    finally:
        await conn.close()

async def fetch_hub_data(lat: float, lon: float, worker_id: str):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                f"{DATA_HUB_URL}?lat={lat}&lon={lon}&worker_id={worker_id}", 
                timeout=15.0
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise HTTPException(status_code=502, detail="Data Hub Unreachable.")

def run_risk_analysis(hub_data):
    w = hub_data["external_disruption"]["weather"]
    aq = hub_data["external_disruption"]["air_quality"]
    features = {"temp_c": w["temp"], "feels_like_c": w["feels_like"], "rainfall_mm": w["rain_1h"], "pm25": aq["pm25"], "pm10": aq["pm10"], "traffic_index": 45}
    try:
        ml_input = pd.DataFrame([features])[RISK_FEATURES]
        risk_score = float(RISK_REGRESSOR.predict(ml_input)[0])
        risk_level = str(RISK_LE.inverse_transform([RISK_MODEL.predict(ml_input)[0]])[0])
    except:
        risk_score, risk_level = 5.0, "MODERATE"
    return {"risk_score": round(max(0.0, min(10.0, risk_score)), 2), "risk_level": risk_level}

def run_income_analysis(hub_data, db_activity):
    b = hub_data.get("business_impact", {}).get("historical_baseline", {})
    m = hub_data.get("business_impact", {}).get("metrics", {})
    features = {
        "earnings_drop_pct": m.get("earnings_drop_pct", 0.0), "order_drop_pct": m.get("order_drop_pct", 0.0),
        "activity_drop_pct": m.get("activity_drop_pct", 0.0), "orders_last_hour": int(db_activity.get("total_orders", 0)),
        "earnings_today": float(db_activity.get("total_earnings", 0.0)), "hours_worked_today": 4.0,
        "avg_orders_7d": b.get("avg_orders_7d", 20.0), "avg_earnings_12w": b.get("avg_earnings_12w", 1500.0),
        "avg_hours_baseline": b.get("avg_hours_baseline", 8.0)
    }
    try:
        ml_input = pd.DataFrame([features])[INCOME_FEATURES]
        income_drop_pred = float(INCOME_REG.predict(ml_input)[0])
        severity = str(INCOME_LE.inverse_transform([INCOME_CLF.predict(ml_input)[0]])[0])
    except:
        income_drop_pred, severity = 15.0, "MEDIUM"
    return {"income_drop_percent": round(max(0.0, income_drop_pred), 2), "severity": severity}

def run_fraud_analysis(hub_data, db_activity):
    b = hub_data["business_impact"]["historical_baseline"]
    m = hub_data["business_impact"]["metrics"]
    curr_orders = int(db_activity["total_orders"])
    features = {
        "activity_drop_pct": m["activity_drop_pct"], "hours_worked_today": 4.0, "earnings_drop_pct": m["earnings_drop_pct"],
        "active_hours": 4.0, "deliveries_completed": curr_orders, "avg_deliveries": b["avg_orders_7d"] / 8.0,
        "movement_distance_km": 12.5, "order_drop_pct": m["order_drop_pct"], "orders_last_hour": curr_orders
    }
    try:
        ml_input = pd.DataFrame([features])[FRAUD_FEATURES]
        fraud_score = float(FRAUD_REG.predict(ml_input)[0])
        fraud_level = str(FRAUD_LE.inverse_transform([FRAUD_CLF.predict(ml_input)[0]])[0])
    except:
        fraud_score, fraud_level = 0.05, "LOW"
    return {"fraud_score": round(max(0.0, min(1.0, fraud_score)), 2), "fraud_level": fraud_level}

def run_decision_engine(risk, income, fraud, hub_data, db_activity):
    # 1. Initialize variables to avoid NameErrors
    potential_payout_pcts = [0.0]  # List must exist before being used
    trigger_reasons = []
    
    # 2. DYNAMIC FETCH: Get the real worker average from Hub Data
    # Fallback to 3822 if the dictionary path doesn't exist
    business_impact = hub_data.get("business_impact", {})
    baseline = business_impact.get("historical_baseline", {})
    historical_weekly_avg = baseline.get("avg_earnings_12w", 3822)
    
    # YOUR CORRECT METHOD: Daily Base = Weekly / 7
    daily_base = historical_weekly_avg / 7
    
    # 3. FRAUD GATE: Only calculate triggers if Fraud is not HIGH
    if fraud.get("fraud_level") != "HIGH":
        w = hub_data.get("external_disruption", {}).get("weather", {})
        rp = hub_data.get("external_disruption", {}).get("risk_parameters", {})
        
        if w.get("rain_1h", 0) > 65 or rp.get("rain_risk", 0) >= 4:
            potential_payout_pcts.append(0.80)
            trigger_reasons.append("Heavy Rainfall Disruption")
            
        if w.get("temp", 0) > 41:
            potential_payout_pcts.append(0.75)
            trigger_reasons.append("Extreme Heat Safety Halt")
            
        if income.get("income_drop_percent", 0) > 60:
            potential_payout_pcts.append(1.00)
            trigger_reasons.append("Severe Verified Income Loss")

    # 4. Fallback for Demo Stability (Moderate Risk)
    # If no weather triggers hit but risk score is high, provide base coverage
    if max(potential_payout_pcts) == 0 and risk.get("risk_score", 0) >= 4.0:
        potential_payout_pcts.append(0.60)
        trigger_reasons.append("Moderate Risk Base Protection")

    # 5. Final Calculations
    trigger_pct = max(potential_payout_pcts)
    payout_amount = round(daily_base * trigger_pct)
    requires_verification = fraud.get("fraud_score", 0) > 0.7

    return {
        "payout_triggered": payout_amount > 0,
        "payout_amount": payout_amount, 
        "requires_verification": requires_verification,
        "payout_strategy": {
            "amount": payout_amount,
            "selected_payout_pct": f"{int(trigger_pct * 100)}%",
            "active_triggers": trigger_reasons or ["Climate Protection Active"],
            "verification_mode": "MANUAL_VERIFY" if requires_verification else "INSTANT_PAY"
        },
        "risk_level": risk.get("risk_level", "LOW"),
        "fraud_level": fraud.get("fraud_level", "LOW"),
        "income_severity": income.get("severity", "LOW")
    }
async def save_payout(worker_id, amount, decision):
    conn = await get_db_conn()
    try:
        await conn.execute('''
            INSERT INTO payouts (worker_id, amount, risk_level, income_severity, fraud_level, status)
            VALUES ($1, $2, $3, $4, $5, $6)
        ''', str(worker_id), float(amount), decision["risk_level"], decision["income_severity"], decision["fraud_level"], 
        'PENDING' if decision["requires_verification"] else 'APPROVED')
        
        row = await conn.fetchrow('SELECT id, created_at FROM payouts WHERE worker_id = $1 ORDER BY created_at DESC LIMIT 1', str(worker_id))
        return dict(row) if row else {"id": 101, "created_at": datetime.now()}
    except:
        return {"id": 0, "created_at": datetime.now()}
    finally:
        await conn.close()

# --- 7. Main API Endpoint ---
@app.post("/api/v1/analyze")
async def analyze_pipeline(req: AnalyzeRequest):
    hub_data = await fetch_hub_data(req.lat, req.lon, req.worker_id)
    db_activity = await fetch_db_activity(req.worker_id)
    
    # ML ANALYSES
    risk = run_risk_analysis(hub_data)
    income = run_income_analysis(hub_data, db_activity)
    fraud = run_fraud_analysis(hub_data, db_activity)

    # ---------------------------------------------------------
    # TEST CASES (Comment out predicted risk_score to test Case 1 or 2)
    # ---------------------------------------------------------
    # Actual Predicted Score (Live Logic):
    risk_score = risk["risk_score"]
    risk_level = risk["risk_level"]
    fraud_score = fraud["fraud_score"]

    #CASE 1: High Risk + Low Fraud (Instant Payout + Auto Alert)
    #risk_score, risk_level, fraud_score = 9.2, "EXTREME", 0.15

    # CASE 2: High Risk + High Fraud (Manual Verification + Auto Alert)
    #risk_score, risk_level, fraud_score = 8.5, "HIGH", 0.88
    # ---------------------------------------------------------

    # 1. Update risk/fraud dicts with test values if needed
    risk["risk_score"], risk["risk_level"] = risk_score, risk_level
    fraud["fraud_score"] = fraud_score

    # 2. Run Decision Engine
    decision = run_decision_engine(risk, income, fraud, hub_data, db_activity)
    
    # 3. FIX: Handle KeyError 'city' safely
    # 2. Extract location info FIRST 
    # This pulls the EXACT city/zone the Flutter user has selected
    location_info = hub_data.get("location", {})
    
    # Safely get city and zone with fallbacks
    city_name = location_info.get("city") or location_info.get("name") or "Coimbatore"
    zone_name = location_info.get("zone") or "Zone 4 — Central"
    
    print(f"📍 [LOCATION] Worker is in City: {city_name}, Zone: {zone_name}")

    # ... (ML logic and test scores go here) ...

    # 4. Autonomous Alert Publication
    # Python now uses the dynamic zone_name instead of a hardcoded string
    await publish_auto_alert(
        req.worker_id, 
        risk_score, 
        risk_level, 
        city_name, 
        zone_name
    )

    # 5. Database Logging
    db_record = await save_payout(req.worker_id, decision["payout_amount"], decision)
    
    # Inside analyze_pipeline function in main.py
    
    return {
        "status": "success",
        "requires_verification": decision["requires_verification"],
        "metadata": {"worker_id": req.worker_id, "place": city_name},
        "analysis": {
            "risk": risk, 
            "income": income, 
            "fraud": fraud,
            # 🚀 CHANGE THIS LINE: from "trigger_analysis" to "payout_strategy"
            "payout_strategy": decision["payout_strategy"] 
        },
        "payout_record": {
            "payout_id": db_record.get("id"),
            "amount": decision["payout_amount"],
            "triggered_at": db_record.get("created_at"),
            "status": "APPROVED" if (decision["payout_triggered"] and not decision["requires_verification"]) else "PENDING"
        }
    }

@app.get("/")
def health():
    return {"status": "Aegis Master Pipeline v3.0 Live", "database_connected": bool(DATABASE_URL)}

# --- 8. Cloud Deployment Setup ---
if __name__ == "__main__":
    import uvicorn
    # 🚀 RECTIFICATION: Use the PORT environment variable provided by Render/Railway
    # Default to 5000 if running locally
    port = int(os.getenv("PORT", 5000))
    
    # 🚀 RECTIFICATION: Host MUST be '0.0.0.0' for external access
    uvicorn.run(app, host="0.0.0.0", port=port)