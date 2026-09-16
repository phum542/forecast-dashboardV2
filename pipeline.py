#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ============================================================
# SaaS Metrics Pipeline (complete, self-contained)
# Run:  python pipeline.py   ->  writes customers.csv + dashboard_data.json
# Then serve the folder:  python -m http.server
# ============================================================
import json
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N_CUSTOMERS = 10_000
DATA_START, DATA_END = "2022-01-01", "2025-06-30"
OUT_JSON = "dashboard_data.json"

SEGMENTS = ["enterprise", "midmarket", "smb"]
BASE_FEE = {"enterprise": 400, "midmarket": 120, "smb": 35}
SEG_PROB = [0.10, 0.35, 0.55]
REGIONS = ["North America", "Europe", "APAC", "Other"]
REGION_PROB = [0.45, 0.30, 0.18, 0.07]

# FIX: region fee multipliers (so region ARPU differs on the dashboard)
REGION_FEE_MULT = {
    "North America": 1.15,
    "Europe": 1.00,
    "APAC": 0.90,
    "Other": 0.85,
}


# ------------------------------------------------------------
# STEP 1-2: Simulate customers
# ------------------------------------------------------------
def simulate_customers():
    n = N_CUSTOMERS
    start_ts = pd.Timestamp(DATA_START)
    end_ts = pd.Timestamp(DATA_END)

    # pd.Series wrapper -> .dt works everywhere downstream
    signup = pd.Series(
        start_ts
        + pd.to_timedelta(
            rng.integers(0, (end_ts - start_ts).days, n), "D"))

    segment = pd.Series(rng.choice(SEGMENTS, n, p=SEG_PROB))
    region = pd.Series(rng.choice(REGIONS, n, p=REGION_PROB))

    # cohort "fit" decays over time -> LTV:CAC decay story
    cohort_idx = (signup.dt.year - 2022) * 12 + signup.dt.month
    fit = 1.0 - 0.055 * (cohort_idx - cohort_idx.min()) / 12.0

    monthly_fee = (np.array([BASE_FEE[s] for s in segment])
                   * rng.uniform(0.85, 1.15, n)
                   * np.array([REGION_FEE_MULT[r] for r in region])
                   * fit.values)

    first_month_hazard = np.clip(rng.normal(0.22, 0.05, n), 0.08, 0.40) * fit.values
    base_hazard = np.clip(rng.normal(0.028, 0.008, n), 0.012, 0.055) * fit.values

    churn_month = np.full(n, -1)  # -1 = still active
    signup_periods = signup.dt.to_period("M")
    end_period = end_ts.to_period("M")
    for i in range(n):
        max_m = int((end_period - signup_periods.iloc[i]).n)
        for m in range(1, max_m + 1):
            h = first_month_hazard[i] if m == 1 else base_hazard[i]
            if m in (11, 12):
                h *= 1.6
            if rng.random() < h:
                churn_month[i] = m
                break

    return pd.DataFrame({
        "customer_id": np.arange(n),
        "signup_date": signup,
        "segment": segment,
        "region": region,
        "monthly_fee": np.round(monthly_fee, 2),
        "churn_month": churn_month,
    })


customers = simulate_customers()
customers.to_csv("customers.csv", index=False)

# ------------------------------------------------------------
# STEP 3: Monthly aggregation
# ------------------------------------------------------------
months = pd.period_range(DATA_START, DATA_END, freq="M")
signup_p = customers.signup_date.dt.to_period("M")
# FIX: month arithmetic on PeriodIndex uses plain ints (no timedelta "M")
churn_p = signup_p + customers.churn_month.clip(lower=0).astype(int)

monthly = []
prev_mrr = prev_churn = prev_cac = prev_arpu = None
first_mrr = None   # จะ set ใน iteration แรก
for p in months:
    active = customers[(signup_p <= p)
                       & ((customers.churn_month < 0) | (churn_p > p))]
    new_cust = int((signup_p == p).sum())
    churned_n = int(((churn_p == p) & (customers.churn_month > 0)).sum())
    start_active = len(active) + churned_n - new_cust
    mrr = float(active.monthly_fee.sum())
    if first_mrr is None:
        first_mrr = mrr
    churn_rate = round(churned_n / max(start_active, 1) * 100, 2)
    spend = 115_000 + rng.normal(0, 12_000) + 4_000 * (p.ordinal % 12)
    cac = round(float(spend) / max(new_cust, 1), 2)
    arpu = round(mrr / max(len(active), 1), 2)
    target = round(first_mrr * 1.10 * (1.07 ** (p.ordinal - months[0].ordinal)), 2)
    row = {
        "month": str(p), "actual_mrr": round(mrr, 2), "target_mrr": target,
        "churn_rate": churn_rate, "cac": cac, "arpu": arpu,
        "new_customers": new_cust, "marketing_spend": round(float(spend), 2),
    }
    row["mom_delta"] = {
        "mrr_pct": None if prev_mrr is None else round((mrr - prev_mrr) / prev_mrr * 100, 2),
        "churn_delta": None if prev_churn is None else round(churn_rate - prev_churn, 2),
        "cac_pct": None if prev_cac is None else round((cac - prev_cac) / prev_cac * 100, 2),
        "arpu_pct": None if prev_arpu is None else round((arpu - prev_arpu) / prev_arpu * 100, 2),
    }
    monthly.append(row)
    prev_mrr, prev_churn, prev_cac, prev_arpu = mrr, churn_rate, cac, arpu

# ------------------------------------------------------------
# STEP 4: 6-month forecast (linear on last 12 months)
# ------------------------------------------------------------
last12 = monthly[-12:]
x = np.arange(len(last12))
y = np.array([m["actual_mrr"] for m in last12])
slope, intercept = np.polyfit(x, y, 1)
last_period = pd.Period(monthly[-1]["month"])
forecast = [{"month": str(last_period + k),
             "forecast_mrr": round(float(slope * (len(last12) - 1 + k) + intercept), 2)}
            for k in range(1, 7)]

# ------------------------------------------------------------
# STEP 5: Cohorts (retention + LTV:CAC)
# ------------------------------------------------------------
cohort_data = []
cac_normal = float(np.median([m["cac"] for m in monthly[-12:]]))
end_p = pd.Timestamp(DATA_END).to_period("M")
for cohort, g in customers.groupby(signup_p):
    max_m = int((end_p - cohort).n)
    observed_life = np.minimum(
        g.churn_month.where(g.churn_month > 0, np.inf), max_m + 0.5)
    surv = {str(m): round(float((observed_life > m).mean() * 100), 1)
            for m in (1, 3, 6, 12)}
    ltv = float(g.monthly_fee.mean() * np.minimum(observed_life, 36).mean())
    cohort_data.append({
        "cohort": str(cohort),
        "size": int(len(g)),
        "age_months": max_m,  # NEW: อายุจริงของ cohort ณ วันสิ้นสุด data (ไว้กรอง benchmark + ให้ frontend แสดง N/A)
        "retention": surv,
        "avg_monthly_fee": round(float(g.monthly_fee.mean()), 2),
        "ltv": round(ltv, 2),
        "ltv_cac_ratio": round(ltv / cac_normal, 2),
    })

# NEW: retention benchmark by age — เฉลี่ยเฉพาะ cohort ที่อายุถึง month นั้นจริงเท่านั้น
# (cohort อายุน้อยจะมีค่า 0.0 ในเดือนที่ยังไม่ถึง ถ้าเอามาเฉลี่ยด้วย benchmark จะต่ำเกินจริง)
retention_benchmark = {}
for m in (1, 3, 6, 12):
    vals = [c["retention"][str(m)] for c in cohort_data if c["age_months"] >= m]
    if vals:
        retention_benchmark[str(m)] = round(float(np.mean(vals)), 1)

# ------------------------------------------------------------
# STEP 6: Segments + Region matrix
# ------------------------------------------------------------
active_all = customers[customers.churn_month < 0]
total_mrr = float(active_all.monthly_fee.sum())

seg_rows = []
for seg in SEGMENTS:
    g = active_all[active_all.segment == seg]
    churned_pct = float((customers[customers.segment == seg].churn_month > 0).mean() * 100)
    spark = []
    for p in months[-6:]:
        mask = (g.signup_date.dt.to_period("M") <= p)
        spark.append(round(float(g[mask].monthly_fee.sum()), 2))
    seg_rows.append({
        "segment": seg,
        "customers": int(len(g)),
        "mrr": round(float(g.monthly_fee.sum()), 2),
        "share_pct": round(float(g.monthly_fee.sum()) / total_mrr * 100, 1),
        "churn_rate": round(churned_pct, 1),
        "sparkline_6m": spark,
    })

region_matrix = {}
for region in REGIONS:
    region_matrix[region] = {}
    for seg in SEGMENTS:
        g = active_all[(active_all.region == region) & (active_all.segment == seg)]
        region_matrix[region][seg] = {
            "customers": int(len(g)),
            "mrr": round(float(g.monthly_fee.sum()), 2),
        }

# NEW: region totals + share_pct (ไว้ทำ share bar / concentration ใน frontend)
region_rows = []
for region in REGIONS:
    r_mrr = round(sum(c["mrr"] for c in region_matrix[region].values()), 2)
    region_rows.append({
        "region": region,
        "mrr": r_mrr,
        "share_pct": round(r_mrr / total_mrr * 100, 1) if total_mrr else 0.0,
        "matrix": region_matrix[region],
    })

# ------------------------------------------------------------
# STEP 7: Anomaly detection (last 18 months)
# ------------------------------------------------------------
anomalies = []
for key in ("actual_mrr", "churn_rate", "cac", "arpu", "new_customers", "marketing_spend"):
    series = [m[key] for m in monthly]
    for idx in range(max(0, len(monthly) - 18), len(monthly)):
        window = series[max(0, idx - 12):idx]
        if len(window) < 6:
            continue
        mu, sd = float(np.mean(window)), float(np.std(window))
        v = float(series[idx])
        z = (v - mu) / sd if sd > 0 else 0.0
        # CHANGED: เพิ่ม delta_from_mean + delta_pct เพื่อแปลงเป็น impact statement ฝั่ง frontend
        if z >= 3 or v > mu + 1.96 * sd:
            anomalies.append({"month": monthly[idx]["month"], "metric": key,
                              "direction": "spike", "value": round(v, 2),
                              "z_score": round(z, 2),
                              "delta_from_mean": round(v - mu, 2),
                              "delta_pct": round((v - mu) / mu * 100, 1) if mu else 0.0})
        elif z <= -3 or v < mu - 1.96 * sd:
            anomalies.append({"month": monthly[idx]["month"], "metric": key,
                              "direction": "drop", "value": round(v, 2),
                              "z_score": round(z, 2),
                              "delta_from_mean": round(v - mu, 2),
                              "delta_pct": round((v - mu) / mu * 100, 1) if mu else 0.0})
anomalies.sort(key=lambda a: a["month"], reverse=True)

# ------------------------------------------------------------
# STEP 8: Assemble JSON (plain Python types only)
# ------------------------------------------------------------
latest = monthly[-1]
dashboard = {
    "meta": {"generated_at": pd.Timestamp.now("UTC").isoformat(),  # เดิม utcnow() ถูก deprecate
             "data_end": DATA_END, "synthetic": True},
    "kpi_latest": {
        "mrr": float(latest["actual_mrr"]),
        "churn_rate": float(latest["churn_rate"]),
        "cac": float(latest["cac"]),
        "arpu": float(latest["arpu"]),
        "mom_delta": {k: (float(v) if v is not None else None)
                      for k, v in latest["mom_delta"].items()},
    },
    "monthly": monthly,
    "forecast": forecast,
    "segments": seg_rows,
    "cohorts": cohort_data,
    "retention_benchmark": retention_benchmark,   # ➕ เพิ่ม
    "regions": region_rows,                       # ➕ เพิ่ม
    "anomalies": anomalies,
    "region_matrix": region_matrix,               # คงไว้ (frontend เดิมยังอ่านอยู่)
}
with open(OUT_JSON, "w") as f:
    json.dump(dashboard, f, indent=1, default=float)

print(f"OK -> {OUT_JSON}: {len(monthly)} months, {len(cohort_data)} cohorts, "
      f"{len(anomalies)} anomalies, "
      f"benchmark ages: {list(retention_benchmark.keys())}")   # (optional) ปรับ print ให้เห็น field ใหม่



# In[2]:


customers.groupby("region")["monthly_fee"].mean()


# In[ ]:




