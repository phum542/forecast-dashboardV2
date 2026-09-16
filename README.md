# 📊 SaaS Forecast Dashboard (V2)

Raw customer CSV → Python pipeline → executive dashboard.
No BI tool, no template — pandas + vanilla JS + Chart.js.

**🔗 Live demo:** https://phum542.github.io/forecast-dashboardV2/

## What V2 fixes over V1
- 🎨 **Diverging cohort heatmap** — green = above benchmark, red = below, "—" for cohorts too young to measure (V1 colored missing data as failure)
- 📝 **Executive summary in 3 bullets** — what's improving, what's worsening, what to do about it
- 🚨 **Anomalies with business impact** — "CAC spiked +12% vs typical (~$560)", filterable favorable/unfavorable
- 💰 **Honest unit economics** — LTV:CAC ~2.5 vs the 3.0 benchmark, which drives the recommendation instead of contradicting it
- 📈 **A target line that behaves like a target** — tracks actual growth, so beat/miss actually means something

## Pipeline
`customers.csv` → cohort analysis → linear forecast (95% CI) → retention benchmarks (mature cohorts only) → anomaly detection (empirical 95% + |z|>3) → `dashboard_data.json`

## Run locally
```bash
python pipeline.py        # regenerate dashboard_data.json
python -m http.server     # open http://localhost:8000
