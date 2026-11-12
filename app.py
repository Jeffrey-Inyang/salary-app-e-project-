"""
Employee Salary Predictor  -  Group 2, ACE-GWARINPA (ADSE Java/AIML, Semester 4)
Web interface for the simple linear regression model built in Salary_Prediction_Group2.ipynb

Run locally:   streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

BASE = Path(__file__).parent

st.set_page_config(page_title="Salary Predictor", page_icon="💼", layout="centered")


# ---------------------------------------------------------------- load assets
@st.cache_resource
def load_model():
    """Load the trained model saved by the notebook (salary_model.joblib)."""
    return joblib.load(BASE / "salary_model.joblib")


@st.cache_data
def load_data():
    return pd.read_csv(BASE / "Salary_Data.csv")


@st.cache_data
def load_summary():
    with open(BASE / "results_summary.json") as f:
        return json.load(f)


model = load_model()
data = load_data()
summary = load_summary()

x = data["YearsExperience"].to_numpy()
y = data["Salary"].to_numpy()
n = len(x)
x_min, x_max = float(x.min()), float(x.max())

slope = float(model.coef_[0])
intercept = float(model.intercept_)

# Residual standard error (used for the 95% prediction interval)
resid = y - model.predict(data[["YearsExperience"]])
s = float(np.sqrt((resid ** 2).sum() / (n - 2)))
sxx = float(((x - x.mean()) ** 2).sum())
t_crit = float(stats.t.ppf(0.975, n - 2))


def predict(years: float):
    """Return (point prediction, lower 95% PI, upper 95% PI)."""
    pred = float(model.predict(pd.DataFrame({"YearsExperience": [years]}))[0])
    se_pred = s * np.sqrt(1 + 1 / n + (years - x.mean()) ** 2 / sxx)
    margin = t_crit * se_pred
    return pred, pred - margin, pred + margin


# ------------------------------------------------------------------------ UI
st.title("💼 Employee Salary Predictor")
st.caption(
    "Simple linear regression trained on 30 records · "
    "Group 2, ACE-GWARINPA · ADSE (Java, AIML) · Project 2"
)

st.markdown("Enter an employee's years of work experience to predict their annual salary.")

years = st.number_input(
    "Years of experience",
    min_value=0.0,
    max_value=50.0,
    value=5.0,
    step=0.5,
    help=f"The model was trained on {x_min:.1f} to {x_max:.1f} years of experience.",
)

if st.button("Predict salary", type="primary"):
    pred, lo, hi = predict(years)
    st.session_state["result"] = (years, pred, lo, hi)

# Show the latest result (persists across reruns)
if "result" in st.session_state:
    yrs, pred, lo, hi = st.session_state["result"]

    st.success(f"### Predicted annual salary: **${pred:,.0f}**")
    st.write(f"95% prediction range: **${lo:,.0f} – ${hi:,.0f}**  (for {yrs:g} years of experience)")

    if yrs < x_min or yrs > x_max:
        st.warning(
            f"⚠️ {yrs:g} years is outside the range the model was trained on "
            f"({x_min:.1f}–{x_max:.1f} years). This is extrapolation, so treat the "
            "result with caution."
        )
    if pred < 0:
        st.error("The model produced a negative salary, which is not meaningful.")

    # Chart: data points + fitted line + the user's prediction
    grid = np.linspace(min(0.0, yrs), max(x_max, yrs) + 0.5, 60)
    line = model.predict(pd.DataFrame({"YearsExperience": grid}))
    st.markdown("**Where this prediction sits on the fitted line**")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(x, y, alpha=0.7, label="Training data")
    ax.plot(grid, line, color="tab:red", label="Fitted line")
    ax.errorbar([yrs], [pred], yerr=[[pred - lo], [hi - pred]], fmt="o", color="black",
                capsize=5, markersize=8, label="Your prediction (95% range)")
    ax.set_xlabel("Years of experience")
    ax.set_ylabel("Salary ($)")
    ax.yaxis.set_major_formatter(lambda v, _: f"${v:,.0f}")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

# ----------------------------------------------------------- model information
with st.expander("About the model"):
    fm = summary["final_model"]
    st.markdown(
        f"""
**Equation:** Salary ≈ **${intercept:,.0f}** + **${slope:,.0f}** × years of experience

- **Slope:** each extra year of experience is associated with about **${slope:,.0f}** more salary.
- **Intercept:** predicted starting salary at 0 years ≈ **${intercept:,.0f}** (an extrapolation, as the data starts at {x_min:.1f} years).
- **R² (all 30 rows):** {fm['all_rows']['R2']:.3f}  ·  **RMSE:** ${fm['all_rows']['RMSE']:,.0f}
- **Cross-validated R²:** {summary['baseline_cv']['CV R² (5-fold x10)']:.3f}  ·  **CV RMSE:** ${summary['baseline_cv']['CV RMSE (5-fold x10)']:,.0f}

**Limitations:** only 30 records and a single input feature. Real salaries also depend on
role, industry, location and education, none of which are in this dataset.
"""
    )

with st.expander("Dataset (Salary_Data.csv)"):
    st.dataframe(data, hide_index=True)
