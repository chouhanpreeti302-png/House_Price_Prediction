import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from datetime import date

# =========================
# Page + theme styling
# =========================
st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")

st.markdown("""
<style>
    .title {font-size: 2.6rem; font-weight: 800; margin-bottom: 0.1rem;}
    .subtitle {color:#9aa4b2; font-size:1rem; margin-top:0;}
    .card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 16px;
    }
    .soft {
        color:#9aa4b2;
        font-size:0.9rem;
    }
    .pill {
        display:inline-block; padding:0.25rem 0.6rem;
        border-radius:999px; font-weight:600; font-size:0.85rem;
        border:1px solid rgba(56,189,248,0.25);
        background:rgba(56,189,248,0.10);
        color: rgb(56,189,248);
    }
    .danger-pill {
        border:1px solid rgba(239,68,68,0.25);
        background:rgba(239,68,68,0.10);
        color: rgb(239,68,68);
    }
    .hr {border-bottom:1px solid rgba(255,255,255,0.08); margin: 14px 0;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🏠 House Price Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">CatBoost + consistent preprocessing (label encoding, imputation, scaling) • returns expm1(log_pred)</div>', unsafe_allow_html=True)

# =========================
# Artifact paths
# =========================
MODEL_PATH = "catboost_model.pkl"
ENC_PATH   = "label_encoders.pkl"
IMP_PATH   = "imputer.pkl"
SCL_PATH   = "scaler.pkl"

# =========================
# Feature schema (must match training)
# =========================
FEATURE_ORDER = [
    "Location","Size","Bedrooms","Bathrooms","Year Built","Type",
    "Sold_Year","Sold_Month","Sold_Quarter","Property_Age","Condition_Ordinal"
]
CAT_COLS = ["Location","Type"]
NUM_COLS = ["Size","Bedrooms","Bathrooms","Year Built","Sold_Year","Sold_Month",
            "Sold_Quarter","Property_Age","Condition_Ordinal"]
COND_MAP = {"Poor":0, "Fair":1, "Good":2, "New":3}

# =========================
# Sidebar: status + debug
# =========================
st.sidebar.header("⚙️ Controls")
show_debug = st.sidebar.toggle("Show Debug Panel", value=False)

st.sidebar.markdown("### 📦 Artifacts")
for f in [MODEL_PATH, ENC_PATH, IMP_PATH, SCL_PATH]:
    st.sidebar.write("✅" if os.path.exists(f) else "❌", f)

# =========================
# Load artifacts (with helpful errors)
# =========================
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENC_PATH)
    imputer = joblib.load(IMP_PATH)
    scaler = joblib.load(SCL_PATH)
    return model, encoders, imputer, scaler

try:
    model, label_encoders, imputer, scaler = load_artifacts()
    st.sidebar.markdown('<span class="pill">Model Ready</span>', unsafe_allow_html=True)
except Exception as e:
    st.sidebar.markdown('<span class="pill danger-pill">Model NOT Ready</span>', unsafe_allow_html=True)
    st.error("❌ Could not load artifacts.")
    st.exception(e)
    st.info("Your error screenshot shows `No module named catboost` → run: `pip install catboost` in your venv.")
    if show_debug:
        st.sidebar.markdown("### Debug Info")
        st.sidebar.code(os.getcwd())
        st.sidebar.code("\n".join(sorted(os.listdir("."))))
    st.stop()

if show_debug:
    st.sidebar.markdown("### Debug Info")
    st.sidebar.write("Working directory:")
    st.sidebar.code(os.getcwd())
    st.sidebar.write("Files in folder:")
    st.sidebar.code("\n".join(sorted(os.listdir("."))))

# =========================
# Helpers
# =========================
def feature_engineer_row(location, size, bedrooms, bathrooms, year_built, condition, ptype, date_sold):
    dt = pd.to_datetime(date_sold, errors="coerce")
    if pd.isna(dt):
        raise ValueError("Invalid Date Sold. Use YYYY-MM-DD.")

    sold_year = int(dt.year)
    sold_month = int(dt.month)
    sold_quarter = int((sold_month - 1)//3 + 1)

    if condition not in COND_MAP:
        raise ValueError(f"Condition must be one of {list(COND_MAP.keys())}")

    row = {
        "Location": str(location),
        "Size": float(size),
        "Bedrooms": float(bedrooms),
        "Bathrooms": float(bathrooms),
        "Year Built": float(year_built),
        "Type": str(ptype),
        "Sold_Year": sold_year,
        "Sold_Month": sold_month,
        "Sold_Quarter": sold_quarter,
        "Property_Age": sold_year - float(year_built),
        "Condition_Ordinal": float(COND_MAP[condition]),
    }
    return pd.DataFrame([row], columns=FEATURE_ORDER)

def preprocess_one(X: pd.DataFrame):
    Xc = X.copy()

    # Label encoding (reject unseen categories)
    for col in CAT_COLS:
        le = label_encoders[col]
        val = str(Xc.loc[0, col])
        if val not in le.classes_:
            raise ValueError(f"Unknown {col}='{val}'. Known values: {list(le.classes_)}")
        Xc[col] = le.transform([val])[0]

    # Impute
    X_imp = pd.DataFrame(imputer.transform(Xc), columns=Xc.columns)

    # Scale numeric cols
    X_imp[NUM_COLS] = scaler.transform(X_imp[NUM_COLS])

    return Xc, X_imp

def predict_price(X: pd.DataFrame):
    _, X_scaled = preprocess_one(X)
    pred_log = float(model.predict(X_scaled.values)[0])     # model predicts log1p(price)
    pred = float(np.expm1(pred_log))                       # back to real price
    return pred, pred_log, X_scaled

# =========================
# Main UI
# =========================
loc_choices = list(label_encoders["Location"].classes_)
type_choices = list(label_encoders["Type"].classes_)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

left, right = st.columns([1.25, 1])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧾 Enter Property Details")

    c1, c2 = st.columns(2)
    with c1:
        location = st.selectbox("📍 Location", loc_choices)
        ptype = st.selectbox("🏘️ Type", type_choices)
        condition = st.selectbox("🛠️ Condition", list(COND_MAP.keys()), index=2)

    with c2:
        size = st.number_input("📐 Size (sqft)", 100.0, 20000.0, 1800.0, 10.0)
        bedrooms = st.number_input("🛏️ Bedrooms", 0.0, 20.0, 3.0, 1.0)
        bathrooms = st.number_input("🛁 Bathrooms", 0.0, 20.0, 2.0, 1.0)

    year_built = st.number_input("🏗️ Year Built", 1800.0, float(date.today().year), 2005.0, 1.0)
    date_sold = st.date_input("🗓️ Date Sold", value=date(2024, 6, 1))

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🔮 Prediction Output")

    X1 = feature_engineer_row(location, size, bedrooms, bathrooms, year_built, condition, ptype, date_sold)

    if st.button("💰 Predict", type="primary", use_container_width=True):
        try:
            pred, pred_log, X_scaled = predict_price(X1)

            k1, k2 = st.columns(2)
            k1.metric("Predicted Price", f"{pred:,.2f}")
            k2.metric("Predicted log1p(Price)", f"{pred_log:.4f}")

            st.caption("✅ Your model was trained on log1p(price), and we return expm1(pred_log).")

            with st.expander("Advanced: See engineered + scaled features"):
                st.markdown("**Engineered row**")
                st.dataframe(X1, use_container_width=True)
                st.markdown("**Scaled row (what model receives)**")
                st.dataframe(pd.DataFrame(X_scaled, columns=X_scaled.columns), use_container_width=True)

        except Exception as e:
            st.error(str(e))

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# =========================
# Batch predictions
# =========================
st.subheader("📦 Batch Predictions (CSV Upload)")
st.markdown('<div class="soft">Required columns: Location, Size, Bedrooms, Bathrooms, Year Built, Condition, Type, Date Sold</div>', unsafe_allow_html=True)

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.dataframe(df.head(20), use_container_width=True)

    required = ["Location","Size","Bedrooms","Bathrooms","Year Built","Condition","Type","Date Sold"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
    else:
        if st.button("Run Batch Prediction", type="primary"):
            preds, errs = [], []
            for _, r in df.iterrows():
                try:
                    Xi = feature_engineer_row(
                        r["Location"], r["Size"], r["Bedrooms"], r["Bathrooms"],
                        r["Year Built"], r["Condition"], r["Type"], r["Date Sold"]
                    )
                    p, _, _ = predict_price(Xi)
                    preds.append(p)
                    errs.append("")
                except Exception as e:
                    preds.append(np.nan)
                    errs.append(str(e))

            out = df.copy()
            out["Predicted_Price"] = preds
            out["Error"] = errs

            st.success(" Done.")
            st.dataframe(out.head(50), use_container_width=True)

            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download predictions CSV",
                data=csv_bytes,
                file_name="house_price_predictions.csv",
                mime="text/csv"
            )
