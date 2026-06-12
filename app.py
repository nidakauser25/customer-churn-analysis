import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ================= CONFIG =================
st.set_page_config(
    page_title="Customer Churn Analysis Platform",
    page_icon="📊",
    layout="wide"
)

# ================= LOAD =================
model = joblib.load("models/churn_model.pkl")
encoders = joblib.load("models/encoders.pkl")

df = pd.read_csv("data/cleaned_customer_churn.csv")
df["Churn"] = df["Churn"].astype(str)

# ================= SESSION =================
if "login" not in st.session_state:
    st.session_state.login = False

def logout():
    st.session_state.login = False

# ================= LOGIN =================
def login_page():

    st.markdown("""
    <style>
    .stApp {
        background: radial-gradient(circle at top, #0B1220, #020617);
    }

    h1 {
        color: #38BDF8 !important;
        text-align: center;
        font-weight: 800;
    }

    label {
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("📊 Customer Churn Platform")
    st.info("Demo Login → Username: admin | Password: admin123")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

    if submit:
        if username.strip() == "admin" and password.strip() == "admin123":
            st.session_state.login = True
            st.rerun()
        else:
            st.error("Invalid credentials")

# ================= GATE =================
if not st.session_state.login:
    login_page()
    st.stop()

# ================= GLOBAL UI =================
st.markdown("""
<style>

.stApp {
    background: radial-gradient(circle at top, #0B1220, #020617);
    color: white;
}

h1, h2, h3 {
    color: #38BDF8 !important;
}

label {
    color: white !important;
}

div[data-baseweb="select"] {
    background-color: white !important;
    border-radius: 8px;
}

div[data-baseweb="select"] div {
    color: black !important;
}

div[role="option"] {
    color: black !important;
    background-color: white !important;
}

div[role="option"]:hover {
    background-color: #E5E7EB !important;
}

[data-testid="stMetric"] {
    background: linear-gradient(145deg, #1E293B, #0F172A);
    padding: 18px;
    border-radius: 16px;
    border: 1px solid #334155;
}

[data-testid="stMetricValue"] {
    color: white !important;
    font-size: 28px !important;
    font-weight: 900;
}

[data-testid="stMetricLabel"] {
    color: #CBD5E1 !important;
}

.stButton>button {
    background: linear-gradient(90deg, #2563EB, #3B82F6);
    color: white;
    border-radius: 10px;
    height: 45px;
    font-weight: 600;
}

section[data-testid="stSidebar"] {
    background-color: #0B1220;
}

/* FIX: White font for sidebar radio navigation menu items */
section[data-testid="stSidebar"] input[type="radio"] + div,
section[data-testid="stSidebar"] input[type="radio"] + div * {
    color: white !important;
}

section[data-testid="stSidebar"] .row-widget.stRadio label {
    color: white !important;
}

section[data-testid="stSidebar"] div.row-widget.stRadio > div > label {
    color: white !important;
}

</style>
""", unsafe_allow_html=True)

# ================= NAV =================
menu = st.sidebar.radio(
    "Navigation",
    ["🏠 Dashboard", "🔮 Predict Churn", "📊 Analytics"]
)

st.sidebar.button("🚪 Logout", on_click=logout)

# ================= DASHBOARD =================
if menu == "🏠 Dashboard":

    st.title("📊 Customer Churn Analysis Platform")

    churn_rate = round(df["Churn"].value_counts(normalize=True).get("Yes", 0) * 100, 2)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Model Accuracy", "81.69%")

    with col2:
        st.metric("Customers", "7043")

    with col3:
        st.metric("Churn Rate", f"{churn_rate}%")

    with col4:
        st.metric("Status", "ACTIVE 🚀")

    st.divider()

    colA, colB = st.columns(2)

    with colA:
        st.info("""
        • Month-to-month contracts → high churn  
        • Fiber optic users → high risk  
        • Long tenure → stable customers  
        • Electronic payment → more churn  
        """)

    with colB:
        churn_counts = df["Churn"].value_counts()

        fig = px.bar(
            x=churn_counts.index,
            y=churn_counts.values,
            color=churn_counts.index,
            color_discrete_map={"Yes": "#EF4444", "No": "#3B82F6"}
        )

        fig.update_layout(
            plot_bgcolor="#0F172A",
            paper_bgcolor="#0F172A",

            legend=dict(
                font=dict(color="white"),
                title=dict(font=dict(color="white"))
            ),

            font=dict(color="white")
        )

        st.plotly_chart(fig, use_container_width=True)

# ================= PREDICTION =================
elif menu == "🔮 Predict Churn":

    st.title("🔮 Prediction Engine")

    gender = st.selectbox("Gender", ["Male", "Female"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    monthly = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)
    total = st.number_input("Total Charges", 0.0, 10000.0, 500.0)

    if st.button("Predict Risk"):

        input_data = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": 0,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": internet,
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": contract,
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": monthly,
            "TotalCharges": total
        }])

        for col in input_data.columns:
            if col in encoders:
                input_data[col] = encoders[col].transform(input_data[col].astype(str))

        prob = model.predict_proba(input_data)[0][1]

        st.metric("Churn Probability", f"{prob*100:.2f}%")

        if prob < 0.30:
            st.success("🟢 LOW RISK")
        elif prob < 0.70:
            st.warning("🟡 MEDIUM RISK")
        else:
            st.error("🔴 HIGH RISK")

# ================= ANALYTICS =================
elif menu == "📊 Analytics":

    st.title("📊 Business Analytics Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(px.histogram(df, x="tenure", color="Churn"), use_container_width=True)

    with col2:
        st.plotly_chart(px.box(df, x="Contract", y="MonthlyCharges", color="Churn"), use_container_width=True)

    st.metric(
        "Revenue at Risk",
        f"${df[df['Churn']=='Yes']['MonthlyCharges'].sum():,.2f}"
    )