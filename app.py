import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="ChurnAI | Customer Churn Prediction",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("churn_ann_model.keras")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")

    return model, scaler, feature_columns


try:
    model, scaler, feature_columns = load_model()
    model_status = True
    model_error = None

except Exception as e:
    model_status = False
    model_error = str(e)


# =========================================================
# CUSTOM CSS
# =========================================================

st.html("""
<style>

/* =====================================================
   GLOBAL
===================================================== */

.stApp {
    background: #f5f7fc;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1450px;
}

/* Normal text */
.stMarkdown,
.stText,
p {
    font-size: 17px;
}

/* =====================================================
   STREAMLIT HEADINGS
===================================================== */

h1 {
    font-size: 36px !important;
}

h2 {
    font-size: 29px !important;
}

h3 {
    font-size: 24px !important;
}

/* =====================================================
   SIDEBAR
===================================================== */

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #111b3d 0%,
        #0c1632 100%
    );
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

.sidebar-title {
    font-size: 34px;
    font-weight: 800;
    margin-bottom: 8px;
}

.sidebar-subtitle {
    font-size: 16px;
    color: #cbd5e1 !important;
    line-height: 1.6;
}

.sidebar-box {
    margin-top: 40px;
    padding: 25px 18px;
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 15px;
    text-align: center;
    background: rgba(255,255,255,0.04);
    font-size: 16px;
}

.sidebar-footer {
    margin-top: 100px;
    font-size: 14px;
    color: #94a3b8 !important;
}

/* Sidebar radio */
section[data-testid="stSidebar"] label {
    font-size: 17px !important;
}

/* =====================================================
   HEADER
===================================================== */

.top-header {
    background: white;
    padding: 15px 24px;
    border-radius: 15px;
    margin-bottom: 20px;
    border: 1px solid #e5e7eb;
    font-size: 17px;
}

/* =====================================================
   HERO
===================================================== */

.hero {
    background: linear-gradient(
        110deg,
        #173b83 0%,
        #123c91 45%,
        #4032a8 100%
    );

    border-radius: 15px;
    padding: 32px 36px;
    color: white;
    margin-bottom: 20px;

    box-shadow: 0 10px 30px rgba(32,55,130,0.18);
}

.hero h1 {
    color: white;
    font-size: 38px !important;
    margin-bottom: 12px;
}

.hero p {
    color: #e2e8f0;
    font-size: 18px;
    line-height: 1.7;
}

.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25);
    padding: 10px 17px;
    border-radius: 30px;
    margin-top: 8px;
    font-size: 16px;
}

/* =====================================================
   METRIC CARDS
===================================================== */

.metric-card {
    background: white;
    border-radius: 15px;
    padding: 22px;
    border: 1px solid #e5e7eb;
    min-height: 140px;

    box-shadow: 0 5px 18px rgba(0,0,0,0.04);
}

.metric-icon {
    font-size: 30px;
}

.metric-title {
    color: #64748b;
    font-size: 16px;
    margin-top: 6px;
}

.metric-value {
    color: #15377e;
    font-size: 33px;
    font-weight: 800;
    margin-top: 5px;
}

.metric-sub {
    color: #94a3b8;
    font-size: 14px;
}

/* =====================================================
   CARDS
===================================================== */

.card {
    background: white;
    border-radius: 15px;
    padding: 26px;
    border: 1px solid #e2e8f0;
    margin-top: 18px;

    box-shadow: 0 5px 20px rgba(0,0,0,0.035);
}

.card-title {
    color: #173b83;
    font-size: 22px;
    font-weight: 800;
    margin-bottom: 14px;
}

.card-text {
    color: #475569;
    font-size: 17px;
    line-height: 1.8;
}

/* =====================================================
   WORKFLOW
===================================================== */

.workflow {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 15px;
    padding: 24px;
    margin-top: 18px;
}

.step {
    background: #f8fafc;
    border-radius: 12px;
    padding: 18px 10px;
    text-align: center;
    border: 1px solid #edf0f5;
    min-height: 135px;
}

.step-number {
    width: 38px;
    height: 38px;
    line-height: 38px;
    margin: auto;
    border-radius: 50%;
    background: #2463eb;
    color: white;
    font-weight: bold;
    font-size: 16px;
}

.step-title {
    font-weight: 700;
    color: #1e3a8a;
    margin-top: 10px;
    font-size: 16px;
}

.step-text {
    color: #64748b;
    font-size: 14px;
    margin-top: 5px;
}

/* =====================================================
   ANN
===================================================== */

.layer {
    padding: 11px;
    border-radius: 8px;
    text-align: center;
    margin: 8px 0;
    font-weight: 700;
    font-size: 15px;
}

.input-layer {
    background: #dbeafe;
    color: #1d4ed8;
}

.hidden-layer {
    background: #ede9fe;
    color: #6d28d9;
}

.output-layer {
    background: #dcfce7;
    color: #15803d;
}

/* =====================================================
   RISK
===================================================== */

.risk-high {
    background: #fff1f2;
    border: 1px solid #fecdd3;
    border-radius: 14px;
    padding: 22px;
}

.risk-low {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 14px;
    padding: 22px;
}

.risk-title-high {
    color: #be123c;
    font-weight: 800;
    font-size: 20px;
}

.risk-title-low {
    color: #15803d;
    font-weight: 800;
    font-size: 20px;
}

/* =====================================================
   TECHNOLOGY
===================================================== */

.tech {
    text-align: center;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 17px 5px;
}

.tech-icon {
    font-size: 30px;
}

.tech-name {
    font-weight: 700;
    color: #334155;
    font-size: 15px;
}

/* =====================================================
   FOOTER
===================================================== */

.footer {
    margin-top: 35px;
    padding: 17px;
    border-top: 1px solid #e2e8f0;
    color: #64748b;
    font-size: 14px;
    text-align: center;
}

/* =====================================================
   PREDICTION
===================================================== */

.prediction-result {
    padding: 35px;
    border-radius: 18px;
    text-align: center;
    margin-top: 20px;
}

.prediction-result h1 {
    font-size: 44px !important;
}

.prediction-result h2 {
    font-size: 30px !important;
}

/* =====================================================
   INPUTS
===================================================== */

div[data-baseweb="select"] {
    font-size: 17px !important;
}

div[data-baseweb="select"] * {
    font-size: 17px !important;
}

input {
    font-size: 17px !important;
}

textarea {
    font-size: 17px !important;
}

label {
    font-size: 17px !important;
    font-weight: 600 !important;
}

/* =====================================================
   BUTTONS
===================================================== */

button {
    font-size: 17px !important;
    font-weight: 600 !important;
}

/* =====================================================
   LOGIN
===================================================== */

.login-title {
    font-size: 36px;
}

.login-subtitle {
    font-size: 17px;
}

</style>
""")


# =========================================================
# SESSION STATE
# =========================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "page" not in st.session_state:
    st.session_state.page = "🏠 Dashboard"


# =========================================================
# LOGIN PAGE
# =========================================================

def login_page():

    st.html("<br><br>")

    col1, col2, col3 = st.columns([1, 1.2, 1])

    with col2:

        st.html("""
        <div style="
            background:white;
            padding:40px;
            border-radius:20px;
            border:1px solid #e2e8f0;
            box-shadow:0 15px 40px rgba(0,0,0,0.08);
            text-align:center;
        ">

        <div style="font-size:65px;">🤖</div>

        <h1 style="
            color:#173b83;
            font-size:40px !important;
        ">
        ChurnAI
        </h1>

        <p style="
            color:#64748b;
            font-size:18px;
        ">
        Customer Churn Prediction System
        </p>

        </div>
        """)

        st.write("")

        username = st.text_input(
            "👤 Username",
            placeholder="Enter username"
        )

        password = st.text_input(
            "🔐 Password",
            type="password",
            placeholder="Enter password"
        )

        if st.button(
            "🚀 Login",
            use_container_width=True,
            type="primary"
        ):

            if username == "admin" and password == "admin123":

                st.session_state.logged_in = True
                st.session_state.page = "🏠 Dashboard"

                st.rerun()

            else:
                st.error(
                    "❌ Invalid username or password"
                )


# =========================================================
# SIDEBAR
# =========================================================

def sidebar():

    with st.sidebar:

        st.html("""
        <div class="sidebar-title">
        🤖 ChurnAI
        </div>

        <div class="sidebar-subtitle">
        Customer Churn<br>
        Prediction System
        </div>
        """)

        st.write("")

        pages = [
            "🏠 Dashboard",
            "🤖 Prediction",
            "ℹ️ About Project"
        ]

        # Find current page
        current_page = st.session_state.get(
            "page",
            "🏠 Dashboard"
        )

        if current_page not in pages:
            current_page = "🏠 Dashboard"

        page = st.radio(
            "Navigation",
            pages,
            index=pages.index(current_page),
            label_visibility="collapsed"
        )

        # Save selected page
        st.session_state.page = page

        st.html("""
        <div class="sidebar-box">

            <div style="font-size:50px;">
            🧠
            </div>

            <b style="font-size:17px;">
            Smarter Insights<br>
            Better Retention
            </b>

            <p style="
                font-size:14px;
                color:#cbd5e1 !important;
            ">
            Predict today,<br>
            retain tomorrow.
            </p>

        </div>
        """)

        st.html("""
        <div class="sidebar-footer">
        ChurnAI v1.0<br><br>
        ❤️ Made for a better tomorrow
        </div>
        """)

        st.write("")

        if st.button(
            "🚪 Logout",
            use_container_width=True
        ):

            st.session_state.logged_in = False
            st.session_state.page = "🏠 Dashboard"

            st.rerun()

    return page


# =========================================================
# DASHBOARD
# =========================================================

def dashboard():

    # Header
    st.html("""
    <div class="top-header">
        🔍 &nbsp;
        <span style="color:#94a3b8;">
        Customer Churn Analytics Dashboard
        </span>
    </div>
    """)

    # Hero
    st.html("""
    <div class="hero">

        <h1>👋 Welcome to ChurnAI</h1>

        <p>
        Hello admin! 👋<br>
        This dashboard provides an overview of the Customer Churn
        Prediction System developed using an Artificial Neural Network.
        </p>

        <div class="hero-badge">
        🧠 AI Powered &nbsp; | &nbsp;
        📊 Telecom Customer Analytics
        </div>

    </div>
    """)

    # =====================================================
    # METRICS
    # =====================================================

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.html("""
        <div class="metric-card">

            <div class="metric-icon">
            👥
            </div>

            <div class="metric-title">
            Customers
            </div>

            <div class="metric-value">
            7,043
            </div>

            <div class="metric-sub">
            Dataset Records
            </div>

        </div>
        """)

    with c2:
        st.html("""
        <div class="metric-card">

            <div class="metric-icon">
            📊
            </div>

            <div class="metric-title">
            Features
            </div>

            <div class="metric-value">
            21
            </div>

            <div class="metric-sub">
            Original Features
            </div>

        </div>
        """)

    with c3:
        st.html("""
        <div class="metric-card">

            <div class="metric-icon">
            🧠
            </div>

            <div class="metric-title">
            Model
            </div>

            <div class="metric-value">
            ANN
            </div>

            <div class="metric-sub">
            Deep Learning
            </div>

        </div>
        """)

    with c4:
        st.html("""
        <div class="metric-card">

            <div class="metric-icon">
            🎯
            </div>

            <div class="metric-title">
            Output
            </div>

            <div class="metric-value">
            Binary
            </div>

            <div class="metric-sub">
            Churn / No Churn
            </div>

        </div>
        """)

    # =====================================================
    # WHAT DOES CHURNAI DO + ANN
    # =====================================================

    left, right = st.columns(2)

    with left:

        st.html("""
        <div class="card">

            <div class="card-title">
            💡 What Does ChurnAI Do?
            </div>

            <div class="card-text">

            ChurnAI predicts whether a telecom customer
            is likely to leave the service.

            <br><br>

            The system analyzes customer information such as:

            <br><br>

            ✔ Customer tenure<br>
            ✔ Contract type<br>
            ✔ Monthly charges<br>
            ✔ Internet service<br>
            ✔ Payment method<br>
            ✔ Customer support services

            </div>

        </div>
        """)

    with right:

        st.html("""
        <div class="card">

            <div class="card-title">
            🧠 Artificial Neural Network
            </div>

            <div class="card-text">

            The prediction engine uses an Artificial Neural
            Network to learn complex patterns in customer data.

            </div>

            <br>

            <b style="
                color:#334155;
                font-size:17px;
            ">
            Model Architecture
            </b>

            <div class="layer input-layer">
            Input Features
            </div>

            <div class="layer hidden-layer">
            Dense Layer — 64 Neurons
            </div>

            <div class="layer hidden-layer">
            Dense Layer — 32 Neurons
            </div>

            <div class="layer hidden-layer">
            Dense Layer — 16 Neurons
            </div>

            <div class="layer output-layer">
            Sigmoid Output
            </div>

        </div>
        """)

    # =====================================================
    # WORKFLOW
    # =====================================================

    st.html("""
    <div class="workflow">

        <div class="card-title">
        ⚙️ Machine Learning Workflow
        </div>

    </div>
    """)

    steps = [
        (
            "1",
            "🗄️",
            "Data Collection",
            "Telecom customer dataset"
        ),
        (
            "2",
            "🧹",
            "Data Cleaning",
            "Missing values & preprocessing"
        ),
        (
            "3",
            "💻",
            "Feature Encoding",
            "Convert categorical data"
        ),
        (
            "4",
            "⚙️",
            "ANN Training",
            "Train neural network"
        ),
        (
            "5",
            "📈",
            "Prediction",
            "Churn / No Churn"
        )
    ]

    cols = st.columns(5)

    for i, (num, icon, title, text) in enumerate(steps):

        with cols[i]:

            st.html(f"""
            <div class="step">

                <div class="step-number">
                {num}
                </div>

                <div style="
                    font-size:28px;
                    margin-top:8px;
                ">
                {icon}
                </div>

                <div class="step-title">
                {title}
                </div>

                <div class="step-text">
                {text}
                </div>

            </div>
            """)

    # =====================================================
    # RISK + TECHNOLOGY
    # =====================================================

    left, right = st.columns(2)

    with left:

        st.html("""
        <div class="card">

            <div class="card-title">
            ⚠️ Understanding Churn Risk
            </div>

            <div class="risk-high">

                <div class="risk-title-high">
                🔴 High Churn Risk
                </div>

                <p style="
                    color:#475569;
                    font-size:15px;
                ">
                Probability ≥ 50%
                </p>

                <p style="
                    color:#64748b;
                    font-size:15px;
                ">
                Customer has a higher predicted probability
                of leaving the service.
                </p>

                <b style="
                    color:#be123c;
                    font-size:16px;
                ">
                Recommended Actions:
                </b>

                <p style="
                    color:#64748b;
                    font-size:14px;
                ">
                • Retention offers<br>
                • Personalized support<br>
                • Discounts<br>
                • Loyalty benefits
                </p>

            </div>

            <br>

            <div class="risk-low">

                <div class="risk-title-low">
                🟢 Low Churn Risk
                </div>

                <p style="
                    color:#475569;
                    font-size:15px;
                ">
                Probability &lt; 50%
                </p>

                <p style="
                    color:#64748b;
                    font-size:15px;
                ">
                Customer has a lower predicted probability
                of leaving the service.
                </p>

                <b style="
                    color:#15803d;
                    font-size:16px;
                ">
                Recommended Actions:
                </b>

                <p style="
                    color:#64748b;
                    font-size:14px;
                ">
                • Continue engagement<br>
                • Maintain service quality<br>
                • Build customer relationship
                </p>

            </div>

        </div>
        """)

    with right:

        st.html("""
        <div class="card">

            <div class="card-title">
            🛠️ Technology Stack
            </div>

        </div>
        """)

        tech_cols = st.columns(5)

        technologies = [
            ("🐍", "Python"),
            ("🧠", "TensorFlow"),
            ("🐼", "Pandas"),
            ("📊", "Scikit-learn"),
            ("🌐", "Streamlit")
        ]

        for i, (icon, name) in enumerate(technologies):

            with tech_cols[i]:

                st.html(f"""
                <div class="tech">

                    <div class="tech-icon">
                    {icon}
                    </div>

                    <div class="tech-name">
                    {name}
                    </div>

                </div>
                """)

        st.write("")

        st.html("""
        <div style="
            background:#eff6ff;
            border:1px solid #bfdbfe;
            border-radius:13px;
            padding:22px;
        ">

            <div style="
                color:#1d4ed8;
                font-size:20px;
                font-weight:800;
            ">
            🚀 Quick Start
            </div>

            <p style="
                color:#64748b;
                font-size:16px;
            ">
            Ready to analyze a customer?
            </p>

        </div>
        """)

        st.write("")

        if st.button(
            "🤖 Open Churn Prediction →",
            use_container_width=True,
            type="primary"
        ):

            st.session_state.page = "🤖 Prediction"
            st.rerun()

    # =====================================================
    # FOOTER
    # =====================================================

    st.html("""
    <div class="footer">

        <b>ChurnAI</b>
        &nbsp; | &nbsp;

        Customer Churn Prediction using
        Artificial Neural Network

        &nbsp; | &nbsp;

        v1.0

    </div>
    """)


# =========================================================
# PREDICTION PAGE
# =========================================================

def prediction_page():

    st.title("🤖 Customer Churn Prediction")

    st.write(
        "Enter customer information below and let the ANN model "
        "predict the churn probability."
    )

    st.divider()

    # =====================================================
    # CUSTOMER DETAILS
    # =====================================================

    st.subheader("👤 Customer Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

    with col2:

        senior_citizen = st.selectbox(
            "Senior Citizen",
            ["No", "Yes"]
        )

    with col3:

        partner = st.selectbox(
            "Partner",
            ["No", "Yes"]
        )

    col1, col2, col3 = st.columns(3)

    with col1:

        dependents = st.selectbox(
            "Dependents",
            ["No", "Yes"]
        )

    with col2:

        tenure = st.number_input(
            "Tenure (Months)",
            min_value=0,
            max_value=72,
            value=12
        )

    with col3:

        phone_service = st.selectbox(
            "Phone Service",
            ["No", "Yes"]
        )

    # =====================================================
    # INTERNET
    # =====================================================

    st.subheader("🌐 Internet & Services")

    col1, col2, col3 = st.columns(3)

    with col1:

        multiple_lines = st.selectbox(
            "Multiple Lines",
            [
                "No phone service",
                "No",
                "Yes"
            ]
        )

    with col2:

        internet_service = st.selectbox(
            "Internet Service",
            [
                "DSL",
                "Fiber optic",
                "No"
            ]
        )

    with col3:

        online_security = st.selectbox(
            "Online Security",
            [
                "No internet service",
                "No",
                "Yes"
            ]
        )

    col1, col2, col3 = st.columns(3)

    with col1:

        online_backup = st.selectbox(
            "Online Backup",
            [
                "No internet service",
                "No",
                "Yes"
            ]
        )

    with col2:

        device_protection = st.selectbox(
            "Device Protection",
            [
                "No internet service",
                "No",
                "Yes"
            ]
        )

    with col3:

        tech_support = st.selectbox(
            "Tech Support",
            [
                "No internet service",
                "No",
                "Yes"
            ]
        )

    col1, col2 = st.columns(2)

    with col1:

        streaming_tv = st.selectbox(
            "Streaming TV",
            [
                "No internet service",
                "No",
                "Yes"
            ]
        )

    with col2:

        streaming_movies = st.selectbox(
            "Streaming Movies",
            [
                "No internet service",
                "No",
                "Yes"
            ]
        )

    # =====================================================
    # CONTRACT
    # =====================================================

    st.subheader("💳 Contract & Billing")

    col1, col2, col3 = st.columns(3)

    with col1:

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

    with col2:

        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["No", "Yes"]
        )

    with col3:

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    col1, col2 = st.columns(2)

    with col1:

        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            value=70.0
        )

    with col2:

        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            value=float(monthly_charges * tenure)
        )

    st.write("")

    # =====================================================
    # PREDICT
    # =====================================================

    if st.button(
        "🚀 Predict Customer Churn",
        use_container_width=True,
        type="primary"
    ):

        # Check model
        if not model_status:

            st.error(
                "Model files could not be loaded. "
                "Please check "
                "churn_ann_model.keras, scaler.pkl "
                "and feature_columns.pkl."
            )

            return

        try:

            # =================================================
            # CREATE INPUT
            # =================================================

            input_data = pd.DataFrame({

                "gender": [gender],

                "SeniorCitizen": [
                    1 if senior_citizen == "Yes" else 0
                ],

                "Partner": [partner],

                "Dependents": [dependents],

                "tenure": [tenure],

                "PhoneService": [phone_service],

                "MultipleLines": [multiple_lines],

                "InternetService": [internet_service],

                "OnlineSecurity": [online_security],

                "OnlineBackup": [online_backup],

                "DeviceProtection": [device_protection],

                "TechSupport": [tech_support],

                "StreamingTV": [streaming_tv],

                "StreamingMovies": [streaming_movies],

                "Contract": [contract],

                "PaperlessBilling": [paperless_billing],

                "PaymentMethod": [payment_method],

                "MonthlyCharges": [monthly_charges],

                "TotalCharges": [total_charges]

            })

            # =================================================
            # ONE-HOT ENCODING
            # =================================================

            input_encoded = pd.get_dummies(
                input_data
            )

            # =================================================
            # MATCH TRAINING COLUMNS
            # =================================================

            input_encoded = input_encoded.reindex(
                columns=feature_columns,
                fill_value=0
            )

            # =================================================
            # SCALING
            # =================================================

            input_scaled = scaler.transform(
                input_encoded
            )

            # =================================================
            # PREDICTION
            # =================================================

            prediction = model.predict(
                input_scaled,
                verbose=0
            )

            probability = float(
                prediction[0][0]
            )

            churn_probability = probability * 100

            # =================================================
            # RESULT
            # =================================================

            st.divider()

            if probability >= 0.50:

                st.html(f"""

                <div class="prediction-result"
                    style="
                    background:#fff1f2;
                    border:2px solid #fecdd3;
                    ">

                    <div style="
                        font-size:65px;
                    ">
                    ⚠️
                    </div>

                    <h2 style="
                        color:#be123c;
                    ">
                    High Churn Risk
                    </h2>

                    <h1 style="
                        color:#be123c;
                    ">
                    {churn_probability:.2f}%
                    </h1>

                    <p style="
                        color:#64748b;
                        font-size:17px;
                    ">
                    Probability that this customer
                    may leave the service.
                    </p>

                </div>

                """)

                st.progress(
                    min(churn_probability / 100, 1.0)
                )

                st.warning(
                    "Recommended: Consider retention offers, "
                    "personalized support and loyalty benefits."
                )

            else:

                st.html(f"""

                <div class="prediction-result"
                    style="
                    background:#f0fdf4;
                    border:2px solid #bbf7d0;
                    ">

                    <div style="
                        font-size:65px;
                    ">
                    ✅
                    </div>

                    <h2 style="
                        color:#15803d;
                    ">
                    Low Churn Risk
                    </h2>

                    <h1 style="
                        color:#15803d;
                    ">
                    {churn_probability:.2f}%
                    </h1>

                    <p style="
                        color:#64748b;
                        font-size:17px;
                    ">
                    Probability that this customer
                    may leave the service.
                    </p>

                </div>

                """)

                st.progress(
                    min(churn_probability / 100, 1.0)
                )

                st.success(
                    "Customer shows relatively low churn risk. "
                    "Continue good service and engagement."
                )

        except Exception as e:

            st.error(
                f"Prediction failed: {str(e)}"
            )


# =========================================================
# ABOUT PAGE
# =========================================================

def about_page():

    st.title("ℹ️ About ChurnAI")

    st.html("""
    <div class="card">

        <div class="card-title">
        🎯 Project Objective
        </div>

        <div class="card-text">

        The objective of this project is to develop a machine
        learning based Customer Churn Prediction System using an
        <b>Artificial Neural Network (ANN)</b>.

        <br><br>

        The system predicts whether a telecom customer is likely
        to leave the company based on customer demographic,
        service and billing information.

        </div>

    </div>
    """)

    st.subheader("🧠 Model Architecture")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Input",
            "Features"
        )

    with col2:
        st.metric(
            "Hidden Layer 1",
            "64 Neurons"
        )

    with col3:
        st.metric(
            "Hidden Layer 2",
            "32 Neurons"
        )

    with col4:
        st.metric(
            "Output",
            "Sigmoid"
        )

    st.subheader("🔄 Project Workflow")

    workflow = pd.DataFrame({

        "Step": [
            "1. Data Collection",
            "2. Data Cleaning",
            "3. Feature Encoding",
            "4. Feature Scaling",
            "5. ANN Training",
            "6. Model Evaluation",
            "7. Prediction"
        ],

        "Description": [
            "Telco customer dataset",
            "Missing values and preprocessing",
            "Convert categorical variables",
            "StandardScaler",
            "Artificial Neural Network",
            "Accuracy, Precision, Recall, F1",
            "Churn / No Churn"
        ]

    })

    st.dataframe(
        workflow,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("🛠️ Technologies Used")

    st.info(
        "Python • Pandas • NumPy • Scikit-learn • "
        "TensorFlow/Keras • Streamlit"
    )

    st.subheader("📌 Project Output")

    st.success(
        "The model provides a churn probability and classifies "
        "customers into High Churn Risk or Low Churn Risk."
    )


# =========================================================
# MAIN APP
# =========================================================

if not st.session_state.logged_in:

    login_page()

else:

    selected_page = sidebar()

    if selected_page == "🏠 Dashboard":

        dashboard()

    elif selected_page == "🤖 Prediction":

        prediction_page()

    elif selected_page == "ℹ️ About Project":

        about_page()