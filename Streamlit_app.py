import streamlit as st
import pandas as pd
import joblib
from io import BytesIO
from fpdf import FPDF

# Set page configuration
st.set_page_config(
    page_title="AI Predictive Methods for Credit Underwriting",
    page_icon="💸",
    layout="wide"
)

# Load trained model
model_path = 'best_features_model.pkl'
try:
    model = joblib.load(model_path)
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error(f"Model file not found: {model_path}")
    st.stop()

# Initialize session state
if "loan_details" not in st.session_state:
    st.session_state["loan_details"] = {
        "full_name": "",
        "email": "",
        "phone": "",
        "cibil_score": 750,
        "income_annum": 5000000,
        "loan_amount": 2000000,
        "loan_term": 24,
        "loan_percent_income": 20.0,
        "active_loans": 1,
        "gender": "Men",
        "marital_status": "Single",
        "employee_status": "employed",
        "residence_type": "OWN",
        "loan_purpose": "Personal",
        "emi": None,
        "id_proof": None,
        "address_proof": None
    }

# Navigation menu
step = st.radio(
    "Navigate through the steps:",
    ["Personal Information", "Loan Details", "Upload Documents", "Final Decision"]
)

# Step 1: Personal Information
if step == "Personal Information":
    st.markdown("### Step 1: Personal Information")
    st.session_state["loan_details"]["full_name"] = st.text_input("Full Name")
    st.session_state["loan_details"]["email"] = st.text_input("Email Address")
    st.session_state["loan_details"]["phone"] = st.text_input("Phone Number")

# Step 2: Loan Details
elif step == "Loan Details":
    st.markdown("### Step 2: Loan Details")
    st.session_state["loan_details"]["cibil_score"] = st.slider("CIBIL Score (300-900):", 300, 900, 750)
    st.session_state["loan_details"]["income_annum"] = st.number_input("Annual Income (INR):", min_value=0, step=10000)
    st.session_state["loan_details"]["loan_amount"] = st.number_input("Loan Amount (INR):", min_value=0, step=10000)
    st.session_state["loan_details"]["loan_term"] = st.number_input("Loan Term (Months):", min_value=1, step=1)
    st.session_state["loan_details"]["loan_percent_income"] = st.number_input("Loan Percent of Income (%):", min_value=0.0, step=0.1)
    st.session_state["loan_details"]["active_loans"] = st.number_input("Number of Active Loans:", min_value=0, step=1)
    st.session_state["loan_details"]["gender"] = st.selectbox("Gender:", ["Men", "Women"])
    st.session_state["loan_details"]["marital_status"] = st.selectbox("Marital Status:", ["Single", "Married"])
    st.session_state["loan_details"]["employee_status"] = st.selectbox("Employment Status:", ["employed", "self employed", "unemployed", "student"])
    st.session_state["loan_details"]["residence_type"] = st.selectbox("Residence Type:", ["MORTGAGE", "OWN", "RENT"])
    st.session_state["loan_details"]["loan_purpose"] = st.selectbox("Loan Purpose:", ["Vehicle", "Personal", "Home Renovation", "Education", "Medical", "Other"])

# Step 3: Upload Documents
elif step == "Upload Documents":
    st.markdown("### Step 3: Upload Documents")
    st.session_state["loan_details"]["id_proof"] = st.file_uploader("Upload ID Proof")
    st.session_state["loan_details"]["address_proof"] = st.file_uploader("Upload Address Proof")

# Step 4: Final Decision
elif step == "Final Decision":
    st.markdown("### Step 4: Final Decision")
    loan_details = st.session_state["loan_details"]
    input_data = pd.DataFrame({
        "cibil_score": [loan_details["cibil_score"]],
        "income_annum": [loan_details["income_annum"]],
        "loan_amount": [loan_details["loan_amount"]],
        "loan_term": [loan_details["loan_term"]],
        "loan_percent_income": [loan_details["loan_percent_income"]],
        "active_loans": [loan_details["active_loans"]]
    })
    
    try:
        prediction = model.predict(input_data)
        if prediction[0] == 1:
            st.error("Loan Rejected ❌")
        else:
            st.success("Loan Approved ✅")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
