# app.py
import streamlit as st
import pandas as pd
import joblib
from scipy.sparse import hstack

# --- Streamlit page config ---
st.set_page_config(page_title="ROI Predictor", layout="wide")

# --- Custom CSS for styling ---
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .st-emotion-cache-16txtl3 {
        padding: 2rem 1.5rem;
    }
    h1 {
        color: #4A90E2;
        text-align: center;
    }
    .stButton>button {
        background-color: #4A90E2;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        transition-duration: 0.4s;
    }
    .stButton>button:hover {
        background-color: #357ABD;
        color: white;
    }
    .roi-output {
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        color: #4A90E2;
        padding: 1rem;
        border: 2px solid #4A90E2;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
     .st-emotion-cache-1vo6xi6{
            }       
</style>
""", unsafe_allow_html=True)

st.title("📈 Marketing Campaign ROI Predictor")

# Placeholder for the ROI output
roi_placeholder = st.empty()

st.write("Enter your campaign details below and get a predicted ROI!")

with st.expander("📊 Model Performance Metrics"):
    st.metric(label="R² Score", value="0.8024")
    st.write(" My model explains ~80.24% of the variance in ROI, which is solid for a real-world marketing dataset.")
    st.metric(label="Mean Squared Error (MSE)", value="0.5946")
    st.write("The R² score indicates that the model explains approximately 80.24% of the variance in the ROI, suggesting strong predictive accuracy.")

# === Load saved components ===
model = joblib.load("rf_model_safe.pkl")   # Random Forest safe version
encoder = joblib.load("encoder.pkl")       # OneHotEncoder for minimal features
scaler = joblib.load("scaler.pkl")         # StandardScaler

# --- Columns used in minimal feature model ---
NUMERIC_COLS = ['Duration', 'Acquisition_Cost', 'Clicks', 'Impressions', 'Conversion_Rate', 'Engagement_Score']
CATEGORICAL_COLS = ['Campaign_Type', 'Target_Audience', 'Channel_Used', 'Location', 'Customer_Segment']

# --- Main Area for Inputs ---
st.header("Campaign Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Campaign Setup")
    campaign_type = st.selectbox("Campaign Type", ["Social Media", "Email", "Search", "Display", "Influencer"])
    channel_used = st.selectbox("Channel Used", ["Instagram", "Email", "Google Ads"])
    duration = st.number_input("Duration (days)", min_value=1, max_value=365, value=30)
    acquisition_cost = st.number_input("Acquisition Cost ($)", min_value=0.0, value=12500.0, step=100.0, format="%.2f")

with col2:
    st.subheader("Audience")
    target_audience = st.selectbox("Target Audience", ["All Ages", "Men 18-24", "Men 25-34", "Women 25-34", "Women 35-44"])
    location = st.selectbox("Location", ["Los Angeles", "New York", "Chicago", "Miami", "Houston"])
    customer_segment = st.selectbox("Customer Segment", ["Foodies", "Tech Enthusiasts", "Outdoor Adventurers", "Health & Wellness", "Fashionistas"])

with col3:
    st.subheader("Performance Metrics")
    impressions = st.number_input("Impressions", min_value=0, value=100000, step=1000)
    clicks = st.number_input("Clicks", min_value=0, value=5000, step=100)
    conversion_rate = st.slider("Conversion Rate", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
    engagement_score = st.slider("Engagement Score", min_value=0.0, max_value=10.0, value=7.5, step=0.1)


# --- Prepare Input DataFrame ---
input_df = pd.DataFrame([{
    "Campaign_Type": campaign_type,
    "Target_Audience": target_audience,
    "Channel_Used": channel_used,
    "Location": location,
    "Customer_Segment": customer_segment,
    "Duration": duration,
    "Acquisition_Cost": acquisition_cost,
    "Clicks": clicks,
    "Impressions": impressions,
    "Conversion_Rate": conversion_rate,
    "Engagement_Score": engagement_score
}])

# --- Feature Engineering ---
input_df['CTR'] = input_df['Clicks'] / input_df['Impressions']
# Since we don't have the actual ROI to calculate ROI_per_Cost, we'll have to make an assumption or use a placeholder.
# For now, let's use a placeholder value. This is a limitation of the current app design.
input_df['ROI_per_Cost'] = 0 

# --- Encode Categorical Features ---
try:
    X_encoded_sparse = encoder.transform(input_df[CATEGORICAL_COLS])
except ValueError as e:
    st.error(f"Encoder error: {e}")
    st.stop()

# --- Numeric Features ---
X_numeric = input_df[NUMERIC_COLS + ['CTR', 'ROI_per_Cost']].astype(float)

# --- Combine numeric + encoded ---
X_combined = hstack([X_numeric, X_encoded_sparse])

# --- Scale ---
try:
    X_scaled = scaler.transform(X_combined)
except ValueError as e:
    st.error(f"Scaler error: {e}")
    st.stop()

# --- Prediction Button and Output ---
if st.button("Predict ROI"):
    # --- Encode Categorical Features ---
    try:
        X_encoded_sparse = encoder.transform(input_df[CATEGORICAL_COLS])
    except ValueError as e:
        st.error(f"Encoder error: {e}")
        st.stop()

    # --- Numeric Features ---
    X_numeric = input_df[NUMERIC_COLS + ['CTR', 'ROI_per_Cost']].astype(float)

    # --- Combine numeric + encoded ---
    X_combined = hstack([X_numeric, X_encoded_sparse])

    # --- Scale ---
    try:
        X_scaled = scaler.transform(X_combined)
    except ValueError as e:
        st.error(f"Scaler error: {e}")
        st.stop()

    # --- Debug feature count ---
    n_model_features = model.n_features_in_
    n_input_features = X_scaled.shape[1]

    if n_input_features != n_model_features:
        st.warning(f"❌ Feature mismatch! Model expects {n_model_features}, but input has {n_input_features}. Prediction skipped.")
    else:
        # --- Predict ROI ---
        predicted_roi = model.predict(X_scaled)[0]
        net_profit = predicted_roi * acquisition_cost
        
        roi_placeholder.markdown(
            f"""
            <div class='roi-output'>
                Predicted ROI: {predicted_roi:.2f}
                <br>
                <span style='font-size: 1.5rem; color: #555;'>Net Profit: ${net_profit:,.2f}</span>
            </div>
            """, 
            unsafe_allow_html=True
        )
