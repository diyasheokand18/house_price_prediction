import streamlit as st
import joblib
import numpy as np

# Load Model
model = joblib.load("house_model.pkl")

# Page Configuration
st.set_page_config(
    page_title="HomeValue AI",
    page_icon="🏡",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>

.stApp{
background: linear-gradient(135deg,#F5F7FA,#E3F2FD,#D4F1F4);
}

h1{
text-align:center;
color:white;
font-size:48px;
}

h3{
text-align:center;
color:#E5E7EB;
}

div[data-testid="stNumberInput"]{
background:white;
padding:12px;
border-radius:12px;
}

div[data-testid="stSelectbox"]{
background:white;
padding:12px;
border-radius:12px;
}

.stButton>button{
width:100%;
background:#00C853;
color:white;
font-size:22px;
font-weight:bold;
border-radius:12px;
height:60px;
}

.stButton>button:hover{
background:#009624;
}

.result{
background:white;
padding:20px;
border-radius:15px;
text-align:center;
font-size:30px;
font-weight:bold;
color:#0F172A;
box-shadow:0 8px 25px rgba(0,0,0,0.15);
border-left:6px solid #2563EB;
}

</style>
""", unsafe_allow_html=True)

# Heading
st.title("🏡 HomeValue AI")
st.subheader("Smart House Price Prediction using Machine Learning")

st.write("Fill the property details below to predict the estimated house price.")

col1, col2 = st.columns(2)

with col1:
    bedrooms = st.number_input("🛏 Bedrooms",1,10,3)
    bathrooms = st.number_input("🛁 Bathrooms",1.0,10.0,2.0)
    sqft_living = st.number_input("🏠 Living Area (sqft)",300,10000,1500)
    floors = st.number_input("🏢 Floors",1.0,5.0,1.0)

with col2:
    sqft_lot = st.number_input("🌳 Lot Area (sqft)",500,100000,5000)
    waterfront = st.selectbox("🌊 Waterfront",[0,1])
    view = st.selectbox("🌄 View",[0,1,2,3,4])
    condition = st.selectbox("⭐ Condition",[1,2,3,4,5])

st.write("")

if st.button("🔍 Predict House Price"):

    features=np.array([[
        bedrooms,
        bathrooms,
        sqft_living,
        sqft_lot,
        floors,
        waterfront,
        view,
        condition
    ]])

    prediction=model.predict(features)

    st.markdown(f"""
    <div class="result">
    💰 Estimated House Price <br><br>
    <span style='color:#16A34A;font-size:40px;'>
    ${prediction[0]:,.2f}
    </span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown(
"""
<center>

### 👨‍💻 Developed By

**Diya Sheokand**

**House Price Prediction using Random Forest**

</center>
""",
unsafe_allow_html=True)