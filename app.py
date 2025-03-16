import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Set Page Configuration
st.set_page_config(page_title="Wildlife Species Location Prediction", layout="wide")

# Custom Header
st.markdown("""
    <style>
        .title {
            text-align: center;
            color: #40E0D0;
            font-size: 36px;
            font-weight: bold;
        }
        .sub-title {
            text-align: center;
            color: #444;
            font-size: 20px;
        }
        .highlight-box {
            background-color: #8c39bf;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
            text-align: center;
            font-size: 18px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🐦 Wildlife Species Location Prediction System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'> A Smart AI System for Wildlife Movement Analysis & Prediction</div>", unsafe_allow_html=True)
st.write("")

# Sidebar for File Upload
st.sidebar.header("📂 Upload CSV File")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file:
    # Read CSV
    data = pd.read_csv(uploaded_file)
    data.columns = data.columns.str.strip()  # Clean column names
    st.markdown("<div class='highlight-box'>✅ File Uploaded Successfully!</div>", unsafe_allow_html=True)

    # Display dataset preview
    st.write("### 📌 Dataset Preview")
    st.dataframe(data.head())

    # Dataset Summary
    st.write("### 📊 Dataset Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"- **Total Rows:** {data.shape[0]}")
        st.write(f"- **Total Columns:** {data.shape[1]}")
    with col2:
        st.write("#### 🛠 Missing Values:")
        st.dataframe(data.isnull().sum())

    # Extract Latitude & Longitude
    if "GPS Location" in data.columns:
        data['GPS Location'] = data['GPS Location'].astype(str).fillna('')
        split_locations = data['GPS Location'].str.split(',', expand=True)
        if split_locations.shape[1] == 2:
            data[['latitude', 'longitude']] = split_locations
            data['latitude'] = pd.to_numeric(data['latitude'], errors='coerce')
            data['longitude'] = pd.to_numeric(data['longitude'], errors='coerce')
    elif "latitude" in data.columns and "longitude" in data.columns:
        data['latitude'] = pd.to_numeric(data['latitude'], errors='coerce')
        data['longitude'] = pd.to_numeric(data['longitude'], errors='coerce')
    else:
        st.error("CSV file must contain 'GPS Location' or separate 'latitude' and 'longitude' columns!")
        st.stop()

    # Handle Missing Values
    numeric_cols = data.select_dtypes(include=["number"]).columns
    categorical_cols = data.select_dtypes(include=["object"]).columns

    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
    for col in categorical_cols:
        data[col] = data[col].fillna(data[col].mode()[0])

    # Convert Date Column
    if "Date" in data.columns:
        try:
            data['Date'] = pd.to_datetime(data['Date'], format='%d-%b-%y', errors='coerce')
            data.dropna(subset=['Date'], inplace=True)
            data['Day'] = data['Date'].dt.day
            data['Month'] = data['Date'].dt.month
            data['Year'] = data['Date'].dt.year
        except Exception as e:
            st.error(f"Error converting Date column: {e}")
            st.stop()

    # 📊 Feature Distribution
    st.write("### 📈 Feature Distributions")
    num_columns = data.select_dtypes(include=["number"]).columns
    fig, axes = plt.subplots(nrows=1, ncols=min(3, len(num_columns)), figsize=(18, 5))
    
    for i, col in enumerate(num_columns[:3]):
        sns.histplot(data[col], bins=20, kde=True, ax=axes[i], color="dodgerblue")
        axes[i].set_title(f"Distribution of {col}")
        axes[i].set_xlabel(col)
    
    st.pyplot(fig)

    # Encode Categorical Data
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le

    # Feature Selection
    possible_features = ["Day", "Month", "Year", "Weather", "Food offered"]
    feature_columns = [col for col in possible_features if col in data.columns]

    if not feature_columns:
        st.error("No valid features available for prediction. Please check your dataset.")
        st.stop()

    X = data[feature_columns]
    y_lat = data["latitude"]
    y_lon = data["longitude"]

    # Train-Test Split
    X_train, X_test, y_lat_train, y_lat_test = train_test_split(X, y_lat, test_size=0.2, random_state=42)
    X_train, X_test, y_lon_train, y_lon_test = train_test_split(X, y_lon, test_size=0.2, random_state=42)

    # Train Models
    lat_model = RandomForestRegressor(n_estimators=200, random_state=42)
    lon_model = RandomForestRegressor(n_estimators=200, random_state=42)
    lat_model.fit(X_train, y_lat_train)
    lon_model.fit(X_train, y_lon_train)

    # Predictions & Metrics
    y_lat_pred = lat_model.predict(X_test)
    y_lon_pred = lon_model.predict(X_test)

    st.write("### 📊 Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="📌 Latitude MAE", value=f"{mean_absolute_error(y_lat_test, y_lat_pred):.5f}")
        st.metric(label="📌 Latitude R² Score", value=f"{r2_score(y_lat_test, y_lat_pred):.3f}")
    with col2:
        st.metric(label="📌 Longitude MAE", value=f"{mean_absolute_error(y_lon_test, y_lon_pred):.5f}")
        st.metric(label="📌 Longitude R² Score", value=f"{r2_score(y_lon_test, y_lon_pred):.3f}")

    # Predicted Next Location
    latest_data = X.iloc[-1:].copy()
    future_lat = lat_model.predict(latest_data)[0]
    future_lon = lon_model.predict(latest_data)[0]

    st.write("### 🗺️ Predicted Next Location")
    st.success(f"📍 Latitude: {future_lat:.6f}, Longitude: {future_lon:.6f}")

    # Wildlife Movement Map
    past_locations = data[['latitude', 'longitude']].dropna()
    m = folium.Map(location=[past_locations.iloc[-1]['latitude'], past_locations.iloc[-1]['longitude']], zoom_start=8)

    # Connect all locations with a line
    folium.PolyLine(
        past_locations.values,
        color="blue",
        weight=3,
        opacity=0.7,
        tooltip="Past Movements"
    ).add_to(m)

    # Add markers for all previous locations
    for _, row in past_locations.iterrows():
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            icon=folium.Icon(color="lightblue", icon="cloud")
        ).add_to(m)

    # Start Location Marker
    folium.Marker(
        location=[past_locations.iloc[0]['latitude'], past_locations.iloc[0]['longitude']],
        popup="Starting Point",
        icon=folium.Icon(color="green", icon="flag")
    ).add_to(m)

    # Last Known Location Marker
    folium.Marker(
        location=[past_locations.iloc[-1]['latitude'], past_locations.iloc[-1]['longitude']],
        popup="Last Known Location",
        icon=folium.Icon(color="orange", icon="info-sign")
    ).add_to(m)

    # Predicted Location Marker (Red)
    folium.Marker(
        [future_lat, future_lon],
        popup=f"Predicted Location: ({future_lat:.6f}, {future_lon:.6f})",
        icon=folium.Icon(color="red", icon="map-marker")
    ).add_to(m)

    st_folium(m, returned_objects=[])
