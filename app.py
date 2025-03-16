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

# Set page config
st.set_page_config(page_title="Wildlife Species Location Prediction", layout="wide")
st.title("🐦 Wildlife Species Location Prediction System Made by A10")

st.write("Upload a CSV file to analyze wildlife movement patterns and predict future locations.")

# Upload CSV file
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    # Read CSV
    data = pd.read_csv(uploaded_file)
    data.columns = data.columns.str.strip()  # Clean column names
    st.write("### 📌 Dataset Preview")
    st.write(data.head())

    # Display dataset info
    st.write("### 📊 Dataset Summary")
    st.write(f"- **Total Rows:** {data.shape[0]}")
    st.write(f"- **Total Columns:** {data.shape[1]}")
    st.write("#### Missing Values:")
    st.write(data.isnull().sum())

    # Extract latitude and longitude
    if "GPS Location" in data.columns:
        data['GPS Location'] = data['GPS Location'].astype(str).fillna('')
        split_locations = data['GPS Location'].str.split(',', expand=True)
        if split_locations.shape[1] == 2:  # Ensure correct splitting
            data[['latitude', 'longitude']] = split_locations
            data['latitude'] = pd.to_numeric(data['latitude'], errors='coerce')
            data['longitude'] = pd.to_numeric(data['longitude'], errors='coerce')
        else:
            st.error("Error splitting GPS Location. Ensure format is 'lat,long'")
            st.stop()
    elif "latitude" in data.columns and "longitude" in data.columns:
        data['latitude'] = pd.to_numeric(data['latitude'], errors='coerce')
        data['longitude'] = pd.to_numeric(data['longitude'], errors='coerce')
    else:
        st.error("CSV file must contain 'GPS Location' or separate 'latitude' and 'longitude' columns!")
        st.stop()

    # Handle missing values
    numeric_cols = data.select_dtypes(include=["number"]).columns
    categorical_cols = data.select_dtypes(include=["object"]).columns

    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
    for col in categorical_cols:
        data[col] = data[col].fillna(data[col].mode()[0])

    # Convert 'Date' column
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

    # Feature Distributions
    st.write("### 📈 Feature Distributions")
    num_columns = data.select_dtypes(include=["number"]).columns
    for col in num_columns:
        fig, ax = plt.subplots()
        sns.histplot(data[col], bins=20, kde=True, ax=ax)
        plt.xlabel(col)
        st.pyplot(fig)

    # Encode categorical columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le

    # Define Features (X) and Targets (y)
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

    # Train Random Forest Models
    lat_model = RandomForestRegressor(n_estimators=200, random_state=42)
    lon_model = RandomForestRegressor(n_estimators=200, random_state=42)
    lat_model.fit(X_train, y_lat_train)
    lon_model.fit(X_train, y_lon_train)

    # Make Predictions
    y_lat_pred = lat_model.predict(X_test)
    y_lon_pred = lon_model.predict(X_test)

    # Calculate Metrics
    lat_mae = mean_absolute_error(y_lat_test, y_lat_pred)
    lon_mae = mean_absolute_error(y_lon_test, y_lon_pred)
    lat_r2 = r2_score(y_lat_test, y_lat_pred)
    lon_r2 = r2_score(y_lon_test, y_lon_pred)

    st.write("### 📊 Model Performance")
    st.write(f"- **Latitude Mean Absolute Error (MAE):** {lat_mae:.5f}")
    st.write(f"- **Longitude Mean Absolute Error (MAE):** {lon_mae:.5f}")
    st.write(f"- **Latitude R² Score:** {lat_r2:.3f}")
    st.write(f"- **Longitude R² Score:** {lon_r2:.3f}")

    # Predict Next Location
    latest_data = X.iloc[-1:].copy()
    future_lat = lat_model.predict(latest_data)[0]
    future_lon = lon_model.predict(latest_data)[0]

    st.write("### 🗺️ Predicted Next Location")
    st.write(f"- **Latitude:** {future_lat:.6f}")
    st.write(f"- **Longitude:** {future_lon:.6f}")

    # Feature Importance for Latitude Prediction
    lat_importances = lat_model.feature_importances_
    lon_importances = lon_model.feature_importances_

    feature_names = X.columns

    # Plot Feature Importance for Latitude
    st.write("### 🌟 Feature Importance - Latitude")
    fig, ax = plt.subplots()
    sns.barplot(x=lat_importances, y=feature_names, ax=ax, color="blue")
    sns.barplot(x=lon_importances, y=feature_names, ax=ax, color="red")
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Feature")
    ax.set_title("Feature Importance for Latitude Prediction")
    st.pyplot(fig)

    # Plot Feature Importance for Longitude
    st.write("### 🌟 Feature Importance - Longitude")
    fig, ax = plt.subplots()
    sns.barplot(x=lat_importances, y=feature_names, ax=ax, color="blue")
    sns.barplot(x=lon_importances, y=feature_names, ax=ax, color="red")
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Feature")
    ax.set_title("Feature Importance for Longitude Prediction")
    st.pyplot(fig)


    # Generate Map with Past Movements
    past_locations = data[['latitude', 'longitude']].dropna()
    m = folium.Map(location=[past_locations.iloc[-1]['latitude'], past_locations.iloc[-1]['longitude']], zoom_start=8)

    folium.PolyLine(
        past_locations.values,
        color="blue",
        weight=3,
        opacity=0.7,
        tooltip="Past Movements"
    ).add_to(m)

    folium.Marker(
        location=[past_locations.iloc[0]['latitude'], past_locations.iloc[0]['longitude']],
        popup="Starting Point",
        icon=folium.Icon(color="green", icon="flag")
    ).add_to(m)

    folium.Marker(
        location=[past_locations.iloc[-1]['latitude'], past_locations.iloc[-1]['longitude']],
        popup="Last Known Location",
        icon=folium.Icon(color="orange", icon="info-sign")
    ).add_to(m)

    folium.Marker(
        [future_lat, future_lon],
        popup=f"Predicted Location: ({future_lat:.6f}, {future_lon:.6f})",
        icon=folium.Icon(color="red", icon="map-marker")
    ).add_to(m)

    st.write("### 🌍 Wildlife Movement and Predicted Location")
    st_folium(m, returned_objects=[])
