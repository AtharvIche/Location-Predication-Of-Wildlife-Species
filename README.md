# Major_Project
Final Year Project :- Location Predication of Wildlife Species Using ML
<br>
📌 Final Year Project - Location Prediction of Wildlife Species Using ML
🚀 Developed by: A10

📖 Overview
This project analyzes wildlife movement patterns and predicts future locations of species (e.g., vultures) based on historical tracking data.
The system takes a CSV file with GPS coordinates, timestamps, weather conditions, and food availability, and uses Machine Learning (ML) to predict the species' next probable location.

🛠️ Technologies Used & Purpose
Technology	Purpose
Python	Core programming language for implementation.
Streamlit	Python framework for interactive web applications.
Pandas	Handles data preprocessing and CSV file reading.
Matplotlib & Seaborn	Used for data visualization (histograms, feature distributions).
Folium	Visualizes wildlife movement on interactive maps.
Scikit-learn	Performs machine learning tasks (data processing, model training, evaluation).
RandomForestRegressor	ML model used for predicting future wildlife locations.

🔥 Features
✅ Upload CSV File: Users can upload tracking data for analysis.
✅ Data Cleaning & Processing: Handles missing values, extracts features, and prepares data.
✅ Feature Engineering: Extracts useful insights such as day, month, year, weather, and food availability.
✅ Machine Learning Model: Trained using RandomForestRegressor to predict latitude & longitude.
✅ Model Performance Metrics: Displays MAE (Mean Absolute Error) & R² Score to evaluate accuracy.
✅ Feature Importance Analysis: Identifies the most important factors influencing predictions.
✅ Interactive Map Visualization: Displays past movements & predicted next location.

📌 How It Works
1️⃣ Uploading the CSV File
Users upload a CSV file containing wildlife tracking data (GPS locations, date, weather, food availability).
The system reads the file using pandas and cleans the data.
2️⃣ Data Preprocessing
Removes missing values (mean for numeric, mode for categorical).
Extracts latitude and longitude from "GPS Location" or separate columns.
Converts date to datetime format and extracts day, month, year.
3️⃣ Feature Engineering
Identifies useful features such as:
📍 Day, Month, Year
🌦 Weather Condition
🍽 Food Availability
Converts categorical variables (like Weather) using Label Encoding.
4️⃣ Data Visualization
Uses Matplotlib & Seaborn to generate:
Histograms for feature distributions.
Bar charts for feature importance analysis.
5️⃣ Training the ML Model
Model Used: RandomForestRegressor
Why? ✅ Handles missing data well, ✅ Works with numerical data, ✅ High accuracy.
Steps:
Splits Data: 80% training, 20% testing.
Trains Two Models: One for latitude, another for longitude.
python
Copy
Edit
lat_model = RandomForestRegressor(n_estimators=200, random_state=42)
lon_model = RandomForestRegressor(n_estimators=200, random_state=42)

lat_model.fit(X_train, y_lat_train)
lon_model.fit(X_train, y_lon_train)
6️⃣ Evaluating Model Performance
Mean Absolute Error (MAE): Measures average prediction error.
R² Score: Measures how well the model explains variance.
python
Copy
Edit
lat_mae = mean_absolute_error(y_lat_test, y_lat_pred)
lon_mae = mean_absolute_error(y_lon_test, y_lon_pred)
lat_r2 = r2_score(y_lat_test, y_lat_pred)
lon_r2 = r2_score(y_lon_test, y_lon_pred)
📌 Higher R² and lower MAE indicate better predictions!

7️⃣ Predicting the Next Location
Uses latest available data to predict the next probable latitude & longitude.
python
Copy
Edit
latest_data = X.iloc[-1:].copy()
future_lat = lat_model.predict(latest_data)[0]
future_lon = lon_model.predict(latest_data)[0]

8️⃣ Feature Importance Analysis
Identifies which features contribute most to latitude & longitude predictions.
Uses Seaborn bar plots to display importance scores.
python
Copy
Edit
importances = lat_model.feature_importances_
sns.barplot(x=importances, y=X.columns, palette="viridis")

9️⃣ Wildlife Movement Visualization (Map)
Folium is used to plot:
🟢 Starting Location (Green marker)
🔶 Last Known Location (Orange marker)
🔴 Predicted Next Location (Red marker)
🔵 Past Movement Path (Blue line)
python
Copy
Edit
folium.Marker([future_lat, future_lon], 
    popup=f"Predicted Location: ({future_lat:.6f}, {future_lon:.6f})",
    icon=folium.Icon(color="red", icon="map-marker")
).add_to(m)
🔹 Final Output:
✅ Upload a CSV file with wildlife movement data.
✅ System cleans data, extracts features, and trains ML models.
✅ Predicts the next movement location of the species.
✅ Displays an interactive map showing past movements & the predicted next location.

📊 Results & Key Insights
Metric	Latitude Model	Longitude Model
MAE	Low (Good)	Low (Good)
R² Score	High (Good)	High (Good)
Most Important Features	Weather, Month, Day, Food Offered	Weather, Food Offered, Year
🔹 Higher R² score means the model makes accurate predictions.
🔹 Food availability & weather conditions strongly influence movements.

📌 Summary of Key Concepts
Concept	Description
Data Cleaning	Handling missing values, fixing formats.
Feature Engineering	Extracting date-based and environmental factors.
Machine Learning Model	RandomForestRegressor for predictions.
Model Evaluation	Using MAE & R² Score to measure accuracy.
Prediction	Forecasting the next location of wildlife.
Data Visualization	Using Seaborn (histograms) & Folium (map) for insights.
📢 Conclusion
This project successfully predicts wildlife species movement based on past data.
🔹 Combining ML modeling & visualization, it provides an effective tool for wildlife conservation & tracking.
🔹 The system is scalable and can be improved by adding more features such as temperature, wind speed, and terrain data.


💡 Installation & Running the Project
🔹 Install Dependencies

pip install streamlit pandas scikit-learn matplotlib seaborn folium
🔹 Run the Streamlit App
🔹streamlit run app.py

🚀 Developed by: A10
🔗 GitHub Repository: [Soon]

