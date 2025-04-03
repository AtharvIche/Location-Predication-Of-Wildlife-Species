
Wildlife Species Location Prediction System

Overview
The Wildlife Species Location Prediction System is a machine learning-based application designed to predict the future locations of wildlife species based on historical GPS tracking data and environmental factors. By leveraging the Random Forest Regressor, this system provides accurate predictions, helping conservationists and researchers analyze wildlife movement patterns and make data-driven decisions.  

Features 
✅ CSV Data Upload – Upload wildlife movement datasets for analysis.  
✅ Data Preprocessing & Feature Engineering – Handles missing values, encodes categorical features, and extracts temporal/environmental factors.  
✅ Machine Learning Model – Predicts future latitude and longitude using a trained Random Forest Regressor.  
✅ Feature Importance Analysis – Identifies key factors influencing wildlife movement.  
✅ Interactive Visualizations – Displays historical movement patterns, feature distributions, and predicted locations using Folium maps and Matplotlib.  
✅ Model Performance Metrics – Evaluates prediction accuracy with Mean Absolute Error (MAE) and R² score.  

Tech Stack 
- Frontend:Streamlit  
- Backend: Python (Pandas, NumPy, Scikit-Learn)  
- Visualization: Matplotlib, Seaborn, Folium  
- Machine Learning Model: Random Forest Regressor  


Installation & Usage

Clone the Repository  
terminal
https://github.com/AtharvIche/Location-Predication-Of-Wildlife-Species.git
cd Location-Predication-Of-Wildlife-Species


Install Dependencies 
terminal
pip install -r requirements.txt


Run the Application  
terminal
streamlit run app.py


Data Format 
The dataset should be in CSV format with columns such as:  

Timestamp, Latitude, Longitude, Weather, Temperature, Food_Offered, Health_Condition
Ensure the dataset contains relevant temporal and ecological factors for accurate predictions.  



License
This project is open-source under the MIT License.  

---
