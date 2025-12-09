"""
Quick test to verify that predictions change with different inputs
"""
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

# Load model and scaler
model = joblib.load("models/lightgbm_model.pkl")
scaler = joblib.load("models/scaler.pkl")

def create_test_features(temp, electric_load_a, electric_load_b, hour=12):
    """Create test features for prediction"""
    current_date = datetime.now().replace(hour=hour)
    
    features = {
        'PV_Generation_kW': 100.0,
        'Grid_Import_kW': 300.0,
        'Electric_Load_A_kW': electric_load_a,
        'Electric_Load_B_kW': electric_load_b,
        'Cooling_Load_A_kW': 50.0,
        'Heating_Load_A_kW': 50.0,
        'Solar_Irradiation_W': 200.0,
        'Outdoor_Air_Temp_C': temp,
        'Outdoor_Air_Humidity_percent': 50.0,
        'Wind_Speed_ms': 5.0,
        'Temp_x_Hour': temp * hour,
        
        # Time features
        'Year': current_date.year,
        'Month': current_date.month,
        'Day': current_date.day,
        'Hour': current_date.hour,
        'DayOfWeek': current_date.weekday(),
        'DayOfYear': current_date.timetuple().tm_yday,
        'Quarter': (current_date.month - 1) // 3 + 1,
        'WeekOfYear': current_date.isocalendar()[1],
        'Season': 1,
        'IsWeekend': 0,
        'Hour_sin': np.sin(2 * np.pi * current_date.hour / 24),
        'Hour_cos': np.cos(2 * np.pi * current_date.hour / 24),
        'DayOfWeek_sin': np.sin(2 * np.pi * current_date.weekday() / 7),
        'DayOfWeek_cos': np.cos(2 * np.pi * current_date.weekday() / 7),
        
        # Weather features
        'HDD': max(0, 18.3 - temp),
        'CDD': max(0, temp - 18.3),
        'Is_High_Temp': 1 if temp > 25 else 0,
        'Is_Low_Temp': 1 if temp < 10 else 0,
        'Comfort_Index': temp - (0.55 - 0.0055 * 50) * (temp - 14.5),
        'Is_Solar_Active': 1,
        
        # Lag features
        'Lag_6h_Load': electric_load_a + electric_load_b,
        'Lag_12h_Load': electric_load_a + electric_load_b,
        'Lag_24h_Load': electric_load_a + electric_load_b,
        'Lag_48h_Load': electric_load_a + electric_load_b,
        'Rolling_Mean_12h': electric_load_a + electric_load_b,
        'Rolling_Mean_24h': electric_load_a + electric_load_b,
        'Rolling_Std_24h': 50.0,
        'EWMA_24h_Load': electric_load_a + electric_load_b,
    }
    
    return features

def test_predictions():
    """Test that predictions vary with different inputs"""
    print("Testing Energy Demand Predictions:")
    print("=" * 50)
    
    # Test different scenarios
    scenarios = [
        {"name": "Low Load, Cool", "temp": 15, "load_a": 300, "load_b": 300, "hour": 8},
        {"name": "High Load, Hot", "temp": 30, "load_a": 600, "load_b": 600, "hour": 14},
        {"name": "Medium Load, Mild", "temp": 20, "load_a": 400, "load_b": 400, "hour": 12},
        {"name": "Evening Peak", "temp": 18, "load_a": 550, "load_b": 530, "hour": 19},
        {"name": "Winter High", "temp": 2, "load_a": 600, "load_b": 580, "hour": 10},
    ]
    
    for scenario in scenarios:
        features = create_test_features(
            scenario["temp"], scenario["load_a"], 
            scenario["load_b"], scenario["hour"]
        )
        
        # Convert to DataFrame
        df = pd.DataFrame([features])
        df.columns = df.columns.astype(str)
        
        # Handle missing features
        if hasattr(model, 'feature_names_in_'):
            expected_features = [str(f) for f in model.feature_names_in_]
            for feature in expected_features:
                if feature not in df.columns:
                    df[feature] = 0.0
            df = df[expected_features]
        
        # Make prediction
        try:
            scaled_features = scaler.transform(df)
            prediction = model.predict(scaled_features)[0]
            
            print(f"{scenario['name']:15} | "
                  f"Temp: {scenario['temp']:2d}°C | "
                  f"Load A+B: {scenario['load_a'] + scenario['load_b']:4d} kW | "
                  f"Hour: {scenario['hour']:2d} | "
                  f"Predicted: {prediction:6.1f} kW")
        except Exception as e:
            print(f"{scenario['name']:15} | Error: {e}")
    
    print("=" * 50)
    print("✅ If predictions vary above, the model is working correctly!")

if __name__ == "__main__":
    test_predictions()