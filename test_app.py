"""
Test script to verify the Streamlit app dependencies and model loading
Run this before starting the Streamlit app to ensure everything works
"""

import sys
import os

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing package imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✅ Pandas imported successfully")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        print("✅ Plotly imported successfully")
    except ImportError as e:
        print(f"❌ Plotly import failed: {e}")
        return False
    
    try:
        import lightgbm as lgb
        print("✅ LightGBM imported successfully")
    except ImportError as e:
        print(f"❌ LightGBM import failed: {e}")
        return False
    
    try:
        import joblib
        print("✅ Joblib imported successfully")
    except ImportError as e:
        print(f"❌ Joblib import failed: {e}")
        return False
    
    return True

def test_model_files():
    """Test if model files exist and can be loaded"""
    print("\nTesting model files...")
    
    model_path = "models/lightgbm_model.pkl"
    scaler_path = "models/scaler.pkl"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("   Please ensure the LightGBM model is saved in the models/ folder")
        return False
    else:
        print(f"✅ Model file found: {model_path}")
    
    if not os.path.exists(scaler_path):
        print(f"❌ Scaler file not found: {scaler_path}")
        print("   Please ensure the scaler is saved in the models/ folder")
        return False
    else:
        print(f"✅ Scaler file found: {scaler_path}")
    
    # Try to load the files
    try:
        import joblib
        model = joblib.load(model_path)
        print("✅ Model loaded successfully")
        
        scaler = joblib.load(scaler_path)
        print("✅ Scaler loaded successfully")
        
        return True
    except Exception as e:
        print(f"❌ Error loading model files: {e}")
        return False

def test_sample_prediction():
    """Test a sample prediction to ensure everything works"""
    print("\nTesting sample prediction...")
    
    try:
        import joblib
        import pandas as pd
        import numpy as np
        from datetime import datetime
        
        # Load model and scaler
        model = joblib.load("models/lightgbm_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        
        # Create sample data
        sample_data = {
            'PV_Generation_kW': 100.0,
            'Grid_Import_kW': 300.0,
            'Electric_Load_A_kW': 400.0,
            'Electric_Load_B_kW': 400.0,
            'Cooling_Load_A_kW': 50.0,
            'Heating_Load_A_kW': 50.0,
            'Solar_Irradiation_W': 200.0,
            'Outdoor_Air_Temp_C': 20.0,
            'Outdoor_Air_Humidity_percent': 50.0,
            'Wind_Speed_ms': 5.0,
            'Year': 2023,
            'Month': 6,
            'Day': 15,
            'Hour': 12,
            'DayOfWeek': 3,
            'DayOfYear': 166,
            'Quarter': 2,
            'WeekOfYear': 24,
            'Season': 2,
            'IsWeekend': 0,
            'Hour_sin': np.sin(2 * np.pi * 12 / 24),
            'Hour_cos': np.cos(2 * np.pi * 12 / 24),
            'DayOfWeek_sin': np.sin(2 * np.pi * 3 / 7),
            'DayOfWeek_cos': np.cos(2 * np.pi * 3 / 7),
            'HDD': 0.0,
            'CDD': 1.7,
            'Temp_x_Hour': 240.0,
            'Is_High_Temp': 0,
            'Is_Low_Temp': 0,
            'Comfort_Index': 18.5,
            'Is_Solar_Active': 1,
            'Lag_6h_Load': 800.0,
            'Lag_12h_Load': 800.0,
            'Lag_24h_Load': 800.0,
            'Lag_48h_Load': 800.0,
            'Rolling_Mean_12h': 800.0,
            'Rolling_Mean_24h': 800.0,
            'Rolling_Std_24h': 50.0,
            'EWMA_24h_Load': 800.0,
        }
        
        # Convert to DataFrame
        df = pd.DataFrame([sample_data])
        
        # Convert column names to strings to avoid feature name issues
        df.columns = df.columns.astype(str)
        
        # Handle missing features
        if hasattr(model, 'feature_names_in_'):
            expected_features = [str(f) for f in model.feature_names_in_]
            for feature in expected_features:
                if feature not in df.columns:
                    df[feature] = 0.0
            df = df[expected_features]
        
        # Make prediction
        if scaler is not None:
            scaled_features = scaler.transform(df)
            prediction = model.predict(scaled_features)[0]
        else:
            prediction = model.predict(df)[0]
        
        print(f"✅ Sample prediction successful: {prediction:.2f} kW")
        return True
        
    except Exception as e:
        print(f"❌ Sample prediction failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Streamlit Energy Forecasting App")
    print("=" * 50)
    
    success = True
    
    # Test imports
    if not test_imports():
        success = False
    
    # Test model files
    if not test_model_files():
        success = False
    
    # Test sample prediction
    if not test_sample_prediction():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! You can run the Streamlit app with:")
        print("   streamlit run streamlit_app.py")
    else:
        print("❌ Some tests failed. Please fix the issues above before running the app.")
        print("\n📝 Common solutions:")
        print("   - Install missing packages: pip install -r requirements.txt")
        print("   - Ensure model files are in the models/ folder")
        print("   - Check that the model was trained and saved properly")
    
    return success

if __name__ == "__main__":
    main()