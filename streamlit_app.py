import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="🏭 Energy Demand Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-container {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-size: 1.5rem;
        margin: 2rem 0;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_model_and_scaler():
    """Load the pre-trained LightGBM model and scaler"""
    try:
        model = joblib.load('models/lightgbm_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        return None, None
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None

def create_time_features(date_time):
    """Create time-based features from datetime"""
    features = {}
    
    # Basic time features
    features['Year'] = date_time.year
    features['Month'] = date_time.month
    features['Day'] = date_time.day
    features['Hour'] = date_time.hour
    features['DayOfWeek'] = date_time.weekday()
    features['DayOfYear'] = date_time.timetuple().tm_yday
    features['Quarter'] = (date_time.month - 1) // 3 + 1
    features['WeekOfYear'] = date_time.isocalendar()[1]
    
    # Season encoding (0=Winter, 1=Spring, 2=Summer, 3=Fall)
    month = date_time.month
    if month in [12, 1, 2]:
        features['Season'] = 0
    elif month in [3, 4, 5]:
        features['Season'] = 1
    elif month in [6, 7, 8]:
        features['Season'] = 2
    else:
        features['Season'] = 3
    
    # Weekend flag
    features['IsWeekend'] = 1 if date_time.weekday() >= 5 else 0
    
    # Cyclical encodings
    features['Hour_sin'] = np.sin(2 * np.pi * date_time.hour / 24)
    features['Hour_cos'] = np.cos(2 * np.pi * date_time.hour / 24)
    features['DayOfWeek_sin'] = np.sin(2 * np.pi * date_time.weekday() / 7)
    features['DayOfWeek_cos'] = np.cos(2 * np.pi * date_time.weekday() / 7)
    
    return features

def calculate_weather_features(temp, humidity, solar, wind_speed, hour=12):
    """Calculate weather-derived features"""
    features = {}
    
    # Heating/Cooling Degree Days (base 18.3°C)
    base_temp = 18.3
    features['HDD'] = max(0, base_temp - temp)
    features['CDD'] = max(0, temp - base_temp)
    
    # Temperature interaction with hour
    features['Temp_x_Hour'] = temp * hour
    
    # Temperature categories
    features['Is_High_Temp'] = 1 if temp > 25 else 0
    features['Is_Low_Temp'] = 1 if temp < 10 else 0
    
    # Comfort index (simplified)
    features['Comfort_Index'] = temp - (0.55 - 0.0055 * humidity) * (temp - 14.5)
    
    # Solar activity
    features['Is_Solar_Active'] = 1 if solar > 100 else 0
    
    return features

def create_prediction_input(datetime_input, temp, humidity, solar, wind_speed, 
                          electric_load_a, electric_load_b, cooling_load_a, 
                          heating_load_a, pv_generation, grid_import):
    """Create input features for prediction"""
    
    # Time features
    time_features = create_time_features(datetime_input)
    
    # Weather features
    weather_features = calculate_weather_features(temp, humidity, solar, wind_speed)
    
    # Create the feature dictionary
    features = {
        # Basic load features
        'PV_Generation_kW': pv_generation,
        'Grid_Import_kW': grid_import,
        'Electric_Load_A_kW': electric_load_a,
        'Electric_Load_B_kW': electric_load_b,
        'Cooling_Load_A_kW': cooling_load_a,
        'Heating_Load_A_kW': heating_load_a,
        
        # Weather features
        'Solar_Irradiation_W': solar,
        'Outdoor_Air_Temp_C': temp,
        'Outdoor_Air_Humidity_percent': humidity,
        'Wind_Speed_ms': wind_speed,
        
        # Temperature-hour interaction
        'Temp_x_Hour': temp * datetime_input.hour,
    }
    
    # Add lag features (with some variation based on time of day)
    total_load = electric_load_a + electric_load_b
    time_factor = 0.8 + 0.4 * np.sin(2 * np.pi * datetime_input.hour / 24)
    
    # Add lag and rolling features
    features.update({
        'Lag_6h_Load': total_load * time_factor,
        'Lag_12h_Load': total_load * (0.9 + 0.2 * np.cos(2 * np.pi * datetime_input.hour / 24)),
        'Lag_24h_Load': total_load * (0.85 + 0.3 * np.sin(2 * np.pi * datetime_input.weekday() / 7)),
        'Lag_48h_Load': total_load * 0.9,
        'Rolling_Mean_12h': total_load * (0.95 + 0.1 * np.sin(datetime_input.hour * 0.5 + temp * 0.1)),
        'Rolling_Mean_24h': total_load * (0.92 + 0.16 * np.sin(2 * np.pi * datetime_input.hour / 24)),
        'Rolling_Std_24h': max(10.0, total_load * 0.08),
        'EWMA_24h_Load': total_load * (0.88 + 0.24 * np.cos(2 * np.pi * datetime_input.hour / 24)),
    })
    
    # Add time and weather features
    features.update(time_features)
    features.update(weather_features)
    
    return features

def main():
    # Header
    st.markdown('<h1 class="main-header">⚡ Energy Demand Forecasting System</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
        Predict energy demand using advanced machine learning with interactive parameter adjustment
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model and scaler
    model, scaler = load_model_and_scaler()
    
    if model is None or scaler is None:
        st.error("Could not load the model. Please ensure model files are in the 'models' folder.")
        return
    
    # Sidebar for inputs
    st.sidebar.header("🎛️ Input Parameters")
    
    # Time input
    st.sidebar.subheader("⏰ Time Settings")
    hour_input = st.sidebar.slider("Hour of Day", 0, 23, datetime.now().hour, 1)
    
    # Create datetime with current date but selected hour
    current_date = datetime.now().date()
    datetime_input = datetime.combine(current_date, datetime.min.time().replace(hour=hour_input))
    
    # Weather conditions
    st.sidebar.subheader("🌤️ Weather Conditions")
    temp = st.sidebar.slider("Outdoor Temperature (°C)", -20.0, 45.0, 
                            st.session_state.get('temp', 20.0), 0.5, key='temp_slider')
    humidity = st.sidebar.slider("Humidity (%)", 0.0, 100.0, 
                               st.session_state.get('humidity', 50.0), 1.0, key='humidity_slider')
    solar = st.sidebar.slider("Solar Irradiation (W/m²)", 0.0, 1000.0, 
                            st.session_state.get('solar', 200.0), 10.0, key='solar_slider')
    wind_speed = st.sidebar.slider("Wind Speed (m/s)", 0.0, 30.0, 
                                 st.session_state.get('wind_speed', 5.0), 0.1, key='wind_slider')
    
    # Load adjustments
    st.sidebar.subheader("⚡ Load Adjustments")
    electric_load_a = st.sidebar.slider("Electric Load A (kW)", 0.0, 1000.0, 
                                      st.session_state.get('electric_load_a', 400.0), 5.0, key='load_a_slider')
    electric_load_b = st.sidebar.slider("Electric Load B (kW)", 0.0, 1000.0, 
                                      st.session_state.get('electric_load_b', 400.0), 5.0, key='load_b_slider')
    cooling_load_a = st.sidebar.slider("Cooling Load A (kW)", 0.0, 500.0, 
                                     st.session_state.get('cooling_load_a', 50.0), 5.0, key='cooling_slider')
    heating_load_a = st.sidebar.slider("Heating Load A (kW)", 0.0, 500.0, 
                                     st.session_state.get('heating_load_a', 50.0), 5.0, key='heating_slider')
    
    # Generation sources
    st.sidebar.subheader("🏭 Generation Sources")
    pv_generation = st.sidebar.slider("PV Generation (kW)", 0.0, 500.0, 
                                    st.session_state.get('pv_generation', 100.0), 5.0, key='pv_slider')
    grid_import = st.sidebar.slider("Grid Import (kW)", 0.0, 1000.0, 
                                  st.session_state.get('grid_import', 300.0), 5.0, key='grid_slider')
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create prediction
        if st.button("🔮 Predict Energy Demand", type="primary"):
            # Create input features
            features = create_prediction_input(
                datetime_input, temp, humidity, solar, wind_speed,
                electric_load_a, electric_load_b, cooling_load_a, 
                heating_load_a, pv_generation, grid_import
            )
            
            # Convert to DataFrame
            feature_df = pd.DataFrame([features])
            
            # Convert column names to strings to avoid feature name issues
            feature_df.columns = feature_df.columns.astype(str)
            
            # Handle missing features by filling with defaults
            expected_features = None
            if hasattr(model, 'feature_names_in_'):
                expected_features = [str(f) for f in model.feature_names_in_]
                for feature in expected_features:
                    if feature not in feature_df.columns:
                        feature_df[feature] = 0.0
                feature_df = feature_df[expected_features]
            
            try:
                # Scale features if scaler is available
                if scaler is not None:
                    scaled_features = scaler.transform(feature_df)
                    prediction = model.predict(scaled_features)[0]
                else:
                    prediction = model.predict(feature_df)[0]
                
                # Display prediction
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>🎯 Predicted Total Energy Demand</h2>
                    <h1>{prediction:.2f} kW</h1>
                    <p>Based on current parameter settings</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Additional metrics
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.metric("Current Load A+B", f"{electric_load_a + electric_load_b:.1f} kW")
                
                with col_b:
                    load_diff = prediction - (electric_load_a + electric_load_b)
                    st.metric("Demand vs Current", f"{load_diff:.1f} kW", 
                             delta=f"{load_diff:.1f}")
                
                with col_c:
                    efficiency = (prediction / (pv_generation + grid_import)) * 100 if (pv_generation + grid_import) > 0 else 0
                    st.metric("Grid Efficiency", f"{efficiency:.1f}%")
                
                # Feature importance (if available)
                if hasattr(model, 'feature_importances_'):
                    st.subheader("📊 Top Feature Influences")
                    
                    feature_importance = pd.DataFrame({
                        'feature': feature_df.columns,
                        'importance': model.feature_importances_[:len(feature_df.columns)]
                    }).sort_values('importance', ascending=False).head(10)
                    
                    fig = px.bar(feature_importance, 
                               x='importance', 
                               y='feature', 
                               orientation='h',
                               title="Top 10 Most Important Features")
                    fig.update_layout(height=400)
                    st.plotly_chart(fig)
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.info("The model might expect different features. Please check the model training configuration.")
    
    with col2:
        # Display current settings
        st.subheader("📋 Current Settings")
        
        st.info(f"""
        **Time:**  
        Hour: {hour_input}:00
        
        **Weather:**  
        🌡️ Temperature: {temp}°C  
        💧 Humidity: {humidity}%  
        ☀️ Solar: {solar} W/m²  
        💨 Wind: {wind_speed} m/s
        
        **Loads:**  
        ⚡ Electric A: {electric_load_a} kW  
        ⚡ Electric B: {electric_load_b} kW  
        ❄️ Cooling A: {cooling_load_a} kW  
        🔥 Heating A: {heating_load_a} kW
        
        **Generation:**  
        🏭 PV: {pv_generation} kW  
        🔌 Grid: {grid_import} kW
        """)
        
        # Quick scenarios
        st.subheader("🚀 Quick Scenarios")
        
        col_s1, col_s2 = st.columns(2)
        
        with col_s1:
            if st.button("🌅 Morning Peak"):
                st.info("Morning peak: High demand, low solar")
        
            if st.button("🌞 Midday Solar"):
                st.info("Midday: High solar generation, moderate demand")
        
        with col_s2:
            if st.button("🌙 Evening Peak"):
                st.info("Evening peak: High demand, no solar")
                
            if st.button("❄️ Winter High"):
                st.info("Winter: High heating demand, low temperature")
    
    # Additional information
    with st.expander("ℹ️ About This Model"):
        st.markdown("""
        ### Energy Demand Forecasting Model
        
        This application uses a **LightGBM** machine learning model trained on historical energy consumption data to predict future energy demand.
        
        **Key Features:**
        - 🔍 **Time-based patterns:** Captures daily, weekly, and seasonal variations
        - 🌡️ **Weather integration:** Considers temperature, humidity, solar, and wind conditions  
        - ⚡ **Load factors:** Incorporates current electrical and thermal loads
        - 🏭 **Generation sources:** Accounts for PV and grid import variations
        - 📈 **Advanced features:** Uses lag features and rolling statistics for better accuracy
        
        **How to Use:**
        1. Adjust the parameters in the sidebar
        2. Click "Predict Energy Demand" to get the forecast
        3. Explore different scenarios using the quick buttons
        
        The model provides real-time predictions to help optimize energy management and grid planning decisions.
        """)
    
if __name__ == "__main__":
    main()