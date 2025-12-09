# Energy Consumption Time Series Forecasting

An interactive web application for predicting energy demand using machine learning. Built with Streamlit and LightGBM.

## 🚀 Features

- **Interactive Prediction**: Real-time energy demand forecasting with adjustable parameters
- **Weather Integration**: Incorporates temperature, humidity, solar radiation, and wind speed
- **Load Adjustment**: Sliders for different types of energy loads (electric, cooling, heating)
- **Visualization**: Feature importance charts and prediction metrics
- **Scenario Testing**: Quick preset scenarios for different conditions

## 📋 Prerequisites

- Python 3.8 or higher
- Required Python packages (see requirements.txt)

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Ameytillu/energy_consumption_Time_series_forecasting.git
   cd energy_consumption_Time_series_forecasting
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure model files are available:**
   Make sure the following files exist in the `models/` folder:
   - `lightgbm_model.pkl` - The trained LightGBM model
   - `scaler.pkl` - The feature scaler

## 🚀 Running the Application

1. **Start the Streamlit app:**
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Open your browser:**
   The app will automatically open in your default browser at `http://localhost:8501`

## 📊 How to Use

### 1. Time Settings
- Select the date and time for prediction
- The model considers temporal patterns (hourly, daily, seasonal)

### 2. Weather Conditions
- **Temperature**: Outdoor temperature in Celsius (-20°C to 45°C)
- **Humidity**: Relative humidity percentage (0% to 100%)
- **Solar Irradiation**: Solar energy in W/m² (0 to 1000)
- **Wind Speed**: Wind speed in m/s (0 to 30)

### 3. Load Adjustments
- **Electric Load A & B**: Primary electrical loads in kW
- **Cooling Load A**: Air conditioning and cooling systems in kW
- **Heating Load A**: Heating systems in kW

### 4. Generation Sources
- **PV Generation**: Solar photovoltaic generation in kW
- **Grid Import**: Power imported from the electrical grid in kW

### 5. Make Predictions
- Adjust parameters using the sliders in the sidebar
- Click **"Predict Energy Demand"** to get the forecast
- View the predicted total energy demand and additional metrics

## 🎯 Understanding the Results

The app provides several key outputs:

- **Predicted Total Energy Demand**: The main prediction in kW
- **Current Load A+B**: Sum of current electrical loads
- **Demand vs Current**: Difference between predicted and current load
- **Grid Efficiency**: Efficiency percentage based on generation sources
- **Feature Importance**: Chart showing which factors most influence the prediction

## 🔧 Model Information

The application uses a **LightGBM** (Light Gradient Boosting Machine) model trained on historical energy consumption data. The model incorporates:

- **Time-based features**: Hour, day, week, month, season patterns
- **Weather features**: Temperature effects, heating/cooling degree days
- **Lag features**: Historical load patterns from previous hours/days
- **Rolling statistics**: Moving averages and standard deviations
- **Cyclical encoding**: Proper handling of time cycles

## 📈 Quick Scenarios

Use the quick scenario buttons to test different conditions:

- **Morning Peak**: Typical morning high-demand period
- **Midday Solar**: High solar generation during midday
- **Evening Peak**: Evening high-demand period
- **Winter High**: High heating demand in winter

## 🛠️ Project Structure

```
energy_consumption_Time_series_forecasting/
├── streamlit_app.py          # Main Streamlit application
├── requirements.txt          # Python dependencies
├── models/                   # Trained model files
│   ├── lightgbm_model.pkl   # LightGBM model
│   └── scaler.pkl           # Feature scaler
├── Data_Output/             # Processed datasets
├── notebooks/               # Jupyter notebooks for analysis
└── README.md               # This file
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋‍♂️ Support

If you have any questions or issues:

1. Check the existing issues in the GitHub repository
2. Create a new issue with a detailed description
3. Include error messages and steps to reproduce

## 🔮 Future Enhancements

- [ ] Real-time data integration
- [ ] Multiple model comparison
- [ ] Historical data visualization
- [ ] Export predictions to CSV
- [ ] Advanced scenario planning
- [ ] Model performance metrics dashboard

---

**Built with ❤️ using Streamlit, LightGBM, and modern data science practices.**