# Prophet Transaction Forecasting Dashboard Documentation

This documentation provides a comprehensive technical overview of the Prophet-based transaction forecasting system, detailing the purpose and mechanics of each module and function.

## 🏗️ System Overview
The application is a full-stack forecasting solution designed to predict transaction trends. It integrates a **Facebook Prophet** backend with an interactive **Plotly Dash** frontend, specifically tailored for regional requirements like Hijri calendar support and Indian financial formatting.

---

## 🛠️ Module & Function Details

### 1. Feature Engineering Module (`prophet_feature_engineering.py`)
This module is responsible for transforming raw data into a format suitable for the Prophet model.

*   **`create_prophet_features(df, date_col, campaign_col)`**: 
    *   Performs **Prophet-safe feature engineering** by cleaning and structuring the input dataframe.
    *   **Hijri Integration**: Utilises the `hijri_converter` library to generate features based on the Islamic calendar, which is critical for capturing lunar-driven transaction patterns.
    *   **Regressor Management**: Specifically prepares the `campaign_count` column to be used as an external regressor in the model.

### 2. Backend Engine (`backend_prophet.py`)
The backend manages the machine learning lifecycle, from training to executing predictions.

*   **`format_indian_number(x)`**: 
    *   A specialised utility that converts raw numerical values into the **Indian Number System (Crore/Lakh)**.
    *   This ensures that large transaction values are presented in a familiar format for regional financial reporting.
*   **`train_model()`**: 
    *   Executed automatically on startup to initialize the global `MODEL` and `REGRESSORS`.
    *   It loads the training data, applies feature engineering, and fits the Prophet model with specified regressors.
*   **`build_future_features(future_df)`**: 
    *   Ensures structural integrity between training and prediction phases.
    *   It applies the same feature engineering logic to the `future_df` that was used during training, ensuring all expected regressor columns are present.
*   **`predict_transaction(start_date, end_date)`**: 
    *   Acts as the primary API for the frontend.
    *   It generates a future date range, builds the necessary features, runs the Prophet forecast, and returns the results for the specified window.

### 3. Frontend Application (`frontend.py`)
The frontend provides a web-based interface for interacting with the forecasting model.

*   **`app.layout`**: 
    *   Defines the **Visual Architecture** of the dashboard using `dash.html` and `dash.dcc` components.
    *   Includes a customised **Global CSS** section to implement a professional "Reference Design" theme (Navy, Teal, Gold, and Dark tones).
*   **`update(n_clicks, start, end)`**: 
    *   A multi-output **Dash Callback** triggered by the "Predict" button.
    *   **Input Processing**: Takes the user-selected date range and calls the backend `predict_transaction` function.
    *   **Visual Generation**: Updates eight distinct UI elements simultaneously:
        1.  **`forecast-graph`**: A Plotly visualization of predicted trends.
        2.  **Summary Stats**: Updates `total-output`, `max-output`, `min-output`, and `avg-output` using the Indian number formatter.
        3.  **Data Table**: Generates the `forecast-table` and `table-count` for a detailed breakdown.
        4.  **`pie-chart`**: Renders a distributional view of the forecasted data.

---

## 🚀 Execution Workflow

1.  **Initialisation**: The system starts by running `train_model()`, which prepares the environment for instant predictions.
2.  **User Interaction**: The user selects a date range on the dashboard and clicks "Predict".
3.  **Data Flow**: The `update` callback sends the dates to the backend, which processes the forecast through the feature-engineered pipeline.
4.  **Rendering**: The dashboard UI is updated asynchronously with the new forecasts, formatted statistics, and interactive charts.