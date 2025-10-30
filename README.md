# MMM Dashboard Basic

A minimalist Flask dashboard for visualizing Marketing Mix Model (MMM) results from Meta's Robyn framework.

## Overview

This dashboard provides a comprehensive view of marketing effectiveness, including:
- Media ROAS and ROI analysis
- Contribution decomposition
- Adstock decay patterns
- Competition tracking
- Trend analysis

## Installation

```bash
# Clone the repository
git clone https://github.com/credas/MMM-dashboard-basic.git
cd MMM-dashboard-basic

# Install dependencies
pip install flask pandas numpy scipy

# Run the application
python app.py
```

## Project Structure

```
MMM-dashboard-basic/
├── app.py                    # Flask application
├── templates/
│   └── dashboard.html        # Dashboard template
├── models/
│   └── FinBee/
│       └── loans1-model/
│           ├── model_config.json         # Model configuration
│           ├── pareto_aggregated.csv     # Aggregated model results
│           ├── pareto_alldecomp_matrix.csv  # Time series decomposition
│           ├── pareto_hyperparameters.csv   # Model hyperparameters
│           └── RobynModel-3_329_5.json     # Robyn model export
```

## Configuration

The `model_config.json` file defines:
- Model metadata (title, client, KPI)
- Media channel categorization (awareness/performance, ATL/digital)
- Competitor mapping
- Context variables
- Variable groupings

## Key Calculations

### 1. ROAS (Return on Ad Spend)
```
ROAS = Total Return / Total Investment
```
- Calculated from `roi_total` column in pareto_aggregated.csv
- Shows efficiency of each media channel

### 2. Media Response Decay (Adstock)
Using Weibull CDF transformation:
```python
# Week 1: Immediate effect only
week1_pct = immediate_pct * 100

# Subsequent weeks: Carryover decay
cdf_values = weibull_min.cdf(weeks/max_weeks, c=shape, scale=scale)
week2_pct = carryover_pct * (cdf_week2 - cdf_week1) * 100
```

**Key concepts:**
- **Immediate effect**: Impact realized in the first measurement period
- **Carryover effect**: Delayed impact that decays over time
- **Weibull parameters**: Shape controls decay curve, Scale controls time horizon

### 3. ROI Curves at Different Spending Levels
Using Hill saturation transformation:
```python
response_ratio = (spend^s) / ((spend^s) + (gamma^s))
```
- Shows diminishing returns as spending increases
- Helps identify optimal spending levels

### 4. Contribution Analysis
Groups variables into business-meaningful categories:
- **Our Media**: Split by awareness/performance
- **Competition**: Individual competitors and groups
- **Base Sales**: Intercept + trend (baseline) and seasonality
- **Context**: External factors (temperature, interest rates)

### 5. Yearly Trend Analysis
```
Trend % = (Trend Contribution / Total KPI) * 100
```
- Isolates long-term growth/decline from other effects
- Shows underlying business momentum

### 6. Competition Pressure
Monthly aggregation of:
- Our media investments
- Competitor GRP pressure by brand
- Actual KPI performance

### 7. Confidence Intervals
- Uses bootstrapped CI from Robyn model
- Shows uncertainty range for each channel's ROI

### 8. Seasonality Patterns
- Weekly seasonality index (excluding trend)
- Combines seasonal and holiday effects

## Data Sources

### pareto_aggregated.csv
- Model-level summary statistics
- Channel coefficients and contributions
- ROI and spend share metrics
- Confidence intervals

### pareto_alldecomp_matrix.csv
- Weekly time series decomposition
- Actual vs predicted values
- Individual channel contributions over time

### pareto_hyperparameters.csv
- Adstock parameters (scale, shape)
- Saturation parameters (alpha, gamma)
- Model performance metrics

## Key Metrics Explained

- **R²**: Model fit quality (0-1, higher is better)
- **NRMSE**: Normalized Root Mean Square Error (lower is better)
- **Effect Share**: % of total sales driven by each channel
- **Spend Share**: % of total budget allocated to each channel
- **Efficiency**: Effect Share / Spend Share (>1 means over-performing)

## Customization

To adapt for your own MMM model:

1. Update `model_config.json` with your variables
2. Place Robyn output files in the model directory
3. Adjust `MODEL_ID` in app.py to match your selected model
4. Modify `MEDIA_CHANNELS` list if needed

## Design Philosophy

This dashboard follows minimalist principles:
- No fancy styling or animations
- Black text on white background
- Simple tables and layouts
- Focus on data clarity over aesthetics
- Wireframe-like appearance

## License

MIT License

## Contributing

Pull requests are welcome. For major changes, please open an issue first.