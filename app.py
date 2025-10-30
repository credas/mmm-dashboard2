from flask import Flask, render_template
import pandas as pd
import numpy as np
from scipy.stats import weibull_min
import os

app = Flask(__name__)

# Path to model results
MODEL_DIR = 'models/FinBee/loans1-model'
MODEL_PATH = f'{MODEL_DIR}/pareto_aggregated.csv'
DECOMP_PATH = f'{MODEL_DIR}/pareto_alldecomp_matrix.csv'
CONFIG_PATH = f'{MODEL_DIR}/model_config.json'

# Selected model ID
MODEL_ID = '3_329_5'

# Load model configuration
try:
    with open(CONFIG_PATH, 'r') as f:
        import json
        model_config = json.load(f)
        MEDIA_CHANNELS = list(model_config.get('media_spend', {}).keys())
except:
    # Fallback if config not found
    model_config = {}
    MEDIA_CHANNELS = ['TV', 'TV2', 'RADIO', 'RADIO2', 'META', 'GADS', 'YT']

@app.route('/')
def dashboard():
    # Read the aggregated results
    df = pd.read_csv(MODEL_PATH)
    
    # Filter for specific model and media channels
    media_df = df[(df['solID'] == MODEL_ID) & (df['rn'].isin(MEDIA_CHANNELS))]
    
    # Extract relevant columns for media
    media_data = []
    for _, row in media_df.iterrows():
        # ROAS = ROI Total (already calculated in the data)
        roas = float(row['roi_total']) if pd.notna(row['roi_total']) else 0.0
        
        # Only include media if it has spend > 0
        if pd.notna(row['total_spend']) and float(row['total_spend']) > 0:
            media_data.append({
                'channel': row['rn'],
                'investment': float(row['total_spend']),
                'roas': roas
            })
    
    # Sort by ROAS descending
    media_data.sort(key=lambda x: x['roas'], reverse=True)
    
    # Get all variables for the model
    all_vars_df = df[df['solID'] == MODEL_ID]
    
    # Extract all variables and their contributions
    all_variables = []
    for _, row in all_vars_df.iterrows():
        variable_name = row['rn']
        contribution = float(row['xDecompPerc']) * 100 if pd.notna(row['xDecompPerc']) else 0.0
        
        # Skip variables with 0 contribution
        if abs(contribution) > 0.001:
            all_variables.append({
                'variable': variable_name,
                'contribution': contribution
            })
    
    # Sort by absolute contribution descending
    all_variables.sort(key=lambda x: abs(x['contribution']), reverse=True)
    
    # Create grouped contribution analysis
    grouped_contributions = {
        'our_media': {
            'total': 0,
            'awareness': {'total': 0, 'channels': {}},
            'performance': {'total': 0, 'channels': {}}
        },
        'competition': {
            'total': 0,
            'competitors': {},
            'groups': {}
        },
        'base_sales': {
            'total': 0,
            'base': 0,
            'season': 0
        },
        'context': {
            'total': 0,
            'variables': {}
        }
    }
    
    # Process all variables and group them
    for var in all_variables:
        var_name = var['variable']
        contribution = var['contribution']
        
        # Our media channels
        if var_name in model_config.get('media_spend', {}):
            media_info = model_config['media_spend'][var_name]
            approach = media_info.get('approach', '').lower()
            
            grouped_contributions['our_media']['total'] += contribution
            
            if approach == 'awareness':
                grouped_contributions['our_media']['awareness']['total'] += contribution
                grouped_contributions['our_media']['awareness']['channels'][var_name] = {
                    'contribution': contribution,
                    'name': media_info.get('name', var_name)
                }
            elif approach == 'performance':
                grouped_contributions['our_media']['performance']['total'] += contribution
                grouped_contributions['our_media']['performance']['channels'][var_name] = {
                    'contribution': contribution,
                    'name': media_info.get('name', var_name)
                }
        
        # Competition
        elif var_name in model_config.get('competitors', {}):
            comp_info = model_config['competitors'][var_name]
            comp_name = comp_info.get('name', var_name)
            comp_group = comp_info.get('group', 'OTHER')
            
            grouped_contributions['competition']['total'] += contribution
            
            # Individual competitors
            if comp_name not in grouped_contributions['competition']['competitors']:
                grouped_contributions['competition']['competitors'][comp_name] = 0
            grouped_contributions['competition']['competitors'][comp_name] += contribution
            
            # Competition groups
            if comp_group not in grouped_contributions['competition']['groups']:
                grouped_contributions['competition']['groups'][comp_group] = 0
            grouped_contributions['competition']['groups'][comp_group] += contribution
        
        # Base sales components
        elif var_name in ['(Intercept)', 'trend', 'season', 'holiday']:
            grouped_contributions['base_sales']['total'] += contribution
            if var_name in ['(Intercept)', 'trend']:
                grouped_contributions['base_sales']['base'] += contribution
            else:
                grouped_contributions['base_sales']['season'] += contribution
        
        # Context variables
        elif var_name in model_config.get('context_variables', {}):
            context_info = model_config['context_variables'][var_name]
            grouped_contributions['context']['total'] += contribution
            grouped_contributions['context']['variables'][var_name] = {
                'contribution': contribution,
                'name': context_info.get('short_name', var_name)
            }
    
    # Get model metrics (R² and NRMSE) and date range
    model_metrics = {}
    if not all_vars_df.empty:
        first_row = all_vars_df.iloc[0]
        model_metrics['r2'] = float(first_row['rsq_train']) if pd.notna(first_row['rsq_train']) else 0.0
        model_metrics['nrmse'] = float(first_row['nrmse']) if pd.notna(first_row['nrmse']) else 0.0
    
    # Get date range from decomp data
    decomp_df = pd.read_csv(DECOMP_PATH)
    date_df = decomp_df[decomp_df['solID'] == MODEL_ID]['ds']
    if not date_df.empty:
        model_metrics['date_from'] = date_df.iloc[0]
        model_metrics['date_to'] = date_df.iloc[-1]
    
    # Calculate totals
    total_investment = sum(m['investment'] for m in media_data)
    
    # Get actual vs predicted values
    kpi_df = decomp_df[decomp_df['solID'] == MODEL_ID].copy()
    
    # Prepare KPI data
    kpi_data = []
    for _, row in kpi_df.iterrows():
        kpi_data.append({
            'date': row['ds'],
            'actual': float(row['dep_var']) if pd.notna(row['dep_var']) else 0.0,
            'predicted': float(row['depVarHat']) if pd.notna(row['depVarHat']) else 0.0
        })
    
    # Sort by date
    kpi_data.sort(key=lambda x: x['date'])
    
    # Limit to last 20 weeks for readability
    kpi_data = kpi_data[-20:] if len(kpi_data) > 20 else kpi_data
    
    # Calculate yearly investments and returns
    yearly_data = []
    if not kpi_df.empty:
        kpi_df['year'] = pd.to_datetime(kpi_df['ds']).dt.year
        yearly_stats = []
        
        for year in kpi_df['year'].unique():
            year_df = kpi_df[kpi_df['year'] == year]
            
            # Calculate total media investment for the year
            year_investment = 0
            for channel in MEDIA_CHANNELS:
                if channel in year_df.columns:
                    year_investment += year_df[channel].sum()
            
            # Calculate total return (contribution) for the year
            year_return = year_df['depVarHat'].sum()
            
            yearly_stats.append({
                'year': int(year),
                'investment': year_investment,
                'return': year_return,
                'roi': year_return / year_investment if year_investment > 0 else 0
            })
        
        # Sort by year and calculate YoY changes
        yearly_stats.sort(key=lambda x: x['year'])
        for i in range(len(yearly_stats)):
            if i > 0:
                prev = yearly_stats[i-1]
                curr = yearly_stats[i]
                inv_change = ((curr['investment'] - prev['investment']) / prev['investment'] * 100) if prev['investment'] > 0 else 0
                ret_change = ((curr['return'] - prev['return']) / prev['return'] * 100) if prev['return'] > 0 else 0
            else:
                inv_change = ret_change = 0
            
            yearly_data.append({
                'year': yearly_stats[i]['year'],
                'investment': yearly_stats[i]['investment'],
                'return': yearly_stats[i]['return'],
                'roi': yearly_stats[i]['roi'],
                'inv_change': inv_change,
                'ret_change': ret_change
            })
    
    # Calculate monthly competition pressure
    monthly_pressure_data = []
    if not kpi_df.empty:
        kpi_df['year_month'] = pd.to_datetime(kpi_df['ds']).dt.to_period('M')
        
        for period in kpi_df['year_month'].unique():
            month_df = kpi_df[kpi_df['year_month'] == period]
            
            # Calculate our media investment for the month
            our_investment = 0
            for channel in MEDIA_CHANNELS:
                if channel in month_df.columns:
                    our_investment += month_df[channel].sum()
            
            # Calculate competition pressure by competitor
            competitor_pressure = {}
            total_comp_pressure = 0
            
            for comp_var, comp_info in model_config.get('competitors', {}).items():
                if comp_var in month_df.columns:
                    comp_name = comp_info.get('name', comp_var)
                    pressure = month_df[comp_var].sum()
                    
                    if comp_name not in competitor_pressure:
                        competitor_pressure[comp_name] = 0
                    competitor_pressure[comp_name] += pressure
                    total_comp_pressure += abs(pressure)
            
            # Get actual KPI for the month
            monthly_kpi = month_df['dep_var'].sum()
            
            monthly_pressure_data.append({
                'month': str(period),
                'our_investment': our_investment,
                'competitor_pressure': competitor_pressure,
                'total_comp_pressure': total_comp_pressure,
                'kpi_actual': monthly_kpi
            })
    
    # Sort by month and limit to last 12 months
    monthly_pressure_data.sort(key=lambda x: x['month'])
    monthly_pressure_data = monthly_pressure_data[-12:] if len(monthly_pressure_data) > 12 else monthly_pressure_data
    
    # Calculate trend analysis by year
    trend_analysis = []
    if not kpi_df.empty and 'trend' in kpi_df.columns:
        # Group by year
        for year in sorted(kpi_df['year'].unique()):
            year_df = kpi_df[kpi_df['year'] == year]
            
            # Calculate total KPI (actual sales) for the year
            total_kpi = year_df['dep_var'].sum()
            
            # Calculate total trend contribution for the year
            total_trend = year_df['trend'].sum()
            
            # Calculate trend as percentage of total KPI
            trend_pct = (total_trend / total_kpi * 100) if total_kpi > 0 else 0
            
            trend_analysis.append({
                'year': int(year),
                'total_kpi': total_kpi,
                'total_trend': total_trend,
                'trend_pct': trend_pct
            })
    
    # Natural seasonality (weekly, excluding trend)
    seasonality_data = []
    if not kpi_df.empty:
        # Group by week number
        kpi_df['week_num'] = pd.to_datetime(kpi_df['ds']).dt.isocalendar().week
        
        for week in range(1, 53):
            week_data = kpi_df[kpi_df['week_num'] == week]
            if not week_data.empty:
                avg_season = week_data['season'].mean() if 'season' in week_data.columns else 0
                avg_holiday = week_data['holiday'].mean() if 'holiday' in week_data.columns else 0
                seasonality_data.append({
                    'week': week,
                    'seasonality': avg_season + avg_holiday
                })
    
    # Share of spend vs share of contribution
    spend_contrib_data = []
    for _, row in media_df.iterrows():
        if pd.notna(row['spend_share']) and pd.notna(row['effect_share']):
            spend_contrib_data.append({
                'channel': row['rn'],
                'spend_share': float(row['spend_share']) * 100,
                'effect_share': float(row['effect_share']) * 100,
                'efficiency': float(row['effect_share']) / float(row['spend_share']) if float(row['spend_share']) > 0 else 0
            })
    
    # CI intervals for media channels
    ci_data = []
    for _, row in media_df.iterrows():
        if pd.notna(row['ci_low']) and pd.notna(row['ci_up']):
            ci_data.append({
                'channel': row['rn'],
                'roi_mean': float(row['roi_total']) if pd.notna(row['roi_total']) else 0,
                'ci_low': float(row['ci_low']),
                'ci_up': float(row['ci_up'])
            })
    
    # Immediate vs carryover effects and store for adstock calculation
    carryover_data = []
    carryover_dict = {}  # Store for use in adstock calculation
    for _, row in media_df.iterrows():
        if pd.notna(row['carryover_pct']):
            carryover_pct = float(row['carryover_pct']) * 100
            immediate_pct = 100 - carryover_pct
            carryover_data.append({
                'channel': row['rn'],
                'immediate': immediate_pct,
                'carryover': carryover_pct
            })
            carryover_dict[row['rn']] = {
                'immediate': immediate_pct,
                'carryover': carryover_pct
            }
    
    # Create ROI curves table using actual hyperparameters
    roi_curves = []
    spending_levels = [0.5, 0.75, 1.0, 1.25, 1.5]  # 50%, 75%, 100%, 125%, 150%
    
    # Read hyperparameters for the model
    hyper_df = pd.read_csv('models/FinBee/loans1-model/pareto_hyperparameters.csv')
    hyper_row = hyper_df[hyper_df['solID'] == MODEL_ID].iloc[0] if not hyper_df[hyper_df['solID'] == MODEL_ID].empty else None
    
    for media in media_data:
        if media['roas'] > 0:
            curve_data = {
                'channel': media['channel'],
                'current_spend': media['investment'],
                'current_roi': media['roas'],
                'levels': []
            }
            
            # Get hyperparameters for this channel if available
            alpha = gamma = None
            if hyper_row is not None:
                alpha_col = f"{media['channel']}_alphas"
                gamma_col = f"{media['channel']}_gammas"
                if alpha_col in hyper_row and gamma_col in hyper_row:
                    alpha = float(hyper_row[alpha_col])
                    gamma = float(hyper_row[gamma_col])
            
            # For each spending level, calculate response using Hill transformation
            current_spend = media['investment']
            current_response = media['roas'] * current_spend  # Total response at current spend
            
            for level in spending_levels:
                spend = current_spend * level
                
                if alpha and gamma:
                    # Hill transformation: y = alpha * x^s / (x^s + gamma^s)
                    # where s = 1 for standard Hill
                    response_ratio = (spend ** 1) / ((spend ** 1) + (gamma ** 1))
                    base_response_ratio = (current_spend ** 1) / ((current_spend ** 1) + (gamma ** 1))
                    
                    # Scale the response
                    if base_response_ratio > 0:
                        response = current_response * (response_ratio / base_response_ratio)
                    else:
                        response = 0
                else:
                    # Fallback to simple diminishing returns if no hyperparameters
                    roi_factor = 1.0 - (level - 1.0) * 0.3 if level > 1.0 else 1.0 + (1.0 - level) * 0.15
                    response = current_response * level * roi_factor
                
                roi = response / spend if spend > 0 else 0
                
                curve_data['levels'].append({
                    'spend': spend,
                    'roi': max(0, roi),
                    'level_pct': int(level * 100)
                })
            
            roi_curves.append(curve_data)
    
    # Calculate adstock decay table
    adstock_data = []
    adstock_type = "Weibull CDF"  # From the model JSON
    
    for media in media_data:
        # Get Weibull parameters for this channel
        scale = shape = None
        if hyper_row is not None:
            scale_col = f"{media['channel']}_scales"
            shape_col = f"{media['channel']}_shapes"
            if scale_col in hyper_row and shape_col in hyper_row:
                scale = float(hyper_row[scale_col])
                shape = float(hyper_row[shape_col])
        
        if media['channel'] in carryover_dict:
            # Get the immediate effect percentage from the model
            immediate_pct = carryover_dict[media['channel']]['immediate'] / 100
            carryover_pct = carryover_dict[media['channel']]['carryover'] / 100
            
            if carryover_pct > 0 and scale and shape:
                # Calculate how the carryover is distributed over time using the Weibull CDF
                # Since data is weekly, we work with weeks as time units
                
                # Create weekly time points
                max_weeks = 52  # One year of weeks
                weeks = np.arange(0, max_weeks + 1)
                
                # Calculate CDF values for weeks
                # The scale parameter is already in the correct unit (proportion of max time)
                cdf_values = weibull_min.cdf(weeks / max_weeks, c=shape, scale=scale)
                
                # Week 1: Just the immediate effect
                week1_pct = immediate_pct * 100
                
                # Week 2: Carryover that happens in week 2
                week1_cdf = cdf_values[1] if len(cdf_values) > 1 else 0
                week2_cdf = cdf_values[2] if len(cdf_values) > 2 else 0
                week2_carryover = week2_cdf - week1_cdf
                week2_pct = carryover_pct * week2_carryover * 100
                
                # Week 3: Carryover that happens in week 3
                week3_cdf = cdf_values[3] if len(cdf_values) > 3 else 0
                week3_carryover = week3_cdf - week2_cdf
                week3_pct = carryover_pct * week3_carryover * 100
                
                # Week 4: Carryover that happens in week 4
                week4_cdf = cdf_values[4] if len(cdf_values) > 4 else 0
                week4_carryover = week4_cdf - week3_cdf
                week4_pct = carryover_pct * week4_carryover * 100
                
                # First month: Sum of all 4 weeks
                month1_pct = week1_pct + week2_pct + week3_pct + week4_pct
                
                # Other months: 100% - Month 1
                other_pct = 100 - month1_pct
                
                # Ensure we don't have negative values
                week2_pct = max(0, week2_pct)
                week3_pct = max(0, week3_pct)
                week4_pct = max(0, week4_pct)
                month1_pct = min(100, max(0, month1_pct))
                other_pct = max(0, other_pct)
            else:
                # No carryover or no parameters
                week1_pct = immediate_pct * 100
                week2_pct = 0
                week3_pct = 0
                week4_pct = 0
                month1_pct = immediate_pct * 100
                other_pct = 0
            
            adstock_data.append({
                'channel': media['channel'],
                'week1': week1_pct,
                'week2': week2_pct,
                'week3': week3_pct,
                'week4': week4_pct,
                'month1_total': month1_pct,
                'other_months': other_pct
            })
    
    return render_template('dashboard.html', 
                         media_data=media_data,
                         all_variables=all_variables,
                         kpi_data=kpi_data,
                         roi_curves=roi_curves,
                         yearly_data=yearly_data,
                         seasonality_data=seasonality_data,
                         spend_contrib_data=spend_contrib_data,
                         ci_data=ci_data,
                         carryover_data=carryover_data,
                         adstock_data=adstock_data,
                         adstock_type=adstock_type,
                         total_investment=total_investment,
                         model_id=MODEL_ID,
                         model_metrics=model_metrics,
                         model_config=model_config,
                         grouped_contributions=grouped_contributions,
                         monthly_pressure_data=monthly_pressure_data,
                         trend_analysis=trend_analysis)

if __name__ == '__main__':
    app.run(debug=True)