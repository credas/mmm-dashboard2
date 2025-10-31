from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from scipy.stats import weibull_min
from scipy.optimize import minimize
import os
from datetime import datetime

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

@app.route('/optimize')
def optimize():
    # Read raw data to get actual spend amounts
    raw_df = pd.read_csv(f'{MODEL_DIR}/raw_data.csv')
    
    # Get date range from decomp data
    decomp_df = pd.read_csv(DECOMP_PATH)
    model_df = decomp_df[decomp_df['solID'] == MODEL_ID].copy()
    
    if model_df.empty:
        return "No data found for model", 404
    
    # Get date range
    date_min = model_df['ds'].min()
    date_max = model_df['ds'].max()
    
    # Calculate average weekly spend for each media channel from raw data
    avg_spends = {}
    for channel in MEDIA_CHANNELS:
        if channel in raw_df.columns:
            avg_spends[channel] = raw_df[channel].mean()
    
    return render_template('optimize.html',
                         date_min=date_min,
                         date_max=date_max,
                         media_channels=MEDIA_CHANNELS,
                         avg_spends=avg_spends,
                         model_config=model_config,
                         model_id=MODEL_ID)

@app.route('/api/get_averages', methods=['POST'])
def get_averages():
    data = request.json
    start_date = data['start_date']
    end_date = data['end_date']
    
    # Read raw data to get actual spend
    raw_df = pd.read_csv(f'{MODEL_DIR}/raw_data.csv')
    
    # Filter by date range
    mask = (raw_df['weekstart'] >= start_date) & (raw_df['weekstart'] <= end_date)
    period_df = raw_df[mask]
    
    if period_df.empty:
        return jsonify({'error': 'No data found for selected period'}), 400
    
    # Read model parameters for additional metrics
    agg_df = pd.read_csv(MODEL_PATH)
    media_df = agg_df[(agg_df['solID'] == MODEL_ID) & (agg_df['rn'].isin(MEDIA_CHANNELS))]
    
    hyper_df = pd.read_csv('models/FinBee/loans1-model/pareto_hyperparameters.csv')
    hyper_row = hyper_df[hyper_df['solID'] == MODEL_ID].iloc[0] if not hyper_df[hyper_df['solID'] == MODEL_ID].empty else None
    
    
    # Calculate metrics for each media channel
    channel_metrics = {}
    for channel in MEDIA_CHANNELS:
        if channel in period_df.columns and channel in media_df['rn'].values:
            avg_spend = float(period_df[channel].mean())
            
            # Get model parameters
            channel_row = media_df[media_df['rn'] == channel].iloc[0]
            mean_spend = float(channel_row['mean_spend'])
            mean_spend_adstocked = float(channel_row['mean_spend_adstocked'])
            coef = float(channel_row['coef'])
            
            # Calculate adstock multiplier
            adstock_mult = mean_spend_adstocked / mean_spend if mean_spend > 0 else 1.0
            
            # Get hyperparameters
            alpha = gamma = scale = shape = None
            if hyper_row is not None:
                alpha_col = f"{channel}_alphas"
                inflexion_col = f"{channel}_inflexion"
                scale_col = f"{channel}_scales"
                shape_col = f"{channel}_shapes"
                
                if alpha_col in hyper_row:
                    alpha = float(hyper_row[alpha_col])
                if inflexion_col in hyper_row:
                    gamma = float(hyper_row[inflexion_col])
                if scale_col in hyper_row:
                    scale = float(hyper_row[scale_col])
                if shape_col in hyper_row:
                    shape = float(hyper_row[shape_col])
            
            # Calculate saturation and ROI at average spend
            adstocked_spend = avg_spend * adstock_mult
            if alpha and gamma:
                saturation = (adstocked_spend ** alpha) / (adstocked_spend ** alpha + gamma ** alpha)
                response = coef * saturation
                roi = response / avg_spend if avg_spend > 0 else 0
            else:
                saturation = 0
                roi = 0
            
            # Calculate adstock2 and saturation2 using weekly simulation
            adstocked_sum2 = 0
            saturated_sum2 = 0
            
            if scale and shape and alpha and gamma:
                # Simulate weekly adstock decay over selected period
                n_weeks = len(period_df)
                total_weeks = n_weeks + 52  # Include carryover weeks
                
                # Initialize array to store adstocked spend for each week
                weekly_adstock = np.zeros(total_weeks)
                
                # Use scipy's Weibull distribution
                from scipy.stats import weibull_min
                
                # Calculate carryover multiplier (total multiplier - immediate effect)
                carryover_mult = adstock_mult - 1.0 if adstock_mult > 1.0 else 0.0
                
                # For each week of spend, add immediate effect + distribute carryover
                for spend_week in range(n_weeks):
                    # Immediate effect in the same week
                    weekly_adstock[spend_week] += avg_spend * 1.0
                    
                    # Distribute carryover effects according to Weibull decay
                    for lag in range(1, min(52, total_weeks - spend_week)):
                        target_week = spend_week + lag
                        
                        # Get the portion of carryover for this lag
                        cdf_current = weibull_min.cdf((lag + 1)/52, c=shape, scale=scale)
                        cdf_previous = weibull_min.cdf(lag/52, c=shape, scale=scale)
                        decay_portion = cdf_current - cdf_previous
                        
                        # Apply this portion of the total carryover
                        carryover_amount = avg_spend * carryover_mult * decay_portion
                        weekly_adstock[target_week] += carryover_amount
                
                # Sum total adstocked spend
                adstocked_sum2 = np.sum(weekly_adstock)
                
                # Now apply saturation to each week's total adstocked spend
                for week_idx, week_total in enumerate(weekly_adstock):
                    if week_total > 0:
                        # Apply saturation to the week's total adstocked spend
                        week_saturation = (week_total ** alpha) / (week_total ** alpha + gamma ** alpha)
                        week_response = coef * week_saturation
                        saturated_sum2 += week_response
                
                # Calculate ROI2
                roi2 = saturated_sum2 / (avg_spend * n_weeks) if avg_spend > 0 and n_weeks > 0 else 0
            else:
                roi2 = roi  # Fallback to simple ROI
            
            
            channel_metrics[channel] = {
                'avg_spend': avg_spend,
                'adstocked_spend': round(adstocked_spend, 0),
                'adstock_mult': round(adstock_mult, 3),
                'saturation_pct': round(saturation * 100, 1),
                'roi': round(roi, 1),
                'coef': coef,
                'alpha': alpha,
                'gamma': gamma,
                'scale': scale,
                'shape': shape,
                'adstocked_sum2': round(adstocked_sum2, 0),
                'saturated_sum2': round(saturated_sum2, 0),
                'roi2': round(roi2, 1),
                'n_weeks': n_weeks if 'n_weeks' in locals() else len(period_df)
            }
    
    return jsonify({
        'channels': channel_metrics,
        'weeks': len(period_df),
        'total_spend': float(sum(period_df[ch].sum() for ch in MEDIA_CHANNELS if ch in period_df.columns))
    })

@app.route('/api/optimize', methods=['POST'])
def run_optimization():
    data = request.json
    start_date = data['start_date']
    end_date = data['end_date']
    budget_constraints = data['budget_constraints']
    
    # Read both raw data (for spend) and decomp data (for model parameters)
    raw_df = pd.read_csv(f'{MODEL_DIR}/raw_data.csv')
    decomp_df = pd.read_csv(DECOMP_PATH)
    model_df = decomp_df[decomp_df['solID'] == MODEL_ID].copy()
    
    # Filter raw data by date range for actual spend
    raw_mask = (raw_df['weekstart'] >= start_date) & (raw_df['weekstart'] <= end_date)
    raw_period_df = raw_df[raw_mask].copy()
    
    # Filter model data by date range for decomposition
    model_mask = (model_df['ds'] >= start_date) & (model_df['ds'] <= end_date)
    model_period_df = model_df[model_mask].copy()
    
    if raw_period_df.empty or model_period_df.empty:
        return jsonify({'error': 'No data found for selected period'}), 400
    
    n_weeks = len(raw_period_df)
    
    # Read model parameters
    agg_df = pd.read_csv(MODEL_PATH)
    media_df = agg_df[(agg_df['solID'] == MODEL_ID) & (agg_df['rn'].isin(MEDIA_CHANNELS))]
    
    hyper_df = pd.read_csv('models/FinBee/loans1-model/pareto_hyperparameters.csv')
    hyper_row = hyper_df[hyper_df['solID'] == MODEL_ID].iloc[0] if not hyper_df[hyper_df['solID'] == MODEL_ID].empty else None
    
    # Build optimization parameters for each channel
    channels_data = {}
    active_channels = []
    
    for channel in MEDIA_CHANNELS:
        if channel in raw_period_df.columns and channel in budget_constraints:
            constraint = budget_constraints[channel]
            
            # Get model parameters
            channel_row = media_df[media_df['rn'] == channel]
            if channel_row.empty:
                continue
                
            # Get total spend and decomp values for the entire model period
            total_spend = float(channel_row['total_spend'].iloc[0]) if not channel_row['total_spend'].empty else 0
            total_decomp = float(channel_row['xDecompAgg'].iloc[0]) if not channel_row['xDecompAgg'].empty else 0
            
            # Get additional model data
            mean_response = float(channel_row['mean_response'].iloc[0]) if 'mean_response' in channel_row and not channel_row['mean_response'].empty else 0
            mean_spend_adstocked = float(channel_row['mean_spend_adstocked'].iloc[0]) if 'mean_spend_adstocked' in channel_row and not channel_row['mean_spend_adstocked'].empty else 0
            
            channel_data = {
                'name': channel,
                'current_spend': raw_period_df[channel].sum(),
                'avg_weekly': raw_period_df[channel].mean(),
                'min_weekly': constraint['min'],
                'max_weekly': constraint['max'],
                'coef': float(channel_row['coef'].iloc[0]) if not channel_row['coef'].empty else 0,
                'total_spend': total_spend,
                'total_decomp': total_decomp,
                'base_roas': total_decomp / total_spend if total_spend > 0 else 0,
                'mean_response': mean_response,
                'mean_spend_adstocked': mean_spend_adstocked
            }
            
            # Get hyperparameters
            if hyper_row is not None:
                alpha_col = f"{channel}_alphas"
                gamma_col = f"{channel}_gammas"
                inflexion_col = f"{channel}_inflexion"
                scale_col = f"{channel}_scales"
                shape_col = f"{channel}_shapes"
                
                if alpha_col in hyper_row:
                    channel_data['alpha'] = float(hyper_row[alpha_col])
                    # Use inflexion if available, otherwise use gamma
                    if inflexion_col in hyper_row and pd.notna(hyper_row[inflexion_col]):
                        channel_data['gamma'] = float(hyper_row[inflexion_col])
                    else:
                        channel_data['gamma'] = float(hyper_row[gamma_col])
                if scale_col in hyper_row:
                    channel_data['scale'] = float(hyper_row[scale_col])
                    channel_data['shape'] = float(hyper_row[shape_col])
            
            channels_data[channel] = channel_data
            active_channels.append(channel)
    
    # Initial spend allocation (current average)
    x0 = np.array([channels_data[ch]['avg_weekly'] for ch in active_channels])
    
    # Bounds for optimization
    bounds = [(channels_data[ch]['min_weekly'], channels_data[ch]['max_weekly']) for ch in active_channels]
    
    # Objective function: maximize total response with efficiency penalty
    def objective(x):
        total_response = 0
        efficiency_penalty = 0
        
        for i, channel in enumerate(active_channels):
            weekly_spend = x[i]
            ch_data = channels_data[channel]
            
            # Calculate response for the weekly spend
            response = calculate_response(weekly_spend, ch_data)
            total_response += response
            
            # Add penalty for very low saturation (inefficient spend)
            if 'alpha' in ch_data and 'gamma' in ch_data:
                adstock_mult = ch_data.get('mean_spend_adstocked', 0) / ch_data.get('avg_weekly', 1)
                adstocked = weekly_spend * adstock_mult
                saturation = (adstocked ** ch_data['alpha']) / (adstocked ** ch_data['alpha'] + ch_data['gamma'] ** ch_data['alpha'])
                
                # Penalize if saturation is below 5% (too little spend to be effective)
                if saturation < 0.05:
                    efficiency_penalty += (0.05 - saturation) * 100000
        
        # Return negative for minimization (multiply by weeks for total)
        return -(total_response * n_weeks) + efficiency_penalty
    
    # Run optimization
    result = minimize(objective, x0, method='SLSQP', bounds=bounds)
    
    if not result.success:
        return jsonify({'error': 'Optimization failed: ' + result.message}), 500
    
    # Calculate results
    optimized_spends = result.x
    
    # Prepare results for display
    initial_results = []
    optimized_results = []
    
    total_initial_spend = 0
    total_initial_response = 0
    total_optimized_spend = 0
    total_optimized_response = 0
    
    for i, channel in enumerate(active_channels):
        ch_data = channels_data[channel]
        
        # Initial allocation
        initial_spend = ch_data['avg_weekly']
        initial_response = calculate_response(initial_spend, ch_data)
        
        # Calculate saturation for initial spend
        initial_saturation = 0
        if 'alpha' in ch_data and 'gamma' in ch_data:
            adstock_mult = ch_data.get('mean_spend_adstocked', 0) / ch_data.get('avg_weekly', 1)
            adstocked = initial_spend * adstock_mult
            initial_saturation = (adstocked ** ch_data['alpha']) / (adstocked ** ch_data['alpha'] + ch_data['gamma'] ** ch_data['alpha'])
        
        initial_results.append({
            'channel': channel,
            'spend': initial_spend * n_weeks,
            'weekly_spend': initial_spend,
            'response': initial_response * n_weeks,
            'response_pct': 0,  # Will calculate after total
            'spend_pct': 0,
            'roas': initial_response / initial_spend if initial_spend > 0 else 0,
            'saturation': initial_saturation * 100
        })
        
        total_initial_spend += initial_spend * n_weeks
        total_initial_response += initial_response * n_weeks
        
        # Optimized allocation
        opt_spend = optimized_spends[i]
        opt_response = calculate_response(opt_spend, ch_data)
        
        # Calculate saturation for optimized spend
        opt_saturation = 0
        if 'alpha' in ch_data and 'gamma' in ch_data:
            adstock_mult = ch_data.get('mean_spend_adstocked', 0) / ch_data.get('avg_weekly', 1)
            adstocked = opt_spend * adstock_mult
            opt_saturation = (adstocked ** ch_data['alpha']) / (adstocked ** ch_data['alpha'] + ch_data['gamma'] ** ch_data['alpha'])
        
        optimized_results.append({
            'channel': channel,
            'spend': opt_spend * n_weeks,
            'weekly_spend': opt_spend,
            'response': opt_response * n_weeks,
            'response_pct': 0,  # Will calculate after total
            'spend_pct': 0,
            'roas': opt_response / opt_spend if opt_spend > 0 else 0,
            'saturation': opt_saturation * 100
        })
        
        total_optimized_spend += opt_spend * n_weeks
        total_optimized_response += opt_response * n_weeks
    
    # Calculate percentages
    for res in initial_results:
        res['spend_pct'] = (res['spend'] / total_initial_spend * 100) if total_initial_spend > 0 else 0
        res['response_pct'] = (res['response'] / total_initial_response * 100) if total_initial_response > 0 else 0
    
    for res in optimized_results:
        res['spend_pct'] = (res['spend'] / total_optimized_spend * 100) if total_optimized_spend > 0 else 0
        res['response_pct'] = (res['response'] / total_optimized_response * 100) if total_optimized_response > 0 else 0
    
    # Calculate lift
    spend_change = ((total_optimized_spend - total_initial_spend) / total_initial_spend * 100) if total_initial_spend > 0 else 0
    response_change = ((total_optimized_response - total_initial_response) / total_initial_response * 100) if total_initial_response > 0 else 0
    
    return jsonify({
        'success': True,
        'period': {
            'start': start_date,
            'end': end_date,
            'weeks': n_weeks
        },
        'initial': {
            'results': initial_results,
            'total_spend': total_initial_spend,
            'total_response': total_initial_response,
            'total_roas': total_initial_response / total_initial_spend if total_initial_spend > 0 else 0
        },
        'optimized': {
            'results': optimized_results,
            'total_spend': total_optimized_spend,
            'total_response': total_optimized_response,
            'total_roas': total_optimized_response / total_optimized_spend if total_optimized_spend > 0 else 0
        },
        'lift': {
            'spend_change': spend_change,
            'response_change': response_change
        }
    })

def calculate_response(spend, channel_data):
    """Calculate response for a given spend level using Robyn's transformation pipeline
    
    Pipeline: Raw Spend → Adstock → Saturation → Response
    """
    
    if spend <= 0:
        return 0
    
    # Get model parameters
    coef = channel_data.get('coef', 0)
    mean_spend = channel_data.get('avg_weekly', 0)
    mean_spend_adstocked = channel_data.get('mean_spend_adstocked', 0)
    
    # Calculate adstock multiplier
    if mean_spend > 0 and mean_spend_adstocked > 0:
        adstock_mult = mean_spend_adstocked / mean_spend
    else:
        adstock_mult = 1.0
    
    # Apply adstock transformation
    adstocked_spend = spend * adstock_mult
    
    # Get saturation parameters
    alpha = channel_data.get('alpha', 0)
    gamma = channel_data.get('gamma', 0)  # inflexion point
    
    if alpha > 0 and gamma > 0:
        # Apply Hill saturation
        saturation = (adstocked_spend ** alpha) / (adstocked_spend ** alpha + gamma ** alpha)
        
        # Calculate response
        response = coef * saturation
    else:
        # Fallback to linear scaling if parameters missing
        mean_response = channel_data.get('mean_response', 0)
        if mean_spend > 0:
            response = mean_response * (spend / mean_spend)
        else:
            response = 0
    
    return response

if __name__ == '__main__':
    app.run(debug=True)