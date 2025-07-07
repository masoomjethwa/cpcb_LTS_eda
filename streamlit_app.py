#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit Dashboard for CPCB Air Quality Data (1990–2015)

This script creates an interactive web dashboard using Streamlit to analyze
and visualize CPCB air quality data. It includes filtering, KPI metrics,
various time series plots, geographical animations, anomaly detection,
seasonal decomposition, ARIMA forecasting, and time series clustering.

To run this app:
1. Make sure you have the 'CPCB_data_from1990_2015.csv' file in a directory
   accessible by the script (update DATA_FILE_PATH if needed).
2. Install necessary libraries:
   pip install streamlit pandas numpy plotly seaborn matplotlib statsmodels scikit-learn
3. Run from your terminal:
   streamlit run streamlit_app.py

@author: codernumber1
coding assistant: 891273
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.arima.model import ARIMA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import os

# Set Streamlit page configuration
st.set_page_config(layout="wide", page_title="CPCB Air Quality Dashboard")

# Define the path to your CSV file.
# IMPORTANT: Update this path if your CSV is not in the same directory as this script.
DATA_FILE_PATH = "CPCB_data_from1990_2015.csv"

# --- Data Loading and Preprocessing ---
@st.cache_data
def load_and_preprocess_data(file_path):
    """
    Loads the CPCB dataset and performs initial cleaning and preprocessing.
    This function is cached by Streamlit for performance.
    """
    if not os.path.exists(file_path):
        st.error(f"Error: Data file not found at '{file_path}'. "
                 "Please ensure the CSV is in the correct location.")
        st.stop() # Stop the app if data file is not found

    try:
        df = pd.read_csv(file_path, encoding='latin-1', low_memory=False)
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        st.stop()

    # Replace 'NA' strings with NaN values
    df.replace('NA', np.nan, inplace=True)

    # Custom function to parse 'sampling_date'
    def parse_sampling_date_custom(date_str):
        if pd.isna(date_str):
            return pd.NaT
        try:
            date_part = str(date_str).split("-")[-1].strip()
            return datetime.strptime(date_part, "M%m%Y")
        except (ValueError, IndexError):
            return pd.NaT

    df["sampling_date"] = df["sampling_date"].apply(parse_sampling_date_custom)

    # Convert 'date' column to datetime, handling errors
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors='coerce', dayfirst=True)
    else:
        st.warning("Column 'date' not found in the dataset. Some features might be affected.")

    # Extract temporal features
    df["year"] = df["sampling_date"].dt.year
    df["month"] = df["sampling_date"].dt.month
    df["day"] = df["sampling_date"].dt.day

    # List of pollutant columns
    pollutant_cols = ["so2", "no2", "rspm", "spm", "pm2_5"]

    # Convert pollutant columns to numeric, coercing errors
    for col in pollutant_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Impute missing values in pollutant columns with their mean
    for col in pollutant_cols:
        if df[col].isnull().sum() > 0:
            mean_val = df[col].mean()
            df.loc[:, col] = df[col].fillna(mean_val)

    # Drop rows where essential columns for analysis are missing
    df.dropna(subset=["sampling_date", "state", "location"], inplace=True)

    # Melt pollutant columns for easier plotting
    df_melted = df.melt(
        id_vars=[col for col in df.columns if col not in pollutant_cols],
        value_vars=pollutant_cols,
        var_name='pollutant',
        value_name='value'
    )
    df_melted.dropna(subset=['value'], inplace=True) # Drop rows where 'value' is NaN after melting

    return df, df_melted, pollutant_cols

df_original, df_melted, POLLUTANT_COLUMNS = load_and_preprocess_data(DATA_FILE_PATH)

# --- Sidebar Filters ---
st.sidebar.header("🔧 Filters")

# Get unique values for filters from the preprocessed data
all_states = sorted(df_original['state'].dropna().unique())
all_locations = sorted(df_original['location'].dropna().unique())

selected_states = st.sidebar.multiselect(
    "Select State(s)",
    options=all_states,
    default=all_states[:3] if len(all_states) > 3 else all_states
)

selected_locations = st.sidebar.multiselect(
    "Select Location(s)",
    options=all_locations,
    default=all_locations[:5] if len(all_locations) > 5 else all_locations
)

selected_pollutants = st.sidebar.multiselect(
    "Select Pollutant(s)",
    options=POLLUTANT_COLUMNS,
    default=["so2", "no2", "rspm"]
)

# Date range filter
min_date = df_original["sampling_date"].min()
max_date = df_original["sampling_date"].max()

if pd.isna(min_date) or pd.isna(max_date):
    st.error("Date range could not be determined from the data. Check 'sampling_date' column.")
    st.stop()

date_range_selection = st.sidebar.date_input(
    "Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Ensure date_range_selection has two dates before proceeding
if len(date_range_selection) == 2:
    start_date = pd.to_datetime(date_range_selection[0])
    end_date = pd.to_datetime(date_range_selection[1])
else:
    st.warning("Please select a start and end date for the filter.")
    start_date = min_date
    end_date = max_date

use_rolling_avg = st.sidebar.checkbox("Apply 30-day Rolling Average to Time Series", value=False)

# Filter the data based on sidebar selections
filtered_df = df_original[
    df_original['state'].isin(selected_states) &
    df_original['location'].isin(selected_locations) &
    df_original['sampling_date'].between(start_date, end_date)
].copy() # Use .copy() to avoid SettingWithCopyWarning

# Filter melted data for relevant plots
filtered_melted_df = df_melted[
    df_melted['state'].isin(selected_states) &
    df_melted['location'].isin(selected_locations) &
    df_melted['pollutant'].isin(selected_pollutants) &
    df_melted['date'].between(start_date, end_date)
].copy()

# --- Main Dashboard Content ---
st.title("🧪 CPCB Air Quality Dashboard (1990–2015)")

if filtered_df.empty or filtered_melted_df.empty:
    st.warning("No data available for the selected filters. Please adjust your selections.")
else:
    # --- 10. KPI Metrics Box ---
    st.subheader("📊 KPI Metrics for Selected Pollutants")
    kpi_cols = st.columns(len(selected_pollutants) * 2) # Two metrics per pollutant (mean, max)
    col_idx = 0
    for pollutant in selected_pollutants:
        values = filtered_df[pollutant].dropna()
        if not values.empty:
            kpi_cols[col_idx].metric(f"{pollutant.upper()} - Mean", f"{values.mean():.2f}")
            kpi_idx = col_idx + 1 if col_idx + 1 < len(kpi_cols) else col_idx # Ensure index is within bounds
            kpi_cols[kpi_idx].metric(f"{pollutant.upper()} - Max", f"{values.max():.2f}")
            col_idx += 2
        else:
            st.write(f"No data for {pollutant.upper()} in selected range.")

    st.markdown("---")

    # --- 2. Time Series Panel (Line Plot + Rolling Average Toggle) ---
    st.subheader("📈 Pollutant Trends Over Time")
    st.write("View the concentration of selected pollutants over the chosen date range.")
    for pollutant in selected_pollutants:
        ts_data = filtered_melted_df[filtered_melted_df['pollutant'] == pollutant].dropna(subset=['date', 'value'])
        if not ts_data.empty:
            fig = px.line(
                ts_data,
                x="date",
                y="value",
                color="location",
                title=f"{pollutant.upper()} Concentration Over Time",
                labels={'value': f'{pollutant.upper()} Concentration', 'date': 'Date'}
            )
            if use_rolling_avg:
                for loc in ts_data["location"].unique():
                    sub_ts = ts_data[ts_data["location"] == loc].set_index("date")
                    # Ensure the data is numeric before rolling calculation
                    sub_ts['value'] = pd.to_numeric(sub_ts['value'], errors='coerce')
                    rolling_mean = sub_ts['value'].rolling("30D").mean().dropna()
                    if not rolling_mean.empty:
                        fig.add_scatter(x=rolling_mean.index, y=rolling_mean.values,
                                        mode='lines', name=f"{loc} (30D Rolling Avg)",
                                        line=dict(dash='dash', width=2))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No time series data for {pollutant.upper()} in the selected filters.")

    st.markdown("---")

    # --- 3. Multi-Location Compare (Facet Line Plot) ---
    st.subheader("🏙️ Multi-Location Comparison")
    st.write("Compare pollutant trends across selected locations in separate panels.")
    # Limit selection to a reasonable number of locations for readability
    if len(selected_locations) > 6:
        st.info("Displaying trends for the first 6 selected locations for clarity in this view.")
        compare_locations = selected_locations[:6]
    else:
        compare_locations = selected_locations

    compare_df = filtered_melted_df[
        filtered_melted_df['location'].isin(compare_locations)
    ].dropna(subset=['date', 'value', 'location', 'pollutant'])

    if not compare_df.empty:
        # Use a consistent date range for all facets for better comparison
        # This might mean some facets have gaps if data is missing for a period.
        fig_facet = px.line(
            compare_df,
            x="date",
            y="value",
            color="pollutant",
            facet_col="location",
            facet_col_wrap=3,
            title="Pollutant Trends Across Locations",
            labels={'value': 'Concentration', 'date': 'Date'},
            height=500
        )
        fig_facet.update_yaxes(matches=None) # Allow independent y-axes for better comparison across locations
        fig_facet.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1])) # Clean facet titles
        st.plotly_chart(fig_facet, use_container_width=True)
    else:
        st.info("No data to compare across selected locations and pollutants.")

    st.markdown("---")

    # --- 4. Monthly Heatmap View ---
    st.subheader("🗓️ Monthly Heatmap View")
    st.write("Visualize average pollutant levels by month and year to identify seasonal patterns.")
    heatmap_data = filtered_df.copy()
    heatmap_data["Year"] = heatmap_data["sampling_date"].dt.year
    heatmap_data["Month"] = heatmap_data["sampling_date"].dt.month

    for pollutant in selected_pollutants:
        pivot_table = heatmap_data.pivot_table(values=pollutant, index="Month", columns="Year", aggfunc="mean")
        if not pivot_table.empty:
            fig, ax = plt.subplots(figsize=(12, 5))
            sns.heatmap(pivot_table, cmap="viridis", annot=False, fmt=".1f", linewidths=.5, ax=ax,
                        cbar_kws={'label': f'Average {pollutant.upper()} Concentration'})
            ax.set_title(f"Monthly Heatmap of {pollutant.upper()}")
            st.pyplot(fig)
            plt.close(fig) # Close plot to free memory
        else:
            st.info(f"No heatmap data for {pollutant.upper()} in the selected filters.")

    st.markdown("---")

    # --- 5. Top Cities Ranking ---
    st.subheader("🏆 Top Cities by Mean Pollution")
    st.write("See which cities have the highest average pollutant concentrations.")
    for pollutant in selected_pollutants:
        top_cities = filtered_df.groupby("location")[pollutant].mean().sort_values(ascending=False).head(10).reset_index()
        if not top_cities.empty:
            fig = px.bar(top_cities, x=pollutant, y="location", orientation="h",
                         title=f"Top 10 Cities - Mean {pollutant.upper()}",
                         labels={pollutant: f'Mean {pollutant.upper()} Concentration', 'location': 'City'},
                         color=pollutant, color_continuous_scale=px.colors.sequential.Viridis)
            fig.update_layout(yaxis={'categoryorder':'total ascending'}) # Order bars from smallest to largest
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No top cities data for {pollutant.upper()} in the selected filters.")

    st.markdown("---")

    # --- 6. Animated Map (optional) ---
    st.subheader("🗺️ Animated Pollution Spread (Simulated Locations)")
    st.write("Watch how average pollutant levels change across locations over the years. (Locations are simulated for demonstration).")

    # Aggregate data by year and location for the animation
    animated_map_data = filtered_melted_df.groupby(['year', 'location', 'pollutant'])['value'].mean().reset_index()

    # Add simulated latitude and longitude for demonstration purposes.
    # These ranges approximate India's geographical boundaries.
    # We use a dictionary to ensure consistent lat/lon for each unique location.
    @st.cache_data
    def get_simulated_lat_lon(locations):
        lat_map = {loc: np.random.uniform(9.0, 34.0) for loc in locations}
        lon_map = {loc: np.random.uniform(69.0, 96.0) for loc in locations}
        return lat_map, lon_map

    unique_locations_in_data = animated_map_data['location'].unique()
    location_lat_map, location_lon_map = get_simulated_lat_lon(unique_locations_in_data)

    animated_map_data["lat"] = animated_map_data["location"].map(location_lat_map)
    animated_map_data["lon"] = animated_map_data["location"].map(location_lon_map)

    # Drop rows where lat/lon might be NaN if location was not mapped
    animated_map_data.dropna(subset=['lat', 'lon', 'value'], inplace=True)

    if not animated_map_data.empty:
        # Allow user to select pollutant for animation
        animated_pollutant = st.selectbox("Select Pollutant for Animated Map", selected_pollutants)
        animated_map_filtered = animated_map_data[animated_map_data['pollutant'] == animated_pollutant]

        if not animated_map_filtered.empty:
            fig_animated_map = px.scatter_mapbox(
                animated_map_filtered,
                lat="lat",
                lon="lon",
                color="value", # Color by pollutant value
                size="value", # Size by pollutant value
                animation_frame="year", # Animate over years
                animation_group="location", # Group animation by location
                hover_name="location",
                mapbox_style="carto-positron", # Or "open-street-map", "stamen-terrain"
                zoom=3,
                center={"lat": 22.5, "lon": 80}, # Approximate center of India
                title=f"Animated {animated_pollutant.upper()} Concentration by Location Over Years (Simulated)"
            )
            fig_animated_map.update_layout(transition={'duration': 5000}) # Slow down animation
            st.plotly_chart(fig_animated_map, use_container_width=True)
        else:
            st.info(f"No data for {animated_pollutant.upper()} to animate.")
    else:
        st.info("No data available for animated map after filtering and simulating coordinates.")

    st.markdown("---")

    # --- 7. Correlation Over Time Plot ---
    st.subheader("🤝 Rolling Correlation Between Pollutants")
    st.write("Analyze how the relationship between two pollutants changes over time.")
    # Allow selection of two pollutants for correlation
    col_corr1, col_corr2 = st.columns(2)
    corr_pollutant1 = col_corr1.selectbox("Select Pollutant 1", POLLUTANT_COLUMNS, index=0)
    corr_pollutant2 = col_corr2.selectbox("Select Pollutant 2", POLLUTANT_COLUMNS, index=1)

    if corr_pollutant1 == corr_pollutant2:
        st.warning("Please select two different pollutants for correlation analysis.")
    else:
        # Aggregate data to monthly average for correlation calculation
        corr_data = filtered_df[['sampling_date', 'location', corr_pollutant1, corr_pollutant2]].dropna()
        if not corr_data.empty:
            corr_data = corr_data.set_index('sampling_date')
            rolling_correlations = []
            for loc in corr_data['location'].unique():
                loc_df = corr_data[corr_data['location'] == loc].resample('ME').mean().dropna()
                if len(loc_df) > 30: # Need enough data points for rolling correlation
                    loc_df['rolling_corr'] = loc_df[corr_pollutant1].rolling(window=30, min_periods=10).corr(loc_df[corr_pollutant2])
                    loc_df['location'] = loc
                    rolling_correlations.append(loc_df.reset_index())

            if rolling_correlations:
                rolling_corr_df = pd.concat(rolling_correlations)
                fig_rolling_corr = px.line(
                    rolling_corr_df.dropna(subset=['rolling_corr']),
                    x='sampling_date',
                    y='rolling_corr',
                    color='location',
                    title=f'Rolling Correlation (30-Day) between {corr_pollutant1.upper()} and {corr_pollutant2.upper()}',
                    labels={'rolling_corr': 'Correlation Coefficient', 'sampling_date': 'Date'}
                )
                st.plotly_chart(fig_rolling_corr, use_container_width=True)
            else:
                st.info("Not enough data points for rolling correlation for selected pollutants/locations.")
        else:
            st.info("No data available for selected pollutants for correlation analysis.")

    st.markdown("---")

    # --- 8. Interactive Time Series (Plotly) ---
    st.subheader("🔍 Interactive Pollutant Trends")
    st.write("Explore detailed pollutant trends with zoom and range selection capabilities.")
    interactive_pollutant = st.selectbox("Select Pollutant for Interactive Trend", selected_pollutants)

    interactive_data = filtered_melted_df[
        (filtered_melted_df['pollutant'] == interactive_pollutant)
    ].dropna(subset=['date', 'value'])

    if not interactive_data.empty:
        fig_interactive = px.line(
            interactive_data,
            x='date',
            y='value',
            color='location',
            title=f'Interactive {interactive_pollutant.upper()} Concentration Over Time by Location',
            labels={'value': f'{interactive_pollutant.upper()} Concentration', 'date': 'Date'}
        )
        fig_interactive.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )
        st.plotly_chart(fig_interactive, use_container_width=True)
    else:
        st.info(f"No interactive time series data for {interactive_pollutant.upper()} in the selected filters.")

    st.markdown("---")

    # --- 9. Pollution Spike Detector (Z-score or Threshold) ---
    st.subheader("🚨 Pollution Spike Detector")
    st.write("Identify anomalous spikes in pollutant concentrations using Z-score.")
    col_spike_loc, col_spike_poll = st.columns(2)
    spike_location = col_spike_loc.selectbox("Select Location for Spike Detection", selected_locations)
    spike_pollutant = col_spike_poll.selectbox("Select Pollutant for Spike Detection", selected_pollutants)
    spike_threshold = st.slider("Z-score Threshold for Spikes", min_value=1.0, max_value=5.0, value=3.0, step=0.5)

    spike_data = filtered_df[(filtered_df['location'] == spike_location)][['sampling_date', spike_pollutant]].dropna()
    if not spike_data.empty:
        location_series = spike_data.set_index('sampling_date')[spike_pollutant].sort_index()
        mean_val = location_series.mean()
        std_val = location_series.std()

        if std_val > 0:
            location_zscore = (location_series - mean_val) / std_val
            spikes = location_series[location_zscore.abs() > spike_threshold]

            fig, ax = plt.subplots(figsize=(14, 7))
            ax.plot(location_series.index, location_series.values, label=f'{spike_location} {spike_pollutant.upper()}', color='blue', alpha=0.7)
            ax.scatter(spikes.index, spikes.values, color='red', s=50, zorder=5, label=f'Spikes (Z > {spike_threshold})')
            ax.set_title(f'{spike_pollutant.upper()} Pollution Spikes for {spike_location}', fontsize=16)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel(f'{spike_pollutant.upper()} Concentration', fontsize=12)
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info(f"Standard deviation for {spike_pollutant.upper()} in {spike_location} is zero. Cannot detect spikes.")
    else:
        st.info(f"No data for {spike_pollutant.upper()} in {spike_location} for spike detection.")

    st.markdown("---")

    # --- 9. Trend + Seasonality Decomposition (STL) ---
    st.subheader("📉 Trend + Seasonality Decomposition (STL)")
    st.write("Decompose a time series into its trend, seasonal, and residual components.")
    col_stl_loc, col_stl_poll = st.columns(2)
    stl_location = col_stl_loc.selectbox("Select Location for STL", selected_locations, key="stl_loc_select")
    stl_pollutant = col_stl_poll.selectbox("Select Pollutant for STL", selected_pollutants, key="stl_poll_select")

    decomp_data = filtered_df[(filtered_df['location'] == stl_location)][['sampling_date', stl_pollutant]].dropna()
    if not decomp_data.empty:
        decomp_series = decomp_data.set_index("sampling_date")[stl_pollutant].resample("ME").mean().dropna()

        if len(decomp_series) > 24: # STL requires at least two full cycles (e.g., 2 years for monthly data)
            try:
                stl = STL(decomp_series, seasonal=13, period=12, robust=True)
                res = stl.fit()

                fig, ax = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
                res.observed.plot(ax=ax[0], title="Observed")
                res.trend.plot(ax=ax[1], title="Trend")
                res.seasonal.plot(ax=ax[2], title="Seasonal")
                res.resid.plot(ax=ax[3], title="Residual")
                fig.suptitle(f'STL Decomposition of {stl_pollutant.upper()} in {stl_location} (Monthly)', fontsize=16, y=1.02)
                plt.tight_layout(rect=[0, 0.03, 1, 0.98])
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.warning(f"Could not perform STL decomposition for {stl_pollutant.upper()} in {stl_location}: {e}")
                st.info("STL decomposition requires sufficient, regular data points. Try a different location/pollutant or a longer date range.")
        else:
            st.info(f"Not enough data points for meaningful STL decomposition for {stl_pollutant.upper()} in {stl_location} (need at least 2 years of monthly data).")
    else:
        st.info(f"No data for {stl_pollutant.upper()} in {stl_location} for STL decomposition.")

    st.markdown("---")

    # --- ARIMA Forecasting ---
    st.subheader("🔮 ARIMA Forecasting")
    st.write("Forecast future pollutant levels using an ARIMA model.")
    col_arima_loc, col_arima_poll = st.columns(2)
    arima_location = col_arima_loc.selectbox("Select Location for ARIMA", selected_locations, key="arima_loc_select")
    arima_pollutant = col_arima_poll.selectbox("Select Pollutant for ARIMA", selected_pollutants, key="arima_poll_select")
    forecast_steps = st.slider("Forecast Months", 6, 36, step=6, value=12)

    arima_data = filtered_df[(filtered_df['location'] == arima_location)][['sampling_date', arima_pollutant]].dropna()
    if not arima_data.empty:
        arima_series = arima_data.set_index("sampling_date")[arima_pollutant].resample("ME").mean().dropna()

        if len(arima_series) > 24: # ARIMA typically needs a reasonable amount of data
            try:
                # Using a simple (1,1,1) order, this can be optimized
                model = ARIMA(arima_series, order=(1,1,1))
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=forecast_steps)
                future_dates = pd.date_range(start=arima_series.index[-1] + pd.offsets.MonthBegin(1), periods=forecast_steps, freq='ME')
                forecast_df = pd.DataFrame({"Forecast": forecast.values}, index=future_dates)

                fig, ax = plt.subplots(figsize=(10, 4))
                arima_series.plot(ax=ax, label="Historical")
                forecast_df.plot(ax=ax, label="Forecast", style="--")
                ax.set_title(f"{arima_pollutant.upper()} Forecast for {arima_location}")
                ax.set_xlabel("Date")
                ax.set_ylabel("Concentration")
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.7)
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.warning(f"Could not perform ARIMA forecasting for {arima_pollutant.upper()} in {arima_location}: {e}")
                st.info("ARIMA models require stationary data and sufficient history. Try a different location/pollutant or a longer date range.")
        else:
            st.info(f"Not enough data points for meaningful ARIMA forecasting for {arima_pollutant.upper()} in {arima_location} (need more historical data).")
    else:
        st.info(f"No data for {arima_pollutant.upper()} in {arima_location} for ARIMA forecasting.")

    st.markdown("---")

    # --- Clustering with Time Series (KMeans) ---
    st.subheader("📊 Time Series Clustering")
    st.write("Cluster locations/states based on their pollutant trends using KMeans.")
    cluster_by_option = st.radio("Cluster by:", ("State", "Location"))
    cluster_pollutant = st.selectbox("Select Pollutant for Clustering", selected_pollutants, key="cluster_poll_select")
    num_clusters = st.slider("Number of Clusters (K)", min_value=2, max_value=min(10, len(selected_locations) if cluster_by_option == "Location" else len(selected_states)), value=3)

    if cluster_by_option == "State":
        group_cols = ['state', 'pollutant']
        id_col = 'state'
    else: # Location
        group_cols = ['location', 'pollutant']
        id_col = 'location'

    ts_cluster_data = filtered_melted_df.groupby(group_cols + [pd.Grouper(key='date', freq='ME')])['value'].mean().unstack(level='date')
    ts_cluster_data.fillna(0, inplace=True)

    # Filter for the selected pollutant for clustering
    if cluster_pollutant in ts_cluster_data.index.get_level_values('pollutant'):
        ts_cluster_filtered = ts_cluster_data.loc[(slice(None), cluster_pollutant), :].droplevel('pollutant')
    else:
        st.info(f"No data for {cluster_pollutant.upper()} to cluster by {cluster_by_option}.")
        ts_cluster_filtered = pd.DataFrame() # Empty DataFrame to skip clustering

    if not ts_cluster_filtered.empty and ts_cluster_filtered.shape[0] > 1 and num_clusters >= 1:
        if ts_cluster_filtered.shape[0] < num_clusters:
            st.warning(f"Number of clusters ({num_clusters}) is greater than the number of unique {cluster_by_option.lower()}s ({ts_cluster_filtered.shape[0]}) with data for {cluster_pollutant.upper()}. Adjusting K to {ts_cluster_filtered.shape[0]}.")
            num_clusters = ts_cluster_filtered.shape[0]
            if num_clusters < 2: # Need at least 2 for meaningful clustering
                st.info(f"Not enough {cluster_by_option.lower()}s with data for {cluster_pollutant.upper()} to perform clustering.")
                ts_cluster_filtered = pd.DataFrame() # Set empty to skip
        
        if not ts_cluster_filtered.empty and num_clusters >= 2: # Re-check after adjustment
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(ts_cluster_filtered)

            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            ts_cluster_filtered['cluster'] = kmeans.fit_predict(scaled_features)

            # Prepare data for plotting cluster trends
            cluster_trends_for_plot = ts_cluster_filtered.groupby('cluster').mean().reset_index()
            # Melt the date columns into a single 'date' column
            date_cols = [col for col in cluster_trends_for_plot.columns if isinstance(col, pd.Timestamp)]
            cluster_trends_melted = cluster_trends_for_plot.melt(
                id_vars=['cluster'],
                value_vars=date_cols,
                var_name='date',
                value_name='average_value'
            )

            fig_cluster = px.line(
                cluster_trends_melted,
                x='date',
                y='average_value',
                color='cluster',
                title=f'Average {cluster_pollutant.upper()} Trends for Each Cluster by {cluster_by_option} (Monthly Averages)',
                labels={'average_value': f'Average {cluster_pollutant.upper()} Value', 'date': 'Date'}
            )
            st.plotly_chart(fig_cluster, use_container_width=True)

            # Display which locations/states belong to which cluster
            st.subheader(f"Cluster Assignments for {cluster_pollutant.upper()}")
            cluster_assignments = ts_cluster_filtered['cluster'].reset_index()
            cluster_assignments.columns = [id_col, 'cluster']
            st.dataframe(cluster_assignments)
        else:
            st.info(f"Not enough {cluster_by_option.lower()}s with data for {cluster_pollutant.upper()} to perform clustering with selected K.")
    else:
        st.info(f"Insufficient data for clustering {cluster_by_option.lower()}s based on {cluster_pollutant.upper()} trends.")

    st.markdown("---")

    # --- Download Filtered Data Button ---
    st.subheader("📥 Download Filtered Data")
    st.write("Download the currently filtered dataset as a CSV file.")
    csv = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name="filtered_cpcb_data.csv",
        mime="text/csv",
    )

