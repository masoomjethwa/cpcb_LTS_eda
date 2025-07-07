#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit Dashboard for CPCB Air Quality Data (1990–2015)

This script creates an interactive web dashboard using Streamlit to analyze
and visualize CPCB air quality data. It includes filtering, KPI metrics,
various time series plots, geographical animations, anomaly detection,
seasonal decomposition, ARIMA forecasting, and time series clustering,
along with a comprehensive set of advanced EDA visualizations.

To run this app:
1. Make sure you have the 'CPCB_data_from1990_2015.csv' file in a directory
   accessible by the script (update DATA_FILE_PATH if needed).
2. Install necessary libraries:
   pip install streamlit pandas numpy plotly seaborn matplotlib statsmodels scikit-learn wordcloud networkx joypy
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
import io # For handling plot downloads
import networkx as nx # Import networkx for Network Graph
import joypy # Import joypy for Ridgeline plots
from scipy.cluster.hierarchy import linkage, dendrogram # Import for Dendrogram

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

    # --- Diagnostic Output: Raw Data Head ---
    st.sidebar.subheader("Raw Data Head (Diagnostic)")
    st.sidebar.dataframe(df.head())

    # --- Diagnostic Output: Raw Data Info ---
    st.sidebar.subheader("Raw Data Info (Diagnostic)")
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.sidebar.text(buffer.getvalue())

    # --- Diagnostic Output: Raw Data Describe ---
    st.sidebar.subheader("Raw Data Describe (Diagnostic)")
    st.sidebar.dataframe(df.describe())


    # Convert 'sampling_date' column to datetime, handling errors
    # Using dayfirst=True as is common for Indian datasets, errors='coerce' turns unparseable into NaT
    if "sampling_date" in df.columns:
        df["sampling_date"] = pd.to_datetime(df["sampling_date"], errors='coerce', dayfirst=True)
    else:
        st.warning("Column 'sampling_date' not found in the dataset. Please check your CSV file.")

    # Convert 'date' column to datetime if it exists, handling errors
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors='coerce', dayfirst=True)
    else:
        # If 'date' column doesn't exist, create it from 'sampling_date' for consistency in melted_df
        if "sampling_date" in df.columns:
            df["date"] = df["sampling_date"]
        else:
            st.error("Neither 'date' nor 'sampling_date' column found. Cannot proceed with date-based analysis.")
            st.stop()


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

# --- Diagnostic Output: Filtered Data Status ---
st.subheader("Filtered Data Status (Diagnostic)")
st.write(f"Filtered DataFrame shape: {filtered_df.shape}")
st.dataframe(filtered_df.head())


# --- Main Dashboard Content ---
st.title("🧪 CPCB Air Quality Dashboard (1990–2015)")

if filtered_df.empty or filtered_melted_df.empty:
    st.warning("No data available for the selected filters. Please adjust your selections.")
else:
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview & KPIs",
        "📈 Time Series Analysis",
        "🎨 Advanced EDA Plots",
        "📊 Time Series Clustering",
        "📥 Data & Plot Downloads"
    ])

    with tab1:
        # --- 10. KPI Metrics Box ---
        st.header("📊 Key Performance Indicators")
        kpi_cols = st.columns(len(selected_pollutants) * 2) # Two metrics per pollutant (mean, max)
        col_idx = 0
        for pollutant in selected_pollutants:
            values = filtered_df[pollutant].dropna()
            if not values.empty:
                kpi_cols[col_idx].metric(f"{pollutant.upper()} - Mean", f"{values.mean():.2f}")
                if col_idx + 1 < len(kpi_cols):
                    kpi_cols[col_idx + 1].metric(f"{pollutant.upper()} - Max", f"{values.max():.2f}")
                col_idx += 2
            else:
                st.info(f"No data for {pollutant.upper()} in selected range for KPI metrics.")

        st.markdown("---")

        # --- 4. Monthly Heatmap View ---
        st.header("🗓️ Monthly Heatmap View")
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
        st.header("🏆 Top Cities by Mean Pollution")
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

    with tab2:
        st.header("📈 Time Series Analysis")

        # --- 2. Time Series Panel (Line Plot + Rolling Average Toggle) ---
        st.subheader("Pollutant Trends Over Time")
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
        if len(selected_locations) > 6:
            st.info("Displaying trends for the first 6 selected locations for clarity in this view.")
            compare_locations = selected_locations[:6]
        else:
            compare_locations = selected_locations

        compare_df = filtered_melted_df[
            filtered_melted_df['location'].isin(compare_locations)
        ].dropna(subset=['date', 'value', 'location', 'pollutant'])

        if not compare_df.empty:
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
            fig_facet.update_yaxes(matches=None)
            fig_facet.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
            st.plotly_chart(fig_facet, use_container_width=True)
        else:
            st.info("No data to compare across selected locations and pollutants.")

        st.markdown("---")

        # --- 7. Correlation Over Time Plot ---
        st.subheader("🤝 Rolling Correlation Between Pollutants")
        st.write("Analyze how the relationship between two pollutants changes over time.")
        col_corr1, col_corr2 = st.columns(2)
        corr_pollutant1 = col_corr1.selectbox("Select Pollutant 1", POLLUTANT_COLUMNS, index=0)
        corr_pollutant2 = col_corr2.selectbox("Select Pollutant 2", POLLUTANT_COLUMNS, index=1)

        if corr_pollutant1 == corr_pollutant2:
            st.warning("Please select two different pollutants for correlation analysis.")
        else:
            # Ensure the pollutants are numerical and drop NaNs for these specific columns
            # Also, keep 'location' to filter by it later.
            corr_data = filtered_df[['sampling_date', 'location', corr_pollutant1, corr_pollutant2]].dropna(subset=[corr_pollutant1, corr_pollutant2])
            if not corr_data.empty:
                corr_data = corr_data.set_index('sampling_date')
                rolling_correlations = []
                for loc in corr_data['location'].unique():
                    # Filter by location, then explicitly select *only* the numeric pollutant columns
                    # before resampling and calculating the mean.
                    loc_df_numeric = corr_data[corr_data['location'] == loc][[corr_pollutant1, corr_pollutant2]].resample('ME').mean().dropna()

                    if len(loc_df_numeric) > 30: # Need enough data points for rolling correlation
                        loc_df_numeric['rolling_corr'] = loc_df_numeric[corr_pollutant1].rolling(window=30, min_periods=10).corr(loc_df_numeric[corr_pollutant2])
                        loc_df_numeric['location'] = loc # Add location back for plotting
                        rolling_correlations.append(loc_df_numeric.reset_index())

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
                    st.info("Not enough data points for rolling correlation for selected pollutants/locations after resampling.")
            else:
                st.info("No data available for selected pollutants for correlation analysis after dropping NaNs.")

        st.markdown("---")

        # --- 8. Interactive Time Series (Plotly) ---
        st.subheader("🔍 Interactive Pollutant Trends")
        st.write("Explore detailed pollutant trends with zoom and range selection capabilities.")
        interactive_pollutant = st.selectbox("Select Pollutant for Interactive Trend", selected_pollutants, key="interactive_poll")

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
        spike_location = col_spike_loc.selectbox("Select Location for Spike Detection", selected_locations, key="spike_loc")
        spike_pollutant = col_spike_poll.selectbox("Select Pollutant for Spike Detection", selected_pollutants, key="spike_poll")
        spike_threshold = st.slider("Z-score Threshold for Spikes", min_value=1.0, max_value=5.0, value=3.0, step=0.5, key="spike_thresh")

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
        forecast_steps = st.slider("Forecast Months", 6, 36, step=6, value=12, key="forecast_steps")

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

    with tab3:
        st.header("🎨 Advanced EDA Plots")

        # --- Univariate Visualizations (Histograms & Boxplots) ---
        st.subheader("Univariate Pollutant Distributions")
        for pollutant in selected_pollutants:
            with st.expander(f"Distribution of {pollutant.upper()}"):
                fig, ax = plt.subplots(1, 2, figsize=(12, 5))
                sns.histplot(filtered_df[pollutant].dropna(), kde=True, bins=50, color='skyblue', edgecolor='black', ax=ax[0])
                ax[0].set_title(f"Distribution of {pollutant.upper()}")
                ax[0].set_xlabel(pollutant)
                ax[0].set_ylabel("Frequency")
                ax[0].grid(True, linestyle='--', alpha=0.6)

                sns.boxplot(x=filtered_df[pollutant].dropna(), color='lightcoral', ax=ax[1])
                ax[1].set_title(f"Boxplot of {pollutant.upper()}")
                ax[1].set_xlabel(pollutant)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        st.markdown("---")

        # --- Bivariate Analysis: Correlation Heatmap ---
        st.subheader("Bivariate Analysis: Correlation Heatmap")
        correlation_matrix = filtered_df[selected_pollutants].corr()
        if not correlation_matrix.empty:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5, fmt=".2f", ax=ax)
            ax.set_title("Correlation Heatmap of Selected Pollutants")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Not enough data to compute correlation heatmap for selected pollutants.")

        st.markdown("---")

        # --- Sankey Diagram ---
        st.subheader("Sankey Diagram: State to Pollutant Flow")
        sankey_data = filtered_melted_df.groupby(["state", "pollutant"]).size().reset_index(name='count')
        if not sankey_data.empty:
            labels = list(sankey_data["state"].unique()) + list(sankey_data["pollutant"].unique())
            state_map = {k: i for i, k in enumerate(sankey_data["state"].unique())}
            pollutant_map = {k: i + len(sankey_data["state"].unique()) for i, k in enumerate(sankey_data["pollutant"].unique())}

            fig_sankey = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15, thickness=20, line=dict(color="black", width=0.5), label=labels, color="blue"
                ),
                link=dict(
                    source=sankey_data["state"].map(state_map),
                    target=sankey_data["pollutant"].map(pollutant_map),
                    value=sankey_data["count"]
                )
            )])
            fig_sankey.update_layout(title_text="Sankey Diagram: State to Pollutant Flow", font_size=10)
            st.plotly_chart(fig_sankey, use_container_width=True)
        else:
            st.info("No data for Sankey Diagram with current filters.")

        st.markdown("---")

        # --- Network Graph ---
        st.subheader("Network Graph: Pollutant-State Relationships")
        G = nx.Graph()
        edges = filtered_melted_df.groupby(["state", "pollutant"]).size().reset_index(name="weight")
        if not edges.empty:
            for _, row in edges.iterrows():
                G.add_edge(row["state"], row["pollutant"], weight=row["weight"])

            fig, ax = plt.subplots(figsize=(12, 10))
            pos = nx.spring_layout(G, k=0.3, iterations=50)
            node_sizes = [G.degree(node) * 100 for node in G.nodes()]
            edge_widths = [d['weight'] / edges['weight'].max() * 5 for (u, v, d) in G.edges(data=True)]

            nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=node_sizes, alpha=0.9, ax=ax)
            nx.draw_networkx_edges(G, pos, edge_color='gray', width=edge_widths, alpha=0.6, ax=ax)
            nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', ax=ax)
            ax.set_title("Pollutant-State Network Graph", fontsize=16)
            ax.axis("off")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No data for Network Graph with current filters.")

        st.markdown("---")

        # --- Heatmap with Dendrogram ---
        st.subheader("Heatmap with Dendrogram: State Clustering by Pollutant Levels")
        # Pivot table to get states as index, pollutants as columns, and mean value as data
        pivot_table_dendro = filtered_melted_df.pivot_table(values="value", index="state", columns="pollutant", aggfunc="mean")
        pivot_table_dendro.fillna(0, inplace=True)

        if not pivot_table_dendro.empty and pivot_table_dendro.shape[0] > 1:
            try:
                linkage_matrix = linkage(pivot_table_dendro, method='ward')
                fig, ax = plt.subplots(figsize=(14, 8))
                dendrogram(linkage_matrix, labels=pivot_table_dendro.index, orientation='left', leaf_font_size=8, ax=ax)
                ax.set_title("Dendrogram of States by Pollutant Levels (Hierarchical Clustering)", fontsize=16)
                ax.set_xlabel("Distance")
                ax.set_ylabel("State")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.info(f"Could not generate Dendrogram: {e}. Ensure enough states and data points for clustering.")
        else:
            st.info("Not enough data or states to generate Dendrogram with current filters.")

        st.markdown("---")

        # --- Ridgeline Plot (Joyplot) ---
        st.subheader("Ridgeline Plot: Pollutant Value Distributions by State")
        ridgeline_data = filtered_melted_df.dropna(subset=['state', 'value']).copy()
        if not ridgeline_data.empty and len(ridgeline_data['state'].unique()) > 1:
            try:
                import joypy # Import joypy here to ensure it's available when needed
                fig_ridgeline, axes = joypy.joyplot(
                    ridgeline_data,
                    by="state",
                    column="value",
                    figsize=(12, 8),
                    xlabels=True,
                    ylabels=True,
                    overlap=1,
                    grid=True,
                    title="Ridgeline Plot of Pollutant Values by State",
                    linewidth=1,
                    alpha=0.7
                )
                plt.xlabel("Pollutant Value")
                plt.ylabel("State Density")
                plt.tight_layout()
                st.pyplot(fig_ridgeline)
                plt.close(fig_ridgeline)
            except Exception as e:
                st.info(f"Could not generate Ridgeline Plot: {e}. Ensure enough states and data for distribution plots.")
        else:
            st.info("Not enough data or states to generate Ridgeline Plot with current filters.")

        st.markdown("---")

        # --- Sunburst Chart ---
        st.subheader("Sunburst Chart: Pollutant Values by State, Location, and Pollutant Type")
        sunburst_data = filtered_melted_df.dropna(subset=['state', 'location', 'pollutant']).copy()
        if not sunburst_data.empty:
            fig_sunburst = px.sunburst(
                sunburst_data,
                path=['state', 'location', 'pollutant'],
                values='value',
                title="Sunburst Chart: Pollutant Values by State, Location, and Pollutant Type"
            )
            st.plotly_chart(fig_sunburst, use_container_width=True)
        else:
            st.info("No data for Sunburst Chart with current filters.")

        st.markdown("---")

        # --- Parallel Coordinates Plot ---
        st.subheader("Parallel Coordinates Plot: Pollutant Value, Year, and Month")
        sample_df_pc = filtered_melted_df[['value', 'year', 'month']].dropna()
        if not sample_df_pc.empty:
            sample_pc = sample_df_pc.sample(min(2000, len(sample_df_pc)), random_state=42)
            fig_pc = px.parallel_coordinates(
                sample_pc,
                dimensions=['value', 'year', 'month'],
                title="Parallel Coordinates Plot: Pollutant Value, Year, and Month"
            )
            st.plotly_chart(fig_pc, use_container_width=True)
        else:
            st.info("No data for Parallel Coordinates Plot with current filters.")

        st.markdown("---")

        # --- Treemap ---
        st.subheader("Treemap: Pollutant Values by State and Pollutant Type")
        treemap_data = filtered_melted_df.dropna(subset=['state', 'pollutant']).copy()
        if not treemap_data.empty:
            fig_treemap = px.treemap(
                treemap_data,
                path=['state', 'pollutant'],
                values='value',
                title="Treemap: Pollutant Values by State and Pollutant Type"
            )
            st.plotly_chart(fig_treemap, use_container_width=True)
        else:
            st.info("No data for Treemap with current filters.")

        st.markdown("---")

        # --- Gantt Chart (Simplified) ---
        st.subheader("Gantt Chart: Sample Monitoring Periods by Location")
        gantt_data = filtered_df[['location', 'date']].drop_duplicates().sort_values(by='date')
        gantt_data.dropna(subset=['location', 'date'], inplace=True)
        gantt_data = gantt_data.head(min(100, len(gantt_data))) # Limit for readability
        if not gantt_data.empty:
            gantt_data["End"] = gantt_data["date"] + pd.Timedelta(days=10) # Simulate duration
            fig_gantt = px.timeline(
                gantt_data,
                x_start="date",
                x_end="End",
                y="location",
                title="Gantt Chart: Sample Monitoring Periods by Location"
            )
            fig_gantt.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_gantt, use_container_width=True)
        else:
            st.info("No data for Gantt Chart with current filters.")

        st.markdown("---")

        # --- Trellis Plot / Faceted Plot ---
        st.subheader("Trellis Plot: Distribution of Pollutant Values by State")
        trellis_data = filtered_melted_df.dropna(subset=['state', 'value']).copy()
        if not trellis_data.empty and len(trellis_data['state'].unique()) > 1:
            try:
                g_trellis = sns.FacetGrid(trellis_data, col="state", col_wrap=3, height=4, aspect=1.2, sharex=True, sharey=False)
                g_trellis.map(sns.histplot, "value", bins=30, kde=True, color='teal', edgecolor='black')
                g_trellis.set_titles("State: {col_name}")
                g_trellis.set_axis_labels("Pollutant Value", "Frequency")
                plt.suptitle("Trellis Plot: Distribution of Pollutant Values by State", y=1.02, fontsize=16)
                plt.tight_layout(rect=[0, 0.03, 1, 0.98])
                st.pyplot(g_trellis.fig)
                plt.close(g_trellis.fig)
            except Exception as e:
                st.info(f"Could not generate Trellis Plot: {e}. Ensure enough states and data points.")
        else:
            st.info("Not enough data or states to generate Trellis Plot with current filters.")

        st.markdown("---")

        # --- Violin Plot with Swarm Plot Overlay ---
        st.subheader("Violin + Swarm Plot: Pollutant Values by State")
        violin_swarm_data = filtered_melted_df.dropna(subset=['state', 'value']).copy()
        if not violin_swarm_data.empty and len(violin_swarm_data['state'].unique()) > 1:
            fig, ax = plt.subplots(figsize=(14, 8))
            sns.violinplot(x="state", y="value", data=violin_swarm_data, inner=None, color="lightgray", linewidth=1.5, ax=ax)
            sample_swarm = violin_swarm_data.sample(min(5000, len(violin_swarm_data)), random_state=42)
            sns.swarmplot(x="state", y="value", data=sample_swarm, size=3, color='darkblue', alpha=0.6, ax=ax)
            ax.set_title("Violin + Swarm Plot: Pollutant Values by State", fontsize=16)
            ax.set_xlabel("State")
            ax.set_ylabel("Pollutant Value")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Not enough data or states to generate Violin + Swarm Plot with current filters.")

        st.markdown("---")

        # --- Hexbin Plot ---
        st.subheader("Hexbin Plot: Pollutant Value Density by Month")
        df_hex = filtered_melted_df[['value', 'month']].dropna()
        if not df_hex.empty:
            fig, ax = plt.subplots(figsize=(10, 7))
            hb = ax.hexbin(df_hex['month'], df_hex['value'], gridsize=30, cmap='Blues', edgecolors='none')
            ax.set_xlabel('Month')
            ax.set_ylabel('Pollutant Value')
            ax.set_title("Hexbin Plot: Pollutant Value Density by Month")
            fig.colorbar(hb, ax=ax, label='Count in Bin')
            ax.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No data for Hexbin Plot with current filters.")

        st.markdown("---")

        # --- Streamgraph (Simulated as Stacked Area Chart) ---
        st.subheader("Streamgraph (Simulated): Mean Pollutant Concentration Over Years")
        stream_data = filtered_melted_df.groupby(["year", "pollutant"])["value"].mean().unstack(fill_value=0)
        if not stream_data.empty:
            fig, ax = plt.subplots(figsize=(14, 7))
            stream_data.plot.area(
                ax=ax,
                alpha=0.7,
                linewidth=0
            )
            ax.set_title("Streamgraph (Simulated): Mean Pollutant Concentration Over Years")
            ax.set_xlabel("Year")
            ax.set_ylabel("Mean Concentration")
            ax.legend(title="Pollutant", bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No data for Streamgraph with current filters.")

        st.markdown("---")

        # --- Geographic Heatmap (Simulated) ---
        st.subheader("Geographic Heatmap: Pollutant Value Density (Simulated Locations)")
        df_geo = filtered_melted_df.copy()
        @st.cache_data
        def get_simulated_lat_lon_for_eda(locations):
            lat_map = {loc: np.random.uniform(9.0, 34.0) for loc in locations}
            lon_map = {loc: np.random.uniform(69.0, 96.0) for loc in locations}
            return lat_map, lon_map

        unique_locations_in_filtered = df_geo['location'].unique()
        location_lat_map_eda, location_lon_map_eda = get_simulated_lat_lon_for_eda(unique_locations_in_filtered)

        df_geo["lat"] = df_geo["location"].map(location_lat_map_eda)
        df_geo["lon"] = df_geo["location"].map(location_lon_map_eda)
        df_geo.dropna(subset=['lat', 'lon', 'value'], inplace=True)

        if not df_geo.empty:
            geo_pollutant = st.selectbox("Select Pollutant for Geographic Heatmap", selected_pollutants, key="geo_poll")
            df_geo_filtered = df_geo[df_geo['pollutant'] == geo_pollutant]
            if not df_geo_filtered.empty:
                fig_geo_heatmap = px.density_mapbox(
                    df_geo_filtered,
                    lat="lat",
                    lon="lon",
                    z="value",
                    radius=10,
                    center=dict(lat=22.5, lon=80),
                    zoom=3,
                    mapbox_style="stamen-terrain",
                    title=f"Geographic Heatmap: {geo_pollutant.upper()} Density (Simulated Locations)"
                )
                st.plotly_chart(fig_geo_heatmap, use_container_width=True)
            else:
                st.info(f"No data for {geo_pollutant.upper()} for Geographic Heatmap.")
        else:
            st.info("No data for Geographic Heatmap with current filters.")

        st.markdown("---")

        # --- Word Cloud ---
        st.subheader("Word Cloud of Locations")
        from wordcloud import WordCloud # Re-import locally for clarity if not at top
        text_data_wc = " ".join(filtered_df["location"].astype(str).dropna().tolist())
        if text_data_wc:
            wordcloud = WordCloud(
                width=1000, height=500,
                background_color="white",
                collocations=True,
                min_font_size=10,
                colormap='viridis'
            ).generate(text_data_wc)

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            ax.set_title("Word Cloud of Locations (with Collocations)", fontsize=16)
            plt.tight_layout(pad=0)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No location data to generate Word Cloud with current filters.")

    with tab4:
        st.header("📊 Time Series Clustering")

        # --- Clustering with Time Series (KMeans) ---
        st.subheader("Time Series Clustering")
        st.write("Cluster locations/states based on their pollutant trends using KMeans.")
        cluster_by_option = st.radio("Cluster by:", ("State", "Location"), key="cluster_by_radio")
        cluster_pollutant = st.selectbox("Select Pollutant for Clustering", selected_pollutants, key="cluster_poll_select")

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

        # Calculate max_clusters based on available data
        max_clusters_available = ts_cluster_filtered.shape[0] if not ts_cluster_filtered.empty else 0
        num_clusters = st.slider(
            "Number of Clusters (K)",
            min_value=2,
            max_value=min(10, max_clusters_available) if max_clusters_available >= 2 else 2,
            value=min(3, max_clusters_available) if max_clusters_available >= 2 else 2,
            key="num_clusters_slider"
        )

        if not ts_cluster_filtered.empty and ts_cluster_filtered.shape[0] > 1 and num_clusters >= 2:
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(ts_cluster_filtered)

            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            ts_cluster_filtered['cluster'] = kmeans.fit_predict(scaled_features)

            # Prepare data for plotting cluster trends
            cluster_trends_for_plot = ts_cluster_filtered.groupby('cluster').mean().reset_index()
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
            if max_clusters_available < 2:
                st.info(f"Not enough unique {cluster_by_option.lower()}s with data for {cluster_pollutant.upper()} to perform clustering (need at least 2).")
            else:
                st.info(f"Insufficient data for clustering {cluster_by_option.lower()}s based on {cluster_pollutant.upper()} trends with selected K.")


    with tab5:
        st.header("📥 Data & Plot Downloads")
        st.write("Download the currently filtered dataset or individual plots.")

        # --- Download Filtered Data Button ---
        st.subheader("Download Filtered Data")
        st.write("Download the currently filtered dataset as a CSV file.")
        csv_data = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Filtered Data as CSV",
            data=csv_data,
            file_name="filtered_cpcb_data.csv",
            mime="text/csv",
        )

        st.markdown("---")

        st.subheader("Download Individual Plots")
        st.write("Select a plot type to download its most recently generated version.")

        plot_types = [
            "Monthly Heatmap", "Top Cities Bar Chart", "Animated Map",
            "Rolling Correlation Plot", "Interactive Time Series",
            "Pollution Spike Detector", "STL Decomposition", "ARIMA Forecast",
            "Univariate Distributions", "Correlation Heatmap", "Sankey Diagram",
            "Network Graph", "Dendrogram", "Ridgeline Plot", "Sunburst Chart",
            "Parallel Coordinates Plot", "Treemap", "Gantt Chart", "Trellis Plot",
            "Violin + Swarm Plot", "Hexbin Plot", "Streamgraph", "Geographic Heatmap",
            "Word Cloud", "Time Series Clustering"
        ]
        selected_plot_to_download = st.selectbox("Choose a plot to download:", plot_types)

        # Logic to regenerate and download the selected plot
        if st.button(f"Download {selected_plot_to_download}"):
            if selected_plot_to_download == "Monthly Heatmap":
                heatmap_data_download = filtered_df.copy()
                heatmap_data_download["Year"] = heatmap_data_download["sampling_date"].dt.year
                heatmap_data_download["Month"] = heatmap_data_download["sampling_date"].dt.month
                if not selected_pollutants:
                    st.warning("Please select at least one pollutant in the sidebar to download this plot.")
                else:
                    pollutant_for_download = selected_pollutants[0] # Just take the first one for download
                    pivot_table_download = heatmap_data_download.pivot_table(values=pollutant_for_download, index="Month", columns="Year", aggfunc="mean")
                    if not pivot_table_download.empty:
                        fig, ax = plt.subplots(figsize=(12, 5))
                        sns.heatmap(pivot_table_download, cmap="viridis", annot=False, fmt=".1f", linewidths=.5, ax=ax, cbar_kws={'label': f'Average {pollutant_for_download.upper()} Concentration'})
                        ax.set_title(f"Monthly Heatmap of {pollutant_for_download.upper()}")
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                        st.download_button(label="Download Plot", data=buf.getvalue(), file_name=f"monthly_heatmap_{pollutant_for_download}.png", mime="image/png")
                        plt.close(fig)
                    else:
                        st.info("No data to generate this plot for download.")

            elif selected_plot_to_download == "Top Cities Bar Chart":
                if not selected_pollutants:
                    st.warning("Please select at least one pollutant in the sidebar to download this plot.")
                else:
                    pollutant_for_download = selected_pollutants[0]
                    top_cities_download = filtered_df.groupby("location")[pollutant_for_download].mean().sort_values(ascending=False).head(10).reset_index()
                    if not top_cities_download.empty:
                        fig = px.bar(top_cities_download, x=pollutant_for_download, y="location", orientation="h", title=f"Top 10 Cities - Mean {pollutant_for_download.upper()}")
                        buf = io.BytesIO()
                        fig.write_html(buf, auto_open=False)
                        st.download_button(label="Download Plot", data=buf.getvalue(), file_name=f"top_cities_bar_chart_{pollutant_for_download}.html", mime="text/html")
                    else:
                        st.info("No data to generate this plot for download.")

            elif selected_plot_to_download == "Animated Map":
                animated_map_data_download = filtered_melted_df.groupby(['year', 'location', 'pollutant'])['value'].mean().reset_index()
                unique_locations_in_filtered_download = animated_map_data_download['location'].unique()
                # Ensure get_simulated_lat_lon is defined or imported for download logic
                @st.cache_data
                def get_simulated_lat_lon(locations):
                    lat_map = {loc: np.random.uniform(9.0, 34.0) for loc in locations}
                    lon_map = {loc: np.random.uniform(69.0, 96.0) for loc in locations}
                    return lat_map, lon_map

                location_lat_map_download, location_lon_map_download = get_simulated_lat_lon(unique_locations_in_filtered_download)
                animated_map_data_download["lat"] = animated_map_data_download["location"].map(location_lat_map_download)
                animated_map_data_download["lon"] = animated_map_data_download["location"].map(location_lon_map_download)
                animated_map_data_download.dropna(subset=['lat', 'lon', 'value'], inplace=True)

                if not selected_pollutants:
                    st.warning("Please select at least one pollutant in the sidebar to download this plot.")
                else:
                    pollutant_for_download = selected_pollutants[0]
                    animated_map_filtered_download = animated_map_data_download[animated_map_data_download['pollutant'] == pollutant_for_download]

                    if not animated_map_filtered_download.empty:
                        fig = px.scatter_mapbox(animated_map_filtered_download, lat="lat", lon="lon", color="value", size="value",
                                                animation_frame="year", animation_group="location", hover_name="location",
                                                mapbox_style="carto-positron", zoom=3, center={"lat": 22.5, "lon": 80},
                                                title=f"Animated {pollutant_for_download.upper()} Concentration Over Years (Simulated)")
                        buf = io.BytesIO()
                        fig.write_html(buf, auto_open=False)
                        st.download_button(label="Download Plot", data=buf.getvalue(), file_name=f"animated_map_{pollutant_for_download}.html", mime="text/html")
                    else:
                        st.info("No data to generate this plot for download.")

            elif selected_plot_to_download == "Rolling Correlation Plot":
                if len(selected_pollutants) < 2:
                    st.warning("Please select at least two pollutants in the sidebar to download this plot.")
                else:
                    corr_pollutant1_download = selected_pollutants[0]
                    corr_pollutant2_download = selected_pollutants[1]
                    corr_data_download = filtered_df[['sampling_date', 'location', corr_pollutant1_download, corr_pollutant2_download]].dropna(subset=[corr_pollutant1_download, corr_pollutant2_download])
                    if not corr_data_download.empty:
                        corr_data_download = corr_data_download.set_index('sampling_date')
                        rolling_correlations_download = []
                        for loc in corr_data_download['location'].unique():
                            # Explicitly select only numeric columns for resampling and mean
                            loc_df_numeric_download = corr_data_download[corr_data_download['location'] == loc][[corr_pollutant1_download, corr_pollutant2_download]].resample('ME').mean().dropna()
                            if len(loc_df_numeric_download) > 30:
                                loc_df_numeric_download['rolling_corr'] = loc_df_numeric_download[corr_pollutant1_download].rolling(window=30, min_periods=10).corr(loc_df_numeric_download[corr_pollutant2_download])
                                loc_df_numeric_download['location'] = loc
                                rolling_correlations_download.append(loc_df_numeric_download.reset_index())
                        if rolling_correlations_download:
                            rolling_corr_df_download = pd.concat(rolling_correlations_download)
                            fig = px.line(rolling_corr_df_download.dropna(subset=['rolling_corr']), x='sampling_date', y='rolling_corr', color='location',
                                            title=f'Rolling Correlation (30-Day) between {corr_pollutant1_download.upper()} and {corr_pollutant2_download.upper()}')
                            buf = io.BytesIO()
                            fig.write_html(buf, auto_open=False)
                            st.download_button(label="Download Plot", data=buf.getvalue(), file_name=f"rolling_correlation_{corr_pollutant1_download}_{corr_pollutant2_download}.html", mime="text/html")
                        else:
                            st.info("Not enough data to generate this plot for download.")
                    else:
                        st.info("No data to generate this plot for download.")

            elif selected_plot_to_download == "Interactive Time Series":
                if not selected_pollutants:
                    st.warning("Please select at least one pollutant in the sidebar to download this plot.")
                else:
                    pollutant_for_download = selected_pollutants[0]
                    interactive_data_download = filtered_melted_df[(filtered_melted_df['pollutant'] == pollutant_for_download)].dropna(subset=['date', 'value'])
                    if not interactive_data_download.empty:
                        fig = px.line(interactive_data_download, x='date', y='value', color='location',
                                        title=f'Interactive {pollutant_for_download.upper()} Concentration Over Time by Location')
                        fig.update_xaxes(rangeslider_visible=True)
                        buf = io.BytesIO()
                        fig.write_html(buf, auto_open=False)
                        st.download_button(label="Download Plot", data=buf.getvalue(), file_name=f"interactive_time_series_{pollutant_for_download}.html", mime="text/html")
                    else:
                        st.info("No data to generate this plot for download.")

            elif selected_plot_to_download == "Pollution Spike Detector":
                if not selected_locations or not selected_pollutants:
                    st.warning("Please select at least one location and pollutant in the sidebar to download this plot.")
                else:
                    spike_location_download = selected_locations[0]
                    spike_pollutant_download = selected_pollutants[0]
                    spike_data_download = filtered_df[(filtered_df['location'] == spike_location_download)][['sampling_date', spike_pollutant_download]].dropna()
                    if not spike_data_download.empty:
                        location_series_download = spike_data_download.set_index('sampling_date')[spike_pollutant_download].sort_index()
                        mean_val = location_series_download.mean()
                        std_val = location_series_download.std()
                        if std_val > 0:
                            location_zscore_download = (location_series_download - mean_val) / std_val
                            spikes_download = location_series_download[location_zscore_download.abs() > 3.0] # Using default threshold for download
                            fig, ax = plt.subplots(figsize=(14, 7))
                            ax.plot(location_series_download.index, location_series_download.values, label=f'{spike_location_download} {spike_pollutant_download.upper()}', color='blue', alpha=0.7)
                            ax.scatter(spikes_download.index, spikes_download.values, color='red', s=50, zorder=5, label='Spikes (Z > 3.0)')
                            ax.set_title(f'{spike_pollutant_download.upper()} Pollution Spikes for {spike_location_download}')
                            buf = io.BytesIO()
                            fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                            st.download_button(label="Download Plot", data=buf.getvalue(), file_name=f"spike_detector_{spike_location_download}_{spike_pollutant_download}.png", mime="image/png")
                            plt.close(fig)
                        else:
                            st.info("Standard deviation is zero. Cannot detect spikes for download.")
                    else:
                        st.info("No data to generate this plot for download.")

            elif selected_plot_to_download == "STL Decomposition":
                if not selected_locations or not selected_pollutants:
                    st.warning("Please select at least one location and pollutant in the sidebar to download this plot.")
                else:
                    stl_location_download = selected_locations[0]
                    stl_pollutant_download = selected_pollutants[0]
                    decomp_data_download = filtered_df[(filtered_df['location'] == stl_location_download)][['sampling_date', stl_pollutant_download]].dropna()
                    if not decomp_data_download.empty:
                        decomp_series_download = decomp_data_download.set_index("sampling_date")[stl_pollutant_download].resample("ME").mean().dropna()
                        if len(decomp_series_download) > 24:
                            try:
                                stl = STL(decomp_series_download, seasonal=13, period=12, robust=True)
                                res = stl.fit()
                                fig, ax = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
                                res.observed.plot(ax=ax[0], title="Observed")
                                res.trend.plot(ax=ax[1], title="Trend")
                                res.seasonal.plot(ax=ax[2], title="Seasonal")
                                res.resid.plot(ax=ax[3], title="Residual")
                                fig.suptitle(f'STL Decomposition of {stl_pollutant_download.upper()} in {stl_location_download} (Monthly)', fontsize=16, y=1.02)
                                plt.tight_layout(rect=[0, 0.03, 1, 0.98])
                                buf = io.BytesIO()
                                fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                                st.download_button(label="Download Plot", data=buf.getvalue(), file_name=f"stl_decomposition_{stl_location_download}_{stl_pollutant_download}.png", mime="image/png")
                                plt.close(fig)
                            except Exception as e:
                                st.info(f"Could not generate STL Decomposition plot for download: {e}")
                        else:
                            st.info("Not enough data for STL Decomposition for download.")
                    else:
                        st.info("No data to generate this plot for download.")

            elif selected_plot_to_download == "ARIMA Forecast":
                if not selected_locations or not selected_pollutants:
                    st.warning("Please select at least one location and pollutant in the sidebar to download this plot.")
                else:
                    arima_location_download = selected_locations[0]
                    arima_pollutant_download = selected_pollutants[0]
                    arima_data_download = filtered_df[(filtered_df['location'] == arima_location_download)][['sampling_date', arima_pollutant_download]].dropna()
                    if not arima_data_download.empty:
                        arima_series_download = arima_data_download.set_index("sampling_date")[arima_pollutant_download].resample("ME").mean().dropna()
                        if len(arima_series_download) > 24:
                            try:
                                model = ARIMA(arima_series_download, order=(1,1,1))
                                model_fit = model.fit()
                                forecast = model_fit.forecast(steps=12) # Default 12 months for download
                                future_dates = pd.date_range(start=arima_series_download.index[-1] + pd.offsets.MonthBegin(1), periods=12, freq='ME')
                                forecast_df = pd.DataFrame({"Forecast": forecast.values}, index=future_dates)
                                fig, ax = plt.subplots(figsize=(10, 4))
                                arima_series_download.plot(ax=ax, label="Historical")
                                forecast_df.plot(ax=ax, label="Forecast", style="--")
                                ax.set_title(f"{arima_pollutant_download.upper()} Forecast for {arima_location_download}")
                                buf = io.BytesIO()
                                fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                                st.download_button(label="Download Plot", data=buf.getvalue(), file_name=f"arima_forecast_{arima_location_download}_{arima_pollutant_download}.png", mime="image/png")
                                plt.close(fig)
                            except Exception as e:
                                st.info(f"Could not generate ARIMA Forecast plot for download: {e}")
                        else:
                            st.info("Not enough data for ARIMA Forecast for download.")
                    else:
                        st.info("No data to generate this plot for download.")

            elif selected_plot_to_download == "Univariate Distributions":
                if not selected_pollutants:
                    st.warning("Please select at least one pollutant in the sidebar to download this plot.")
                else:
                    pollutant_for_download = selected_pollutants[0]
                    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
                    sns.histplot(filtered_df[pollutant_for_download].dropna(), kde=True, bins=50, color='skyblue', edgecolor='black', ax=ax[0])
                    ax[0].set_title(f"Distribution of {pollutant_for_download.upper()}")
                    sns.boxplot(x=filtered_df[pollutant_for_download].dropna(), color='lightcoral', ax=ax[1])
                    ax[1].set_title(f"Boxplot of {pollutant_for_download.upper()}")
                    plt.tight_layout()
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                    st.download_button(label="Download Plot", data=buf.getvalue(), file_name=f"univariate_distribution_{pollutant_for_download}.png", mime="image/png")
                    plt.close(fig)

            elif selected_plot_to_download == "Correlation Heatmap":
                if len(selected_pollutants) < 2:
                    st.warning("Please select at least two pollutants in the sidebar to download this plot.")
                else:
                    correlation_matrix_download = filtered_df[selected_pollutants].corr()
                    if not correlation_matrix_download.empty:
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(correlation_matrix_download, annot=True, cmap="coolwarm", linewidths=0.5, fmt=".2f", ax=ax)
                        ax.set_title("Correlation Heatmap of Selected Pollutants")
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                        st.download_button(label="Download Plot", data=buf.getvalue(), file_name="correlation_heatmap.png", mime="image/png")
                        plt.close(fig)
                    else:
                        st.info("No data to generate this plot for download.")

            elif selected_plot_to_download == "Sankey Diagram":
                sankey_data_download = filtered_melted_df.groupby(["state", "pollutant"]).size().reset_index(name='count')
                if not sankey_data_download.empty:
                    labels = list(sankey_data_download["state"].unique()) + list(sankey_data_download["pollutant"].unique())
                    state_map = {k: i for i, k in enumerate(sankey_data_download["state"].unique())}
                    pollutant_map = {k: i + len(sankey_data_download["state"].unique()) for i, k in enumerate(sankey_data_download["pollutant"].unique())}
                    fig = go.Figure(data=[go.Sankey(node=dict(label=labels), link=dict(source=sankey_data_download["state"].map(state_map), target=sankey_data_download["pollutant"].map(pollutant_map), value=sankey_data_download["count"]))])
                    buf = io.BytesIO()
                    fig.write_html(buf, auto_open=False)
                    st.download_button(label="Download Plot", data=buf.getvalue(), file_name="sankey_diagram.html", mime="text/html")
                else:
                    st.info("No data to generate this plot for download.")

            elif selected_plot_to_download == "Network Graph":
                edges_download = filtered_melted_df.groupby(["state", "pollutant"]).size().reset_index(name="weight")
                if not edges_download.empty:
                    G = nx.Graph()
                    for _, row in edges_download.iterrows():
                        G.add_edge(row["state"], row["pollutant"], weight=row["weight"])
                    fig, ax = plt.subplots(figsize=(12, 10))
                    pos = nx.spring_layout(G, k=0.3, iterations=50)
                    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=[G.degree(node) * 100 for node in G.nodes()], alpha=0.9, ax=ax)
                    nx.draw_networkx_edges(G, pos, edge_color='gray', width=[d['weight'] / edges_download['weight'].max() * 5 for (u, v, d) in G.edges(data=True)], alpha=0.6, ax=ax)
                    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', ax=ax)
                    ax.set_title("Pollutant-State Network Graph")
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                    st.download_button(label="Download Plot", data=buf.getvalue(), file_name="network_graph.png", mime="image/png")
                    plt.close(fig)
                else:
                    st.info("No data to generate this plot for download.")

            elif selected_plot_to_download == "Dendrogram":
                pivot_table_dendro_download = filtered_melted_df.pivot_table(values="value", index="state", columns="pollutant", aggfunc="mean").fillna(0)
                if not pivot_table_dendro_download.empty and pivot_table_dendro_download.shape[0] > 1:
                    try:
                        linkage_matrix_download = linkage(pivot_table_dendro_download, method='ward')
                        fig, ax = plt.subplots(figsize=(14, 8))
                        dendrogram(linkage_matrix_download, labels=pivot_table_dendro_download.index, orientation='left', leaf_font_size=8, ax=ax)
                        ax.set_title("Dendrogram of States by Pollutant Levels (Hierarchical Clustering)")
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                        st.download_button(label="Download Plot", data=buf.getvalue(), file_name="dendrogram.png", mime="image/png")
                        plt.close(fig)
                    except Exception as e:
                        st.info(f"Could not generate Dendrogram for download: {e}")
                else:
                    st.info("No data to generate this plot for download.")

            elif selected_plot_to_download == "Ridgeline Plot":
                ridgeline_data_download = filtered_melted_df.dropna(subset=['state', 'value']).copy()
                if not ridgeline_data_download.empty and len(ridgeline_data_download['state'].unique()) > 1:
                    try:
                        import joypy # Ensure joypy is imported here for download logic
                        fig_ridgeline, axes = joypy.joyplot(ridgeline_data_download, by="state", column="value", figsize=(12, 8), xlabels=True, ylabels=True, overlap=1, grid=True, title="Ridgeline Plot of Pollutant Values by State")
                        buf = io.BytesIO()
                        fig_ridgeline.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                        st.download_button(label="Download Plot", data=buf.getvalue(), file_name="ridgeline_plot.png", mime="image/png")
                        plt.close(fig_ridgeline)
                    except Exception as e:
                        st.info(f"Could not generate Ridgeline Plot for download: {e}")
                else:
                    st.info("No data to generate this plot for download.")

            elif selected_plot_to_download == "Sunburst Chart":
                sunburst_data_download = filtered_melted_df.dropna(subset=['state', 'location', 'pollutant']).copy()
                if not sunburst_data_download.empty:
                    fig = px.sunburst(sunburst_data_download, path=['state', 'location', 'pollutant'], values='value', title="Sunburst Chart: Pollutant Values by State, Location, and Pollutant Type")
                    buf = io.BytesIO()
                    fig.write_html(buf, auto_open=False)
                    st.download_button(label="Download Plot", data=buf.getvalue(), file_name="sunburst_chart.html", mime="text/html")
                else:
                    st.info("No data to generate this plot for download.")

            elif selected_plot_to_download == "Parallel Coordinates Plot":
                sample_df_pc_download = filtered_melted_df[['value', 'year', 'month']].dropna()
                if not sample_df_pc_download.empty:
                    sample_pc_download = sample_df_pc_download.sample(min(2000, len(sample_df_pc_download)), random_state=42)
                    fig = px.parallel_coordinates(sample_pc_download, dimensions=['value', 'year', 'month'], title="Parallel Coordinates Plot: Pollutant Value, Year, and Month")
                    buf = io.BytesIO()
                    fig.write_html(buf, auto_open=False)
                    st.download_button(label="Download Plot", data=buf.getvalue(), file_name="parallel_coordinates.html", mime="text/html")
                else:
                    st.info("No data to generate this plot for download.")

            elif selected_plot_to_download == "Treemap":
                treemap_data_download = filtered_melted_df.dropna(subset=['state', 'pollutant']).copy()
                if not treemap_data_download.empty:
                    fig = px.treemap(treemap_data_download, path=['state', 'pollutant'], values='value', title="Treemap: Pollutant Values by State and Pollutant Type")
                    buf = io.BytesIO()
                    fig.write_html(buf, auto_open=False)
                    st.download_button(label="Download Plot", data=buf.getvalue(), file_name="treemap.html", mime="text/html")
                else:
                    st.info("No data to generate this plot for download.")

            elif selected_plot_to_download == "Gantt Chart":
                gantt_data_download = filtered_df[['location', 'date']].drop_duplicates().sort_values(by='date').dropna(subset=['location', 'date'])
                gantt_data_download = gantt_data_download.head(min(100, len(gantt_data_download)))
                if not gantt_data_download.empty:
                    gantt_data_download["End"] = gantt_data_download["date"] + pd.Timedelta(days=10)
                    fig = px.timeline(gantt_data_download, x_start="date", x_end="End", y="location", title="Gantt Chart: Sample Monitoring Periods by Location")
                    buf = io.BytesIO()
                    fig.write_html(buf, auto_open=False)
                    st.download_button(label="Download Plot", data=buf.getvalue(), file_name="gantt_chart.html", mime="text/html")
                else:
                    st.info("No data to generate this plot for download.")

            elif selected_plot_to_download == "Trellis Plot":
                trellis_data_download = filtered_melted_df.dropna(subset=['state', 'value']).copy()
                if not trellis_data_download.empty and len(trellis_data_download['state'].unique()) > 1:
                    try:
                        g_trellis = sns.FacetGrid(trellis_data_download, col="state", col_wrap=3, height=4, aspect=1.2, sharex=True, sharey=False)
                        g_trellis.map(sns.histplot, "value", bins=30, kde=True, color='teal', edgecolor='black')
                        plt.suptitle("Trellis Plot: Distribution of Pollutant Values by State", y=1.02, fontsize=16)
                        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
                        buf = io.BytesIO()
                        g_trellis.fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                        st.download_button(label="Download Plot", data=buf.getvalue(), file_name="trellis_plot.png", mime="image/png")
                        plt.close(g_trellis.fig)
                    except Exception as e:
                        st.info(f"Could not generate Trellis Plot for download: {e}")
                else:
                    st.info("No data to generate this plot for download.")

            elif selected_plot_to_download == "Violin + Swarm Plot":
                violin_swarm_data_download = filtered_melted_df.dropna(subset=['state', 'value']).copy()
                if not violin_swarm_data_download.empty and len(violin_swarm_data_download['state'].unique()) > 1:
                    fig, ax = plt.subplots(figsize=(14, 8))
                    sns.violinplot(x="state", y="value", data=violin_swarm_data_download, inner=None, color="lightgray", linewidth=1.5, ax=ax)
                    sample_swarm = violin_swarm_data_download.sample(min(5000, len(violin_swarm_data_download)), random_state=42)
                    sns.swarmplot(x="state", y="value", data=sample_swarm, size=3, color='darkblue', alpha=0.6, ax=ax)
                    ax.set_title("Violin + Swarm Plot: Pollutant Values by State")
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                    st.download_button(label="Download Plot", data=buf.getvalue(), file_name="violin_swarm_plot.png", mime="image/png")
                    plt.close(fig)
                else:
                    st.info("No data to generate this plot for download.")

            elif selected_plot_to_download == "Hexbin Plot":
                df_hex_download = filtered_melted_df[['value', 'month']].dropna()
                if not df_hex_download.empty:
                    fig, ax = plt.subplots(figsize=(10, 7))
                    hb = ax.hexbin(df_hex_download['month'], df_hex_download['value'], gridsize=30, cmap='Blues', edgecolors='none')
                    ax.set_title("Hexbin Plot: Pollutant Value Density by Month")
                    fig.colorbar(hb, ax=ax, label='Count in Bin')
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                    st.download_button(label="Download Plot", data=buf.getvalue(), file_name="hexbin_plot.png", mime="image/png")
                    plt.close(fig)
                else:
                    st.info("No data to generate this plot for download.")

            elif selected_plot_to_download == "Streamgraph":
                stream_data_download = filtered_melted_df.groupby(["year", "pollutant"])["value"].mean().unstack(fill_value=0)
                if not stream_data_download.empty:
                    fig, ax = plt.subplots(figsize=(14, 7))
                    stream_data_download.plot.area(ax=ax, alpha=0.7, linewidth=0)
                    ax.set_title("Streamgraph (Simulated): Mean Pollutant Concentration Over Years")
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                    st.download_button(label="Download Plot", data=buf.getvalue(), file_name="streamgraph.png", mime="image/png")
                    plt.close(fig)
                else:
                    st.info("No data to generate this plot for download.")

            elif selected_plot_to_download == "Geographic Heatmap":
                df_geo_download = filtered_melted_df.copy()
                # Ensure get_simulated_lat_lon_for_eda is defined or imported for download logic
                @st.cache_data
                def get_simulated_lat_lon_for_eda(locations):
                    lat_map = {loc: np.random.uniform(9.0, 34.0) for loc in locations}
                    lon_map = {loc: np.random.uniform(69.0, 96.0) for loc in locations}
                    return lat_map, lon_map

                unique_locations_in_filtered_download = df_geo_download['location'].unique()
                location_lat_map_download, location_lon_map_download = get_simulated_lat_lon_for_eda(unique_locations_in_filtered_download)
                df_geo_download["lat"] = df_geo_download["location"].map(location_lat_map_download)
                df_geo_download["lon"] = df_geo_download["location"].map(location_lon_map_download)
                df_geo_download.dropna(subset=['lat', 'lon', 'value'], inplace=True)

                if not selected_pollutants:
                    st.warning("Please select at least one pollutant in the sidebar to download this plot.")
                else:
                    pollutant_for_download = selected_pollutants[0]
                    df_geo_filtered_download = df_geo_download[df_geo_download['pollutant'] == pollutant_for_download]
                    if not df_geo_filtered_download.empty:
                        fig = px.density_mapbox(df_geo_filtered_download, lat="lat", lon="lon", z="value", radius=10,
                                                center=dict(lat=22.5, lon=80), zoom=3, mapbox_style="stamen-terrain",
                                                title=f"Geographic Heatmap: {pollutant_for_download.upper()} Density (Simulated Locations)")
                        buf = io.BytesIO()
                        fig.write_html(buf, auto_open=False)
                        st.download_button(label="Download Plot", data=buf.getvalue(), file_name=f"geographic_heatmap_{pollutant_for_download}.html", mime="text/html")
                    else:
                        st.info("No data to generate this plot for download.")

            elif selected_plot_to_download == "Word Cloud":
                text_data_wc_download = " ".join(filtered_df["location"].astype(str).dropna().tolist())
                if text_data_wc_download:
                    from wordcloud import WordCloud
                    wordcloud = WordCloud(width=1000, height=500, background_color="white", collocations=True, min_font_size=10, colormap='viridis').generate(text_data_wc_download)
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.imshow(wordcloud, interpolation="bilinear")
                    ax.axis("off")
                    ax.set_title("Word Cloud of Locations (with Collocations)")
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                    st.download_button(label="Download Plot", data=buf.getvalue(), file_name="wordcloud.png", mime="image/png")
                    plt.close(fig)
                else:
                    st.info("No data to generate this plot for download.")

            elif selected_plot_to_download == "Time Series Clustering":
                # Re-run the clustering logic to get the plot data
                cluster_by_option_download = "State" # Default for download
                cluster_pollutant_download = selected_pollutants[0] if selected_pollutants else None

                if cluster_pollutant_download:
                    group_cols_download = ['state', 'pollutant'] if cluster_by_option_download == "State" else ['location', 'pollutant']
                    ts_cluster_data_download = filtered_melted_df.groupby(group_cols_download + [pd.Grouper(key='date', freq='ME')])['value'].mean().unstack(level='date')
                    ts_cluster_data_download.fillna(0, inplace=True)

                    if cluster_pollutant_download in ts_cluster_data_download.index.get_level_values('pollutant'):
                        ts_cluster_filtered_download = ts_cluster_data_download.loc[(slice(None), cluster_pollutant_download), :].droplevel('pollutant')
                    else:
                        ts_cluster_filtered_download = pd.DataFrame()

                    if not ts_cluster_filtered_download.empty and ts_cluster_filtered_download.shape[0] > 1:
                        num_clusters_download = min(3, ts_cluster_filtered_download.shape[0])
                        if num_clusters_download >= 2:
                            scaler = StandardScaler()
                            scaled_features = scaler.fit_transform(ts_cluster_filtered_download)
                            kmeans = KMeans(n_clusters=num_clusters_download, random_state=42, n_init=10)
                            ts_cluster_filtered_download['cluster'] = kmeans.fit_predict(scaled_features)

                            cluster_trends_for_plot_download = ts_cluster_filtered_download.groupby('cluster').mean().reset_index()
                            date_cols_download = [col for col in cluster_trends_for_plot_download.columns if isinstance(col, pd.Timestamp)]
                            cluster_trends_melted_download = cluster_trends_for_plot_download.melt(
                                id_vars=['cluster'], value_vars=date_cols_download, var_name='date', value_name='average_value'
                            )
                            fig = px.line(cluster_trends_melted_download, x='date', y='average_value', color='cluster',
                                            title=f'Average {cluster_pollutant_download.upper()} Trends for Each Cluster by {cluster_by_option_download} (Monthly Averages)')
                            buf = io.BytesIO()
                            fig.write_html(buf, auto_open=False)
                            st.download_button(label="Download Plot", data=buf.getvalue(), file_name=f"time_series_clustering_{cluster_pollutant_download}.html", mime="text/html")
                        else:
                            st.info("Not enough data to generate this plot for download.")
                    else:
                        st.info("No data to generate this plot for download.")
                else:
                    st.warning("Please select at least one pollutant in the sidebar to download this plot.")
