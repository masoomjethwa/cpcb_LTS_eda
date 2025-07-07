#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EDA Script for CPCB Air Quality Data (1990–2015)

This script performs an Exploratory Data Analysis (EDA) on the CPCB air quality
data from 1990 to 2015. It covers data loading, initial inspection, cleaning,
missing value handling, descriptive statistics, and univariate and bivariate
visualizations. It also includes an advanced plotting section for deeper insights.

@author: codernumber1
"coding assistant": 78945 # A random number generated for this session.
"""

# -----------------------------
# 📦 Required Libraries
# -----------------------------
# To install these libraries, use pip:
# pip install pandas numpy seaborn matplotlib missingno plotly plotly-express wordcloud networkx joypy scikit-learn statsmodels

import random
import os
from datetime import datetime
from itertools import combinations # For Network Graph (potential future use for node connections)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
import plotly.express as px
import plotly.graph_objects as go # For more advanced Plotly features
from wordcloud import WordCloud # For creating word clouds
import networkx as nx # For network graphs
from scipy.cluster.hierarchy import linkage, dendrogram # For hierarchical clustering in heatmaps
import joypy # For Ridgeline plots
from sklearn.cluster import KMeans # For clustering time series data
from sklearn.preprocessing import StandardScaler # For scaling data before clustering
from statsmodels.tsa.seasonal import STL # For Seasonal-Trend decomposition using Loess

# -----------------------------
# 📁 Data Path and Loading
# -----------------------------
# Define the path to your CSV file.
# Replace "path to files" with the actual directory where your data is stored.
DATA_FILE_PATH = "CPCB_data_from1990_2015.csv"

# Check if the data file exists before attempting to load
if not os.path.exists(DATA_FILE_PATH):
    print(f"Error: Data file not found at '{DATA_FILE_PATH}'. "
          "Please update DATA_FILE_PATH to the correct location.")
    exit()

# Load the CPCB dataset efficiently using pandas.
# The 'encoding' is changed to 'latin-1' to handle potential decoding errors
# like 'UnicodeDecodeError' which often occur with non-UTF-8 encoded CSV files.
# 'low_memory=False' is used to prevent mixed-type inference warnings for large files.
try:
    df = pd.read_csv(DATA_FILE_PATH, encoding='latin-1', low_memory=False)
    print(f"Successfully loaded data from '{DATA_FILE_PATH}' with 'latin-1' encoding.")
except Exception as e:
    print(f"Error loading CSV file: {e}")
    print("If the error persists, try changing the 'encoding' parameter (e.g., 'cp1252').")
    exit()

# -----------------------------
# 🔍 Initial Data Inspection
# -----------------------------
print("\n--- Initial Data Inspection ---")
print(f"Shape of the dataset (rows, columns): {df.shape}")
print("\nFirst 5 rows of the dataset:")
print(df.head())

print("\nColumn Information and Data Types:")
# Display concise summary of the DataFrame, including data types and non-null values.
df.info()

# -----------------------------
# 🧹 Data Cleaning and Transformation
# -----------------------------
print("\n--- Data Cleaning and Transformation ---")

# Replace all occurrences of the string 'NA' with NumPy's NaN (Not a Number).
# This prepares the data for proper numerical conversion and missing value handling.
df.replace('NA', np.nan, inplace=True)
print("Replaced 'NA' strings with NaN values.")

def parse_sampling_date(date_str: str) -> pd.Timestamp | None:
    """
    Custom function to parse the 'sampling_date' column.
    Handles formats like "February - M021990" by extracting the M%m%Y part.

    Args:
        date_str (str): The date string to parse.

    Returns:
        pd.Timestamp | None: A pandas Timestamp object if parsing is successful,
                             otherwise pd.NaT (Not a Time) for invalid dates.
    """
    if pd.isna(date_str):
        return pd.NaT
    try:
        # Split by '-' and take the last part, then strip whitespace.
        # Example: "February - M021990" -> "M021990"
        date_part = date_str.split("-")[-1].strip()
        return datetime.strptime(date_part, "M%m%Y")
    except (ValueError, IndexError):
        # Catch errors during parsing (e.g., if format is unexpected or string is empty)
        return pd.NaT

# Apply the custom parsing function to the 'sampling_date' column.
df["sampling_date"] = df["sampling_date"].apply(parse_sampling_date)
print("Converted 'sampling_date' to datetime objects.")

# Convert the 'date' column to datetime.
# 'errors='coerce'' will turn any unparseable dates into NaT.
# 'dayfirst=True' is used based on the sample format '01-02-1990'.
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors='coerce', dayfirst=True)
    print("Converted 'date' column to datetime objects.")
else:
    print("Column 'date' not found in the dataset. Skipping conversion.")

# Extract temporal features (year, month, day) from 'sampling_date'.
# These can be useful for time-series analysis and aggregations.
df["year"] = df["sampling_date"].dt.year
df["month"] = df["sampling_date"].dt.month
df["day"] = df["sampling_date"].dt.day
print("Extracted 'year', 'month', 'day' from 'sampling_date'.")

# List of columns containing air pollutant concentrations.
POLLUTANT_COLUMNS = ["so2", "no2", "rspm", "spm", "pm2_5"]

# Convert pollutant columns to numeric types.
# 'errors='coerce'' will convert any non-numeric values (including NaNs from 'NA' replacement)
# into NaN, allowing for numerical operations.
for col in POLLUTANT_COLUMNS:
    df[col] = pd.to_numeric(df[col], errors='coerce')
print(f"Converted pollutant columns {POLLUTANT_COLUMNS} to numeric types.")

# -----------------------------
# 🔍 Missing Values Analysis
# -----------------------------
print("\n--- Missing Values Analysis ---")

# Initialize plot counter for consistent numbering across all plots
plot_counter = 0

# Calculate the percentage of missing values for each column.
missing_percent = df.isnull().mean() * 100
print("\nMissing Data Percentage per Column (sorted descending):")
print(missing_percent.sort_values(ascending=False))

# Visualize missing data patterns using the missingno library.
# The matrix plot shows the presence/absence of data for each column.
plot_counter += 1
plt.figure(figsize=(10, 6))
msno.matrix(df, color=(0.2, 0.2, 0.3))
plt.title("Missing Data Matrix", fontsize=16)
plt.savefig(f"plots/plot_{plot_counter}_missing_data_matrix.png", dpi=300) # Save with high DPI
plt.close() # Close the plot to free memory
print(f"Plot {plot_counter}: Missing Data Matrix saved to plots/plot_{plot_counter}_missing_data_matrix.png")


# The heatmap shows the correlation of missingness between columns.
# A value close to 1 means if one column is missing, the other is likely missing too.
plot_counter += 1
plt.figure(figsize=(10, 8))
msno.heatmap(df, cmap="viridis")
plt.title("Missing Data Correlation Heatmap", fontsize=16)
plt.savefig(f"plots/plot_{plot_counter}_missing_data_correlation_heatmap.png", dpi=300) # Save with high DPI
plt.close() # Close the plot to free memory
print(f"Plot {plot_counter}: Missing Data Correlation Heatmap saved to plots/plot_{plot_counter}_missing_data_correlation_heatmap.png")

# -----------------------------
# 🧼 Impute Missing Values
# -----------------------------
print("\n--- Imputing Missing Values ---")

# Simple mean imputation for numerical pollutant columns.
# This fills NaN values with the mean of the respective column.
# For more robust analysis, consider advanced imputation techniques
# like K-Nearest Neighbors (KNN) imputation or MICE (Multiple Imputation by Chained Equations).
for col in POLLUTANT_COLUMNS:
    if df[col].isnull().sum() > 0:
        mean_val = df[col].mean()
        # Using .loc for direct assignment to avoid SettingWithCopyWarning
        df.loc[:, col] = df[col].fillna(mean_val)
        print(f"Imputed missing values in '{col}' with its mean ({mean_val:.2f}).")
    else:
        print(f"No missing values found in '{col}'.")

# Verify that there are no more missing values in pollutant columns
print("\nMissing values after imputation:")
print(df[POLLUTANT_COLUMNS].isnull().sum())

# -----------------------------
# 📊 Descriptive Statistics
# -----------------------------
print("\n--- Descriptive Statistics ---")

# Generate summary statistics for all numerical columns.
# This includes count, mean, standard deviation, min, max, and quartiles.
print("\nDescriptive Statistics for Numerical Columns:")
print(df[POLLUTANT_COLUMNS + ["year", "month", "day"]].describe().T)
df[POLLUTANT_COLUMNS + ["year", "month", "day"]].describe().T.to_csv("plot_data/Plot_descriptive_statistics_data.csv")
print("Descriptive statistics saved to plot_data/Plot_descriptive_statistics_data.csv")


# Identify and list categorical columns for further analysis.
CATEGORICAL_COLUMNS = ["state", "location", "agency", "type", "location_monitoring_station"]

print("\nUnique Value Counts for Categorical Variables:")
for col in CATEGORICAL_COLUMNS:
    print(f"\n--- Column: '{col}' ---")
    # Display unique values and their counts, including NaN values if any remain.
    print(df[col].value_counts(dropna=False))
    df[col].value_counts(dropna=False).to_csv(f"plot_data/Plot_{col}_value_counts_data.csv")
    print(f"Value counts for '{col}' saved to plot_data/Plot_{col}_value_counts_data.csv")
    # Optionally, check for inconsistencies or typos in categorical data here.
    # E.g., df[col].value_counts().head(50) to spot variations.

# -----------------------------
# 📈 Univariate Visualizations
# -----------------------------
print("\n--- Univariate Visualizations (Pollutant Distributions) ---")

# Create histograms and box plots for each numerical pollutant column.
# Histograms show the distribution shape (skewness, kurtosis).
# Box plots help identify outliers and the spread of the data.
for col in POLLUTANT_COLUMNS:
    plot_counter += 1
    plt.figure(figsize=(10, 5))

    # Histogram with Kernel Density Estimate (KDE)
    plt.subplot(1, 2, 1) # 1 row, 2 columns, first plot
    sns.histplot(df[col], kde=True, bins=50, color='skyblue', edgecolor='black')
    plt.title(f"Distribution of {col}", fontsize=14)
    plt.xlabel(col, fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Boxplot
    plt.subplot(1, 2, 2) # 1 row, 2 columns, second plot
    sns.boxplot(x=df[col], color='lightcoral')
    plt.title(f"Boxplot of {col}", fontsize=14)
    plt.xlabel(col, fontsize=12)
    plt.tight_layout() # Adjust layout to prevent overlapping titles/labels
    plt.savefig(f"plots/plot_{plot_counter}_univariate_{col}_distribution.png", dpi=300) # Save with high DPI
    plt.close() # Close the plot to free memory
    print(f"Plot {plot_counter}: Univariate plots for {col} saved to plots/plot_{plot_counter}_univariate_{col}_distribution.png")

# -----------------------------
# 📉 Bivariate Analysis: Correlation Heatmap
# -----------------------------
print("\n--- Bivariate Analysis: Correlation Heatmap ---")

# Calculate the Pearson correlation matrix for all numerical pollutant columns.
# This quantifies the linear relationships between pairs of pollutants.
correlation_matrix = df[POLLUTANT_COLUMNS].corr()

# Visualize the correlation matrix using a heatmap.
# 'annot=True' displays the correlation values on the heatmap.
# 'cmap="coolwarm"' provides a diverging color map to easily distinguish positive/negative correlations.
plot_counter += 1
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5, fmt=".2f")
plt.title("Correlation Heatmap of Air Pollutants", fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(f"plots/plot_{plot_counter}_correlation_heatmap.png", dpi=300) # Save with high DPI
plt.close() # Close the plot to free memory
correlation_matrix.to_csv(f"plot_data/Plot_{plot_counter}_correlation_matrix_data.csv")
print(f"Plot {plot_counter}: Correlation Heatmap saved to plots/plot_{plot_counter}_correlation_heatmap.png and data to plot_data/Plot_{plot_counter}_correlation_matrix_data.csv")

# -----------------------------
# 🕰️ Time Series Trends Analysis
# -----------------------------
print("\n--- Time Series Trends Analysis ---")

# Group the data by 'year' and calculate the mean concentration for each pollutant.
# This helps observe long-term trends in air quality.
df_yearly_avg = df.groupby("year")[POLLUTANT_COLUMNS].mean().reset_index()

# Plot the yearly average concentration of each pollutant.
plot_counter += 1
plt.figure(figsize=(14, 7))
for col in POLLUTANT_COLUMNS:
    plt.plot(df_yearly_avg["year"], df_yearly_avg[col],
             marker='o', linestyle='-', linewidth=2, label=col)

plt.title("Yearly Average Concentration of Pollutants (1990-2015)", fontsize=16)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Average Concentration", fontsize=12)
plt.legend(title="Pollutant", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(df_yearly_avg["year"].unique()[::2], rotation=45) # Show every other year for readability
plt.tight_layout()
plt.savefig(f"plots/plot_{plot_counter}_yearly_average_pollutant_trends.png", dpi=300) # Save with high DPI
plt.close() # Close the plot to free memory
df_yearly_avg.to_csv(f"plot_data/Plot_{plot_counter}_yearly_average_pollutant_trends_data.csv", index=False)
print(f"Plot {plot_counter}: Yearly Average Pollutant Trends plot saved to plots/plot_{plot_counter}_yearly_average_pollutant_trends.png and data to plot_data/Plot_{plot_counter}_yearly_average_pollutant_trends_data.csv")

# -----------------------------
# 🎨 Advanced Visualizations (14 Plots)
# -----------------------------
print("\n--- Generating Advanced Visualizations ---")

# Create output directories for plots and plot data
os.makedirs("plots", exist_ok=True)
os.makedirs("plot_data", exist_ok=True)
print("Created 'plots/' and 'plot_data/' directories.")

# Melt pollutant columns for easier plotting with 'pollutant' and 'value' columns.
# This reshapes the DataFrame from wide format (so2, no2, etc. as separate columns)
# to long format (a single 'pollutant' column and a 'value' column).
df_melted = df.melt(
    id_vars=[col for col in df.columns if col not in POLLUTANT_COLUMNS],
    value_vars=POLLUTANT_COLUMNS,
    var_name='pollutant',
    value_name='value'
)
# Drop rows where 'value' is NaN after melting, ensuring all values are valid for plotting.
df_melted.dropna(subset=['value'], inplace=True)
print("Melted pollutant columns into 'pollutant' and 'value' for advanced plotting.")

# 1. Sankey Diagram: Visualizes flow and relationships between categories.
# Here, showing the flow from states to pollutants based on the count of records.
plot_counter += 1
print(f"Generating Plot {plot_counter}: Sankey Diagram...")
sankey_data = df_melted.groupby(["state", "pollutant"]).size().reset_index(name='count')
# Create unique labels for nodes (states and pollutants)
labels = list(sankey_data["state"].unique()) + list(sankey_data["pollutant"].unique())
# Map states and pollutants to numerical indices for source and target
state_map = {k: i for i, k in enumerate(sankey_data["state"].unique())}
pollutant_map = {k: i + len(sankey_data["state"].unique()) for i, k in enumerate(sankey_data["pollutant"].unique())}

fig1 = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=labels,
        color="blue"
    ),
    link=dict(
        source=sankey_data["state"].map(state_map),
        target=sankey_data["pollutant"].map(pollutant_map),
        value=sankey_data["count"]
    )
)])
fig1.update_layout(title_text="Sankey Diagram: State to Pollutant Flow", font_size=10)
fig1.write_html(f"plots/plot_{plot_counter}_sankey_diagram.html")
sankey_data.to_csv(f"plot_data/Plot_{plot_counter}_sankey_diagram_data.csv", index=False) # Updated CSV name
print(f"Plot {plot_counter}: Sankey Diagram saved to plots/plot_{plot_counter}_sankey_diagram.html and data to plot_data/Plot_{plot_counter}_sankey_diagram_data.csv")

# 2. Network Graph (Force-Directed Layout): Represents relationships and connections.
# Here, nodes are states and pollutants, and edges represent their co-occurrence.
plot_counter += 1
print(f"Generating Plot {plot_counter}: Network Graph...")
G = nx.Graph()
# Group by state and pollutant to get the weight (e.g., count of records) for edges
edges = df_melted.groupby(["state", "pollutant"]).size().reset_index(name="weight")
for _, row in edges.iterrows():
    G.add_edge(row["state"], row["pollutant"], weight=row["weight"])

plt.figure(figsize=(12, 10))
# Use spring_layout for a force-directed layout
pos = nx.spring_layout(G, k=0.3, iterations=50) # k regulates distance between nodes
node_sizes = [G.degree(node) * 100 for node in G.nodes()] # Size nodes by degree
edge_widths = [d['weight'] / edges['weight'].max() * 5 for (u, v, d) in G.edges(data=True)] # Scale edge widths

nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=node_sizes, alpha=0.9)
nx.draw_networkx_edges(G, pos, edge_color='gray', width=edge_widths, alpha=0.6)
nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')
plt.title("Pollutant-State Network Graph", fontsize=16)
plt.axis("off") # Hide axes
plt.tight_layout()
plt.savefig(f"plots/plot_{plot_counter}_network_graph.png", dpi=300) # Save with high DPI
plt.close() # Close the plot to free memory
edges.to_csv(f"plot_data/Plot_{plot_counter}_network_graph_data.csv", index=False) # Updated CSV name
print(f"Plot {plot_counter}: Network Graph saved to plots/plot_{plot_counter}_network_graph.png and data to plot_data/Plot_{plot_counter}_network_graph_data.csv")

# 3. Heatmap with Dendrograms: Visualizes correlations or similarities with hierarchical clustering.
# Shows how states cluster based on their average pollutant levels.
plot_counter += 1
print(f"Generating Plot {plot_counter}: Heatmap with Dendrogram...")
# Pivot table to get states as index, pollutants as columns, and mean value as data
pivot_table = df_melted.pivot_table(values="value", index="state", columns="pollutant", aggfunc="mean")
# Fill any NaN values that might result from the pivot (e.g., if a state has no data for a pollutant)
pivot_table.fillna(0, inplace=True) # Using 0 for missing values; consider mean/median if appropriate

# Perform hierarchical clustering on the states (rows)
linkage_matrix = linkage(pivot_table, method='ward') # 'ward' minimizes variance within clusters

plt.figure(figsize=(14, 8))
# Create dendrogram
dendrogram(linkage_matrix, labels=pivot_table.index, orientation='left', leaf_font_size=8)
plt.title("Dendrogram of States by Pollutant Levels (Hierarchical Clustering)", fontsize=16)
plt.xlabel("Distance", fontsize=12)
plt.ylabel("State", fontsize=12)
plt.tight_layout()
plt.savefig(f"plots/plot_{plot_counter}_dendrogram.png", dpi=300) # Save with high DPI
plt.close() # Close the plot to free memory
pivot_table.to_csv(f"plot_data/Plot_{plot_counter}_hierarchical_clustering_data.csv") # Updated CSV name for clarity
print(f"Plot {plot_counter}: Dendrogram saved to plots/plot_{plot_counter}_dendrogram.png and data to plot_data/Plot_{plot_counter}_hierarchical_clustering_data.csv")

# 4. Ridgeline Plot (Joyplot): Shows the distribution of a numerical variable across several categories, with overlapping densities.
# Useful for comparing pollutant value distributions across different states.
plot_counter += 1
print(f"Generating Plot {plot_counter}: Ridgeline Plot...")
# Ensure 'value' column is numeric and 'state' is categorical
if not pd.api.types.is_numeric_dtype(df_melted['value']):
    df_melted['value'] = pd.to_numeric(df_melted['value'], errors='coerce')
    df_melted.dropna(subset=['value'], inplace=True)

# joypy requires a specific structure, ensure 'value' and 'state' are present
if not df_melted.empty:
    fig_ridgeline, axes = joypy.joyplot(
        df_melted,
        by="state",
        column="value",
        figsize=(12, 8),
        xlabels=True,
        ylabels=True,
        overlap=1, # Adjust overlap for better visualization
        grid=True,
        title="Ridgeline Plot of Pollutant Values by State",
        linewidth=1,
        alpha=0.7
    )
    plt.xlabel("Pollutant Value", fontsize=12)
    plt.ylabel("State Density", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"plots/plot_{plot_counter}_ridgeline_plot.png", dpi=300) # Save with high DPI
    plt.close() # Close the plot to free memory
    df_melted[['state', 'value']].to_csv(f"plot_data/Plot_{plot_counter}_ridgeline_plot_data.csv", index=False) # Updated CSV name
    print(f"Plot {plot_counter}: Ridgeline Plot saved to plots/plot_{plot_counter}_ridgeline_plot.png and data to plot_data/Plot_{plot_counter}_ridgeline_plot_data.csv")
else:
    print(f"Skipping Plot {plot_counter}: Ridgeline Plot: df_melted is empty after cleaning.")


# 5. Sunburst Chart: Displays hierarchical data in a radial layout, ideal for exploring part-to-whole relationships.
# Visualizes the distribution of pollutant values across state, location, and pollutant type.
plot_counter += 1
print(f"Generating Plot {plot_counter}: Sunburst Chart...")
# IMPORTANT: Drop rows with NaN in path columns before generating Sunburst chart
# Plotly's Sunburst chart does not handle None/NaN values in its hierarchy path.
sunburst_data = df_melted.dropna(subset=['state', 'location', 'pollutant']).copy()

if not sunburst_data.empty:
    fig5 = px.sunburst(
        sunburst_data,
        path=['state', 'location', 'pollutant'], # Define the hierarchy
        values='value', # Values to determine segment sizes
        title="Sunburst Chart: Pollutant Values by State, Location, and Pollutant Type"
    )
    fig5.write_html(f"plots/plot_{plot_counter}_sunburst_chart.html")
    sunburst_data[['state', 'location', 'pollutant', 'value']].to_csv(f"plot_data/Plot_{plot_counter}_sunburst_chart_data.csv", index=False) # Updated CSV name
    print(f"Plot {plot_counter}: Sunburst Chart saved to plots/plot_{plot_counter}_sunburst_chart.html and data to plot_data/Plot_{plot_counter}_sunburst_chart_data.csv")
else:
    print(f"Skipping Plot {plot_counter}: Sunburst Chart: Insufficient data after dropping NaNs in hierarchy path columns.")


# 6. Parallel Coordinates Plot: Visualizes multi-dimensional data, allowing the observation of relationships and clusters.
# Shows how different numerical attributes (value, year, month) relate for a sample of data points.
plot_counter += 1
print(f"Generating Plot {plot_counter}: Parallel Coordinates Plot...")
# Sample a smaller subset for better visualization with parallel coordinates, as it can get cluttered.
# Ensure 'value', 'year', 'month' are numeric and present.
sample_df = df_melted[['value', 'year', 'month']].dropna()
if not sample_df.empty:
    sample = sample_df.sample(min(2000, len(sample_df)), random_state=42) # Sample up to 2000 rows
    fig6 = px.parallel_coordinates(
        sample,
        dimensions=['value', 'year', 'month'],
        title="Parallel Coordinates Plot: Pollutant Value, Year, and Month"
    )
    fig6.write_html(f"plots/plot_{plot_counter}_parallel_coordinates.html")
    sample.to_csv(f"plot_data/Plot_{plot_counter}_parallel_coordinates_data.csv", index=False) # Updated CSV name
    print(f"Plot {plot_counter}: Parallel Coordinates Plot saved to plots/plot_{plot_counter}_parallel_coordinates.html and data to plot_data/Plot_{plot_counter}_parallel_coordinates_data.csv")
else:
    print(f"Skipping Plot {plot_counter}: Parallel Coordinates Plot: Insufficient data after dropping NaNs.")

# 7. Treemap: Displays hierarchical data as a set of nested rectangles, with sizes representing proportions.
# Shows the proportion of pollutant values across states and specific pollutants.
plot_counter += 1
print(f"Generating Plot {plot_counter}: Treemap...")
# IMPORTANT: Drop rows with NaN in path columns before generating Treemap
treemap_data = df_melted.dropna(subset=['state', 'pollutant']).copy()

if not treemap_data.empty:
    fig7 = px.treemap(
        treemap_data,
        path=['state', 'pollutant'], # Define the hierarchy
        values='value', # Values to determine rectangle sizes
        title="Treemap: Pollutant Values by State and Pollutant Type"
    )
    fig7.write_html(f"plots/plot_{plot_counter}_treemap.html")
    treemap_data[['state', 'pollutant', 'value']].to_csv(f"plot_data/Plot_{plot_counter}_treemap_data.csv", index=False) # Updated CSV name
    print(f"Plot {plot_counter}: Treemap saved to plots/plot_{plot_counter}_treemap.html and data to plot_data/Plot_{plot_counter}_treemap_data.csv")
else:
    print(f"Skipping Plot {plot_counter}: Treemap: Insufficient data after dropping NaNs in hierarchy path columns.")

# 8. Gantt Chart (simplified for visualizing campaigns/periods):
# This is a conceptual example for visualizing time-based events.
# Here, it simulates monitoring periods for different locations.
plot_counter += 1
print(f"Generating Plot {plot_counter}: Gantt Chart...")
# Use original df as 'date' is a primary key for this visualization
df_gantt = df[['location', 'date']].drop_duplicates().sort_values(by='date')
# Drop rows with NaN in 'location' or 'date' for Gantt chart
df_gantt.dropna(subset=['location', 'date'], inplace=True)

# Take a reasonable sample for visualization, as 4M rows would be too much
df_gantt = df_gantt.head(min(100, len(df_gantt))) # Limit to 100 entries for readability
if not df_gantt.empty:
    # Assume each monitoring instance lasts for 10 days for visualization purposes
    df_gantt["End"] = df_gantt["date"] + pd.Timedelta(days=10)
    fig8 = px.timeline(
        df_gantt,
        x_start="date",
        x_end="End",
        y="location",
        title="Gantt Chart: Sample Monitoring Periods by Location"
    )
    fig8.update_yaxes(autorange="reversed") # Locations from top to bottom
    fig8.write_html(f"plots/plot_{plot_counter}_gantt_chart.html")
    df_gantt.to_csv(f"plot_data/Plot_{plot_counter}_gantt_chart_data.csv", index=False) # Updated CSV name
    print(f"Plot {plot_counter}: Gantt Chart saved to plots/plot_{plot_counter}_gantt_chart.html and data to plot_data/Plot_{plot_counter}_gantt_chart_data.csv")
else:
    print(f"Skipping Plot {plot_counter}: Gantt Chart: Insufficient data for locations/dates.")

# 9. Trellis Plot / Faceted Plot (Complex Grid Layouts): Creates a grid of multiple subplots.
# Shows the distribution of pollutant values for each state in separate subplots.
plot_counter += 1
print(f"Generating Plot {plot_counter}: Trellis Plot...")
# Ensure 'value' column is numeric for histplot
# Drop rows with NaN in 'state' or 'value' for FacetGrid
trellis_data = df_melted.dropna(subset=['state', 'value']).copy()

if not trellis_data.empty:
    g = sns.FacetGrid(trellis_data, col="state", col_wrap=3, height=4, aspect=1.2, sharex=True, sharey=False)
    g.map(sns.histplot, "value", bins=30, kde=True, color='teal', edgecolor='black')
    g.set_titles("State: {col_name}")
    g.set_axis_labels("Pollutant Value", "Frequency")
    plt.suptitle("Trellis Plot: Distribution of Pollutant Values by State", y=1.02, fontsize=16) # Adjust suptitle position
    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout to make space for suptitle
    plt.savefig(f"plots/plot_{plot_counter}_trellis_plot.png", dpi=300) # Save with high DPI
    plt.close() # Close the plot to free memory
    trellis_data[['state', 'value']].to_csv(f"plot_data/Plot_{plot_counter}_trellis_plot_data.csv", index=False)
    print(f"Plot {plot_counter}: Trellis Plot saved to plots/plot_{plot_counter}_trellis_plot.png and data to plot_data/Plot_{plot_counter}_trellis_plot_data.csv")
else:
    print(f"Skipping Plot {plot_counter}: Trellis Plot: Insufficient data after dropping NaNs.")

# 10. Violin Plot with Swarm Plot Overlay: Combines density distribution with individual data points.
# Provides a rich view of data distribution and individual observations for pollutant values across states.
plot_counter += 1
print(f"Generating Plot {plot_counter}: Violin Plot with Swarm Overlay...")
# Drop rows with NaN in 'state' or 'value' for Violin/Swarm plot
violin_swarm_data = df_melted.dropna(subset=['state', 'value']).copy()

plt.figure(figsize=(14, 8))
if not violin_swarm_data.empty:
    # Violin plot shows the distribution shape
    sns.violinplot(x="state", y="value", data=violin_swarm_data, inner=None, color="lightgray", linewidth=1.5)
    # Swarm plot overlays individual data points, useful for smaller datasets or to see density
    # For very large datasets, consider sns.boxplot + sns.stripplot(jitter=0.2) or just violin.
    # Limiting swarm plot to a sample for performance on large datasets
    sample_swarm = violin_swarm_data.sample(min(5000, len(violin_swarm_data)), random_state=42) # Sample up to 5000 points
    sns.swarmplot(x="state", y="value", data=sample_swarm, size=3, color='darkblue', alpha=0.6)
    plt.title("Violin + Swarm Plot: Pollutant Values by State", fontsize=16)
    plt.xlabel("State", fontsize=12)
    plt.ylabel("Pollutant Value", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"plots/plot_{plot_counter}_violin_swarm_plot.png", dpi=300) # Save with high DPI
    plt.close() # Close the plot to free memory
    violin_swarm_data[['state', 'value']].to_csv(f"plot_data/Plot_{plot_counter}_violin_swarm_plot_data.csv", index=False)
    print(f"Plot {plot_counter}: Violin + Swarm Plot saved to plots/plot_{plot_counter}_violin_swarm_plot.png and data to plot_data/Plot_{plot_counter}_violin_swarm_plot_data.csv")
else:
    print(f"Skipping Plot {plot_counter}: Violin + Swarm Plot: Insufficient data after dropping NaNs.")

# 11. Hexbin Plot (for large scatter data): Useful for visualizing the density of points in a scatter plot.
# Shows the density of pollutant values across different months.
plot_counter += 1
print(f"Generating Plot {plot_counter}: Hexbin Plot...")
plt.figure(figsize=(10, 7))
df_hex = df_melted[['value', 'month']].dropna()
if not df_hex.empty:
    plt.hexbin(df_hex['month'], df_hex['value'], gridsize=30, cmap='Blues', edgecolors='none')
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Pollutant Value', fontsize=12)
    plt.title("Hexbin Plot: Pollutant Value Density by Month", fontsize=16)
    plt.colorbar(label='Count in Bin')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"plots/plot_{plot_counter}_hexbin_plot.png", dpi=300) # Save with high DPI
    plt.close() # Close the plot to free memory
    df_hex.to_csv(f"plot_data/Plot_{plot_counter}_hexbin_plot_data.csv", index=False)
    print(f"Plot {plot_counter}: Hexbin Plot saved to plots/plot_{plot_counter}_hexbin_plot.png and data to plot_data/Plot_{plot_counter}_hexbin_plot_data.csv")
else:
    print(f"Skipping Plot {plot_counter}: Hexbin Plot: Insufficient data after dropping NaNs.")

# 12. Streamgraph (simulated as stacked area chart): Shows changes in quantities over time, creating a flowing shape.
# Visualizes the contribution of each pollutant to the total average concentration over years.
plot_counter += 1
print(f"Generating Plot {plot_counter}: Streamgraph (Simulated)...")
# Group by year and pollutant, then unstack to get pollutants as columns
stream_data = df_melted.groupby(["year", "pollutant"])["value"].mean().unstack(fill_value=0)
if not stream_data.empty:
    plt.figure(figsize=(14, 7))
    stream_data.plot.area(
        ax=plt.gca(), # Use current axes
        figsize=(14, 7),
        alpha=0.7,
        linewidth=0 # No border lines for area
    )
    plt.title("Streamgraph (Simulated): Mean Pollutant Concentration Over Years", fontsize=16)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Mean Concentration", fontsize=12)
    plt.legend(title="Pollutant", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"plots/plot_{plot_counter}_streamgraph.png", dpi=300) # Save with high DPI
    plt.close() # Close the plot to free memory
    stream_data.to_csv(f"plot_data/Plot_{plot_counter}_streamgraph_data.csv")
    print(f"Plot {plot_counter}: Streamgraph (Simulated) saved to plots/plot_{plot_counter}_streamgraph.png and data to plot_data/Plot_{plot_counter}_streamgraph_data.csv")
else:
    print(f"Skipping Plot {plot_counter}: Streamgraph: Stream data is empty.")

# 13. Geographic Heatmap (Choropleth Map with Custom Tiling): Visualizing data intensity across geographical regions.
# This example simulates latitude/longitude for demonstration. For real use, replace with actual coordinates.
plot_counter += 1
print(f"Generating Plot {plot_counter}: Geographic Heatmap (Simulated)...")
# Add simulated latitude and longitude for demonstration purposes.
# These ranges approximate India's geographical boundaries.
# In a real scenario, you would map 'location' or 'location_monitoring_station' to actual coordinates.
df_geo = df_melted.copy()
# Ensure unique locations get consistent lat/lon if possible, or just assign randomly for demo
unique_locations = df_geo['location'].unique()
# Use tighter ranges to ensure points generally appear within India's landmass
location_lat_map = {loc: np.random.uniform(9.0, 34.0) for loc in unique_locations} # Latitudes for India
location_lon_map = {loc: np.random.uniform(69.0, 96.0) for loc in unique_locations} # Longitudes for India

df_geo["lat"] = df_geo["location"].map(location_lat_map)
df_geo["lon"] = df_geo["location"].map(location_lon_map)

# Drop rows where lat/lon might be NaN if location was NaN
df_geo.dropna(subset=['lat', 'lon', 'value'], inplace=True)

if not df_geo.empty:
    fig13 = px.density_mapbox(
        df_geo,
        lat="lat",
        lon="lon",
        z="value", # Color intensity by pollutant value
        radius=10, # Radius of the density circles
        center=dict(lat=22.5, lon=80), # Center of the map (approximate center of India)
        zoom=3, # Zoom level
        mapbox_style="stamen-terrain", # Map style (e.g., "open-street-map", "carto-positron", "stamen-terrain")
        title="Geographic Heatmap: Pollutant Value Density (Simulated Locations)"
    )
    fig13.write_html(f"plots/plot_{plot_counter}_geo_heatmap.html")
    df_geo[['lat', 'lon', 'value', 'location']].to_csv(f"plot_data/Plot_{plot_counter}_geo_heatmap_data.csv", index=False)
    print(f"Plot {plot_counter}: Geographic Heatmap saved to plots/plot_{plot_counter}_geo_heatmap.html and data to plot_data/Plot_{plot_counter}_geo_heatmap_data.csv")
else:
    print(f"Skipping Plot {plot_counter}: Geographic Heatmap: Insufficient data for plotting or missing lat/lon.")

# 14. Word Cloud with Collocations/Sentiment: Visualizes frequently occurring words.
# Here, it shows common words in the 'location' column, with collocations.
plot_counter += 1
print(f"Generating Plot {plot_counter}: Word Cloud...")
# Ensure 'location' column is treated as strings for word cloud generation
text_data = " ".join(df["location"].astype(str).dropna().tolist())

if text_data:
    # collocations=True enables finding and displaying phrases that frequently appear together.
    wordcloud = WordCloud(
        width=1000, height=500,
        background_color="white",
        collocations=True, # Detect and display collocations
        min_font_size=10,
        colormap='viridis' # Color map for words
    ).generate(text_data)

    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off") # Hide axes
    plt.title("Word Cloud of Locations (with Collocations)", fontsize=16)
    plt.tight_layout(pad=0)
    plt.savefig(f"plots/plot_{plot_counter}_wordcloud.png", dpi=300) # Save with high DPI
    plt.close() # Close the plot to free memory
    pd.DataFrame({'text': [text_data]}).to_csv(f"plot_data/Plot_{plot_counter}_wordcloud_data.csv", index=False)
    print(f"Plot {plot_counter}: Word Cloud saved to plots/plot_{plot_counter}_wordcloud.png and data to plot_data/Plot_{plot_counter}_wordcloud_data.csv")
else:
    print(f"Skipping Plot {plot_counter}: Word Cloud: No text data available from 'location' column.")

# -----------------------------
# 📈 Advanced Plots for Location-Based Time Series Insights (10 Plots)
# -----------------------------
print("\n--- Generating Location-Based Time Series Insight Plots ---")

# Ensure 'date' column is available and valid for time series plots
if 'date' not in df.columns or df['date'].isnull().all():
    print("Skipping location-based time series plots: 'date' column is missing or entirely null.")
else:
    # Prepare data for time series plots: Aggregate by date, location, and pollutant
    # This ensures a consistent time index for time series operations.
    df_time_series = df_melted.copy()
    df_time_series['date'] = pd.to_datetime(df_time_series['date'], errors='coerce')
    df_time_series.dropna(subset=['date', 'location', 'pollutant', 'value'], inplace=True)
    df_time_series = df_time_series.sort_values(by=['date', 'location', 'pollutant'])
    print("Prepared time series data for location-based insights.")

    # Select a few representative locations for plots that might become too crowded
    # if all locations are plotted. Adjust `num_sample_locations` as needed.
    num_sample_locations = 5
    sample_locations = df_time_series['location'].value_counts().nlargest(num_sample_locations).index.tolist()
    df_sampled_locations = df_time_series[df_time_series['location'].isin(sample_locations)].copy()
    print(f"Selected {num_sample_locations} top locations for some time series plots: {sample_locations}")


    # 1. Line Plot (multi-series): Trend over time for each pollutant per city/state
    plot_counter += 1
    print(f"Generating Plot {plot_counter}: Multi-series Line Plot (State-Pollutant Trends)...")
    # Aggregate by date, state, and pollutant for overall trends
    state_pollutant_trends = df_time_series.groupby(['date', 'state', 'pollutant'])['value'].mean().reset_index()
    fig_line_multi = px.line(
        state_pollutant_trends,
        x='date',
        y='value',
        color='pollutant',
        line_dash='state', # Differentiate states with line styles
        title='Average Pollutant Concentration Over Time by State and Pollutant',
        labels={'value': 'Average Concentration', 'date': 'Date'},
        hover_data={'state': True, 'pollutant': True, 'value': ':.2f'}
    )
    fig_line_multi.write_html(f"plots/plot_{plot_counter}_line_plot_multi_series.html")
    state_pollutant_trends.to_csv(f"plot_data/Plot_{plot_counter}_line_plot_multi_series_data.csv", index=False)
    print(f"Plot {plot_counter}: Multi-series Line Plot saved to plots/plot_{plot_counter}_line_plot_multi_series.html and data to plot_data/Plot_{plot_counter}_line_plot_multi_series_data.csv")

    # 2. Rolling Average Plot: Smooth trends using 7/30 day rolling averages
    plot_counter += 1
    print(f"Generating Plot {plot_counter}: Rolling Average Plot (PM2.5 for a sample location)...")
    # Focus on PM2.5 for a single sample location for clarity
    if not df_sampled_locations.empty and 'pm2_5' in df_sampled_locations['pollutant'].unique():
        pm25_data = df_sampled_locations[df_sampled_locations['pollutant'] == 'pm2_5'].copy()
        if not pm25_data.empty:
            # Resample to daily frequency and fill missing with mean for rolling average calculation
            pm25_daily_avg = pm25_data.groupby(['date', 'location'])['value'].mean().unstack().ffill().bfill()
            if not pm25_daily_avg.empty:
                # Select one location to plot rolling average
                plot_location = pm25_daily_avg.columns[0] if not pm25_daily_avg.columns.empty else None
                if plot_location:
                    pm25_daily_avg[f'{plot_location}_7D_RA'] = pm25_daily_avg[plot_location].rolling(window=7, min_periods=1).mean()
                    pm25_daily_avg[f'{plot_location}_30D_RA'] = pm25_daily_avg[plot_location].rolling(window=30, min_periods=1).mean()

                    plt.figure(figsize=(14, 7))
                    plt.plot(pm25_daily_avg.index, pm25_daily_avg[plot_location], label=f'{plot_location} Daily PM2.5', alpha=0.6)
                    plt.plot(pm25_daily_avg.index, pm25_daily_avg[f'{plot_location}_7D_RA'], label=f'{plot_location} 7-Day Rolling Avg', color='orange')
                    plt.plot(pm25_daily_avg.index, pm25_daily_avg[f'{plot_location}_30D_RA'], label=f'{plot_location} 30-Day Rolling Avg', color='red')
                    plt.title(f'PM2.5 Rolling Averages for {plot_location}', fontsize=16)
                    plt.xlabel('Date', fontsize=12)
                    plt.ylabel('PM2.5 Concentration', fontsize=12)
                    plt.legend()
                    plt.grid(True, linestyle='--', alpha=0.7)
                    plt.tight_layout()
                    plt.savefig(f"plots/plot_{plot_counter}_rolling_average_pm25.png", dpi=300)
                    plt.close()
                    pm25_daily_avg[[plot_location, f'{plot_location}_7D_RA', f'{plot_location}_30D_RA']].to_csv(f"plot_data/Plot_{plot_counter}_rolling_average_pm25_data.csv")
                    print(f"Plot {plot_counter}: Rolling Average Plot saved to plots/plot_{plot_counter}_rolling_average_pm25.png and data to plot_data/Plot_{plot_counter}_rolling_average_pm25_data.csv")
                else:
                    print(f"Skipping Plot {plot_counter}: Rolling Average Plot: No valid location to plot.")
            else:
                print(f"Skipping Plot {plot_counter}: Rolling Average Plot: PM2.5 daily average data is empty.")
        else:
            print(f"Skipping Plot {plot_counter}: Rolling Average Plot: PM2.5 data for sampled locations is empty.")
    else:
        print(f"Skipping Plot {plot_counter}: Rolling Average Plot: No sampled locations or PM2.5 data available.")

    # 3. Seasonal Decomposition Plot (STL): Break time series into Trend + Seasonality + Residual
    plot_counter += 1
    print(f"Generating Plot {plot_counter}: Seasonal Decomposition Plot (STL) for a sample series...")
    # Select one state and one pollutant (e.g., 'Maharashtra' and 'rspm') for decomposition
    sample_series_data = df_time_series[(df_time_series['state'] == 'Maharashtra') &
                                        (df_time_series['pollutant'] == 'rspm')].copy()
    if not sample_series_data.empty:
        # Resample to a monthly frequency for STL decomposition
        # Changed 'M' to 'ME' for month end frequency to avoid FutureWarning
        sample_series_monthly = sample_series_data.set_index('date')['value'].resample('ME').mean().fillna(method='ffill').fillna(method='bfill')

        if len(sample_series_monthly) > 24: # STL requires at least two full cycles (e.g., 2 years for monthly data)
            try:
                # Use STL decomposition
                stl = STL(sample_series_monthly, seasonal=13, period=12, robust=True) # seasonal=13 for odd number, period=12 for monthly
                res = stl.fit()

                fig_stl = res.plot()
                fig_stl.set_size_inches(12, 8)
                fig_stl.suptitle('STL Decomposition of RSPM in Maharashtra (Monthly)', fontsize=16, y=1.02)
                plt.tight_layout(rect=[0, 0.03, 1, 0.98])
                plt.savefig(f"plots/plot_{plot_counter}_stl_decomposition_rspm.png", dpi=300)
                plt.close(fig_stl)
                pd.DataFrame({
                    'observed': res.observed,
                    'trend': res.trend,
                    'seasonal': res.seasonal,
                    'resid': res.resid
                }).to_csv(f"plot_data/Plot_{plot_counter}_stl_decomposition_rspm_data.csv")
                print(f"Plot {plot_counter}: Seasonal Decomposition Plot saved to plots/plot_{plot_counter}_stl_decomposition_rspm.png and data to plot_data/Plot_{plot_counter}_stl_decomposition_rspm_data.csv")
            except Exception as e:
                print(f"Skipping Plot {plot_counter}: STL Decomposition Plot: Error during decomposition - {e}")
        else:
            print(f"Skipping Plot {plot_counter}: STL Decomposition Plot: Not enough data points for meaningful decomposition (less than 2 years of monthly data).")
    else:
        print(f"Skipping Plot {plot_counter}: STL Decomposition Plot: No data for 'Maharashtra' and 'rspm'.")

    # 4. Heatmap (Month × Year): Heatmap of average pollutant levels by month-year
    plot_counter += 1
    print(f"Generating Plot {plot_counter}: Heatmap (Month x Year) for PM2.5...")
    # Aggregate PM2.5 by year and month
    pm25_monthly_avg = df_time_series[df_time_series['pollutant'] == 'pm2_5'].groupby(['year', 'month'])['value'].mean().unstack(fill_value=0)
    if not pm25_monthly_avg.empty:
        plt.figure(figsize=(12, 8))
        sns.heatmap(pm25_monthly_avg, cmap='viridis', annot=True, fmt=".1f", linewidths=.5, cbar_kws={'label': 'Average PM2.5 Concentration'})
        plt.title('Average PM2.5 Concentration by Month and Year', fontsize=16)
        plt.xlabel('Month', fontsize=12)
        plt.ylabel('Year', fontsize=12)
        plt.tight_layout()
        plt.savefig(f"plots/plot_{plot_counter}_heatmap_month_year_pm25.png", dpi=300)
        plt.close()
        pm25_monthly_avg.to_csv(f"plot_data/Plot_{plot_counter}_heatmap_month_year_pm25_data.csv")
        print(f"Plot {plot_counter}: Heatmap (Month x Year) saved to plots/plot_{plot_counter}_heatmap_month_year_pm25.png and data to plot_data/Plot_{plot_counter}_heatmap_month_year_pm25_data.csv")
    else:
        print(f"Skipping Plot {plot_counter}: Heatmap (Month x Year): No PM2.5 data available for aggregation.")

    # 5. Facet Line Plot (One panel per location): Small multiples by city/state
    plot_counter += 1
    print(f"Generating Plot {plot_counter}: Facet Line Plot (Pollutant Trends by Sampled Location)...")
    if not df_sampled_locations.empty:
        g_facet = sns.relplot(
            data=df_sampled_locations,
            x="date",
            y="value",
            col="location", # Create separate panels for each location
            hue="pollutant", # Color lines by pollutant
            kind="line",
            col_wrap=2, # Wrap columns after 2 plots
            height=4, aspect=1.5,
            facet_kws={'sharex': True, 'sharey': False} # Share x-axis, but allow independent y-scales
        )
        g_facet.set_titles("Location: {col_name}")
        g_facet.set_axis_labels("Date", "Pollutant Value")
        plt.suptitle("Pollutant Trends by Sampled Location", y=1.02, fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        plt.savefig(f"plots/plot_{plot_counter}_facet_line_plot_locations.png", dpi=300)
        plt.close()
        df_sampled_locations[['date', 'location', 'pollutant', 'value']].to_csv(f"plot_data/Plot_{plot_counter}_facet_line_plot_locations_data.csv", index=False)
        print(f"Plot {plot_counter}: Facet Line Plot saved to plots/plot_{plot_counter}_facet_line_plot_locations.png and data to plot_data/Plot_{plot_counter}_facet_line_plot_locations_data.csv")
    else:
        print(f"Skipping Plot {plot_counter}: Facet Line Plot: No sampled locations data available.")

    # 6. Animated Scatter/Map Plot (Time axis): Animated bubble map of pollutant spread
    plot_counter += 1
    print(f"Generating Plot {plot_counter}: Animated Scatter/Map Plot (PM2.5 by Year, Simulated Lat/Lon)...")
    df_animated_geo = df_time_series.groupby(['year', 'location'])['value'].mean().reset_index()
    df_animated_geo = df_animated_geo[df_animated_geo['location'].isin(sample_locations)].copy() # Use sampled locations
    df_animated_geo = df_animated_geo.dropna(subset=['location', 'value'])

    # Add simulated lat/lon for the sampled locations for consistency with plot 13
    # Use tighter ranges to ensure points generally appear within India's landmass
    unique_sampled_locations = df_animated_geo['location'].unique()
    sampled_location_lat_map = {loc: np.random.uniform(9.0, 34.0) for loc in unique_sampled_locations} # Latitudes for India
    sampled_location_lon_map = {loc: np.random.uniform(69.0, 96.0) for loc in unique_sampled_locations} # Longitudes for India

    df_animated_geo["lat"] = df_animated_geo["location"].map(sampled_location_lat_map)
    df_animated_geo["lon"] = df_animated_geo["location"].map(sampled_location_lon_map)
    df_animated_geo.dropna(subset=['lat', 'lon'], inplace=True) # Drop if any lat/lon mapping failed

    if not df_animated_geo.empty:
        fig_animated_map = px.scatter_mapbox(
            df_animated_geo,
            lat="lat",
            lon="lon",
            color="value", # Color by pollutant value
            size="value", # Size by pollutant value
            animation_frame="year", # Animate over years
            animation_group="location", # Group animation by location
            hover_name="location",
            mapbox_style="carto-positron", # Or "open-street-map", "stamen-terrain"
            zoom=3,
            center={"lat": 22.5, "lon": 80},
            title="Animated Pollutant Concentration by Location Over Years (Simulated)"
        )
        fig_animated_map.update_layout(transition = {'duration': 5000}) # Slow down animation
        fig_animated_map.write_html(f"plots/plot_{plot_counter}_animated_map_plot.html")
        df_animated_geo[['year', 'location', 'value', 'lat', 'lon']].to_csv(f"plot_data/Plot_{plot_counter}_animated_map_plot_data.csv", index=False)
        print(f"Plot {plot_counter}: Animated Scatter/Map Plot saved to plots/plot_{plot_counter}_animated_map_plot.html and data to plot_data/Plot_{plot_counter}_animated_map_plot_data.csv")
    else:
        print(f"Skipping Plot {plot_counter}: Animated Scatter/Map Plot: Insufficient data or missing lat/lon for sampled locations.")

    # 7. Correlation Over Time Plot: Rolling correlation between pollutants
    plot_counter += 1
    print(f"Generating Plot {plot_counter}: Rolling Correlation Plot (SO2 vs NO2, State-level)...")
    # Focus on two pollutants (e.g., SO2 and NO2) and aggregate monthly by state
    corr_data = df_time_series[df_time_series['pollutant'].isin(['so2', 'no2'])].copy()
    if not corr_data.empty:
        corr_pivot = corr_data.pivot_table(index=['date', 'state'], columns='pollutant', values='value').reset_index()
        corr_pivot.set_index('date', inplace=True)

        # Calculate rolling correlation for each state
        rolling_correlations = []
        for state in corr_pivot['state'].unique():
            state_df = corr_pivot[corr_pivot['state'] == state].dropna(subset=['so2', 'no2'])
            if len(state_df) > 30: # Need enough data points for rolling correlation
                state_df['rolling_corr'] = state_df['so2'].rolling(window=30, min_periods=10).corr(state_df['no2'])
                state_df['state'] = state # Add state back after rolling
                rolling_correlations.append(state_df.reset_index())

        if rolling_correlations:
            rolling_corr_df = pd.concat(rolling_correlations)
            fig_rolling_corr = px.line(
                rolling_corr_df.dropna(subset=['rolling_corr']),
                x='date',
                y='rolling_corr',
                color='state',
                title='Rolling Correlation (30-Day) between SO2 and NO2 by State',
                labels={'rolling_corr': 'Correlation Coefficient', 'date': 'Date'},
                hover_data={'state': True, 'rolling_corr': ':.2f'}
            )
            fig_rolling_corr.write_html(f"plots/plot_{plot_counter}_rolling_correlation_plot.html")
            rolling_corr_df.to_csv(f"plot_data/Plot_{plot_counter}_rolling_correlation_data.csv", index=False)
            print(f"Plot {plot_counter}: Rolling Correlation Plot saved to plots/plot_{plot_counter}_rolling_correlation_plot.html and data to plot_data/Plot_{plot_counter}_rolling_correlation_data.csv")
        else:
            print(f"Skipping Plot {plot_counter}: Rolling Correlation Plot: Not enough data for rolling correlation for any state.")
    else:
        print(f"Skipping Plot {plot_counter}: Rolling Correlation Plot: No SO2 or NO2 data available.")

    # 8. Interactive Time Series (Plotly): Hoverable, zoomable pollutant trends
    plot_counter += 1
    print(f"Generating Plot {plot_counter}: Interactive Time Series Plot (PM2.5 for Sampled Locations)...")
    if not df_sampled_locations.empty and 'pm2_5' in df_sampled_locations['pollutant'].unique():
        interactive_pm25 = df_sampled_locations[df_sampled_locations['pollutant'] == 'pm2_5'].copy()
        if not interactive_pm25.empty:
            fig_interactive = px.line(
                interactive_pm25,
                x='date',
                y='value',
                color='location',
                title='Interactive PM2.5 Concentration Over Time by Location',
                labels={'value': 'PM2.5 Concentration', 'date': 'Date'},
                hover_data={'state': True, 'location': True, 'value': ':.2f'}
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
            fig_interactive.write_html(f"plots/plot_{plot_counter}_interactive_time_series.html")
            interactive_pm25.to_csv(f"plot_data/Plot_{plot_counter}_interactive_time_series_data.csv", index=False)
            print(f"Plot {plot_counter}: Interactive Time Series Plot saved to plots/plot_{plot_counter}_interactive_time_series.html and data to plot_data/Plot_{plot_counter}_interactive_time_series_data.csv")
        else:
            print(f"Skipping Plot {plot_counter}: Interactive Time Series Plot: No PM2.5 data for sampled locations.")
    else:
        print(f"Skipping Plot {plot_counter}: Interactive Time Series Plot: No sampled locations or PM2.5 data available.")

    # 9. Pollution Spike Detector (Z-score or Threshold): Highlight anomalous spikes in pollution
    plot_counter += 1
    print(f"Generating Plot {plot_counter}: Pollution Spike Detector Plot (RSPM, Sample Location)...")
    if not df_sampled_locations.empty and 'rspm' in df_sampled_locations['pollutant'].unique():
        rspm_spike_data = df_sampled_locations[df_sampled_locations['pollutant'] == 'rspm'].copy()
        if not rspm_spike_data.empty:
            # Select one location for spike detection
            spike_location = rspm_spike_data['location'].iloc[0] if not rspm_spike_data['location'].empty else None
            if spike_location:
                location_rspm = rspm_spike_data[rspm_spike_data['location'] == spike_location].set_index('date')['value'].sort_index()
                # Calculate Z-score
                mean_rspm = location_rspm.mean()
                std_rspm = location_rspm.std()
                if std_rspm > 0: # Avoid division by zero
                    location_rspm_zscore = (location_rspm - mean_rspm) / std_rspm
                    # Define a threshold for spikes (e.g., Z-score > 3)
                    spike_threshold = 3
                    spikes = location_rspm[location_rspm_zscore.abs() > spike_threshold]

                    plt.figure(figsize=(14, 7))
                    plt.plot(location_rspm.index, location_rspm.values, label=f'{spike_location} RSPM', color='blue', alpha=0.7)
                    plt.scatter(spikes.index, spikes.values, color='red', s=50, zorder=5, label=f'Spikes (Z > {spike_threshold})')
                    plt.title(f'RSPM Pollution Spikes for {spike_location}', fontsize=16)
                    plt.xlabel('Date', fontsize=12)
                    plt.ylabel('RSPM Concentration', fontsize=12)
                    plt.legend()
                    plt.grid(True, linestyle='--', alpha=0.7)
                    plt.tight_layout()
                    plt.savefig(f"plots/plot_{plot_counter}_pollution_spike_detector.png", dpi=300)
                    plt.close()
                    pd.DataFrame({'date': location_rspm.index, 'value': location_rspm.values, 'is_spike': location_rspm_zscore.abs() > spike_threshold}).to_csv(f"plot_data/Plot_{plot_counter}_pollution_spike_detector_data.csv", index=False)
                    print(f"Plot {plot_counter}: Pollution Spike Detector Plot saved to plots/plot_{plot_counter}_pollution_spike_detector.png and data to plot_data/Plot_{plot_counter}_pollution_spike_detector_data.csv")
                else:
                    print(f"Skipping Plot {plot_counter}: Pollution Spike Detector Plot: Standard deviation for RSPM in {spike_location} is zero.")
            else:
                print(f"Skipping Plot {plot_counter}: Pollution Spike Detector Plot: No valid location to plot.")
        else:
            print(f"Skipping Plot {plot_counter}: Pollution Spike Detector Plot: No RSPM data for sampled locations.")
    else:
        print(f"Skipping Plot {plot_counter}: Pollution Spike Detector Plot: No sampled locations or RSPM data available.")

    # 10. Clustering with Time Series (KMeans): Cluster cities based on pollutant trends
    plot_counter += 1
    print(f"Generating Plot {plot_counter}: Time Series Clustering Plot (KMeans on State-Pollutant Monthly Averages)...")
    # Aggregate monthly average for each state-pollutant combination
    # Changed 'M' to 'ME' for month end frequency to avoid FutureWarning
    ts_cluster_data = df_time_series.groupby(['state', 'pollutant', pd.Grouper(key='date', freq='ME')])['value'].mean().unstack(level='date')
    ts_cluster_data.fillna(0, inplace=True) # Fill NaNs for clustering

    # Flatten the time series for clustering (state-pollutant as rows, monthly values as columns)
    ts_cluster_flat = ts_cluster_data.stack().unstack(level=[0, 1]) # Unstack to get (state, pollutant) as columns
    ts_cluster_flat = ts_cluster_flat.T # Transpose to get (state, pollutant) as rows for clustering

    if not ts_cluster_flat.empty and ts_cluster_flat.shape[0] > 1: # Need at least 2 samples to cluster
        # Scale the data before clustering
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(ts_cluster_flat)

        # Determine optimal number of clusters (e.g., using Elbow Method - not implemented here, but a good next step)
        n_clusters = min(5, scaled_features.shape[0]) # Choose a reasonable number of clusters, max 5

        if n_clusters > 1:
            # UserWarning: KMeans is known to have a memory leak on Windows with MKL,
            # when there are less chunks than available threads. You can avoid it by
            # setting the environment variable OMP_NUM_THREADS=1.
            # This warning is noted but not directly addressed in code as it's an environment setting.
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) # n_init for robust centroid initialization
            ts_cluster_flat['cluster'] = kmeans.fit_predict(scaled_features)

            # Visualize clusters (e.g., plot average trend for each cluster)
            # Step 1: Reset the MultiIndex of ts_cluster_flat to make 'state' and 'pollutant' regular columns.
            df_clustered_long = ts_cluster_flat.reset_index()

            # Step 2: Melt the date columns to create a 'date' column and a 'value' column.
            # Get the list of date columns dynamically from the original ts_cluster_flat columns (before adding 'cluster')
            # These are the columns that were the DatetimeIndex before transposing.
            # The columns of df_clustered_long are now 'state', 'pollutant', date1, date2, ..., 'cluster'
            date_cols_for_melt = [col for col in df_clustered_long.columns if isinstance(col, pd.Timestamp)]

            df_clustered_melted = df_clustered_long.melt(
                id_vars=['state', 'pollutant', 'cluster'],
                value_vars=date_cols_for_melt,
                var_name='date',
                value_name='value'
            )

            # Step 3: Group by cluster and date to get the average trend for each cluster over time.
            cluster_trends_for_plot = df_clustered_melted.groupby(['cluster', 'date'])['value'].mean().reset_index()

            fig_cluster = px.line(
                cluster_trends_for_plot,
                x='date',
                y='value',
                color='cluster',
                title='Average Pollutant Trends for Each Cluster (Monthly Averages)',
                labels={'value': 'Average Pollutant Value', 'date': 'Date'}
            )
            fig_cluster.write_html(f"plots/plot_{plot_counter}_time_series_clustering.html")
            cluster_trends_for_plot.to_csv(f"plot_data/Plot_{plot_counter}_time_series_clustering_data.csv", index=False)
            print(f"Plot {plot_counter}: Time Series Clustering Plot saved to plots/plot_{plot_counter}_time_series_clustering.html and data to plot_data/Plot_{plot_counter}_time_series_clustering_data.csv")
        else:
            print(f"Skipping Plot {plot_counter}: Time Series Clustering Plot: Not enough data points to form more than one cluster.")
    else:
        print(f"Skipping Plot {plot_counter}: Time Series Clustering Plot: Insufficient data for clustering time series.")

print("\n--- EDA Script Finished ---")
