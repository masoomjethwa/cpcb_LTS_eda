---

```markdown
# 🧪 CPCB Air Quality Dashboard (1990–2015)

This interactive **Streamlit** dashboard enables in-depth **Exploratory Data Analysis (EDA)** and visualization of historical air quality data from the Central Pollution Control Board (CPCB) spanning 1990 to 2015. It's built to help users identify trends, patterns, anomalies, and insights across Indian cities and states.

---

## 🚀 Features

The dashboard includes a comprehensive suite of data exploration tools:

### 🎛️ Interactive Filters
- Filter by **State**, **City (Location)**, **Pollutant**, and **Date Range**

### 📊 KPI Metrics
- View **Mean**, **Max**, **Min**, and **Standard Deviation** for each pollutant

### 📈 Time Series Analysis
- Line charts for pollutants over time
- Optional **30-day rolling averages**
- Compare multiple locations
- **Z-score based anomaly detection**
- **STL decomposition** into trend, seasonality, and residual
- **ARIMA-based forecasting**

### 📍 Advanced EDA Visualizations
- Distribution plots: Histograms, Boxplots
- Relationship maps: Correlation heatmaps, Sankey diagrams, Network graphs
- **Hierarchical exploration**: Treemap, Sunburst
- **Multi-dimensional** views: Parallel Coordinates, Violin + Swarm, Ridgeline (Joyplots)
- **Geospatial patterns** (simulated with grouped heatmaps)
- **Hexbin density plots**, **Streamgraphs**, **Dendrograms**
- Word clouds of locations for qualitative insights

### 🧪 Time Series Clustering
- Group states/cities based on pollutant trends using **KMeans clustering**

### 💾 Export Options
- Download filtered data as **CSV**
- Save plots as **PNG** or **interactive HTML**

---

## 📂 Data Source

The dashboard uses:

**`CPCB_data_from1990_2015.csv`**

Make sure this dataset is placed in the same folder as the Streamlit script. This file should contain the following columns:
```

stn\_code, sampling\_date, state, location, agency, type, so2, no2, rspm, spm, location\_monitoring\_station, pm2\_5, date

````

---

## 📚 Understanding the Tools

### 🔍 What is EDA?

**Exploratory Data Analysis (EDA)** is the systematic process of:
- Understanding data distributions and outliers
- Revealing hidden trends and anomalies
- Testing hypotheses visually
- Cleaning and preparing data for modeling

In this project, EDA helps:
- Track air pollution across time and geography
- Spot seasonal or unusual pollutant events
- Identify regions at environmental risk

### 🌐 Why Streamlit?

**Streamlit** is a modern, lightweight Python framework that:
- Turns scripts into shareable web apps
- Requires **no HTML or JS**
- Integrates directly with **Pandas, Plotly, Matplotlib, Seaborn**, etc.
- Is ideal for **data scientists and analysts**

---

## 🛠️ How to Run the App

### Step 1: Save the Code

Save the main script as:
```bash
streamlit_app.py
````

### Step 2: Prepare the Data

Place `CPCB_data_from1990_2015.csv` in the same folder.

### Step 3: Install Dependencies

Use the following command in your terminal (inside your project or virtual environment):

```bash
pip install streamlit pandas numpy plotly seaborn matplotlib statsmodels scikit-learn wordcloud networkx joypy scipy
```

### Step 4: Run the Dashboard

Launch the app with:

```bash
streamlit run streamlit_app.py
```

Then open the app in your web browser (default: [http://localhost:8501](http://localhost:8501)).

---

## 📦 Dependencies

* `streamlit` – Web framework
* `pandas`, `numpy` – Data manipulation
* `matplotlib`, `seaborn`, `plotly` – Visualizations
* `statsmodels` – Time series decomposition & ARIMA
* `scikit-learn` – Clustering (KMeans)
* `wordcloud` – Text-based insights
* `networkx` – Graph-based visualizations
* `joypy` – Ridgeline plots
* `scipy` – Stats and signal processing

---

## ✅ Recommended Project Structure

```
📁 your_project/
├── streamlit_app.py
├── CPCB_data_from1990_2015.csv
├── README.md
├── 📁 plots/         # Optional: Save exported plots here
├── 📁 output/        # Optional: Save transformed data here
└── requirements.txt
```

---

## 💡 Tip: Create a `requirements.txt` for Sharing

```bash
pip freeze > requirements.txt
```

---

## 📬 Feedback or Improvements?

Feel free to fork this project or raise issues for suggestions, improvements, or feature requests. Built to support data-driven environmental awareness.

---

> ✨ Designed for analysts, researchers, and policymakers exploring India's air quality evolution over 25 years.

```

---

Get `requirements.txt` file. 

```