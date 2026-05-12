# 🛒 2025 Dongwon x KAIST AI Competition

## Next-Gen Sales Demand Forecasting via LLM-driven Persona Simulation

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Competition](https://img.shields.io/badge/Status-Completed-orange.svg)

This repository contains the project for the **'2025 Dongwon x KAIST AI Competition'**. The project focuses on predicting monthly sales volume for unreleased products by simulating a synthetic market using LLMs and stochastic modeling.

---

## 🚀 Executive Summary (TL;DR)
- **The Problem**: Predicting monthly sales volume for unreleased products with zero historical sales (Cold-Start Problem).
- **The Solution**: Developed a Synthetic Market Research framework simulating 12 months of sales behavior using LLM-generated personas and Monte Carlo simulation.
- **The Result**: Successfully captured seasonal trends and marketing sensitivity, providing a scalable way to run "What-If" scenarios before launch.

---

## 🛠️ Tech Stack (기술 스택)
- **Core Logic**: Python
- **Generative AI**: Google Gemini API
- **Machine Learning**: LightGBM
- **Simulation**: Monte Carlo Method
- **Data Processing**: Pandas, NumPy

---

## 📌 1. Problem Definition (문제 정의)
- **Background**: Traditional demand forecasting fails for new products with zero historical sales.
- **Objective**: To solve the "cold-start" problem in demand forecasting by creating a simulation engine that mimics real consumer behavior.
- **Vision**: "Empowering product managers with data-driven 'What-If' scenario testing before launch."

---

## 🔬 2. Methodology & Architecture (방법론 및 아키텍처)
Our approach integrates Generative AI, Machine Learning, and Stochastic Simulation into a unified pipeline:

### 2.1 System Architecture
```mermaid
graph TD
    A[Product Info CSV] --> B(LLM Adapter)
    B -->|Generate| C[Persona Cache]
    A --> D(Naver Price Search)
    D --> E[Category Prices CSV]
    E --> F(LightGBM Trainer)
    F -->|Model| G[Price Model PKL]
    
    C --> H(Simulation Engine)
    G --> H
    A --> H
    
    H -->|Monte Carlo Ensemble| I[Final Demand Forecast]
    I --> J[Submission CSV]
```

### 2.2 Core Components
- **LLM Persona Engine**: Generates highly specific consumer segments (Age, Income, Lifestyle, Channel Preference) with unique utility functions. Uses a SHA-1 signature-based caching system to guarantee reproducibility.
- **Price Intelligence (ML)**: Scrapes competitor prices via Naver API and trains a LightGBM Regressor to predict optimal list prices. *(Note: Trained on a small sample of 12 rows for PoC; pipeline is designed to scale.)*
- **Monte Carlo Simulation**: Simulates individual purchase decisions across 12 months, factoring in Advertising GRPs, Distribution (ACV), and Seasonal trends.

---

## 📊 3. Results & Impact (결과 및 임팩트)
- **Accurate Cold-Start Forecasting**: Successfully generated a 12-month demand curve for products with no prior sales history.
- **Behavioral Realism**: Captured the complex interplay between marketing spend (Ad GRPs), distribution ramp-up, and consumer price sensitivity.
- **Business Application**: This framework allows Dongwon to run scenarios (e.g., "What if we increase price by 10% and double ad spend?") before actual product launch.

### ⚠️ Technical Notes & Limitations
- **Model Performance**: The LightGBM model was trained on a sample of 12 rows (yielding R² ~0.0). This serves as a structural Proof-of-Concept. For production use, the scraper should be run at scale to gather larger training datasets.
- **Simulation Scale**: The pipeline defaults to 400 personas for robust simulation, though cached results may show smaller batches used during testing to optimize API costs.

---

## 📁 Repository Structure
```text
├── notebooks/                  # Jupyter notebooks for experimentation
│   ├── 01_experiments_pipeline.ipynb
│   └── 02_final_demand_forecast.ipynb
├── data/                       # Input data files
│   ├── category_prices.csv     # Competitor prices gathered from Naver
│   └── product_info.csv        # Product specifications for forecasting
├── results/                    # Output files and reports
│   ├── monthly_forecast.csv
│   └── submission_3.csv        # Final submission file
├── forecast_pipeline.py        # Master pipeline script (Gather prices, Train, Predict)
├── simulation_pipeline.py      # Simulation logic
└── README.md                   # Project documentation
```

---

## ⚙️ 4. Reproducibility (재현성)
The entire pipeline is modularized into `forecast_pipeline.py`. Follow these steps to reproduce the results.

### 4.1 Gather Competitor Prices
```bash
python forecast_pipeline.py gather_prices --product_csv product_info.csv
```

### 4.2 Train Price Prediction Model
```bash
python forecast_pipeline.py train_price_model --prices_csv category_prices.csv
```

### 4.3 Run Simulation & Generate Submission
```bash
python forecast_pipeline.py make_submission --product_csv product_info.csv --use_llm 0 --mc_runs 7
```

---
## 👥 Contributors
- **Junhyung L.** (Project Lead)

---
*Refactored and polished to meet professional software engineering standards for the [Data Analyst Portfolio](https://github.com/junhyung-L/Resume/blob/main/Portfolio/README.md).*
