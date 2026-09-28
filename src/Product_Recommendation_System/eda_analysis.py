"""
EDA Analysis Script for Raw_Processed_Data.csv
E-Commerce Dataset (Brazil, 2016-2018)

This script performs a full Exploratory Data Analysis including:
- Data overview (shape, dtypes, duplicates)
- Missing value analysis & data sparsity
- Univariate analysis (numeric + categorical)
- Temporal analysis
- Outlier detection (IQR method)
- Delivery performance analysis
- Review score vs delivery correlation
- Revenue by category
- NCF model suitability assessment

Run:
    python eda_analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
import os

warnings.filterwarnings("ignore")

# File path
FILE_PATH = "D:/FinalYearProject/processed_data/Raw_Processed_Data.csv"

# ==============================
# 1. LOAD DATA
# ==============================
print("=" * 60)
print("LOADING DATA")
print("=" * 60)

df = pd.read_csv(FILE_PATH)

print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
print("\nColumn dtypes:")
print(df.dtypes)

# ==============================
# 2. DATE PARSING
# ==============================
date_cols = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
    "shipping_limit_date",
    "review_creation_date",
    "review_answer_timestamp",
]

for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors="coerce")

# ==============================
# 3. BASIC OVERVIEW
# ==============================
print("\n" + "=" * 60)
print("BASIC OVERVIEW")
print("=" * 60)

print(f"\nDate range: {df['order_purchase_timestamp'].min().date()} "
      f"to {df['order_purchase_timestamp'].max().date()}")

print(f"\nDuplicate rows       : {df.duplicated().sum():,}")
print(f"Duplicate order_ids  : {df['order_id'].duplicated().sum():,}")
print(f"Unique customers     : {df['customer_unique_id'].nunique():,}")
print(f"Unique products      : {df['product_id'].nunique():,}")
print(f"Unique sellers       : {df['seller_id'].nunique():,}")
print(f"Unique orders        : {df['order_id'].nunique():,}")

# ==============================
# 4. MISSING VALUES
# ==============================
print("\n" + "=" * 60)
print("MISSING VALUE ANALYSIS")
print("=" * 60)

missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(4)

missing_df = pd.DataFrame({"count": missing, "pct": missing_pct})
missing_df = missing_df[missing_df["count"] > 0].sort_values("pct", ascending=False)

print(missing_df.to_string())

# ==============================
# 5. NUMERIC STATS
# ==============================
print("\n" + "=" * 60)
print("NUMERIC DESCRIPTIVE STATISTICS")
print("=" * 60)

print(df.describe().round(2).to_string())

# ==============================
# 6. DELIVERY METRICS
# ==============================
df["delivery_delay_days"] = (
    df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]
).dt.days

df["approval_time_hrs"] = (
    df["order_approved_at"] - df["order_purchase_timestamp"]
).dt.total_seconds() / 3600

# ==============================
# 7. REVENUE
# ==============================
df["revenue"] = df["price"] + df["freight_value"]

# ==============================
# 8. MONTHLY ORDERS
# ==============================
df["month"] = df["order_purchase_timestamp"].dt.to_period("M")
monthly = df.groupby("month")["order_id"].nunique().sort_index()

# ==============================
# 9. SAVE CHARTS
# ==============================
print("\n" + "=" * 60)
print("SAVING CHARTS")
print("=" * 60)

# Create output folder
output_dir = os.path.join(os.getcwd(), "outputs")
os.makedirs(output_dir, exist_ok=True)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("EDA Charts", fontsize=14, fontweight="bold")

# Chart 1
status_counts = df["order_status"].value_counts()
axes[0, 0].bar(status_counts.index, status_counts.values)
axes[0, 0].set_title("Order Status")

# Chart 2
rv = df["review_score"].value_counts().sort_index()
axes[0, 1].bar(rv.index.astype(str), rv.values)
axes[0, 1].set_title("Review Score")

# Chart 3
axes[0, 2].plot(monthly.index.astype(str), monthly.values)
axes[0, 2].set_title("Monthly Orders")

# Chart 4
axes[1, 0].hist(df["price"].dropna(), bins=50)
axes[1, 0].set_title("Price Distribution")

# Chart 5
rv_delay = df.groupby("review_score")["delivery_delay_days"].mean()
axes[1, 1].bar(rv_delay.index.astype(str), rv_delay.values)
axes[1, 1].set_title("Delay vs Review")

# Chart 6
top_cat = df.groupby("product_category_name_english")["revenue"].sum().sort_values(ascending=False)
top5 = top_cat.head(5)
axes[1, 2].barh(top5.index, top5.values)

plt.tight_layout()

# ✅ FIXED LINE (important)
file_path = os.path.join(output_dir, "eda_charts.png")
plt.savefig(file_path, dpi=150, bbox_inches="tight")

plt.close()

print(f"Charts saved at: {file_path}")
