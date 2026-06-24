import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from scipy.stats import skew, kurtosis, zscore

# =====================================================
# CREATE OUTPUT FOLDER
# =====================================================

os.makedirs("outputs", exist_ok=True)

# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_csv("industry_dataset.csv")

print("=" * 60)
print("DATASET SHAPE")
print("=" * 60)
print(df.shape)

print("\nFIRST 5 RECORDS")
print(df.head())

print("\nDATASET INFO")
print(df.info())

print("\nMISSING VALUES")
print(df.isnull().sum())

# =====================================================
# DATE CONVERSION
# =====================================================

df["created_date"] = pd.to_datetime(
    df["created_date"],
    format="%d-%m-%Y"
)

# =====================================================
# NUMERICAL & CATEGORICAL COLUMNS
# =====================================================

num_cols = [
    "employee_count",
    "annual_revenue_million",
    "profit_margin_percent",
    "founded_year",
    "customer_count",
    "market_rating"
]

cat_cols = [
    "industry",
    "country",
    "region"
]

# =====================================================
# UNIVARIATE ANALYSIS
# =====================================================

print("\n")
print("=" * 60)
print("UNIVARIATE ANALYSIS")
print("=" * 60)

for col in num_cols:

    print(f"\n----- {col} -----")

    print("Mean:", round(df[col].mean(), 2))
    print("Median:", round(df[col].median(), 2))
    print("Std Dev:", round(df[col].std(), 2))
    print("Minimum:", df[col].min())
    print("Maximum:", df[col].max())
    print("Skewness:", round(skew(df[col]), 4))
    print("Kurtosis:", round(kurtosis(df[col]), 4))

# =====================================================
# HISTOGRAMS
# =====================================================

for col in num_cols:

    plt.figure(figsize=(8, 5))

    sns.histplot(df[col], kde=True)

    plt.title(f"Distribution of {col}")

    plt.tight_layout()

    plt.savefig(f"outputs/{col}_histogram.png")

    plt.close()

print("\nHistogram plots saved successfully.")

# =====================================================
# CATEGORICAL ANALYSIS
# =====================================================

print("\n")
print("=" * 60)
print("CATEGORICAL ANALYSIS")
print("=" * 60)

for col in cat_cols:

    print(f"\nFrequency Distribution of {col}")
    print(df[col].value_counts())

    plt.figure(figsize=(8, 5))

    sns.countplot(
        x=df[col]
    )

    plt.title(col)

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig(f"outputs/{col}_countplot.png")

    plt.close()

print("\nCategorical plots saved successfully.")

# =====================================================
# OUTLIER DETECTION
# =====================================================

print("\n")
print("=" * 60)
print("OUTLIER DETECTION")
print("=" * 60)

for col in num_cols:

    plt.figure(figsize=(8, 4))

    sns.boxplot(
        x=df[col]
    )

    plt.title(f"Boxplot of {col}")

    plt.tight_layout()

    plt.savefig(f"outputs/{col}_boxplot.png")

    plt.close()

print("\nBoxplots saved successfully.")

# =====================================================
# IQR METHOD
# =====================================================

print("\nIQR OUTLIER DETECTION")

for col in num_cols:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[
        (df[col] < lower) |
        (df[col] > upper)
    ]

    print(f"{col}: {len(outliers)} outliers")

# =====================================================
# Z SCORE METHOD
# =====================================================

print("\nZ-SCORE OUTLIER DETECTION")

z_scores = np.abs(
    zscore(df[num_cols])
)

print((z_scores > 3).sum(axis=0))

# =====================================================
# CORRELATION MATRIX
# =====================================================

print("\n")
print("=" * 60)
print("CORRELATION MATRIX")
print("=" * 60)

corr = df[num_cols].corr()

print(corr)

# =====================================================
# CORRELATION HEATMAP
# =====================================================

plt.figure(figsize=(10, 6))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig("outputs/correlation_heatmap.png")

plt.close()

# =====================================================
# EMPLOYEE VS REVENUE
# =====================================================

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="employee_count",
    y="annual_revenue_million"
)

plt.title("Employee Count vs Revenue")

plt.tight_layout()

plt.savefig("outputs/employee_vs_revenue.png")

plt.close()

# =====================================================
# CUSTOMER VS REVENUE
# =====================================================

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="customer_count",
    y="annual_revenue_million"
)

plt.title("Customer Count vs Revenue")

plt.tight_layout()

plt.savefig("outputs/customer_vs_revenue.png")

plt.close()

# =====================================================
# INDUSTRY VS REVENUE
# =====================================================

industry_revenue = (
    df.groupby("industry")["annual_revenue_million"]
    .mean()
    .sort_values(ascending=False)
)

print("\nAVERAGE REVENUE BY INDUSTRY")
print(industry_revenue)

plt.figure(figsize=(8, 5))

industry_revenue.plot(kind="bar")

plt.ylabel("Average Revenue")

plt.title("Industry vs Revenue")

plt.tight_layout()

plt.savefig("outputs/industry_vs_revenue.png")

plt.close()

# =====================================================
# REGION VS EMPLOYEE COUNT
# =====================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="region",
    y="employee_count"
)

plt.title("Region vs Employee Count")

plt.tight_layout()

plt.savefig("outputs/region_vs_employee_count.png")

plt.close()

# =====================================================
# COUNTRY VS MARKET RATING
# =====================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="country",
    y="market_rating"
)

plt.title("Country vs Market Rating")

plt.tight_layout()

plt.savefig("outputs/country_vs_market_rating.png")

plt.close()

# =====================================================
# MULTIVARIATE ANALYSIS
# =====================================================

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="employee_count",
    y="annual_revenue_million",
    hue="industry"
)

plt.title("Industry × Revenue × Employee Count")

plt.tight_layout()

plt.savefig("outputs/industry_revenue_employee_analysis.png")

plt.close()

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="market_rating",
    y="profit_margin_percent",
    hue="region"
)

plt.title("Region × Rating × Profit Margin")

plt.tight_layout()

plt.savefig("outputs/region_rating_profit_margin_analysis.png")

plt.close()

# =====================================================
# BUSINESS INSIGHTS
# =====================================================

print("\n")
print("=" * 60)
print("BUSINESS INSIGHTS")
print("=" * 60)

top_industry = (
    df.groupby("industry")["annual_revenue_million"]
    .mean()
    .idxmax()
)

print("Highest Revenue Industry:", top_industry)

top_region = (
    df.groupby("region")["market_rating"]
    .mean()
    .idxmax()
)

print("Highest Rated Region:", top_region)

emp_rev_corr = df["employee_count"].corr(
    df["annual_revenue_million"]
)

cust_rev_corr = df["customer_count"].corr(
    df["annual_revenue_million"]
)

print(f"Employee-Revenue Correlation: {emp_rev_corr:.4f}")
print(f"Customer-Revenue Correlation: {cust_rev_corr:.4f}")

if abs(emp_rev_corr) > 0.7:
    print("Strong relationship between employee count and revenue")
elif abs(emp_rev_corr) > 0.3:
    print("Moderate relationship between employee count and revenue")
else:
    print("Weak relationship between employee count and revenue")

if abs(cust_rev_corr) > 0.7:
    print("Strong relationship between customer count and revenue")
elif abs(cust_rev_corr) > 0.3:
    print("Moderate relationship between customer count and revenue")
else:
    print("Weak relationship between customer count and revenue")

print("\nEDA COMPLETED SUCCESSFULLY")
print("All graphs saved in the 'outputs' folder.")