import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis, zscore

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("industry_dataset.csv")

print("="*60)
print("DATASET SHAPE")
print("="*60)
print(df.shape)

print("\nFIRST 5 RECORDS")
print(df.head())

print("\nDATASET INFO")
print(df.info())

print("\nMISSING VALUES")
print(df.isnull().sum())

# =========================
# DATE CONVERSION
# =========================

df["created_date"] = pd.to_datetime(
    df["created_date"],
    format="%d-%m-%Y"
)

# =========================
# NUMERICAL & CATEGORICAL
# =========================

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

# =========================
# UNIVARIATE ANALYSIS
# =========================

print("\n")
print("="*60)
print("UNIVARIATE ANALYSIS")
print("="*60)

for col in num_cols:

    print(f"\n----- {col} -----")

    print("Mean:", round(df[col].mean(),2))
    print("Median:", round(df[col].median(),2))
    print("Std Dev:", round(df[col].std(),2))
    print("Minimum:", df[col].min())
    print("Maximum:", df[col].max())
    print("Skewness:", round(skew(df[col]),2))
    print("Kurtosis:", round(kurtosis(df[col]),2))

# =========================
# HISTOGRAMS
# =========================

for col in num_cols:

    plt.figure(figsize=(8,5))

    sns.histplot(df[col], kde=True)

    plt.title(f"Distribution of {col}")

    plt.tight_layout()

    plt.show()

# =========================
# CATEGORICAL ANALYSIS
# =========================

print("\n")
print("="*60)
print("CATEGORICAL ANALYSIS")
print("="*60)

for col in cat_cols:

    print(f"\nFrequency Distribution of {col}")
    print(df[col].value_counts())

    plt.figure(figsize=(8,5))

    sns.countplot(
        x=df[col]
    )

    plt.title(col)

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.show()

# =========================
# OUTLIER DETECTION
# =========================

print("\n")
print("="*60)
print("OUTLIER DETECTION")
print("="*60)

for col in num_cols:

    plt.figure(figsize=(8,4))

    sns.boxplot(
        x=df[col]
    )

    plt.title(f"Boxplot of {col}")

    plt.tight_layout()

    plt.show()

# =========================
# IQR METHOD
# =========================

print("\nIQR OUTLIERS")

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

# =========================
# Z SCORE
# =========================

print("\nZ-SCORE OUTLIERS")

z_scores = np.abs(
    zscore(df[num_cols])
)

print(
    (z_scores > 3).sum(axis=0)
)

# =========================
# BIVARIATE ANALYSIS
# =========================

print("\n")
print("="*60)
print("CORRELATION MATRIX")
print("="*60)

corr = df[num_cols].corr()

print(corr)

# =========================
# HEATMAP
# =========================

plt.figure(figsize=(10,6))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.show()

# =========================
# EMPLOYEE VS REVENUE
# =========================

plt.figure(figsize=(8,5))

sns.scatterplot(
    data=df,
    x="employee_count",
    y="annual_revenue_million"
)

plt.title(
    "Employee Count vs Revenue"
)

plt.tight_layout()

plt.show()

# =========================
# CUSTOMER VS REVENUE
# =========================

plt.figure(figsize=(8,5))

sns.scatterplot(
    data=df,
    x="customer_count",
    y="annual_revenue_million"
)

plt.title(
    "Customer Count vs Revenue"
)

plt.tight_layout()

plt.show()

# =========================
# INDUSTRY VS REVENUE
# =========================

industry_revenue = df.groupby(
    "industry"
)["annual_revenue_million"].mean()

print("\nAVERAGE REVENUE BY INDUSTRY")

print(industry_revenue)

plt.figure(figsize=(8,5))

industry_revenue.plot(
    kind="bar"
)

plt.ylabel(
    "Average Revenue"
)

plt.title(
    "Industry vs Revenue"
)

plt.tight_layout()

plt.show()

# =========================
# REGION VS EMPLOYEE COUNT
# =========================

plt.figure(figsize=(8,5))

sns.boxplot(
    data=df,
    x="region",
    y="employee_count"
)

plt.title(
    "Region vs Employee Count"
)

plt.tight_layout()

plt.show()

# =========================
# COUNTRY VS MARKET RATING
# =========================

plt.figure(figsize=(8,5))

sns.boxplot(
    data=df,
    x="country",
    y="market_rating"
)

plt.title(
    "Country vs Market Rating"
)

plt.tight_layout()

plt.show()

# =========================
# MULTIVARIATE ANALYSIS
# =========================

plt.figure(figsize=(10,6))

sns.scatterplot(
    data=df,
    x="employee_count",
    y="annual_revenue_million",
    hue="industry"
)

plt.title(
    "Industry Revenue Employee Analysis"
)

plt.tight_layout()

plt.show()

plt.figure(figsize=(10,6))

sns.scatterplot(
    data=df,
    x="market_rating",
    y="profit_margin_percent",
    hue="region"
)

plt.title(
    "Region Rating Profit Margin Analysis"
)

plt.tight_layout()

plt.show()

# =========================
# BUSINESS INSIGHTS
# =========================

print("\n")
print("="*60)
print("BUSINESS INSIGHTS")
print("="*60)

top_industry = df.groupby(
    "industry"
)["annual_revenue_million"].mean().idxmax()

print(
    "Highest Revenue Industry:",
    top_industry
)

top_region = df.groupby(
    "region"
)["market_rating"].mean().idxmax()

print(
    "Highest Rated Region:",
    top_region
)

emp_rev_corr = df[
    ["employee_count",
     "annual_revenue_million"]
].corr().iloc[0,1]

print(
    "Employee-Revenue Correlation:",
    round(emp_rev_corr,4)
)

cust_rev_corr = df[
    ["customer_count",
     "annual_revenue_million"]
].corr().iloc[0,1]

print(
    "Customer-Revenue Correlation:",
    round(cust_rev_corr,4)
)

print("\nEDA COMPLETED SUCCESSFULLY")