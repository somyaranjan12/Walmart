# walmart_descriptive_app_cloud.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import matplotlib.ticker as ticker

# ======================================
# Helper functions
# ======================================
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    buf.seek(0)
    return buf.getvalue()

def describe_skew(mean_val, median_val):
    if pd.isna(mean_val) or pd.isna(median_val):
        return "distribution is unclear due to lack of data"
    if mean_val > median_val * 1.2:
        return "distribution is **right-skewed** (a few high-value transactions)"
    elif median_val > mean_val * 1.2:
        return "distribution is **left-skewed**"
    else:
        return "distribution is **fairly symmetric**"

# ======================================
# Streamlit Config
# ======================================
st.set_page_config(
    page_title="Walmart Sales Descriptive EDA (Cloud)",
    layout="wide"
)

st.title("📊 Walmart Sales Descriptive Analytics Dashboard (Streamlit Cloud)")
st.markdown(
    """
This app performs **descriptive analytics (EDA)** on the Walmart Sales dataset with:

- Interactive filters (**Date, Category, Region, Channel**)
- Visualizations for key business questions
- **Automatic conclusions** based on the **currently filtered data**

> On Streamlit Cloud:  
> - Either keep `Walmart Sales_Dataset_Original.csv` in the same repo as this file, or  
> - Upload the CSV using the sidebar uploader.
"""
)

# ======================================
# Load data
# ======================================
@st.cache_data
def load_data(uploaded_file=None):
    """
    Priority:
    1. If user uploads a CSV, use that.
    2. Else, try to load 'Walmart Sales_Dataset_Original.csv' from repo.
    """
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        # For Streamlit Cloud: this file must be present in the repository
        df = pd.read_csv("Walmart Sales_Dataset_Original.csv")

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    return df

st.sidebar.header("📂 Data Source")

uploaded_file = st.sidebar.file_uploader(
    "Upload Walmart Sales CSV (optional)",
    type=["csv"],
    help="If not uploaded, the app will load 'Walmart Sales_Dataset_Original.csv' from the repo."
)

try:
    df_raw = load_data(uploaded_file)
except Exception as e:
    st.error(
        "❌ Could not load data.\n\n"
        "If you are on Streamlit Cloud, make sure the file "
        "`Walmart Sales_Dataset_Original.csv` is in the same GitHub repo folder "
        "as this app, **or** upload a CSV using the sidebar.\n\n"
        f"Details: {e}"
    )
    st.stop()

# ======================================
# Sidebar Filters
# ======================================
st.sidebar.header("🎛 Filters (apply to all EDA)")

df = df_raw.copy()

# Date filter
if "Date" in df.columns and df["Date"].notna().any():
    min_date = df["Date"].min()
    max_date = df["Date"].max()
    date_range = st.sidebar.date_input(
        "Date range",
        (min_date.date(), max_date.date()),
        min_value=min_date.date(),
        max_value=max_date.date()
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        mask = (df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))
        df = df[mask]

# Category filter
if "Category" in df.columns:
    categories = sorted(df["Category"].dropna().unique().tolist())
    selected_categories = st.sidebar.multiselect(
        "Category",
        options=categories,
        default=categories
    )
    if selected_categories:
        df = df[df["Category"].isin(selected_categories)]
    else:
        df = df.iloc[0:0]  # no selection -> empty

# Region filter
if "Region" in df.columns:
    regions = sorted(df["Region"].dropna().unique().tolist())
    selected_regions = st.sidebar.multiselect(
        "Region",
        options=regions,
        default=regions
    )
    if selected_regions:
        df = df[df["Region"].isin(selected_regions)]
    else:
        df = df.iloc[0:0]

# Channel filter
if "Channel" in df.columns:
    channels = sorted(df["Channel"].dropna().unique().tolist())
    selected_channels = st.sidebar.multiselect(
        "Channel",
        options=channels,
        default=channels
    )
    if selected_channels:
        df = df[df["Channel"].isin(selected_channels)]
    else:
        df = df.iloc[0:0]

if df.empty:
    st.warning("⚠ No data available for the current filter selection. Please adjust filters on the left.")
    st.stop()

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

# ======================================
# Top-level info
# ======================================
st.subheader("📁 Filtered Dataset Preview")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows (after filters)", df.shape[0])
c2.metric("Columns", df.shape[1])
c3.metric("Numeric Columns", len(num_cols))
c4.metric("Categorical Columns", len(cat_cols))

st.dataframe(df.head())

st.download_button(
    label="⬇ Download Filtered Dataset (CSV)",
    data=df_to_csv_bytes(df),
    file_name="walmart_filtered_dataset.csv",
    mime="text/csv"
)

st.markdown("---")

# ======================================
# Navigation
# ======================================
st.sidebar.header("📌 Analysis Sections")
section = st.sidebar.radio(
    "Go to section:",
    (
        "Distribution Analysis",
        "Category & Subcategory Performance",
        "Regional & City Insights",
        "Sales Over Time & Weekdays",
        "Customer, Channel & Payment",
        "Discount & Profitability",
        "Inventory vs Demand",
        "Correlation Analysis"
    )
)

# ======================================
# 1️⃣ Distribution Analysis
# ======================================
if section == "Distribution Analysis":
    st.header("📊 Distribution of Key Metrics")

    # Total Sales
    if "Total_Sales" in df.columns:
        st.subheader("Distribution of Total Sales per Transaction")

        fig, ax = plt.subplots()
        ax.hist(df["Total_Sales"].dropna(), bins=30)
        ax.set_xlabel("Total Sales")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Total Sales")
        st.pyplot(fig)

        mean_ts = df["Total_Sales"].mean()
        median_ts = df["Total_Sales"].median()
        max_ts = df["Total_Sales"].max()
        skew_desc = describe_skew(mean_ts, median_ts)

        st.markdown(
            f"""
**Conclusion (for current filters):**  
- Average bill value is **{mean_ts:,.2f}** and median is **{median_ts:,.2f}**, so the {skew_desc}.  
- The highest transaction in this view is **{max_ts:,.2f}**.  
- To increase revenue, push more customers toward the higher bill values (bundles, cross-sell, premium packs).
"""
        )

    st.markdown("---")

    # Profit
    if "Profit" in df.columns:
        st.subheader("Distribution of Profit per Transaction")

        fig, ax = plt.subplots()
        ax.hist(df["Profit"].dropna(), bins=30)
        ax.set_xlabel("Profit")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Profit")
        st.pyplot(fig)

        mean_p = df["Profit"].mean()
        median_p = df["Profit"].median()
        min_p = df["Profit"].min()
        max_p = df["Profit"].max()

        st.markdown(
            f"""
**Conclusion (for current filters):**  
- Average profit per transaction is **{mean_p:,.2f}** (median **{median_p:,.2f}**, range **{min_p:,.2f}** to **{max_p:,.2f}**).  
- If there are many low or negative profit transactions, they may come from **heavy discounts or low-margin SKUs**.
"""
        )

    st.markdown("---")

    # Age
    if "Age" in df.columns:
        st.subheader("Customer Age Distribution")

        fig, ax = plt.subplots()
        ax.hist(df["Age"].dropna(), bins=20)
        ax.set_xlabel("Age")
        ax.set_ylabel("Count")
        ax.set_title("Age Distribution of Customers")
        st.pyplot(fig)

        avg_age = df["Age"].mean()
        min_age = df["Age"].min()
        max_age = df["Age"].max()

        st.markdown(
            f"""
**Conclusion (for current filters):**  
- Average customer age is **{avg_age:,.1f} years** (range **{min_age:.0f}–{max_age:.0f}**).  
- Use this to design **age-appropriate promotions** for the segment you're currently viewing.
"""
        )

# ======================================
# 2️⃣ Category & Subcategory Performance
# ======================================
elif section == "Category & Subcategory Performance":
    st.header("📦 Category & Subcategory Performance")

    # Category-wise total sales
    if {"Category", "Total_Sales"}.issubset(df.columns):
        st.subheader("Total Sales by Category")

        cat_sales = df.groupby("Category")["Total_Sales"].sum().sort_values(ascending=False)

        fig, ax = plt.subplots()
        cat_sales.plot(kind="bar", ax=ax)
        ax.set_xlabel("Category")
        ax.set_ylabel("Total Sales")
        ax.set_title("Total Sales by Category (Current Filters)")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        st.pyplot(fig)

        total_sales = cat_sales.sum()
        top_cat = cat_sales.idxmax()
        top_cat_val = cat_sales.max()
        share = (top_cat_val / total_sales * 100) if total_sales != 0 else 0

        st.markdown(
            f"""
**Conclusion (for current filters):**  
- Category **`{top_cat}`** is the top contributor with **{share:.1f}%** of total sales in this filtered view.  
- Low-performing categories may need **better promotion, pricing review, or assortment changes**.
"""
        )

    st.markdown("---")

    # Sub-category total sales
    if {"Sub_Category", "Total_Sales"}.issubset(df.columns):
        st.subheader("Total Sales by Sub-Category")

        sub_sales = df.groupby("Sub_Category")["Total_Sales"].sum().sort_values(ascending=False)

        fig2, ax2 = plt.subplots(figsize=(10, 5))
        sub_sales.plot(kind="bar", ax=ax2)
        ax2.set_xlabel("Sub-Category")
        ax2.set_ylabel("Total Sales")
        ax2.set_title("Total Sales by Sub-Category (Current Filters)")
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90, ha="right")
        st.pyplot(fig2)

        top_sub = sub_sales.idxmax()
        top_sub_val = sub_sales.max()

        st.markdown(
            f"""
**Conclusion (for current filters):**  
- Sub-category **`{top_sub}`** leads with sales of **{top_sub_val:,.2f}**.  
- These are good candidates for **feature displays, end-caps, and cross-sell combos**.
"""
        )

    st.markdown("---")

    # Profit margin by Category
    if {"Category", "Profit", "Total_Sales"}.issubset(df.columns):
        st.subheader("Average Profit Margin by Category")

        df_margin = df[df["Total_Sales"] != 0].copy()
        df_margin["Profit_Margin"] = df_margin["Profit"] / df_margin["Total_Sales"]

        cat_margin = df_margin.groupby("Category")["Profit_Margin"].mean().sort_values(ascending=False)

        fig3, ax3 = plt.subplots()
        cat_margin.plot(kind="bar", ax=ax3)
        ax3.set_xlabel("Category")
        ax3.set_ylabel("Average Profit Margin")
        ax3.set_title("Average Profit Margin by Category (Current Filters)")
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha="right")
        st.pyplot(fig3)

        best_cat = cat_margin.idxmax()
        best_margin = cat_margin.max()

        st.markdown(
            f"""
**Conclusion (for current filters):**  
- Category **`{best_cat}`** has the **highest average margin** at **{best_margin:.2%}**.  
- High-margin categories are ideal for **premium positioning and targeted upsell** campaigns.
"""
        )

# ======================================
# 3️⃣ Regional & City Insights
# ======================================
elif section == "Regional & City Insights":
    st.header("🌍 Regional & City Insights")

    if {"Region", "Total_Sales"}.issubset(df.columns):
        st.subheader("Total Sales by Region")

        reg_sales = df.groupby("Region")["Total_Sales"].sum().sort_values(ascending=False)

        fig, ax = plt.subplots()
        reg_sales.plot(kind="bar", ax=ax)
        ax.set_xlabel("Region")
        ax.set_ylabel("Total Sales")
        ax.set_title("Total Sales by Region (Current Filters)")
        st.pyplot(fig)

        top_reg = reg_sales.idxmax()
        top_reg_val = reg_sales.max()

        st.markdown(
            f"""
**Conclusion (for current filters):**  
- Region **`{top_reg}`** is currently the **best-performing region** with sales of **{top_reg_val:,.2f}**.  
- Regions with lower bars may benefit from **localized assortment, regional offers, or logistics improvements.**
"""
        )

    st.markdown("---")

    if {"City", "Total_Sales"}.issubset(df.columns):
        st.subheader("Top 10 Cities by Sales")

        city_sales = df.groupby("City")["Total_Sales"].sum().sort_values(ascending=False).head(10)

        fig2, ax2 = plt.subplots()
        city_sales.plot(kind="bar", ax=ax2)
        ax2.set_xlabel("City")
        ax2.set_ylabel("Total Sales")
        ax2.set_title("Top 10 Cities by Sales (Current Filters)")
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")
        st.pyplot(fig2)

        top_city = city_sales.idxmax()
        top_city_val = city_sales.max()

        st.markdown(
            f"""
**Conclusion (for current filters):**  
- **`{top_city}`** is the top city with **{top_city_val:,.2f}** in sales.  
- These cities are ideal for **pilot programs, loyalty campaigns, and new concept launches.**
"""
        )

# ======================================
# 4️⃣ Sales Over Time & Weekdays
# ======================================
elif section == "Sales Over Time & Weekdays":
    st.header("⏱ Sales Over Time & Weekday Patterns")

    if "Date" in df.columns and "Total_Sales" in df.columns:
        ts_df = df.sort_values("Date").copy()
        daily = ts_df.groupby("Date")["Total_Sales"].sum()

        st.subheader("Daily Total Sales Trend")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(daily.index, daily.values)
        ax.set_xlabel("Date")
        ax.set_ylabel("Total Sales")
        ax.set_title("Daily Total Sales (Current Filters)")
        plt.xticks(rotation=45)
        st.pyplot(fig)

        first_date = daily.index.min()
        last_date = daily.index.max()

        st.markdown(
            f"""
**Conclusion (for current filters):**  
- Sales trend shown from **{first_date.date()}** to **{last_date.date()}**.  
- Spikes highlight **high-demand days** that are useful for planning **future promotions and inventory**.
"""
        )

        st.markdown("---")

        # Weekday pattern
        if "Quantity" in df.columns:
            st.subheader("Sales Volume by Weekday")

            ts_df = ts_df[ts_df["Date"].notna()]
            ts_df["Weekday"] = ts_df["Date"].dt.day_name()
            weekday_qty = ts_df.groupby("Weekday")["Quantity"].sum()

            order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            weekday_qty = weekday_qty.reindex(order)

            fig2, ax2 = plt.subplots()
            weekday_qty.plot(kind="bar", ax=ax2)
            ax2.set_xlabel("Weekday")
            ax2.set_ylabel("Total Quantity")
            ax2.set_title("Sales Volume by Weekday (Current Filters)")
            st.pyplot(fig2)

            top_day = weekday_qty.idxmax()
            top_day_val = weekday_qty.max()

            st.markdown(
                f"""
**Conclusion (for current filters):**  
- **`{top_day}`** is the **busiest day** with **{top_day_val:,.0f} units** sold.  
- Use this for **staffing, inventory allocation, and weekend/weekday promotion design.**
"""
            )

# ======================================
# 5️⃣ Customer, Channel & Payment
# ======================================
elif section == "Customer, Channel & Payment":
    st.header("👥 Customer, Channel & Payment Behaviour")

    if {"Channel", "Total_Sales"}.issubset(df.columns):
        st.subheader("Total Sales by Channel")

        ch_sales = df.groupby("Channel")["Total_Sales"].sum().sort_values(ascending=False)

        fig, ax = plt.subplots()
        ch_sales.plot(kind="bar", ax=ax)
        ax.set_xlabel("Channel")
        ax.set_ylabel("Total Sales")
        ax.set_title("Sales by Channel (Current Filters)")
        st.pyplot(fig)

        top_ch = ch_sales.idxmax()
        top_ch_val = ch_sales.max()

        st.markdown(
            f"""
**Conclusion (for current filters):**  
- **`{top_ch}`** is the leading channel with sales of **{top_ch_val:,.2f}**.  
- Other channels may need **better integration, UX, or marketing** to catch up.
"""
        )

    st.markdown("---")

    if "Payment_Method" in df.columns:
        st.subheader("Payment Method Distribution")

        pay_counts = df["Payment_Method"].value_counts()

        fig2, ax2 = plt.subplots()
        pay_counts.plot(kind="bar", ax=ax2)
        ax2.set_xlabel("Payment Method")
        ax2.set_ylabel("Number of Transactions")
        ax2.set_title("Payment Method Distribution (Current Filters)")
        st.pyplot(fig2)

        top_pay = pay_counts.idxmax()
        top_pay_val = pay_counts.max()

        st.markdown(
            f"""
**Conclusion (for current filters):**  
- Most customers use **`{top_pay}`** (**{top_pay_val} transactions**).  
- Ensuring a **smooth, reliable experience** for this payment mode is critical.
"""
        )

# ======================================
# 6️⃣ Discount & Profitability
# ======================================
elif section == "Discount & Profitability":
    st.header("💸 Discount vs Profitability")

    if {"Discount", "Profit"}.issubset(df.columns):
        st.subheader("Profit vs Discount (Scatter)")

        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x="Discount", y="Profit", ax=ax, alpha=0.6)
        ax.set_xlabel("Discount")
        ax.set_ylabel("Profit")
        ax.set_title("Profit vs Discount (Current Filters)")
        st.pyplot(fig)

        corr_val = df["Discount"].corr(df["Profit"])

        st.markdown(
            f"""
**Conclusion (for current filters):**  
- The correlation between **Discount** and **Profit** is **{corr_val:.2f}**.  
- A strong negative value would indicate that **higher discounts reduce profit sharply**.  
- Around zero suggests discounts are not the main driver of profit in this filtered segment.
"""
        )

# ======================================
# 7️⃣ Inventory vs Demand
# ======================================
elif section == "Inventory vs Demand":
    st.header("📦 Inventory vs Demand (Stock vs Quantity)")

    if {"Stock_Level", "Quantity"}.issubset(df.columns):
        st.subheader("Stock Level vs Quantity Sold")

        fig, ax = plt.subplots()
        ax.scatter(df["Stock_Level"], df["Quantity"], alpha=0.4)
        ax.set_xlabel("Stock Level")
        ax.set_ylabel("Quantity Sold")
        ax.set_title("Stock vs Quantity (Current Filters)")
        st.pyplot(fig)

        corr_sq = df["Stock_Level"].corr(df["Quantity"])

        st.markdown(
            f"""
**Conclusion (for current filters):**  
- Correlation between **Stock Level** and **Quantity Sold** is **{corr_sq:.2f}**.  
- A positive correlation suggests **keeping enough stock helps meet demand**, while scattered points with low correlation mean other factors (price, promotion) dominate.  
- Points with high stock but low quantity sold indicate **overstock**, and low stock but high quantity sold indicate **stock-out risk**.
"""
        )
    else:
        st.info("Columns `Stock_Level` and/or `Quantity` not available.")

# ======================================
# 8️⃣ Correlation Analysis
# ======================================
elif section == "Correlation Analysis":
    st.header("📌 Correlation Heatmap of Numeric Features")

    if len(num_cols) < 2:
        st.warning("Not enough numeric columns for correlation analysis.")
    else:
        corr = df[num_cols].corr()

        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Heatmap (Current Filters)")
        st.pyplot(fig)

        corr_unstack = corr.where(~np.eye(corr.shape[0], dtype=bool))
        max_corr = corr_unstack.abs().stack().idxmax()
        max_val = corr.loc[max_corr[0], max_corr[1]]

        st.markdown(
            f"""
**Conclusion (for current filters):**  
- The strongest numeric relationship is between **`{max_corr[0]}`** and **`{max_corr[1]}`** with correlation **{max_val:.2f}**.  
- Strong positive relationships can be leveraged for **feature engineering**, forecasting, or building composite KPIs.
"""
        )
