import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------------------
# Page Setup
# ----------------------------
st.set_page_config(page_title="EMS Crash Injury Disparities", page_icon="🚑", layout="wide")

# ----------------------------
# Data Loading and Preprocessing (Directly from GitHub Repo)
# ----------------------------
@st.cache_data(show_spinner="Processing data...")
def postprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Performs post-processing like type conversion after data loading."""
    if 'Year' in df.columns:
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype('Int64')
    if 'AgeGroup' in df.columns:
        order = ['0-24','25-34','35-44','45-54','55-64','65-74','75-84','85+']
        df['AgeGroup'] = pd.Categorical(df['AgeGroup'], categories=order, ordered=True)
    return df

@st.cache_data(show_spinner="Loading sample data...")
def load_data_from_repo(file_path: str) -> pd.DataFrame:
    """Loads and preprocesses the CSV file from the GitHub repository."""
    df = pd.read_csv(file_path, low_memory=False)
    df = postprocess(df)
    return df

# --- Main data loading execution ---
try:
    # Specify the name of the sample data file uploaded to the GitHub repository.
    df = load_data_from_repo('sampled_ems_data_100k.csv')
except FileNotFoundError:
    st.error("Error: 'sampled_ems_data_100k.csv' file not found.")
    st.info("Please ensure the data file is in the same GitHub repository as app.py.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the data: {e}")
    st.stop()

# use full dataset everywhere
fdf = df

# ----------------------------
# Sidebar: Page Navigation
# ----------------------------
st.sidebar.success(f"Data loaded successfully!\n({len(fdf):,} rows)")
st.sidebar.header("Navigate")

pages = {
    "🏠 Overview": "overview",
    "🕵️ Handling Missing Values": "missing_values",
    "🧹 Handling Data Duplicates": "data_duplicates",
    "🏛️ US Census Data Merging": "census_merging",
    "📊 Visualization": "visualization",
}
page = st.sidebar.radio("Go to", list(pages.keys()))


# ----------------------------
# Helper
# ----------------------------
def safe_is_numeric(col):
    try:
        return pd.api.types.is_numeric_dtype(col)
    except:
        return False

# ----------------------------
# Page Content
# ----------------------------
if page == "🏠 Overview":
    st.title("🚑 EMS-Reported Crash Injury Disparities: A Policy Analysis Tool")
    
    st.markdown("""
    This dashboard provides key insights into traffic injury disparities across the U.S. By analyzing national EMS data, I identify high-risk demographic subgroups to support data-driven policy and targeted safety interventions.
    """)

    st.subheader("Target Audience & Application")
    st.markdown("""
    This tool is designed for **government, public health, and transportation safety agencies**. The analysis helps answer critical questions for resource allocation and policy-making:
    - Which demographic groups (by age, race, gender) are most vulnerable to specific traffic injuries?
    - How do these patterns vary by region and urbanicity?
    - Where can interventions be most effectively targeted to improve transportation equity?
    """)

    st.subheader("Data at a Glance")
    st.markdown("""
    - **Source**: National EMS Information System (NEMSIS), 2018-2022.
    - **Full Dataset**: The complete research dataset contains ~6 million records.
    - **App Dataset**: For interactive performance, this dashboard uses a **100,000-record sample** to illustrate key trends.
    """)
    
    st.subheader("Data Preview")
    st.dataframe(fdf.head())

elif page == "🕵️ Handling Missing Values":
    st.title("🕵️ Handling Missing Values")
    st.markdown("""
    Before analysis, identifying and handling missing values is crucial. A simple check for `NaN` (Not a Number) values often misses text-based entries that represent missing data, known as **semantic missing values**.
    """)

    st.subheader("Step 1: Uncovering Semantic Missing Values")
    st.markdown("Many columns contained text like 'unknown' or 'not recorded' instead of a standard `NaN`. I identified these common null-like values to standardize them.")
    common_nulls = [
        "not recorded", "not applicable", "not known", "unknown", "missing",
        "none", "null", "n/a", "na", "not available", "refused", "blank", "", "nan"
    ]
    st.code(f"Common semantic nulls targeted:\n{common_nulls}", language='python')

    # Define the cleaning function to use it on the sample data
    def normalize_and_replace_nulls(df_to_clean):
        dfc = df_to_clean.copy()
        for col in dfc.columns:
            if pd.api.types.is_object_dtype(dfc[col]) or pd.api.types.is_string_dtype(dfc[col]):
                dfc[col] = dfc[col].where(
                    dfc[col].isna(),
                    dfc[col].astype(str).str.lower().str.strip()
                )
                dfc[col] = dfc[col].replace(common_nulls, np.nan)
        return dfc

    # Clean the sample dataframe for visualization
    fdf_cleaned = normalize_and_replace_nulls(fdf)

    st.subheader("Step 2: Visualizing True Missingness")
    st.markdown("After replacing all semantic nulls with standard `NaN`, I created a heatmap to visualize the true extent and pattern of missing data across the columns in my sample.")
    
    # Create and display the heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(fdf_cleaned.isnull(), cbar=False, cmap="viridis", ax=ax)
    ax.set_title("Missing Values Heatmap (After Cleaning)")
    st.pyplot(fig)
    st.caption("Each yellow line represents a missing value. This visualization helps identify columns with significant data gaps.")

    st.subheader("Step 3: Deep Dive into 'Age Units'")
    st.markdown("A key area of concern was the `ageinyear` column, which could be misinterpreted without its corresponding `Age Units` (e.g., an age of 11 could mean years or months).")

    if 'Age Units' in fdf_cleaned.columns:
        age_units_counts = fdf_cleaned['Age Units'].fillna('Missing').value_counts().reset_index()
        age_units_counts.columns = ['Age Units', 'Count']
        fig_age_units = px.bar(
            age_units_counts, x='Age Units', y='Count', color='Age Units',
            text='Count', title='Distribution of Age Units'
        )
        fig_age_units.update_traces(texttemplate='%{text:,}', textposition='outside')
        st.plotly_chart(fig_age_units, use_container_width=True)

        non_years_df = fdf_cleaned[fdf_cleaned['Age Units'].fillna('Missing') != 'years']
        st.write(f"**Finding:** There are **{len(non_years_df):,} rows** in the sample where the age unit is not 'years'. Most of these correspond to infants.")
        st.dataframe(non_years_df[['ageinyear', 'Age Units']].head())
    else:
        st.warning("'Age Units' column not found in the dataset.")
    
    st.subheader("Step 4: The Solution - Focusing on Age Groups")
    st.success("""
    **Action Taken:** Since my analysis focuses on disparities across broader age **groups** (e.g., '0-24', '25-34'), and not on fine-grained age differences for infants, I removed rows where the `Age Units` were not 'years' in my full dataset. This ensures consistency without sacrificing the core objectives of the study.
    """)

elif page == "🧹 Handling Data Duplicates":
    st.title("🧹 Handling Data Duplicates")
    st.markdown("""
    Data quality is paramount. My first step was to check for duplicate records. While no **perfectly identical rows** were found, I investigated potential **semantic duplicates** based on the primary incident identifier.
    """)

    st.subheader("Step 1: Identifying Duplicates by Incident ID (`PcrKey`)")
    st.markdown("`PcrKey` should be a unique key for each EMS incident. I checked if any `PcrKey` appeared more than once.")
    
    total_count = len(fdf)
    unique_count = fdf['PcrKey'].nunique()
    duplicated_incidents = total_count - unique_count

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows in Sample", f"{total_count:,}")
    col2.metric("Unique Incident Keys", f"{unique_count:,}")
    col3.metric("Rows with Duplicated Keys", f"{duplicated_incidents:,}", delta=f"-{duplicated_incidents:,} potential errors", delta_color="inverse")

    st.subheader("Step 2: Investigating the Cause of Duplicates")
    st.markdown("The duplicated rows were not perfectly identical, so I formed several hypotheses to explain the cause.")
    
    dup_keys_series = fdf['PcrKey'].value_counts()
    dup_keys_list = dup_keys_series[dup_keys_series > 1].index
    dup_df = fdf[fdf['PcrKey'].isin(dup_keys_list)]

    with st.expander("Hypothesis 1: Cross-Year Duplicates"):
        cross_year = dup_df.groupby('PcrKey')['Year'].nunique().value_counts()
        st.write("All duplicated incidents appear within the same year, ruling this out as a primary cause.")
        st.dataframe(cross_year.reset_index().rename(columns={'index': 'Number of Unique Years', 'Year': 'Count of Incidents'}))

    with st.expander("Hypothesis 2: Multi-Patient Duplicates"):
        age_col = 'Age_Number' if 'Age_Number' in dup_df.columns else 'PcrKey'
        multi_patient = dup_df.groupby('PcrKey')[['Gender', age_col]].nunique()
        is_multi = ((multi_patient['Gender'] > 1) | (multi_patient[age_col] > 1)).sum()
        st.write(f"I checked if duplicated keys had different gender or age values. **Result: {is_multi} cases found.** This is not the cause.")

    with st.expander("Hypothesis 3: Revision Duplicates"):
        time_cols = [c for c in fdf.columns if 'Time' in c]
        if time_cols:
            revision_like = dup_df.groupby('PcrKey')[time_cols].nunique().max(axis=1) > 1
            st.write(f"I checked for differences in timestamps across records with the same key. **Result: {revision_like.sum()} cases found.** This is also not the cause.")
        else:
            st.warning("No time-related columns found in the sample data to perform this check.")

    st.subheader("Step 3: The Finding - Data Entry Errors")
    st.markdown("""
    With my initial hypotheses disproven, I manually inspected a sample of the duplicated records. The investigation revealed the true cause: **minor data entry errors**.

    Specifically, for the same incident (`PcrKey`), all columns were identical *except for `Race`*. This suggests EMS teams occasionally created multiple records for a single patient due to accidental misclassification of race.
    """)
    
    race_diffs = dup_df.groupby('PcrKey')['Race'].nunique()
    keys_with_diff_race = race_diffs[race_diffs > 1].index
    
    if not keys_with_diff_race.empty:
        example_key = keys_with_diff_race[0]
        example_df = dup_df[dup_df['PcrKey'] == example_key].sort_values('Race')
        st.dataframe(example_df)
        st.caption(f"Example: The two rows above share the same `PcrKey` ({example_key}) but have different `Race` values. All other fields are identical.")
    else:
        st.warning("A clear example of race discrepancy was not found in this specific 100k sample, but the pattern was confirmed in the full dataset.")
        st.dataframe(dup_df.head(2))

    st.subheader("Step 4: The Solution - Removing Erroneous Records")
    st.markdown("""
    Since these duplicates represent data entry mistakes rather than distinct events or patients, they hold no value for imputation and could skew the analysis.
    """)
    st.success(f"**Action Taken:** In my full 6-million-row research dataset, all identified duplicate rows were removed to ensure the integrity of the modeling results.")

elif page == "🏛️ US Census Data Merging":
    st.title("🏛️ US Census Data Merging: The Next Step")
    st.markdown("""
    To accurately assess injury disparities, a simple count of incidents is insufficient. A group might have more injuries simply because their population is larger. The critical step is to **normalize** my EMS data with population counts to calculate rates (e.g., incidents per 100,000 people).
    """)

    st.subheader("Goal: Creating Population-Adjusted Rates")
    st.markdown("""
    My objective is to merge the EMS crash data with the **2018–2022 ACS 5-Year Estimates** from the U.S. Census Bureau. This will allow me to create a detailed population table structured like this:
    """)
    st.code("""
    # Target structure for the population data
    Division		Sex	  Race	   AgeGroup	 Population
    East North Central	Male	  Black	   0-24		  12,300
    East North Central	Male	  Black	   25-34	  4,800
    ...
    """, language='python')
    st.markdown("I attempted this merge using Python's Census library to acquire this granular data.")

    st.subheader("Current Status & Next Steps")
    st.info("""
    The data merging process presented some initial challenges. Therefore, I decided to proceed with a preliminary exploratory data analysis (EDA) first to understand the raw patterns in the data. The visualizations on the next page reflect this initial step.
    """)

elif page == "📊 Visualization":
    st.title("📊 Key Visualizations")
    
    st.warning("""
    **Important Caveat:** The charts presented on this page are based on **raw incident counts**, not population-adjusted rates.
    
    This means the rankings and proportions are preliminary and are expected to **change significantly** once the population data is successfully merged. The current visualizations show *where* incidents are occurring in the dataset, while the final, normalized data will show *which groups are disproportionately affected*.
    """)

    col1, col2 = st.columns(2)

    # Chart 1: Gender Donut
    with col1:
        if 'Gender' in fdf:
            gender_counts = fdf.dropna(subset=['Gender']).groupby('Gender').size().reset_index(name='Count')
            fig_gender = px.pie(gender_counts, names='Gender', values='Count', hole=0.4,
                                title="Crash Counts by Gender")
            fig_gender.update_traces(textinfo='percent+label', pull=[0.04]*len(gender_counts))
            st.plotly_chart(fig_gender, use_container_width=True)

    # Chart 2: Race Bar
    with col2:
        if 'Race' in fdf:
            race_counts = (fdf.dropna(subset=['Race'])
                              .groupby('Race').size().reset_index(name='Count')
                              .sort_values('Count', ascending=False))
            fig_race = px.bar(race_counts, x='Race', y='Count', color='Race',
                             title="Crash Counts by Race")
            fig_race.update_layout(xaxis_tickangle=35)
            st.plotly_chart(fig_race, use_container_width=True)
    
    st.divider()
    
    col3, col4 = st.columns(2)

    # Chart 3: Year trend
    with col3:
        if 'Year' in fdf:
            year_counts = fdf.dropna(subset=['Year']).groupby('Year').size().reset_index(name='Count')
            year_counts['Year'] = year_counts['Year'].astype('Int64')
            fig_year = px.line(year_counts, x='Year', y='Count', markers=True,
                          title='Crash Counts by Year')
            fig_year.update_layout(xaxis=dict(dtick=1))
            st.plotly_chart(fig_year, use_container_width=True)

    # Chart 4: Division bar
    with col4:
        if 'USCensusDivision' in fdf:
            div_counts = (fdf.dropna(subset=['USCensusDivision'])
                             .groupby('USCensusDivision').size()
                             .reset_index(name='Count').sort_values('Count', ascending=False))
            fig_div = px.bar(div_counts, x='USCensusDivision', y='Count', color='USCensusDivision',
                         title='Crash Counts by U.S. Census Division')
            fig_div.update_layout(xaxis_tickangle=35, showlegend=False)
            st.plotly_chart(fig_div, use_container_width=True)