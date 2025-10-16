import pandas as pd
import plotly.express as px
import streamlit as st

# ----------------------------
# 페이지 기본 설정
# ----------------------------
st.set_page_config(page_title="EMS Crash Injury Disparities", page_icon="🚑", layout="wide")

# ----------------------------
# 데이터 로딩 및 전처리 (GitHub Repo에서 직접 로드)
# ----------------------------
@st.cache_data(show_spinner="데이터 처리 중...")
def postprocess(df: pd.DataFrame) -> pd.DataFrame:
    """데이터 로드 후 타입 변환 등 후처리를 수행합니다."""
    if 'Year' in df.columns:
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype('Int64')
    if 'AgeGroup' in df.columns:
        order = ['0-24','25-34','35-44','45-54','55-64','65-74','75-84','85+']
        df['AgeGroup'] = pd.Categorical(df['AgeGroup'], categories=order, ordered=True)
    return df

@st.cache_data(show_spinner="샘플 데이터 로딩 중...")
def load_data_from_repo(file_path: str) -> pd.DataFrame:
    """GitHub 리포지토리 내의 CSV 파일을 로드하고 전처리합니다."""
    df = pd.read_csv(file_path, low_memory=False)
    df = postprocess(df)
    return df

# --- 메인 데이터 로딩 실행 ---
try:
    # GitHub 리포지토리에 함께 올린 샘플 데이터 파일명을 여기에 적습니다.
    df = load_data_from_repo('sampled_ems_data_100k.csv')
except FileNotFoundError:
    st.error("오류: 'sampled_ems_data_100k.csv' 파일을 찾을 수 없습니다.")
    st.info("app.py 파일과 동일한 GitHub 리포지토리 안에 데이터 파일이 있는지 확인해주세요.")
    st.stop()
except Exception as e:
    st.error(f"데이터를 로드하는 중 오류가 발생했습니다: {e}")
    st.stop()

# use full dataset everywhere
fdf = df

# ----------------------------
# 사이드바: 페이지 네비게이션
# ----------------------------
st.sidebar.success(f"데이터 로딩 완료!\n({len(fdf):,} 행)")
st.sidebar.header("Navigate")

pages = {
    "🏠 Overview": "overview",
    "🧹 Data & Cleaning": "data_cleaning",
    "📈 Univariate EDA": "univariate",
    "🔗 Bivariate EDA": "bivariate",
    "🧭 Temporal & Regional": "temporal_regional",
    "🧪 Modeling Plan": "modeling_plan",
    "ℹ️ About": "about",
}
page = st.sidebar.radio("페이지 이동", list(pages.keys()))


# ----------------------------
# Helper
# ----------------------------
def safe_is_numeric(col):
    try:
        return pd.api.types.is_numeric_dtype(col)
    except:
        return False

# ----------------------------
# 각 페이지 콘텐츠 (이 부분은 변경되지 않았습니다)
# ----------------------------
if page == "🏠 Overview":
    st.title("🚑 Disparities in EMS-Reported Crash Injury Outcomes")
    st.markdown("""
This app organizes **exploratory data analysis (EDA)** for U.S. **NEMSIS (2018–2022)** EMS crash records into a presentation-ready storyline.

**Goals**
- Understand **pre-normalization** patterns by race, gender, and age group.
- Later, merge **ACS 2018–2022 5-year** populations to compute **rates per 100,000** and fit models.
- Support a clear narrative for talks, posters, and papers.
    """)

    st.subheader("Dataset Snapshot")
    c1, c2 = st.columns([1,1])
    with c1:
        st.write("**Rows:**", f"{len(fdf):,}")
        st.write("**Columns:**", f"{len(fdf.columns):,}")
    with c2:
        if 'Year' in fdf: st.write("**Years present:**", ", ".join(map(str, sorted(set(fdf['Year'].dropna())))))
        if 'USCensusDivision' in fdf: st.write("**Divisions present:**", ", ".join(sorted(set(fdf['USCensusDivision'].dropna()))))
    st.info("Move through the pages to cover: Univariate → Bivariate → Temporal/Regional → Modeling plan.")

elif page == "🧹 Data & Cleaning":
    st.title("🧹 Data Overview & Cleaning")
    st.markdown("""
- Each row corresponds to an **EMS-attended crash record**.
- Key columns: `Gender`, `Race`, `AgeGroup`, `USCensusDivision`, `Urbanicity`, `Year`, etc.
- Use this page to quickly **check missingness** and **numeric distributions**.
    """)
    st.subheader("Missingness (top 20 by % NA)")
    miss = fdf.isna().mean().sort_values(ascending=False).head(20) * 100
    st.bar_chart(miss)

    st.subheader("Numeric Distribution Quick Check")
    num_cols = [c for c in fdf.columns if safe_is_numeric(fdf[c])]
    if num_cols:
        sel_num = st.selectbox("Choose a numeric column", num_cols, index=0)
        fig = px.histogram(fdf, x=sel_num, nbins=50, title=f"Distribution of {sel_num}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No numeric columns detected.")

elif page == "📈 Univariate EDA":
    st.title("📈 Univariate EDA")
    col1, col2 = st.columns(2)

    # Gender — Donut
    if 'Gender' in fdf:
        gender_counts = fdf.dropna(subset=['Gender']).groupby('Gender').size().reset_index(name='Count')
        with col1:
            fig = px.pie(gender_counts, names='Gender', values='Count', hole=0.4,
                         title="Crash Counts by Gender")
            fig.update_traces(textinfo='percent+label', pull=[0.04]*len(gender_counts))
            st.plotly_chart(fig, use_container_width=True)

    # Race — Bar
    if 'Race' in fdf:
        race_counts = (fdf.dropna(subset=['Race'])
                          .groupby('Race').size().reset_index(name='Count')
                          .sort_values('Count', ascending=False))
        with col2:
            fig = px.bar(race_counts, x='Race', y='Count', color='Race',
                         title="Crash Counts by Race")
            fig.update_layout(xaxis_tickangle=35)
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    # AgeGroup — Bar
    if 'AgeGroup' in fdf:
        age_counts = fdf.dropna(subset=['AgeGroup']).groupby('AgeGroup').size().reset_index(name='Count')
        if pd.api.types.is_categorical_dtype(age_counts['AgeGroup']):
            age_counts = age_counts.sort_values('AgeGroup')
        fig = px.bar(age_counts, x='AgeGroup', y='Count', text='Count', color='AgeGroup',
                     title="Crash Counts by Age Group")
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

elif page == "🔗 Bivariate EDA":
    st.title("🔗 Bivariate EDA")
    tabs = st.tabs(["Gender × Race", "AgeGroup × Gender", "Race × Urbanicity"])

    # Gender × Race
    with tabs[0]:
        if set(['Gender','Race']).issubset(fdf.columns):
            cross = fdf.dropna(subset=['Gender','Race']).groupby(['Gender','Race']).size().reset_index(name='Count')
            fig = px.density_heatmap(cross, x='Race', y='Gender', z='Count',
                                     color_continuous_scale='Viridis',
                                     title="Crash Counts — Gender × Race")
            fig.update_layout(xaxis_tickangle=35)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Columns `Gender` and `Race` required.")

    # AgeGroup × Gender
    with tabs[1]:
        if set(['AgeGroup','Gender']).issubset(fdf.columns):
            ag = fdf.dropna(subset=['AgeGroup','Gender']).groupby(['AgeGroup','Gender']).size().reset_index(name='Count')
            if 'AgeGroup' in ag and ag['AgeGroup'].dtype.name == 'category':
                ag = ag.sort_values('AgeGroup')
            fig = px.bar(ag, x='AgeGroup', y='Count', color='Gender', barmode='group',
                         title="Crash Counts — Age Group × Gender")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Columns `AgeGroup` and `Gender` required.")

    # Race × Urbanicity
    with tabs[2]:
        if set(['Race','Urbanicity']).issubset(fdf.columns):
            cu = fdf.dropna(subset=['Race','Urbanicity']).groupby(['Race','Urbanicity']).size().reset_index(name='Count')
            fig = px.density_heatmap(cu, x='Urbanicity', y='Race', z='Count',
                                     color_continuous_scale='Plasma',
                                     title="Crash Counts — Race × Urbanicity")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Columns `Race` and `Urbanicity` required.")

elif page == "🧭 Temporal & Regional":
    st.title("🧭 Temporal & Regional Patterns")
    sub1, sub2 = st.columns(2)

    # Year trend
    if 'Year' in fdf:
        year_counts = fdf.dropna(subset=['Year']).groupby('Year').size().reset_index(name='Count')
        year_counts['Year'] = year_counts['Year'].astype('Int64')
        fig = px.line(year_counts, x='Year', y='Count', markers=True,
                      title='Crash Counts by Year')
        fig.update_layout(xaxis=dict(dtick=1))
        sub1.plotly_chart(fig, use_container_width=True)

    # Division bar
    if 'USCensusDivision' in fdf:
        div_counts = (fdf.dropna(subset=['USCensusDivision'])
                         .groupby('USCensusDivision').size()
                         .reset_index(name='Count').sort_values('Count', ascending=False))
        fig = px.bar(div_counts, x='USCensusDivision', y='Count', color='USCensusDivision',
                     title='Crash Counts by U.S. Census Division')
        fig.update_layout(xaxis_tickangle=35, showlegend=False)
        sub2.plotly_chart(fig, use_container_width=True)

    st.info("After merging ACS 5-year populations, extend this page with **per-100k rate** maps/heatmaps.")
