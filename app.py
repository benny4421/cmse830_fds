import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="EMS Crash Injury Disparities", page_icon="🚑", layout="wide")

# ----------------------------
# Data Loader
# ----------------------------
@st.cache_data(show_spinner=False)
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    # dtype 정리
    if 'Year' in df.columns:
        # 연도 float → int → str 로도 활용 가능하지만, 여기선 int 보장
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype('Int64')
    # 표준 카테고리 정렬(있을 때만)
    if 'AgeGroup' in df.columns:
        age_order = ['0-24','25-34','35-44','45-54','55-64','65-74','75-84','85+']
        df['AgeGroup'] = pd.Categorical(df['AgeGroup'], categories=age_order, ordered=True)
    return df

# ----------------------------
# Sidebar: 데이터 소스 선택
# ----------------------------
st.sidebar.header("📂 Data Source")
default_path = "/content/ems_merged_clean.csv"
csv_path = st.sidebar.text_input("CSV path", value=default_path, help="코랩에서 저장한 CSV 경로를 입력하세요.")
uploaded = st.sidebar.file_uploader("...또는 파일 업로드", type=["csv"])
if uploaded is not None:
    df = load_data(uploaded)
else:
    df = load_data(csv_path)

# 공통 필터 위젯 (여러 페이지에서 재사용)
st.sidebar.header("🔎 Filters")
# 존재 여부 체크 후 동적 필터 생성
def get_vals(col):
    return sorted([v for v in df[col].dropna().unique().tolist()]) if col in df.columns else []

years = get_vals('Year')
divisions = get_vals('USCensusDivision')
races = get_vals('Race')
genders = get_vals('Gender')
agegroups = get_vals('AgeGroup')
urbanicity = get_vals('Urbanicity')

year_sel = st.sidebar.multiselect("Year", years, default=years)
div_sel  = st.sidebar.multiselect("US Census Division", divisions, default=divisions)
race_sel = st.sidebar.multiselect("Race", races, default=races)
gen_sel  = st.sidebar.multiselect("Gender", genders, default=genders)
age_sel  = st.sidebar.multiselect("Age Group", agegroups, default=agegroups)
urb_sel  = st.sidebar.multiselect("Urbanicity", urbanicity, default=urbanicity)

def apply_filters(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    if 'Year' in out and len(year_sel)>0: out = out[out['Year'].isin(year_sel)]
    if 'USCensusDivision' in out and len(div_sel)>0: out = out[out['USCensusDivision'].isin(div_sel)]
    if 'Race' in out and len(race_sel)>0: out = out[out['Race'].isin(race_sel)]
    if 'Gender' in out and len(gen_sel)>0: out = out[out['Gender'].isin(gen_sel)]
    if 'AgeGroup' in out and len(age_sel)>0: out = out[out['AgeGroup'].isin(age_sel)]
    if 'Urbanicity' in out and len(urb_sel)>0: out = out[out['Urbanicity'].isin(urb_sel)]
    return out

fdf = apply_filters(df)

# ----------------------------
# Navigation (단일 파일 멀티페이지)
# ----------------------------
pages = {
    "🏠 Overview": "overview",
    "🧹 Data & Cleaning": "data_cleaning",
    "📈 Univariate EDA": "univariate",
    "🔗 Bivariate EDA": "bivariate",
    "🧭 Temporal & Regional": "temporal_regional",
    "🧪 Modeling Plan": "modeling_plan",
    "ℹ️ About": "about",
}
page = st.sidebar.radio("Navigate", list(pages.keys()))

# ----------------------------
# Helper: 안전 집계
# ----------------------------
def safe_count(series):
    try:
        return series.size
    except:
        return len(series)

# ----------------------------
# Pages
# ----------------------------
if page == "🏠 Overview":
    st.title("🚑 Disparities in EMS-Reported Crash Injury Outcomes")
    st.markdown("""
    이 앱은 **NEMSIS(2018–2022)** EMS 데이터의 **기초/탐색적 분석(EDA)** 을 스토리라인으로 정리합니다.  
    - **핵심 질문:** 인종·성별·연령대 간 **인구 보정 전** 패턴을 우선 점검  
    - 이후 **ACS 5-year(2018–2022)** 인구를 병합해 100k당 사고율 분석 및 모델링으로 확장  
    - 발표/시연을 위한 **페이지별 내러티브 구성**
    """)

    st.subheader("Current Filters Summary")
    left, right = st.columns([1,1])
    with left:
        st.write("**Rows (after filters):**", len(fdf))
        if 'Year' in fdf: st.write("**Years:**", ", ".join(map(str, sorted(set(fdf['Year'].dropna())))))
        if 'USCensusDivision' in fdf: st.write("**Divisions:**", ", ".join(sorted(set(fdf['USCensusDivision'].dropna()))))
    with right:
        if 'Gender' in fdf: st.write("**Genders:**", ", ".join(sorted(set(fdf['Gender'].dropna()))))
        if 'Race' in fdf: st.write("**Races:**", ", ".join(sorted(set(fdf['Race'].dropna()))))
        if 'AgeGroup' in fdf: st.write("**Age Groups:**", ", ".join([str(x) for x in fdf['AgeGroup'].dropna().unique()]))

    st.info("다음 페이지로 이동해 기초 분포(단변량) → 관계(이변량) → 시간·지역 패턴 순으로 살펴보세요.")

elif page == "🧹 Data & Cleaning":
    st.title("🧹 Data Overview & Cleaning")
    st.markdown("""
    - 각 행은 **EMS가 응답한 개별 사건/환자**  
    - 주요 변수: `Gender`, `Race`, `AgeGroup`, `USCensusDivision`, `Urbanicity`, `Year` 등  
    - 이 페이지는 **결측/이상치 점검**과 **라벨 표준화 여부**를 빠르게 확인하기 위한 용도
    """)
    # 결측치 요약
    st.subheader("Missingness Snapshot")
    miss = fdf.isna().mean().sort_values(ascending=False).head(20)*100
    st.bar_chart(miss)

    st.subheader("Numeric Distribution Quick Check")
    num_cols = [c for c in fdf.columns if pd.api.types.is_numeric_dtype(fdf[c])]
    if len(num_cols):
        sel_num = st.selectbox("Numeric column", num_cols, index=0)
        fig = px.histogram(fdf, x=sel_num, nbins=50, title=f"Distribution of {sel_num}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("수치형 칼럼을 찾지 못했어요.")

elif page == "📈 Univariate EDA":
    st.title("📈 Univariate EDA — 단변량 분포")
    col1, col2 = st.columns(2)

    # Gender — Donut
    if 'Gender' in fdf:
        gender_counts = fdf.dropna(subset=['Gender']).groupby('Gender').size().reset_index(name='Count')
        with col1:
            fig = px.pie(gender_counts, names='Gender', values='Count', hole=0.4,
                         title="🚻 Crash Counts by Gender")
            fig.update_traces(textinfo='percent+label', pull=[0.04]*len(gender_counts))
            st.plotly_chart(fig, use_container_width=True)

    # Race — Bar
    if 'Race' in fdf:
        race_counts = (fdf.dropna(subset=['Race']).groupby('Race').size().reset_index(name='Count')
                         .sort_values('Count', ascending=False))
        with col2:
            fig = px.bar(race_counts, x='Race', y='Count', color='Race',
                         title="🌎 Crash Counts by Race")
            fig.update_layout(xaxis_tickangle=35)
            st.plotly_chart(fig, use_container_width=True)

    # AgeGroup — Bar
    st.divider()
    if 'AgeGroup' in fdf:
        age_counts = (fdf.dropna(subset=['AgeGroup']).groupby('AgeGroup').size().reset_index(name='Count'))
        if pd.api.types.is_categorical_dtype(age_counts['AgeGroup']):
            age_counts = age_counts.sort_values('AgeGroup')
        fig = px.bar(age_counts, x='AgeGroup', y='Count', text='Count', color='AgeGroup',
                     title="👶🧓 Crash Counts by Age Group")
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

elif page == "🔗 Bivariate EDA":
    st.title("🔗 Bivariate EDA — 두 변수 간 관계")
    tabs = st.tabs(["Gender × Race", "AgeGroup × Gender", "Race × Urbanicity"])

    # Gender × Race Heatmap
    with tabs[0]:
        if set(['Gender','Race']).issubset(fdf.columns):
            cross = fdf.dropna(subset=['Gender','Race']).groupby(['Gender','Race']).size().reset_index(name='Count')
            fig = px.density_heatmap(cross, x='Race', y='Gender', z='Count',
                                     color_continuous_scale='Viridis',
                                     title="Gender × Race — Crash Counts")
            fig.update_layout(xaxis_tickangle=35)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Gender / Race 칼럼이 필요합니다.")

    # AgeGroup × Gender Grouped Bar
    with tabs[1]:
        if set(['AgeGroup','Gender']).issubset(fdf.columns):
            ag = fdf.dropna(subset=['AgeGroup','Gender']).groupby(['AgeGroup','Gender']).size().reset_index(name='Count')
            # 순서 보장
            if 'AgeGroup' in ag and ag['AgeGroup'].dtype.name == 'category':
                ag = ag.sort_values('AgeGroup')
            fig = px.bar(ag, x='AgeGroup', y='Count', color='Gender', barmode='group',
                         title="Age Group × Gender — Crash Counts")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("AgeGroup / Gender 칼럼이 필요합니다.")

    # Race × Urbanicity Heatmap
    with tabs[2]:
        if set(['Race','Urbanicity']).issubset(fdf.columns):
            cu = fdf.dropna(subset=['Race','Urbanicity']).groupby(['Race','Urbanicity']).size().reset_index(name='Count')
            fig = px.density_heatmap(cu, x='Urbanicity', y='Race', z='Count',
                                     color_continuous_scale='Plasma',
                                     title="Race × Urbanicity — Crash Counts")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Race / Urbanicity 칼럼이 필요합니다.")

elif page == "🧭 Temporal & Regional":
    st.title("🧭 Temporal & Regional Patterns")
    sub1, sub2 = st.columns(2)

    # Year trend
    if 'Year' in fdf:
        year_counts = fdf.dropna(subset=['Year']).groupby('Year').size().reset_index(name='Count')
        # Year를 category처럼 보이게
        year_counts['Year'] = year_counts['Year'].astype('Int64')
        fig = px.line(year_counts, x='Year', y='Count', markers=True,
                      title='📅 Crash Counts by Year')
        fig.update_layout(xaxis=dict(dtick=1))
        sub1.plotly_chart(fig, use_container_width=True)

    # Division bar
    if 'USCensusDivision' in fdf:
        div_counts = (fdf.dropna(subset=['USCensusDivision']).groupby('USCensusDivision')
                        .size().reset_index(name='Count').sort_values('Count', ascending=False))
        fig = px.bar(div_counts, x='USCensusDivision', y='Count', color='USCensusDivision',
                     title='📍 Crash Counts by US Census Division')
        fig.update_layout(xaxis_tickangle=35, showlegend=False)
        sub2.plotly_chart(fig, use_container_width=True)

    st.info("※ 인구 보정(ACS merge) 이후에는 Division 단위의 **per 100k rate** 지도/히트맵으로 확장 예정.")

elif page == "🧪 Modeling Plan":
    st.title("🧪 Modeling Plan (Preview)")
    st.markdown("""
    **Phase 1 — Count model (Negative Binomial)**  
    - Target: `InjuryCount` (subgroup별 EMS 사건 수)  
    - Offset: `log(Population)` (ACS 5-year)  
    - Formula (예시):  
      ```
      InjuryCount ~ Race + Gender + AgeGroup + CensusRegion + Urbanicity + Year
      + offset(log(Population))
      ```
    **Phase 2 — Individual-level injury outcomes (Multinomial Logistic)**  
    - Target: Chief Complaint / Primary Symptom (다범주)  
    - Predictors: Race, Gender, Age, Region, Urbanicity, Year  
    - Weighted 버전: 인구구성 반영 (population_prop / sample_prop)
    """)
    st.success("지금 페이지들은 **인구 보정 전 탐색** 단계. 다음 스프린트에서 ACS 병합 후 per 100k 시각화 & 모델로 확장!")

elif page == "ℹ️ About":
    st.title("ℹ️ About this App")
    st.markdown("""
    - **Dataset:** NEMSIS EMS (2018–2022), *pre-normalization EDA*  
    - **Goal:** 인종·성별·연령대·지역별 패턴을 탐색하고, 인구 보정/모델링 방향을 제시  
    - **Author:** Jungbum “Ben” Cho @ MSU — Transportation Safety / Data Science
    """)
    st.caption("Tip: 사이드바에서 연도·지역·인종·성별 필터를 바꿔가며 스토리를 전개하세요.")
