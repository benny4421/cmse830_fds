
# 🚑 EMS-Reported Crash Injury Disparities (2018–2022)

**Goal.** Identify demographic disparities in EMS‐attended crash injuries across the U.S. by age, gender, race, region/division—and build a policy-facing dashboard.

**Deployed App.** https://cmse830fds-3zuewsuxruovjwtxwiyqh9.streamlit.app/

**Notebook.** `EMS_Analysis_Project (4).ipynb` (included in repo)

**Data Scale.** ~6,000,000 rows (research dataset).
**App/Repo Data.** For performance, the Streamlit app and GitHub CSV use a **1/60 sample (≈100k rows)**.

---

##  Project Overview

This project analyzes nationally representative EMS data (NEMSIS, 2018–2022) to examine disparities in crash injuries by **AgeGroup × Gender × Race × U.S. Census Division**.
To report **rates** (not just counts), the pipeline integrates **ACS 2018–2022 5-year estimates** to build population denominators aligned to `Division × Sex × Race × AgeGroup`.

---

##  Data Sources

1. **NEMSIS (2018–2022)** – EMS incidents with demographics, times, location context (used as main outcome data).
2. **U.S. Census / ACS 5-Year (2018–2022)** – Population denominators by `Division × Sex × Race × AgeGroup` (built via `censusdata` in Python).

> **Note:** The app ships with a **sampled CSV** (`sampled_ems_data_100k.csv`) for reproducibility and snappy interactivity. The full 6M dataset was used for the research-grade pipeline but is not distributed due to size.

---

##  Data Collection & Preparation (Rubric 1)

* **Two distinct sources**: NEMSIS (outcomes) + ACS 5-year (population).

* **Cleaning**

  * **Duplicate investigation** using `PcrKey` (unique incident ID). Found *semantic duplicates* where all fields matched except `Race` → treated as **data entry errors** and **removed** in the full dataset.
  * **Semantic missing values** normalized: strings like `"unknown"`, `"not recorded"`, `"n/a"` were mapped to proper `NaN` before analysis.
  * **Data types**: Standardized `Year` to integer; ordered `AgeGroup` as categorical (`0–24, 25–34, …, 85+`).
  * **Age unit consistency**: Dropped records where `Age Units != 'years'` (infants, months/days) because the analysis focuses on **AgeGroup bins**.

* **Encoding**

  * `Urbanicity` → **ordinal code** (`rural=0, suburban=1, urban=2, metro=3`).
  * `Possible Injury` → **one-hot** (`PossibleInjury_yes/no/unknown`).
  * (For modeling/imputation) Light one-hot on `Gender`, `Race`, `USCensusDivision` as needed.

---

##  Exploratory Data Analysis & Visualization (Rubric 2)

**App page “ Visualization” (raw counts; pre-normalization):**

* **Pie / Donut** – Crash counts by **Gender**.
* **Bar** – Crash counts by **Race** (sorted).
* **Line** – Crash counts by **Year** (2018–2022).
* **Bar** – Crash counts by **U.S. Census Division**.

**Notes**

* The dashboard **explicitly warns** that these are **raw counts** (not yet population-adjusted).
* Statistical summaries and count tables are shown inline; typed columns and ordered `AgeGroup` ensure appropriate encodings in plots.

---

##  Data Processing & Missing Data Strategy (Rubric 3)

### What I tried (and why I changed)

* **Initial idea: Mean imputation** for `ageinyear`

  * Visualized before/after → produced a huge **spike at the mean**, biasing `AgeGroup` bins.
  * **Rejected** due to distribution distortion.

* **Final approach: Regression-based stochastic imputation** (scalable for 6M rows)

  * **Target**: `ageinyear`
  * **Predictors** (minimal, fast, interpretable): `Urbanicity_code`, `Year`, `EMSTotalCallTimeMin`
  * **Procedure**: Fit linear regression on complete cases → predict missing ages → add random noise from residual std (**stochastic**) → clip to `[0,110]`.
  * **Why**: Preserves variability and relationships without the compute cost of MICE on 6M rows.

> When predictors for missing rows had NaNs, we **imputed only the predictors** with **median** (fast & safe) so the regression model can run. The **true imputation target** is `ageinyear`, not the predictors.

---

## 🏛️ Population Denominators & ACS Merge (Dashboard page “🏛️ US Census Data Merging”)

* Built a population table using **ACS 2018–2022 5-Year** via `censusdata`:

  * Tables: Race-iterated **B01001** series (e.g., `B01001B/C/D/E/H/I`) by **state**, then mapped to **U.S. Census Divisions**, and aggregated to:
  * **`Division × Sex × Race × AgeGroup`** with AgeGroup constructed from B01001 male/female line items:

    * `0–24`: Under5, 5–9, 10–14, 15–17, 18–19, 20–24
    * `25–34`, `35–44`, `45–54`, `55–64`, `65–74`, `75–84`, `85+`
* **QA check**: Sum of AgeGroup buckets equals ACS **sex totals** per Division & Race (Diff = 0), ensuring correctness.
* **Next**: After aligning labels (`USCensusDivision/Gender/Race/AgeGroup` → `Division/Sex/Race/AgeGroup`), we left-join `Population` into the EMS aggregates and compute:

  * `Rate_per_100k = 100000 × InjuryCount / Population`
  * `offset = log(Population)` for count models.

---

##  Modeling Plan

1. **Count model** (population-adjusted)

```r
glm.nb(InjuryCount ~ Race + Gender + AgeGroup + CensusRegion + RuralUrban + Year
       + offset(log(Population)))
```

(Implemented in Python with `statsmodels` NB GLM.)

2. **Individual-level outcomes** 

* **Multinomial logistic regression**, with unweighted vs **population-weighted** variants.
* `weight = population_prop / EMS_sample_prop` to describe disparities **relative to population**.

---

##  Streamlit App (Rubric 4)

**Pages**

* ** Overview** – Project goals, audiences, and dataset preview.
* ** Handling Data Duplicates** – `PcrKey` uniqueness audit; semantic duplicate diagnosis; remediation rationale.
* ** Handling Missing Values** – Semantic null cleaning; heatmaps before/after; **mean imputation failure** demo; `Age Units` analysis & decision.
* ** US Census Data Merging** – Why rates matter; target population schema; status & next steps.
* ** Visualization** – 4 key charts with interactive Plotly components.

**Interactive elements**

* Sidebar **page navigation**
* Interactive **Plotly** charts (hover, filter by page choice)
* In-app **dataframes** and expandable diagnostic sections

**Deployment**

* Deployed publicly via Streamlit Cloud .
* Caching (`@st.cache_data`) keeps UI responsive on 100k sample.

---

##  GitHub Repository (Rubric 5)

**Suggested structure**

```
.
├── app.py                         # Streamlit app (shown above)
├── EMS_Analysis_Project (4).ipynb # Main analysis notebook
├── sampled_ems_data_100k.csv      # 1/60 sampled data for app/repo
├── data/                          # (optional) scripts or smaller assets
├── models/                        # (optional) saved model summaries
└── README.md                      # This file
```

**How to run locally**

```bash

# 1) install deps
pip install -r requirements.txt


# 2) run app
streamlit run app.py
```





