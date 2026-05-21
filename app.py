import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# ── Page config with logo ──────────────────────────────────────────────────────
logo = Image.open("src/UnderCode.png")
st.set_page_config(
    page_title="RePort · Recruitment Support",
    page_icon=logo,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');
  :root {
    --bg:#0d0f14;--surface:#151820;--card:#1c2030;--border:#262c3d;
    --accent:#4f8ef7;--accent2:#f7854f;--success:#4fcf8e;--danger:#f76f6f;
    --muted:#6b7592;--text:#e8eaf2;--subtext:#9ba3bc;
  }
  html,body,[class*="css"]{background:var(--bg)!important;color:var(--text)!important;font-family:'DM Sans',sans-serif;}
  section[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border);}
  section[data-testid="stSidebar"] *{color:var(--text)!important;}
  .stTabs [data-baseweb="tab-list"]{background:var(--surface);border-bottom:1px solid var(--border);gap:0;padding:0 24px;}
  .stTabs [data-baseweb="tab"]{background:transparent;color:var(--subtext)!important;font-family:'DM Sans',sans-serif;font-weight:500;font-size:14px;padding:14px 24px;border-bottom:2px solid transparent;}
  .stTabs [aria-selected="true"]{color:var(--accent)!important;border-bottom:2px solid var(--accent)!important;background:transparent!important;}
  .stTabs [data-baseweb="tab-panel"]{padding:28px 4px 0;}
  div[data-testid="metric-container"]{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:20px 24px;}
  div[data-testid="metric-container"] label{color:var(--subtext)!important;font-size:12px;text-transform:uppercase;letter-spacing:.06em;}
  div[data-testid="metric-container"] [data-testid="stMetricValue"]{color:var(--text)!important;font-size:28px;font-weight:600;}
  .stButton>button{background:var(--accent)!important;color:#fff!important;border:none!important;border-radius:8px!important;font-weight:600!important;padding:10px 28px!important;transition:opacity .2s;}
  .stButton>button:hover{opacity:.85;}
  [data-testid="stFileUploader"]{background:var(--card);border:1px dashed var(--border);border-radius:12px;padding:16px;}
  [data-testid="stDataFrame"]{border:1px solid var(--border);border-radius:12px;overflow:hidden;}
  hr{border-color:var(--border)!important;}
  .section-title{font-family:'DM Serif Display',serif;font-size:20px;color:var(--text);margin-bottom:2px;}
  .section-sub{font-size:13px;color:var(--subtext);margin-bottom:18px;}
  .tab-desc{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:16px 20px;margin-bottom:24px;font-size:14px;color:var(--subtext);line-height:1.7;}
  .tab-desc strong{color:var(--text);}
  .badge-accept{display:inline-block;background:rgba(79,207,142,.15);color:#4fcf8e;border:1px solid rgba(79,207,142,.35);padding:6px 18px;border-radius:100px;font-weight:600;font-size:15px;}
  .badge-reject{display:inline-block;background:rgba(247,111,111,.15);color:#f76f6f;border:1px solid rgba(247,111,111,.35);padding:6px 18px;border-radius:100px;font-weight:600;font-size:15px;}
  .hero{background:linear-gradient(135deg,#1c2030 0%,#141828 100%);border:1px solid var(--border);border-radius:18px;padding:32px 36px;margin-bottom:28px;}
  .hero h1{font-family:'DM Serif Display',serif;font-size:32px;margin:0 0 6px;}
  .hero p{color:var(--subtext);font-size:15px;margin:0;}
  .oar-hint{background:rgba(79,142,247,.08);border:1px solid rgba(79,142,247,.25);border-radius:8px;padding:10px 14px;font-size:13px;color:var(--subtext);margin-top:8px;}
  .insight-card{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:18px 20px;margin-bottom:10px;}
  .insight-icon{font-size:22px;margin-bottom:6px;}
  .insight-title{font-size:13px;font-weight:600;color:var(--text);margin:0 0 6px;}
  .insight-body{font-size:13px;color:var(--subtext);line-height:1.6;margin:0;}
  .insight-tag-good{display:inline-block;background:rgba(79,207,142,.15);color:#4fcf8e;border:1px solid rgba(79,207,142,.3);padding:2px 10px;border-radius:100px;font-size:11px;font-weight:600;margin-bottom:8px;}
  .insight-tag-warn{display:inline-block;background:rgba(247,133,79,.15);color:#f7854f;border:1px solid rgba(247,133,79,.3);padding:2px 10px;border-radius:100px;font-size:11px;font-weight:600;margin-bottom:8px;}
  .insight-tag-bad{display:inline-block;background:rgba(247,111,111,.15);color:#f76f6f;border:1px solid rgba(247,111,111,.3);padding:2px 10px;border-radius:100px;font-size:11px;font-weight:600;margin-bottom:8px;}
  .whatif-card{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:20px 24px;}
  .whatif-winner{border:1px solid rgba(79,207,142,.4)!important;background:rgba(79,207,142,.05)!important;}
  .whatif-label{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--subtext);margin:0 0 4px;}
  .whatif-val{font-size:18px;font-weight:600;color:var(--text);}
  .compare-row{display:flex;justify-content:space-between;padding:8px 0;border-bottom:0.5px solid var(--border);font-size:13px;}
  .compare-row:last-child{border-bottom:none;}
  .compare-label{color:var(--subtext);}
  .compare-val{color:var(--text);font-weight:500;}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
DEPARTMENTS = ["Engineering","Sales","Product","HR","Marketing","Finance"]
SOURCES     = ["Referral","LinkedIn","Recruiter","Job Portal"]

DEPT_JOBS = {
    "Engineering": ["Software Engineer","DevOps Engineer","Backend Developer","Data Engineer"],
    "Finance":     ["Accountant","Finance Manager","Financial Analyst","Payroll Specialist"],
    "HR":          ["Talent Acquisition","HR Coordinator","Recruitment Specialist","HR Manager"],
    "Marketing":   ["Marketing Specialist","Social Media Manager","Content Strategist","SEO Analyst"],
    "Product":     ["UX Designer","Product Manager","UI Designer","Product Analyst"],
    "Sales":       ["Account Executive","Business Development Manager","Sales Associate","Sales Representative"],
}

PT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans",color="#9ba3bc"),
    xaxis=dict(gridcolor="#1e2537",linecolor="#262c3d",zeroline=False),
    yaxis=dict(gridcolor="#1e2537",linecolor="#262c3d",zeroline=False),
    colorway=["#4f8ef7","#f7854f","#4fcf8e","#f76f6f","#a78bfa","#fbbf24","#38bdf8"],
)

# ── Load resources ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_bundle():
    return joblib.load("models/recruitment_model.joblib")

@st.cache_data
def load_data():
    return pd.read_csv("data/raw/recruitment_efficiency_improved.csv")

bundle         = load_bundle()
MODEL          = bundle['model']
FINAL_FEATURES = bundle['final_features']
DEPT_MAP       = bundle['dept_map']
JOBTITLE_MAP   = bundle['jobtitle_map']
SOURCE_MAP     = bundle['source_map']
DEPTSRC_MAP    = bundle['deptsrc_map']
GLOBAL_MEAN    = bundle['global_mean']
SMOOTH_K       = bundle['smooth_k']
TIME_MEAN      = bundle['time_mean_train']
TIME_STD       = bundle['time_std_train']
COST_MEAN      = bundle['cost_mean_train']
COST_STD       = bundle['cost_std_train']
DEPT_MEDIANS   = bundle['dept_medians']   # columns: dept_median_cph, dept_median_tth, dept_median_oar

raw_df = load_data()

# ── Feature engineering ────────────────────────────────────────────────────────
def engineer_features(df):
    d = df.copy()
    d['cost_per_applicant'] = d['cost_per_hire'] / d['num_applicants']
    d['cost_per_day']       = d['cost_per_hire'] / d['time_to_hire_days']
    d['applicants_per_day'] = d['num_applicants'] / d['time_to_hire_days']
    d['is_senior_role']     = d['job_title'].str.contains(
        r'\b(manager|specialist|executive|strategist)\b', case=False, na=False).astype(int)
    d['dept_src_key']       = d['department'].astype(str)+'_'+d['source'].astype(str)
    d['difficulty_additive'] = (
        (d['time_to_hire_days']-TIME_MEAN)/TIME_STD +
        (d['cost_per_hire']    -COST_MEAN)/COST_STD
    ) / 2
    d2 = d.join(DEPT_MEDIANS, on='department')
    if 'offer_acceptance_rate' in d.columns:
        d['drei'] = (
            (d['cost_per_hire'].values          < d2['dept_median_cph'].values) &
            (d['time_to_hire_days'].values       < d2['dept_median_tth'].values) &
            (d['offer_acceptance_rate'].values   > d2['dept_median_oar'].values)
        ).astype(int)
    else:
        d['drei'] = 0
    d['dept_oar_mean']        = d['department'].map(DEPT_MAP).fillna(GLOBAL_MEAN)
    d['jobtitle_oar_mean']    = d['job_title'].map(JOBTITLE_MAP).fillna(GLOBAL_MEAN)
    d['source_oar_mean']      = d['source'].map(SOURCE_MAP).fillna(GLOBAL_MEAN)
    d['dept_source_oar_mean'] = d['dept_src_key'].map(DEPTSRC_MAP).fillna(GLOBAL_MEAN)
    out = d[FINAL_FEATURES].copy()
    out.replace([np.inf,-np.inf], np.nan, inplace=True)
    for col in out.columns:
        med = out[col].median()
        out[col].fillna(med if not np.isnan(med) else 0, inplace=True)
    return out

def predict(df_input):
    X = engineer_features(df_input)
    return MODEL.predict(X), MODEL.predict_proba(X)

def validate_batch(df, required_cols):
    """
    Validates a batch upload DataFrame.
    Returns a dict:
      - errors:   list of blocking issues (prevent prediction)
      - warnings: list of non-blocking issues (prediction runs with fixes)
      - cleaned:  cleaned DataFrame ready for prediction (if no errors)
    """
    errors   = []
    warnings = []
    df       = df.copy()

    # ── 1. Empty file ────────────────────────────────────────────────────
    if df.empty:
        errors.append("The uploaded file is empty. Please upload a CSV with at least one row of data.")
        return {"errors": errors, "warnings": warnings, "cleaned": None}

    # ── 2. Missing required columns ──────────────────────────────────────
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        errors.append(f"Missing required column(s): {', '.join(f'`{c}`' for c in missing)}. "
                      f"Please download the template to see the correct format.")
        return {"errors": errors, "warnings": warnings, "cleaned": None}

    # ── 3. Completely empty required columns ─────────────────────────────
    for col in required_cols:
        if df[col].isna().all():
            errors.append(f"Column `{col}` is entirely empty. Please provide values for all required columns.")

    if errors:
        return {"errors": errors, "warnings": warnings, "cleaned": None}

    # ── 4. Numeric type validation ───────────────────────────────────────
    numeric_cols = ['num_applicants','time_to_hire_days','cost_per_hire','offer_acceptance_rate']
    for col in numeric_cols:
        non_numeric = pd.to_numeric(df[col], errors='coerce').isna() & df[col].notna()
        if non_numeric.any():
            count = non_numeric.sum()
            errors.append(f"Column `{col}` contains {count} non-numeric value(s) "
                          f"(e.g. row {non_numeric.idxmax()+2}). This column must contain numbers only.")
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if errors:
        return {"errors": errors, "warnings": warnings, "cleaned": None}

    # ── 5. Zero / negative values ────────────────────────────────────────
    for col, label in [('num_applicants','Number of Applicants'),
                       ('time_to_hire_days','Time to Hire'),
                       ('cost_per_hire','Cost per Hire')]:
        bad = df[col] <= 0
        if bad.any():
            errors.append(f"`{label}` has {bad.sum()} row(s) with zero or negative values. "
                          f"All values must be greater than 0.")

    if errors:
        return {"errors": errors, "warnings": warnings, "cleaned": None}

    # ── 6. OAR range validation ──────────────────────────────────────────
    oar_out = (df['offer_acceptance_rate'] < 0.00) | (df['offer_acceptance_rate'] > 1.00)
    if oar_out.any():
        count = oar_out.sum()
        warnings.append(f"{count} row(s) have `offer_acceptance_rate` outside the valid range (0.00–1.00). "
                        f"These rows will be clamped to the nearest valid value (0.00 or 1.00) before prediction.")
        df['offer_acceptance_rate'] = df['offer_acceptance_rate'].clip(0.00, 1.00)

    # ── 7. Unknown categorical values ────────────────────────────────────
    valid_depts   = set(DEPARTMENTS)
    valid_sources = set(SOURCES)
    valid_jobs    = set(j for jobs in DEPT_JOBS.values() for j in jobs)

    unk_dept = ~df['department'].isin(valid_depts)
    if unk_dept.any():
        uniq = df.loc[unk_dept, 'department'].unique().tolist()
        warnings.append(f"{unk_dept.sum()} row(s) have unrecognized department value(s): "
                        f"{', '.join(str(u) for u in uniq[:5])}. "
                        f"Target encoding will fall back to the global average for these rows.")

    unk_src = ~df['source'].isin(valid_sources)
    if unk_src.any():
        uniq = df.loc[unk_src, 'source'].unique().tolist()
        warnings.append(f"{unk_src.sum()} row(s) have unrecognized source value(s): "
                        f"{', '.join(str(u) for u in uniq[:5])}. "
                        f"Target encoding will fall back to the global average for these rows.")

    unk_job = ~df['job_title'].isin(valid_jobs)
    if unk_job.any():
        uniq = df.loc[unk_job, 'job_title'].unique().tolist()
        warnings.append(f"{unk_job.sum()} row(s) have unrecognized job title(s): "
                        f"{', '.join(str(u) for u in uniq[:5])}. "
                        f"Target encoding will fall back to the global average for these rows.")

    # ── 8. Missing values in required columns (partial) ──────────────────
    for col in required_cols:
        n_null = df[col].isna().sum()
        if n_null > 0:
            warnings.append(f"`{col}` has {n_null} missing value(s). "
                            f"These rows will use the column median as a fallback.")

    return {"errors": errors, "warnings": warnings, "cleaned": df}

def generate_insights(row, dept_medians, source_map, global_mean):
    """
    Generate detailed HR-friendly insights for a single candidate prediction.
    row: dict with department, job_title, source, num_applicants,
         time_to_hire_days, cost_per_hire, offer_acceptance_rate
    """
    dept     = row['department']
    source   = row['source']
    apps     = row['num_applicants']
    tth      = row['time_to_hire_days']
    cph      = row['cost_per_hire']
    oar      = row['offer_acceptance_rate']

    med_cph  = float(dept_medians.loc[dept, 'dept_median_cph']) if dept in dept_medians.index else 5215
    med_tth  = float(dept_medians.loc[dept, 'dept_median_tth']) if dept in dept_medians.index else 47
    med_oar  = float(dept_medians.loc[dept, 'dept_median_oar']) if dept in dept_medians.index else global_mean

    src_oar  = source_map.get(source, global_mean)
    cpa      = cph / apps if apps > 0 else 0
    apd      = apps / tth if tth > 0 else 0

    insights = []

    # ① Cost Efficiency
    cost_diff = cph - med_cph
    if cph < med_cph * 0.85:
        tag  = "good"
        body = (f"Your cost per hire (${cph:,.0f}) is well below the {dept} department median "
                f"(${med_cph:,.0f}), indicating an efficient sourcing process. "
                f"Keep leveraging the same channels to maintain this performance.")
    elif cph <= med_cph * 1.10:
        tag  = "warn"
        body = (f"Your cost per hire (${cph:,.0f}) is close to the {dept} department median "
                f"(${med_cph:,.0f}). There is some room to optimize — consider whether external "
                f"recruiters or agency fees can be reduced without impacting quality.")
    else:
        tag  = "bad"
        body = (f"Your cost per hire (${cph:,.0f}) is significantly above the {dept} department "
                f"median (${med_cph:,.0f}), an overspend of ${cost_diff:,.0f}. "
                f"Review reliance on external agencies and consider shifting budget toward "
                f"direct channels such as Referral or LinkedIn.")
    insights.append({"icon":"💰","title":"Cost Efficiency","tag":tag,"body":body})

    # ② Time Efficiency
    time_diff = tth - med_tth
    if tth < med_tth * 0.85:
        tag  = "good"
        body = (f"Time to hire ({tth} days) is well below the {dept} median ({med_tth:.0f} days). "
                f"A fast process signals strong pipeline readiness and a positive candidate experience, "
                f"both of which support higher offer acceptance.")
    elif tth <= med_tth * 1.15:
        tag  = "warn"
        body = (f"Time to hire ({tth} days) is near the {dept} median ({med_tth:.0f} days). "
                f"Consider streamlining interview rounds or reducing decision lag between stages "
                f"to keep candidates engaged and prevent drop-offs.")
    else:
        tag  = "bad"
        body = (f"Time to hire ({tth} days) exceeds the {dept} median by {time_diff:.0f} days. "
                f"Prolonged processes are a leading cause of offer rejection — candidates often "
                f"accept competing offers during long waits. Aim to reduce to under {med_tth:.0f} days "
                f"by limiting interview rounds and accelerating internal approvals.")
    insights.append({"icon":"⏱️","title":"Time to Hire","tag":tag,"body":body})

    # ③ Source Quality
    best_src  = max(source_map, key=source_map.get)
    best_oar  = source_map[best_src]
    if src_oar >= global_mean * 1.02:
        tag  = "good"
        body = (f"{source} has an average OAR of {src_oar:.1%} in your dataset, which is above "
                f"the global average ({global_mean:.1%}). This is a strong acquisition channel "
                f"for your organization — continue prioritizing it.")
    elif src_oar >= global_mean * 0.97:
        tag  = "warn"
        body = (f"{source} has an average OAR of {src_oar:.1%}, close to the global average "
                f"({global_mean:.1%}). If acceptance rates remain flat, consider testing "
                f"{best_src} (avg OAR: {best_oar:.1%}) for this role type.")
    else:
        tag  = "bad"
        body = (f"{source} has a below-average OAR ({src_oar:.1%} vs global average {global_mean:.1%}). "
                f"Consider switching to {best_src} (avg OAR: {best_oar:.1%}), which consistently "
                f"yields better acceptance outcomes in your recruitment data.")
    insights.append({"icon":"📣","title":"Source Quality","tag":tag,"body":body})

    # ④ Pipeline Volume
    if apd >= 4.0:
        tag  = "warn"
        body = (f"With {apps} applicants over {tth} days ({apd:.1f} applicants/day), your pipeline "
                f"volume is very high. A large volume of low-intent applications can dilute match "
                f"quality. Consider tightening the job description or adding a pre-screening step "
                f"to improve candidate-role fit.")
    elif apd >= 1.5:
        tag  = "good"
        body = (f"Pipeline velocity of {apd:.1f} applicants/day is healthy, suggesting the role "
                f"is attracting genuine interest. Maintaining this balance between volume and "
                f"quality is key to sustaining high offer acceptance rates.")
    else:
        tag  = "bad"
        body = (f"With only {apd:.1f} applicants/day over {tth} days, pipeline velocity is low. "
                f"This may indicate an unattractive job description, a niche skill set, or limited "
                f"sourcing reach. Consider refreshing the JD, broadening the source mix, or "
                f"offering more competitive compensation to increase applicant interest.")
    insights.append({"icon":"👥","title":"Pipeline Volume","tag":tag,"body":body})

    # ⑤ Overall Difficulty
    diff = ((tth - 47.19) / 23.86 + (cph - 5214.8) / 2731) / 2
    if diff < -0.3:
        tag  = "good"
        body = (f"Overall recruitment difficulty is low — both cost and time are below average. "
                f"This is an efficient process that creates a positive candidate experience. "
                f"Document this approach and replicate it for similar roles.")
    elif diff <= 0.3:
        tag  = "warn"
        body = (f"Overall recruitment difficulty is moderate. The process is neither notably "
                f"efficient nor problematic. Focus on the specific areas flagged above to push "
                f"the outcome toward High OAR.")
    else:
        tag  = "bad"
        body = (f"Overall recruitment difficulty is high — both cost and time are above average, "
                f"which tends to frustrate candidates and increase the risk of offer rejection. "
                f"Prioritize reducing time-to-hire first, as it has the most direct impact on "
                f"candidate experience and acceptance likelihood.")
    insights.append({"icon":"🎯","title":"Overall Recruitment Difficulty","tag":tag,"body":body})

    return insights

# ── Session state for dynamic job title ───────────────────────────────────────
if 'selected_dept' not in st.session_state:
    st.session_state.selected_dept = DEPARTMENTS[0]

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(logo, width=120)
    st.markdown("## RePort")
    st.markdown("<span style='color:#6b7592;font-size:13px'>Recruitment Support Platform</span>", unsafe_allow_html=True)
    st.divider()
    st.markdown("**📂 Custom Data**")
    uploaded = st.file_uploader("Upload a CSV to override", type=["csv"])
    st.divider()
    st.markdown("**🔍 Filters — Overview Tab**")
    dept_filter = st.multiselect("Department", DEPARTMENTS, default=DEPARTMENTS)
    src_filter  = st.multiselect("Source",     SOURCES,     default=SOURCES)
    st.divider()
    st.markdown("<span style='color:#6b7592;font-size:12px'>Model: XGBoost + SMOTE<br>Target: OAR ≥ 0.70 → High OAR<br>AUC-ROC: 0.71 · n=5,000</span>", unsafe_allow_html=True)

# ── Data ───────────────────────────────────────────────────────────────────────
df_base = pd.read_csv(uploaded) if uploaded else raw_df.copy()
df = df_base[df_base["department"].isin(dept_filter) & df_base["source"].isin(src_filter)].copy()
df['oar_class'] = np.where(df['offer_acceptance_rate']>=0.70,'High OAR (≥0.70)','Low OAR (<0.70)')

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>RePort — Recruitment Support</h1>
  <p>RePort is a comprehensive recruitment support platform designed to help HR teams monitor efficiency,
  analyze hiring pipelines, track key metrics, and predict candidate offer decisions with machine learning — all in one place.</p>
</div>""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📊  Candidate Overview","🎯  Candidate Predictor"])

# ════════════════════════════════════════════════════════
#  TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════
with tab1:

    # Change #3 — HR-friendly tab description
    st.markdown("""
    <div class="tab-desc">
    This section provides a complete picture of your recruitment data. Use it to monitor key hiring metrics —
    such as how much it costs to hire, how long the process takes, and how often candidates accept your offers.
    The charts below break down these metrics by <strong>department</strong>, <strong>job title</strong>, and
    <strong>recruitment source</strong>, helping HR teams quickly spot patterns, compare performance, and identify
    which channels or roles need attention.
    </div>""", unsafe_allow_html=True)

    avg_cph      = df['cost_per_hire'].mean()
    avg_tth      = df['time_to_hire_days'].mean()
    avg_oar      = df['offer_acceptance_rate'].mean()
    high_oar_pct = (df['offer_acceptance_rate']>=0.70).mean()*100

    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Total Records",             f"{len(df):,}")
    k2.metric("Avg Cost per Hire",         f"${avg_cph:,.0f}")
    k3.metric("Avg Time to Hire",          f"{avg_tth:.1f} days")
    k4.metric("Avg Offer Acceptance Rate", f"{avg_oar:.1%}")
    k5.metric("High OAR Rate (≥70%)",      f"{high_oar_pct:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 1
    c1,c2 = st.columns(2)
    with c1:
        st.markdown('<p class="section-title">Records by Source</p><p class="section-sub">Candidate volume and OAR class per acquisition channel</p>', unsafe_allow_html=True)
        src_grp = df.groupby(['source','oar_class']).size().reset_index(name='count')
        fig = px.bar(src_grp, x='source', y='count', color='oar_class', barmode='group',
                     color_discrete_map={'High OAR (≥0.70)':'#4fcf8e','Low OAR (<0.70)':'#f76f6f'})
        fig.update_layout(**PT, margin=dict(t=10,b=10,l=0,r=0),
                          legend=dict(title='',orientation='h',y=1.08),
                          xaxis_title='', yaxis_title='Candidates')
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<p class="section-title">Records by Department</p><p class="section-sub">Average Offer Acceptance Rate per department</p>', unsafe_allow_html=True)
        dept_grp = df.groupby('department').agg(avg_oar=('offer_acceptance_rate','mean')).reset_index().sort_values('avg_oar',ascending=False)
        fig = go.Figure(go.Bar(
            x=dept_grp['department'], y=dept_grp['avg_oar']*100,
            marker_color='#4f8ef7',
            text=(dept_grp['avg_oar']*100).round(1).astype(str)+'%', textposition='outside',
        ))
        fig.update_layout(**PT, margin=dict(t=10,b=10,l=0,r=0),
                          yaxis_title='Avg OAR (%)', xaxis_title='',
                          yaxis_range=[0, dept_grp['avg_oar'].max()*115])
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    # Row 2
    c3,c4 = st.columns(2)
    with c3:
        st.markdown('<p class="section-title">Records by Job Title</p><p class="section-sub">Candidate volume by role (top 15)</p>', unsafe_allow_html=True)
        jt_grp = df.groupby('job_title').size().reset_index(name='count').sort_values('count').tail(15)
        fig = go.Figure(go.Bar(
            x=jt_grp['count'], y=jt_grp['job_title'], orientation='h',
            marker_color='#a78bfa', text=jt_grp['count'], textposition='outside',
        ))
        # Change #1 — right margin increased significantly so labels are never clipped
        fig.update_layout(**PT, margin=dict(t=10,b=10,l=0,r=70),
                          xaxis_title='Candidates', yaxis_title='', height=420,
                          xaxis_range=[0, jt_grp['count'].max()*1.20])
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        st.markdown('<p class="section-title">Offer Acceptance Rate Distribution</p><p class="section-sub">Full OAR spread with 0.70 classification threshold</p>', unsafe_allow_html=True)
        fig = go.Figure()
        for cls,color in [('High OAR (≥0.70)','#4fcf8e'),('Low OAR (<0.70)','#f76f6f')]:
            sub = df[df['oar_class']==cls]
            fig.add_trace(go.Histogram(x=sub['offer_acceptance_rate'],name=cls,
                                       marker_color=color,opacity=0.75,nbinsx=30))
        fig.add_vline(x=0.70,line_dash='dash',line_color='#f7854f',line_width=2,
                      annotation_text='Threshold 0.70',annotation_font_color='#f7854f')
        fig.update_layout(**PT,barmode='overlay',height=420,
                          margin=dict(t=10,b=10,l=0,r=0),
                          xaxis_title='Offer Acceptance Rate',yaxis_title='Count',
                          legend=dict(orientation='h',y=1.08))
        st.plotly_chart(fig, use_container_width=True)

    # Row 3
    c5,c6 = st.columns(2)
    with c5:
        st.markdown('<p class="section-title">Cost & Time to Hire</p><p class="section-sub">Distribution by OAR class</p>', unsafe_allow_html=True)
        fig = make_subplots(rows=1,cols=2,subplot_titles=['Cost per Hire ($)','Time to Hire (days)'])
        for cls,color in [('High OAR (≥0.70)','#4fcf8e'),('Low OAR (<0.70)','#f76f6f')]:
            sub = df[df['oar_class']==cls]
            fig.add_trace(go.Box(y=sub['cost_per_hire'],name=cls,marker_color=color,
                                 showlegend=(cls=='High OAR (≥0.70)')),row=1,col=1)
            fig.add_trace(go.Box(y=sub['time_to_hire_days'],name=cls,marker_color=color,
                                 showlegend=False),row=1,col=2)
        fig.update_layout(**PT,margin=dict(t=50,b=10,l=0,r=0),
                          legend=dict(orientation='h',y=1.15),height=390)
        fig.update_xaxes(showgrid=False)
        st.plotly_chart(fig, use_container_width=True)

    with c6:
        st.markdown('<p class="section-title">XGBoost Feature Importances</p><p class="section-sub">Engineered features ranked by predictive power</p>', unsafe_allow_html=True)
        fi_vals = MODEL.feature_importances_
        fi_df   = pd.DataFrame({'feature': FINAL_FEATURES, 'importance': fi_vals}).sort_values('importance')
        fig = go.Figure(go.Bar(
            x=fi_df['importance'], y=fi_df['feature'], orientation='h',
            marker_color='#f7854f',
            text=fi_df['importance'].round(3).astype(str), textposition='outside',
        ))
        fig.update_layout(**PT,margin=dict(t=10,b=10,r=80,l=0),height=390,
                          xaxis_title='Importance',yaxis_title='',
                          xaxis_range=[0, fi_df['importance'].max()*1.25])
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    # KPI by source
    st.markdown("---")
    st.markdown('<p class="section-title">KPI Summary by Source</p><p class="section-sub">Average Cost per Hire, Time to Hire, and OAR per acquisition channel</p>', unsafe_allow_html=True)
    src_kpi = df.groupby('source').agg(
        avg_cost=('cost_per_hire','mean'),
        avg_time=('time_to_hire_days','mean'),
        avg_oar=('offer_acceptance_rate','mean'),
    ).reset_index().sort_values('avg_oar',ascending=False)
    fig = make_subplots(rows=1,cols=3,subplot_titles=['Avg Cost per Hire ($)','Avg Time to Hire (days)','Avg OAR (%)'])
    colors = ['#4f8ef7','#f7854f','#4fcf8e','#a78bfa']
    for i,(col,sfx) in enumerate([('avg_cost','$'),('avg_time',' d'),('avg_oar','%')],1):
        vals = src_kpi[col]*(100 if col=='avg_oar' else 1)
        fig.add_trace(go.Bar(x=src_kpi['source'],y=vals,marker_color=colors[:len(src_kpi)],
                             text=vals.round(1).astype(str)+sfx,textposition='outside',
                             showlegend=False),row=1,col=i)
    fig.update_layout(**PT,margin=dict(t=30,b=10,l=0,r=0),height=340)
    fig.update_traces(marker_line_width=0)
    st.plotly_chart(fig, use_container_width=True)

    # Raw table
    st.markdown("---")
    st.markdown('<p class="section-title">Candidate Records</p>', unsafe_allow_html=True)
    show_cols=['recruitment_id','department','job_title','source',
               'num_applicants','time_to_hire_days','cost_per_hire','offer_acceptance_rate','oar_class']
    st.dataframe(df[show_cols].reset_index(drop=True), use_container_width=True, height=300)
    st.download_button("⬇ Download Filtered Data",
                       df.to_csv(index=False).encode(),"filtered_data.csv","text/csv")

# ════════════════════════════════════════════════════════
#  TAB 2 — PREDICTOR
# ════════════════════════════════════════════════════════
with tab2:

    # Change #3 & #4 — HR-friendly description, includes OAR threshold, removes technical info-box
    st.markdown("""
    <div class="tab-desc">
    This section uses a machine learning model to predict whether a recruitment process is likely to result in a
    <strong>High OAR (≥ 0.70)</strong> or <strong>Low OAR (&lt; 0.70)</strong>. Simply fill in the details for
    a candidate or role — such as department, source, number of applicants, time to hire, and expected cost —
    and the model will instantly estimate the likelihood of the candidate accepting the offer.
    You can also upload a CSV file to run predictions for multiple candidates at once.
    </div>""", unsafe_allow_html=True)

    mode = st.radio("", ["🧑 Single Candidate","📋 Batch Upload"], horizontal=True, label_visibility="collapsed")
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Single ──────────────────────────────────────────────────────────────
    if mode == "🧑 Single Candidate":
        st.markdown('<p class="section-title">Single Candidate Predictor</p><p class="section-sub">Enter recruitment details to predict offer acceptance outcome</p>', unsafe_allow_html=True)

        # Department outside the form — job titles update instantly on change
        dept_sel     = st.selectbox("Department", DEPARTMENTS + ["✏️ Other (type manually)"], key="pred_dept")
        if dept_sel == "✏️ Other (type manually)":
            department = st.text_input("Enter Department manually",
                placeholder="e.g. Legal, Customer Success",
                help="This value is not in the training data — dept medians and target encoding will use global averages as fallback.",
                key="pred_dept_text")
            if not department:
                department = "Engineering"   # safe default until user types
        else:
            department = dept_sel

        dept_med_oar = float(DEPT_MEDIANS.loc[department, 'dept_median_oar']) \
                       if department in DEPT_MEDIANS.index else 0.651

        with st.form("single_form"):
            r1c1,r1c2,r1c3 = st.columns(3)

            # Option B: selectbox + "Other" free-text fallback
            job_sel   = r1c1.selectbox("Job Title", DEPT_JOBS[department] + ["✏️ Other (type manually)"])
            src_sel   = r1c2.selectbox("Source",    SOURCES + ["✏️ Other (type manually)"])
            st.empty()

            # Reveal text input when "Other" is selected
            if job_sel == "✏️ Other (type manually)":
                job_title = st.text_input("Enter Job Title manually",
                    placeholder="e.g. Machine Learning Engineer",
                    help="This value is not in the training data — target encoding will use the global average as fallback.")
            else:
                job_title = job_sel

            if src_sel == "✏️ Other (type manually)":
                source = st.text_input("Enter Source manually",
                    placeholder="e.g. Instagram, Campus Hiring",
                    help="This value is not in the training data — target encoding will use the global average as fallback.")
            else:
                source = src_sel

            r2c1,r2c2,r2c3 = st.columns(3)
            num_applicants    = r2c1.number_input("Number of Applicants", min_value=10,  max_value=300,   value=150, step=5)
            time_to_hire_days = r2c2.number_input("Time to Hire (days)",  min_value=7,   max_value=89,    value=30)
            cost_per_hire     = r2c3.number_input("Cost per Hire ($)",    min_value=500, max_value=10000, value=5000, step=100)

            # OAR slider — starts at 0.00
            st.markdown("---")
            st.markdown("**Expected Offer Acceptance Rate**")
            oar_input = st.slider(
                "Expected OAR",
                min_value=0.00, max_value=1.00,
                value=round(min(dept_med_oar + 0.05, 1.00), 2),
                step=0.01,
            )
            st.markdown(
                f"<div class='oar-hint'>💡 Not sure what value to use? Start with the department median OAR of "
                f"<strong>{dept_med_oar:.2f}</strong> as your baseline.</div>",
                unsafe_allow_html=True
            )

            submitted = st.form_submit_button("🔮 Predict Outcome")

        # Warn if any custom "Other" values are in use
        custom_fields = []
        if dept_sel == "✏️ Other (type manually)" and department:
            custom_fields.append(f"Department: <strong>{department}</strong>")
        if 'job_sel' in dir() and job_sel == "✏️ Other (type manually)" and job_title:
            custom_fields.append(f"Job Title: <strong>{job_title}</strong>")
        if 'src_sel' in dir() and src_sel == "✏️ Other (type manually)" and source:
            custom_fields.append(f"Source: <strong>{source}</strong>")
        if custom_fields:
            st.markdown(
                f"<div class='oar-hint'>⚠️ Custom value(s) detected — {', '.join(custom_fields)} — "
                f"are not in the training data. The model will use the global average OAR as a fallback "
                f"for target encoding. Predictions may be less accurate for these inputs.</div>",
                unsafe_allow_html=True
            )

        if submitted:
            st.session_state.selected_dept = department
            inp = pd.DataFrame([{
                'department':           department,
                'job_title':            job_title,
                'source':               source,
                'num_applicants':       num_applicants,
                'time_to_hire_days':    time_to_hire_days,
                'cost_per_hire':        cost_per_hire,
                'offer_acceptance_rate':oar_input,
            }])
            preds, probas = predict(inp)
            prob_high = probas[0][1]
            prob_low  = probas[0][0]

            # ── Prediction result ──────────────────────────────────────────
            st.markdown("---")
            r1,r2,r3 = st.columns([1.2,1,1])
            badge = '<span class="badge-accept">✅ High OAR (≥0.70)</span>' if preds[0]==1 else '<span class="badge-reject">❌ Low OAR (&lt;0.70)</span>'
            r1.markdown(f"**Prediction**<br>{badge}", unsafe_allow_html=True)
            r2.metric("High OAR Probability", f"{prob_high*100:.1f}%")
            r3.metric("Low OAR Probability",  f"{prob_low*100:.1f}%")

            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=round(prob_high*100,1),
                title={"text":"High OAR Probability","font":{"color":"#9ba3bc","size":14}},
                number={"suffix":"%","font":{"color":"#e8eaf2","size":32}},
                gauge={
                    "axis":{"range":[0,100],"tickcolor":"#6b7592"},
                    "bar":{"color":"#4f8ef7"},"bgcolor":"#1c2030","bordercolor":"#262c3d",
                    "steps":[
                        {"range":[0,40], "color":"rgba(247,111,111,0.15)"},
                        {"range":[40,60],"color":"rgba(247,191,79,0.10)"},
                        {"range":[60,100],"color":"rgba(79,207,142,0.15)"},
                    ],
                    "threshold":{"line":{"color":"#f7854f","width":3},"thickness":0.75,"value":50},
                }
            ))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",font_color="#9ba3bc",
                              height=280,margin=dict(t=40,b=20,l=40,r=40))
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("🔍 View engineered features sent to the model"):
                feat_df = engineer_features(inp)
                st.dataframe(feat_df.T.rename(columns={0:"Value"}).round(4), use_container_width=True)

            # ── Feature 1: Detailed Insights & Recommendations ─────────────
            st.markdown("---")
            st.markdown('<p class="section-title">📋 Insights & Recommendations</p>'
                        '<p class="section-sub">Detailed analysis of each recruitment factor to help HR take action</p>',
                        unsafe_allow_html=True)

            insights = generate_insights(
                row={
                    'department':department,'job_title':job_title,'source':source,
                    'num_applicants':num_applicants,'time_to_hire_days':time_to_hire_days,
                    'cost_per_hire':cost_per_hire,'offer_acceptance_rate':oar_input,
                },
                dept_medians=DEPT_MEDIANS,
                source_map=SOURCE_MAP,
                global_mean=GLOBAL_MEAN,
            )

            TAG_LABELS = {"good":"✅ On Track","warn":"⚠️ Needs Attention","bad":"❌ Action Required"}

            # Display in 2 columns, 5 cards total (3 left, 2 right)
            col_ins_l, col_ins_r = st.columns(2)
            for i, ins in enumerate(insights):
                col = col_ins_l if i % 2 == 0 else col_ins_r
                tag_class = f"insight-tag-{ins['tag']}"
                tag_label = TAG_LABELS[ins['tag']]
                col.markdown(f"""
                <div class="insight-card">
                  <div class="insight-icon">{ins['icon']}</div>
                  <p class="insight-title">{ins['title']}</p>
                  <span class="{tag_class}">{tag_label}</span>
                  <p class="insight-body">{ins['body']}</p>
                </div>""", unsafe_allow_html=True)

            # ── Feature 2: What-If Scenario Comparison ─────────────────────
            st.markdown("---")
            st.markdown('<p class="section-title">🔀 What-If Scenario Comparison</p>'
                        '<p class="section-sub">Compare two recruitment approaches for the same role to find the more effective strategy</p>',
                        unsafe_allow_html=True)
            st.markdown(
                f"<div class='oar-hint'>ℹ️ Department (<strong>{department}</strong>) and Job Title "
                f"(<strong>{job_title}</strong>) are inherited from your prediction above. "
                f"Adjust the recruitment conditions below to compare two scenarios.</div>",
                unsafe_allow_html=True
            )
            st.markdown("<br>", unsafe_allow_html=True)

            # Store dept/job in session so comparison persists across reruns
            st.session_state['wi_dept']    = department
            st.session_state['wi_job']     = job_title
            st.session_state['wi_src_def'] = source
            st.session_state['wi_apps_def']= num_applicants
            st.session_state['wi_tth_def'] = time_to_hire_days
            st.session_state['wi_cph_def'] = cost_per_hire
            st.session_state['wi_oar_def'] = oar_input

        # ── What-if inputs and results OUTSIDE the if-submitted block ─────────
        # This prevents Streamlit from resetting everything on widget interaction
        if 'wi_dept' in st.session_state:
            wi_dept = st.session_state['wi_dept']
            wi_job  = st.session_state['wi_job']

            wi_col_a, wi_col_b = st.columns(2)

            with wi_col_a:
                st.markdown("#### Scenario A")
                wa_src  = st.selectbox("Source", SOURCES, key="wa_src",
                                       index=SOURCES.index(st.session_state.get('wi_src_def', SOURCES[0])))
                wa_apps = st.number_input("Number of Applicants", min_value=10,  max_value=300,
                                          value=st.session_state.get('wi_apps_def', 150), step=5, key="wa_apps")
                wa_tth  = st.number_input("Time to Hire (days)",  min_value=7,   max_value=89,
                                          value=st.session_state.get('wi_tth_def', 30),  key="wa_tth")
                wa_cph  = st.number_input("Cost per Hire ($)",    min_value=500, max_value=10000,
                                          value=st.session_state.get('wi_cph_def', 5000), step=100, key="wa_cph")
                wa_oar  = st.slider("Expected OAR", 0.00, 1.00,
                                    st.session_state.get('wi_oar_def', 0.70), 0.01, key="wa_oar")

            with wi_col_b:
                st.markdown("#### Scenario B")
                wb_src  = st.selectbox("Source", SOURCES, key="wb_src")
                wb_apps = st.number_input("Number of Applicants", min_value=10,  max_value=300,
                                          value=150, step=5, key="wb_apps")
                wb_tth  = st.number_input("Time to Hire (days)",  min_value=7,   max_value=89,
                                          value=30,   key="wb_tth")
                wb_cph  = st.number_input("Cost per Hire ($)",    min_value=500, max_value=10000,
                                          value=4000, step=100, key="wb_cph")
                wb_oar  = st.slider("Expected OAR", 0.00, 1.00,
                                    round(min(st.session_state.get('wi_oar_def', 0.70)+0.10, 1.0), 2),
                                    0.01, key="wb_oar")

            if st.button("⚡ Run Comparison"):
                inp_a = pd.DataFrame([{'department':wi_dept,'job_title':wi_job,'source':wa_src,
                    'num_applicants':wa_apps,'time_to_hire_days':wa_tth,
                    'cost_per_hire':wa_cph,'offer_acceptance_rate':wa_oar}])
                inp_b = pd.DataFrame([{'department':wi_dept,'job_title':wi_job,'source':wb_src,
                    'num_applicants':wb_apps,'time_to_hire_days':wb_tth,
                    'cost_per_hire':wb_cph,'offer_acceptance_rate':wb_oar}])

                pred_a, proba_a = predict(inp_a)
                pred_b, proba_b = predict(inp_b)
                ph_a   = proba_a[0][1]
                ph_b   = proba_b[0][1]
                winner = "A" if ph_a >= ph_b else "B"

                # Store results in session_state so they persist
                st.session_state['wi_results'] = {
                    'ph_a':ph_a,'ph_b':ph_b,'pred_a':int(pred_a[0]),'pred_b':int(pred_b[0]),
                    'winner':winner,
                    'wa_src':wa_src,'wa_apps':wa_apps,'wa_tth':wa_tth,'wa_cph':wa_cph,'wa_oar':wa_oar,
                    'wb_src':wb_src,'wb_apps':wb_apps,'wb_tth':wb_tth,'wb_cph':wb_cph,'wb_oar':wb_oar,
                }

            # Render results from session_state (persists across reruns)
            if 'wi_results' in st.session_state:
                r = st.session_state['wi_results']
                ph_a   = r['ph_a'];   ph_b   = r['ph_b']
                winner = r['winner']
                wa_src = r['wa_src']; wb_src = r['wb_src']
                wa_tth = r['wa_tth']; wb_tth = r['wb_tth']
                wa_cph = r['wa_cph']; wb_cph = r['wb_cph']
                wa_oar = r['wa_oar']; wb_oar = r['wb_oar']
                wa_apps= r['wa_apps'];wb_apps= r['wb_apps']

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### Comparison Results")

                cls_a  = "whatif-card whatif-winner" if winner=="A" else "whatif-card"
                cls_b  = "whatif-card whatif-winner" if winner=="B" else "whatif-card"
                label_a= "✅ High OAR" if r['pred_a']==1 else "❌ Low OAR"
                label_b= "✅ High OAR" if r['pred_b']==1 else "❌ Low OAR"
                crown_a= " 👑 Better outcome" if winner=="A" else ""
                crown_b= " 👑 Better outcome" if winner=="B" else ""

                res_a, res_b = st.columns(2)
                with res_a:
                    st.markdown(f"""
                    <div class="{cls_a}">
                      <p class="whatif-label">Scenario A{crown_a}</p>
                      <p class="whatif-val">{label_a} &nbsp; {ph_a*100:.1f}%</p>
                      <br>
                      <div class="compare-row"><span class="compare-label">Source</span><span class="compare-val">{wa_src}</span></div>
                      <div class="compare-row"><span class="compare-label">Applicants</span><span class="compare-val">{wa_apps}</span></div>
                      <div class="compare-row"><span class="compare-label">Time to Hire</span><span class="compare-val">{wa_tth} days</span></div>
                      <div class="compare-row"><span class="compare-label">Cost per Hire</span><span class="compare-val">${wa_cph:,}</span></div>
                      <div class="compare-row"><span class="compare-label">Expected OAR</span><span class="compare-val">{wa_oar:.2f}</span></div>
                    </div>""", unsafe_allow_html=True)

                with res_b:
                    st.markdown(f"""
                    <div class="{cls_b}">
                      <p class="whatif-label">Scenario B{crown_b}</p>
                      <p class="whatif-val">{label_b} &nbsp; {ph_b*100:.1f}%</p>
                      <br>
                      <div class="compare-row"><span class="compare-label">Source</span><span class="compare-val">{wb_src}</span></div>
                      <div class="compare-row"><span class="compare-label">Applicants</span><span class="compare-val">{wb_apps}</span></div>
                      <div class="compare-row"><span class="compare-label">Time to Hire</span><span class="compare-val">{wb_tth} days</span></div>
                      <div class="compare-row"><span class="compare-label">Cost per Hire</span><span class="compare-val">${wb_cph:,}</span></div>
                      <div class="compare-row"><span class="compare-label">Expected OAR</span><span class="compare-val">{wb_oar:.2f}</span></div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                fig_wi = go.Figure()
                fig_wi.add_trace(go.Bar(
                    name="Scenario A", x=["High OAR Probability"],
                    y=[round(ph_a*100,1)],
                    marker_color="#4fcf8e" if r['pred_a']==1 else "#f76f6f",
                    text=[f"{ph_a*100:.1f}%"], textposition="outside",
                ))
                fig_wi.add_trace(go.Bar(
                    name="Scenario B", x=["High OAR Probability"],
                    y=[round(ph_b*100,1)],
                    marker_color="#4fcf8e" if r['pred_b']==1 else "#f76f6f",
                    text=[f"{ph_b*100:.1f}%"], textposition="outside",
                    marker_pattern_shape="x",
                ))
                fig_wi.add_hline(y=50, line_dash="dash", line_color="#f7854f",
                                 annotation_text="Decision threshold (50%)",
                                 annotation_font_color="#f7854f")
                fig_wi.update_layout(**PT, barmode="group", height=300,
                                     margin=dict(t=20,b=10,l=0,r=0),
                                     yaxis_range=[0,110], yaxis_title="Probability (%)",
                                     legend=dict(orientation="h", y=1.1))
                fig_wi.update_traces(marker_line_width=0)
                st.plotly_chart(fig_wi, use_container_width=True)

                diff_pct   = abs(ph_a - ph_b)*100
                better_src = wa_src if winner=="A" else wb_src
                better_tth = wa_tth if winner=="A" else wb_tth
                better_cph = wa_cph if winner=="A" else wb_cph
                st.markdown(f"""
                <div class="tab-desc">
                <strong>💡 Recommendation:</strong> Scenario {winner} yields a higher acceptance probability
                by <strong>{diff_pct:.1f} percentage points</strong>.
                The better outcome uses <strong>{better_src}</strong> as the sourcing channel,
                with a time to hire of <strong>{better_tth} days</strong> and a cost per hire of
                <strong>${better_cph:,}</strong>. Consider adopting these conditions for the
                <strong>{wi_dept} — {wi_job}</strong> role to improve offer acceptance likelihood.
                </div>""", unsafe_allow_html=True)

    # ── Batch ────────────────────────────────────────────────────────────────
    else:
        st.markdown('<p class="section-title">Batch Candidate Predictor</p><p class="section-sub">Upload a CSV and get predictions for all candidates at once</p>', unsafe_allow_html=True)

        required_cols = ['department','job_title','source','num_applicants',
                         'time_to_hire_days','cost_per_hire','offer_acceptance_rate']
        st.download_button("⬇ Download CSV Template",
                           pd.DataFrame(columns=required_cols).to_csv(index=False).encode(),
                           "candidate_template.csv","text/csv")
        st.markdown(
            f"<p style='color:#6b7592;font-size:12px'>Required columns: {', '.join(f'<code>{c}</code>' for c in required_cols)}<br>"
            f"<code>offer_acceptance_rate</code>: enter the expected OAR for each candidate (0.00–1.00).</p>",
            unsafe_allow_html=True
        )

        batch_file = st.file_uploader("Upload candidates CSV", type=["csv"], key="batch")

        if batch_file:
            try:
                batch_df = pd.read_csv(batch_file)
                st.markdown(f"<p style='color:#9ba3bc;font-size:13px'>📂 File loaded: <strong>{len(batch_df):,} rows</strong> detected — running validation…</p>", unsafe_allow_html=True)

                # ── Validation ────────────────────────────────────────────
                result = validate_batch(batch_df, required_cols)

                # Show blocking errors
                if result['errors']:
                    st.markdown("""
                    <div style="background:rgba(247,111,111,.08);border:1px solid rgba(247,111,111,.35);
                    border-radius:10px;padding:16px 20px;margin-bottom:12px;">
                    <p style="color:#f76f6f;font-weight:600;font-size:14px;margin:0 0 10px;">
                    ❌ Validation Failed — Please fix the following issues before proceeding:</p>
                    </div>""", unsafe_allow_html=True)
                    for i, err in enumerate(result['errors'], 1):
                        st.error(f"**Issue {i}:** {err}")
                    st.markdown(
                        "<p style='color:#9ba3bc;font-size:13px;margin-top:8px;'>"
                        "💡 Download the CSV template above to see the correct format.</p>",
                        unsafe_allow_html=True
                    )

                else:
                    # Show non-blocking warnings
                    if result['warnings']:
                        st.markdown("""
                        <div style="background:rgba(247,133,79,.08);border:1px solid rgba(247,133,79,.35);
                        border-radius:10px;padding:16px 20px;margin-bottom:12px;">
                        <p style="color:#f7854f;font-weight:600;font-size:14px;margin:0 0 10px;">
                        ⚠️ Validation Passed with Warnings — Predictions will run, but please review:</p>
                        </div>""", unsafe_allow_html=True)
                        for warn in result['warnings']:
                            st.warning(warn)

                    # ── All clear — run predictions ────────────────────────
                    batch_df = result['cleaned']
                    st.markdown(
                        "<div style='background:rgba(79,207,142,.08);border:1px solid rgba(79,207,142,.35);"
                        "border-radius:10px;padding:12px 18px;margin-bottom:16px;'>"
                        "<p style='color:#4fcf8e;font-weight:600;font-size:13px;margin:0;'>"
                        f"✅ Validation passed — running predictions on {len(batch_df):,} candidate(s)…</p></div>",
                        unsafe_allow_html=True
                    )

                    preds, probas = predict(batch_df)
                    batch_df['prediction']      = np.where(preds==1,'High OAR (≥0.70)','Low OAR (<0.70)')
                    batch_df['prob_high_oar_%'] = (probas[:,1]*100).round(1)
                    batch_df['prob_low_oar_%']  = (probas[:,0]*100).round(1)

                    b1,b2,b3 = st.columns(3)
                    b1.metric("Total Processed",    f"{len(batch_df):,}")
                    b2.metric("Predicted High OAR", f"{(preds==1).sum():,}")
                    b3.metric("Predicted Low OAR",  f"{(preds==0).sum():,}")

                    cp,cb = st.columns(2)
                    with cp:
                        vc = batch_df['prediction'].value_counts()
                        fig = go.Figure(go.Pie(labels=vc.index,values=vc.values,
                            marker_colors=['#4fcf8e','#f76f6f'],hole=0.55,textinfo='label+percent'))
                        fig.update_layout(**PT,margin=dict(t=10,b=10,l=0,r=0),showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                    with cb:
                        fig = px.histogram(batch_df,x='prob_high_oar_%',color='prediction',
                            color_discrete_map={'High OAR (≥0.70)':'#4fcf8e','Low OAR (<0.70)':'#f76f6f'},
                            nbins=20,title='High OAR Probability Distribution')
                        fig.update_layout(**PT,margin=dict(t=30,b=10,l=0,r=0),
                            title_font_color='#9ba3bc',title_font_size=13,
                            xaxis_title='High OAR Probability (%)',yaxis_title='Count')
                        st.plotly_chart(fig, use_container_width=True)

                    st.dataframe(batch_df.reset_index(drop=True), use_container_width=True, height=320)
                    st.download_button("⬇ Download Predictions",
                                       batch_df.to_csv(index=False).encode(),
                                       "predictions.csv","text/csv")

            except Exception as e:
                st.markdown(
                    "<div style='background:rgba(247,111,111,.08);border:1px solid rgba(247,111,111,.35);"
                    "border-radius:10px;padding:14px 18px;'>"
                    "<p style='color:#f76f6f;font-weight:600;font-size:14px;margin:0 0 6px;'>❌ Unexpected Error</p>"
                    f"<p style='color:#9ba3bc;font-size:13px;margin:0;'>The file could not be processed: {e}<br>"
                    "Please ensure the file is a valid CSV and matches the required format.</p></div>",
                    unsafe_allow_html=True
                )
        else:
            st.markdown("""
            <div style="background:#1c2030;border:1px dashed #262c3d;border-radius:14px;padding:40px;text-align:center;color:#6b7592;">
              <div style="font-size:40px;margin-bottom:12px">📋</div>
              <div style="font-size:15px">Upload a CSV file to run batch predictions</div>
              <div style="font-size:13px;margin-top:6px">Download the template above to see the required format</div>
            </div>""", unsafe_allow_html=True)
