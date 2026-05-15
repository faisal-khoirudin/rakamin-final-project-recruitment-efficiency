import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="RecruitIQ · Recruitment Intelligence",
    page_icon="🎯",
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
  .badge-accept{display:inline-block;background:rgba(79,207,142,.15);color:#4fcf8e;border:1px solid rgba(79,207,142,.35);padding:6px 18px;border-radius:100px;font-weight:600;font-size:15px;}
  .badge-reject{display:inline-block;background:rgba(247,111,111,.15);color:#f76f6f;border:1px solid rgba(247,111,111,.35);padding:6px 18px;border-radius:100px;font-weight:600;font-size:15px;}
  .hero{background:linear-gradient(135deg,#1c2030 0%,#141828 100%);border:1px solid var(--border);border-radius:18px;padding:32px 36px;margin-bottom:28px;}
  .hero h1{font-family:'DM Serif Display',serif;font-size:32px;margin:0 0 6px;}
  .hero p{color:var(--subtext);font-size:15px;margin:0;}
  .info-box{background:rgba(79,142,247,.08);border:1px solid rgba(79,142,247,.25);border-radius:10px;padding:14px 18px;font-size:13px;color:var(--subtext);margin-bottom:18px;}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
DEPARTMENTS = ["Engineering","Sales","Product","HR","Marketing","Finance"]
SOURCES     = ["Referral","LinkedIn","Recruiter","Job Portal"]
JOB_TITLES  = [
    "Software Engineer","Account Executive","UX Designer","DevOps Engineer",
    "Talent Acquisition","Marketing Specialist","Accountant","HR Coordinator",
    "Recruitment Specialist","Business Development Manager","Sales Associate",
    "Backend Developer","Finance Manager","Product Manager","Social Media Manager",
    "Content Strategist","SEO Analyst","Financial Analyst","Sales Representative",
    "UI Designer","Product Analyst","Data Engineer","Payroll Specialist","HR Manager",
]
FINAL_FEATURES = [
    'cost_per_applicant','cost_per_day','applicants_per_day',
    'is_senior_role','difficulty_additive','drei',
    'dept_oar_mean','jobtitle_oar_mean','source_oar_mean','dept_source_oar_mean',
]
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
    return pd.read_csv("data/recruitment_efficiency_improved.csv")

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

# ── Feature engineering (mirrors Stage 4 notebook exactly) ────────────────────
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

    # DREI: uses exact column names from Stage 4 notebook dept_medians
    # dept_medians columns: dept_median_cph, dept_median_tth, dept_median_oar
    d2 = d.join(DEPT_MEDIANS, on='department')
    if 'offer_acceptance_rate' in d.columns:
        d['drei'] = (
            (d['cost_per_hire'].values            < d2['dept_median_cph'].values) &
            (d['time_to_hire_days'].values         < d2['dept_median_tth'].values) &
            (d['offer_acceptance_rate'].values     > d2['dept_median_oar'].values)
        ).astype(int)
    else:
        d['drei'] = 0   # conservative default — no OAR available for new candidates

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

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 RecruitIQ")
    st.markdown("<span style='color:#6b7592;font-size:13px'>Recruitment Intelligence Platform</span>", unsafe_allow_html=True)
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
  <h1>🎯 RecruitIQ Dashboard</h1>
  <p>Monitor recruitment KPIs, explore candidate pipelines, and predict offer acceptance outcomes —
  powered by your <em>recruitment_efficiency_improved</em> dataset and Stage 4 XGBoost model.</p>
</div>""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📊  Candidate Overview","🤖  Candidate Predictor"])

# ════════════════════════════════════════════════════════
#  TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════
with tab1:
    avg_cph      = df['cost_per_hire'].mean()
    avg_tth      = df['time_to_hire_days'].mean()
    avg_oar      = df['offer_acceptance_rate'].mean()
    high_oar_pct = (df['offer_acceptance_rate']>=0.70).mean()*100

    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Total Records",            f"{len(df):,}")
    k2.metric("Avg Cost per Hire",        f"${avg_cph:,.0f}")
    k3.metric("Avg Time to Hire",         f"{avg_tth:.1f} days")
    k4.metric("Avg Offer Acceptance Rate",f"{avg_oar:.1%}")
    k5.metric("High OAR Rate (≥70%)",     f"{high_oar_pct:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 1: by source + by department
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

    # Row 2: by job title + OAR distribution
    c3,c4 = st.columns(2)
    with c3:
        st.markdown('<p class="section-title">Records by Job Title</p><p class="section-sub">Candidate volume by role (top 15)</p>', unsafe_allow_html=True)
        jt_grp = df.groupby('job_title').size().reset_index(name='count').sort_values('count').tail(15)
        fig = go.Figure(go.Bar(
            x=jt_grp['count'], y=jt_grp['job_title'], orientation='h',
            marker_color='#a78bfa', text=jt_grp['count'], textposition='outside',
        ))
        fig.update_layout(**PT, margin=dict(t=10,b=10,l=0,r=0),
                          xaxis_title='Candidates', yaxis_title='', height=420)
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

    # Row 3: Cost & Time boxes + Feature importance
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
        fig.update_layout(**PT,margin=dict(t=30,b=10,l=0,r=0),
                          legend=dict(orientation='h',y=1.1),height=370)
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
        fig.update_layout(**PT,margin=dict(t=10,b=10,r=10,l=0),height=370,
                          xaxis_title='Importance',yaxis_title='')
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
    st.markdown("""
    <div class="info-box">
    ℹ️ Predicts whether a recruitment record will result in <strong>High OAR (≥ 0.70)</strong> or <strong>Low OAR (&lt; 0.70)</strong>,
    using the same feature engineering pipeline from your Stage 4 notebook (ratio features, difficulty additive, DREI, target encoding).
    </div>""", unsafe_allow_html=True)

    mode = st.radio("", ["🧑 Single Candidate","📋 Batch Upload"], horizontal=True, label_visibility="collapsed")
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Single ──────────────────────────────────────────────────────────────
    if mode == "🧑 Single Candidate":
        st.markdown('<p class="section-title">Single Candidate Predictor</p><p class="section-sub">Enter recruitment details to predict offer acceptance outcome</p>', unsafe_allow_html=True)

        with st.form("single_form"):
            r1c1,r1c2,r1c3 = st.columns(3)
            department        = r1c1.selectbox("Department", DEPARTMENTS)
            job_title         = r1c2.selectbox("Job Title",  JOB_TITLES)
            source            = r1c3.selectbox("Source",     SOURCES)

            r2c1,r2c2,r2c3 = st.columns(3)
            num_applicants    = r2c1.number_input("Number of Applicants", min_value=10,  max_value=300,   value=150, step=5)
            time_to_hire_days = r2c2.number_input("Time to Hire (days)",  min_value=7,   max_value=89,    value=30)
            cost_per_hire     = r2c3.number_input("Cost per Hire ($)",    min_value=500, max_value=10000, value=5000, step=100)

            submitted = st.form_submit_button("🔮 Predict Outcome")

        if submitted:
            inp = pd.DataFrame([{
                'department':department,'job_title':job_title,'source':source,
                'num_applicants':num_applicants,'time_to_hire_days':time_to_hire_days,
                'cost_per_hire':cost_per_hire,
            }])
            preds, probas = predict(inp)
            prob_high = probas[0][1]
            prob_low  = probas[0][0]

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

    # ── Batch ────────────────────────────────────────────────────────────────
    else:
        st.markdown('<p class="section-title">Batch Candidate Predictor</p><p class="section-sub">Upload a CSV and get predictions for all candidates at once</p>', unsafe_allow_html=True)

        required_cols = ['department','job_title','source','num_applicants','time_to_hire_days','cost_per_hire']
        st.download_button("⬇ Download CSV Template",
                           pd.DataFrame(columns=required_cols).to_csv(index=False).encode(),
                           "candidate_template.csv","text/csv")
        st.markdown(f"<p style='color:#6b7592;font-size:12px'>Required: {', '.join(f'<code>{c}</code>' for c in required_cols)}</p>", unsafe_allow_html=True)

        batch_file = st.file_uploader("Upload candidates CSV", type=["csv"], key="batch")

        if batch_file:
            try:
                batch_df = pd.read_csv(batch_file)
                missing = [c for c in required_cols if c not in batch_df.columns]
                if missing:
                    st.error(f"❌ Missing columns: {', '.join(missing)}")
                else:
                    st.info(f"✓ {len(batch_df):,} candidates loaded — running predictions…")
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
                st.error(f"❌ Error: {e}")
        else:
            st.markdown("""
            <div style="background:#1c2030;border:1px dashed #262c3d;border-radius:14px;padding:40px;text-align:center;color:#6b7592;">
              <div style="font-size:40px;margin-bottom:12px">📋</div>
              <div style="font-size:15px">Upload a CSV file to run batch predictions</div>
              <div style="font-size:13px;margin-top:6px">Download the template above to see the required format</div>
            </div>""", unsafe_allow_html=True)
