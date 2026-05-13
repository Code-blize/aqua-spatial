# ============================================================
# AQUA-SPATIAL INTELLIGENCE PLATFORM
# Enugu State Water Security Dashboard
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import pickle
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Aqua-Spatial | Enugu Water Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .stApp { background-color: #07111f; color: #e2f4fb; }

    [data-testid="stSidebar"] {
        background: #050d18;
        border-right: 1px solid #0e2a45;
    }
    [data-testid="stSidebar"] * { color: #8ab8cc !important; }
    [data-testid="stSidebar"] .stMarkdown b { color: #e2f4fb !important; }

    /* Sidebar multiselect tags */
    [data-testid="stMultiSelect"] span[data-baseweb="tag"] {
        background-color: #0e2a45 !important;
        color: #e2f4fb !important;
    }
    [data-testid="stMultiSelect"] span[data-baseweb="tag"] span {
        color: #e2f4fb !important;
    }

    /* Streamlit default text overrides */
    .stMarkdown p { color: #8ab8cc; }
    label { color: #8ab8cc !important; }

    /* KPI Cards */
    .kpi-card {
        background: #07111f;
        border: 1px solid #0e2a45;
        border-top: 2px solid #0ea5c9;
        border-radius: 4px;
        padding: 20px 24px;
        height: 130px;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 400;
        color: #e2f4fb;
        letter-spacing: -0.5px;
        line-height: 1.2;
    }
    .kpi-value.critical { color: #e05252; }
    .kpi-value.warn     { color: #e07d30; }
    .kpi-value.good     { color: #22c77a; }
    .kpi-value.info     { color: #0ea5c9; }
    .kpi-label {
        font-size: 0.65rem;
        color: #3d7a9e;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .kpi-sublabel {
        font-size: 0.7rem;
        color: #4a7a99;
        margin-top: 4px;
    }
    .kpi-bar {
        margin-top: 10px;
        height: 2px;
        background: #0e2a45;
        border-radius: 1px;
        overflow: hidden;
    }
    .kpi-bar-fill {
        height: 100%;
        border-radius: 1px;
    }

    /* Section headers */
    .section-header {
        font-size: 0.65rem;
        color: #0ea5c9;
        text-transform: uppercase;
        letter-spacing: 3px;
        margin-bottom: 16px;
        padding-bottom: 10px;
        border-bottom: 1px solid #0e2a45;
    }

    /* Hero */
    .hero-tag {
        font-size: 0.65rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: #0ea5c9;
        margin-bottom: 10px;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 300;
        color: #e2f4fb;
        letter-spacing: -1px;
        line-height: 1.15;
        margin-bottom: 10px;
    }
    .hero-title span { color: #0ea5c9; font-weight: 500; }
    .hero-sub {
        font-size: 0.8rem;
        color: #4a7a99;
        letter-spacing: 0.5px;
    }

    /* Status pills */
    .status-pill {
        background: #051a0f;
        border: 1px solid #0a6640;
        color: #22c77a;
        font-size: 0.68rem;
        padding: 5px 14px;
        border-radius: 3px;
        letter-spacing: 1px;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 8px;
        text-transform: uppercase;
    }
    .status-dot {
        width: 6px;
        height: 6px;
        background: #22c77a;
        border-radius: 50%;
        display: inline-block;
    }
    .alert-pill {
        background: #150d04;
        border: 1px solid #7a3a08;
        color: #e07d30;
        font-size: 0.68rem;
        padding: 5px 14px;
        border-radius: 3px;
        letter-spacing: 1px;
        display: inline-block;
        text-transform: uppercase;
    }

    /* Dataframe */
    [data-testid="stDataFrame"] { border: 1px solid #0e2a45 !important; }
    [data-testid="stDataFrame"] * { color: #e2f4fb !important; }

    /* Hide Streamlit chrome */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data
def load_data():
    base = os.path.dirname(__file__)
    return pd.read_csv(os.path.join(base, 'dashboard_data.csv'))

@st.cache_data
def load_geodata():
    base = os.path.dirname(__file__)
    with open(os.path.join(base, '..', 'data', 'enugu_data.pkl'), 'rb') as f:
        data = pickle.load(f)
    return data['enugu']

@st.cache_data
def load_importance():
    base = os.path.dirname(__file__)
    return pd.read_csv(os.path.join(base, 'feature_importance.csv'))

df          = load_data()
enugu       = load_geodata()
importance_df = load_importance()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:10px 0 24px 0; border-bottom:1px solid #0e2a45;'>
        <div style='color:#0ea5c9; font-weight:700; font-size:1rem; letter-spacing:3px'>AQUA-SPATIAL</div>
        <div style='color:#3d7a9e; font-size:0.65rem; letter-spacing:2px; margin-top:6px'>
            WATER INTELLIGENCE PLATFORM
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Filter by Confidence**")
    confidence_filter = st.multiselect(
        "Confidence",
        options=['High', 'Medium', 'Low'],
        default=['High', 'Medium', 'Low'],
        label_visibility='collapsed'
    )

    st.markdown("---")
    st.markdown("**Intervention Threshold**")
    threshold = st.slider(
        "Threshold", min_value=0.0, max_value=1.0,
        value=0.35, step=0.05, label_visibility='collapsed'
    )

    st.markdown("---")
    st.markdown("**Map Layer**")
    map_layer = st.radio(
        "Layer",
        options=['Predicted Stress', 'Functionality Rate',
                 'Terrain Difficulty', 'Drilling Difficulty'],
        label_visibility='collapsed'
    )

    layer_map = {
        'Predicted Stress':   'predicted_stress',
        'Functionality Rate': 'functionality_rate',
        'Terrain Difficulty': 'terrain_difficulty',
        'Drilling Difficulty':'drilling_difficulty'
    }
    selected_layer = layer_map[map_layer]

    st.markdown("---")
    st.markdown("""
    <div style='color:#3d7a9e; font-size:0.65rem; line-height:2; text-transform:uppercase; letter-spacing:1px'>
        Satellite: NASA/USGS via GEE<br>
        Model: RF + GB Ensemble<br>
        R&#178; = 0.60 &nbsp;|&nbsp; MAE = 0.069
    </div>
    <br>
    <div style='color:#e05252; font-size:0.65rem; letter-spacing:1px; text-transform:uppercase'>
        Ground data is synthetic
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# FILTER DATA
# ============================================================
filtered_df  = df[df['confidence'].isin(confidence_filter)].copy()
priority_lgas = df[df['predicted_stress'] >= threshold]

# ============================================================
# HEADER
# ============================================================
col_title, col_badge = st.columns([3, 1])

with col_title:
    st.markdown("""
    <div class='hero-tag'>Enugu State &nbsp;&middot;&nbsp; Nigeria &nbsp;&middot;&nbsp; 2026</div>
    <div class='hero-title'>Water Security<br>Intelligence <span>Command</span></div>
    <div class='hero-sub'>Geospatial decision support for water infrastructure planning</div>
    """, unsafe_allow_html=True)

with col_badge:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='text-align:right'>
        <div class='status-pill'>
            <span class='status-dot'></span>GEE Satellite Active
        </div><br>
        <div class='alert-pill'>
            {len(priority_lgas)} LGAs Require Intervention
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# ROW 1 — KPI CARDS
# ============================================================
st.markdown("<div class='section-header'>State Overview</div>", unsafe_allow_html=True)

k1, k2, k3, k4, k5 = st.columns(5)

most_stressed    = df.loc[df['predicted_stress'].idxmax()]
avg_stress       = df['predicted_stress'].mean()
avg_functionality = df['functionality_rate'].mean()
total_pop        = df['total_pop_served'].sum()
high_conf        = len(df[df['confidence'] == 'High'])

with k1:
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-label'>Most Critical LGA</div>
        <div class='kpi-value critical'>{most_stressed['lga_name']}</div>
        <div class='kpi-sublabel'>Score {most_stressed['predicted_stress']:.3f}</div>
        <div class='kpi-bar'>
            <div class='kpi-bar-fill' style='width:{most_stressed["predicted_stress"]*100:.0f}%; background:#e05252'></div>
        </div>
    </div>""", unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-label'>Avg State Stress</div>
        <div class='kpi-value warn'>{avg_stress:.2f}</div>
        <div class='kpi-sublabel'>0 = No stress &nbsp;|&nbsp; 1 = Critical</div>
        <div class='kpi-bar'>
            <div class='kpi-bar-fill' style='width:{avg_stress*100:.0f}%; background:#e07d30'></div>
        </div>
    </div>""", unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-label'>Avg Functionality</div>
        <div class='kpi-value good'>{avg_functionality:.0f}%</div>
        <div class='kpi-sublabel'>Water points operational</div>
        <div class='kpi-bar'>
            <div class='kpi-bar-fill' style='width:{avg_functionality:.0f}%; background:#22c77a'></div>
        </div>
    </div>""", unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-label'>Population Served</div>
        <div class='kpi-value info'>{total_pop:,.0f}</div>
        <div class='kpi-sublabel'>Across all water points</div>
        <div class='kpi-bar'>
            <div class='kpi-bar-fill' style='width:65%; background:#0ea5c9'></div>
        </div>
    </div>""", unsafe_allow_html=True)

with k5:
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-label'>High Confidence LGAs</div>
        <div class='kpi-value good'>{high_conf}/17</div>
        <div class='kpi-sublabel'>Reliable predictions</div>
        <div class='kpi-bar'>
            <div class='kpi-bar-fill' style='width:{high_conf/17*100:.0f}%; background:#22c77a'></div>
        </div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# ROW 2 — MAP + STRESS RANKING
# ============================================================
st.markdown("<div class='section-header'>Spatial Intelligence Map</div>", unsafe_allow_html=True)

map_col, rank_col = st.columns([3, 2])

with map_col:
    enugu_map = enugu.merge(
        df[['lga_name','predicted_stress','confidence',
            'functionality_rate','terrain_difficulty',
            'drilling_difficulty','total_pop_served']],
        left_on='NAME_2', right_on='lga_name', how='left'
    )

    m = folium.Map(location=[6.5, 7.4], zoom_start=8, tiles='CartoDB dark_matter')

    colorscale_map = {
        'predicted_stress':   'YlOrRd',
        'functionality_rate': 'RdYlGn',
        'terrain_difficulty': 'YlOrBr',
        'drilling_difficulty':'PuRd'
    }

    folium.Choropleth(
        geo_data=enugu_map.__geo_interface__,
        data=enugu_map,
        columns=['NAME_2', selected_layer],
        key_on='feature.properties.NAME_2',
        fill_color=colorscale_map[selected_layer],
        fill_opacity=0.75,
        line_opacity=0.9,
        line_color='#0ea5c9',
        line_weight=1.5,
        legend_name=map_layer,
        nan_fill_color='#1a2744'
    ).add_to(m)

    folium.GeoJson(
        enugu_map.__geo_interface__,
        style_function=lambda x: {'fillOpacity': 0, 'weight': 0},
        tooltip=folium.GeoJsonTooltip(
            fields=['NAME_2','predicted_stress','confidence',
                    'functionality_rate','total_pop_served'],
            aliases=['LGA','Stress Score','Confidence',
                     'Functionality %','Population Served'],
            localize=True,
            style="""
                background-color: #050d18;
                border: 1px solid #0ea5c9;
                border-radius: 4px;
                color: #e2f4fb;
                font-family: monospace;
                font-size: 12px;
                padding: 10px;
            """
        )
    ).add_to(m)

    for _, row in df[df['predicted_stress'] >= threshold].iterrows():
        lga_geo = enugu_map[enugu_map['NAME_2'] == row['lga_name']]
        if len(lga_geo) > 0:
            centroid = lga_geo.geometry.centroid.values[0]
            folium.CircleMarker(
                location=[centroid.y, centroid.x],
                radius=8, color='#e05252',
                fill=True, fill_color='#e05252', fill_opacity=0.85,
                popup=folium.Popup(
                    f"<b style='color:#e05252'>{row['lga_name']}</b><br>"
                    f"Stress: {row['predicted_stress']:.3f}<br>"
                    f"Confidence: {row['confidence']}",
                    max_width=200
                )
            ).add_to(m)

    st_folium(m, width=700, height=500)

with rank_col:
    rank_df = df.sort_values('predicted_stress', ascending=True)
    bar_colors = rank_df['confidence'].map({
        'High': '#22c77a', 'Medium': '#e07d30', 'Low': '#e05252'
    })

    fig_rank = go.Figure()
    fig_rank.add_trace(go.Bar(
        y=rank_df['lga_name'],
        x=rank_df['upper_bound'] - rank_df['lower_bound'],
        base=rank_df['lower_bound'],
        orientation='h',
        marker_color='rgba(14,165,201,0.08)',
        marker_line_width=0,
        name='Confidence Range',
        hoverinfo='skip'
    ))
    fig_rank.add_trace(go.Bar(
        y=rank_df['lga_name'],
        x=rank_df['predicted_stress'],
        orientation='h',
        marker_color=bar_colors,
        name='Predicted Stress',
        text=rank_df['predicted_stress'].round(3),
        textposition='outside',
        textfont=dict(color='#8ab8cc', size=10),
        hovertemplate='<b>%{y}</b><br>Stress: %{x:.3f}<extra></extra>'
    ))
    fig_rank.add_vline(
        x=threshold, line_dash="dash", line_color="#0ea5c9",
        annotation_text="Threshold",
        annotation_font_color="#0ea5c9",
        annotation_font_size=10
    )
    fig_rank.update_layout(
        title=dict(text="Water Stress Rankings", font=dict(color='#e2f4fb', size=13)),
        paper_bgcolor='#07111f', plot_bgcolor='#07111f',
        font=dict(color='#8ab8cc', size=11),
        showlegend=False, height=500,
        margin=dict(l=10, r=50, t=40, b=10),
        xaxis=dict(gridcolor='#0e2a45', range=[0, 1.15], title='Stress Score'),
        yaxis=dict(gridcolor='#0e2a45'),
        barmode='overlay'
    )
    st.plotly_chart(fig_rank, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# ROW 3 — FEATURE IMPORTANCE + SCATTER
# ============================================================
st.markdown("<div class='section-header'>Model Intelligence</div>", unsafe_allow_html=True)

feat_col, scatter_col = st.columns(2)

with feat_col:
    top_features = importance_df.sort_values('importance', ascending=True).tail(10)
    feat_colors  = ['#0ea5c9' if i >= 7 else '#1e3a5f' for i in range(len(top_features))]

    fig_feat = go.Figure(go.Bar(
        y=top_features['feature'],
        x=top_features['importance'],
        orientation='h',
        marker_color=feat_colors,
        text=top_features['importance'].round(3),
        textposition='outside',
        textfont=dict(color='#8ab8cc', size=10),
        hovertemplate='<b>%{y}</b><br>Importance: %{x:.3f}<extra></extra>'
    ))
    fig_feat.update_layout(
        title=dict(text="Top 10 Predictive Features", font=dict(color='#e2f4fb', size=13)),
        paper_bgcolor='#07111f', plot_bgcolor='#07111f',
        font=dict(color='#8ab8cc', size=11), height=400,
        margin=dict(l=10, r=70, t=40, b=10),
        xaxis=dict(gridcolor='#0e2a45', title='Importance Score'),
        yaxis=dict(gridcolor='#0e2a45')
    )
    st.plotly_chart(fig_feat, use_container_width=True)

with scatter_col:
    fig_scatter = px.scatter(
        df, x='heat_veg_stress', y='predicted_stress',
        size='total_pop_served', color='confidence',
        color_discrete_map={'High':'#22c77a','Medium':'#e07d30','Low':'#e05252'},
        hover_name='lga_name',
        hover_data={
            'predicted_stress':':.3f',
            'heat_veg_stress':':.1f',
            'total_pop_served':':,.0f'
        },
        labels={
            'heat_veg_stress':'Heat-Vegetation Stress Index (LST/NDVI)',
            'predicted_stress':'Predicted Water Stress Score',
            'confidence':'Confidence'
        },
        title='Satellite Stress Index vs Predicted Water Stress'
    )
    fig_scatter.update_layout(
        paper_bgcolor='#07111f', plot_bgcolor='#07111f',
        font=dict(color='#8ab8cc', size=11), height=400,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(gridcolor='#0e2a45'),
        yaxis=dict(gridcolor='#0e2a45'),
        legend=dict(
            bgcolor='#0d1b2a',
            bordercolor='#0e2a45',
            borderwidth=1,
            font=dict(color='#e2f4fb')
        )
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# ROW 4 — DATA TABLE
# ============================================================
st.markdown("<div class='section-header'>LGA Intelligence Report</div>", unsafe_allow_html=True)

table_df = filtered_df[[
    'lga_name','predicted_stress','confidence',
    'lower_bound','upper_bound','functionality_rate',
    'total_stress_rate','avg_depth_m','elevation',
    'ndvi','soil_moisture','lst_celsius',
    'total_pop_served','area_km2'
]].sort_values('predicted_stress', ascending=False).copy()

table_df.columns = [
    'LGA','Stress Score','Confidence','Lower Bound','Upper Bound',
    'Functionality %','Total Stress %','Avg Depth (m)','Elevation (m)',
    'NDVI','Soil Moisture','LST (C)','Pop Served','Area (km2)'
]

def highlight_stress(val):
    if isinstance(val, float) and 0 <= val <= 1:
        if val >= 0.4:   return 'color: #e05252; font-weight: bold'
        elif val >= 0.3: return 'color: #e07d30; font-weight: bold'
        else:            return 'color: #22c77a'
    return ''

def highlight_confidence(val):
    mapping = {
        'High':   'color: #22c77a; font-weight: bold',
        'Medium': 'color: #e07d30; font-weight: bold',
        'Low':    'color: #e05252; font-weight: bold'
    }
    return mapping.get(val, '')

styled_table = table_df.style\
    .map(highlight_stress,    subset=['Stress Score'])\
    .map(highlight_confidence, subset=['Confidence'])\
    .format({
        'Stress Score':    '{:.3f}',
        'Lower Bound':     '{:.3f}',
        'Upper Bound':     '{:.3f}',
        'Functionality %': '{:.1f}',
        'Total Stress %':  '{:.1f}',
        'Avg Depth (m)':   '{:.1f}',
        'Elevation (m)':   '{:.0f}',
        'NDVI':            '{:.3f}',
        'Soil Moisture':   '{:.3f}',
        'LST (C)':         '{:.1f}',
        'Pop Served':      '{:,.0f}',
        'Area (km2)':      '{:.0f}'
    })

st.dataframe(styled_table, use_container_width=True, height=400)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#3d7a9e; font-size:0.7rem; padding:12px; line-height:2.2; letter-spacing:0.5px'>
    <b style='color:#0ea5c9'>Aqua-Spatial Intelligence Platform</b>
    &nbsp;&nbsp;|&nbsp;&nbsp;
    Enugu State Water Security Initiative
    &nbsp;&nbsp;|&nbsp;&nbsp;
    Satellite Data: NASA Landsat 8 &middot; SMAP &middot; SRTM &middot; MODIS via Google Earth Engine
    &nbsp;&nbsp;|&nbsp;&nbsp;
    Model: Random Forest + Gradient Boosting Ensemble &nbsp; R&#178;=0.60 &nbsp; MAE=0.069
    <br>
    <span style='color:#e05252'>
        Ground truth water point data is synthetic and calibrated against UNICEF WASH Nigeria benchmarks.
        For demonstration purposes only.
    </span>
</div>
""", unsafe_allow_html=True)
