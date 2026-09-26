import streamlit as st
import requests
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import os
import io
import streamlit.components.v1 as components
from streamlit.components.v1 import html
from io import BytesIO

from streamlit_lottie import st_lottie

st.set_page_config(
    layout="wide",
    page_title="HCPredict",
    page_icon="🧬",
    initial_sidebar_state="collapsed",
)


# ===========================================================================
# DESIGN TOKENS
# ===========================================================================
INK_0 = "#05040A"
INK_1 = "#08060D"
INK_2 = "#0D0712"
CORAL = "#FF4F87"
CORAL_2 = "#FF5C8A"
CORAL_3 = "#FF6F9D"
ORANGE = "#FF8A65"
ORANGE_2 = "#FF9E78"
ORANGE_3 = "#FFB08A"
PURPLE = "#8B4DFF"
PURPLE_2 = "#A66CFF"
TEXT = "#F7EAF2"
TEXT_2 = "#EADDE7"
TEXT_DIM = "#B9AAB8"

PLOT_COLORWAY = [CORAL, ORANGE, PURPLE_2, ORANGE_3, PURPLE, CORAL_3, "#E8C9FF"]
BIO_SCALE = [[0.0, "#0D0712"], [0.35, "#4A2470"], [0.7, PURPLE_2], [1.0, ORANGE_3]]


def style_fig(fig):
    """Apply the bioluminescent theme to any plotly figure. Minimal grid, no chart chrome."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=PLOT_COLORWAY,
        font=dict(family="Inter, sans-serif", color=TEXT_DIM, size=12),
        title_font=dict(family="Cormorant Garamond, serif", color=TEXT, size=20),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_DIM, size=11)),
        margin=dict(l=36, r=22, t=34, b=36),
        hoverlabel=dict(bgcolor="#140A18", font=dict(color=TEXT, family="Inter"), bordercolor=CORAL),
    )
    fig.update_xaxes(showgrid=False, linecolor="rgba(247,234,242,0.14)",
                     tickfont=dict(color=TEXT_DIM, size=11), title_font=dict(size=11))
    fig.update_yaxes(gridcolor="rgba(166,108,255,0.10)", zerolinecolor="rgba(166,108,255,0.16)",
                     linecolor="rgba(0,0,0,0)", tickfont=dict(color=TEXT_DIM, size=11),
                     title_font=dict(size=11))
    return fig


# App state management for loading overlay
if 'initial_load' not in st.session_state:
    st.session_state.initial_load = True


# Function to load Lottie animation
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


# ===========================================================================
# DATA LAYER — behaviour unchanged
# ===========================================================================
def init_db():
    """Initialize SQLite database with required tables if they don't exist"""
    conn = sqlite3.connect('npi_survey_data.db')
    c = conn.cursor()

    c.execute('''
    CREATE TABLE IF NOT EXISTS data_status (
        data_type TEXT PRIMARY KEY,
        uploaded BOOLEAN,
        last_updated TIMESTAMP
    )
    ''')

    c.execute('''
    CREATE TABLE IF NOT EXISTS csv_data (
        data_type TEXT PRIMARY KEY,
        csv_content BLOB
    )
    ''')

    c.execute("INSERT OR IGNORE INTO data_status VALUES ('npi', 0, NULL)")
    c.execute("INSERT OR IGNORE INTO data_status VALUES ('survey', 0, NULL)")

    conn.commit()
    conn.close()


def check_data_status():
    """Check if required data has been uploaded"""
    conn = sqlite3.connect('npi_survey_data.db')
    c = conn.cursor()

    c.execute("SELECT data_type, uploaded, last_updated FROM data_status")
    results = c.fetchall()

    status = {}
    for data_type, uploaded, last_updated in results:
        status[data_type] = {'uploaded': bool(uploaded), 'last_updated': last_updated}

    conn.close()
    return status


def store_csv_data(data_type, csv_file, skip_validation=False):
    """Store uploaded CSV data in the database.

    skip_validation=True lets a caller that has *already* successfully parsed this
    exact content (e.g. right after `pd.read_csv` in the Save handlers, or for the
    known-good embedded sample data) skip re-parsing the whole CSV a second time here.
    This duplicate parse was part of why saving NPI data felt slow on larger files.
    """
    conn = sqlite3.connect('npi_survey_data.db')
    c = conn.cursor()

    # Read and validate CSV content
    csv_content = csv_file.read()
    if not csv_content:
        conn.close()
        st.error(f"The uploaded {data_type} CSV file is empty.")
        return False

    if not skip_validation:
        try:
            # Attempt to parse the CSV to ensure it's valid
            pd.read_csv(io.BytesIO(csv_content))
        except pd.errors.EmptyDataError:
            conn.close()
            st.error(f"The uploaded {data_type} CSV file is empty or invalid.")
            return False
        except Exception as e:
            conn.close()
            st.error(f"Invalid {data_type} CSV file: {str(e)}")
            return False

    # Store the content
    c.execute("INSERT OR REPLACE INTO csv_data VALUES (?, ?)", (data_type, sqlite3.Binary(csv_content)))

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("UPDATE data_status SET uploaded = 1, last_updated = ? WHERE data_type = ?",
              (now, data_type))

    conn.commit()
    conn.close()

    st.markdown(
        f"<div class='log-line'>signal stored · {data_type} · {len(csv_content)} bytes</div>",
        unsafe_allow_html=True
    )
    return True


def load_csv_data(data_type):
    """Load CSV data from the database"""
    conn = sqlite3.connect('npi_survey_data.db')
    c = conn.cursor()

    c.execute("SELECT csv_content FROM csv_data WHERE data_type = ?", (data_type,))
    result = c.fetchone()

    if result:
        csv_content = result[0]
        if not csv_content:  # Check if content is empty
            conn.close()
            st.error(f"No valid CSV content found for {data_type} in the database.")
            return None
        try:
            df = pd.read_csv(io.BytesIO(csv_content))
            conn.close()
            return df
        except pd.errors.EmptyDataError:
            conn.close()
            st.error(f"The {data_type} CSV file is empty or invalid.")
            return None
        except Exception as e:
            conn.close()
            st.error(f"Error reading {data_type} CSV from database: {str(e)}")
            return None

    conn.close()
    st.warning(f"No {data_type} data found in the database.")
    return None


def clear_data(data_type):
    """Clear specific data from the database"""
    conn = sqlite3.connect('npi_survey_data.db')
    c = conn.cursor()

    c.execute("DELETE FROM csv_data WHERE data_type = ?", (data_type,))

    c.execute("UPDATE data_status SET uploaded = 0, last_updated = NULL WHERE data_type = ?",
              (data_type,))

    conn.commit()
    conn.close()

    return True


# Function to convert hour:minute to minutes since midnight
def to_minutes(hour, minute):
    return hour * 60 + minute


# Function to calculate active time window considering midnight spanning
def calculate_active_window(row):
    login_mins = to_minutes(row['login_hour'], row['login_minute'])
    logout_mins = to_minutes(row['logout_hour'], row['logout_minute'])
    login_date = datetime.strptime(row['login_date'], "%Y-%m-%d")
    logout_date = datetime.strptime(row['logout_date'], "%Y-%m-%d")

    if logout_date > login_date or (logout_date == login_date and logout_mins < login_mins):
        active_time = (1440 - login_mins) + logout_mins
    else:
        active_time = logout_mins - login_mins

    return login_mins, logout_mins, active_time


# Preprocess npi.csv: Extract time patterns and verify usage time
# Vectorized (was row-wise .apply() calling datetime.strptime per row, which was the
# main bottleneck behind the slow "Save NPI data" step on larger files). Produces
# numerically identical output to the previous row-by-row version.
def preprocess_npi_data(npi_df):
    npi_df = npi_df.copy()

    login_hour = npi_df['login_hour'].astype(int)
    login_minute = npi_df['login_minute'].astype(int)
    logout_hour = npi_df['logout_hour'].astype(int)
    logout_minute = npi_df['logout_minute'].astype(int)

    login_mins = login_hour * 60 + login_minute
    logout_mins = logout_hour * 60 + logout_minute

    login_date = pd.to_datetime(npi_df['login_date'])
    logout_date = pd.to_datetime(npi_df['logout_date'])

    spans_midnight = (logout_date > login_date) | ((logout_date == login_date) & (logout_mins < login_mins))
    active_time = np.where(spans_midnight, (1440 - login_mins) + logout_mins, logout_mins - login_mins)

    npi_df['login_mins'] = login_mins
    npi_df['logout_mins'] = logout_mins
    npi_df['calculated_active_time'] = active_time
    npi_df['usage_time_valid'] = (npi_df['calculated_active_time'] - npi_df['Usage Time (mins)']).abs() <= 5

    return npi_df


# Function to check if an NPI is active in a given time slot
def is_active_in_timeslot(row, target_time, window_size=60):
    target_mins = to_minutes(target_time[0], target_time[1])

    half_window = window_size // 2
    slot_start = max(0, target_mins - half_window)
    slot_end = min(1439, target_mins + half_window)

    login_mins = row['login_mins']
    logout_mins = row['logout_mins']

    if logout_mins < login_mins:
        if slot_start <= 1439 and login_mins <= slot_end:
            return True
        if slot_end >= 0 and logout_mins >= slot_start:
            return True
        return False
    else:
        return max(login_mins, slot_start) <= min(logout_mins, slot_end)


# Extract features for the model
def extract_features(row, target_time):
    target_mins = to_minutes(target_time[0], target_time[1])

    hour = target_time[0]
    minute = target_time[1]

    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    is_morning = 1 if 6 <= hour < 12 else 0
    is_afternoon = 1 if 12 <= hour < 18 else 0
    is_evening = 1 if 18 <= hour < 24 else 0
    is_night = 1 if 0 <= hour < 6 else 0

    survey_attempts_normalized = row['Count of Survey Attempts'] / 10.0
    usage_time_normalized = row['Usage Time (mins)'] / 240.0

    region_features = [
        row.get('Region_Midwest', 0),
        row.get('Region_Northeast', 0),
        row.get('Region_South', 0),
        row.get('Region_West', 0)
    ]

    specialty_features = [
        row.get('Speciality_Cardiology', 0),
        row.get('Speciality_General Practice', 0),
        row.get('Speciality_Neurology', 0),
        row.get('Speciality_Oncology', 0),
        row.get('Speciality_Orthopedics', 0),
        row.get('Speciality_Pediatrics', 0),
        row.get('Speciality_Radiology', 0)
    ]

    login_mins = row['login_mins']
    logout_mins = row['logout_mins']

    if logout_mins < login_mins:
        if login_mins <= target_mins:
            time_since_login = target_mins - login_mins
            time_until_logout = (1440 - target_mins) + logout_mins
        else:
            time_since_login = (1440 - login_mins) + target_mins
            time_until_logout = logout_mins - target_mins
    else:
        if login_mins <= target_mins <= logout_mins:
            time_since_login = target_mins - login_mins
            time_until_logout = logout_mins - target_mins
        elif target_mins < login_mins:
            time_since_login = -1 * (login_mins - target_mins)
            time_until_logout = (logout_mins - login_mins) + abs(time_since_login)
        else:
            time_until_logout = -1 * (target_mins - logout_mins)
            time_since_login = (logout_mins - login_mins) + abs(time_until_logout)

    time_since_login_normalized = time_since_login / 1440.0
    time_until_logout_normalized = time_until_logout / 1440.0

    inside_window = 1 if is_active_in_timeslot(row, target_time) else 0

    return [hour_sin, hour_cos, is_morning, is_afternoon, is_evening, is_night,
            survey_attempts_normalized, usage_time_normalized,
            time_since_login_normalized, time_until_logout_normalized,
            inside_window] + region_features + specialty_features


# Generate training data for the Random Forest model
def generate_training_data(npi_df, survey_df):
    X_train = []
    y_train = []

    npi_survey_map = {}
    for _, survey_row in survey_df.iterrows():
        npi = survey_row['NPI']
        attempt_time = (survey_row['attempt_hour'], survey_row['attempt_minute'])

        if npi not in npi_survey_map:
            npi_survey_map[npi] = []

        npi_survey_map[npi].append(attempt_time)

    for _, row in npi_df.iterrows():
        npi = row['NPI']
        survey_times = npi_survey_map.get(npi, [])

        for hour in range(0, 24, 2):
            for minute in [0, 30]:
                target_time = (hour, minute)
                features = extract_features(row, target_time)

                participated = 0
                for s_time in survey_times:
                    s_mins = to_minutes(s_time[0], s_time[1])
                    t_mins = to_minutes(target_time[0], target_time[1])
                    if abs(s_mins - t_mins) <= 30:
                        participated = 1
                        break

                X_train.append(features)
                y_train.append(participated)

    return np.array(X_train), np.array(y_train)


# Train the Random Forest model on NPI survey participation patterns
@st.cache_resource(show_spinner=False)
def train_rf_model(npi_df, survey_df):
    X, y = generate_training_data(npi_df, survey_df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_model = RandomForestClassifier(
        n_estimators=10,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    # stored so the metric card can show the real number instead of a hardcoded one
    st.session_state.model_accuracy = accuracy

    return rf_model


# Main function to analyze survey participation
def analyze_survey_participation(survey_id, time_str, survey_df, npi_df, rf_model):
    survey_row = survey_df[survey_df['Survey ID'] == survey_id]
    if survey_row.empty:
        return "Survey ID not found."

    try:
        hh, mm = map(int, time_str.split(':'))
        if hh < 0 or hh > 23 or mm < 0 or mm > 59:
            return "Invalid time format. Please use HH:MM in 24-hour format."
        target_time = (hh, mm)
    except Exception:
        return "Invalid time format. Please use HH:MM in 24-hour format."

    survey_participants = set(survey_df[survey_df['Survey ID'] == survey_id]['NPI'].tolist())

    active_npis = []

    for _, row in npi_df.iterrows():
        npi = row['NPI']

        is_active = is_active_in_timeslot(row, target_time)

        if is_active:
            participated = npi in survey_participants

            if participated:
                participation_prob = 1.0
            else:
                features = extract_features(row, target_time)
                participation_prob = rf_model.predict_proba([features])[0][1]

            region = "Unknown"
            if row.get('Region_Midwest', 0) == 1:
                region = "Midwest"
            elif row.get('Region_Northeast', 0) == 1:
                region = "Northeast"
            elif row.get('Region_South', 0) == 1:
                region = "South"
            elif row.get('Region_West', 0) == 1:
                region = "West"

            state_columns = [col for col in row.index if col.startswith('State_')]
            state = "Unknown"
            for state_col in state_columns:
                if row[state_col] == 1:
                    state = state_col.replace('State_', '')
                    break

            specialty_columns = [col for col in row.index if col.startswith('Speciality_')]
            specialty = "Unknown"
            for specialty_col in specialty_columns:
                if row[specialty_col] == 1:
                    specialty = specialty_col.replace('Speciality_', '')
                    break

            active_npis.append({
                'NPI': npi,
                'Participated': participated,
                'Participation Probability': participation_prob,
                'Survey Attempts History': row['Count of Survey Attempts'],
                'Usage Time': row['Usage Time (mins)'],
                'Active Window': f"{row['login_hour']:02d}:{row['login_minute']:02d} to {row['logout_hour']:02d}:{row['logout_minute']:02d}",
                'Region': region,
                'State': state,
                'Specialty': specialty
            })

    active_npis_sorted = sorted(active_npis, key=lambda x: x['Participation Probability'], reverse=True)

    participants_count = sum(1 for npi in active_npis if npi['Participated'])
    active_npi_count = len(active_npis)

    participation_percentage = (participants_count / active_npi_count * 100) if active_npi_count > 0 else 0

    output = {
        'Survey ID': survey_id,
        'Analysis Time': time_str,
        'Total NPIs in Database': len(npi_df),
        'Active NPIs at Analysis Time': active_npi_count,
        'Survey Participants Among Active NPIs': participants_count,
        'Participation Percentage': participation_percentage,
        'Active NPIs with Participation Probability': active_npis_sorted
    }

    return output


def _active_mask_vectorized(login_mins, logout_mins, target_mins, window_size=60):
    """Vectorized equivalent of is_active_in_timeslot's boolean logic, applied to whole
    numpy arrays at once instead of one row of the dataframe at a time. Same semantics,
    just without the O(rows) Python-level loop per timeslot."""
    half_window = window_size // 2
    slot_start = max(0, target_mins - half_window)
    slot_end = min(1439, target_mins + half_window)

    crosses_midnight = logout_mins < login_mins
    normal_active = np.maximum(login_mins, slot_start) <= np.minimum(logout_mins, slot_end)
    cross_active = (login_mins <= slot_end) | (logout_mins >= slot_start)
    return np.where(crosses_midnight, cross_active, normal_active)


# Function to analyze active NPIs at different times.
# Cached + vectorized: this used to run a full Python-level row loop (npi_df.iterrows())
# 48 times (every half hour across a day) on every single Streamlit rerun, even ones
# unrelated to this chart — the main reason "NPI Activity by Time" felt slow. It now
# operates on numpy arrays and is cached against the actual data.
@st.cache_data(show_spinner=False)
def analyze_active_npis_by_time(npi_df):
    login_arr = npi_df['login_mins'].to_numpy()
    logout_arr = npi_df['logout_mins'].to_numpy()

    time_counts = {}
    for hour in range(24):
        for minute in [0, 30]:
            time_str = f"{hour:02d}:{minute:02d}"
            target_mins = to_minutes(hour, minute)
            mask = _active_mask_vectorized(login_arr, logout_arr, target_mins)
            time_counts[time_str] = int(mask.sum())

    return time_counts


# Create visualizations for region, state, and specialty distributions
def create_visualizations(active_npi_data):
    df = pd.DataFrame(active_npi_data)

    region_counts = df['Region'].value_counts().reset_index()
    region_counts.columns = ['Region', 'Count']

    fig_region = px.bar(
        region_counts, x='Region', y='Count', title='', color='Region',
        color_discrete_sequence=PLOT_COLORWAY, labels={'Count': 'Active NPIs'}, height=440
    )

    state_counts = df['State'].value_counts().reset_index()
    state_counts.columns = ['State', 'Count']
    state_counts = state_counts.head(15)

    fig_state = px.bar(
        state_counts, x='State', y='Count', title='', color='State',
        color_discrete_sequence=PLOT_COLORWAY, labels={'Count': 'Active NPIs'}, height=380
    )

    specialty_counts = df['Specialty'].value_counts().reset_index()
    specialty_counts.columns = ['Specialty', 'Count']

    fig_specialty = px.pie(
        specialty_counts, values='Count', names='Specialty', title='',
        color_discrete_sequence=PLOT_COLORWAY, hole=0.62, height=340
    )
    fig_specialty.update_traces(marker=dict(line=dict(color=INK_1, width=2)))

    region_participation = df.groupby('Region')['Participated'].agg(['sum', 'count']).reset_index()
    region_participation['Participation Rate'] = (region_participation['sum'] / region_participation['count'] * 100).round(2)
    region_participation.columns = ['Region', 'Participants', 'Total', 'Participation Rate (%)']

    fig_region_participation = px.bar(
        region_participation, x='Region', y='Participation Rate (%)', title='',
        color='Region', color_discrete_sequence=PLOT_COLORWAY, height=340
    )

    specialty_participation = df.groupby('Specialty')['Participated'].agg(['sum', 'count']).reset_index()
    specialty_participation['Participation Rate'] = (specialty_participation['sum'] / specialty_participation['count'] * 100).round(2)
    specialty_participation.columns = ['Specialty', 'Participants', 'Total', 'Participation Rate (%)']

    fig_specialty_participation = px.bar(
        specialty_participation, x='Specialty', y='Participation Rate (%)', title='',
        color='Specialty', color_discrete_sequence=PLOT_COLORWAY, height=340
    )

    for f in (fig_region, fig_state, fig_specialty, fig_region_participation, fig_specialty_participation):
        style_fig(f)

    return fig_region, fig_state, fig_specialty, fig_region_participation, fig_specialty_participation


# ===========================================================================
# STYLE SYSTEM
# 01 tokens · 02 base + atmosphere · 03 typography · 04 nav · 05 cards ·
# 06 buttons · 07 inputs + upload · 08 tabs · 09 guidelines drawer ·
# 10 nav rail · 11 motion · 12 responsive
# ===========================================================================
GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;0,600;1,300;1,400&family=Inter:wght@300;400;500;600;700&family=Manrope:wght@400;500;600;700&display=swap');

/* ---------- 01 · TOKENS ---------- */
:root {
    --ink-0:#05040A; --ink-1:#08060D; --ink-2:#0D0712;
    --coral:#FF4F87; --coral-2:#FF5C8A; --coral-3:#FF6F9D;
    --orange:#FF8A65; --orange-2:#FF9E78; --orange-3:#FFB08A;
    --purple:#8B4DFF; --purple-2:#A66CFF;
    --text:#F7EAF2; --text-2:#EADDE7; --dim:#B9AAB8;
    --card-bg: rgba(20,10,24,0.55);
    --card-bd: 1px solid rgba(255,105,150,0.18);
    /* organic / asymmetric radii instead of uniform rounded rectangles */
    --r-lg: 34px 34px 34px 10px;
    --r-lg-alt: 34px 10px 34px 34px;
    --r-md: 24px 24px 24px 8px;
    --r-sm: 16px;
    --r-pill: 999px;
    --serif:'Cormorant Garamond', Georgia, serif;
    --sans:'Inter', 'Manrope', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* ---------- 02 · BASE + ATMOSPHERE ---------- */
/* Deep black / near-black base with dark purple atmosphere, pink-coral bioluminescent
   glow and warm orange highlights — layered radial gradients standing in for the
   reference image so no external asset is required. Replace with
   `background-image:url(...)` to swap in a real asset later if desired. */
.stApp {
    background:
        radial-gradient(1500px 900px at 86% 4%,  rgba(255,79,135,0.24), transparent 58%),
        radial-gradient(1200px 820px at 100% 30%, rgba(255,138,101,0.17), transparent 56%),
        radial-gradient(1000px 760px at 68% 46%, rgba(255,111,157,0.12), transparent 60%),
        radial-gradient(1300px 940px at 4%  20%, rgba(139,77,255,0.20), transparent 62%),
        radial-gradient(1000px 780px at 50% 118%, rgba(166,108,255,0.13), transparent 60%),
        radial-gradient(800px 600px at 30% 0%, rgba(139,77,255,0.10), transparent 55%),
        linear-gradient(180deg, #030209 0%, #07050C 42%, #0B0610 78%, #0D0712 100%);
    background-attachment: fixed;
}
.stApp [data-testid="stAppViewContainer"] { background:transparent !important; }
[data-testid="stHeader"] { background:transparent !important; }
/* Full viewport canvas — keep the interface layered over the atmosphere instead
   of putting it inside a narrow centered dashboard window. */
.block-container {
    padding-top:2.2rem !important;
    width:100% !important;
    max-width:none !important;
    margin:0 !important;
    padding-left:clamp(20px, 3vw, 56px) !important;
    padding-right:clamp(20px, 3vw, 56px) !important;
}

html, body, [class*="css"], p, li, span, label,
div[data-testid="stMarkdownContainer"] { font-family:var(--sans); color:var(--text-2); }

.bio-field { position:fixed; inset:0; z-index:0; pointer-events:none; overflow:hidden; }
.bio-field::before {
    content:""; position:absolute; width:min(68vw, 980px); aspect-ratio:1;
    top:-15%; right:-13%; border-radius:50%;
    background:
        radial-gradient(ellipse 18% 48% at 50% 4%, rgba(255,79,135,0.34), transparent 72%),
        radial-gradient(ellipse 18% 48% at 50% 96%, rgba(166,108,255,0.26), transparent 72%),
        radial-gradient(ellipse 48% 18% at 4% 50%, rgba(255,138,101,0.26), transparent 72%),
        radial-gradient(ellipse 48% 18% at 96% 50%, rgba(255,111,157,0.30), transparent 72%),
        radial-gradient(ellipse 24% 42% at 20% 18%, rgba(139,77,255,0.24), transparent 72%),
        radial-gradient(ellipse 24% 42% at 80% 82%, rgba(255,176,138,0.22), transparent 72%),
        radial-gradient(circle at 50% 50%, rgba(255,92,138,0.48), rgba(139,77,255,0.10) 42%, transparent 70%);
    filter:blur(22px); opacity:0.72; mix-blend-mode:screen;
    transform:rotate(-16deg); animation:flowerDrift 42s ease-in-out infinite;
}
.bio-field::after {
    content:""; position:absolute; width:min(44vw, 620px); aspect-ratio:1;
    top:3%; right:5%; border-radius:50%;
    background:radial-gradient(circle, rgba(255,176,138,0.22) 0 3%, rgba(255,79,135,0.14) 16%, transparent 62%);
    filter:blur(36px); opacity:0.56; mix-blend-mode:screen;
    animation:flowerPulse 16s ease-in-out infinite;
}
@keyframes flowerDrift {
    0%,100% { transform:rotate(-16deg) scale(0.96) translate(0,0); }
    50% { transform:rotate(-4deg) scale(1.05) translate(-34px,34px); }
}
@keyframes flowerPulse {
    0%,100% { transform:scale(0.92); opacity:0.42; }
    50% { transform:scale(1.08); opacity:0.72; }
}
.bio-blob { position:absolute; border-radius:50%; filter:blur(86px); opacity:0.46; mix-blend-mode:screen; }
.b1 { width:620px; height:620px; top:-12%; right:-4%;
      background:radial-gradient(circle at 38% 38%, rgba(255,79,135,0.64), transparent 68%);
      animation:drift1 38s ease-in-out infinite; }
.b2 { width:520px; height:520px; top:22%; right:4%;
      background:radial-gradient(circle at 55% 45%, rgba(255,138,101,0.48), transparent 68%);
      animation:drift2 48s ease-in-out infinite; }
.b3 { width:560px; height:560px; bottom:-16%; left:-8%;
      background:radial-gradient(circle at 50% 50%, rgba(139,77,255,0.46), transparent 70%);
      animation:drift3 56s ease-in-out infinite; }
.b4 { width:400px; height:400px; top:48%; right:22%;
      background:radial-gradient(circle at 50% 50%, rgba(255,176,138,0.30), transparent 70%);
      animation:drift2 44s ease-in-out infinite 3s; }
.b5 { width:360px; height:360px; top:6%; left:30%;
      background:radial-gradient(circle at 50% 50%, rgba(166,108,255,0.22), transparent 72%);
      animation:drift3 50s ease-in-out infinite 6s; }
@keyframes drift1 { 0%,100%{transform:translate(0,0) scale(1);}    50%{transform:translate(-70px,80px) scale(1.12);} }
@keyframes drift2 { 0%,100%{transform:translate(0,0) scale(1.06);} 50%{transform:translate(60px,-70px) scale(0.92);} }
@keyframes drift3 { 0%,100%{transform:translate(0,0) scale(0.96);} 50%{transform:translate(80px,-60px) scale(1.12);} }

.mote { position:absolute; width:3px; height:3px; border-radius:50%;
        background:var(--orange-3); box-shadow:0 0 10px var(--coral); opacity:0.6; }
.m1{top:20%;left:22%;animation:breathe 11s ease-in-out infinite;}
.m2{top:62%;left:74%;animation:breathe 15s ease-in-out infinite 2s;}
.m3{top:38%;left:52%;animation:breathe 13s ease-in-out infinite 4s;}
.m4{top:80%;left:34%;animation:breathe 17s ease-in-out infinite 1s;}
.m5{top:12%;left:66%;animation:breathe 14s ease-in-out infinite 5s;}
.m6{top:70%;left:12%;animation:breathe 19s ease-in-out infinite 3s;}
.m7{top:30%;left:84%;animation:breathe 12s ease-in-out infinite 2.5s;}
.m8{top:88%;left:60%;animation:breathe 16s ease-in-out infinite 4.5s;}
@keyframes breathe { 0%,100%{opacity:0.12; transform:scale(0.8);} 50%{opacity:0.9; transform:scale(1.7);} }
section.main, section[data-testid="stSidebar"] { position:relative; z-index:1; }
/* Keeps the header row (nav + Guidelines toggle) clickable above the drawer/scrim. */
#topbar-anchor + div[data-testid="stHorizontalBlock"] { position:relative; z-index:1002; }

/* ---------- 03 · TYPOGRAPHY ---------- */
/* UI font sizes raised roughly 5-6px across labels, body copy and helper text for
   readability; the large display headings stay proportionally much larger. */
h1, h2, h3, h4 { font-family:var(--serif) !important; color:var(--text) !important; font-weight:400 !important; }
h1 { font-size:80px !important; line-height:1.02 !important; letter-spacing:-0.5px; }
h2 { font-size:48px !important; line-height:1.1 !important; }
h3 { font-size:36px !important; }
h4 { font-size:24px !important; }
.hero-h { font-family:var(--serif); font-size:clamp(52px, 6.6vw, 92px); line-height:1.0;
          color:var(--text); font-weight:400; letter-spacing:-1px; margin:16px 0 0 0; }
.hero-h em { font-style:italic; font-weight:300;
             background:linear-gradient(96deg, var(--coral), var(--orange) 55%, var(--purple-2));
             -webkit-background-clip:text; background-clip:text; color:transparent; }
.eyebrow { font-family:var(--sans); font-size:13px; font-weight:600; letter-spacing:3.2px;
           text-transform:uppercase; color:var(--coral-3); }
.lede { font-family:var(--sans); font-size:20px; line-height:1.7; color:var(--dim); max-width:500px; }
.log-line { font-family:var(--sans); font-size:14px; letter-spacing:1.6px; text-transform:uppercase;
            color:var(--dim); opacity:0.75; margin:8px 0 2px; }

.sec { margin: 8px 0 26px 0; }
.sec .eyebrow { display:block; margin-bottom:12px; }
.sec-h { font-family:var(--serif); font-size:clamp(36px,3.6vw,52px); line-height:1.08;
         color:var(--text); font-weight:400; margin:0; }
.sec-d { font-family:var(--sans); font-size:18px; line-height:1.6; color:var(--dim); margin-top:10px; max-width:600px; }

.status-pill { display:inline-flex; align-items:center; gap:8px; font-family:var(--sans);
    font-size:14px; letter-spacing:0.8px; color:var(--text-2); padding:9px 18px;
    border-radius:var(--r-pill); border:1px solid rgba(166,255,190,0.28);
    background:rgba(120,255,180,0.08); margin-bottom:16px; }
.or-divider { display:flex; align-items:center; gap:14px; margin:18px 0; color:var(--dim);
    font-family:var(--sans); font-size:12px; letter-spacing:2px; text-transform:uppercase; }
.or-divider::before, .or-divider::after { content:""; flex:1; height:1px;
    background:rgba(255,255,255,0.10); }
.upload-label { font-family:var(--sans); font-size:14px; font-weight:500; letter-spacing:0.6px;
    color:var(--dim); margin-bottom:10px; }
.data-card-h { font-family:var(--serif); font-size:38px; margin-top:10px; color:var(--text); }

/* ---------- 04 · NAV ---------- */
.nav {
    display:flex; align-items:center; justify-content:space-between; gap:20px;
    padding:13px 26px; margin:0 0 18px 0;
    border-radius:999px; border:1px solid rgba(255,255,255,0.06);
    background:rgba(13,7,18,0.5); backdrop-filter:blur(16px);
}
.nav-brand { font-family:var(--serif); font-size:26px; color:var(--text); letter-spacing:0.4px; }
.nav-dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:10px;
           background:var(--coral); box-shadow:0 0 12px var(--coral); animation:breathe 5s ease-in-out infinite; }
.nav-links { display:flex; gap:32px; }
.nav-links span { font-family:var(--sans); font-size:15px; font-weight:500; letter-spacing:1.4px; text-transform:uppercase;
                  color:var(--dim); position:relative; padding-bottom:5px; transition:color .35s ease; }
.nav-links span::after { content:""; position:absolute; left:0; bottom:0; height:1px; width:0;
                         background:linear-gradient(90deg, var(--coral), var(--orange));
                         box-shadow:0 0 8px var(--coral); transition:width .45s cubic-bezier(.2,.8,.3,1); }
.nav-links span:hover { color:var(--text); }
.nav-links span:hover::after { width:100%; }
.nav-cta { font-family:var(--sans); font-size:13px; font-weight:600; letter-spacing:1.6px;
           text-transform:uppercase; color:var(--text); padding:10px 22px; border-radius:999px;
           border:1px solid rgba(255,79,135,0.45);
           background:linear-gradient(135deg, rgba(255,79,135,0.20), rgba(139,77,255,0.16)); }

/* ---------- 05 · CARDS ---------- */
/* Organic, asymmetric shapes (one corner stays sharp) instead of uniform rounded
   rectangles — layered translucent glass panels with a soft glow, not neon. */
.card, .metric, .instrument, .data-card {
    background:var(--card-bg); border:var(--card-bd); backdrop-filter:blur(18px);
    box-shadow:0 14px 40px rgba(0,0,0,0.38); position:relative; overflow:hidden;
}
.card { border-radius:var(--r-lg); padding:28px 30px; margin-bottom:22px;
        transition:border-color .45s ease, transform .45s ease, box-shadow .45s ease; }
.card::before { content:""; position:absolute; inset:-1px; border-radius:inherit; pointer-events:none;
        background:radial-gradient(420px 160px at 12% 0%, rgba(255,79,135,0.12), transparent 70%),
                   radial-gradient(360px 160px at 92% 100%, rgba(139,77,255,0.12), transparent 70%); }
.card:hover { transform:translateY(-3px); border-color:rgba(255,105,150,0.34);
              box-shadow:0 20px 50px rgba(0,0,0,0.46), 0 0 26px rgba(255,79,135,0.10); }
.card-h { font-family:var(--serif); font-size:27px; color:var(--text); margin:0; }
.card-d { font-family:var(--sans); font-size:16px; color:var(--dim); margin:6px 0 16px; }

/* metrics — pill / capsule shaped rather than boxy rectangles */
.metrics-grid { display:grid; grid-template-columns:repeat(4, 1fr); gap:18px; margin-bottom:8px; }
.metric { border-radius:var(--r-pill) / 42px; padding:22px 30px 20px 32px; display:flex;
          flex-direction:column; gap:2px; }
.metric::before { content:""; position:absolute; inset:-1px; border-radius:inherit; pointer-events:none;
        background:radial-gradient(280px 130px at 100% 0%, rgba(255,138,101,0.18), transparent 70%); }
.metric .ico { font-size:16px; color:var(--coral-3); opacity:0.9; }
.metric .val { font-family:var(--serif); font-size:46px; line-height:1.05; color:var(--text); margin-top:8px; }
.metric .key { font-family:var(--sans); font-size:13px; font-weight:600; letter-spacing:2.2px;
               text-transform:uppercase; color:var(--dim); margin-top:8px; }

.instrument { border-radius:var(--r-lg-alt); padding:38px 42px; margin-bottom:26px; }
.instrument::before { content:""; position:absolute; inset:-1px; border-radius:inherit; pointer-events:none;
        background:radial-gradient(520px 200px at 50% -10%, rgba(255,79,135,0.14), transparent 72%); }

/* data input panels (NPI / Survey) — same family, alternating corner so the pair
   reads as two halves of one composition rather than duplicate boxes */
.data-card { border-radius:var(--r-lg); padding:30px 32px 32px; margin-bottom:22px; min-height:260px; }
.data-card::before { content:""; position:absolute; inset:-1px; border-radius:inherit; pointer-events:none;
        background:radial-gradient(360px 160px at 100% 0%, rgba(255,79,135,0.12), transparent 70%); }

/* ---------- 06 · BUTTONS ---------- */
/* Larger, premium pill buttons: bigger type, taller hit area, hover lift + glow +
   an animated light sweep (kept subtle — one sweep per hover, no looping shimmer). */
.stButton > button, .stDownloadButton > button {
    font-family:var(--sans) !important; font-size:15px !important; font-weight:600 !important;
    letter-spacing:1.2px !important; text-transform:uppercase !important; color:var(--text) !important;
    background:linear-gradient(135deg, rgba(255,79,135,0.22), rgba(139,77,255,0.16)) !important;
    border:1px solid rgba(255,105,150,0.40) !important;
    border-radius:999px !important; padding:15px 32px !important; min-height:52px !important;
    position:relative; overflow:hidden;
    transition:box-shadow .4s ease, border-color .4s ease, transform .25s cubic-bezier(.2,.8,.3,1) !important;
}
.stButton > button::before, .stDownloadButton > button::before {
    content:""; position:absolute; top:0; left:-70%; width:55%; height:100%;
    background:linear-gradient(120deg, rgba(255,255,255,0), rgba(255,176,138,0.34), rgba(255,255,255,0));
    transform:skewX(-18deg); transition:left .8s ease;
}
.stButton > button:hover::before, .stDownloadButton > button:hover::before { left:130%; }
.stButton > button:hover, .stDownloadButton > button:hover {
    border-color:var(--coral) !important; color:#fff !important;
    transform:translateY(-2px) !important;
    box-shadow:0 0 24px rgba(255,79,135,0.30), 0 0 40px rgba(255,138,101,0.14) !important;
}
.stButton > button[kind="primary"] {
    background:linear-gradient(135deg, var(--coral), #FF7A6E) !important;
    border:1px solid rgba(255,176,138,0.60) !important; color:#1A0710 !important; font-weight:700 !important;
    box-shadow:0 0 30px rgba(255,79,135,0.30), inset 0 0 18px rgba(255,176,138,0.25) !important;
}
.stButton > button[kind="primary"]:hover { transform:translateY(-2px) !important; }

/* ---------- 07 · INPUTS + UPLOAD ---------- */
div[data-testid="stTextInput"] input, div[data-testid="stNumberInput"] input {
    background:rgba(8,6,13,0.75) !important; color:var(--text) !important;
    border:1px solid rgba(166,108,255,0.28) !important; border-radius:var(--r-sm) !important;
    font-family:var(--sans) !important; font-size:20px !important; padding:14px 16px !important;
    transition:box-shadow .35s ease, border-color .35s ease;
}
div[data-testid="stTextInput"] input:focus, div[data-testid="stNumberInput"] input:focus {
    border-color:var(--coral) !important; box-shadow:0 0 18px rgba(255,79,135,0.26) !important;
}
div[data-testid="stWidgetLabel"] p {
    font-family:var(--sans) !important; font-size:15px !important; font-weight:600 !important;
    letter-spacing:2px !important; text-transform:uppercase; color:var(--dim) !important;
}
div[data-testid="stFileUploader"] section {
    border:1px dashed rgba(255,105,150,0.40) !important; border-radius:var(--r-md) !important;
    background:rgba(20,10,24,0.45) !important; padding:20px !important;
    transition:border-color .4s ease, box-shadow .4s ease;
}
div[data-testid="stFileUploader"] section:hover {
    border-color:var(--coral) !important; box-shadow:0 0 24px rgba(255,79,135,0.14) !important;
}
div[data-testid="stFileUploader"] section small,
div[data-testid="stFileUploader"] section span,
div[data-testid="stFileUploader"] section div {
    color:var(--dim) !important; font-family:var(--sans) !important; font-size:15px !important;
}
div[data-testid="stFileUploader"] button {
    font-size:14px !important; padding:10px 20px !important;
}
details[data-testid="stExpander"], div[data-testid="stExpander"] {
    background:rgba(20,10,24,0.45) !important; border:var(--card-bd) !important;
    border-radius:var(--r-lg) !important; backdrop-filter:blur(18px);
}
details[data-testid="stExpander"] summary p { color:var(--text) !important; letter-spacing:1.4px; font-size:16px !important; }

/* ---------- 08 · TABS + DATA ---------- */
div[data-baseweb="tab-list"] { background:transparent !important; gap:12px;
                               border-bottom:1px solid rgba(255,255,255,0.06); }
button[data-baseweb="tab"] { background:transparent !important; color:var(--dim) !important;
    font-family:var(--sans) !important; font-size:16px !important; font-weight:600 !important;
    letter-spacing:1.2px; text-transform:uppercase; padding:12px 6px !important; }
button[data-baseweb="tab"][aria-selected="true"] { color:var(--text) !important; }
div[data-baseweb="tab-highlight"] { background:linear-gradient(90deg, var(--coral), var(--orange)) !important; }
div[data-testid="stDataFrame"] { border:var(--card-bd); border-radius:var(--r-md); overflow:hidden;
                                 background:rgba(20,10,24,0.55); font-size:15px; }
div[data-testid="stAlert"] { background:rgba(255,79,135,0.09) !important;
    border:1px solid rgba(255,105,150,0.26) !important; border-radius:var(--r-sm) !important;
    color:var(--text) !important; font-family:var(--sans) !important; font-size:15px !important; }

/* ---------- 09 · GUIDELINES DRAWER ---------- */
/* A real anchor (not a bare div) so clicking anywhere on the scrim navigates to
   "?gd=0", which Python reads and uses to close the drawer — this is what makes
   "click outside to close" actually work, on top of the × and the header toggle. */
a.drawer-scrim { position:fixed; inset:0; background:rgba(5,4,10,0.55); backdrop-filter:blur(3px);
                 z-index:998; animation:fadeIn .35s ease both; display:block; cursor:pointer;
                 pointer-events:auto; }
.drawer {
    position:fixed; top:0; right:0; height:100vh; width:min(420px, 92vw); z-index:999;
    background:linear-gradient(180deg, rgba(20,10,24,0.94), rgba(8,6,13,0.96));
    border-left:1px solid rgba(255,105,150,0.20); backdrop-filter:blur(22px);
    padding:46px 34px; overflow-y:auto; overscroll-behavior:contain;
    box-shadow:-30px 0 70px rgba(0,0,0,0.55); isolation:isolate;
    animation:drawerIn .55s cubic-bezier(.2,.8,.3,1) both;
    border-radius:28px 0 0 28px;
}
@keyframes drawerIn { from{opacity:0; transform:translateX(46px); filter:blur(8px);}
                      to{opacity:1; transform:translateX(0); filter:blur(0);} }
@keyframes fadeIn { from{opacity:0;} to{opacity:1;} }
.drawer-head { display:flex; align-items:flex-start; justify-content:space-between; gap:12px; }
.drawer h3 { font-size:40px !important; margin:10px 0 26px 0; }
.drawer-close { flex:none; width:40px; height:40px; border-radius:50%; display:flex;
    align-items:center; justify-content:center; font-size:24px; line-height:1; color:var(--text);
    text-decoration:none; border:1px solid rgba(255,105,150,0.35);
    background:rgba(255,79,135,0.10); transition:background .3s ease, transform .3s ease; }
.drawer-close:hover { background:rgba(255,79,135,0.22); transform:rotate(90deg); }
.drawer-close:focus-visible { outline:2px solid var(--orange-3); outline-offset:4px; }
.g-row { display:flex; gap:18px; align-items:flex-start; padding:16px 0;
         border-bottom:1px solid rgba(255,255,255,0.05); }
.g-num { flex:none; width:38px; height:38px; border-radius:50%; display:flex; align-items:center;
         justify-content:center; font-family:var(--sans); font-size:13px; font-weight:600; color:var(--text);
         border:1px solid rgba(255,105,150,0.35);
         background:radial-gradient(circle at 35% 30%, rgba(255,79,135,0.30), rgba(139,77,255,0.14));
         box-shadow:0 0 16px rgba(255,79,135,0.18); }
.g-txt { font-family:var(--sans); font-size:17px; line-height:1.6; color:var(--text-2); padding-top:6px; }
.g-tip { margin-top:28px; padding:20px 22px; border-radius:var(--r-md);
         border:1px solid rgba(166,108,255,0.22); background:rgba(139,77,255,0.08); }

/* ---------- 10 · NAV RAIL ---------- */
section[data-testid="stSidebar"] {
    width:212px !important; min-width:212px !important;
    background:linear-gradient(180deg, rgba(8,6,13,0.92), rgba(16,8,22,0.92));
    border-right:1px solid rgba(255,255,255,0.05); backdrop-filter:blur(18px);
}
section[data-testid="stSidebar"] * { color:var(--text-2); font-family:var(--sans); }
.rail-item { display:flex; align-items:center; gap:12px; padding:13px 14px; border-radius:var(--r-sm);
             font-size:15px; letter-spacing:1px; text-transform:uppercase; color:var(--dim);
             transition:background .3s ease, color .3s ease; }
.rail-item:hover { background:rgba(255,79,135,0.10); color:var(--text); }
.rail-item.on { color:var(--text); background:rgba(255,79,135,0.14); }

/* ---------- 11 · MOTION ---------- */
@keyframes riseIn { from{opacity:0; transform:translateY(24px);} to{opacity:1; transform:translateY(0);} }
.rise   { animation:riseIn .9s cubic-bezier(.2,.8,.3,1) both; }
.rise-2 { animation:riseIn .9s cubic-bezier(.2,.8,.3,1) .12s both; }
.rise-3 { animation:riseIn .9s cubic-bezier(.2,.8,.3,1) .24s both; }
.spacer-xl { height:118px; } .spacer-lg { height:82px; } .spacer-md { height:46px; }

.result { display:flex; align-items:center; justify-content:space-between; gap:36px; flex-wrap:wrap; }
.result-verdict { font-family:var(--serif); font-size:clamp(38px,4vw,58px); line-height:1.05; color:var(--text); }
.result-sub { font-family:var(--sans); font-size:19px; color:var(--dim); margin-top:10px; max-width:460px; line-height:1.7; }
.ring-wrap { position:relative; width:170px; height:170px; flex:none; }
.ring-wrap svg { transform:rotate(-90deg); }
.ring-bg { fill:none; stroke:rgba(255,255,255,0.07); stroke-width:6; }
.ring-fg { fill:none; stroke:url(#ringgrad); stroke-width:6; stroke-linecap:round;
           filter:drop-shadow(0 0 8px rgba(255,79,135,0.55));
           animation:ringDraw 1.5s cubic-bezier(.2,.8,.3,1) both; }
@keyframes ringDraw { from { stroke-dashoffset:var(--dash-full); } }
.ring-num { position:absolute; inset:0; display:flex; flex-direction:column;
            align-items:center; justify-content:center; animation:riseIn .9s ease .5s both; }
.ring-num b { font-family:var(--serif); font-size:42px; font-weight:400; color:var(--text); }
.ring-num small { font-family:var(--sans); font-size:12px; letter-spacing:2.2px;
                  text-transform:uppercase; color:var(--dim); margin-top:4px; }

/* ---------- 12 · RESPONSIVE ---------- */
@media (max-width: 1100px) {
    .metrics-grid { grid-template-columns:repeat(2, 1fr); }
}
@media (max-width: 900px) {
    .block-container { padding-left:1rem !important; padding-right:1rem !important; }
    h1 { font-size:46px !important; }
    .hero-h { font-size:44px; }
    .nav { flex-direction:column; gap:14px; border-radius:24px; padding:18px; }
    .nav-links { flex-wrap:wrap; justify-content:center; gap:16px; }
    .instrument { padding:24px 20px; }
    .data-card { padding:24px 22px 26px; }
    .drawer { width:100vw; padding:34px 22px; border-radius:0; }
    .spacer-xl { height:64px; } .spacer-lg { height:48px; }
    .result { flex-direction:column; align-items:flex-start; }
    section[data-testid="stSidebar"] { width:168px !important; min-width:168px !important; }
    .metrics-grid { grid-template-columns:1fr 1fr; gap:12px; }
    .metric { padding:18px 20px 16px 22px; }
    .metric .val { font-size:36px; }
}
@media (max-width: 560px) {
    .metrics-grid { grid-template-columns:1fr; }
}
</style>

<div class="bio-field">
    <div class="bio-blob b1"></div><div class="bio-blob b2"></div><div class="bio-blob b3"></div>
    <div class="bio-blob b4"></div><div class="bio-blob b5"></div>
    <div class="mote m1"></div><div class="mote m2"></div><div class="mote m3"></div>
    <div class="mote m4"></div><div class="mote m5"></div><div class="mote m6"></div>
    <div class="mote m7"></div><div class="mote m8"></div>
</div>
"""


OPENING_ANIMATION = """
<style>
.boot { position:fixed; inset:0; z-index:10000;
    background:radial-gradient(900px 620px at 50% 45%, #10061A 0%, #05040A 72%);
    display:flex; flex-direction:column; justify-content:center; align-items:center; gap:16px; }
.boot-seed { position:absolute; width:10px; height:10px; border-radius:50%;
    background:#FF4F87; box-shadow:0 0 30px #FF4F87; animation:seed 1.6s ease-out forwards; }
@keyframes seed { 0%{transform:scale(0);opacity:0;} 25%{transform:scale(1);opacity:1;} 100%{transform:scale(0.2);opacity:0;} }
.boot-ecg { width:320px; height:70px; animation:ecgOut 2.6s ease-in forwards; }
.boot-ecg path { fill:none; stroke:#FF6F9D; stroke-width:2.2; stroke-linecap:round;
    filter:drop-shadow(0 0 8px #FF4F87); stroke-dasharray:620; stroke-dashoffset:620;
    animation:trace 1.5s ease-out .3s forwards; }
@keyframes trace { to{stroke-dashoffset:0;} }
@keyframes ecgOut { 0%,60%{opacity:1;filter:blur(0);} 100%{opacity:0;filter:blur(8px);transform:scale(1.14);} }
.boot-title { font-family:'Cormorant Garamond',serif; font-size:60px; font-weight:400; color:#F7EAF2;
    letter-spacing:1px; opacity:0; animation:titleIn 1s ease-out 2.1s forwards;
    text-shadow:0 0 40px rgba(255,79,135,0.42); }
.boot-sub { font-family:'Inter',sans-serif; font-size:10px; letter-spacing:4px; text-transform:uppercase;
    color:#B9AAB8; opacity:0; animation:titleIn .9s ease-out 2.6s forwards; }
@keyframes titleIn { from{opacity:0;transform:translateY(14px);filter:blur(6px);}
                     to{opacity:1;transform:translateY(0);filter:blur(0);} }
</style>
<div class="boot">
    <div class="boot-seed"></div>
    <svg class="boot-ecg" viewBox="0 0 320 70">
        <path d="M0,35 L95,35 L108,35 L118,12 L131,58 L143,35 L158,35 L168,22 L178,48 L188,35 L320,35"/>
    </svg>
    <div class="boot-title">HCPredict</div>
    <div class="boot-sub">Understanding people. Through data.</div>
</div>
"""


HERO_ORGANISM = """
<div style="display:flex; justify-content:center; align-items:center; min-height:420px;">
<svg viewBox="0 0 460 420" width="100%" style="max-width:540px; overflow:visible;">
  <defs>
    <radialGradient id="core" cx="50%" cy="50%">
      <stop offset="0%"   stop-color="#FFB08A" stop-opacity="0.95"/>
      <stop offset="34%"  stop-color="#FF5C8A" stop-opacity="0.58"/>
      <stop offset="68%"  stop-color="#8B4DFF" stop-opacity="0.22"/>
      <stop offset="100%" stop-color="#8B4DFF" stop-opacity="0"/>
    </radialGradient>
    <linearGradient id="thread" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%"   stop-color="#FF4F87"/>
      <stop offset="52%"  stop-color="#FF8A65"/>
      <stop offset="100%" stop-color="#A66CFF"/>
    </linearGradient>
    <filter id="soft"><feGaussianBlur stdDeviation="4"/></filter>
  </defs>

  <circle cx="240" cy="205" r="150" fill="url(#core)">
    <animate attributeName="r" values="142;164;142" dur="13s" repeatCount="indefinite"/>
    <animate attributeName="opacity" values="0.7;1;0.7" dur="13s" repeatCount="indefinite"/>
  </circle>

  <g stroke="url(#thread)" fill="none" stroke-width="1.4" filter="url(#soft)">
    <path d="M40,300 C130,268 140,168 240,152 C340,136 372,82 440,110">
      <animate attributeName="stroke-opacity" values="0.3;0.95;0.3" dur="10s" repeatCount="indefinite"/></path>
    <path d="M28,160 C140,200 168,272 268,272 C356,272 392,326 452,296">
      <animate attributeName="stroke-opacity" values="0.9;0.28;0.9" dur="14s" repeatCount="indefinite"/></path>
    <path d="M244,18 C204,112 300,158 272,242 C252,306 312,342 298,404">
      <animate attributeName="stroke-opacity" values="0.38;0.88;0.38" dur="16s" repeatCount="indefinite"/></path>
    <path d="M110,40 C186,96 148,208 222,254 C288,296 274,352 356,372">
      <animate attributeName="stroke-opacity" values="0.8;0.22;0.8" dur="19s" repeatCount="indefinite"/></path>
    <path d="M60,382 C150,356 196,300 240,206 C276,132 350,140 420,190">
      <animate attributeName="stroke-opacity" values="0.24;0.7;0.24" dur="22s" repeatCount="indefinite"/></path>
  </g>

  <g fill="#FFB08A">
    <circle cx="240" cy="152" r="3"><animate attributeName="r" values="2;5.2;2" dur="7s" repeatCount="indefinite"/></circle>
    <circle cx="268" cy="272" r="2.6" fill="#FF6F9D"><animate attributeName="r" values="1.6;4.6;1.6" dur="9s" repeatCount="indefinite"/></circle>
    <circle cx="148" cy="188" r="2.2" fill="#A66CFF"><animate attributeName="r" values="1.4;4;1.4" dur="11s" repeatCount="indefinite"/></circle>
    <circle cx="352" cy="96"  r="2.4"><animate attributeName="r" values="1.5;4.2;1.5" dur="8s" repeatCount="indefinite"/></circle>
    <circle cx="300" cy="360" r="2" fill="#FF4F87"><animate attributeName="r" values="1.2;3.8;1.2" dur="13s" repeatCount="indefinite"/></circle>
  </g>

  <g>
    <circle r="2.8" fill="#FFB08A" opacity="0.9">
      <animateMotion dur="11s" repeatCount="indefinite"
        path="M40,300 C130,268 140,168 240,152 C340,136 372,82 440,110"/></circle>
    <circle r="2.4" fill="#A66CFF" opacity="0.85">
      <animateMotion dur="15s" repeatCount="indefinite"
        path="M28,160 C140,200 168,272 268,272 C356,272 392,326 452,296"/></circle>
    <circle r="2.2" fill="#FF6F9D" opacity="0.8">
      <animateMotion dur="17s" repeatCount="indefinite"
        path="M244,18 C204,112 300,158 272,242 C252,306 312,342 298,404"/></circle>
  </g>
</svg>
</div>
"""


PREDICT_ANIMATION = """
<style>
.pa { display:flex; align-items:center; justify-content:center; gap:22px; flex-wrap:wrap; padding:40px 0; }
.pa span { font-family:'Inter',sans-serif; font-size:10.5px; letter-spacing:3px; text-transform:uppercase;
           color:#B9AAB8; opacity:0; animation:paIn .35s ease forwards; }
.pa span:nth-of-type(1){animation-delay:0s;} .pa span:nth-of-type(2){animation-delay:.28s;}
.pa span:nth-of-type(3){animation-delay:.56s;} .pa span:nth-of-type(4){animation-delay:.84s; color:#FF6F9D;}
.pa i { width:26px; height:1px; background:linear-gradient(90deg,#FF4F87,#FF8A65);
        display:inline-block; opacity:0; animation:paIn .3s ease forwards; }
.pa i:nth-of-type(1){animation-delay:.16s;} .pa i:nth-of-type(2){animation-delay:.44s;}
.pa i:nth-of-type(3){animation-delay:.72s;}
@keyframes paIn { from{opacity:0; transform:translateY(6px);} to{opacity:1; transform:translateY(0);} }
</style>
<div class="pa">
    <span>Input signals</span><i></i><span>Analyzing</span><i></i><span>Model</span><i></i><span>Result</span>
</div>
"""


def section_header(eyebrow, heading, description=""):
    desc = f'<div class="sec-d">{description}</div>' if description else ""
    return f"""
    <div class="sec rise">
        <span class="eyebrow">{eyebrow}</span>
        <div class="sec-h">{heading}</div>
        {desc}
    </div>
    """


NAV_BAR = """
<div class="nav rise">
    <div class="nav-brand"><span class="nav-dot"></span>HCPredict</div>
    <div class="nav-links">
        <span>Home</span><span>Predict</span><span>Explore</span><span>Insights</span><span>About</span>
    </div>
    <div class="nav-cta">Get started</div>
</div>
"""

FOOTER = """
<div style="padding-top:22px; border-top:1px solid rgba(255,255,255,0.06);
            display:flex; justify-content:space-between; align-items:flex-end; flex-wrap:wrap; gap:18px;">
    <div>
        <div class="nav-brand" style="font-size:22px;"><span class="nav-dot"></span>HCPredict</div>
        <div class="sec-d" style="margin-top:6px;">Understanding people. Through data.</div>
    </div>
    <div class="log-line">● system online</div>
</div>
"""

GUIDELINES_STEPS = [
    ("01", "Choose sample data or upload your own CSV."),
    ("02", "Process NPI data."),
    ("03", "Process Survey data."),
    ("04", "Continue to the analysis section."),
    ("05", "Explore plots and insights."),
]


def guidelines_drawer():
    rows = "".join(
        f'<div class="g-row"><div class="g-num">{n}</div><div class="g-txt">{t}</div></div>'
        for n, t in GUIDELINES_STEPS
    )
    return f"""
    <a href="?gd=0" target="_self" class="drawer-scrim" aria-label="Close Guidelines"></a>
    <div class="drawer" role="dialog" aria-modal="true" aria-labelledby="guidelines-title">
        <div class="drawer-head">
            <div>
                <span class="eyebrow">Guidelines</span>
                <h3 id="guidelines-title">How this works</h3>
            </div>
            <a href="?gd=0" target="_self" class="drawer-close" aria-label="Close Guidelines">×</a>
        </div>
        {rows}
        <div class="g-tip">
            <span class="eyebrow">Tip</span>
            <div class="g-txt" style="padding-top:8px;">
                Close this guide with the × button, the Guidelines control, or Escape.
            </div>
        </div>
    </div>
    """


def guidelines_keyboard_bridge():
    """Let Escape close the custom drawer without leaving a stale scrim behind.

    Streamlit renders the drawer as HTML, so the key listener lives in a tiny
    same-page component and navigates through the same `gd=0` close path as the
    visible X and scrim. The listener is removed when the component is replaced
    on the next rerun.
    """
    components.html(
        """
        <script>
        (() => {
            try {
                const parentWindow = window.parent;
                const parentDocument = parentWindow.document;
                const closeGuidelines = (event) => {
                    if (event.key !== "Escape") return;
                    event.preventDefault();
                    const url = new URL(parentWindow.location.href);
                    url.searchParams.set("gd", "0");
                    parentWindow.location.assign(url.toString());
                };
                parentDocument.addEventListener("keydown", closeGuidelines);
                window.addEventListener("unload", () => {
                    parentDocument.removeEventListener("keydown", closeGuidelines);
                });
            } catch (error) {
                // Cross-origin component sandboxes may not expose the parent
                // document; the visible X and scrim remain fully functional.
            }
        })();
        </script>
        """,
        height=0,
        scrolling=False,
    )


def result_panel(verdict, sub, pct):
    """Premium result panel with an animated ring. pct is a measured percentage, not a mock."""
    circumference = 2 * 3.14159 * 76
    dash = circumference * (pct / 100.0)
    return f"""
    <div class="card rise" style="padding:38px 40px;">
      <div class="result">
        <div>
          <span class="eyebrow">Prediction</span>
          <div class="result-verdict" style="margin-top:12px;">{verdict}</div>
          <div class="result-sub">{sub}</div>
        </div>
        <div class="ring-wrap">
          <svg width="170" height="170" viewBox="0 0 170 170">
            <defs>
              <linearGradient id="ringgrad" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0%" stop-color="#FF4F87"/>
                <stop offset="60%" stop-color="#FF8A65"/>
                <stop offset="100%" stop-color="#A66CFF"/>
              </linearGradient>
            </defs>
            <circle class="ring-bg" cx="85" cy="85" r="76"></circle>
            <circle class="ring-fg" cx="85" cy="85" r="76"
                    style="--dash-full:{circumference:.1f};
                           stroke-dasharray:{dash:.1f} {circumference:.1f};"></circle>
          </svg>
          <div class="ring-num"><b>{pct:.0f}%</b><small>Participation</small></div>
        </div>
      </div>
    </div>
    """


def chart_card(title, description=""):
    desc = f'<div class="card-d">{description}</div>' if description else '<div style="height:10px;"></div>'
    return f'<div class="card"><div class="card-h">{title}</div>{desc}'


def main():
    # Initialize the database
    init_db()

    # ---- session state ----
    for key, default in [
        ('initial_load', True), ('show_gif', False), ('show_transition', False),
        ('show_analysis', False), ('slideshow_completed', False), ('npi_file', None),
        ('survey_file', None), ('npi_df', None), ('survey_df', None),
        ('rf_model', None), ('model_accuracy', None), ('show_guidelines', False),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # ---- guidelines: close via query-param ----
    # The scrim and × button are regular same-page links. They trigger a clean
    # Streamlit rerun, and this branch removes only the drawer flag before the
    # rest of the page is rendered. This prevents a stale fixed scrim from
    # surviving in the DOM and blocking pointer events after close.
    try:
        if st.query_params.get("gd") == "0":
            st.session_state.show_guidelines = False
            del st.query_params["gd"]
    except Exception:
        pass

    # ---- opening sequence (~3s) ----
    if st.session_state.initial_load:
        loading_placeholder = st.empty()
        with loading_placeholder:
            st.markdown(OPENING_ANIMATION, unsafe_allow_html=True)
            time.sleep(3.2)
        loading_placeholder.empty()
        st.session_state.initial_load = False

    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

    # ---- slim navigation rail ----
    with st.sidebar:
        st.markdown(
            '<div class="nav-brand" style="font-size:20px; margin-bottom:22px;">'
            '<span class="nav-dot"></span>HCPredict</div>'
            '<div class="rail-item on">◆ &nbsp;Home</div>'
            '<div class="rail-item">◇ &nbsp;Predict</div>'
            '<div class="rail-item">◇ &nbsp;Explore</div>'
            '<div class="rail-item">◇ &nbsp;Insights</div>'
            '<div class="rail-item">◇ &nbsp;About</div>'
            '<div style="margin-top:30px;"><span class="eyebrow">System</span>'
            '<div class="log-line" style="margin-top:8px;">● online</div></div>',
            unsafe_allow_html=True
        )

    # ---- header: nav + guidelines toggle ----
    # The anchor below gives the following row (nav + toggle button) its own stacking
    # context above the drawer/scrim (see CSS "#topbar-anchor + div"), so the toggle
    # button stays clickable even while the drawer is open — a second, always-working
    # way to close Guidelines in addition to the × and click-outside.
    st.markdown('<div id="topbar-anchor"></div>', unsafe_allow_html=True)
    nav_col, g_col = st.columns([5, 1])
    with nav_col:
        st.markdown(NAV_BAR, unsafe_allow_html=True)
    with g_col:
        g_label = "Guidelines ×" if st.session_state.show_guidelines else "Guidelines"
        if st.button(g_label, key="guidelines_toggle", use_container_width=True):
            st.session_state.show_guidelines = not st.session_state.show_guidelines
            st.rerun()

    if st.session_state.show_guidelines:
        guidelines_keyboard_bridge()
        st.markdown(guidelines_drawer(), unsafe_allow_html=True)

    # Check if data is already in the database
    data_status = check_data_status()

    # =======================================================================
    # LANDING — hero, then upload
    # =======================================================================
    if not st.session_state.show_analysis and not st.session_state.show_transition:
        hero_left, hero_right = st.columns([1.05, 1], gap="large")

        with hero_left:
            st.markdown("""
            <div style="padding-top:38px;">
                <span class="eyebrow rise">Human behavior. Data intelligence.</span>
                <div class="hero-h rise-2">Smarter<br><em>Healthcare</em><br>Connections</div>
                <p class="lede rise-3" style="margin-top:24px;">
                    AI-powered predictions to understand HCP availability and engagement.
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
            if st.button("Try prediction", key="hero_cta", type="primary"):
                st.session_state.show_guidelines = True
                st.rerun()

        with hero_right:
            st.markdown(HERO_ORGANISM, unsafe_allow_html=True)

        st.markdown('<div class="spacer-xl"></div>', unsafe_allow_html=True)

        st.markdown(section_header(
            "Data Input",
            "Data Input",
            "Use the sample data to try HCPredict immediately, or upload your own NPI and Survey CSVs."
        ), unsafe_allow_html=True)

        top_l, top_r = st.columns([5, 1])
        with top_r:
            if st.button("Reset database", key="reset_db", use_container_width=True):
                conn = sqlite3.connect('npi_survey_data.db')
                c = conn.cursor()
                c.execute("DROP TABLE IF EXISTS data_status")
                c.execute("DROP TABLE IF EXISTS csv_data")
                conn.commit()
                conn.close()
                init_db()  # Recreate tables
                st.session_state.npi_df = None
                st.session_state.survey_df = None
                st.success("Database reset successfully!")
                st.session_state.show_analysis = False
                st.session_state.show_transition = False
                st.rerun()

        st.markdown('<div class="spacer-md"></div>', unsafe_allow_html=True)

        # The exact sample bytes already used for the download buttons — reused as-is
        # for the direct "Use sample data" path below, so it's the same real sample
        # data either way, never fabricated on the fly.
        nsample_data = b"""NPI,login_date,login_hour,login_minute,logout_date,logout_hour,logout_minute,Region_Midwest,Region_Northeast,Region_South,Region_West,Speciality_Cardiology,Speciality_General Practice,Speciality_Neurology,Speciality_Oncology,Speciality_Orthopedics,Speciality_Pediatrics,Speciality_Radiology,State_TX,State_CA,Count of Survey Attempts,Usage Time (mins)
1234567890,2024-01-10,8,30,2024-01-10,10,0,1,0,0,0,0,1,0,0,0,0,0,1,0,5,90
1234567891,2024-01-11,9,0,2024-01-11,11,30,0,1,0,0,0,1,0,0,0,0,1,0,0,3,120
1234567892,2024-01-12,10,0,2024-01-12,12,0,0,0,1,0,0,1,0,0,0,0,0,1,0,2,110
1234567893,2024-01-13,14,0,2024-01-13,16,30,0,0,0,1,1,0,0,0,0,0,1,0,0,4,150
"""
        ssample_data = b"""Survey ID,NPI,attempt_hour,attempt_minute
100010,1234567890,9,0
100010,1234567891,10,30
100010,1234567892,11,0
100010,1234567893,15,0
"""

        npi_col, survey_col = st.columns(2, gap="large")

        # ---------------- NPI ----------------
        with npi_col:
            st.markdown('<div class="data-card rise">', unsafe_allow_html=True)
            st.markdown(
                '<span class="eyebrow">Source 01</span>'
                '<div class="data-card-h">NPI Data</div>'
                '<div class="sec-d">Login windows, usage time, region and specialty signals.</div>',
                unsafe_allow_html=True
            )

            if data_status['npi']['uploaded']:
                st.markdown(
                    f"<div class='status-pill'>● uploaded · {data_status['npi']['last_updated']}</div>",
                    unsafe_allow_html=True
                )
                if st.button("Clear NPI data", key="clear_npi", use_container_width=True):
                    clear_data('npi')
                    st.session_state.npi_df = None
                    st.success("NPI data cleared successfully!")
                    st.session_state.show_analysis = False
                    st.rerun()
            else:
                if st.button("Use sample NPI data", key="use_sample_npi",
                             type="primary", use_container_width=True):
                    with st.spinner("Loading sample signals..."):
                        try:
                            sample_df = pd.read_csv(io.BytesIO(nsample_data))
                            st.session_state.npi_df = preprocess_npi_data(sample_df)
                            if store_csv_data('npi', BytesIO(nsample_data), skip_validation=True):
                                st.success("Sample NPI data loaded!")
                                new_status = check_data_status()
                                if new_status['survey']['uploaded']:
                                    st.session_state.show_transition = True
                                st.rerun()
                        except Exception as e:
                            st.error(f"Error loading sample NPI data: {str(e)}")

                st.markdown('<div class="or-divider"><span>or</span></div>', unsafe_allow_html=True)
                st.markdown('<div class="upload-label">Upload your own NPI CSV</div>', unsafe_allow_html=True)

                npi_file = st.file_uploader("Drag and drop file here", type=['csv'], key="npi_uploader",
                                            label_visibility="collapsed")
                st.session_state.npi_file = npi_file
                if st.session_state.npi_file is not None:
                    if st.button("Save NPI data", key="save_npi", use_container_width=True):
                        with st.spinner("Reading signals..."):
                            try:
                                npi_df = pd.read_csv(st.session_state.npi_file)
                                if npi_df.empty:
                                    st.error("The uploaded NPI CSV file is empty.")
                                else:
                                    st.session_state.npi_df = preprocess_npi_data(npi_df)
                                    st.session_state.npi_file.seek(0)
                                    # already parsed successfully above, so skip the
                                    # redundant re-parse inside store_csv_data
                                    if store_csv_data('npi', st.session_state.npi_file, skip_validation=True):
                                        st.success("NPI data uploaded successfully!")
                                        new_status = check_data_status()
                                        if new_status['survey']['uploaded']:
                                            st.session_state.show_transition = True
                                        st.rerun()
                            except Exception as e:
                                st.error(f"Error processing NPI CSV: {str(e)}")

                st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)
                st.download_button(
                    label="↓ Download sample NPI file",
                    data=BytesIO(nsample_data),
                    file_name="npi2_sample_4_rows.csv",
                    mime="text/csv",
                    key="npi_sample_download",
                    use_container_width=True
                )
            st.markdown('</div>', unsafe_allow_html=True)

        # ---------------- SURVEY ----------------
        with survey_col:
            st.markdown('<div class="data-card rise-2">', unsafe_allow_html=True)
            st.markdown(
                '<span class="eyebrow">Source 02</span>'
                '<div class="data-card-h">Survey Data</div>'
                '<div class="sec-d">Recorded survey attempts, by NPI and time of attempt.</div>',
                unsafe_allow_html=True
            )

            if data_status['survey']['uploaded']:
                st.markdown(
                    f"<div class='status-pill'>● uploaded · {data_status['survey']['last_updated']}</div>",
                    unsafe_allow_html=True
                )
                if st.button("Clear survey data", key="clear_survey", use_container_width=True):
                    clear_data('survey')
                    st.session_state.survey_df = None
                    st.success("Survey data cleared successfully!")
                    st.session_state.show_analysis = False
                    st.rerun()
            else:
                if st.button("Use sample survey data", key="use_sample_survey",
                             type="primary", use_container_width=True):
                    with st.spinner("Loading sample signals..."):
                        try:
                            sample_df = pd.read_csv(io.BytesIO(ssample_data))
                            if sample_df.empty:
                                st.error("Sample survey data is empty.")
                            else:
                                st.session_state.survey_df = sample_df
                                if store_csv_data('survey', BytesIO(ssample_data), skip_validation=True):
                                    st.success("Sample survey data loaded!")
                                    new_status = check_data_status()
                                    if new_status['npi']['uploaded']:
                                        st.session_state.show_transition = True
                                    st.rerun()
                        except Exception as e:
                            st.error(f"Error loading sample survey data: {str(e)}")

                st.markdown('<div class="or-divider"><span>or</span></div>', unsafe_allow_html=True)
                st.markdown('<div class="upload-label">Upload your own Survey CSV</div>', unsafe_allow_html=True)

                survey_file = st.file_uploader("Drag and drop file here", type=['csv'], key="survey_uploader",
                                               label_visibility="collapsed")
                st.session_state.survey_file = survey_file
                if st.session_state.survey_file is not None:
                    if st.button("Save survey data", key="save_survey", use_container_width=True):
                        with st.spinner("Reading signals..."):
                            try:
                                survey_df = pd.read_csv(st.session_state.survey_file)
                                if survey_df.empty:
                                    st.error("The uploaded Survey CSV file is empty.")
                                else:
                                    st.session_state.survey_df = survey_df
                                    st.session_state.survey_file.seek(0)
                                    if store_csv_data('survey', st.session_state.survey_file, skip_validation=True):
                                        st.success("Survey data uploaded successfully!")
                                        new_status = check_data_status()
                                        if new_status['npi']['uploaded']:
                                            st.session_state.show_transition = True
                                        st.rerun()
                            except Exception as e:
                                st.error(f"Error processing Survey CSV: {str(e)}")

                st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)
                st.download_button(
                    label="↓ Download sample survey file",
                    data=BytesIO(ssample_data),
                    file_name="survey2_first_4_rows.csv",
                    mime="text/csv",
                    key="survey_sample_download",
                    use_container_width=True
                )
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="spacer-lg"></div>', unsafe_allow_html=True)
        st.markdown(FOOTER, unsafe_allow_html=True)

    # =======================================================================
    # TRANSITION
    # =======================================================================
    # NOTE: this used to also require st.session_state.npi_df / survey_df to already be
    # populated in memory. That silently broke the flow whenever a data type had been
    # uploaded in an earlier session (status flag already true in the DB) but this
    # session's in-memory cache had never been (re)filled — the "Survey data is not
    # saving" bug. The analysis section below always reloads both from the DB directly,
    # so gating on the session-only copies here was unnecessary and wrong.
    if st.session_state.show_transition:
        transition_container = st.empty()

        transition_container.markdown("""
        <style>
        .flow { position:fixed; inset:0; z-index:1000;
            background:radial-gradient(900px 640px at 50% 50%, #12061C 0%, #05040A 74%);
            display:flex; flex-direction:column; justify-content:center; align-items:center; gap:28px; }
        .flow svg { width:min(680px,86vw); }
        .flow path { fill:none; stroke:url(#fg); stroke-width:1.5; opacity:0.85;
            stroke-dasharray:520; stroke-dashoffset:520; animation:draw 1.5s ease-out forwards; }
        .flow path:nth-child(2){animation-delay:.14s;} .flow path:nth-child(3){animation-delay:.28s;}
        .flow path:nth-child(4){animation-delay:.42s;}
        @keyframes draw { to{stroke-dashoffset:0;} }
        .flow-core { width:88px; height:88px; border-radius:50%;
            background:radial-gradient(circle at 40% 40%, #FFB08A, #FF4F87 46%, rgba(139,77,255,0) 72%);
            box-shadow:0 0 74px rgba(255,79,135,0.5); animation:pulse 2.4s ease-in-out infinite; }
        @keyframes pulse { 0%,100%{transform:scale(0.92);} 50%{transform:scale(1.12);} }
        .flow-t { font-family:'Inter',sans-serif; font-size:10px; letter-spacing:5px;
                  text-transform:uppercase; color:#B9AAB8; }
        .flow-d { font-family:'Cormorant Garamond',serif; font-size:34px; color:#F7EAF2; opacity:0;
                  animation:fu .9s ease-out 3.1s forwards; }
        @keyframes fu { from{opacity:0;transform:translateY(14px);} to{opacity:1;transform:translateY(0);} }
        </style>
        <div class="flow">
            <svg viewBox="0 0 680 200">
                <defs><linearGradient id="fg" x1="0" y1="0" x2="1" y2="0">
                    <stop offset="0%" stop-color="#8B4DFF"/><stop offset="58%" stop-color="#FF4F87"/>
                    <stop offset="100%" stop-color="#FF8A65"/></linearGradient></defs>
                <path d="M20,30 C220,30 260,100 340,100"/>
                <path d="M20,80 C220,80 270,100 340,100"/>
                <path d="M20,130 C220,130 270,100 340,100"/>
                <path d="M20,180 C220,180 260,100 340,100"/>
            </svg>
            <div class="flow-core"></div>
            <div class="flow-t">Analyzing signals</div>
            <div class="flow-d">Prediction surface ready</div>
        </div>
        """, unsafe_allow_html=True)

        time.sleep(4.2)

        st.session_state.show_transition = False
        st.session_state.show_analysis = True
        transition_container.markdown(
            '<div style="position:fixed; inset:0; background:#05040A; z-index:1000;"></div>',
            unsafe_allow_html=True
        )
        time.sleep(0.15)
        transition_container.empty()
        st.rerun()

    # =======================================================================
    # ANALYSIS — metrics, prediction, result, analytics, footer
    # =======================================================================
    if st.session_state.show_analysis:
        with st.spinner("Loading signals..."):
            processing_placeholder = st.empty()
            processing_placeholder.empty()

            npi_df = load_csv_data('npi')
            survey_df = load_csv_data('survey')

            if npi_df is None or survey_df is None:
                st.error("Failed to load data. Please upload valid NPI and Survey CSV files.")
                st.session_state.show_analysis = False
                st.session_state.show_slideshow = False
                st.rerun()
                return

            npi_df = preprocess_npi_data(npi_df)

        # Train the Random Forest model. train_rf_model is cached on the actual data
        # (npi_df, survey_df) rather than on a closure, so a reset + re-upload of new
        # data correctly retrains instead of silently reusing a stale model.
        rf_model = train_rf_model(npi_df, survey_df)

        # ---------------- METRICS (measured, not mocked) ----------------
        n_records = len(npi_df)
        n_specialties = len([c for c in npi_df.columns if c.startswith('Speciality_')])
        n_regions = len([c for c in npi_df.columns if c.startswith('Region_')])
        acc = st.session_state.get('model_accuracy')
        acc_txt = f"{acc * 100:.0f}%" if acc is not None else "—"

        metrics_html = '<div class="metrics-grid">' + "".join(
            f'<div class="metric rise"><div class="ico">{icon}</div>'
            f'<div class="val">{val}</div><div class="key">{key}</div></div>'
            for icon, val, key in [
                ("◇", f"{n_records:,}", "HCP Records"),
                ("◈", f"{n_specialties}", "Specialties"),
                ("◉", f"{n_regions}", "Regions"),
                ("◍", acc_txt, "Model Accuracy"),
            ]
        ) + '</div>'
        st.markdown(metrics_html, unsafe_allow_html=True)

        st.markdown('<div class="spacer-lg"></div>', unsafe_allow_html=True)

        # ---------------- PREDICTION INSTRUMENT ----------------
        st.markdown(section_header(
            "Input Signals",
            "Input Signals",
            "Provide the behavioral signals and let the model analyze."
        ), unsafe_allow_html=True)

        st.markdown('<div class="instrument rise-2">', unsafe_allow_html=True)
        i1, i2, i3 = st.columns([1, 1, 1], gap="large")
        with i1:
            survey_id = st.number_input("Survey ID", min_value=100000, max_value=999999,
                                        value=100010, key="survey_id")
        with i2:
            time_str = st.text_input("Analysis time (HH:MM)", value="00:30", key="time_input")
        with i3:
            st.markdown("<div style='height:26px;'></div>", unsafe_allow_html=True)
            run_button = st.button("Predict now", key="run_button", type="primary",
                                   help="Run the analysis")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="spacer-md"></div>', unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["Survey analysis", "NPI distribution", "Time patterns"])

        with tab1:
            if 'run_analysis_triggered' in st.session_state and st.session_state.run_analysis_triggered:
                anim = st.empty()
                anim.markdown(PREDICT_ANIMATION, unsafe_allow_html=True)
                result = analyze_survey_participation(survey_id, time_str, survey_df, npi_df, rf_model)
                time.sleep(1.2)
                anim.empty()
                st.session_state.run_analysis_triggered = False

                if isinstance(result, str):
                    st.error(result)
                else:
                    pct = result['Participation Percentage']
                    active_n = result['Active NPIs at Analysis Time']
                    if pct >= 60:
                        verdict = "High activity"
                    elif pct >= 25:
                        verdict = "Moderate activity"
                    else:
                        verdict = "Low activity"
                    st.markdown(
                        result_panel(
                            verdict,
                            f"{active_n} NPIs are inside their active window at {time_str}. "
                            f"{result['Survey Participants Among Active NPIs']} of them took survey "
                            f"{result['Survey ID']}.",
                            pct
                        ),
                        unsafe_allow_html=True
                    )

                    if result['Active NPIs with Participation Probability']:
                        fig_region, fig_state, fig_specialty, fig_region_part, fig_specialty_part = create_visualizations(
                            result['Active NPIs with Participation Probability']
                        )

                        # primary chart — full width
                        st.markdown(chart_card("Active NPIs by Region",
                                               "Where the reachable audience sits at this moment."),
                                    unsafe_allow_html=True)
                        st.plotly_chart(fig_region, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                        c1, c2 = st.columns(2, gap="large")
                        with c1:
                            st.markdown(chart_card("By specialty"), unsafe_allow_html=True)
                            st.plotly_chart(fig_specialty, use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        with c2:
                            st.markdown(chart_card("Participation rate by region"), unsafe_allow_html=True)
                            st.plotly_chart(fig_region_part, use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)

                        c3, c4 = st.columns(2, gap="large")
                        with c3:
                            st.markdown(chart_card("Top 15 states"), unsafe_allow_html=True)
                            st.plotly_chart(fig_state, use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        with c4:
                            st.markdown(chart_card("Participation rate by specialty"), unsafe_allow_html=True)
                            st.plotly_chart(fig_specialty_part, use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)

                    results_df = pd.DataFrame(result['Active NPIs with Participation Probability'])
                    st.markdown('<div class="spacer-md"></div>', unsafe_allow_html=True)
                    st.markdown(section_header("Detail", "Active NPIs",
                                               "Every NPI inside its active window, ranked by modelled probability."),
                                unsafe_allow_html=True)
                    st.dataframe(results_df, use_container_width=True)

                    csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download results as CSV",
                        data=csv,
                        file_name=f"survey_{survey_id}_analysis_{time_str.replace(':', '')}.csv",
                        mime='text/csv',
                        use_container_width=True
                    )
            else:
                st.session_state.run_analysis_triggered = False
                if run_button:
                    st.session_state.run_analysis_triggered = True
                    st.rerun()

        with tab2:
            st.markdown(section_header("Explore", "NPI Distribution",
                                       "The full population, before any time filter is applied."),
                        unsafe_allow_html=True)

            region_data = {}
            state_data = {}
            specialty_data = {}
            region_specialty_counts = {}

            for _, row in npi_df.iterrows():
                region = "Unknown"
                for region_name in ['Midwest', 'Northeast', 'South', 'West']:
                    if row.get(f'Region_{region_name}', 0) == 1:
                        region = region_name
                        break
                region_data[region] = region_data.get(region, 0) + 1

                state = "Unknown"
                state_columns = [col for col in row.index if col.startswith('State_')]
                for state_col in state_columns:
                    if row[state_col] == 1:
                        state = state_col.replace('State_', '')
                        break
                state_data[state] = state_data.get(state, 0) + 1

                specialty = "Unknown"
                specialty_columns = [col for col in row.index if col.startswith('Speciality_')]
                for specialty_col in specialty_columns:
                    if row[specialty_col] == 1:
                        specialty = specialty_col.replace('Speciality_', '')
                        break
                specialty_data[specialty] = specialty_data.get(specialty, 0) + 1

                key = (region, specialty)
                region_specialty_counts[key] = region_specialty_counts.get(key, 0) + 1

            region_df = pd.DataFrame([{'Region': k, 'Count': v} for k, v in region_data.items()])
            state_df = pd.DataFrame([{'State': k, 'Count': v} for k, v in state_data.items()]).sort_values('Count', ascending=False).head(15)
            specialty_df = pd.DataFrame([{'Specialty': k, 'Count': v} for k, v in specialty_data.items()])

            region_specialty_data = []
            for (region, specialty), count in region_specialty_counts.items():
                region_specialty_data.append({'Region': region, 'Specialty': specialty, 'Count': count})
            region_specialty_df = pd.DataFrame(region_specialty_data)
            pivot_df = region_specialty_df.pivot_table(values='Count', index='Specialty', columns='Region', fill_value=0)

            st.markdown(chart_card("NPI count by region and specialty",
                                   "Density across the two strongest grouping signals."),
                        unsafe_allow_html=True)
            fig_heatmap = px.imshow(pivot_df, title='',
                                    labels=dict(x="Region", y="Specialty", color="NPI count"),
                                    color_continuous_scale=BIO_SCALE, height=460)
            st.plotly_chart(style_fig(fig_heatmap), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            c1, c2 = st.columns(2, gap="large")
            with c1:
                st.markdown(chart_card("NPIs by region"), unsafe_allow_html=True)
                fig_region_all = px.bar(region_df, x='Region', y='Count', title='', color='Region',
                                        color_discrete_sequence=PLOT_COLORWAY, height=340)
                st.plotly_chart(style_fig(fig_region_all), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with c2:
                st.markdown(chart_card("NPIs by specialty"), unsafe_allow_html=True)
                fig_specialty_all = px.pie(specialty_df, values='Count', names='Specialty', title='',
                                           color_discrete_sequence=PLOT_COLORWAY, hole=0.62, height=340)
                fig_specialty_all.update_traces(marker=dict(line=dict(color=INK_1, width=2)))
                st.plotly_chart(style_fig(fig_specialty_all), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown(chart_card("Top 15 states"), unsafe_allow_html=True)
            fig_state_all = px.bar(state_df, x='State', y='Count', title='', color='State',
                                   color_discrete_sequence=PLOT_COLORWAY, height=380)
            st.plotly_chart(style_fig(fig_state_all), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with tab3:
            st.markdown(section_header("Insights", "NPI Activity by Time",
                                       "When the audience is reachable across a 24-hour cycle."),
                        unsafe_allow_html=True)

            time_counts = analyze_active_npis_by_time(npi_df)
            time_df = pd.DataFrame([{'Time': k, 'Active NPIs': v} for k, v in time_counts.items()])

            hour_counts = {}
            for time_key, count in time_counts.items():
                hour = int(time_key.split(':')[0])
                hour_str = f"{hour:02d}:00"
                hour_counts[hour_str] = hour_counts.get(hour_str, 0) + count / 2
            hour_df = pd.DataFrame([{'Hour': k, 'Active NPIs': v} for k, v in hour_counts.items()])
            hour_matrix = np.zeros((7, 24))
            for i in range(7):
                for j, (_, row) in enumerate(hour_df.iterrows()):
                    hour_matrix[i, j] = row['Active NPIs']
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            hours = [f"{h:02d}:00" for h in range(24)]

            login_hour_counts = npi_df['login_hour'].value_counts().reset_index()
            login_hour_counts.columns = ['Hour', 'Count']
            login_hour_counts = login_hour_counts.sort_values('Hour')
            logout_hour_counts = npi_df['logout_hour'].value_counts().reset_index()
            logout_hour_counts.columns = ['Hour', 'Count']
            logout_hour_counts = logout_hour_counts.sort_values('Hour')

            st.markdown(chart_card("Active NPIs throughout the day",
                                   "Half-hour resolution across the full cycle."),
                        unsafe_allow_html=True)
            fig_time = px.line(time_df, x='Time', y='Active NPIs', title='', markers=True, height=460,
                               color_discrete_sequence=[CORAL])
            fig_time.update_traces(line=dict(width=2.4), marker=dict(size=5, color=ORANGE_3))
            st.plotly_chart(style_fig(fig_time), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown(chart_card("Active NPIs by hour and day"), unsafe_allow_html=True)
            fig_heatmap2 = px.imshow(hour_matrix, labels=dict(x="Hour of day", y="Day of week", color="Active NPIs"),
                                     x=hours, y=days, title='', color_continuous_scale=BIO_SCALE, height=360)
            st.plotly_chart(style_fig(fig_heatmap2), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            c1, c2 = st.columns(2, gap="large")
            with c1:
                st.markdown(chart_card("Login hours"), unsafe_allow_html=True)
                fig_login = px.bar(login_hour_counts, x='Hour', y='Count', title='', height=320,
                                   color_discrete_sequence=[CORAL])
                st.plotly_chart(style_fig(fig_login), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with c2:
                st.markdown(chart_card("Logout hours"), unsafe_allow_html=True)
                fig_logout = px.bar(logout_hour_counts, x='Hour', y='Count', title='', height=320,
                                    color_discrete_sequence=[PURPLE_2])
                st.plotly_chart(style_fig(fig_logout), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="spacer-lg"></div>', unsafe_allow_html=True)
        st.markdown(FOOTER, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
