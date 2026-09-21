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

st.set_page_config(layout="wide", page_title="HCPredict", page_icon="🧬")


# ---------------------------------------------------------------------------
# THEME CONSTANTS — bioluminescent intelligence
# ---------------------------------------------------------------------------
INK_0 = "#08070D"
INK_1 = "#0D0912"
INK_2 = "#120B16"
CORAL = "#FF4F87"
CORAL_SOFT = "#FF6B9D"
PEACH = "#FF9B7A"
PEACH_SOFT = "#FFB38A"
PURPLE = "#8B4DFF"
PURPLE_SOFT = "#B06CFF"
TEXT = "#EADCF2"
TEXT_DIM = "#A493B4"

PLOT_COLORWAY = [CORAL, PEACH, PURPLE_SOFT, PEACH_SOFT, PURPLE, CORAL_SOFT, "#E8C9FF"]
BIO_SCALE = [[0.0, INK_2], [0.35, "#4B2A6B"], [0.7, PURPLE_SOFT], [1.0, PEACH]]


def style_fig(fig):
    """Apply the bioluminescent theme to any plotly figure."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=PLOT_COLORWAY,
        font=dict(family="DM Sans, sans-serif", color=TEXT, size=13),
        title_font=dict(family="Playfair Display, serif", color=TEXT, size=18),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_DIM)),
        margin=dict(l=40, r=25, t=45, b=40),
        hoverlabel=dict(bgcolor=INK_2, font=dict(color=TEXT), bordercolor=CORAL),
    )
    fig.update_xaxes(gridcolor="rgba(176,108,255,0.14)", zerolinecolor="rgba(176,108,255,0.2)",
                     linecolor="rgba(234,220,242,0.2)", tickfont=dict(color=TEXT_DIM))
    fig.update_yaxes(gridcolor="rgba(176,108,255,0.14)", zerolinecolor="rgba(176,108,255,0.2)",
                     linecolor="rgba(234,220,242,0.2)", tickfont=dict(color=TEXT_DIM))
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


# Database setup functions
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


def store_csv_data(data_type, csv_file):
    """Store uploaded CSV data in the database"""
    conn = sqlite3.connect('npi_survey_data.db')
    c = conn.cursor()

    # Read and validate CSV content
    csv_content = csv_file.read()
    if not csv_content:
        conn.close()
        st.error(f"The uploaded {data_type} CSV file is empty.")
        return False

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
        f"<div class='signal-log'>signal stored · {data_type} · {len(csv_content)} bytes</div>",
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
def preprocess_npi_data(npi_df):
    npi_df[['login_mins', 'logout_mins', 'calculated_active_time']] = npi_df.apply(
        lambda row: pd.Series(calculate_active_window(row)), axis=1
    )

    npi_df['usage_time_valid'] = npi_df.apply(
        lambda row: abs(row['calculated_active_time'] - row['Usage Time (mins)']) <= 5, axis=1
    )

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
    # kept as a value so the stat strip can show the real number instead of a hardcoded one
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


# Function to analyze active NPIs at different times
def analyze_active_npis_by_time(npi_df):
    time_counts = {}
    for hour in range(24):
        for minute in [0, 30]:
            target_time = (hour, minute)
            time_str = f"{hour:02d}:{minute:02d}"

            active_count = sum(1 for _, row in npi_df.iterrows() if is_active_in_timeslot(row, target_time))
            time_counts[time_str] = active_count

    return time_counts


# Create visualizations for region, state, and specialty distributions
def create_visualizations(active_npi_data):
    df = pd.DataFrame(active_npi_data)

    region_counts = df['Region'].value_counts().reset_index()
    region_counts.columns = ['Region', 'Count']

    fig_region = px.bar(
        region_counts,
        x='Region',
        y='Count',
        title='Active NPIs by Region',
        color='Region',
        color_discrete_sequence=PLOT_COLORWAY,
        labels={'Count': 'Number of Active NPIs'},
        height=400
    )

    state_counts = df['State'].value_counts().reset_index()
    state_counts.columns = ['State', 'Count']
    state_counts = state_counts.head(15)

    fig_state = px.bar(
        state_counts,
        x='State',
        y='Count',
        title='Active NPIs by State (Top 15)',
        color='State',
        color_discrete_sequence=PLOT_COLORWAY,
        labels={'Count': 'Number of Active NPIs'},
        height=500
    )

    specialty_counts = df['Specialty'].value_counts().reset_index()
    specialty_counts.columns = ['Specialty', 'Count']

    fig_specialty = px.pie(
        specialty_counts,
        values='Count',
        names='Specialty',
        title='Active NPIs by Specialty',
        color_discrete_sequence=PLOT_COLORWAY,
        hole=0.55,
        height=400
    )
    fig_specialty.update_traces(marker=dict(line=dict(color=INK_1, width=2)))

    region_participation = df.groupby('Region')['Participated'].agg(['sum', 'count']).reset_index()
    region_participation['Participation Rate'] = (region_participation['sum'] / region_participation['count'] * 100).round(2)
    region_participation.columns = ['Region', 'Participants', 'Total', 'Participation Rate (%)']

    fig_region_participation = px.bar(
        region_participation,
        x='Region',
        y='Participation Rate (%)',
        title='Participation Rate by Region',
        color='Region',
        color_discrete_sequence=PLOT_COLORWAY,
        height=400
    )

    specialty_participation = df.groupby('Specialty')['Participated'].agg(['sum', 'count']).reset_index()
    specialty_participation['Participation Rate'] = (specialty_participation['sum'] / specialty_participation['count'] * 100).round(2)
    specialty_participation.columns = ['Specialty', 'Participants', 'Total', 'Participation Rate (%)']

    fig_specialty_participation = px.bar(
        specialty_participation,
        x='Specialty',
        y='Participation Rate (%)',
        title='Participation Rate by Specialty',
        color='Specialty',
        color_discrete_sequence=PLOT_COLORWAY,
        height=400
    )

    for f in (fig_region, fig_state, fig_specialty, fig_region_participation, fig_specialty_participation):
        style_fig(f)

    return fig_region, fig_state, fig_specialty, fig_region_participation, fig_specialty_participation


# ---------------------------------------------------------------------------
# THEME MARKUP HELPERS
# ---------------------------------------------------------------------------
GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,500;0,600;1,500&family=DM+Sans:wght@300;400;500;700&family=DM+Mono:wght@300;400&display=swap');

:root {
    --ink-0:#08070D;
    --ink-1:#0D0912;
    --ink-2:#120B16;
    --coral:#FF4F87;
    --coral-soft:#FF6B9D;
    --peach:#FF9B7A;
    --peach-soft:#FFB38A;
    --purple:#8B4DFF;
    --purple-soft:#B06CFF;
    --text:#EADCF2;
    --text-dim:#A493B4;
}

/* ---- base surface ---- */
.stApp {
    background:
        radial-gradient(1100px 620px at 12% -8%, rgba(255,79,135,0.16), transparent 62%),
        radial-gradient(900px 560px at 92% 8%, rgba(139,77,255,0.18), transparent 60%),
        radial-gradient(1000px 700px at 50% 108%, rgba(255,155,122,0.10), transparent 62%),
        linear-gradient(180deg, #08070D 0%, #0D0912 45%, #120B16 100%);
    background-attachment: fixed;
}
.stApp [data-testid="stAppViewContainer"] { background: transparent !important; }
[data-testid="stHeader"] { background: transparent !important; }

html, body, [class*="css"], p, li, span, label, div[data-testid="stMarkdownContainer"] {
    font-family:'DM Sans', sans-serif;
    color: var(--text);
}

/* ---- living background: slow drifting bioluminescent forms ---- */
.bio-field { position:fixed; inset:0; z-index:0; pointer-events:none; overflow:hidden; }
.bio-blob {
    position:absolute; border-radius:50%;
    filter: blur(70px);
    opacity:0.5;
    mix-blend-mode: screen;
}
.b1 { width:520px; height:520px; top:-8%; left:-6%;
      background: radial-gradient(circle at 35% 35%, rgba(255,79,135,0.55), transparent 68%);
      animation: drift1 34s ease-in-out infinite; }
.b2 { width:460px; height:460px; top:32%; right:-8%;
      background: radial-gradient(circle at 60% 40%, rgba(139,77,255,0.50), transparent 68%);
      animation: drift2 44s ease-in-out infinite; }
.b3 { width:400px; height:400px; bottom:-10%; left:28%;
      background: radial-gradient(circle at 50% 50%, rgba(255,155,122,0.42), transparent 70%);
      animation: drift3 52s ease-in-out infinite; }
@keyframes drift1 { 0%,100%{transform:translate(0,0) scale(1);} 50%{transform:translate(90px,70px) scale(1.14);} }
@keyframes drift2 { 0%,100%{transform:translate(0,0) scale(1.05);} 50%{transform:translate(-80px,-60px) scale(0.92);} }
@keyframes drift3 { 0%,100%{transform:translate(0,0) scale(0.95);} 50%{transform:translate(60px,-70px) scale(1.12);} }

.bio-spark { position:absolute; width:3px; height:3px; border-radius:50%;
             background: var(--peach-soft); box-shadow:0 0 10px var(--coral); opacity:0.7; }
.s1{top:22%;left:18%;animation:breathe 9s ease-in-out infinite;}
.s2{top:64%;left:72%;animation:breathe 13s ease-in-out infinite 1.5s;}
.s3{top:41%;left:47%;animation:breathe 11s ease-in-out infinite 3s;}
.s4{top:78%;left:31%;animation:breathe 15s ease-in-out infinite 2s;}
.s5{top:14%;left:81%;animation:breathe 12s ease-in-out infinite 4s;}
@keyframes breathe { 0%,100%{opacity:0.15; transform:scale(0.8);} 50%{opacity:0.95; transform:scale(1.6);} }

/* keep real content above the field */
section.main, section[data-testid="stSidebar"] { position:relative; z-index:1; }

/* ---- typography ---- */
h1 {
    font-family:'Playfair Display', serif !important;
    font-weight:600 !important;
    color: var(--text) !important;
    letter-spacing:0.3px;
}
h2, h3, h4 {
    font-family:'Playfair Display', serif !important;
    font-weight:500 !important;
    color: var(--text) !important;
    letter-spacing:0.2px;
}
.eyebrow {
    font-family:'DM Mono', monospace;
    font-size:12px; letter-spacing:4px; text-transform:uppercase;
    color: var(--coral-soft);
}
.signal-log {
    font-family:'DM Mono', monospace; font-size:11px; letter-spacing:1.5px;
    color: var(--text-dim); text-transform:uppercase; opacity:0.75; margin:6px 0 2px 0;
}

/* ---- floating nav ---- */
.nav-bar {
    display:flex; align-items:center; justify-content:space-between;
    padding:12px 22px; margin:4px 0 26px 0;
    border:1px solid rgba(255,255,255,0.07);
    border-radius:999px;
    background: rgba(18,11,22,0.55);
    backdrop-filter: blur(12px);
}
.nav-brand { font-family:'Playfair Display', serif; font-size:19px; color:var(--text); letter-spacing:0.5px; }
.nav-dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:9px;
           background: var(--coral); box-shadow:0 0 12px var(--coral); animation: breathe 4s ease-in-out infinite; }
.nav-links { display:flex; gap:26px; }
.nav-links span {
    font-size:13px; letter-spacing:1.4px; text-transform:uppercase; color:var(--text-dim);
    position:relative; padding-bottom:4px; transition:color .35s ease;
}
.nav-links span::after {
    content:""; position:absolute; left:0; bottom:0; height:1px; width:0;
    background:linear-gradient(90deg, var(--coral), var(--peach));
    box-shadow:0 0 8px var(--coral); transition:width .45s cubic-bezier(.2,.8,.3,1);
}
.nav-links span:hover { color:var(--text); }
.nav-links span:hover::after { width:100%; }

/* ---- glass cards ---- */
.glass-card, .graph-card {
    background: linear-gradient(160deg, rgba(255,255,255,0.045), rgba(255,255,255,0.015));
    border:1px solid rgba(255,255,255,0.08);
    border-radius:20px;
    padding:20px 22px;
    margin-bottom:22px;
    box-shadow: 0 18px 46px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
    transition: border-color .4s ease, box-shadow .4s ease, transform .4s ease;
}
.glass-card:hover, .graph-card:hover {
    transform: translateY(-3px);
    border-color: rgba(255,79,135,0.35);
    box-shadow: 0 22px 54px rgba(0,0,0,0.55), 0 0 26px rgba(255,79,135,0.12);
}
.glass-card h4, .graph-card h4 { text-align:left; margin:0 0 6px 0; font-size:17px; }

/* ---- stat fragments ---- */
.stat-frag {
    background: linear-gradient(160deg, rgba(255,255,255,0.05), rgba(255,255,255,0.012));
    border:1px solid rgba(255,255,255,0.08);
    border-radius:18px; padding:18px 16px; text-align:left;
    position:relative; overflow:hidden;
}
.stat-frag::after {
    content:""; position:absolute; width:120px; height:120px; right:-50px; top:-50px;
    background: radial-gradient(circle, rgba(255,79,135,0.30), transparent 68%);
    filter: blur(14px); animation: breathe 7s ease-in-out infinite;
}
.stat-val { font-family:'Playfair Display', serif; font-size:34px; color:var(--text); line-height:1.1; }
.stat-key { font-family:'DM Mono', monospace; font-size:11px; letter-spacing:2.4px;
            text-transform:uppercase; color:var(--text-dim); margin-top:6px; }

/* ---- buttons: light sweeps across, no colour jump ---- */
.stButton > button, .stDownloadButton > button {
    font-family:'DM Sans', sans-serif !important;
    font-size:13px !important; letter-spacing:1.6px !important; text-transform:uppercase !important;
    color: var(--text) !important;
    background: linear-gradient(135deg, rgba(255,79,135,0.20), rgba(139,77,255,0.18)) !important;
    border:1px solid rgba(255,79,135,0.45) !important;
    border-radius:999px !important;
    padding:9px 26px !important;
    position:relative; overflow:hidden;
    transition: box-shadow .45s ease, border-color .45s ease !important;
}
.stButton > button::before, .stDownloadButton > button::before {
    content:""; position:absolute; top:0; left:-70%; width:55%; height:100%;
    background: linear-gradient(120deg, rgba(255,255,255,0), rgba(255,179,138,0.38), rgba(255,255,255,0));
    transform: skewX(-18deg); transition:left .75s ease;
}
.stButton > button:hover::before, .stDownloadButton > button:hover::before { left:130%; }
.stButton > button:hover, .stDownloadButton > button:hover {
    border-color: var(--coral) !important;
    box-shadow: 0 0 22px rgba(255,79,135,0.35) !important;
    color:#fff !important;
}
.stButton > button:focus, .stDownloadButton > button:focus { box-shadow:0 0 20px rgba(255,79,135,0.4) !important; }

/* ---- inputs as one instrument ---- */
div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input {
    background: rgba(18,11,22,0.75) !important;
    color: var(--text) !important;
    border:1px solid rgba(176,108,255,0.35) !important;
    border-radius:12px !important;
    font-family:'DM Mono', monospace !important;
    letter-spacing:1px;
    transition: box-shadow .35s ease, border-color .35s ease;
}
div[data-testid="stTextInput"] input:focus,
div[data-testid="stNumberInput"] input:focus {
    border-color: var(--coral) !important;
    box-shadow: 0 0 16px rgba(255,79,135,0.30) !important;
}
div[data-testid="stWidgetLabel"] p {
    font-family:'DM Mono', monospace !important;
    font-size:11px !important; letter-spacing:2.4px !important;
    text-transform:uppercase; color: var(--text-dim) !important;
}

/* ---- uploader ---- */
div[data-testid="stFileUploader"] section {
    border:1px dashed rgba(255,79,135,0.45) !important;
    border-radius:16px !important;
    background: rgba(18,11,22,0.55) !important;
}
div[data-testid="stFileUploader"] section small,
div[data-testid="stFileUploader"] section span { color: var(--text-dim) !important; }

/* ---- expander ---- */
details[data-testid="stExpander"], div[data-testid="stExpander"] {
    background: rgba(13,9,18,0.55) !important;
    border:1px solid rgba(255,255,255,0.07) !important;
    border-radius:20px !important;
    backdrop-filter: blur(10px);
}
details[data-testid="stExpander"] summary p { color: var(--text) !important; letter-spacing:1.5px; }

/* ---- tabs ---- */
div[data-baseweb="tab-list"] { background:transparent !important; gap:8px; border-bottom:1px solid rgba(255,255,255,0.06); }
button[data-baseweb="tab"] {
    background:transparent !important; color: var(--text-dim) !important;
    font-family:'DM Mono', monospace !important; letter-spacing:1.6px; font-size:12px !important;
    text-transform:uppercase;
}
button[data-baseweb="tab"][aria-selected="true"] { color: var(--text) !important; }
div[data-baseweb="tab-highlight"] { background: linear-gradient(90deg, var(--coral), var(--peach)) !important; }

/* ---- section heading (replaces the old violet blocks) ---- */
.section-title {
    display:flex; align-items:center; gap:14px;
    font-family:'Playfair Display', serif; font-size:30px; color:var(--text);
    margin:6px 0 16px 0;
}
.section-title .orb {
    width:10px; height:10px; border-radius:50%;
    background: var(--coral); box-shadow:0 0 16px var(--coral);
    animation: breathe 5s ease-in-out infinite;
}
.section-rule { height:1px; flex:1;
    background: linear-gradient(90deg, rgba(255,79,135,0.55), rgba(139,77,255,0.25), transparent); }

/* ---- status pills ---- */
.status-indicator { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:8px; }
.status-green { background:#FFB38A; box-shadow:0 0 10px #FF9B7A; }
.status-red { background:#8B4DFF; box-shadow:0 0 10px #8B4DFF; opacity:0.75; }
.status-line { font-family:'DM Mono', monospace; font-size:12px; letter-spacing:1.6px;
               color: var(--text-dim); text-transform:uppercase; }

/* ---- sidebar ---- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D0912 0%, #150C1C 100%);
    border-right:1px solid rgba(255,255,255,0.06);
}
section[data-testid="stSidebar"] * { color: var(--text); }

/* ---- dataframe / alerts ---- */
div[data-testid="stDataFrame"] {
    border:1px solid rgba(255,255,255,0.08); border-radius:16px; overflow:hidden;
    background: rgba(18,11,22,0.6);
}
div[data-testid="stAlert"] {
    background: rgba(255,79,135,0.10) !important;
    border:1px solid rgba(255,79,135,0.30) !important;
    border-radius:14px !important; color: var(--text) !important;
}

@keyframes magicReveal { from{opacity:0; transform:translateY(-8px);} to{opacity:1; transform:translateY(0);} }
.magic-instructions { animation: magicReveal .5s ease; }

@keyframes riseIn { from{opacity:0; transform:translateY(22px);} to{opacity:1; transform:translateY(0);} }
.rise { animation: riseIn .9s cubic-bezier(.2,.8,.3,1) both; }
.rise-2 { animation: riseIn .9s cubic-bezier(.2,.8,.3,1) .15s both; }
.rise-3 { animation: riseIn .9s cubic-bezier(.2,.8,.3,1) .3s both; }
</style>

<div class="bio-field">
    <div class="bio-blob b1"></div>
    <div class="bio-blob b2"></div>
    <div class="bio-blob b3"></div>
    <div class="bio-spark s1"></div>
    <div class="bio-spark s2"></div>
    <div class="bio-spark s3"></div>
    <div class="bio-spark s4"></div>
    <div class="bio-spark s5"></div>
</div>
"""


OPENING_ANIMATION = """
<style>
.boot {
    position:fixed; inset:0; z-index:10000;
    background: radial-gradient(900px 600px at 50% 45%, #140C1A 0%, #08070D 70%);
    display:flex; flex-direction:column; justify-content:center; align-items:center; gap:18px;
}
.boot-seed {
    position:absolute; width:10px; height:10px; border-radius:50%;
    background:#FF4F87; box-shadow:0 0 28px #FF4F87;
    animation: seed 1.6s ease-out forwards;
}
@keyframes seed {
    0% { transform:scale(0); opacity:0; }
    25% { transform:scale(1); opacity:1; }
    100% { transform:scale(0.2); opacity:0; }
}
.boot-ecg { width:320px; height:70px; animation: ecgOut 2.6s ease-in forwards; }
.boot-ecg path {
    fill:none; stroke:#FF6B9D; stroke-width:2.2; stroke-linecap:round;
    filter: drop-shadow(0 0 7px #FF4F87);
    stroke-dasharray: 620; stroke-dashoffset: 620;
    animation: trace 1.5s ease-out .35s forwards;
}
@keyframes trace { to { stroke-dashoffset:0; } }
@keyframes ecgOut { 0%,62%{opacity:1; filter:blur(0);} 100%{opacity:0; filter:blur(7px); transform:scale(1.12);} }
.boot-particles { position:absolute; display:flex; gap:16px; opacity:0; animation: partsIn .9s ease-out 1.9s forwards; }
.boot-particles i {
    width:5px; height:5px; border-radius:50%; display:block;
    background:#FF9B7A; box-shadow:0 0 12px #FF9B7A;
    animation: gather 1.1s ease-in-out 2.1s forwards;
}
.boot-particles i:nth-child(2n){ background:#B06CFF; box-shadow:0 0 12px #8B4DFF; }
@keyframes partsIn { to{opacity:1;} }
@keyframes gather { to { transform:translateY(-10px) scale(0.3); opacity:0; } }
.boot-title {
    font-family:'Playfair Display', serif; font-size:52px; color:#EADCF2; letter-spacing:1px;
    opacity:0; animation: titleIn 1s ease-out 2.2s forwards;
    text-shadow:0 0 34px rgba(255,79,135,0.45);
}
.boot-sub {
    font-family:'DM Mono', monospace; font-size:12px; letter-spacing:4px; text-transform:uppercase;
    color:#A493B4; opacity:0; animation: titleIn .9s ease-out 2.7s forwards;
}
@keyframes titleIn { from{opacity:0; transform:translateY(14px); filter:blur(6px);} to{opacity:1; transform:translateY(0); filter:blur(0);} }
</style>
<div class="boot">
    <div class="boot-seed"></div>
    <svg class="boot-ecg" viewBox="0 0 320 70">
        <path d="M0,35 L95,35 L108,35 L118,12 L131,58 L143,35 L158,35 L168,22 L178,48 L188,35 L320,35"/>
    </svg>
    <div class="boot-particles"><i></i><i></i><i></i><i></i><i></i><i></i><i></i></div>
    <div class="boot-title">HCPredict</div>
    <div class="boot-sub">Understanding people. Through data.</div>
</div>
"""


HERO_ORGANISM = """
<div style="display:flex; justify-content:center; align-items:center; min-height:260px;">
<svg viewBox="0 0 420 300" width="100%" style="max-width:460px; overflow:visible;">
  <defs>
    <radialGradient id="core" cx="50%" cy="50%">
      <stop offset="0%" stop-color="#FFB38A" stop-opacity="0.95"/>
      <stop offset="45%" stop-color="#FF4F87" stop-opacity="0.55"/>
      <stop offset="100%" stop-color="#8B4DFF" stop-opacity="0"/>
    </radialGradient>
    <linearGradient id="thread" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#FF4F87"/>
      <stop offset="55%" stop-color="#FF9B7A"/>
      <stop offset="100%" stop-color="#B06CFF"/>
    </linearGradient>
    <filter id="soft"><feGaussianBlur stdDeviation="3.4"/></filter>
  </defs>

  <circle cx="210" cy="150" r="118" fill="url(#core)">
    <animate attributeName="r" values="112;128;112" dur="11s" repeatCount="indefinite"/>
    <animate attributeName="opacity" values="0.65;0.95;0.65" dur="11s" repeatCount="indefinite"/>
  </circle>

  <g stroke="url(#thread)" fill="none" stroke-width="1.5" opacity="0.85" filter="url(#soft)">
    <path d="M60,210 C120,190 130,120 200,110 C270,100 300,60 370,80">
      <animate attributeName="stroke-opacity" values="0.35;0.95;0.35" dur="9s" repeatCount="indefinite"/>
    </path>
    <path d="M50,120 C130,150 150,200 230,200 C300,200 330,240 390,215">
      <animate attributeName="stroke-opacity" values="0.9;0.3;0.9" dur="13s" repeatCount="indefinite"/>
    </path>
    <path d="M210,20 C180,90 250,120 230,180 C215,230 260,250 250,285">
      <animate attributeName="stroke-opacity" values="0.4;0.85;0.4" dur="15s" repeatCount="indefinite"/>
    </path>
    <path d="M100,40 C160,80 130,160 190,190 C240,215 230,260 300,270">
      <animate attributeName="stroke-opacity" values="0.75;0.25;0.75" dur="17s" repeatCount="indefinite"/>
    </path>
  </g>

  <g fill="#FFB38A">
    <circle cx="200" cy="110" r="3.2"><animate attributeName="r" values="2;5;2" dur="6s" repeatCount="indefinite"/></circle>
    <circle cx="230" cy="200" r="2.6" fill="#FF6B9D"><animate attributeName="r" values="1.6;4.4;1.6" dur="8s" repeatCount="indefinite"/></circle>
    <circle cx="130" cy="120" r="2.2" fill="#B06CFF"><animate attributeName="r" values="1.4;4;1.4" dur="10s" repeatCount="indefinite"/></circle>
    <circle cx="300" cy="70" r="2.4"><animate attributeName="r" values="1.5;4.2;1.5" dur="7s" repeatCount="indefinite"/></circle>
    <circle cx="255" cy="255" r="2" fill="#FF4F87"><animate attributeName="r" values="1.2;3.6;1.2" dur="12s" repeatCount="indefinite"/></circle>
  </g>

  <g>
    <circle r="3" fill="#FFB38A" opacity="0.9">
      <animateMotion dur="9s" repeatCount="indefinite"
        path="M60,210 C120,190 130,120 200,110 C270,100 300,60 370,80"/>
    </circle>
    <circle r="2.6" fill="#B06CFF" opacity="0.85">
      <animateMotion dur="13s" repeatCount="indefinite"
        path="M50,120 C130,150 150,200 230,200 C300,200 330,240 390,215"/>
    </circle>
    <circle r="2.4" fill="#FF6B9D" opacity="0.8">
      <animateMotion dur="15s" repeatCount="indefinite"
        path="M210,20 C180,90 250,120 230,180 C215,230 260,250 250,285"/>
    </circle>
  </g>
</svg>
</div>
"""


def section_title(icon, text):
    return f"""
    <div class="section-title rise">
        <span class="orb"></span>
        <span>{icon} {text}</span>
        <span class="section-rule"></span>
    </div>
    """


def main():
    # Initialize the database
    init_db()

    # Sidebar instructions
    with st.sidebar:
        if "show_instructions" not in st.session_state:
            st.session_state.show_instructions = False

        st.markdown(
            "<div class='eyebrow' style='margin-bottom:6px;'>✦ Guide</div>"
            "<h4 style='margin-top:0; text-align:left;'>How this works</h4>",
            unsafe_allow_html=True
        )
        if st.button("✨ Show steps"):
            st.session_state.show_instructions = not st.session_state.show_instructions

        if st.session_state.show_instructions:
            st.markdown("""
            <div class="glass-card magic-instructions">
                <ol style="padding-left:18px; line-height:1.9; font-family:'DM Sans', sans-serif;">
                    <li>Download the <b>NPI file</b> first</li>
                    <li>Then download the <b>Survey file</b></li>
                    <li>Click <b>Save</b> under each section</li>
                    <li>You will then be redirected to the next page</li>
                    <li>There you can enter a <b>time period</b></li>
                    <li>The available staff will be shown as <b>plots and graphs, according to region</b></li>
                </ol>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style="margin-top:26px;">
            <div class="eyebrow">System Status</div>
            <div class="status-line" style="margin-top:8px;">
                <span class="status-indicator status-green"></span> Online
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Initialize session state variables
    if 'initial_load' not in st.session_state:
        st.session_state.initial_load = True
    if 'show_gif' not in st.session_state:
        st.session_state.show_gif = False
    if 'show_transition' not in st.session_state:
        st.session_state.show_transition = False
    if 'show_analysis' not in st.session_state:
        st.session_state.show_analysis = False
    if 'slideshow_completed' not in st.session_state:
        st.session_state.slideshow_completed = False
    if 'npi_file' not in st.session_state:
        st.session_state.npi_file = None
    if 'survey_file' not in st.session_state:
        st.session_state.survey_file = None
    if 'npi_df' not in st.session_state:
        st.session_state.npi_df = None
    if 'survey_df' not in st.session_state:
        st.session_state.survey_df = None
    if 'rf_model' not in st.session_state:
        st.session_state.rf_model = None
    if 'model_accuracy' not in st.session_state:
        st.session_state.model_accuracy = None

    # Opening animation: seed -> heartbeat -> particles -> HCPredict (~3s)
    if st.session_state.initial_load:
        loading_placeholder = st.empty()
        with loading_placeholder:
            st.markdown(OPENING_ANIMATION, unsafe_allow_html=True)
            time.sleep(3.2)
        loading_placeholder.empty()
        st.session_state.initial_load = False

    # Inject the global theme
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

    # Check if data is already in the database
    data_status = check_data_status()

    # Initial view (shown until both files are uploaded)
    if not st.session_state.show_analysis and not st.session_state.show_transition:
        # Thin floating navigation
        st.markdown("""
        <div class="nav-bar rise">
            <div class="nav-brand"><span class="nav-dot"></span>HCPredict</div>
            <div class="nav-links">
                <span>Home</span><span>Predict</span><span>Explore</span><span>Insights</span><span>About</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        hero_left, hero_right = st.columns([1.15, 1])

        with hero_left:
            st.markdown("""
            <div style="padding-top:26px;">
                <div class="eyebrow rise">Human behavior. Data intelligence.</div>
                <h1 style="text-align:left; font-size:56px; line-height:1.08; margin:14px 0 0 0;" class="rise-2">
                    Smarter<br>
                    <span style="font-style:italic; background:linear-gradient(90deg,#FF4F87,#FF9B7A,#B06CFF);
                                 -webkit-background-clip:text; background-clip:text; color:transparent;">Healthcare</span><br>
                    Connections
                </h1>
                <p class="rise-3" style="color:#A493B4; font-size:17px; line-height:1.7; max-width:440px; margin-top:18px;">
                    AI-powered prediction to understand HCP availability and engagement.
                    Start by loading the signals below.
                </p>
            </div>
            """, unsafe_allow_html=True)

        with hero_right:
            st.markdown(HERO_ORGANISM, unsafe_allow_html=True)

        st.markdown("<div style='margin-bottom:34px;'></div>", unsafe_allow_html=True)

        # Data upload section
        with st.expander("Input Signals", expanded=True):
            st.markdown("""
            <div style="margin-bottom:14px;">
                <div class="eyebrow">Step 00</div>
                <p style="color:#A493B4; margin-top:8px;">
                    Provide the behavioral signals and let the model analyze.
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Reset Database Button
            if st.button("Reset Database", key="reset_db"):
                conn = sqlite3.connect('npi_survey_data.db')
                c = conn.cursor()
                c.execute("DROP TABLE IF EXISTS data_status")
                c.execute("DROP TABLE IF EXISTS csv_data")
                conn.commit()
                conn.close()
                init_db()  # Recreate tables
                st.success("Database reset successfully!")
                st.session_state.show_analysis = False
                st.session_state.show_transition = False
                st.rerun()

            # ---------------- NPI DATA ----------------
            st.markdown(section_title("🧬", "NPI Data"), unsafe_allow_html=True)

            nsample_data = b"""NPI,login_date,login_hour,login_minute,logout_date,logout_hour,logout_minute,Region_Midwest,Region_Northeast,Region_South,Region_West,Speciality_Cardiology,Speciality_General Practice,Speciality_Neurology,Speciality_Oncology,Speciality_Orthopedics,Speciality_Pediatrics,Speciality_Radiology,State_TX,State_CA,Count of Survey Attempts,Usage Time (mins)
1234567890,2024-01-10,8,30,2024-01-10,10,0,1,0,0,0,0,1,0,0,0,0,0,1,0,5,90
1234567891,2024-01-11,9,0,2024-01-11,11,30,0,1,0,0,0,1,0,0,0,0,1,0,0,3,120
1234567892,2024-01-12,10,0,2024-01-12,12,0,0,0,1,0,0,1,0,0,0,0,0,1,0,2,110
1234567893,2024-01-13,14,0,2024-01-13,16,30,0,0,0,1,1,0,0,0,0,0,1,0,0,4,150
"""
            st.download_button(
                label="Download Sample NPI File",
                data=BytesIO(nsample_data),
                file_name="npi2_sample_4_rows.csv",
                mime="text/csv",
                key="npi_sample_download"
            )

            if data_status['npi']['uploaded']:
                st.markdown(f"""
                <p class='status-line' style='text-align:left; margin-top:12px;'>
                    <span class='status-indicator status-green'></span>
                    Status: uploaded on {data_status['npi']['last_updated']}
                </p>
                """, unsafe_allow_html=True)
                if st.button("Clear NPI Data", key="clear_npi"):
                    clear_data('npi')
                    st.success("NPI data cleared successfully!")
                    st.session_state.show_analysis = False

                    st.rerun()
            else:
                st.markdown("""
                <p class='status-line' style='text-align:left; margin-top:12px;'>
                    <span class='status-indicator status-red'></span>
                    Status: awaiting signal
                </p>
                """, unsafe_allow_html=True)
                npi_file = st.file_uploader("Upload npi csv", type=['csv'], key="npi_uploader")

                st.session_state.npi_file = npi_file
                if st.session_state.npi_file is not None:
                    if st.button("Save NPI Data", key="save_npi"):
                        with st.spinner("Reading signals..."):
                            try:
                                npi_df = pd.read_csv(st.session_state.npi_file)
                                if npi_df.empty:
                                    st.error("The uploaded NPI CSV file is empty.")
                                else:
                                    st.session_state.npi_df = preprocess_npi_data(npi_df)
                                    st.session_state.npi_file.seek(0)
                                    if store_csv_data('npi', st.session_state.npi_file):
                                        st.markdown("""
                                        <div class="glass-card" style="text-align:center;">
                                            <div class="eyebrow">Signal received</div>
                                            <div style="font-family:'Playfair Display',serif; font-size:30px; margin-top:8px;
                                                        background:linear-gradient(90deg,#FF4F87,#FF9B7A,#B06CFF);
                                                        -webkit-background-clip:text; background-clip:text; color:transparent;">
                                                NPI data absorbed
                                            </div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        time.sleep(1.6)
                                        st.success("NPI data uploaded successfully!")
                                        new_data_status = check_data_status()
                                        if new_data_status['survey']['uploaded']:
                                            st.session_state.show_transition = True
                                            st.rerun()
                            except Exception as e:
                                st.error(f"Error processing NPI CSV: {str(e)}")

            st.markdown("<div style='margin-bottom:38px;'></div>", unsafe_allow_html=True)

            # ---------------- SURVEY DATA ----------------
            st.markdown(section_title("📡", "Survey Data"), unsafe_allow_html=True)

            ssample_data = b"""Survey ID,NPI,attempt_hour,attempt_minute
100010,1234567890,9,0
100010,1234567891,10,30
100010,1234567892,11,0
100010,1234567893,15,0
"""
            st.download_button(
                label="Download Sample Survey File",
                data=BytesIO(ssample_data),
                file_name="survey2_first_4_rows.csv",
                mime="text/csv",
                key="survey_sample_download"
            )

            if data_status['survey']['uploaded']:
                st.markdown(f"""
                <p class='status-line' style='text-align:left; margin-top:12px;'>
                    <span class='status-indicator status-green'></span>
                    Status: uploaded on {data_status['survey']['last_updated']}
                </p>
                """, unsafe_allow_html=True)
                if st.button("Clear Survey Data", key="clear_survey"):
                    clear_data('survey')
                    st.success("Survey data cleared successfully!")
                    st.session_state.show_analysis = False

                    st.rerun()
            else:
                st.markdown("""
                <p class='status-line' style='text-align:left; margin-top:12px;'>
                    <span class='status-indicator status-red'></span>
                    Status: awaiting signal
                </p>
                """, unsafe_allow_html=True)
                survey_file = st.file_uploader("Upload survey csv", type=['csv'], key="survey_uploader")

                st.session_state.survey_file = survey_file
                if st.session_state.survey_file is not None:
                    if st.button("Save Survey Data", key="save_survey"):
                        with st.spinner("Reading signals..."):
                            try:
                                survey_df = pd.read_csv(st.session_state.survey_file)
                                if survey_df.empty:
                                    st.error("The uploaded Survey CSV file is empty.")
                                else:
                                    st.session_state.survey_df = survey_df
                                    st.session_state.survey_file.seek(0)
                                    if store_csv_data('survey', st.session_state.survey_file):
                                        st.markdown("""
                                        <div class="glass-card" style="text-align:center;">
                                            <div class="eyebrow">Signal received</div>
                                            <div style="font-family:'Playfair Display',serif; font-size:30px; margin-top:8px;
                                                        background:linear-gradient(90deg,#FF4F87,#FF9B7A,#B06CFF);
                                                        -webkit-background-clip:text; background-clip:text; color:transparent;">
                                                Survey data absorbed
                                            </div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        time.sleep(1.4)
                                        st.markdown("<div class='signal-log'>analyzing signals...</div>", unsafe_allow_html=True)
                                        time.sleep(0.8)
                                        st.markdown("<div class='signal-log'>model ready</div>", unsafe_allow_html=True)
                                        st.success("Survey data uploaded successfully!")
                                        new_data_status = check_data_status()
                                        if new_data_status['npi']['uploaded']:
                                            st.session_state.show_transition = True
                                            st.rerun()
                            except Exception as e:
                                st.error(f"Error processing Survey CSV: {str(e)}")

    # Transition: signals converge into the model
    if st.session_state.show_transition and st.session_state.npi_df is not None and st.session_state.survey_df is not None:
        transition_container = st.empty()

        transition_container.markdown("""
        <style>
        .flow-page {
            position:fixed; inset:0; z-index:1000;
            background: radial-gradient(900px 620px at 50% 50%, #150C1C 0%, #08070D 72%);
            display:flex; flex-direction:column; justify-content:center; align-items:center; gap:26px;
        }
        .flow-svg { width:min(680px, 86vw); }
        .flow-svg path {
            fill:none; stroke:url(#flowgrad); stroke-width:1.6; opacity:0.85;
            stroke-dasharray:520; stroke-dashoffset:520;
            animation: draw 1.6s ease-out forwards;
        }
        .flow-svg path:nth-child(2){ animation-delay:.15s; }
        .flow-svg path:nth-child(3){ animation-delay:.3s; }
        .flow-svg path:nth-child(4){ animation-delay:.45s; }
        @keyframes draw { to { stroke-dashoffset:0; } }
        .flow-core {
            width:88px; height:88px; border-radius:50%;
            background: radial-gradient(circle at 40% 40%, #FFB38A, #FF4F87 45%, rgba(139,77,255,0) 72%);
            box-shadow:0 0 70px rgba(255,79,135,0.55);
            animation: corePulse 2.4s ease-in-out infinite;
        }
        @keyframes corePulse { 0%,100%{transform:scale(0.92);} 50%{transform:scale(1.12);} }
        .flow-text {
            font-family:'DM Mono', monospace; font-size:12px; letter-spacing:5px;
            text-transform:uppercase; color:#A493B4;
        }
        .flow-done {
            font-family:'Playfair Display', serif; font-size:30px; color:#EADCF2;
            opacity:0; animation: fadeUp .9s ease-out 3.4s forwards;
        }
        @keyframes fadeUp { from{opacity:0; transform:translateY(14px);} to{opacity:1; transform:translateY(0);} }
        </style>
        <div class="flow-page">
            <svg class="flow-svg" viewBox="0 0 680 200">
                <defs>
                    <linearGradient id="flowgrad" x1="0" y1="0" x2="1" y2="0">
                        <stop offset="0%" stop-color="#8B4DFF"/>
                        <stop offset="60%" stop-color="#FF4F87"/>
                        <stop offset="100%" stop-color="#FF9B7A"/>
                    </linearGradient>
                </defs>
                <path d="M20,30 C220,30 260,100 340,100"/>
                <path d="M20,80 C220,80 270,100 340,100"/>
                <path d="M20,130 C220,130 270,100 340,100"/>
                <path d="M20,180 C220,180 260,100 340,100"/>
            </svg>
            <div class="flow-core"></div>
            <div class="flow-text">Analyzing signals</div>
            <div class="flow-done">Prediction surface ready</div>
        </div>
        """, unsafe_allow_html=True)

        time.sleep(4.6)

        st.session_state.show_transition = False
        st.session_state.show_analysis = True
        transition_container.markdown("""
        <div style="position:fixed; inset:0; background:#08070D; z-index:1000;"></div>
        """, unsafe_allow_html=True)
        time.sleep(0.15)
        transition_container.empty()
        st.rerun()

    # Post-upload view
    if st.session_state.show_analysis:
        st.markdown("""
        <div class="nav-bar rise">
            <div class="nav-brand"><span class="nav-dot"></span>HCPredict</div>
            <div class="nav-links">
                <span>Home</span><span>Predict</span><span>Explore</span><span>Insights</span><span>About</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("Loading signals from the store..."):
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

        # Train the Random Forest model
        @st.cache_resource
        def get_rf_model():
            return train_rf_model(npi_df, survey_df)

        rf_model = get_rf_model()

        # ---- stat fragments (real numbers from the loaded data) ----
        n_records = len(npi_df)
        n_specialties = len([c for c in npi_df.columns if c.startswith('Speciality_')])
        n_regions = len([c for c in npi_df.columns if c.startswith('Region_')])
        acc = st.session_state.get('model_accuracy')
        acc_txt = f"{acc * 100:.0f}%" if acc is not None else "—"

        s1, s2, s3, s4 = st.columns(4)
        for col, val, key in [
            (s1, f"{n_records:,}", "HCP Records"),
            (s2, f"{n_specialties}", "Specialties"),
            (s3, f"{n_regions}", "Regions"),
            (s4, acc_txt, "Model Accuracy"),
        ]:
            with col:
                st.markdown(f"""
                <div class="stat-frag rise">
                    <div class="stat-val">{val}</div>
                    <div class="stat-key">{key}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<div style='margin-bottom:34px;'></div>", unsafe_allow_html=True)

        # Input Section
        st.markdown("""
        <div style="text-align:center; margin-bottom:6px;" class="rise">
            <div class="eyebrow">Input Signals</div>
            <h2 style="font-size:36px; margin:10px 0 6px 0;">Analysis Parameters</h2>
            <p style="color:#A493B4;">Provide the behavioral signals and let the model analyze.</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 0.5, 2])
        with col2:
            survey_id = st.number_input("Survey ID", min_value=100000, max_value=999999, value=100010, key="survey_id")
            time_str = st.text_input("Analysis Time (HH:MM)", value="00:30", key="time_input")
            run_button = st.button("Predict Now →", key="run_button", help="Click to run the analysis")

        st.markdown("<div style='margin-bottom:22px;'></div>", unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["📈 Survey Analysis", "🌍 NPI Distribution", "⏰ Time Patterns"])

        with tab1:
            if 'run_analysis_triggered' in st.session_state and st.session_state.run_analysis_triggered:
                with st.spinner("Analyzing signals..."):
                    result = analyze_survey_participation(survey_id, time_str, survey_df, npi_df, rf_model)
                st.session_state.run_analysis_triggered = False

                if isinstance(result, str):
                    st.error(result)
                else:
                    if result['Active NPIs with Participation Probability']:
                        fig_region, fig_state, fig_specialty, fig_region_part, fig_specialty_part = create_visualizations(
                            result['Active NPIs with Participation Probability']
                        )
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown('<div class="glass-card"><h4>Active NPIs by Region</h4>', unsafe_allow_html=True)
                            st.plotly_chart(fig_region, use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                            st.markdown('<div class="glass-card"><h4>Active NPIs by State (Top 15)</h4>', unsafe_allow_html=True)
                            st.plotly_chart(fig_state, use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        with col2:
                            st.markdown('<div class="glass-card"><h4>Active NPIs by Specialty</h4>', unsafe_allow_html=True)
                            st.plotly_chart(fig_specialty, use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                            st.markdown('<div class="glass-card"><h4>Participation Rate by Region</h4>', unsafe_allow_html=True)
                            st.plotly_chart(fig_region_part, use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        st.markdown('<div class="glass-card"><h4>Participation Rate by Specialty</h4>', unsafe_allow_html=True)
                        st.plotly_chart(fig_specialty_part, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                    results_df = pd.DataFrame(result['Active NPIs with Participation Probability'])
                    st.markdown(section_title("✦", "Active NPIs"), unsafe_allow_html=True)
                    st.dataframe(results_df, use_container_width=True)

                    csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Results as CSV",
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
            st.markdown(section_title("🌍", "Overall NPI Distribution"), unsafe_allow_html=True)

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

            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="glass-card"><h4>NPIs by Region</h4>', unsafe_allow_html=True)
                fig_region_all = px.bar(region_df, x='Region', y='Count', title='', color='Region',
                                        color_discrete_sequence=PLOT_COLORWAY, height=400)
                st.plotly_chart(style_fig(fig_region_all), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('<div class="glass-card"><h4>Top 15 States</h4>', unsafe_allow_html=True)
                fig_state_all = px.bar(state_df, x='State', y='Count', title='', color='State',
                                       color_discrete_sequence=PLOT_COLORWAY, height=500)
                st.plotly_chart(style_fig(fig_state_all), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="glass-card"><h4>NPIs by Specialty</h4>', unsafe_allow_html=True)
                fig_specialty_all = px.pie(specialty_df, values='Count', names='Specialty', title='',
                                           color_discrete_sequence=PLOT_COLORWAY, hole=0.55, height=400)
                fig_specialty_all.update_traces(marker=dict(line=dict(color=INK_1, width=2)))
                st.plotly_chart(style_fig(fig_specialty_all), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('<div class="glass-card"><h4>NPI Count by Region and Specialty</h4>', unsafe_allow_html=True)
                fig_heatmap = px.imshow(pivot_df, title='',
                                        labels=dict(x="Region", y="Specialty", color="NPI Count"),
                                        color_continuous_scale=BIO_SCALE, height=500)
                st.plotly_chart(style_fig(fig_heatmap), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

        with tab3:
            st.markdown(section_title("⏱", "NPI Activity by Time"), unsafe_allow_html=True)

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

            st.markdown('<div class="glass-card"><h4>Active NPIs Throughout the Day</h4>', unsafe_allow_html=True)
            fig_time = px.line(time_df, x='Time', y='Active NPIs', title='', markers=True, height=400,
                               color_discrete_sequence=[CORAL])
            fig_time.update_traces(line=dict(width=2.4), marker=dict(size=6, color=PEACH))
            st.plotly_chart(style_fig(fig_time), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="glass-card"><h4>Active NPIs by Hour and Day</h4>', unsafe_allow_html=True)
            fig_heatmap = px.imshow(hour_matrix, labels=dict(x="Hour of Day", y="Day of Week", color="Active NPIs"),
                                    x=hours, y=days, title='', color_continuous_scale=BIO_SCALE, height=400)
            st.plotly_chart(style_fig(fig_heatmap), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="glass-card"><h4>Distribution of Login Hours</h4>', unsafe_allow_html=True)
                fig_login = px.bar(login_hour_counts, x='Hour', y='Count', title='', height=400,
                                   color_discrete_sequence=[CORAL])
                st.plotly_chart(style_fig(fig_login), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="glass-card"><h4>Distribution of Logout Hours</h4>', unsafe_allow_html=True)
                fig_logout = px.bar(logout_hour_counts, x='Hour', y='Count', title='', height=400,
                                    color_discrete_sequence=[PURPLE_SOFT])
                st.plotly_chart(style_fig(fig_logout), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

        # ---- footer ----
        st.markdown("""
        <div style="margin-top:50px; padding-top:22px; border-top:1px solid rgba(255,255,255,0.07);
                    display:flex; justify-content:space-between; align-items:flex-end; flex-wrap:wrap; gap:18px;">
            <div>
                <div class="nav-brand" style="font-size:20px;"><span class="nav-dot"></span>HCPredict</div>
                <div style="color:#A493B4; font-size:13px; margin-top:6px;">Understanding people. Through data.</div>
            </div>
            <div class="status-line">
                <span class="status-indicator status-green"></span> System online
            </div>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
