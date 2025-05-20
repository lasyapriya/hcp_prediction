import streamlit as st
import requests
import time
from datetime import datetime,timedelta
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

from streamlit_lottie import st_lottie
st.set_page_config(layout="wide", page_title="NPI Survey Analysis", page_icon="📊")


# App state management for loading overlay
if 'initial_load' not in st.session_state:
    st.session_state.initial_load = True

# Function to load Lottie animation
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load medical animation
lottie_medical = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_jcikwtux.json")

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
    
    st.write(f"Stored {data_type} CSV with size: {len(csv_content)} bytes")  # Debugging
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
            time_since_login_normalized, time_until_logout_normalized, inside_window] + region_features + specialty_features

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
    st.write(f"Random Forest Model Accuracy: {accuracy:.4f}")
    
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
    except:
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
        height=400
    )
    
    region_participation = df.groupby('Region')['Participated'].agg(['sum', 'count']).reset_index()
    region_participation['Participation Rate'] = (region_participation['sum'] / region_participation['count'] * 100).round(2)
    region_participation.columns = ['Region', 'Participants', 'Total', 'Participation Rate (%)']
    
    fig_region_participation = px.bar(
        region_participation,
        x='Region',
        y='Participation Rate (%)',
        title='Participation Rate by Region',
        color='Region',
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
        height=400
    )
    
    return fig_region, fig_state, fig_specialty, fig_region_participation, fig_specialty_participation
# --- Sample Data Byte Strings ---


from io import BytesIO
npi_sample_data = b"""NPI,login_date,login_hour,login_minute,logout_date,logout_hour,logout_minute,Region_Midwest,Region_Northeast,Region_South,Region_West,Speciality_Cardiology,Speciality_General Practice,Speciality_Neurology,Speciality_Oncology,Speciality_Orthopedics,Speciality_Pediatrics,Speciality_Radiology,State_TX,State_CA,Count of Survey Attempts,Usage Time (mins)
1234567890,2024-01-10,8,30,2024-01-10,10,0,1,0,0,0,0,1,0,0,0,0,0,1,0,5,90
1234567891,2024-01-11,9,0,2024-01-11,11,30,0,1,0,0,0,1,0,0,0,0,1,0,0,3,120
1234567892,2024-01-12,10,0,2024-01-12,12,0,0,0,1,0,0,1,0,0,0,0,0,1,0,2,110
1234567893,2024-01-13,14,0,2024-01-13,16,30,0,0,0,1,1,0,0,0,0,0,1,0,0,4,150
"""

survey_sample_data = b"""Survey ID,NPI,attempt_hour,attempt_minute
100010,1234567890,9,0
100010,1234567891,10,30
100010,1234567892,11,0
100010,1234567893,15,0
"""
def main():
    # Initialize the database
    init_db()

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

    # Display loading overlay on initial load
    if st.session_state.initial_load:
        loading_placeholder = st.empty()
        with loading_placeholder:
            st.markdown("""
                <div style="position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; display: flex; justify-content: center; align-items: center; z-index: 10000; background-color: #ffffff;">
                    <img src="https://cdn.pixabay.com/animation/2024/02/01/07/48/07-48-16-574_512.gif" width="350">
                </div>
                <style>
                    body { margin: 0; padding: 0; }
                </style>
            """, unsafe_allow_html=True)
            time.sleep(4)
        loading_placeholder.empty()
        st.session_state.initial_load = False

    # Inject custom CSS with enhanced file uploader styling and full-screen slideshow
    st.markdown("""
    <style>
.stAppViewContainer {
            max-width: 100% !important;
            margin: 0 auto !important;
            padding: 20px !important;
                background:url('https://cdn.dribbble.com/userupload/23646781/file/original-83141e8daaa41110fb9ab34daa6e8718.gif') !important;
            background-size: 16% !important;
                background-repeat: no-repeat !important;
        }
        .corner-gif {
            position: fixed;
            width: 300px;
            height: auto;
            z-index: 10002;
            pointer-events: none;
        }
        #top-left-gif {
            top: 10px;
            left: 10px;
        }
        #bottom-right-gif {
            bottom: 10px;
            right: 10px;
        }
        .main .block-container {
            max-width: 100% !important;
            padding: 0 !important;
            margin: 0 auto !important;
            position: relative;
            z-index: 10001;
        }
        h1 {
            font-family: 'Roboto', sans-serif !important;
            font-size: 36px !important;
            font-weight: 700 !important;
            text-align: center !important;
            padding: 10px !important;
            background: linear-gradient(90deg, #6B46C1, #D53F8C, #ED64A6) !important;
            -webkit-background-clip: text !important;
            background-clip: text !important;
            color: transparent !important;
            animation: flowGradient 4s ease infinite !important;
            transition: transform 0.3s ease !important;
        }
        h1:hover {
            transform: scale(1.05) !important;
        }
        h4 {
            font-family: 'Roboto', sans-serif !important;
            font-size: 34px !important;
            font-weight: 600 !important;
            text-align: center !important;
            padding: 10px !important;
            background: linear-gradient(90deg, #6B46C1, #D53F8C, #ED64A6) !important;
                background-width:80% !important;
                background-height:40% !important;
                            -webkit-background-clip: text !important;
            background-clip: text !important;
            color: transparent !important;
            animation: flowGradient 4s ease infinite !important;
            transition: transform 0.3s ease, color 0.3s ease !important;
        }
        h4:hover {
            transform: scale(1.05) !important;
            color: #6B46C1 !important;
        }
        h3[data-testid="stMarkdownContainer"] p,
        div[data-testid="stExpander"] summary p {
            font-family: 'Roboto', sans-serif !important;
            font-size: 40px !important;
            font-weight: 600 !important;
            text-align: center !important;
            padding: 10px !important;
            background: linear-gradient(90deg, #6B46C1, #D53F8C, #ED64A6) !important;
            -webkit-background-clip: text !important;
            background-clip: text !important;
            color: transparent !important;
            animation: flowGradient 4s ease infinite !important;
            transition: transform 0.3s ease, color 0.3s ease !important;
        }
        h3[data-testid="stMarkdownContainer"] p:hover,
        div[data-testid="stExpander"] summary p:hover {
            transform: scale(1.05) !important;
            color: #6B46C1 !important;
        }
        @keyframes flowGradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        .data-section {
            border: 2px solid #6B46C1;
            border-radius: 10px;
                width:60% !important;
            padding: 15px;
            margin-bottom: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            background: linear-gradient(135deg, #E9D8FD, #B794F4);
        }
        .data-section:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 12px rgba(107, 70, 193, 0.3);
        }
        div[data-testid="stFileUploader"] {
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            width: 100% !important;
            max-width: 550px !important;
            margin: 0 auto !important;
            border: none !important;
            border-radius: 0 !important;
            padding: 0 !important;
            box-shadow: none !important;
            background-color: transparent !important;
        }
        div[data-testid="stFileUploader"] section {
            width: 100% !important;
            border: 4px dashed #6B46C1 !important;
            border-radius: 10px !important;
            padding: 20px !important;
            background: linear-gradient(135deg, #D8B4F8, #E0BBE4) !important;
            transition: all 0.3s ease !important;
                margin-top:0px !important;
                margin-bottom:40px !important;
        }
        div[data-testid="stFileUploader"]:hover section {
            border-color: #D53F8C !important;
            background: linear-gradient(135deg, #E6E6FA, #E0B0FF, #DCD0FF) !important;
        }
        div[data-testid="stFileUploader"] button {
            margin-top: 10px !important;
            width: 150px !important;
            background: linear-gradient(135deg, #6B46C1, #D53F8C) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 5px !important;
            transition: background-color 0.3s ease, transform 0.3s ease !important;
        }
        div[data-testid="stFileUploader"] button:hover {
            background: linear-gradient(135deg, #D53F8C, #ED64A6) !important;
            transform: scale(1.05) !important;
        }
        div[data-testid="stFileUploader"] p {
            font-family: 'Roboto', sans-serif !important;
            font-size: 18px !important;
            color: #6B46C1 !important;
        }
        .stNumberInput input, .stTextInput input {
            width: 200px !important;
            background: linear-gradient(135deg, #E9D8FD, #B794F4) !important;
            color: #6B46C1 !important;
            border: 2px solid #6B46C1 !important;
            border-radius: 5px !important;
            padding: 5px !important;
        }
        .stButton>button {
            width: 150px !important;
            margin-top: 10px;
            background: linear-gradient(135deg, #6B46C1, #D53F8C) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            transition: all 0.3s ease-in-out !important;
            box-shadow: 0 2px 5px rgba(213, 63, 140, 0.5) !important;
            opacity: 0.9;
        }
        .stButton>button:hover {
            background: linear-gradient(135deg, #D53F8C, #ED64A6) !important;
            transform: scale(1.05) !important;
            opacity: 1 !important;
        }
        div[data-testid="stTextInput"] input,
        div[data-testid="stNumberInput"] input {
            font-size: 1.2rem !important;
            padding: 10px !important;
            color: #6B46C1 !important;
            border: 2px solid #6B46C1 !important;
            border-radius: 5px !important;
        }
        div[data-testid="stNumberInput"] p,
        div[data-testid="stTextInput"] p,
        .stTextInput > div > p,
        .stNumberInput > div > p,
        .stTextInput > div > label,
        .stNumberInput > div > label {
            font-size: 26px !important;
            color: #D53F8C !important;
            text-align: center !important;
            font-weight: bold !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            justify-content: center !important;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 28px !important;
            background: linear-gradient(90deg, #6B46C1, #D53F8C, #ED64A6) !important;
            -webkit-background-clip: text !important;
            background-clip: text !important;
            color: transparent !important;
            text-align: center !important;
            font-weight: bold !important;
        }
        .graph-card {
            border: 2px solid #6B46C1;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            padding: 15px;
            background: linear-gradient(135deg, #E9D8FD, #B794F4);
            margin-bottom: 20px;
            overflow-y: auto;
            max-height: 500px;
            width: 80% !important;
            margin-left: auto !important;
            margin-right: auto !important;
            font-size: 16px;
        }
        .graph-card h4 {
            font-size: 24px;
            background: linear-gradient(90deg, #6B46C1, #D53F8C, #ED64A6) !important;
            -webkit-background-clip: text !important;
            background-clip: text !important;
            color: transparent !important;
            margin-bottom: 10px;
            text-align: center;
        }
        .golden-border-box {
            border: 3px solid #6B46C1;
            background: linear-gradient(135deg, #6B46C1, #D53F8C);
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
            margin-bottom: 20px;
            width: 80% !important;
            margin-left: auto !important;
            margin-right: auto !important;
            color: #ffffff;
        }
        .success-msg {
            text-align: center;
            font-size: 30px;
            background: linear-gradient(to right, #D53F8C, #ED64A6);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            margin-top: 10px;
            font-weight: bold;
        }
        .syringe-pop {
            width: 150px;
            margin-top: 10px;
            animation: popBounceFade 5s ease forwards;
            display: block;
            margin-left: auto;
            margin-right: auto;
        }
        .st-emotion-cache-br351g p {
            font-size: 22px;
            color: #6B46C1;
            font-weight: 40px;
        }
        @keyframes popBounceFade {
            0% { transform: scale(0.5); opacity: 0; }
            20% { transform: scale(1.2); opacity: 1; }
            40% { transform: scale(0.9) rotate(-10deg); }
            60% { transform: scale(1.05) rotate(10deg); }
            80% { transform: scale(1) rotate(0deg); }
            100% { transform: scale(1); opacity: 0; }
        }
        .upload-section {
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 5px;
        }
        .status-green {
            background-color: #2ecc71;
        }
        .status-red {
            background-color: #e74c3c;
        }
        .card {
            background: linear-gradient(135deg, #E9D8FD, #B794F4);
            border-left: 4px solid #6B46C1;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 5px;
            font-size: 24px;
            color: #6B46C1;
            width: 80% !important;
            margin-left: auto !important;
            margin-right: auto !important;
            text-align: center;
        }
        @media (max-width: 768px) {
            .graph-card {
                max-height: 400px;
                width: 90% !important;
            }
            .stTextInput > div > label, .stNumberInput > div > label {
                font-size: 22px !important;
            }
            .stTabs [data-baseweb="tab"] {
                font-size: 22px !important;
            }
            .stApp > div:nth-child(2) > div:nth-child(2) > div {
                flex-direction: column !important;
            }
            .stApp > div:nth-child(2) > div:nth-child(2) > div > div {
                width: 100% !important;
                margin-bottom: 20px !important;
            }
        }
        .stNumberInput input, .stTimeInput input {
            width: 100px !important;
                height: 50px !important;
            background: linear-gradient(135deg, #E9D8FD, #B794F4) !important;
            color: #6B46C1 !important;
            border: 2px solid #6B46C1 !important;
            border-radius: 5px !important;
            padding: 10px !important;
        }
       
        .text {
            color: #ffffff;
            font-size: 24px;
            position: absolute;
            bottom: 20px;
            width: 100%;
            text-align: center;
            z-index: 10001;
            background: linear-gradient(90deg, #6B46C1, #D53F8C, #ED64A6);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }
        
        .active {
            background-color: #6B46C1;
        }
        @keyframes fade {
            from { opacity: .4; }
            to { opacity: 1; }
        }
        .proceed-button {
            position: fixed;
            bottom: 50px;
            left: 50%;
            transform: translateX(-50%);
            padding: 15px 30px;
            background: linear-gradient(135deg, #6B46C1, #D53F8C);
            color: #ffffff;
            border: none;
            border-radius: 8px;
            font-size: 18px;
            font-weight: bold;
            cursor: pointer;
            z-index: 10002;
            transition: all 0.3s ease;
            display: none;
        }
        .proceed-button:hover {
            background: linear-gradient(135deg, #D53F8C, #ED64A6);
            transform: translateX(-50%) scale(1.05);
        }
        .stApp [data-testid="stAppViewContainer"].analysis-view {
            background-image: none !important;
            background-color: #ffffff !important;
            transition: background-color 0.5s ease;
        }
        .stApp {
        background-color: #FFFFFF;
    
    }
</style>
    """, unsafe_allow_html=True)

    # Check if data is already in the database
    data_status = check_data_status()

    # Initial view (shown until both files are uploaded)
    if not st.session_state.show_analysis and not st.session_state.show_transition:
        # Lottie animation
        col1, col2, col3 = st.columns([2, 2, 2])
        with col2:
            st_lottie(lottie_medical, speed=1, height=200, key="medical")

        # Header
        st.markdown("""
<div style='
font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif, 
"Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol";
font-size: 80px;
font-weight: 700;
color: #4B006E;
text-align: center;
margin-top: 30px;
'>
HCP Campaign Prediction
</div>
""", unsafe_allow_html=True)
        
        
# Subtitle
        st.markdown("""
<div style='
font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif, 
"Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol";
font-size: 20px;
color: #4B0082;
text-align: center;
margin-top: 10px;
'>
Analyze NPI participation patterns with interactive visualization
</div>
""", unsafe_allow_html=True)
        st.markdown("""
    <div style="margin-bottom: 40px;"></div>
""", unsafe_allow_html=True)
        # Data upload section
        with st.expander("Data Management", expanded=True):
            st.markdown("""
    <style>
    .section-box {
        border: 2px solid #D8BFD8;
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        background-color: transparent;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .section-heading {
        font-size: 26px;
        font-weight: bold;
        color: #800080;
        text-align: center;
        margin-bottom: 10px;
    }
    .status {
        text-align: left;
        font-style: italic;
        margin-bottom: 10px;
        font-size: 18px;
    }
</style>
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
            
            st.markdown('</div>', unsafe_allow_html=True)

            # NPI Data section inside light violet box
            st.markdown("""
                        <style>
                        .centered-box-wrapper {
    display: flex;
    justify-content: center;
    margin-bottom: 20px;
}
         .data-heading {
    background-color: #E9D8FD;
    border-radius: 15px;
    padding: 15px 25px;
    width:60%;
    height:120%;                    
    text-align: center;
 font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif, 
"Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol";
    font-size: 50px;
    font-weight: 700;
    color: #4B0082;
    border: 3px solid #6B46C1;
    position: relative;
    overflow: hidden;
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 12px;
    transition: all 0.3s ease;
}

.data-heading:hover {
    transform: scale(1.03);
    box-shadow: 0 0 15px rgba(107, 70, 193, 0.4);
}

.data-heading::before {
    content: "";
    position: absolute;
    top: 0;
    left: -60%;
    width: 50%;
    height: 100%;
    background: linear-gradient(120deg, rgba(255,255,255,0.2), rgba(255,255,255,0));
    transform: skewX(-20deg);
    transition: left 0.7s ease;
}

.data-heading:hover::before {
    left: 120%;
}

.icon {
    font-size: 46px;
    color: #6B46C1;
}
</style>
<div class="centered-box-wrapper">
<div class="data-heading">
    <span class="icon">🩺</span>
    NPI Data
</div>
                        </div>
  """, unsafe_allow_html=True)
            st.download_button(
    label="Download Sample NPI File",
    data=BytesIO(npi_sample_data),
    file_name="npi2_sample_4_rows.csv",
    mime="text/csv",
    key="npi_sample_download"
            )
            if data_status['npi']['uploaded']:
               st.markdown(f"""
        <p style='text-align: left;'>
            <span class='status-indicator status-green'></span>
            <strong>Status:</strong> Uploaded on {data_status['npi']['last_updated']}
        </p>
    """, unsafe_allow_html=True)
               if st.button("Clear NPI Data", key="clear_npi"):
                   clear_data('npi')
                   st.success("NPI data cleared successfully!")
                   st.session_state.show_analysis = False
                   
                   st.rerun()
            else:
               st.markdown("""
                       <p style='text-align: left;'>
                               <span class='status-indicator status-red'></span>
                               <strong>Status:</strong> Not uploaded
                                 </p>
                                     """, unsafe_allow_html=True)
               npi_file = st.file_uploader("Upload npi csv", type=['csv'], key="npi_uploader")
               from io import BytesIO
               st.download_button(
                label="Download Sample NPI File",
                 data=BytesIO(npi_sample_data),
                 file_name="npi2_sample_4_rows.csv",
                mime="text/csv",
                 key="npi_sample_button"
                             )
               st.session_state.npi_file = npi_file
               if st.session_state.npi_file is not None:
                   if st.button("Save NPI Data", key="save_npi"):
                      with st.spinner("Uploading NPI data..."):
                           try:
                                npi_df = pd.read_csv(st.session_state.npi_file)
                                if npi_df.empty:
                                    st.error("The uploaded NPI CSV file is empty.")
                                else:
                                    st.session_state.npi_df = preprocess_npi_data(npi_df)
                                    st.session_state.npi_file.seek(0)
                                    if store_csv_data('npi', st.session_state.npi_file):
                                      st.markdown(f"""
                                             <div class="success-msg">🎉 Upload Successful!</div>
                                                <img class="syringe-pop" src="https://designmodo.com/wp-content/uploads/2015/09/medical-emergencies.gif" width:"250px" height:"200px" />
                                                  """, unsafe_allow_html=True)
                                      time.sleep(2.5)
                                      st.markdown("<div style='text-align: center; font-size: 30px; color: #6B46C1;'>Data Loaded</div>", unsafe_allow_html=True)
                                      st.success("NPI data uploaded successfully!")
                                      new_data_status = check_data_status()
                                      if new_data_status['survey']['uploaded']:
                                          st.session_state.show_transition = True
                                          st.rerun()
                           except Exception as e:
                                 st.error(f"Error processing NPI CSV: {str(e)}")
            st.markdown("""
    <div style="margin-bottom: 40px;"></div>
""", unsafe_allow_html=True)

            st.markdown("""
                        <style>
                       .centered-box-wrapper {
    display: flex;
    justify-content: center;
    margin-bottom: 20px;
}
         .data-heading {
    background-color: #E9D8FD;
    border-radius: 15px;
    padding: 15px 25px;
    width:60%;
    height:120%;                   
    text-align: center;
 font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif, 
"Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol";
    font-size: 50px;
    font-weight: 700;
    color: #4B0082;
    border: 3px solid #6B46C1;
    position: relative;
    overflow: hidden;
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 12px;
    transition: all 0.3s ease;
}

.data-heading:hover {
    transform: scale(1.03);
    box-shadow: 0 0 15px rgba(107, 70, 193, 0.4);
}

.data-heading::before {
    content: "";
    position: absolute;
    top: 0;
    left: -60%;
    width: 50%;
    height: 100%;
    background: linear-gradient(120deg, rgba(255,255,255,0.2), rgba(255,255,255,0));
    transform: skewX(-20deg);
    transition: left 0.7s ease;
}

.data-heading:hover::before {
    left: 120%;
}

.icon {
    font-size: 46px;
    color: #6B46C1;
}
</style>
<div class="centered-box-wrapper">
<div class="data-heading">
    <span class="icon">📊</span>
    Survey Data
</div>
                        </div>

                                        """, unsafe_allow_html=True)
            st.download_button(
           label="Download Sample Survey File",
           data=BytesIO(survey_sample_data),
           file_name="survey2_first_4_rows.csv",
             mime="text/csv",
              key="survey_sample_download"
            )
            if data_status['survey']['uploaded']:
                   st.markdown(f"""
                        <p style='text-align: left;'>
                           <span class='status-indicator status-green'></span>
                          <strong>Status:</strong> Uploaded on {data_status['survey']['last_updated']}
                      </p>
                             """, unsafe_allow_html=True)
                   if st.button("Clear Survey Data", key="clear_survey"):
                     clear_data('survey')
                     st.success("Survey data cleared successfully!")
                     st.session_state.show_analysis = False
                     
                     st.rerun()
            else:
                  st.markdown("""
                            <p style='text-align: left;'>
                          <span class='status-indicator status-red'></span>
                        <strong>Status:</strong> Not uploaded
                          </p>
                          """, unsafe_allow_html=True)
                  survey_file = st.file_uploader("Upload survey csv", type=['csv'], key="survey_uploader")
                  st.download_button(
                  label="Download Sample Survey File",
                  data=BytesIO(survey_sample_data),
                  file_name="survey2_first_4_rows.csv",
                  mime="text/csv",
                  key="survey_sample_button"
                  )
                  st.session_state.survey_file = survey_file
                  if st.session_state.survey_file is not None:
                       if st.button("Save Survey Data", key="save_survey"):
                           with st.spinner("Uploading survey data..."):
                               try:
                                     survey_df = pd.read_csv(st.session_state.survey_file)
                                     if survey_df.empty:
                                           st.error("The uploaded Survey CSV file is empty.")
                                     else:
                                        st.session_state.survey_df = survey_df
                                        st.session_state.survey_file.seek(0)
                                        if store_csv_data('survey', st.session_state.survey_file):
                                          st.markdown(f"""
                                         <div class="success-msg">🎉 Upload Successful!</div>
                                          <img class="syringe-pop" src="https://designmodo.com/wp-content/uploads/2015/09/medical-emergencies.gif" width:"250px" height:"200px" />
                                           """, unsafe_allow_html=True)
                                          time.sleep(2.5)
                                          st.markdown("<div style='text-align: center; font-size: 30px; color: #6B46C1;'>Data Loaded</div>", unsafe_allow_html=True)
                                          time.sleep(1)
                                          st.markdown("<div style='text-align: center; font-size: 30px; color: #6B46C1;'>Processing Started</div>", unsafe_allow_html=True)
                                          time.sleep(1)
                                          st.markdown("<div style='text-align: center; font-size: 30px; color: #6B46C1;'>Ready for Analysis</div>", unsafe_allow_html=True)
                                          st.success("Survey data uploaded successfully!")
                                          new_data_status = check_data_status()
                                          if new_data_status['npi']['uploaded']:
                                               st.session_state.show_transition = True
                                               st.rerun()
                               except Exception as e:
                                   st.error(f"Error processing Survey CSV: {str(e)}")
    


    
    # Initialize slide index
    if st.session_state.show_transition and st.session_state.npi_df is not None and st.session_state.survey_df is not None:
      transition_container = st.empty()
    
    # First show the bubble animation
      transition_container.markdown("""
    <style>
    @keyframes big-bubble {
        0% { 
            transform: translate(-50%, -50%) scale(0); 
            opacity: 0; 
        }
        20% { 
            opacity: 1; 
        }
        80% { 
            transform: translate(-50%, -50%) scale(30); 
            opacity: 1; 
        }
        100% { 
            transform: translate(-50%, -50%) scale(30); 
            opacity: 0; 
        }
    }
    .bubble-transition {
        position: fixed;
        top: 50%;
        left: 50%;
        width: 50px;
        height: 50px;
        background: linear-gradient(135deg, #6B46C1, #D53F8C);
        border-radius: 50%;
        animation: big-bubble 2s ease-out forwards;
        z-index: 1001;
        pointer-events: none;
        transform: translate(-50%, -50%);
    }
    </style>
    <div class="bubble-transition"></div>
    """, unsafe_allow_html=True)
    
      time.sleep(1.5)  # Let bubble animation complete
    
    # Now show the GIF
      transition_container.markdown("""
    <style>
    .transition-page {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background-color:#EDEDEC; 
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 1000;
    }
    .gif-transition {
         max-width: 100vw;
        max-height: 100vh;
        animation: fadeIn 0.5s ease-out;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: scale(0.9); }
        to { opacity: 1; transform: scale(1); }
    }
    </style>
     <div class="transition-page">
        <img class="gif-tansition" 
             src="https://cdn.dribbble.com/userupload/26483443/file/original-1ab4eb0e3bfcb0f812768ca9e2bc8a5a.gif" 
             style="width:80vw; height:auto;max-width:900px;">
    </div>
    """, unsafe_allow_html=True)
    
      time.sleep(5.5)  # Show GIF for 3 seconds
    
    # Clear everything and proceed to analysis
      
      st.session_state.show_transition = False
      st.session_state.show_analysis = True
      transition_container.markdown("""
<div style="position:fixed; top:0; left:0; width:100vw; height:100vh; background:white; z-index:1000;"></div>
""", unsafe_allow_html=True)
      time.sleep(0.1)
      transition_container.empty()
      st.rerun()


     

     # Post-upload view (shown after slideshow)
    if st.session_state.show_analysis:
        st.markdown("""
        <style>
            .stApp [data-testid="stAppViewContainer"] {
                background: linear-gradient(to bottom right, #F4EFFF, #FFFFFF); !important;
            
                     
            }
            
        </style>
        """, unsafe_allow_html=True)
        with st.spinner("Loading data from database..."):
          processing_placeholder = st.empty()
          with processing_placeholder:
                
               
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
        
        # Input Section with improved styling
        st.markdown("""
        <style>
         

input[type="text"], input[type="time"], input[type="number"] {
    padding: 10px;
    border: none;
    border-radius: 8px;
    background: #f2e1ff;
    box-shadow: 0 0 10px rgba(130, 61, 255, 0.2);
    color: #5a0079;
    font-size: 16px;
    width: 200px;
    transition: 0.3s;
}

input[type="text"]:focus, input[type="time"]:focus, input[type="number"]:focus {
    box-shadow: 0 0 15px rgba(111, 0, 255, 0.5);
    outline: none;
}
                    
.analysis-heading {
    font-size: 36px;
    font-weight: bold;
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif,
"Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol";
    padding: 15px 25px;
    text-align: center;
    color: #4B0082;
    border-radius: 50px;
    background: linear-gradient(270deg, #a26cff, #bc6ff1, #c77dff, #a26cff);
    background-size: 600% 600%;
    animation: gradientFlow 6s ease infinite;
    box-shadow: 0 4px 15px rgba(130, 61, 255, 0.3);
    width: fit-content;
    margin: auto;
    margin-bottom: 30px;
}

@keyframes gradientFlow {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

</style>
    """, unsafe_allow_html=True)
        st.markdown("<div class='analysis-heading'>ANALYSIS PARAMETERS</div>", unsafe_allow_html=True)
       
        col1, col2, col3 = st.columns([1, 0.5, 2])
        with col2:
          survey_id = st.number_input("Survey ID", min_value=100000, max_value=999999, value=100010, key="survey_id")
          time_str = st.text_input("Analysis Time (HH:MM)", value="00:30", key="time_input")
          run_button = st.button("Run Analysis", key="run_button", help="Click to run the analysis")
       
        st.markdown('</div>', unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["📈 Survey Analysis", "🌍 NPI Distribution", "⏰ Time Patterns"])
        
        with tab1:
            if 'run_analysis_triggered' in st.session_state and st.session_state.run_analysis_triggered:
                with st.spinner("Analyzing survey participation..."):
                    
                    result = analyze_survey_participation(survey_id, time_str, survey_df, npi_df, rf_model)
                st.session_state.run_analysis_triggered = False
                
                if isinstance(result, str):
                    st.error(result)
                else:
                    st.markdown("<h3>Analysis Summary</h3>", unsafe_allow_html=True)
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"""
                            <div class='card'>
                                <h4>Total NPIs</h4>
                                <p style='font-size: 24px; color: #3498db;'>{result['Total NPIs in Database']}</p>
                            </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"""
                            <div class='card'>
                                <h4>Active NPIs</h4>
                                <p style='font-size: 24px; color: #3498db;'>{result['Active NPIs at Analysis Time']}</p>
                            </div>
                        """, unsafe_allow_html=True)
                    with col3:
                        st.markdown(f"""
                            <div class='card'>
                                <h4>Participation</h4>
                                <p style='font-size: 24px; color: #3498db;'>{result['Participation Percentage']:.2f}%</p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    if result['Active NPIs with Participation Probability']:
                        fig_region, fig_state, fig_specialty, fig_region_part, fig_specialty_part = create_visualizations(
                            result['Active NPIs with Participation Probability']
                        )
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown('<div class="graph-card"><h4>Active NPIs by Region</h4>', unsafe_allow_html=True)
                            st.plotly_chart(fig_region, use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                            st.markdown('<div class="graph-card"><h4>Active NPIs by State (Top 15)</h4>', unsafe_allow_html=True)
                            st.plotly_chart(fig_state, use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        with col2:
                            st.markdown('<div class="graph-card"><h4>Active NPIs by Specialty</h4>', unsafe_allow_html=True)
                            st.plotly_chart(fig_specialty, use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                            st.markdown('<div class="graph-card"><h4>Participation Rate by Region</h4>', unsafe_allow_html=True)
                            st.plotly_chart(fig_region_part, use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        st.markdown('<div class="graph-card"><h4>Participation Rate by Specialty</h4>', unsafe_allow_html=True)
                        st.plotly_chart(fig_specialty_part, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    results_df = pd.DataFrame(result['Active NPIs with Participation Probability'])
                    st.markdown("<h3>Active NPIs</h3>", unsafe_allow_html=True)
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
            st.markdown("<h3>Overall NPI Distribution</h3>", unsafe_allow_html=True)
            
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
                st.markdown('<div class="graph-card"><h4>NPIs by Region</h4>', unsafe_allow_html=True)
                fig_region_all = px.bar(region_df, x='Region', y='Count', title='', color='Region', height=400)
                st.plotly_chart(fig_region_all, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('<div class="graph-card"><h4>Top 15 States</h4>', unsafe_allow_html=True)
                fig_state_all = px.bar(state_df, x='State', y='Count', title='', color='State', height=500)
                st.plotly_chart(fig_state_all, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="graph-card"><h4>NPIs by Specialty</h4>', unsafe_allow_html=True)
                fig_specialty_all = px.pie(specialty_df, values='Count', names='Specialty', title='', height=400)
                st.plotly_chart(fig_specialty_all, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('<div class="graph-card"><h4>NPI Count by Region and Specialty</h4>', unsafe_allow_html=True)
                fig_heatmap = px.imshow(pivot_df, title='', 
                                        labels=dict(x="Region", y="Specialty", color="NPI Count"), 
                                        color_continuous_scale='Blues', height=500)
                st.plotly_chart(fig_heatmap, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        with tab3:
            st.markdown("<h3>NPI Activity by Time</h3>", unsafe_allow_html=True)
            
            time_counts = analyze_active_npis_by_time(npi_df)
            time_df = pd.DataFrame([{'Time': k, 'Active NPIs': v} for k, v in time_counts.items()])
            
            hour_counts = {}
            for time_str, count in time_counts.items():
                hour = int(time_str.split(':')[0])
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
            
            st.markdown('<div class="graph-card"><h4>Active NPIs Throughout the Day</h4>', unsafe_allow_html=True)
            fig_time = px.line(time_df, x='Time', y='Active NPIs', title='', markers=True, height=400)
            st.plotly_chart(fig_time, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="graph-card"><h4>Active NPIs by Hour and Day</h4>', unsafe_allow_html=True)
            fig_heatmap = px.imshow(hour_matrix, labels=dict(x="Hour of Day", y="Day of Week", color="Active NPIs"),
                                    x=hours, y=days, title='', color_continuous_scale='Viridis', height=400)
            st.plotly_chart(fig_heatmap, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="graph-card"><h4>Distribution of Login Hours</h4>', unsafe_allow_html=True)
                fig_login = px.bar(login_hour_counts, x='Hour', y='Count', title='', height=400)
                st.plotly_chart(fig_login, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="graph-card"><h4>Distribution of Logout Hours</h4>', unsafe_allow_html=True)
                fig_logout = px.bar(logout_hour_counts, x='Hour', y='Count', title='', height=400)
                st.plotly_chart(fig_logout, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
