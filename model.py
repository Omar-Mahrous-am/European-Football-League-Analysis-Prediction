# model.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Football League Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS with background and styling
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)), 
                    url('https://i.pinimg.com/1200x/33/42/59/334259bffb9313a70c369a208af8acbb.jpg');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    
    .main-header {
        text-align: center;
        color: #fad390;
        font-size: 3.5rem;
        font-weight: 800;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.8);
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        text-align: center;
        color: #82ccdd;
        font-size: 1.5rem;
        font-weight: 300;
        margin-bottom: 2rem;
    }
    
    .prediction-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .points-display {
        font-size: 4rem;
        font-weight: 800;
        color: #fad390;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    .confidence-display {
        font-size: 1.5rem;
        color: #82ccdd;
        text-align: center;
    }
    
    .team-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #fad390;
    }
    
    .stButton>button {
        background: linear-gradient(45deg, #e55039, #eb2f06);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-size: 1.2rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(229, 80, 57, 0.6);
    }
</style>
""", unsafe_allow_html=True)

# Load your trained model
@st.cache_resource
def load_model():
    try:
        with open('football_model.pkl', 'rb') as file:
            model_data = pickle.load(file)
        return model_data
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Team and league data with historical averages
leagues_teams = {
    "French Ligue 1": ["PSG", "Marseille", "Lyon", "Monaco", "Lille", "Nice", "Rennes", "Lens"],
    "English Premier League": ["Man City", "Liverpool", "Chelsea", "Arsenal", "Man United", "Tottenham", "Newcastle", "West Ham"],
    "Spanish La Liga": ["Real Madrid", "Barcelona", "Atletico", "Sevilla", "Valencia", "Villarreal", "Real Sociedad", "Athletic Bilbao"],
    "German Bundesliga": ["Bayern Munich", "Dortmund", "Leipzig", "Leverkusen", "Frankfurt", "Wolfsburg", "Gladbach", "Union Berlin"],
    "Italian Serie A": ["Inter Milan", "Juventus", "AC Milan", "Napoli", "Roma", "Lazio", "Atalanta", "Fiorentina"]
}

# Historical performance data (you can replace this with your actual training data)
team_historical_data = {
    "PSG": {"GP": 38, "W": 27, "D": 4, "L": 7, "GF": 89, "GA": 40},
    "Marseille": {"GP": 38, "W": 22, "D": 7, "L": 9, "GF": 67, "GA": 40},
    "Lyon": {"GP": 38, "W": 18, "D": 8, "L": 12, "GF": 65, "GA": 47},
    "Man City": {"GP": 38, "W": 28, "D": 5, "L": 5, "GF": 94, "GA": 33},
    "Liverpool": {"GP": 38, "W": 24, "D": 8, "L": 6, "GF": 85, "GA": 38},
    "Real Madrid": {"GP": 38, "W": 26, "D": 6, "L": 6, "GF": 80, "GA": 31},
    "Barcelona": {"GP": 38, "W": 25, "D": 7, "L": 6, "GF": 76, "GA": 38},
    "Bayern Munich": {"GP": 34, "W": 24, "D": 5, "L": 5, "GF": 92, "GA": 38},
    "Inter Milan": {"GP": 38, "W": 25, "D": 8, "L": 5, "GF": 77, "GA": 32}
}

# For teams not in historical data, use league averages
league_averages = {
    "French Ligue 1": {"GP": 38, "W": 18, "D": 8, "L": 12, "GF": 62, "GA": 48},
    "English Premier League": {"GP": 38, "W": 19, "D": 9, "L": 10, "GF": 68, "GA": 45},
    "Spanish La Liga": {"GP": 38, "W": 20, "D": 7, "L": 11, "GF": 65, "GA": 42},
    "German Bundesliga": {"GP": 34, "W": 16, "D": 7, "L": 11, "GF": 60, "GA": 46},
    "Italian Serie A": {"GP": 38, "W": 19, "D": 8, "L": 11, "GF": 59, "GA": 41}
}

team_colors = {
    "PSG": "#004170", "Marseille": "#00A0E3", "Lyon": "#BF0D3E", "Monaco": "#E2001A", "Lille": "#ED145B",
    "Man City": "#6CABDD", "Liverpool": "#C8102E", "Chelsea": "#034694", "Arsenal": "#EF0107", "Man United": "#DA291C",
    "Real Madrid": "#FFFFFF", "Barcelona": "#A50044", "Atletico": "#CB3524", "Sevilla": "#D00D2B", "Valencia": "#F9A01B",
    "Bayern Munich": "#DC052D", "Dortmund": "#FDE100", "Leipzig": "#DD0C2B", "Leverkusen": "#E32221", "Frankfurt": "#E20000",
    "Inter Milan": "#0068A8", "Juventus": "#000000", "AC Milan": "#FB090B", "Napoli": "#12A0D7", "Roma": "#8B1A3D"
}

def get_team_features(team, league):
    """Get historical features for a team, fall back to league averages if team not found"""
    if team in team_historical_data:
        return team_historical_data[team]
    else:
        return league_averages[league]

def predict_with_model(model_data, team, league):
    try:
        model = model_data['model']
        scaler = model_data['scaler']
        feature_columns = model_data['feature_columns']
        
        # Get historical features for the team
        features = get_team_features(team, league)
        features['GD'] = features['GF'] - features['GA']
        
        # Create input dataframe
        input_data = {col: 0 for col in feature_columns}
        
        # Fill numerical features
        for feature in ['GP', 'W', 'D', 'L', 'GF', 'GA', 'GD']:
            if feature in input_data:
                input_data[feature] = features[feature]
        
        # Fill league one-hot encoding
        league_col = f"League_{league}"
        if league_col in input_data:
            input_data[league_col] = 1
        
        # Convert to dataframe
        input_df = pd.DataFrame([input_data])
        
        # Scale features
        scaled_features = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(scaled_features)[0]
        
        return max(0, round(prediction)), features
    
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

def create_performance_chart(features, predicted_points, team):
    metrics = ['Wins', 'Draws', 'Losses', 'Goals For', 'Goals Against']
    values = [features['W'], features['D'], features['L'], features['GF'], features['GA']]
    
    fig = go.Figure(data=[
        go.Bar(name='Performance', x=metrics, y=values, 
               marker_color=['#27ae60', '#f39c12', '#e74c3c', '#3498db', '#e55039'])
    ])
    
    fig.update_layout(
        title=f'{team} - Historical Performance',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        showlegend=False,
        height=300
    )
    
    return fig

# Load model
model_data = load_model()

# App layout
st.markdown('<div class="main-header">⚽ EUROPEAN FOOTBALL LEAGUE PREDICTOR</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Next Season Prediction Based on Historical Data</div>', unsafe_allow_html=True)

if model_data is None:
    st.error("❌ Could not load the model. Please make sure 'football_model.pkl' is in the same directory.")
else:
    st.success("✅ Model loaded successfully! Select your team below.")
    
    # Main content
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        
        # League selection
        selected_league = st.selectbox(
            "🏆 SELECT LEAGUE",
            options=list(leagues_teams.keys()),
            index=0
        )
        
        # Team selection
        if selected_league:
            selected_team = st.selectbox(
                "👥 SELECT TEAM",
                options=leagues_teams[selected_league],
                index=0
            )
        
        # Predict button
        if st.button("🚀 PREDICT NEXT SEASON POINTS", use_container_width=True):
            if selected_team:
                predicted_points, historical_features = predict_with_model(model_data, selected_team, selected_league)
                
                if predicted_points is not None:
                    # Store in session state
                    st.session_state.prediction = {
                        'points': predicted_points,
                        'team': selected_team,
                        'league': selected_league,
                        'historical_features': historical_features,
                        'timestamp': datetime.now()
                    }
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Team info
        if selected_team:
            historical_data = get_team_features(selected_team, selected_league)
            st.markdown(f"""
            <div class="team-card">
                <h3>🏅 {selected_team}</h3>
                <p><strong>League:</strong> {selected_league}</p>
                <p><strong>Historical Stats (Last Season):</strong></p>
                <p>W: {historical_data['W']} | D: {historical_data['D']} | L: {historical_data['L']}</p>
                <p>GF: {historical_data['GF']} | GA: {historical_data['GA']}</p>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        if 'prediction' in st.session_state:
            pred = st.session_state.prediction
            
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            
            # Main prediction display
            st.markdown(f'<div class="points-display">{pred["points"]}</div>', unsafe_allow_html=True)
            st.markdown('<div class="confidence-display">Predicted Points for Next Season</div>', unsafe_allow_html=True)
            
            # Performance chart
            fig = create_performance_chart(pred['historical_features'], pred['points'], pred['team'])
            st.plotly_chart(fig, use_container_width=True)
            
            # Additional insights
            st.markdown("### 📈 Prediction Analysis")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                win_rate = (pred['historical_features']['W'] / pred['historical_features']['GP']) * 100
                st.metric("Historical Win Rate", f"{win_rate:.1f}%")
            
            with col2:
                points_per_game = pred['points'] / pred['historical_features']['GP']
                st.metric("Projected PPG", f"{points_per_game:.2f}")
            
            with col3:
                goals_per_game = pred['historical_features']['GF'] / pred['historical_features']['GP']
                st.metric("Goals/Game", f"{goals_per_game:.2f}")
            
            with col4:
                if pred['points'] > 80:
                    qualification = "Champions League"
                elif pred['points'] > 65:
                    qualification = "Europa League"
                else:
                    qualification = "Domestic"
                st.metric("Expected", qualification)
            
            # Model info
            st.markdown("---")
            st.markdown("### ℹ️ Model Information")
            st.write(f"**Model Type:** Linear Regression")
            st.write(f"**Features Used:** Historical performance data")
            st.write(f"**Prediction Time:** {pred['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            # Placeholder when no prediction
            st.markdown("""
            <div class="prediction-card" style="text-align: center; padding: 4rem;">
                <h2 style="color: #82ccdd; margin-bottom: 1rem;">🎯 Ready for AI Prediction</h2>
                <p style="color: #ffffff; opacity: 0.8;">Select your league and team, then click predict to see next season's projection!</p>
                <div style="font-size: 5rem; margin: 2rem 0;">🤖</div>
                <p style="color: #fad390; font-size: 1.1rem;">Using historical data and machine learning</p>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: rgba(255,255,255,0.6);">
    <p>© 2024 Football Analytics Pro | Powered by Machine Learning & Historical Data</p>
    <p>No user input required - uses historical team performance data</p>
</div>
""", unsafe_allow_html=True)