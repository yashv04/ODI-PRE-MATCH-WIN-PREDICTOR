import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="Cricket Match Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    .team-card {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 10px 0;
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header"> Cricket Match Predictor</h1>', unsafe_allow_html=True)

# Load pre-trained model and data (you'll need to upload these)
@st.cache_data
def load_data():
    try:
        # Load your trained model, scaler, and encoders
        # For demo purposes, we'll create dummy data
        teams = ['India', 'England', 'Australia', 'Pakistan', 'South Africa', 
                'New Zealand', 'West Indies', 'Sri Lanka', 'Bangladesh', 'Afghanistan']
        
        venues = ['Wankhede Stadium', 'Lord\'s', 'MCG', 'Eden Gardens', 'The Oval',
                 'SCG', 'Old Trafford', 'Headingley', 'Wanderers', 'Cape Town']
        
        # Sample player data
        players_data = {
            'India': ['RG Sharma', 'Shubman Gill', 'V Kohli', 'SS Iyer', 'KL Rahul', 
                     'HH Pandya', 'RA Jadeja', 'Kuldeep Yadav', 'JJ Bumrah', 
                     'Mohammed Shami', 'Mohammed Siraj'],
            'England': ['PD Salt', 'BM Duckett', 'JE Root', 'JC Buttler', 'HC Brook',
                       'LS Livingstone', 'MM Ali', 'AU Rashid', 'MA Wood', 
                       'CR Woakes', 'BA Stokes'],
            # Add more teams as needed
        }
        
        return teams, venues, players_data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return [], [], {}

# Mock prediction function (replace with your actual model)
def predict_match_outcome(match_data):
    """
    Replace this with your actual prediction logic from the notebook
    """
    # Dummy prediction for demonstration
    team1_prob = np.random.uniform(30, 70)
    team2_prob = 100 - team1_prob
    
    return {
        match_data['team1']: round(team1_prob, 2),
        match_data['team2']: round(team2_prob, 2)
    }

# Load data
teams, venues, players_data = load_data()

# Sidebar for match input
st.sidebar.header(" Match Configuration")

with st.sidebar:
    st.subheader("Teams")
    team1 = st.selectbox("Select Team 1", teams, index=0 if teams else 0)
    team2 = st.selectbox("Select Team 2", [t for t in teams if t != team1], 
                        index=0 if teams else 0)
    
    st.subheader("Match Details")
    venue = st.selectbox("Venue", venues if venues else ["Default Venue"])
    match_date = st.date_input("Match Date", datetime.now().date())
    
    neutral_venue = st.checkbox("Neutral Venue", value=False)
    
    st.subheader("Toss Details")
    toss_winner = st.selectbox("Toss Winner", [team1, team2])
    toss_decision = st.selectbox("Toss Decision", ["bat", "field"])

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header(" Match Prediction")
    
    if st.button("Predict Match Outcome", type="primary", use_container_width=True):
        # Prepare match data
        match_data = {
            'team1': team1,
            'team2': team2,
            'venue': venue,
            'date': match_date.strftime('%Y-%m-%d'),
            'neutralvenue': neutral_venue,
            'toss_winner': toss_winner,
            'toss_decision': toss_decision
        }
        
        # Get prediction
        with st.spinner("Analyzing match conditions..."):
            prediction = predict_match_outcome(match_data)
        
        # Display results
        st.subheader("Prediction Results")
        
        # Create probability visualization
        fig = go.Figure(data=[
            go.Bar(name=team1, x=[team1], y=[prediction[team1]], 
                  marker_color='#1f77b4'),
            go.Bar(name=team2, x=[team2], y=[prediction[team2]], 
                  marker_color='#ff7f0e')
        ])
        
        fig.update_layout(
            title="Win Probability",
            yaxis_title="Probability (%)",
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display prediction boxes
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown(f"""
            <div class="prediction-box">
                <h3>{team1}</h3>
                <h2>{prediction[team1]}%</h2>
                <p>Win Probability</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_b:
            st.markdown(f"""
            <div class="prediction-box">
                <h3>{team2}</h3>
                <h2>{prediction[team2]}%</h2>
                <p>Win Probability</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Match insights
        st.subheader("🔍 Match Insights")
        
        insights_col1, insights_col2 = st.columns(2)
        
        with insights_col1:
            st.info(f"**Toss Impact**: {toss_winner} won the toss and chose to {toss_decision}")
            st.info(f"**Venue**: {'Neutral' if neutral_venue else 'Home advantage possible'}")
        
        with insights_col2:
            favorite = team1 if prediction[team1] > prediction[team2] else team2
            margin = abs(prediction[team1] - prediction[team2])
            
            if margin > 20:
                confidence = "High"
            elif margin > 10:
                confidence = "Medium"
            else:
                confidence = "Low"
            
            st.success(f"**Favorite**: {favorite}")
            st.success(f"**Confidence**: {confidence}")

with col2:
    st.header("📈 Statistics")
    
    # Team comparison metrics (dummy data for demo)
    st.subheader("Team Comparison")
    
    metrics = {
        'Team Rating': [np.random.randint(800, 1200), np.random.randint(800, 1200)],
        'Recent Form': [np.random.uniform(0.3, 0.8), np.random.uniform(0.3, 0.8)],
        'H2H Record': [np.random.uniform(0.2, 0.8), np.random.uniform(0.2, 0.8)]
    }
    
    comparison_df = pd.DataFrame(metrics, index=[team1, team2])
    
    for metric in metrics.keys():
        st.metric(
            label=f"{metric} - {team1}",
            value=f"{comparison_df.loc[team1, metric]:.2f}"
        )
        st.metric(
            label=f"{metric} - {team2}",
            value=f"{comparison_df.loc[team2, metric]:.2f}"
        )
    
    # Historical performance chart
    st.subheader("Historical Performance")
    
    # Generate dummy historical data
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='M')
    team1_performance = np.random.uniform(0.4, 0.8, len(dates))
    team2_performance = np.random.uniform(0.4, 0.8, len(dates))
    
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(x=dates, y=team1_performance, 
                                 name=team1, line=dict(color='#1f77b4')))
    fig_hist.add_trace(go.Scatter(x=dates, y=team2_performance, 
                                 name=team2, line=dict(color='#ff7f0e')))
    
    fig_hist.update_layout(
        title="Win Rate Over Time",
        xaxis_title="Date",
        yaxis_title="Win Rate",
        height=300
    )
    
    st.plotly_chart(fig_hist, use_container_width=True)

# Additional features
st.header("Additional Features")

tab1, tab2, tab3 = st.tabs(["Team Analysis", "Player Statistics", "Venue Stats"])

with tab1:
    st.subheader("Team Performance Analysis")
    
    selected_team = st.selectbox("Select Team for Analysis", teams)
    
    # Create dummy team performance data
    performance_data = {
        'Matches Played': np.random.randint(50, 100),
        'Matches Won': np.random.randint(25, 60),
        'Win Rate': np.random.uniform(0.5, 0.8),
        'Average Score': np.random.randint(250, 320),
        'Highest Score': np.random.randint(350, 450)
    }
    
    col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
    
    with col_metrics1:
        st.metric("Matches Played", performance_data['Matches Played'])
        st.metric("Matches Won", performance_data['Matches Won'])
    
    with col_metrics2:
        st.metric("Win Rate", f"{performance_data['Win Rate']:.2%}")
        st.metric("Average Score", performance_data['Average Score'])
    
    with col_metrics3:
        st.metric("Highest Score", performance_data['Highest Score'])

with tab2:
    st.subheader("Player Statistics")
    
    if team1 in players_data:
        st.write(f"**{team1} Squad:**")
        players_df = pd.DataFrame({
            'Player': players_data[team1],
            'Batting Avg': np.random.uniform(25, 55, len(players_data[team1])),
            'Strike Rate': np.random.uniform(80, 130, len(players_data[team1])),
            'Bowling Avg': np.random.uniform(20, 40, len(players_data[team1]))
        })
        st.dataframe(players_df, use_container_width=True)

with tab3:
    st.subheader("Venue Statistics")
    
    venue_stats = pd.DataFrame({
        'Venue': venues[:5] if venues else ['Default'],
        'Matches': np.random.randint(20, 100, 5),
        'Avg Score': np.random.randint(240, 320, 5),
        'Toss Win %': np.random.uniform(0.4, 0.6, 5)
    })
    
    st.dataframe(venue_stats, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
     Cricket Match Predictor | Built with Streamlit | 
    Powered by Machine Learning
</div>
""", unsafe_allow_html=True)
