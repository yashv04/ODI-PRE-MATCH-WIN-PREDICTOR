import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datetime import datetime, timedelta
from collections import defaultdict
import pickle
import warnings
warnings.filterwarnings("ignore")

def load_and_clean_data():
    """Load and clean the cricket match data"""
    print("Loading data...")
    matches_df = pd.read_csv('matches_updated_mens_odi_upto_feb_2025.csv')
    deliveries_df = pd.read_csv('deliveries_updated_mens_odi_upto_feb_2025.csv')
    
    # Data cleaning
    matches_df = matches_df.dropna(subset=['winner', 'toss_winner', 'toss_decision'])
    print(f"Loaded {len(matches_df)} matches and {len(deliveries_df)} deliveries")
    
    return matches_df, deliveries_df

def create_target_variable(matches_df):
    """Create target variable for prediction"""
    def create_label(row):
        return 1 if row['winner'] == row['team1'] else 0
    
    matches_df['winner_label'] = matches_df.apply(create_label, axis=1)
    return matches_df

def calculate_team_ratings(matches_df):
    """Calculate Elo-style team ratings"""
    print("Calculating team ratings...")
    
    # Convert date and add time features
    matches_df['date'] = pd.to_datetime(matches_df['date'])
    matches_df['year'] = matches_df['date'].dt.year
    matches_df['month'] = matches_df['date'].dt.month
    
    team_ratings = defaultdict(lambda: 1000)
    venue_advantages = defaultdict(lambda: defaultdict(int))
    K = 32
    
    rating_history = []
    
    for idx, row in matches_df.sort_values("date").iterrows():
        team1, team2 = row['team1'], row['team2']
        winner = row['winner']
        venue = row['venue']
        
        # Calculate venue advantage
        if venue in venue_advantages[team1]:
            home_bonus = venue_advantages[team1][venue] * 10
        else:
            home_bonus = 50 if not row.get('neutralvenue', True) else 0
        
        R1 = team_ratings[team1] + home_bonus
        R2 = team_ratings[team2]
        
        # Expected probabilities
        E1 = 1 / (1 + 10 ** ((R2 - R1) / 400))
        E2 = 1 - E1
        
        # Actual results
        A1 = 1 if winner == team1 else 0
        A2 = 1 - A1
        
        # Update ratings
        team_ratings[team1] += K * (A1 - E1)
        team_ratings[team2] += K * (A2 - E2)
        
        # Update venue advantages
        if A1 == 1:
            venue_advantages[team1][venue] += 1
            venue_advantages[team2][venue] -= 1
        else:
            venue_advantages[team2][venue] += 1
            venue_advantages[team1][venue] -= 1
        
        rating_history.append({
            'matchId': row['matchId'],
            'date': row['date'],
            'team1': team1,
            'team2': team2,
            'team1_rating': team_ratings[team1],
            'team2_rating': team_ratings[team2]
        })
    
    return pd.DataFrame(rating_history)

def calculate_player_ratings(deliveries_df, matches_df):
    """Calculate batting and bowling ratings for players"""
    print("Calculating player ratings...")
    
    # Batting stats
    batting_stats = deliveries_df.groupby(['matchId', 'batsman']).agg(
        runs=('batsman_runs', 'sum'),
        balls=('ball', 'count'),
        fours=('batsman_runs', lambda x: (x == 4).sum()),
        sixes=('batsman_runs', lambda x: (x == 6).sum())
    ).reset_index()
    
    batting_stats['strike_rate'] = batting_stats['runs'] / batting_stats['balls'] * 100
    batting_stats['batting_rating'] = (
        batting_stats['runs'] * 1.5 +
        batting_stats['strike_rate'] * 0.3 +
        batting_stats['fours'] * 2 +
        batting_stats['sixes'] * 4
    )
    
    # Bowling stats
    deliveries_df['is_wicket'] = deliveries_df['player_dismissed'].notna().astype(int)
    deliveries_df['total_runs'] = deliveries_df['batsman_runs'] + deliveries_df['extras']
    
    bowling_stats = deliveries_df.groupby(['matchId', 'bowler']).agg(
        runs_conceded=('total_runs', 'sum'),
        balls=('ball', 'count'),
        wickets=('is_wicket', 'sum'),
        maidens=('over', lambda x: (deliveries_df.loc[x.index].groupby('over')['total_runs'].sum() == 0).sum())
    ).reset_index()
    
    bowling_stats['overs'] = bowling_stats['balls'] / 6
    bowling_stats['economy'] = bowling_stats['runs_conceded'] / bowling_stats['overs']
    bowling_stats['bowling_rating'] = (
        bowling_stats['wickets'] * 25 -
        bowling_stats['economy'] * 4 +
        bowling_stats['maidens'] * 5
    )
    
    # Create rating history with rolling averages
    def create_rating_history(stats_df, player_col, rating_col, window=15):
        ratings = {}
        for player in stats_df[player_col].unique():
            player_df = stats_df[stats_df[player_col] == player].sort_values('matchId')
            player_df[rating_col] = player_df[rating_col].rolling(window=window, min_periods=3).mean()
            ratings[player] = player_df[['matchId', player_col, rating_col]]
        return pd.concat(ratings.values()).reset_index(drop=True)
    
    batting_rating_df = create_rating_history(batting_stats, 'batsman', 'batting_rating')
    bowling_rating_df = create_rating_history(bowling_stats, 'bowler', 'bowling_rating')
    
    # Merge with match dates
    batting_rating_df = batting_rating_df.merge(matches_df[['matchId', 'date']], on='matchId')
    bowling_rating_df = bowling_rating_df.merge(matches_df[['matchId', 'date']], on='matchId')
    
    return batting_rating_df, bowling_rating_df

def get_playing_xis(deliveries_df):
    """Extract playing XIs from deliveries data"""
    print("Extracting playing XIs...")
    
    playing_xis = []
    for match_id in deliveries_df['matchId'].unique():
        match_deliveries = deliveries_df[deliveries_df['matchId'] == match_id]
        
        teams = match_deliveries['batting_team'].unique()
        if len(teams) < 2:
            continue
        
        team1, team2 = teams[0], teams[1]
        
        team1_players = list(set(
            list(match_deliveries[match_deliveries['batting_team'] == team1]['batsman']) +
            list(match_deliveries[match_deliveries['bowling_team'] == team1]['bowler'])
        ))[:11]
        
        team2_players = list(set(
            list(match_deliveries[match_deliveries['batting_team'] == team2]['batsman']) +
            list(match_deliveries[match_deliveries['bowling_team'] == team2]['bowler'])
        ))[:11]
        
        playing_xis.append({
            'matchId': match_id,
            'team1': team1,
            'team2': team2,
            'team1_players': team1_players,
            'team2_players': team2_players
        })
    
    return pd.DataFrame(playing_xis)

def calculate_team_player_ratings(matches_df, batting_rating_df, bowling_rating_df):
    """Calculate average team batting and bowling ratings"""
    print("Calculating team player ratings...")
    
    def get_latest_rating(player, date, rating_df, player_col, rating_col):
        player_ratings = rating_df[rating_df[player_col] == player]
        past_ratings = player_ratings[player_ratings['date'] < date]
        if past_ratings.empty:
            return 50 if 'batting' in rating_col else 10
        return past_ratings.iloc[-1][rating_col]
    
    team1_bat_ratings = []
    team1_bowl_ratings = []
    team2_bat_ratings = []
    team2_bowl_ratings = []
    
    for _, row in matches_df.iterrows():
        date = row['date']
        team1_players = row['team1_players']
        team2_players = row['team2_players']
        
        # Get ratings for each team
        team1_bat = [get_latest_rating(p, date, batting_rating_df, 'batsman', 'batting_rating')
                     for p in team1_players]
        team1_bowl = [get_latest_rating(p, date, bowling_rating_df, 'bowler', 'bowling_rating')
                      for p in team1_players]
        
        team2_bat = [get_latest_rating(p, date, batting_rating_df, 'batsman', 'batting_rating')
                     for p in team2_players]
        team2_bowl = [get_latest_rating(p, date, bowling_rating_df, 'bowler', 'bowling_rating')
                      for p in team2_players]
        
        # Use top players ratings
        team1_bat_sorted = sorted([r for r in team1_bat if r > 0], reverse=True)[:7]
        team1_bowl_sorted = sorted([r for r in team1_bowl if r > 0], reverse=True)[:5]
        team2_bat_sorted = sorted([r for r in team2_bat if r > 0], reverse=True)[:7]
        team2_bowl_sorted = sorted([r for r in team2_bowl if r > 0], reverse=True)[:5]
        
        team1_bat_ratings.append(np.mean(team1_bat_sorted) if team1_bat_sorted else 50)
        team1_bowl_ratings.append(np.mean(team1_bowl_sorted) if team1_bowl_sorted else 10)
        team2_bat_ratings.append(np.mean(team2_bat_sorted) if team2_bat_sorted else 50)
        team2_bowl_ratings.append(np.mean(team2_bowl_sorted) if team2_bowl_sorted else 10)
    
    return team1_bat_ratings, team1_bowl_ratings, team2_bat_ratings, team2_bowl_ratings

def create_features(matches_df, team_rating_df):
    """Create all features for the model"""
    print("Creating features...")
    
    # Basic features
    matches_df['team1_won_toss'] = (matches_df['toss_winner'] == matches_df['team1']).astype(int)
    matches_df['team1_bats_first'] = (
        (matches_df['toss_winner'] == matches_df['team1']) &
        (matches_df['toss_decision'] == 'bat')
    ).astype(int)
    
    # Venue encoding
    venue_encoder = LabelEncoder()
    matches_df['venue_encoded'] = venue_encoder.fit_transform(matches_df['venue'])
    
    # Merge team ratings
    matches_df = matches_df.merge(team_rating_df, on=['matchId', 'date', 'team1', 'team2'], how='left')
    
    # Rating differences
    matches_df['rating_diff'] = matches_df['team1_rating'] - matches_df['team2_rating']
    matches_df['batting_rating_diff'] = matches_df['team1_avg_bat_rating'] - matches_df['team2_avg_bat_rating']
    matches_df['bowling_rating_diff'] = matches_df['team1_avg_bowl_rating'] - matches_df['team2_avg_bowl_rating']
    
    # Form calculation
    def compute_weighted_form(team, date, window=10, decay=0.9):
        recent_matches = matches_df[
            ((matches_df['team1'] == team) | (matches_df['team2'] == team)) &
            (matches_df['date'] < date)
        ].sort_values('date').tail(window)
        
        if recent_matches.empty:
            return 0.5
        
        wins = recent_matches[recent_matches['winner'] == team]
        weighted_wins = sum([decay**i for i in range(len(wins))])
        weighted_total = sum([decay**i for i in range(len(recent_matches))])
        
        return weighted_wins / weighted_total if weighted_total > 0 else 0.5
    
    matches_df['team1_recent_form'] = matches_df.apply(
        lambda row: compute_weighted_form(row['team1'], row['date']), axis=1
    )
    matches_df['team2_recent_form'] = matches_df.apply(
        lambda row: compute_weighted_form(row['team2'], row['date']), axis=1
    )
    matches_df['form_diff'] = matches_df['team1_recent_form'] - matches_df['team2_recent_form']
    
    # Home advantage
    matches_df['neutralvenue'] = matches_df['neutralvenue'].fillna(True).astype(bool)
    matches_df['team1_home'] = (~matches_df['neutralvenue']).astype(int)
    
    # Time features
    matches_df['month_sin'] = np.sin(2 * np.pi * matches_df['month'] / 12)
    matches_df['month_cos'] = np.cos(2 * np.pi * matches_df['month'] / 12)
    
    # Head-to-head record
    def calculate_h2h_record(team1, team2, date):
        h2h_matches = matches_df[
            (((matches_df['team1'] == team1) & (matches_df['team2'] == team2)) |
             ((matches_df['team1'] == team2) & (matches_df['team2'] == team1))) &
            (matches_df['date'] < date)
        ].tail(20)
        
        if h2h_matches.empty:
            return 0.5
        
        team1_wins = h2h_matches[matches_df['winner'] == team1].shape[0]
        return team1_wins / len(h2h_matches)
    
    matches_df['h2h_advantage'] = matches_df.apply(
        lambda row: calculate_h2h_record(row['team1'], row['team2'], row['date']), axis=1
    )
    
    return matches_df, venue_encoder

def train_models(matches_df):
    """Train and evaluate multiple models"""
    print("Training models...")
    
    # Train-test split
    matches_df['date'] = pd.to_datetime(matches_df['date'])
    train_df = matches_df[matches_df['date'].dt.year <= 2023]
    test_df = matches_df[matches_df['date'].dt.year >= 2024]
    
    # Feature columns
    feature_cols = [
        'team1_won_toss', 'team1_bats_first', 'venue_encoded',
        'team1_rating', 'team2_rating', 'rating_diff',
        'batting_rating_diff', 'bowling_rating_diff',
        'form_diff', 'team1_home', 'h2h_advantage',
        'month_sin', 'month_cos'
    ]
    
    target_col = 'winner_label'
    
    # Prepare data
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df[target_col]
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df[target_col]
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            class_weight='balanced'
        ),
        'Logistic Regression': LogisticRegression(
            C=1.0,
            class_weight='balanced',
            random_state=42,
            max_iter=1000
        )
    }
    
    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        
        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'accuracy': acc,
            'f1_score': f1,
            'predictions': y_pred
        }
        
        print(f"{name} - Accuracy: {acc:.3f}, F1 Score: {f1:.3f}")
    
    # Select best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
    best_model = results[best_model_name]['model']
    
    print(f"\nBest Model: {best_model_name}")
    print(f"Best Accuracy: {results[best_model_name]['accuracy']:.3f}")
    
    return best_model, best_model_name, scaler, feature_cols

def save_model_artifacts(model, model_name, scaler, venue_encoder, feature_cols, 
                        team_rating_df, batting_rating_df, bowling_rating_df, matches_df):
    """Save all model artifacts"""
    print("Saving model artifacts...")
    
    # Save model
    with open('cricket_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save scaler (if used)
    if model_name == 'Logistic Regression':
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
    
    # Save encoders and other artifacts
    with open('venue_encoder.pkl', 'wb') as f:
        pickle.dump(venue_encoder, f)
    
    with open('feature_cols.pkl', 'wb') as f:
        pickle.dump(feature_cols, f)
    
    # Save rating dataframes
    team_rating_df.to_pickle('team_ratings.pkl')
    batting_rating_df.to_pickle('batting_ratings.pkl')
    bowling_rating_df.to_pickle('bowling_ratings.pkl')
    
    # Save matches dataframe for reference
    matches_df.to_pickle('matches_processed.pkl')
    
    # Save model metadata
    metadata = {
        'model_name': model_name,
        'feature_cols': feature_cols,
        'uses_scaler': model_name == 'Logistic Regression'
    }
    
    with open('model_metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    print("Model artifacts saved successfully!")

def main():
    """Main training pipeline"""
    print("Starting cricket match prediction model training...")
    
    # Load data
    matches_df, deliveries_df = load_and_clean_data()
    
    # Create target variable
    matches_df = create_target_variable(matches_df)
    
    # Calculate team ratings
    team_rating_df = calculate_team_ratings(matches_df)
    
    # Calculate player ratings
    batting_rating_df, bowling_rating_df = calculate_player_ratings(deliveries_df, matches_df)
    
    # Get playing XIs
    playing_xis_df = get_playing_xis(deliveries_df)
    matches_df = matches_df.merge(playing_xis_df[['matchId', 'team1_players', 'team2_players']], 
                                  on='matchId', how='left')
    
    # Filter matches with proper playing XI data
    matches_df = matches_df[matches_df['team1_players'].apply(lambda x: isinstance(x, list) and len(x) > 5)]
    matches_df = matches_df[matches_df['team2_players'].apply(lambda x: isinstance(x, list) and len(x) > 5)]
    
    # Calculate team player ratings
    team1_bat, team1_bowl, team2_bat, team2_bowl = calculate_team_player_ratings(
        matches_df, batting_rating_df, bowling_rating_df
    )
    
    matches_df['team1_avg_bat_rating'] = team1_bat
    matches_df['team1_avg_bowl_rating'] = team1_bowl
    matches_df['team2_avg_bat_rating'] = team2_bat
    matches_df['team2_avg_bowl_rating'] = team2_bowl
    
    # Create features
    matches_df, venue_encoder = create_features(matches_df, team_rating_df)
    
    # Train models
    best_model, model_name, scaler, feature_cols = train_models(matches_df)
    
    # Save artifacts
    save_model_artifacts(best_model, model_name, scaler, venue_encoder, feature_cols,
                        team_rating_df, batting_rating_df, bowling_rating_df, matches_df)
    
    print("\nTraining completed successfully!")
    print("Model artifacts saved and ready for deployment.")

if __name__ == "__main__":
    main()
