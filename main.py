import fastf1
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
import warnings

warnings.filterwarnings('ignore')
logging.disable(logging.INFO)
logging.disable(logging.WARNING)

# Enable FastF1 caching
fastf1.Cache.enable_cache  # Create cache directory


class F1Predictor:
    def __init__(self):
        self.driver_encoder = LabelEncoder()
        self.team_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.position_predictor = None
        self.winner_predictor = None

    def load_race_data(self, year_start, year_end):
        """
        Loading match data from multiple seasons
        """
        all_races = []

        for year in range(year_start, year_end + 1):
            print(f"Loading {year} season data...")
            schedule = fastf1.get_event_schedule(year)

            for _, event in schedule.iterrows():
                if event['EventFormat'] == 'conventional' and event['EventDate'] < pd.Timestamp.now():
                    try:
                        session = fastf1.get_session(year, event['RoundNumber'], 'R')
                        session.load()

                        # Get the match results
                        results = session.results

                        # Get quali results
                        quali = fastf1.get_session(year, event['RoundNumber'], 'Q')
                        quali.load()
                        quali_results = quali.results

                        # Merging data
                        for _, driver_result in results.iterrows():
                            driver_code = driver_result['Abbreviation']

                            # Get quli results
                            quali_pos = None
                            if driver_code in quali_results['Abbreviation'].values:
                                quali_pos = \
                                quali_results[quali_results['Abbreviation'] == driver_code]['Position'].values[0]

                            race_data = {
                                'Year': year,
                                'Circuit': event['EventName'],
                                'CircuitID': event['Circuit']['CircuitName'] if 'Circuit' in event else event[
                                    'EventName'],
                                'Driver': driver_code,
                                'Team': driver_result['TeamName'],
                                'GridPosition': driver_result['GridPosition'],
                                'FinalPosition': driver_result['Position'],
                                'QualifyingPosition': quali_pos,
                                'Status': driver_result['Status']
                            }
                            all_races.append(race_data)


                    except Exception as e:
                        print(f"  跳过 {year} {event['EventName']}: {e}")
                        continue
                print(f"  Collected data for {event['EventName']} {year}")

        return pd.DataFrame(all_races)

    def prepare_features(self, df):
        """
        Prepare feature data
        """
        # Encoding categorical variables
        df['Driver_Encoded'] = self.driver_encoder.fit_transform(df['Driver'])
        df['Team_Encoded'] = self.team_encoder.fit_transform(df['Team'])
        df['Circuit_Encoded'] = pd.factorize(df['CircuitID'])[0]

        # Calculation features
        df_features = df.copy()

        # Calculate historical performance
        df_features['Avg_Position_Last_3'] = df_features.groupby('Driver')['FinalPosition'].transform(
            lambda x: x.rolling(3, min_periods=1).mean().shift(1)
        )

        df_features['Grid_Advantage'] = df_features['QualifyingPosition'] - df_features['GridPosition']

        # Handling missing values
        df_features = df_features.fillna(method='ffill').fillna(method='bfill')

        return df_features

    def train(self, df):
        """
        Training prediction model
        """
        # Preparation features
        df = self.prepare_features(df)

        # Feature columns
        feature_columns = ['Driver_Encoded', 'Team_Encoded', 'Circuit_Encoded',
                           'GridPosition', 'QualifyingPosition', 'Avg_Position_Last_3']

        # Filter valid data
        df = df.dropna(subset=feature_columns + ['FinalPosition'])

        X = df[feature_columns]
        y_position = df['FinalPosition']
        y_winner = (df['FinalPosition'] == 1).astype(int)

        # Standardization features
        X_scaled = self.scaler.fit_transform(X)

        # Data splitting
        X_train, X_test, y_pos_train, y_pos_test, y_win_train, y_win_test = train_test_split(
            X_scaled, y_position, y_winner, test_size=0.2, random_state=42
        )

        # Training the location prediction model
        self.position_predictor = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.position_predictor.fit(X_train, y_pos_train)

        # Training the winner prediction model
        self.winner_predictor = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        self.winner_predictor.fit(X_train, y_win_train)

        # Evaluation Model
        pos_pred = self.position_predictor.predict(X_test)
        win_pred = self.winner_predictor.predict(X_test)

        print("\n=== Model Evaluation ===")
        print(f"Position Prediction MAE: {mean_absolute_error(y_pos_test, pos_pred):.2f}")
        print(f"Position Prediction RMSE: {np.sqrt(mean_squared_error(y_pos_test, pos_pred)):.2f}")
        print(f"Winner prediction accuracy: {accuracy_score(y_win_test, win_pred):.2%}")

        # 特征重要性
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': self.position_predictor.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nFeature Importance:")
        print(feature_importance)

        return self

    def predict_race(self, year, round_number):
        """
        Predict the outcome of a specific match
        """
        try:
            # Load qual match data
            quali = fastf1.get_session(year, round_number, 'Q')
            quali.load()
            quali_results = quali.results

            # Load practice data for more features
            practice = fastf1.get_session(year, round_number, 'FP3')
            practice.load()

            # Prepare forecast data
            predictions = []

            for _, driver_data in quali_results.iterrows():
                driver_code = driver_data['Abbreviation']
                team_name = driver_data['TeamName']

                # Encoding features
                driver_encoded = self.driver_encoder.transform([driver_code])[0]
                team_encoded = self.team_encoder.transform([team_name])[0]

                # Using ranked match positions as a basis
                quali_pos = driver_data['Position']
                grid_pos = quali_pos  # 通常排位赛位置就是发车位置

                # Prepare feature vectors
                features = np.array([[
                    driver_encoded,
                    team_encoded,
                    0,  # Circuit encoded (Need to be set according to the track)
                    grid_pos,
                    quali_pos,
                    quali_pos  # Temporarily replace historical average with qualifying position
                ]])

                # 标准化
                features_scaled = self.scaler.transform(features)

                # 预测
                predicted_position = self.position_predictor.predict(features_scaled)[0]
                win_probability = self.winner_predictor.predict_proba(features_scaled)[0][1]

                predictions.append({
                    'Driver': driver_code,
                    'Team': team_name,
                    'Qualifying': quali_pos,
                    'Predicted_Position': round(predicted_position, 2),
                    'Win_Probability': round(win_probability * 100, 2)
                })

            # 排序
            predictions_df = pd.DataFrame(predictions)
            predictions_df = predictions_df.sort_values('Predicted_Position')
            predictions_df['Predicted_Rank'] = range(1, len(predictions_df) + 1)

            return predictions_df

        except Exception as e:
            print(f"Prediction failure: {e}")
            return None


def main():
    """
    Main function: Demonstrates the use of the F1 prediction system
    """
    # Create a predictor
    predictor = F1Predictor()

    print("=" * 60)
    print("F1oreSight is starting")
    print("=" * 60)

    # Load historical data
    print("\n1. Loading F1 historical data...")
    df = predictor.load_race_data(2024, 2024)
    print(f"loaded {len(df)} match records")

    # Training Model
    print("\n2. Training prediction model...")
    predictor.train(df)

    # Predict the next match
    print("\n3. Predict the next match...")
    # Example: Predict the first match of 2024
    prediction = predictor.predict_race(2024, 1)

    if prediction is not None:
        print("\n=== Match prediction results ===")
        print(prediction.to_string(index=False))

        # Showing the top 5 players with the highest probability of winning.
        print("\n=== Most likely to win driver ===")
        top_winners = prediction.nlargest(5, 'Win_Probability')[['Driver', 'Team', 'Win_Probability']]
        print(top_winners.to_string(index=False))


if __name__ == "__main__":
    main()