"""
Random Forest Movie Recommender System

This module implements a Random Forest-based movie recommendation system
using cleaned movie metadata, credits, keywords, and ratings data.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import ast
import re
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class MovieRecommenderConfig:
    """Configuration class for Random Forest Movie Recommender"""
    
    # Data file paths
    DATA_PATH = "/home/woto/Desktop/personal/python/thesis/cleaned_data/"
    MOVIES_FILE = "movies_metadata_cleaned.csv"
    CREDITS_FILE = "credits_cleaned.csv"
    KEYWORDS_FILE = "keywords_cleaned.csv"
    RATINGS_FILE = "ratings_cleaned.csv"
    
    # Random Forest parameters
    N_ESTIMATORS = 100
    MAX_DEPTH = 20
    MIN_SAMPLES_SPLIT = 5
    MIN_SAMPLES_LEAF = 2
    RANDOM_STATE = 42
    N_JOBS = -1
    
    # Feature engineering parameters
    MIN_MOVIE_RATINGS = 10  # Minimum ratings for a movie to be considered
    MIN_USER_RATINGS = 20   # Minimum ratings for a user to be considered
    MAX_USERS = 5000        # Maximum number of users to consider (for performance)
    MAX_ACTORS = 5          # Maximum number of lead actors to consider
    MAX_KEYWORDS = 10       # Maximum number of keywords to consider
    
    # Feature count optimization
    MAX_TOP_ACTORS = 25     # Reduced from 50 - most frequent actors to include
    MAX_TOP_DIRECTORS = 15  # Reduced from 30 - most frequent directors to include
    MAX_TFIDF_FEATURES = 30 # Reduced from 100 - TF-IDF keyword features
    
    # Model evaluation parameters
    TEST_SIZE = 0.2
    CV_FOLDS = 5
    
    # Feature selection
    USE_GENRES = True
    USE_ACTORS = True
    USE_DIRECTORS = True
    USE_KEYWORDS = True
    USE_MOVIE_FEATURES = True
    USE_USER_FEATURES = True


class DataProcessor:
    """Handles data loading and preprocessing"""
    
    def __init__(self, config: MovieRecommenderConfig):
        self.config = config
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load all data files"""
        print("Loading data files...")
        
        movies = pd.read_csv(f"{self.config.DATA_PATH}{self.config.MOVIES_FILE}")
        credits = pd.read_csv(f"{self.config.DATA_PATH}{self.config.CREDITS_FILE}")
        keywords = pd.read_csv(f"{self.config.DATA_PATH}{self.config.KEYWORDS_FILE}")
        ratings = pd.read_csv(f"{self.config.DATA_PATH}{self.config.RATINGS_FILE}")
        
        print(f"Loaded: {len(movies)} movies, {len(credits)} credits, {len(keywords)} keywords, {len(ratings)} ratings")
        return movies, credits, keywords, ratings
    
    def filter_reliable_data(self, ratings: pd.DataFrame) -> pd.DataFrame:
        """Filter ratings to include only reliable movies and active users"""
        print("Filtering reliable data...")
        
        # First, limit to first N users for performance
        unique_users = ratings['userId'].unique()
        selected_users = unique_users[:self.config.MAX_USERS]
        print(f"Limiting to first {len(selected_users)} users out of {len(unique_users)} total users")
        
        # Filter to selected users
        ratings = ratings[ratings['userId'].isin(selected_users)].copy()
        
        # Filter movies with minimum ratings
        movie_counts = ratings.groupby('movieId').size()
        reliable_movies = movie_counts[movie_counts >= self.config.MIN_MOVIE_RATINGS].index
        
        # Filter users with minimum ratings
        user_counts = ratings.groupby('userId').size()
        active_users = user_counts[user_counts >= self.config.MIN_USER_RATINGS].index
        
        # Apply filters
        filtered_ratings = ratings[
            (ratings['movieId'].isin(reliable_movies)) & 
            (ratings['userId'].isin(active_users))
        ].copy()
        
        print(f"Filtered to {len(filtered_ratings)} ratings from {len(filtered_ratings['userId'].unique())} users and {len(filtered_ratings['movieId'].unique())} movies")
        return filtered_ratings
    
    def parse_list_column(self, series: pd.Series, max_items: Optional[int] = None) -> pd.Series:
        """Parse string representations of lists and optionally limit items"""
        def safe_parse(x):
            if pd.isna(x) or x == '[]':
                return []
            try:
                parsed = ast.literal_eval(x)
                if max_items:
                    return parsed[:max_items]
                return parsed
            except:
                return []
        
        return series.apply(safe_parse)
    
    def merge_datasets(self, movies: pd.DataFrame, credits: pd.DataFrame, 
                      keywords: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
        """Merge all datasets into a single DataFrame"""
        print("Merging datasets...")
        
        # Start with ratings as base
        merged = ratings.copy()
        
        # Merge with movies
        merged = merged.merge(movies, left_on='movieId', right_on='id', how='left')
        
        # Merge with credits
        if self.config.USE_ACTORS or self.config.USE_DIRECTORS:
            merged = merged.merge(credits[['id', 'lead_actors', 'directors', 'cast_size']], 
                                left_on='movieId', right_on='id', how='left', suffixes=('', '_credits'))
        
        # Merge with keywords
        if self.config.USE_KEYWORDS:
            merged = merged.merge(keywords, left_on='movieId', right_on='id', how='left', suffixes=('', '_keywords'))
        
        print(f"Merged dataset shape: {merged.shape}")
        return merged
    
    def create_genre_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create genre-based features"""
        if not self.config.USE_GENRES:
            return df
            
        print("Creating genre features...")
        
        # Parse genres
        df['genres_list'] = self.parse_list_column(df['genres'])
        
        # Get all unique genres
        all_genres = set()
        for genres in df['genres_list'].dropna():
            all_genres.update(genres)
        
        # Create binary features for each genre
        for genre in sorted(all_genres):
            df[f'genre_{genre.lower().replace(" ", "_")}'] = df['genres_list'].apply(
                lambda x: 1 if isinstance(x, list) and genre in x else 0
            )
        
        # Add genre count and primary genre features
        df['genre_count_filled'] = df['genre_count'].fillna(0)
        
        return df
    
    def create_cast_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create cast and director features"""
        if not (self.config.USE_ACTORS or self.config.USE_DIRECTORS):
            return df
            
        print("Creating cast features...")
        
        if self.config.USE_ACTORS:
            # Parse lead actors
            df['lead_actors_list'] = self.parse_list_column(df['lead_actors'], self.config.MAX_ACTORS)
            
            # Get top actors based on frequency
            all_actors = []
            for actors in df['lead_actors_list'].dropna():
                all_actors.extend(actors)
            
            actor_counts = pd.Series(all_actors).value_counts()
            top_actors = actor_counts.head(self.config.MAX_TOP_ACTORS).index.tolist()  # Configurable number of top actors
            
            # Create binary features for top actors
            for actor in top_actors:
                safe_actor_name = re.sub(r'[^\w]', '_', actor.lower())
                df[f'actor_{safe_actor_name}'] = df['lead_actors_list'].apply(
                    lambda x: 1 if isinstance(x, list) and actor in x else 0
                )
        
        if self.config.USE_DIRECTORS:
            # Parse directors
            df['directors_list'] = self.parse_list_column(df['directors'])
            
            # Get top directors
            all_directors = []
            for directors in df['directors_list'].dropna():
                all_directors.extend(directors)
            
            director_counts = pd.Series(all_directors).value_counts()
            top_directors = director_counts.head(self.config.MAX_TOP_DIRECTORS).index.tolist()  # Configurable number of top directors
            
            # Create binary features for top directors
            for director in top_directors:
                safe_director_name = re.sub(r'[^\w]', '_', director.lower())
                df[f'director_{safe_director_name}'] = df['directors_list'].apply(
                    lambda x: 1 if isinstance(x, list) and director in x else 0
                )
        
        # Add cast size feature
        df['cast_size_filled'] = df['cast_size'].fillna(0)
        
        return df
    
    def create_keyword_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create keyword-based features using TF-IDF"""
        if not self.config.USE_KEYWORDS:
            return df
            
        print("Creating keyword features...")
        
        # Parse keywords
        df['keywords_list'] = self.parse_list_column(df['keywords'], self.config.MAX_KEYWORDS)
        
        # Convert keywords to text for TF-IDF
        df['keywords_text'] = df['keywords_list'].apply(
            lambda x: ' '.join(x) if isinstance(x, list) else ''
        )
        
        # Apply TF-IDF (limit to top features to avoid overfitting)
        tfidf = TfidfVectorizer(max_features=self.config.MAX_TFIDF_FEATURES, stop_words='english')
        keyword_tfidf = tfidf.fit_transform(df['keywords_text'])
        
        # Create DataFrame with TF-IDF features
        keyword_features = pd.DataFrame(
            keyword_tfidf.toarray(),
            columns=[f'keyword_tfidf_{i}' for i in range(keyword_tfidf.shape[1])],
            index=df.index
        )
        
        # Merge with main DataFrame
        df = pd.concat([df, keyword_features], axis=1)
        
        # Add keyword count feature
        df['keyword_count_filled'] = df['keyword_count'].fillna(0)
        
        return df
    
    def create_movie_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create movie-specific features"""
        if not self.config.USE_MOVIE_FEATURES:
            return df
            
        print("Creating movie features...")
        
        # Fill missing values
        df['release_year_filled'] = df['release_year'].fillna(df['release_year'].median())
        df['runtime_filled'] = df['runtime'].fillna(df['runtime'].median())
        df['budget_filled'] = df['budget'].fillna(0)
        df['revenue_filled'] = df['revenue'].fillna(0)
        df['vote_average_filled'] = df['vote_average'].fillna(df['vote_average'].median())
        df['vote_count_filled'] = df['vote_count'].fillna(0)
        df['popularity_filled'] = df['popularity'].fillna(df['popularity'].median())
        
        # Create derived features
        df['budget_log'] = np.log1p(df['budget_filled'])
        df['revenue_log'] = np.log1p(df['revenue_filled'])
        df['vote_count_log'] = np.log1p(df['vote_count_filled'])
        df['movie_age'] = 2025 - df['release_year_filled']  # Current year - release year
        df['profit'] = df['revenue_filled'] - df['budget_filled']
        df['profit_log'] = np.log1p(np.maximum(df['profit'], 0))
        
        # ROI calculation (avoid division by zero)
        df['roi'] = np.where(df['budget_filled'] > 0, 
                           df['profit'] / df['budget_filled'], 0)
        
        # Encode categorical features
        categorical_features = ['original_language', 'primary_genre']
        for feature in categorical_features:
            if feature in df.columns:
                le = LabelEncoder()
                df[f'{feature}_encoded'] = le.fit_transform(df[feature].fillna('unknown'))
                self.label_encoders[feature] = le
        
        return df
    
    def create_user_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create user-specific features"""
        if not self.config.USE_USER_FEATURES:
            return df
            
        print("Creating user features...")
        
        # User rating statistics
        user_stats = df.groupby('userId').agg({
            'rating': ['mean', 'std', 'count'],
            'rating_year': ['min', 'max']
        }).round(3)
        
        user_stats.columns = ['user_avg_rating', 'user_rating_std', 'user_rating_count',
                             'user_first_rating_year', 'user_last_rating_year']
        user_stats = user_stats.reset_index()
        
        # User activity span
        user_stats['user_activity_span'] = (user_stats['user_last_rating_year'] - 
                                           user_stats['user_first_rating_year'])
        
        # Fill NaN std with 0 (users with only one rating)
        user_stats['user_rating_std'] = user_stats['user_rating_std'].fillna(0)
        
        # Merge with main DataFrame
        df = df.merge(user_stats, on='userId', how='left')
        
        return df


class RandomForestRecommender:
    """Random Forest Movie Recommendation Model"""
    
    def __init__(self, config: MovieRecommenderConfig):
        self.config = config
        self.model = RandomForestRegressor(
            n_estimators=config.N_ESTIMATORS,
            max_depth=config.MAX_DEPTH,
            min_samples_split=config.MIN_SAMPLES_SPLIT,
            min_samples_leaf=config.MIN_SAMPLES_LEAF,
            random_state=config.RANDOM_STATE,
            n_jobs=config.N_JOBS
        )
        self.feature_columns = None
        self.processor = DataProcessor(config)
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and prepare features for training"""
        print("Preparing features for model...")
        
        # Define feature columns to use
        feature_columns = []
        
        # Basic movie features
        if self.config.USE_MOVIE_FEATURES:
            basic_features = [
                'release_year_filled', 'runtime_filled', 'budget_log', 'revenue_log',
                'vote_average_filled', 'vote_count_log', 'popularity_filled',
                'movie_age', 'profit_log', 'roi', 'genre_count_filled'
            ]
            feature_columns.extend([col for col in basic_features if col in df.columns])
            
            # Encoded categorical features
            encoded_features = ['original_language_encoded', 'primary_genre_encoded']
            feature_columns.extend([col for col in encoded_features if col in df.columns])
        
        # User features
        if self.config.USE_USER_FEATURES:
            user_features = [
                'user_avg_rating', 'user_rating_std', 'user_rating_count',
                'user_activity_span'
            ]
            feature_columns.extend([col for col in user_features if col in df.columns])
        
        # Cast features
        if self.config.USE_ACTORS or self.config.USE_DIRECTORS:
            cast_features = [col for col in df.columns if col.startswith(('actor_', 'director_'))]
            feature_columns.extend(cast_features)
            if 'cast_size_filled' in df.columns:
                feature_columns.append('cast_size_filled')
        
        # Genre features
        if self.config.USE_GENRES:
            genre_features = [col for col in df.columns if col.startswith('genre_')]
            feature_columns.extend(genre_features)
        
        # Keyword features
        if self.config.USE_KEYWORDS:
            keyword_features = [col for col in df.columns if col.startswith('keyword_')]
            feature_columns.extend(keyword_features)
            if 'keyword_count_filled' in df.columns:
                feature_columns.append('keyword_count_filled')
        
        # Store feature columns for later use
        self.feature_columns = feature_columns
        
        print(f"Selected {len(feature_columns)} features for training")
        return df[feature_columns]
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Train the Random Forest model"""
        print("Training Random Forest model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.TEST_SIZE, random_state=self.config.RANDOM_STATE
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'train_mse': mean_squared_error(y_train, y_pred_train),
            'test_mse': mean_squared_error(y_test, y_pred_test),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test)
        }
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=self.config.CV_FOLDS, 
                                   scoring='neg_mean_squared_error')
        metrics['cv_mse'] = -cv_scores.mean()
        metrics['cv_mse_std'] = cv_scores.std()
        
        print("Training completed!")
        return metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from trained model"""
        if self.model is None or self.feature_columns is None:
            raise ValueError("Model must be trained first")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using trained model"""
        if self.model is None:
            raise ValueError("Model must be trained first")
        return self.model.predict(X)


def main():
    """Main execution function"""
    print("=== Random Forest Movie Recommender ===\n")
    
    # Initialize configuration
    config = MovieRecommenderConfig()
    processor = DataProcessor(config)
    
    # Load and process data
    movies, credits, keywords, ratings = processor.load_data()
    
    # Filter reliable data
    ratings_filtered = processor.filter_reliable_data(ratings)
    
    # Merge datasets
    merged_data = processor.merge_datasets(movies, credits, keywords, ratings_filtered)
    
    # Apply feature engineering
    print("\n=== Feature Engineering ===")
    merged_data = processor.create_genre_features(merged_data)
    merged_data = processor.create_cast_features(merged_data)
    merged_data = processor.create_keyword_features(merged_data)
    merged_data = processor.create_movie_features(merged_data)
    merged_data = processor.create_user_features(merged_data)
    
    # Remove rows with missing target values
    merged_data = merged_data.dropna(subset=['rating'])
    
    print(f"\nFinal dataset shape: {merged_data.shape}")
    
    # Initialize and train model
    print("\n=== Model Training ===")
    recommender = RandomForestRecommender(config)
    
    # Prepare features
    X = recommender.prepare_features(merged_data)
    y = merged_data['rating']
    
    # Train model
    metrics = recommender.train(X, y)
    
    # Display results
    print("\n=== Training Results ===")
    print(f"Training MSE: {metrics['train_mse']:.4f}")
    print(f"Test MSE: {metrics['test_mse']:.4f}")
    print(f"Training MAE: {metrics['train_mae']:.4f}")
    print(f"Test MAE: {metrics['test_mae']:.4f}")
    print(f"Training R²: {metrics['train_r2']:.4f}")
    print(f"Test R²: {metrics['test_r2']:.4f}")
    print(f"CV MSE: {metrics['cv_mse']:.4f} (±{metrics['cv_mse_std']:.4f})")
    
    # Display feature importance
    print("\n=== Top 20 Most Important Features ===")
    feature_importance = recommender.get_feature_importance()
    print(feature_importance.head(20).to_string(index=False))
    
    return recommender, merged_data, metrics


if __name__ == "__main__":
    recommender, data, metrics = main()
