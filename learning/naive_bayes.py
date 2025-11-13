"""
Naive Bayes Movie Recommender System

This module implements a Naive Bayes-based movie recommendation system
using cleaned movie metadata, credits, keywords, and ratings data.
The system uses Gaussian Naive Bayes for regression and categorical features.
"""

import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, KBinsDiscretizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.ensemble import VotingRegressor
from sklearn.base import BaseEstimator, RegressorMixin
import ast
import re
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class NaiveBayesMovieRecommenderConfig:
    """Configuration class for Naive Bayes Movie Recommender"""
    
    # Data file paths
    DATA_PATH = "/home/woto/Desktop/personal/python/thesis/cleaned_data/"
    MOVIES_FILE = "movies_metadata_cleaned.csv"
    CREDITS_FILE = "credits_cleaned.csv"
    KEYWORDS_FILE = "keywords_cleaned.csv"
    RATINGS_FILE = "ratings_cleaned.csv"
    
    # Model parameters
    RANDOM_STATE = 42
    VAR_SMOOTHING = 1e-9  # Gaussian NB smoothing parameter
    ALPHA = 1.0           # Categorical NB smoothing parameter
    
    # Feature engineering parameters
    MIN_MOVIE_RATINGS = 10  # Minimum ratings for a movie to be considered
    MIN_USER_RATINGS = 20   # Minimum ratings for a user to be considered
    MAX_USERS = 5000        # Maximum number of users to consider (for performance)
    MAX_ACTORS = 5          # Maximum number of lead actors to consider
    MAX_KEYWORDS = 10       # Maximum number of keywords to consider
    
    # Feature count optimization
    MAX_TOP_ACTORS = 20     # Most frequent actors to include as features
    MAX_TOP_DIRECTORS = 10  # Most frequent directors to include as features
    MAX_TOP_GENRES = 15     # Most frequent genres to include as features
    MAX_KEYWORD_FEATURES = 25  # Maximum keyword features from vectorization
    
    # Binning parameters for continuous variables
    N_RATING_BINS = 5       # Number of bins for rating discretization
    N_YEAR_BINS = 8         # Number of bins for release year
    N_RUNTIME_BINS = 6      # Number of bins for runtime
    N_BUDGET_BINS = 5       # Number of bins for budget
    N_POPULARITY_BINS = 5   # Number of bins for popularity
    
    # Model evaluation parameters
    TEST_SIZE = 0.2
    CV_FOLDS = 5
    
    # Feature selection flags
    USE_GENRES = True
    USE_ACTORS = True
    USE_DIRECTORS = True
    USE_KEYWORDS = True
    USE_MOVIE_FEATURES = True
    USE_USER_FEATURES = True
    USE_TEMPORAL_FEATURES = True
    
    # Rating classification thresholds
    LOW_RATING_THRESHOLD = 2.5    # Below this = low rating
    HIGH_RATING_THRESHOLD = 4.0   # Above this = high rating
    # Between thresholds = medium rating


class NaiveBayesDataProcessor:
    """Handles data loading and preprocessing for Naive Bayes models"""
    
    def __init__(self, config: NaiveBayesMovieRecommenderConfig):
        self.config = config
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.discretizers = {}
        
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
    
    def create_categorical_genre_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create categorical genre-based features optimized for Naive Bayes"""
        if not self.config.USE_GENRES:
            return df
            
        print("Creating categorical genre features...")
        
        # Parse genres
        df['genres_list'] = self.parse_list_column(df['genres'])
        
        # Get top genres by frequency
        all_genres = []
        for genres in df['genres_list'].dropna():
            all_genres.extend(genres)
        
        genre_counts = pd.Series(all_genres).value_counts()
        top_genres = genre_counts.head(self.config.MAX_TOP_GENRES).index.tolist()
        
        # Create binary features for top genres only
        for genre in top_genres:
            df[f'genre_{genre.lower().replace(" ", "_")}'] = df['genres_list'].apply(
                lambda x: 1 if isinstance(x, list) and genre in x else 0
            )
        
        # Primary genre as categorical feature
        df['primary_genre_clean'] = df['primary_genre'].fillna('Unknown')
        
        # Genre count binning for categorical treatment
        df['genre_count_filled'] = df['genre_count'].fillna(0)
        df['genre_count_binned'] = pd.cut(df['genre_count_filled'], 
                                        bins=[-1, 0, 1, 2, 3, np.inf], 
                                        labels=['None', 'One', 'Two', 'Three', 'Many'])
        
        return df
    
    def create_categorical_cast_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create categorical cast and director features"""
        if not (self.config.USE_ACTORS or self.config.USE_DIRECTORS):
            return df
            
        print("Creating categorical cast features...")
        
        if self.config.USE_ACTORS:
            # Parse lead actors
            df['lead_actors_list'] = self.parse_list_column(df['lead_actors'], self.config.MAX_ACTORS)
            
            # Get top actors based on frequency
            all_actors = []
            for actors in df['lead_actors_list'].dropna():
                all_actors.extend(actors)
            
            actor_counts = pd.Series(all_actors).value_counts()
            top_actors = actor_counts.head(self.config.MAX_TOP_ACTORS).index.tolist()
            
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
            top_directors = director_counts.head(self.config.MAX_TOP_DIRECTORS).index.tolist()
            
            # Create binary features for top directors
            for director in top_directors:
                safe_director_name = re.sub(r'[^\w]', '_', director.lower())
                df[f'director_{safe_director_name}'] = df['directors_list'].apply(
                    lambda x: 1 if isinstance(x, list) and director in x else 0
                )
        
        # Cast size binning
        df['cast_size_filled'] = df['cast_size'].fillna(0)
        df['cast_size_binned'] = pd.cut(df['cast_size_filled'], 
                                      bins=[-1, 0, 5, 15, 30, np.inf], 
                                      labels=['None', 'Small', 'Medium', 'Large', 'Very Large'])
        
        return df
    
    def create_categorical_keyword_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create keyword-based features using binary encoding for categories"""
        if not self.config.USE_KEYWORDS:
            return df
            
        print("Creating categorical keyword features...")
        
        # Parse keywords
        df['keywords_list'] = self.parse_list_column(df['keywords'], self.config.MAX_KEYWORDS)
        
        # Convert keywords to text for vectorization
        df['keywords_text'] = df['keywords_list'].apply(
            lambda x: ' '.join(x) if isinstance(x, list) else ''
        )
        
        # Use CountVectorizer for binary features (better for Naive Bayes than TF-IDF)
        vectorizer = CountVectorizer(max_features=self.config.MAX_KEYWORD_FEATURES, 
                                   binary=True, stop_words='english')
        keyword_matrix = vectorizer.fit_transform(df['keywords_text'])
        
        # Create DataFrame with binary keyword features
        keyword_features = pd.DataFrame(
            keyword_matrix.toarray(),
            columns=[f'keyword_{word}' for word in vectorizer.get_feature_names_out()],
            index=df.index
        )
        
        # Merge with main DataFrame
        df = pd.concat([df, keyword_features], axis=1)
        
        # Keyword count binning
        df['keyword_count_filled'] = df['keyword_count'].fillna(0)
        df['keyword_count_binned'] = pd.cut(df['keyword_count_filled'], 
                                          bins=[-1, 0, 2, 5, 10, np.inf], 
                                          labels=['None', 'Few', 'Some', 'Many', 'Lots'])
        
        return df
    
    def create_discretized_movie_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create discretized movie-specific features for Naive Bayes"""
        if not self.config.USE_MOVIE_FEATURES:
            return df
            
        print("Creating discretized movie features...")
        
        # Fill missing values first
        df['release_year_filled'] = df['release_year'].fillna(df['release_year'].median())
        df['runtime_filled'] = df['runtime'].fillna(df['runtime'].median())
        df['budget_filled'] = df['budget'].fillna(0)
        df['revenue_filled'] = df['revenue'].fillna(0)
        df['vote_average_filled'] = df['vote_average'].fillna(df['vote_average'].median())
        df['vote_count_filled'] = df['vote_count'].fillna(0)
        df['popularity_filled'] = df['popularity'].fillna(df['popularity'].median())
        
        # Discretize continuous features
        # Release year bins
        year_bins = np.linspace(df['release_year_filled'].min(), 
                               df['release_year_filled'].max(), 
                               self.config.N_YEAR_BINS + 1)
        df['release_year_binned'] = pd.cut(df['release_year_filled'], bins=year_bins, 
                                         include_lowest=True)
        
        # Runtime bins
        df['runtime_binned'] = pd.cut(df['runtime_filled'], 
                                    bins=self.config.N_RUNTIME_BINS,
                                    labels=[f'Runtime_{i}' for i in range(self.config.N_RUNTIME_BINS)])
        
        # Budget bins (log scale for better distribution)
        df['budget_log'] = np.log1p(df['budget_filled'])
        df['budget_binned'] = pd.cut(df['budget_log'], 
                                   bins=self.config.N_BUDGET_BINS,
                                   labels=[f'Budget_{i}' for i in range(self.config.N_BUDGET_BINS)])
        
        # Popularity bins
        df['popularity_binned'] = pd.cut(df['popularity_filled'], 
                                       bins=self.config.N_POPULARITY_BINS,
                                       labels=[f'Popularity_{i}' for i in range(self.config.N_POPULARITY_BINS)])
        
        # Vote average bins (for movie quality)
        df['vote_average_binned'] = pd.cut(df['vote_average_filled'], 
                                         bins=[0, 4, 6, 7, 8, 10],
                                         labels=['Poor', 'Below_Avg', 'Good', 'Very_Good', 'Excellent'])
        
        # Vote count bins (for popularity/reliability)
        df['vote_count_log'] = np.log1p(df['vote_count_filled'])
        df['vote_count_binned'] = pd.cut(df['vote_count_log'], 
                                       bins=5,
                                       labels=[f'VoteCount_{i}' for i in range(5)])
        
        # Movie age
        df['movie_age'] = 2025 - df['release_year_filled']
        df['movie_age_binned'] = pd.cut(df['movie_age'], 
                                      bins=[0, 5, 10, 20, 30, np.inf],
                                      labels=['Recent', 'New', 'Decade_Old', 'Classic', 'Very_Old'])
        
        # Language categories
        df['original_language_clean'] = df['original_language'].fillna('unknown')
        
        # Adult content flag
        df['adult_clean'] = df['adult'].fillna(False).astype(int)
        
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
        
        # Discretize user features
        user_stats['user_avg_rating_binned'] = pd.cut(user_stats['user_avg_rating'], 
                                                     bins=self.config.N_RATING_BINS,
                                                     labels=[f'AvgRating_{i}' for i in range(self.config.N_RATING_BINS)])
        
        user_stats['user_rating_count_binned'] = pd.cut(user_stats['user_rating_count'], 
                                                       bins=5,
                                                       labels=['Low_Activity', 'Moderate', 'Active', 'Very_Active', 'Power_User'])
        
        # Merge with main DataFrame
        df = df.merge(user_stats, on='userId', how='left')
        
        return df
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features from rating timestamps"""
        if not self.config.USE_TEMPORAL_FEATURES:
            return df
            
        print("Creating temporal features...")
        
        # Rating year binning
        df['rating_year_binned'] = pd.cut(df['rating_year'], 
                                        bins=[2000, 2005, 2010, 2015, 2020, 2025],
                                        labels=['Early_2000s', 'Mid_2000s', 'Early_2010s', 'Mid_2010s', 'Recent'])
        
        # Time gap between movie release and rating
        df['rating_delay'] = df['rating_year'] - df['release_year_filled']
        df['rating_delay_binned'] = pd.cut(df['rating_delay'], 
                                         bins=[-np.inf, 0, 1, 5, 10, np.inf],
                                         labels=['Before_Release', 'Same_Year', 'Recent', 'Years_Later', 'Much_Later'])
        
        return df


class HybridNaiveBayesRegressor(BaseEstimator, RegressorMixin):
    """
    Hybrid Naive Bayes regressor that combines Gaussian NB with rating classification
    and categorical features for movie recommendation.
    """
    
    def __init__(self, config: NaiveBayesMovieRecommenderConfig):
        self.config = config
        self.gaussian_nb = GaussianNB(var_smoothing=config.VAR_SMOOTHING)
        self.categorical_nb = CategoricalNB(alpha=config.ALPHA)
        self.rating_encoder = LabelEncoder()
        self.feature_columns_gaussian = None
        self.feature_columns_categorical = None
        
    def _discretize_ratings(self, ratings: np.ndarray) -> np.ndarray:
        """Convert continuous ratings to discrete classes"""
        discrete_ratings = np.zeros_like(ratings, dtype=int)
        discrete_ratings[ratings < self.config.LOW_RATING_THRESHOLD] = 0  # Low
        discrete_ratings[(ratings >= self.config.LOW_RATING_THRESHOLD) & 
                        (ratings < self.config.HIGH_RATING_THRESHOLD)] = 1  # Medium
        discrete_ratings[ratings >= self.config.HIGH_RATING_THRESHOLD] = 2  # High
        return discrete_ratings
    
    def _prepare_features(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Separate features for Gaussian and Categorical NB"""
        
        # Features for Gaussian NB (continuous/ordinal features)
        gaussian_features = []
        for col in X.columns:
            if any(keyword in col.lower() for keyword in ['avg', 'std', 'count', 'span', 'age', 'size', 'log']):
                if X[col].dtype in ['int64', 'float64']:
                    gaussian_features.append(col)
        
        # Features for Categorical NB (binary/categorical features)
        categorical_features = []
        for col in X.columns:
            if any(col.startswith(prefix) for prefix in ['genre_', 'actor_', 'director_', 'keyword_']):
                categorical_features.append(col)
            elif 'binned' in col or 'clean' in col:
                categorical_features.append(col)
        
        self.feature_columns_gaussian = gaussian_features
        self.feature_columns_categorical = categorical_features
        
        # Handle Gaussian features
        if gaussian_features:
            X_gaussian_df = X[gaussian_features].copy()
            X_gaussian = X_gaussian_df.fillna(0).values
        else:
            X_gaussian = np.array([]).reshape(len(X), 0)
        
        # Handle Categorical features more carefully
        if categorical_features:
            X_categorical_df = X[categorical_features].copy()
            # Convert categorical columns to numeric, handling NaN values properly
            for col in categorical_features:
                if X_categorical_df[col].dtype == 'category':
                    # For categorical data, fill NaN with a new category and ensure all values are non-negative
                    X_categorical_df[col] = X_categorical_df[col].cat.add_categories(['unknown'])
                    X_categorical_df[col] = X_categorical_df[col].fillna('unknown')
                    X_categorical_df[col] = X_categorical_df[col].cat.codes
                elif X_categorical_df[col].dtype == 'object':
                    # For object data, fill NaN with 'unknown' and encode
                    X_categorical_df[col] = X_categorical_df[col].fillna('unknown')
                    X_categorical_df[col] = pd.Categorical(X_categorical_df[col]).codes
                else:
                    # For numeric data, just fill with 0
                    X_categorical_df[col] = X_categorical_df[col].fillna(0)
                
                # Ensure all values are non-negative (CategoricalNB requirement)
                if X_categorical_df[col].min() < 0:
                    X_categorical_df[col] = X_categorical_df[col] - X_categorical_df[col].min()
            
            X_categorical = X_categorical_df.values.astype(int)
        else:
            X_categorical = np.array([]).reshape(len(X), 0)
        
        return X_gaussian, X_categorical
    
    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """Train both Gaussian and Categorical Naive Bayes models"""
        
        # Prepare features
        X_gaussian, X_categorical = self._prepare_features(X)
        
        # Discretize target for classification
        y_discrete = self._discretize_ratings(y)
        
        # Train models
        if X_gaussian.shape[1] > 0:
            self.gaussian_nb.fit(X_gaussian, y_discrete)
        
        if X_categorical.shape[1] > 0:
            # Ensure categorical features are integers
            X_categorical = X_categorical.astype(int)
            self.categorical_nb.fit(X_categorical, y_discrete)
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict ratings by combining Gaussian and Categorical NB predictions"""
        
        X_gaussian, X_categorical = self._prepare_features(X)
        
        predictions = []
        
        # Get predictions from both models
        if X_gaussian.shape[1] > 0:
            gaussian_proba = self.gaussian_nb.predict_proba(X_gaussian)
        else:
            gaussian_proba = np.ones((len(X), 3)) / 3  # Uniform probability
        
        if X_categorical.shape[1] > 0:
            X_categorical = X_categorical.astype(int)
            categorical_proba = self.categorical_nb.predict_proba(X_categorical)
        else:
            categorical_proba = np.ones((len(X), 3)) / 3  # Uniform probability
        
        # Combine probabilities (simple average)
        combined_proba = (gaussian_proba + categorical_proba) / 2
        
        # Convert probabilities back to ratings
        # Map class probabilities to continuous ratings
        rating_values = np.array([2.0, 3.5, 4.5])  # Representative values for low, medium, high
        predicted_ratings = np.dot(combined_proba, rating_values)
        
        return predicted_ratings


class NaiveBayesMovieRecommender:
    """Naive Bayes Movie Recommendation System"""
    
    def __init__(self, config: NaiveBayesMovieRecommenderConfig):
        self.config = config
        self.model = HybridNaiveBayesRegressor(config)
        self.feature_columns = None
        self.processor = NaiveBayesDataProcessor(config)
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and prepare features for training"""
        print("Preparing features for Naive Bayes model...")
        
        # Define feature columns to use
        feature_columns = []
        
        # Basic movie features (discretized)
        if self.config.USE_MOVIE_FEATURES:
            basic_features = [
                'release_year_binned', 'runtime_binned', 'budget_binned', 
                'popularity_binned', 'vote_average_binned', 'vote_count_binned',
                'movie_age_binned', 'original_language_clean', 'adult_clean'
            ]
            # Convert categorical columns to category codes for NB
            for col in basic_features:
                if col in df.columns:
                    if df[col].dtype == 'category' or df[col].dtype == 'object':
                        df[f'{col}_encoded'] = pd.Categorical(df[col]).codes
                        feature_columns.append(f'{col}_encoded')
                    else:
                        feature_columns.append(col)
        
        # User features (discretized)
        if self.config.USE_USER_FEATURES:
            user_features = [
                'user_avg_rating', 'user_rating_std', 'user_rating_count',
                'user_activity_span', 'user_avg_rating_binned', 'user_rating_count_binned'
            ]
            for col in user_features:
                if col in df.columns:
                    if df[col].dtype == 'category' or df[col].dtype == 'object':
                        df[f'{col}_encoded'] = pd.Categorical(df[col]).codes
                        feature_columns.append(f'{col}_encoded')
                    else:
                        feature_columns.append(col)
        
        # Cast features
        if self.config.USE_ACTORS or self.config.USE_DIRECTORS:
            cast_features = [col for col in df.columns if col.startswith(('actor_', 'director_'))]
            feature_columns.extend(cast_features)
            
            if 'cast_size_binned' in df.columns:
                df['cast_size_binned_encoded'] = pd.Categorical(df['cast_size_binned']).codes
                feature_columns.append('cast_size_binned_encoded')
        
        # Genre features
        if self.config.USE_GENRES:
            genre_features = [col for col in df.columns if col.startswith('genre_')]
            feature_columns.extend(genre_features)
            
            if 'genre_count_binned' in df.columns:
                df['genre_count_binned_encoded'] = pd.Categorical(df['genre_count_binned']).codes
                feature_columns.append('genre_count_binned_encoded')
        
        # Keyword features
        if self.config.USE_KEYWORDS:
            keyword_features = [col for col in df.columns if col.startswith('keyword_')]
            feature_columns.extend(keyword_features)
            
            if 'keyword_count_binned' in df.columns:
                df['keyword_count_binned_encoded'] = pd.Categorical(df['keyword_count_binned']).codes
                feature_columns.append('keyword_count_binned_encoded')
        
        # Temporal features
        if self.config.USE_TEMPORAL_FEATURES:
            temporal_features = ['rating_year_binned', 'rating_delay_binned']
            for col in temporal_features:
                if col in df.columns:
                    df[f'{col}_encoded'] = pd.Categorical(df[col]).codes
                    feature_columns.append(f'{col}_encoded')
        
        # Store feature columns for later use
        self.feature_columns = feature_columns
        
        print(f"Selected {len(feature_columns)} features for training")
        return df[feature_columns]
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Train the Naive Bayes model"""
        print("Training Naive Bayes model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.TEST_SIZE, random_state=self.config.RANDOM_STATE
        )
        
        # Train model
        self.model.fit(X_train, y_train.values)
        
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
        cv_scores = cross_val_score(self.model, X, y.values, cv=self.config.CV_FOLDS, 
                                   scoring='neg_mean_squared_error')
        metrics['cv_mse'] = -cv_scores.mean()
        metrics['cv_mse_std'] = cv_scores.std()
        
        # Calculate recommendation accuracy metrics
        metrics.update(self._calculate_recommendation_accuracy(y_test.values, y_pred_test))
        
        print("Training completed!")
        return metrics
    
    def _calculate_recommendation_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate recommendation-specific accuracy metrics"""
        
        # 1. Threshold-based accuracy (within ±0.5 rating points)
        within_half_point = np.abs(y_true - y_pred) <= 0.5
        accuracy_0_5 = within_half_point.mean()
        
        # 2. Threshold-based accuracy (within ±1.0 rating points)  
        within_one_point = np.abs(y_true - y_pred) <= 1.0
        accuracy_1_0 = within_one_point.mean()
        
        # 3. Binary classification accuracy (good vs bad movies)
        # Define "good" as rating >= 3.5, "bad" as < 3.5
        y_true_binary = (y_true >= 3.5).astype(int)
        y_pred_binary = (y_pred >= 3.5).astype(int)
        binary_accuracy = (y_true_binary == y_pred_binary).mean()
        
        # 4. Classification accuracy for discrete rating classes
        y_true_discrete = self._discretize_ratings_for_metrics(y_true)
        y_pred_discrete = self._discretize_ratings_for_metrics(y_pred)
        class_accuracy = accuracy_score(y_true_discrete, y_pred_discrete)
        
        # 5. Recommendation precision@k metrics
        # Top 20% predictions vs actual top 20% ratings
        true_top20_threshold = np.percentile(y_true, 80)
        pred_top20_threshold = np.percentile(y_pred, 80)
        
        true_top20 = y_true >= true_top20_threshold
        pred_top20 = y_pred >= pred_top20_threshold
        
        # Precision: Of predicted top 20%, how many are actually top 20%?
        if pred_top20.sum() > 0:
            precision_top20 = (true_top20 & pred_top20).sum() / pred_top20.sum()
        else:
            precision_top20 = 0.0
            
        # Recall: Of actual top 20%, how many did we predict as top 20%?
        if true_top20.sum() > 0:
            recall_top20 = (true_top20 & pred_top20).sum() / true_top20.sum()
        else:
            recall_top20 = 0.0
        
        return {
            'accuracy_within_0_5': accuracy_0_5,
            'accuracy_within_1_0': accuracy_1_0,
            'binary_accuracy': binary_accuracy,
            'class_accuracy': class_accuracy,
            'precision_top20': precision_top20,
            'recall_top20': recall_top20
        }
    
    def _discretize_ratings_for_metrics(self, ratings: np.ndarray) -> np.ndarray:
        """Convert continuous ratings to discrete classes for accuracy calculation"""
        discrete_ratings = np.zeros_like(ratings, dtype=int)
        discrete_ratings[ratings < self.config.LOW_RATING_THRESHOLD] = 0  # Low
        discrete_ratings[(ratings >= self.config.LOW_RATING_THRESHOLD) & 
                        (ratings < self.config.HIGH_RATING_THRESHOLD)] = 1  # Medium
        discrete_ratings[ratings >= self.config.HIGH_RATING_THRESHOLD] = 2  # High
        return discrete_ratings
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using trained model"""
        if self.model is None:
            raise ValueError("Model must be trained first")
        return self.model.predict(X)


def main():
    """Main execution function"""
    print("=== Naive Bayes Movie Recommender ===\n")
    
    # Initialize configuration
    config = NaiveBayesMovieRecommenderConfig()
    processor = NaiveBayesDataProcessor(config)
    
    # Load and process data
    movies, credits, keywords, ratings = processor.load_data()
    
    # Filter reliable data
    ratings_filtered = processor.filter_reliable_data(ratings)
    
    # Merge datasets
    merged_data = processor.merge_datasets(movies, credits, keywords, ratings_filtered)
    
    # Apply feature engineering
    print("\n=== Feature Engineering ===")
    merged_data = processor.create_categorical_genre_features(merged_data)
    merged_data = processor.create_categorical_cast_features(merged_data)
    merged_data = processor.create_categorical_keyword_features(merged_data)
    merged_data = processor.create_discretized_movie_features(merged_data)
    merged_data = processor.create_user_features(merged_data)
    merged_data = processor.create_temporal_features(merged_data)
    
    # Remove rows with missing target values
    merged_data = merged_data.dropna(subset=['rating'])
    
    print(f"\nFinal dataset shape: {merged_data.shape}")
    
    # Initialize and train model
    print("\n=== Model Training ===")
    recommender = NaiveBayesMovieRecommender(config)
    
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
    
    # Display recommendation accuracy metrics
    print("\n=== Recommendation Accuracy Metrics ===")
    print(f"Accuracy within ±0.5 points: {metrics['accuracy_within_0_5']:.4f} ({metrics['accuracy_within_0_5']*100:.1f}%)")
    print(f"Accuracy within ±1.0 points: {metrics['accuracy_within_1_0']:.4f} ({metrics['accuracy_within_1_0']*100:.1f}%)")
    print(f"Binary classification accuracy: {metrics['binary_accuracy']:.4f} ({metrics['binary_accuracy']*100:.1f}%)")
    print(f"Class prediction accuracy: {metrics['class_accuracy']:.4f} ({metrics['class_accuracy']*100:.1f}%)")
    print(f"Precision@Top20%: {metrics['precision_top20']:.4f} ({metrics['precision_top20']*100:.1f}%)")
    print(f"Recall@Top20%: {metrics['recall_top20']:.4f} ({metrics['recall_top20']*100:.1f}%)")
    
    # Display model information
    print("\n=== Model Information ===")
    print(f"Total features used: {len(recommender.feature_columns)}")
    print(f"Gaussian NB features: {len(recommender.model.feature_columns_gaussian) if recommender.model.feature_columns_gaussian else 0}")
    print(f"Categorical NB features: {len(recommender.model.feature_columns_categorical) if recommender.model.feature_columns_categorical else 0}")
    print(f"User limit: {config.MAX_USERS}")
    print(f"Min movie ratings: {config.MIN_MOVIE_RATINGS}")
    print(f"Min user ratings: {config.MIN_USER_RATINGS}")
    
    return recommender, merged_data, metrics


if __name__ == "__main__":
    recommender, data, metrics = main()