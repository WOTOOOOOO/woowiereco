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
    MIN_RELEASE_YEAR = 2000 # Only include movies released after this year
    
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
    USE_USER_FEATURES = False
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
        """
        Load all cleaned movie data files from the specified directory.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: 
                A tuple containing (movies, credits, keywords, ratings) DataFrames
                - movies: Movie metadata including titles, genres, budgets, revenues
                - credits: Cast and crew information for each movie
                - keywords: Movie keywords and themes
                - ratings: User ratings with timestamps
        
        Raises:
            FileNotFoundError: If any of the required data files are missing
            pd.errors.EmptyDataError: If any of the data files are empty
        """
        print("Loading data files...")
        
        movies = pd.read_csv(f"{self.config.DATA_PATH}{self.config.MOVIES_FILE}")
        credits = pd.read_csv(f"{self.config.DATA_PATH}{self.config.CREDITS_FILE}")
        keywords = pd.read_csv(f"{self.config.DATA_PATH}{self.config.KEYWORDS_FILE}")
        ratings = pd.read_csv(f"{self.config.DATA_PATH}{self.config.RATINGS_FILE}")
        
        print(f"Loaded: {len(movies)} movies, {len(credits)} credits, {len(keywords)} keywords, {len(ratings)} ratings")
        return movies, credits, keywords, ratings
    
    def filter_reliable_data(self, ratings: pd.DataFrame) -> pd.DataFrame:
        """
        Filter ratings to include only reliable movies and active users.
        
        This method performs several filtering steps:
        1. Limits the number of users to MAX_USERS for performance
        2. Filters movies with fewer than MIN_MOVIE_RATINGS
        3. Filters users with fewer than MIN_USER_RATINGS
        
        Args:
            ratings (pd.DataFrame): Raw ratings DataFrame
            
        Returns:
            pd.DataFrame: Filtered ratings DataFrame containing only reliable data
        """
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
        """
        Parse string representations of lists and optionally limit items.
        
        Args:
            series (pd.Series): Series containing stringified lists (e.g., "['a', 'b']")
            max_items (Optional[int]): Maximum number of items to keep from each list
            
        Returns:
            pd.Series: Series containing actual Python lists
        """
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
        """
        Merge all datasets into a single DataFrame for analysis.
        
        Combines ratings with movie metadata, credits, and keywords based on movieId.
        
        Args:
            movies (pd.DataFrame): Movie metadata
            credits (pd.DataFrame): Cast and crew information
            keywords (pd.DataFrame): Movie keywords
            ratings (pd.DataFrame): User ratings (filtered)
            
        Returns:
            pd.DataFrame: Merged DataFrame containing all features
        """
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
        
        # Filter movies by release year (only movies after MIN_RELEASE_YEAR)
        print(f"Filtering movies released after {self.config.MIN_RELEASE_YEAR}...")
        initial_count = len(merged)
        merged = merged[merged['release_year'] >= self.config.MIN_RELEASE_YEAR].copy()
        filtered_count = len(merged)
        print(f"Filtered from {initial_count} to {filtered_count} ratings ({initial_count - filtered_count} removed)")
        
        print(f"Merged dataset shape: {merged.shape}")
        return merged
    
    def create_categorical_genre_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create categorical genre-based features optimized for Naive Bayes.
        
        Processes genre information to create:
        1. Binary features for top genres
        2. Primary genre categorical feature
        3. Binned genre count feature
        
        Args:
            df (pd.DataFrame): Input DataFrame with 'genres' column
            
        Returns:
            pd.DataFrame: DataFrame with added genre features
        """
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
        if self.config.MAX_TOP_GENRES is not None:
            top_genres = genre_counts.head(self.config.MAX_TOP_GENRES).index.tolist()
        else:
            top_genres = genre_counts.index.tolist()
        
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
        """
        Create categorical cast and director features.
        
        Processes cast and crew information to create:
        1. Binary features for top actors
        2. Binary features for top directors
        3. Binned cast size feature
        
        Args:
            df (pd.DataFrame): Input DataFrame with 'lead_actors' and 'directors' columns
            
        Returns:
            pd.DataFrame: DataFrame with added cast/crew features
        """
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
            if self.config.MAX_TOP_ACTORS is not None:
                top_actors = actor_counts.head(self.config.MAX_TOP_ACTORS).index.tolist()
            else:
                top_actors = actor_counts.index.tolist()
            
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
            if self.config.MAX_TOP_DIRECTORS is not None:
                top_directors = director_counts.head(self.config.MAX_TOP_DIRECTORS).index.tolist()
            else:
                top_directors = director_counts.index.tolist()
            
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
        """
        Create keyword-based features using binary encoding for categories.
        
        Uses CountVectorizer to create binary features for the most frequent keywords.
        Also creates a binned feature for the number of keywords per movie.
        
        Args:
            df (pd.DataFrame): Input DataFrame with 'keywords' column
            
        Returns:
            pd.DataFrame: DataFrame with added keyword features
        """
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
        """
        Create discretized movie-specific features for Naive Bayes.
        
        Discretizes continuous variables into bins to make them suitable for 
        Categorical Naive Bayes or to improve Gaussian Naive Bayes performance.
        Features processed: release_year, runtime, budget, popularity, vote_average, vote_count.
        
        Args:
            df (pd.DataFrame): Input DataFrame with movie metadata
            
        Returns:
            pd.DataFrame: DataFrame with added discretized features
        """
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
        
        # Revenue bins (log scale)
        df['revenue_log'] = np.log1p(df['revenue_filled'])
        df['revenue_binned'] = pd.cut(df['revenue_log'], 
                                    bins=self.config.N_BUDGET_BINS, # Reuse budget bins count
                                    labels=[f'Revenue_{i}' for i in range(self.config.N_BUDGET_BINS)])

        # Profit and ROI
        df['profit'] = df['revenue_filled'] - df['budget_filled']
        df['profit_log'] = np.log1p(np.maximum(df['profit'], 0))
        df['profit_binned'] = pd.cut(df['profit_log'], 
                                   bins=5,
                                   labels=[f'Profit_{i}' for i in range(5)])
        
        df['roi'] = np.where(df['budget_filled'] > 0, 
                           df['profit'] / df['budget_filled'], 0)
        # Handle infinite ROI if any
        df['roi'] = df['roi'].replace([np.inf, -np.inf], 0)
        df['roi_binned'] = pd.cut(df['roi'], 
                                bins=[-np.inf, 0, 1, 2, 5, np.inf],
                                labels=['Loss', 'Low', 'Medium', 'High', 'Very_High'])
        
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
        """
        Create user-specific features based on rating history.
        
        Calculates and bins user statistics:
        - Average rating (binned)
        - Rating count (binned)
        - Activity span
        
        Args:
            df (pd.DataFrame): Input DataFrame with user ratings
            
        Returns:
            pd.DataFrame: DataFrame with added user features
        """
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
        """
        Create temporal features from rating timestamps.
        
        Extracts and bins temporal information:
        - Day of week
        - Month
        - Time of day (Morning, Afternoon, Evening, Night)
        - Is weekend
        
        Args:
            df (pd.DataFrame): Input DataFrame with 'timestamp' column
            
        Returns:
            pd.DataFrame: DataFrame with added temporal features
        """
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
    
    def _prepare_features(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Separate and prepare features for Gaussian and Categorical NB models.
        
        Splits the input DataFrame into two arrays:
        1. Gaussian features: Continuous variables (avg, std, counts, etc.)
        2. Categorical features: Binary/Categorical variables (genres, actors, bins)
        
        Also handles missing values and ensures non-negative values for CategoricalNB.
        
        Args:
            X (pd.DataFrame): Input feature DataFrame
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple of (X_gaussian, X_categorical) arrays
        """
        
        # Features for Gaussian NB (continuous/ordinal features)
        gaussian_features = []
        for col in X.columns:
            # Skip categorical columns even if they contain keywords like 'age' (e.g. 'keyword_marriage')
            if any(col.startswith(prefix) for prefix in ['genre_', 'actor_', 'director_', 'keyword_']):
                continue
                
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
        """
        Train both Gaussian and Categorical Naive Bayes models.
        
        1. Prepares features for each model type
        2. Discretizes target ratings into classes using LabelEncoder
        3. Fits GaussianNB on continuous features
        4. Fits CategoricalNB on categorical features
        
        Args:
            X (pd.DataFrame): Training features
            y (np.ndarray): Target ratings
            
        Returns:
            self: The trained estimator
        """
        
        # Prepare features
        X_gaussian, X_categorical = self._prepare_features(X)
        
        # Discretize target for classification using LabelEncoder
        # This allows us to predict exact rating values (0.5, 1.0, ..., 5.0)
        # instead of just 3 coarse classes
        y_discrete = self.rating_encoder.fit_transform(y)
        
        # Train models
        if X_gaussian.shape[1] > 0:
            self.gaussian_nb.fit(X_gaussian, y_discrete)
        
        if X_categorical.shape[1] > 0:
            # Ensure categorical features are integers
            X_categorical = X_categorical.astype(int)
            self.categorical_nb.fit(X_categorical, y_discrete)
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict ratings by combining Gaussian and Categorical NB predictions.
        
        1. Gets class probabilities from both models
        2. Averages the probabilities (ensemble)
        3. Converts combined probabilities to continuous ratings using expected value
        
        Args:
            X (pd.DataFrame): Features to predict on
            
        Returns:
            np.ndarray: Predicted continuous ratings
        """
        
        X_gaussian, X_categorical = self._prepare_features(X)
        n_classes = len(self.rating_encoder.classes_)
        
        # Get predictions from both models
        has_gaussian = X_gaussian.shape[1] > 0
        has_categorical = X_categorical.shape[1] > 0
        
        if has_gaussian:
            gaussian_proba = self.gaussian_nb.predict_proba(X_gaussian)
        
        if has_categorical:
            X_categorical = X_categorical.astype(int)
            categorical_proba = self.categorical_nb.predict_proba(X_categorical)
        
        # Combine probabilities
        if has_gaussian and has_categorical:
            combined_proba = (gaussian_proba + categorical_proba) / 2
        elif has_gaussian:
            combined_proba = gaussian_proba
        elif has_categorical:
            combined_proba = categorical_proba
        else:
            # Fallback if no features (shouldn't happen)
            combined_proba = np.ones((len(X), n_classes)) / n_classes
        
        # Convert probabilities back to ratings
        # Use argmax (most likely class) instead of expected value
        # This avoids "averaging out" to the mean and allows predicting extreme values (1.0 or 5.0)
        rating_values = self.rating_encoder.classes_
        most_likely_indices = np.argmax(combined_proba, axis=1)
        predicted_ratings = rating_values[most_likely_indices]
        
        return predicted_ratings


class NaiveBayesMovieRecommender:
    """Naive Bayes Movie Recommendation System"""
    
    def __init__(self, config: NaiveBayesMovieRecommenderConfig):
        self.config = config
        self.model = HybridNaiveBayesRegressor(config)
        self.feature_columns = None
        self.processor = NaiveBayesDataProcessor(config)
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select and prepare features for training.
        
        Collects all relevant features based on configuration:
        - Movie features (binned/encoded)
        - User features (binned/encoded)
        - Cast/Director features
        - Genre features
        - Keyword features
        - Temporal features
        
        Also handles encoding of categorical columns that weren't pre-encoded.
        
        Args:
            df (pd.DataFrame): Full merged DataFrame
            
        Returns:
            pd.DataFrame: DataFrame containing only the selected features for training
        """
        print("Preparing features for Naive Bayes model...")
        
        # Define feature columns to use
        feature_columns = []
        
        # Basic movie features (discretized and continuous)
        if self.config.USE_MOVIE_FEATURES:
            basic_features = [
                'release_year_binned', 'runtime_binned', 'budget_binned', 
                'revenue_binned', 'profit_binned', 'roi_binned',
                'popularity_binned', 'vote_average_binned', 'vote_count_binned',
                'movie_age_binned', 'original_language_clean', 'adult_clean',
                'primary_genre_clean',
                # Continuous features for Gaussian NB
                'release_year_filled', 'runtime_filled', 'budget_filled',
                'revenue_filled', 'profit', 'roi',
                'popularity_filled', 'vote_average_filled', 'vote_count_filled',
                'budget_log', 'revenue_log', 'profit_log', 'vote_count_log'
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
            # Select only specific binary features to avoid grabbing raw text columns
            cast_features = [col for col in df.columns if (col.startswith('actor_') or col.startswith('director_')) 
                           and not col.endswith('_list')]
            feature_columns.extend(cast_features)
            
            if 'cast_size_binned' in df.columns:
                df['cast_size_binned_encoded'] = pd.Categorical(df['cast_size_binned']).codes
                feature_columns.append('cast_size_binned_encoded')
            
            # Add continuous cast size for Gaussian NB
            if 'cast_size_filled' in df.columns:
                feature_columns.append('cast_size_filled')
        
        # Genre features
        if self.config.USE_GENRES:
            # Exclude count/filled/binned variations from the wildcard selection
            genre_features = [col for col in df.columns if col.startswith('genre_') 
                            and 'count' not in col 
                            and not col.endswith('_list')]
            feature_columns.extend(genre_features)
            
            if 'genre_count_binned' in df.columns:
                df['genre_count_binned_encoded'] = pd.Categorical(df['genre_count_binned']).codes
                feature_columns.append('genre_count_binned_encoded')
            
            # Add continuous genre count for Gaussian NB
            if 'genre_count_filled' in df.columns:
                feature_columns.append('genre_count_filled')
        
        # Keyword features
        if self.config.USE_KEYWORDS:
            # Exclude count/filled/binned variations
            keyword_features = [col for col in df.columns if col.startswith('keyword_') 
                              and 'count' not in col 
                              and not col.endswith('_list')
                              and not col.endswith('_text')]
            feature_columns.extend(keyword_features)
            
            if 'keyword_count_binned' in df.columns:
                df['keyword_count_binned_encoded'] = pd.Categorical(df['keyword_count_binned']).codes
                feature_columns.append('keyword_count_binned_encoded')
            
            # Add continuous keyword count for Gaussian NB
            if 'keyword_count_filled' in df.columns:
                feature_columns.append('keyword_count_filled')
        
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
        """
        Train the Naive Bayes model and evaluate performance.
        
        Performs the following steps:
        1. Splits data into train/test sets
        2. Trains the HybridNaiveBayesRegressor
        3. Evaluates on train and test sets (MSE, MAE, R2)
        4. Performs cross-validation
        5. Calculates recommendation-specific accuracy metrics
        
        Args:
            X (pd.DataFrame): Feature DataFrame
            y (pd.Series): Target ratings
            
        Returns:
            Dict: Dictionary containing all performance metrics
        """
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
        };
        
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
        """
        Calculate recommendation-specific accuracy metrics.
        
        Metrics include:
        - Accuracy within ±0.5 and ±1.0 rating points
        - Binary classification accuracy (Like/Dislike)
        - Multi-class accuracy (Low/Medium/High)
        - Precision and Recall for top 20% of recommendations
        
        Args:
            y_true (np.ndarray): True ratings
            y_pred (np.ndarray): Predicted ratings
            
        Returns:
            Dict: Dictionary of accuracy metrics
        """
        
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
        
        # Calculate Confusion Matrix elements
        tp = ((y_pred_binary == 1) & (y_true_binary == 1)).sum()
        tn = ((y_pred_binary == 0) & (y_true_binary == 0)).sum()
        fp = ((y_pred_binary == 1) & (y_true_binary == 0)).sum()
        fn = ((y_pred_binary == 0) & (y_true_binary == 1)).sum()
        
        # Calculate Binary Precision and Recall
        binary_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        binary_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
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
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'binary_precision': binary_precision,
            'binary_recall': binary_recall,
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
    
    def explain_feature_importance_math(self) -> Dict:
        """
        Provide detailed mathematical explanation of how feature importance is calculated.
        
        Returns a dictionary containing:
        1. Gaussian NB explanation (Variance of Class Means, LDA)
        2. Categorical NB explanation (Entropy, Information Gain)
        3. Mathematical formulas used in the calculations
        4. Example calculation for the first Gaussian feature
        
        Returns:
            Dict: Structured explanation of mathematical concepts and formulas
        """
        if self.model is None:
            raise ValueError("Model must be trained first")
            
        explanation = {
            'gaussian_nb_explanation': {},
            'categorical_nb_explanation': {},
            'mathematical_formulas': {}
        }
        
        # Gaussian NB Mathematical Explanation
        if (self.model.feature_columns_gaussian and 
            hasattr(self.model.gaussian_nb, 'theta_') and 
            self.model.gaussian_nb.theta_ is not None):
            
            class_means = self.model.gaussian_nb.theta_
            class_vars = self.model.gaussian_nb.var_ if hasattr(self.model.gaussian_nb, 'var_') else None
            
            explanation['gaussian_nb_explanation'] = {
                'method': 'Variance of Class Means',
                'formula': 'importance_i = Var(μ_i^(class_0), μ_i^(class_1), μ_i^(class_2))',
                'interpretation': 'Higher variance = feature values differ more between classes = more discriminative',
                'n_classes': class_means.shape[0],
                'n_features': class_means.shape[1],
                'class_means_shape': class_means.shape,
                'example_calculation': {}
            }
            
            # Show example calculation for first feature
            if len(self.model.feature_columns_gaussian) > 0:
                feature_name = self.model.feature_columns_gaussian[0]
                means_for_feature = class_means[:, 0]
                variance = np.var(means_for_feature)
                
                explanation['gaussian_nb_explanation']['example_calculation'] = {
                    'feature': feature_name,
                    'class_means': means_for_feature.tolist(),
                    'mean_low_rating': means_for_feature[0],
                    'mean_medium_rating': means_for_feature[1] if len(means_for_feature) > 1 else 0,
                    'mean_high_rating': means_for_feature[2] if len(means_for_feature) > 2 else 0,
                    'calculated_variance': variance,
                    'step_by_step': {
                        'step_1': f"Class means: {means_for_feature.tolist()}",
                        'step_2': f"Overall mean: {np.mean(means_for_feature):.4f}",
                        'step_3': f"Squared deviations: {[(x - np.mean(means_for_feature))**2 for x in means_for_feature]}",
                        'step_4': f"Variance: {variance:.4f}"
                    }
                }
                
            # Fisher LDA explanation if variances available
            if class_vars is not None:
                explanation['gaussian_nb_explanation']['lda_method'] = {
                    'formula': 'LDA_score_i = (between_class_variance_i) / (within_class_variance_i)',
                    'between_class_variance': 'Variance of class means',
                    'within_class_variance': 'Average of variances within each class',
                    'interpretation': 'Higher ratio = better separation between classes relative to spread within classes'
                }
        
        # Categorical NB Mathematical Explanation  
        explanation['categorical_nb_explanation'] = {
            'method': 'Entropy-based Information Gain',
            'formula': 'importance_i = -Σ(p_j * log(p_j)) where p_j are class probabilities for feature i',
            'interpretation': 'Higher entropy = more uncertainty = more informative when resolved',
            'fallback_method': 'Class Prior Variance when entropy calculation fails',
            'note': 'CategoricalNB stores complex probability distributions, making direct importance calculation challenging'
        }
        
        # Mathematical Formulas Reference
        explanation['mathematical_formulas'] = {
            'naive_bayes_assumption': 'P(features|class) = Π P(feature_i|class) [independence assumption]',
            'gaussian_likelihood': 'P(x_i|class) = (1/√(2πσ²)) * exp(-(x_i-μ)²/(2σ²))',
            'categorical_likelihood': 'P(x_i|class) = θ_i where θ_i is learned probability',
            'class_prediction': 'class = argmax_c [P(class) * Π P(feature_i|class)]',
            'variance_formula': 'Var(X) = E[(X - μ)²] = Σ(x_i - μ)²/n',
            'entropy_formula': 'H(X) = -Σ p_i * log(p_i)',
            'fisher_lda': 'J = (μ_1 - μ_2)² / (σ_1² + σ_2²) [for 2 classes]'
        }
        
        return explanation

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Calculate feature importance for Naive Bayes, normalized to sum to 100% (1.0).
        
        Since Naive Bayes doesn't have a native "feature importance" metric like Random Forest,
        we use statistical proxies and normalize them to create a comparable metric.
        
        Methodology:
        1. Gaussian NB (Continuous Features):
           Uses Fisher's LDA Score = (Between-Class Variance) / (Within-Class Variance).
           This measures how well the feature separates the classes relative to the noise.
           
        2. Categorical NB (Discrete Features):
           Uses Variance of Probabilities across classes.
           If a feature's probability distribution is the same for all classes, variance is 0 (unimportant).
           If it varies significantly, it helps distinguish classes.
           
        Normalization:
        - Since the model averages predictions from Gaussian and Categorical components (50/50 weight),
          we normalize feature importances such that:
          - Sum of Gaussian feature importances = 0.5
          - Sum of Categorical feature importances = 0.5
          - Total sum = 1.0 (100%)
        
        Returns:
            pd.DataFrame: DataFrame containing feature importance scores, sorted by importance.
                          Columns: 'feature', 'importance', 'model_type'
        """
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        gaussian_scores = {}
        categorical_scores = {}
        
        # --- 1. Calculate Raw Scores for Gaussian Features ---
        if (self.model.feature_columns_gaussian and 
            hasattr(self.model.gaussian_nb, 'theta_') and 
            self.model.gaussian_nb.theta_ is not None):
            
            # theta_ is mean of each feature per class: (n_classes, n_features)
            class_means = self.model.gaussian_nb.theta_
            # var_ is variance of each feature per class
            class_vars = self.model.gaussian_nb.var_ if hasattr(self.model.gaussian_nb, 'var_') else None
            
            # Calculate variance of means (Between-Class Variance)
            between_class_variance = np.var(class_means, axis=0)
            
            for i, feature in enumerate(self.model.feature_columns_gaussian):
                score = between_class_variance[i]
                
                # If we have within-class variance, use Fisher's LDA score (better)
                if class_vars is not None:
                    within_class_variance = np.mean(class_vars[:, i])
                    if within_class_variance > 1e-10:
                        score = score / within_class_variance
                
                gaussian_scores[feature] = score
        
        # --- 2. Calculate Raw Scores for Categorical Features ---
        if (self.model.feature_columns_categorical and 
            hasattr(self.model.categorical_nb, 'feature_log_prob_') and
            self.model.categorical_nb.feature_log_prob_ is not None):
            
            feature_log_probs = self.model.categorical_nb.feature_log_prob_
            
            try:
                # Handle list of arrays (standard sklearn behavior for CategoricalNB)
                if isinstance(feature_log_probs, list):
                    for i, feature in enumerate(self.model.feature_columns_categorical):
                        if i < len(feature_log_probs):
                            # log_probs shape: (n_classes, n_categories_for_this_feature)
                            log_probs = feature_log_probs[i]
                            probs = np.exp(log_probs)
                            
                            # Calculate variance across classes for each category value
                            # Then sum/mean to get total feature importance
                            # We use sum of variances to capture total discriminative power across all categories
                            category_variances = np.var(probs, axis=0)
                            score = np.sum(category_variances)
                            
                            categorical_scores[feature] = score
                            
                # Handle single array case (unlikely for CategoricalNB but possible in some versions)
                elif hasattr(feature_log_probs, 'shape'):
                    # ...existing code...
                    pass # Fallback or simplified logic if needed
                    
            except Exception as e:
                print(f"Warning: Error calculating categorical importance: {e}")
        
        # --- 3. Normalize Scores ---
        importance_data = []
        
        # Determine weights
        has_gaussian = len(gaussian_scores) > 0
        has_categorical = len(categorical_scores) > 0
        
        if has_gaussian and has_categorical:
            w_gaussian = 0.5
            w_categorical = 0.5
        elif has_gaussian:
            w_gaussian = 1.0
            w_categorical = 0.0
        else:
            w_gaussian = 0.0
            w_categorical = 1.0
            
        # Normalize Gaussian scores
        if has_gaussian:
            total_raw_gaussian = sum(gaussian_scores.values())
            if total_raw_gaussian > 0:
                for feature, raw_score in gaussian_scores.items():
                    normalized_score = (raw_score / total_raw_gaussian) * w_gaussian
                    importance_data.append({
                        'feature': feature,
                        'importance': normalized_score,
                        'model_type': 'Gaussian_NB'
                    })
            else:
                # If all scores are 0, distribute weight evenly
                uniform_score = w_gaussian / len(gaussian_scores)
                for feature in gaussian_scores:
                    importance_data.append({
                        'feature': feature,
                        'importance': uniform_score,
                        'model_type': 'Gaussian_NB'
                    })

        # Normalize Categorical scores
        if has_categorical:
            total_raw_categorical = sum(categorical_scores.values())
            if total_raw_categorical > 0:
                for feature, raw_score in categorical_scores.items():
                    normalized_score = (raw_score / total_raw_categorical) * w_categorical
                    importance_data.append({
                        'feature': feature,
                        'importance': normalized_score,
                        'model_type': 'Categorical_NB'
                    })
            else:
                # If all scores are 0, distribute weight evenly
                uniform_score = w_categorical / len(categorical_scores)
                for feature in categorical_scores:
                    importance_data.append({
                        'feature': feature,
                        'importance': uniform_score,
                        'model_type': 'Categorical_NB'
                    })
        
        if not importance_data:
            return pd.DataFrame()
            
        return pd.DataFrame(importance_data).sort_values('importance', ascending=False)
    
    def get_feature_group_importance(self) -> pd.DataFrame:
        """
        Calculate cumulative importance for feature groups (Genres, Keywords, Cast, etc.).
        
        Aggregates individual feature importance scores into logical groups to provide
        a higher-level view of what drives the model's predictions.
        
        Returns:
            pd.DataFrame: DataFrame containing group importance statistics
        """
        importance_df = self.get_feature_importance()
        if importance_df.empty:
            return pd.DataFrame()
            
        groups = {
            'Genres': [],
            'Keywords': [],
            'Actors': [],
            'Directors': [],
            'Movie Metadata': [],
            'User Features': [],
            'Temporal Features': []
        }
        
        for feature in importance_df['feature']:
            if feature.startswith('genre_') or 'primary_genre' in feature:
                groups['Genres'].append(feature)
            elif feature.startswith('keyword_'):
                groups['Keywords'].append(feature)
            elif feature.startswith('actor_') or 'cast_size' in feature:
                groups['Actors'].append(feature)
            elif feature.startswith('director_'):
                groups['Directors'].append(feature)
            elif feature.startswith('user_'):
                groups['User Features'].append(feature)
            elif feature.startswith('rating_') and 'count' not in feature: 
                # rating_year, rating_delay. user_rating_count is user feature
                groups['Temporal Features'].append(feature)
            else:
                groups['Movie Metadata'].append(feature)
                
        group_importance = []
        for group_name, features in groups.items():
            if not features:
                continue
                
            # Sum importance for features in this group
            group_data = importance_df[importance_df['feature'].isin(features)]
            total_importance = group_data['importance'].sum()
            
            group_importance.append({
                'feature_group': group_name,
                'total_importance': total_importance,
                'feature_count': len(features),
                'top_features': ', '.join(group_data.head(3)['feature'].tolist())
            })
            
        return pd.DataFrame(group_importance).sort_values('total_importance', ascending=False)
    
    def get_class_probabilities_analysis(self, feature_subset: List[str] = None) -> Dict:
        """
        Analyze how features differ across rating classes
        """
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        analysis = {}
        
        # Gaussian NB analysis
        if (hasattr(self.model.gaussian_nb, 'theta_') and 
            self.model.gaussian_nb.theta_ is not None):
            
            features_to_analyze = feature_subset or self.model.feature_columns_gaussian[:10]  # Top 10 if not specified
            
            gaussian_analysis = {}
            class_means = self.model.gaussian_nb.theta_
            class_vars = self.model.gaussian_nb.var_
            
            for i, feature in enumerate(self.model.feature_columns_gaussian):
                if feature in features_to_analyze:
                    gaussian_analysis[feature] = {
                        'class_means': {
                            'low_rating': class_means[0, i] if class_means.shape[0] > 0 else 0,
                            'medium_rating': class_means[1, i] if class_means.shape[0] > 1 else 0,
                            'high_rating': class_means[2, i] if class_means.shape[0] > 2 else 0,
                        },
                        'class_variances': {
                            'low_rating': class_vars[0, i] if class_vars.shape[0] > 0 else 0,
                            'medium_rating': class_vars[1, i] if class_vars.shape[0] > 1 else 0,
                            'high_rating': class_vars[2, i] if class_vars.shape[0] > 2 else 0,
                        }
                    }
            
            analysis['gaussian_features'] = gaussian_analysis
        
        # Categorical NB analysis
        if (hasattr(self.model.categorical_nb, 'feature_log_prob_') and 
            self.model.categorical_nb.feature_log_prob_ is not None):
            
            features_to_analyze_cat = feature_subset or self.model.feature_columns_categorical[:10]
            
            categorical_analysis = {}
            feature_log_probs = self.model.categorical_nb.feature_log_prob_
            
            for i, feature in enumerate(self.model.feature_columns_categorical):
                if feature in features_to_analyze_cat:
                    # Get the most probable categories for each class
                    class_probs = np.exp(feature_log_probs[:, i])
                    
                    categorical_analysis[feature] = {
                        'class_preferences': {
                            'low_rating': float(class_probs[0]) if len(class_probs) > 0 else 0,
                            'medium_rating': float(class_probs[1]) if len(class_probs) > 1 else 0,
                            'high_rating': float(class_probs[2]) if len(class_probs) > 2 else 0,
                        }
                    }
            
            analysis['categorical_features'] = categorical_analysis
        
        return analysis

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using trained model.
        
        Wrapper around the HybridNaiveBayesRegressor's predict method.
        
        Args:
            X (pd.DataFrame): Features to predict on
            
        Returns:
            np.ndarray: Predicted ratings
            
        Raises:
            ValueError: If model has not been trained yet
        """
        if self.model is None:
            raise ValueError("Model must be trained first")
        return self.model.predict(X)
    
    def recommend_for_user(self, user_data: pd.DataFrame, user_id: int, 
                          test_size: float = 0.3, random_state: int = 42) -> Dict:
        """
        Generate recommendations for a specific user
        
        Args:
            user_data: DataFrame containing all user's movie ratings with features
            user_id: ID of the user to generate recommendations for
            test_size: Fraction of user's movies to hold out for testing
            random_state: Random state for reproducible splits
            
        Returns:
            Dictionary containing recommendations and evaluation results
        """
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        print(f"\n=== Generating Recommendations for User {user_id} ===")
        
        # Get user's movie data
        user_movies = user_data[user_data['userId'] == user_id].copy()
        
        if len(user_movies) == 0:
            raise ValueError(f"No data found for user {user_id}")
        
        print(f"User {user_id} has rated {len(user_movies)} movies")
        
        # Split user's movies into training (history) and test (evaluation) sets
        if len(user_movies) < 5:
            print("Warning: User has very few ratings, using all for training")
            train_movies = user_movies.copy()
            test_movies = pd.DataFrame()
        else:
            train_movies, test_movies = train_test_split(
                user_movies, test_size=test_size, random_state=random_state, 
                stratify=None  # Can't stratify continuous targets
            )
        
        print(f"Using {len(train_movies)} movies as history, {len(test_movies)} movies for evaluation")
        
        # Prepare features for test movies (movies to predict)
        if len(test_movies) > 0:
            # Ensure we have all the necessary columns (including encoded ones)
            # Since user_data comes from merged_data which was processed by prepare_features,
            # it should have the encoded columns.
            
            # Check if we need to re-prepare features (if encoded columns are missing)
            # But we assume user_data has them.
            
            # Select features
            X_test_user = test_movies[self.feature_columns]
            y_true_user = test_movies['rating'].values
            
            # Make predictions
            y_pred_user = self.predict(X_test_user)
            
            # Calculate user-specific metrics
            user_metrics = {
                'user_mse': mean_squared_error(y_true_user, y_pred_user),
                'user_mae': mean_absolute_error(y_true_user, y_pred_user),
                'user_r2': r2_score(y_true_user, y_pred_user)
            }
            
            # Add accuracy metrics
            user_metrics.update(self._calculate_recommendation_accuracy(
                y_true_user, y_pred_user
            ))
            
            # Create recommendations DataFrame
            recommendations = test_movies[['movieId', 'title', 'genres', 'release_year', 'rating']].copy()
            recommendations['predicted_rating'] = y_pred_user
            recommendations['rating_diff'] = recommendations['predicted_rating'] - recommendations['rating']
            recommendations = recommendations.sort_values('predicted_rating', ascending=False)
            
        else:
            user_metrics = {}
            recommendations = pd.DataFrame()
        
        # Get user's movie history (training data) for reference
        history = train_movies[['movieId', 'title', 'genres', 'release_year', 'rating']].copy()
        history = history.sort_values('rating', ascending=False)
        
        # Summary statistics
        if len(history) > 0:
            user_stats = {
                'total_movies_rated': len(user_movies),
                'movies_in_history': len(history),
                'movies_for_evaluation': len(test_movies),
                'avg_user_rating': history['rating'].mean(),
                'user_rating_std': history['rating'].std(),
                'favorite_genres': self._get_user_favorite_genres(history),
                'rating_distribution': history['rating'].value_counts().sort_index().to_dict()
            }
        else:
            user_stats = {}
        
        return {
            'user_id': user_id,
            'user_stats': user_stats,
            'movie_history': history,
            'recommendations': recommendations,
            'user_metrics': user_metrics
        }
    
    def _get_user_favorite_genres(self, user_movies: pd.DataFrame, top_n: int = 3) -> List[str]:
        """
        Identify user's favorite genres based on their rating history.
        """
        # Parse genres and calculate average ratings per genre
        genre_ratings = {}
        
        for _, movie in user_movies.iterrows():
            try:
                genres = ast.literal_eval(movie['genres']) if pd.notna(movie['genres']) else []
                for genre in genres:
                    if genre not in genre_ratings:
                        genre_ratings[genre] = []
                    genre_ratings[genre].append(movie['rating'])
            except:
                continue
        
        # Calculate average rating per genre
        genre_avg_ratings = {
            genre: np.mean(ratings) 
            for genre, ratings in genre_ratings.items() 
            if len(ratings) >= 2  # Only consider genres with at least 2 movies
        }
        
        # Sort by average rating and return top genres
        sorted_genres = sorted(genre_avg_ratings.items(), key=lambda x: x[1], reverse=True)
        return [genre for genre, _ in sorted_genres[:top_n]]


def main():
    """
    Main execution function for the Naive Bayes Movie Recommender.
    
    Orchestrates the entire pipeline:
    1. Initializes configuration and data processor
    2. Loads and cleans data
    3. Performs feature engineering
    4. Trains the model
    5. Evaluates performance
    6. Displays detailed results and feature importance analysis
    """
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
    
    # Display Confusion Matrix details
    total_predictions = metrics['tp'] + metrics['tn'] + metrics['fp'] + metrics['fn']
    print(f"\nConfusion Matrix (Threshold = 3.5):")
    print(f"True Positives (TP): {metrics['tp']} ({metrics['tp']/total_predictions*100:.1f}%)")
    print(f"False Positives (FP): {metrics['fp']} ({metrics['fp']/total_predictions*100:.1f}%)")
    print(f"True Negatives (TN): {metrics['tn']} ({metrics['tn']/total_predictions*100:.1f}%)")
    print(f"False Negatives (FN): {metrics['fn']} ({metrics['fn']/total_predictions*100:.1f}%)")
    
    print(f"\nBinary Precision (Threshold=3.5): {metrics['binary_precision']:.4f} ({metrics['binary_precision']*100:.1f}%)")
    print(f"Binary Recall (Threshold=3.5): {metrics['binary_recall']:.4f} ({metrics['binary_recall']*100:.1f}%)")
    
    print(f"\nClass prediction accuracy: {metrics['class_accuracy']:.4f} ({metrics['class_accuracy']*100:.1f}%)")
    print(f"Precision@Top20%: {metrics['precision_top20']:.4f} ({metrics['precision_top20']*100:.1f}%)")
    print(f"Recall@Top20%: {metrics['recall_top20']:.4f} ({metrics['recall_top20']*100:.1f}%)")
    
    # Display model information
    print_feature_summary(recommender.feature_columns)
    
    # Save feature list to file
    with open('naive_bayes_features.txt', 'w') as f:
        for feature in recommender.feature_columns:
            f.write(f"{feature}\n")
    print(f"Feature list saved to naive_bayes_features.txt")

    print(f"\nGaussian NB features: {len(recommender.model.feature_columns_gaussian) if recommender.model.feature_columns_gaussian else 0}")
    print(f"Categorical NB features: {len(recommender.model.feature_columns_categorical) if recommender.model.feature_columns_categorical else 0}")
    print(f"User limit: {config.MAX_USERS}")
    print(f"Min movie ratings: {config.MIN_MOVIE_RATINGS}")
    print(f"Min user ratings: {config.MIN_USER_RATINGS}")
    
    # Display feature importance analysis
    print("\n=== Feature Importance Analysis ===")
    try:
        feature_importance = recommender.get_feature_importance()
        if not feature_importance.empty:
            # Display Group Importance (New)
            print("\n--- Feature Group Importance (Cumulative) ---")
            group_importance = recommender.get_feature_group_importance()
            if not group_importance.empty:
                print(f"{'Feature Group':<20} {'Total Importance':<18} {'Count':<6} {'Top Features'}")
                print("-" * 80)
                for _, row in group_importance.iterrows():
                    print(f"{row['feature_group']:<20} {row['total_importance']:.4f}             {row['feature_count']:<6} {row['top_features']}")
            
            print("\n--- Top 15 Individual Features ---")
            top_features = feature_importance.head(15)
            for _, row in top_features.iterrows():
                print(f"{row['feature']:<40} {row['importance']:.4f} ({row['model_type']})")
            
            # Show breakdown by model type
            print(f"\nGaussian NB Features (top 5):")
            gaussian_features = feature_importance[feature_importance['model_type'] == 'Gaussian_NB'].head(5)
            for _, row in gaussian_features.iterrows():
                print(f"  {row['feature']:<35} {row['importance']:.4f}")
                
            print(f"\nCategorical NB Features (top 5):")
            categorical_features = feature_importance[feature_importance['model_type'] == 'Categorical_NB'].head(5)
            for _, row in categorical_features.iterrows():
                print(f"  {row['feature']:<35} {row['importance']:.4f}")
        else:
            print("No feature importance data available")
            
        # Mathematical Explanation
        print("\n=== MATHEMATICAL EXPLANATION OF FEATURE IMPORTANCE ===")
        try:
            math_explanation = recommender.explain_feature_importance_math()
            
            print("\n1. GAUSSIAN NB FEATURE IMPORTANCE:")
            gauss_exp = math_explanation['gaussian_nb_explanation']
            print(f"   Method: {gauss_exp['method']}")
            print(f"   Formula: {gauss_exp['formula']}")
            print(f"   Interpretation: {gauss_exp['interpretation']}")
            
            if 'example_calculation' in gauss_exp:
                example = gauss_exp['example_calculation']
                print(f"\n   Example with '{example['feature']}':")
                print(f"   - Low rating class mean: {example['mean_low_rating']:.3f}")
                print(f"   - Medium rating class mean: {example['mean_medium_rating']:.3f}")
                print(f"   - High rating class mean: {example['mean_high_rating']:.3f}")
                print(f"   - Variance across classes: {example['calculated_variance']:.4f}")
                print(f"   - Step-by-step: {example['step_by_step']['step_1']}")
                print(f"                   Overall mean = {example['step_by_step']['step_2'].split(': ')[1]}")
                print(f"                   Final variance = {example['calculated_variance']:.4f}")
            
            print(f"\n2. CATEGORICAL NB FEATURE IMPORTANCE:")
            cat_exp = math_explanation['categorical_nb_explanation']
            print(f"   Method: {cat_exp['method']}")
            print(f"   Formula: {cat_exp['formula']}")
            print(f"   Interpretation: {cat_exp['interpretation']}")
            print(f"   Note: {cat_exp['note']}")
            
            print(f"\n3. KEY MATHEMATICAL CONCEPTS:")
            formulas = math_explanation['mathematical_formulas']
            print(f"   - Naive Bayes: {formulas['naive_bayes_assumption']}")
            print(f"   - Gaussian Likelihood: {formulas['gaussian_likelihood']}")
            print(f"   - Variance: {formulas['variance_formula']}")
            print(f"   - Entropy: {formulas['entropy_formula']}")
            
        except Exception as e:
            print(f"Error in mathematical explanation: {e}")
                
            # Interpretation
            print(f"\n=== Feature Importance Interpretation ===")
            print("1. USER BEHAVIOR is the strongest predictor:")
            print("   - user_rating_count: How many movies a user has rated")
            print("   - user_avg_rating: User's average rating tendency")
            print("   - user_activity_span: How long user has been active")
            
            print("\n2. MOVIE CHARACTERISTICS also matter:")
            movie_features = [f for f in top_features['feature'] if any(x in f.lower() for x in ['budget', 'runtime', 'popularity', 'vote', 'age'])]
            if movie_features:
                print("   - " + "\n   - ".join(movie_features[:5]))
                
            print("\n3. CAST/CREW influence ratings:")
            cast_features = [f for f in top_features['feature'] if 'actor_' in f or 'director_' in f]
            if cast_features:
                print("   - " + "\n   - ".join(cast_features[:3]))
        else:
            print("No feature importance data available")
    except Exception as e:
        print(f"Error calculating feature importance: {e}")
    
    # Demonstrate user recommendations
    print("\n=== User Recommendation Demo ===")
    demo_user_recommendations(recommender, merged_data)
    
    return recommender, merged_data, metrics


def demo_user_recommendations(recommender: NaiveBayesMovieRecommender, data: pd.DataFrame, 
                            num_users: int = 3):
    """
    Demonstrate the recommendation system with real user examples.
    
    Selects random users from the dataset and generates personalized recommendations,
    showing how the model performs on individual users with detailed analysis.
    
    Args:
        recommender (NaiveBayesMovieRecommender): Trained recommendation model
        data (pd.DataFrame): Complete dataset with user ratings and features
        num_users (int): Number of users to demonstrate. Defaults to 3.
    """
    
    # Get users with sufficient ratings
    user_counts = data.groupby('userId').size()
    eligible_users = user_counts[user_counts >= 10].index.tolist()
    
    if len(eligible_users) == 0:
        print("No users with sufficient ratings found for demo")
        return
    
    # Select random users
    np.random.seed(42)
    demo_users = np.random.choice(eligible_users, min(num_users, len(eligible_users)), replace=False)
    
    for user_id in demo_users:
        try:
            print(f"\n{'='*60}")
            result = recommender.recommend_for_user(data, user_id, test_size=0.3)
            
            # Display user statistics
            stats = result['user_stats']
            print(f"User {user_id} Profile:")
            print(f"  Total movies rated: {stats.get('total_movies_rated', 'N/A')}")
            print(f"  Average rating: {stats.get('avg_user_rating', 0):.2f}")
            print(f"  Rating std: {stats.get('user_rating_std', 0):.2f}")
            print(f"  Favorite genres: {', '.join(stats.get('favorite_genres', []))}")
            
            # Display complete movie history (all training movies)
            history = result['movie_history']
            if len(history) > 0:
                print(f"\n  Complete Movie History ({len(history)} movies):")
                for _, movie in history.iterrows():
                    genres_str = str(movie['genres'])[:50] + "..." if len(str(movie['genres'])) > 50 else str(movie['genres'])
                    print(f"    {movie['rating']:.1f}★ {movie['title']} ({movie['release_year']}) - {genres_str}")
            
            # Display recommendations and evaluation
            recommendations = result['recommendations']
            if len(recommendations) > 0:
                user_metrics = result['user_metrics']
                print(f"\n  Prediction Performance:")
                print(f"    MAE: {user_metrics.get('user_mae', 0):.3f}")
                print(f"    Accuracy within ±1.0: {user_metrics.get('accuracy_within_1_0', 0)*100:.1f}%")
                print(f"    Binary accuracy: {user_metrics.get('binary_accuracy', 0)*100:.1f}%")
                
                print(f"\n  Predictions vs Actual Ratings:")
                for _, movie in recommendations.head(5).iterrows():
                    genres_str = str(movie['genres'])[:40] + "..." if len(str(movie['genres'])) > 40 else str(movie['genres'])
                    print(f"    Predicted: {movie['predicted_rating']:.1f}★ | Actual: {movie['rating']:.1f}★ | "
                          f"{movie['title']} ({movie['release_year']}) - {genres_str}")
            else:
                print("  No evaluation movies available")
                
        except Exception as e:
            print(f"Error generating recommendations for user {user_id}: {e}")
            continue

def print_feature_summary(features: List[str]):
    """
    Print a standardized summary of features used in the model.
    Groups common feature types (genres, actors, etc.) and lists specific ones.
    """
    print("\n=== Feature Summary ===")
    
    # Define groups and their prefixes
    groups = {
        'Genres': 'genre_',
        'Actors': 'actor_',
        'Directors': 'director_',
        'Keywords': 'keyword_'
    }
    
    # Special handling for temporal features
    temporal_features = ['rating_year', 'rating_month', 'rating_day_of_week', 'is_weekend', 'rating_delay',
                        'rating_year_binned_encoded', 'rating_delay_binned_encoded']
    
    grouped_counts = {k: 0 for k in groups}
    grouped_counts['Temporal'] = 0
    other_features = []
    
    for feature in features:
        matched = False
        for group, prefix in groups.items():
            if feature.startswith(prefix) and 'count' not in feature:
                grouped_counts[group] += 1
                matched = True
                break
        
        if not matched:
            if feature in temporal_features or any(tf in feature for tf in temporal_features):
                grouped_counts['Temporal'] += 1
            else:
                other_features.append(feature)
            
    # Print grouped features
    print("Grouped Features:")
    for group_name, count in grouped_counts.items():
        if count > 0:
            print(f"  - {count} {group_name} features")
            
    # Print other features
    print("\nSpecific Features:")
    for feature in sorted(other_features):
        print(f"  - {feature}")
        
    print(f"\nTotal Features: {len(features)}")


if __name__ == "__main__":
    recommender, data, metrics = main()