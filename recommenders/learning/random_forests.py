"""
Random Forest Movie Recommender System

This module implements a Random Forest-based movie recommendation system
using cleaned movie metadata, credits, keywords, and ratings data.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, MultiLabelBinarizer
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
    LINKS_FILE = "links.csv"
    
    # Random Forest parameters
    N_ESTIMATORS = 100
    MAX_DEPTH = 20
    MIN_SAMPLES_SPLIT = 5
    MIN_SAMPLES_LEAF = 2
    RANDOM_STATE = 42
    N_JOBS = -1
    
    # Feature engineering parameters
    MIN_MOVIE_RATINGS = 5  # Minimum ratings for a movie to be considered
    MIN_USER_RATINGS = 10   # Minimum ratings for a user to be considered
    MAX_USERS = 5000        # Maximum number of users to consider (for performance)
    MAX_ACTORS = 5          # Maximum number of lead actors to consider
    MAX_KEYWORDS = 10       # Maximum number of keywords to consider
    MIN_RELEASE_YEAR = 2000 # Only include movies released after this year
    
    # Feature count optimization
    # Set to None to use all available features (no limit)
    MAX_TOP_ACTORS = 20  # Reduced to match Naive Bayes
    MAX_TOP_DIRECTORS = 10  # Reduced to match Naive Bayes
    MAX_TOP_GENRES = 15     # Most frequent genres to include as features
    MAX_KEYWORD_FEATURES = 25 # Renamed from MAX_TFIDF_FEATURES and reduced to match Naive Bayes
    
    # Model evaluation parameters
    TEST_SIZE = 0.2
    CV_FOLDS = 5
    
    # Feature selection
    USE_GENRES = True
    USE_ACTORS = True
    USE_DIRECTORS = True
    USE_KEYWORDS = True
    USE_MOVIE_FEATURES = True
    USE_USER_FEATURES = False  # Disabled user behavioral features
    USE_TEMPORAL_FEATURES = True  # Enable temporal features


class DataProcessor:
    """Handles data loading and preprocessing"""
    
    def __init__(self, config: MovieRecommenderConfig):
        self.config = config
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all cleaned movie data files from the specified directory.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: 
                A tuple containing (movies, credits, keywords, ratings, links) DataFrames
                - movies: Movie metadata including titles, genres, budgets, revenues
                - credits: Cast and crew information for each movie
                - keywords: Movie keywords and themes
                - ratings: User ratings with timestamps
                - links: Mapping between MovieLens IDs and TMDb IDs
        
        Raises:
            FileNotFoundError: If any of the required data files are missing
            pd.errors.EmptyDataError: If any of the data files are empty
        """
        print("Loading data files...")
        
        movies = pd.read_csv(f"{self.config.DATA_PATH}{self.config.MOVIES_FILE}")
        credits = pd.read_csv(f"{self.config.DATA_PATH}{self.config.CREDITS_FILE}")
        keywords = pd.read_csv(f"{self.config.DATA_PATH}{self.config.KEYWORDS_FILE}")
        ratings = pd.read_csv(f"{self.config.DATA_PATH}{self.config.RATINGS_FILE}")
        links = pd.read_csv(f"{self.config.DATA_PATH}{self.config.LINKS_FILE}")
        
        print(f"Loaded: {len(movies)} movies, {len(credits)} credits, {len(keywords)} keywords, {len(ratings)} ratings, {len(links)} links")
        return movies, credits, keywords, ratings, links
    
    def filter_reliable_data(self, ratings: pd.DataFrame) -> pd.DataFrame:
        """
        Filter the ratings data to include only reliable movies and active users.
        
        This method applies several filters to ensure data quality:
        1. Limits to first N users (configurable via MAX_USERS)
        2. Includes only movies with minimum number of ratings (MIN_MOVIE_RATINGS)
        3. Includes only users with minimum number of ratings (MIN_USER_RATINGS)
        
        Args:
            ratings (pd.DataFrame): Raw ratings DataFrame with columns:
                - userId: User identifier
                - movieId: Movie identifier  
                - rating: Rating value
                - Other columns preserved
        
        Returns:
            pd.DataFrame: Filtered ratings DataFrame containing only reliable 
                         data that meets all minimum thresholds
        
        Note:
            The filtering order is important - user selection happens first
            to improve performance, then movie/user activity filters are applied.
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
        Safely parse string representations of Python lists into actual list objects.
        
        Handles various edge cases including missing values, empty lists, and
        malformed strings. Optionally limits the number of items in each list.
        
        Args:
            series (pd.Series): Series containing string representations of lists
                               (e.g., "['item1', 'item2']")
            max_items (Optional[int]): Maximum number of items to keep from each list.
                                     If None, keeps all items. Defaults to None.
        
        Returns:
            pd.Series: Series with actual Python list objects. Empty lists for
                      missing/malformed values.
        
        Example:
            >>> series = pd.Series(["['a', 'b']", "['c']", None, "[]"])
            >>> parse_list_column(series, max_items=1)
            0    [a]
            1    [c]  
            2    []
            3    []
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
                      keywords: pd.DataFrame, ratings: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
        """
        Merge all movie-related datasets into a single comprehensive DataFrame.
        
        Performs left joins to combine ratings with movie metadata, cast/crew
        information, and keywords. Uses links table to bridge MovieLens IDs and TMDb IDs.
        Also applies release year filtering to focus on modern movies (post-2000).
        
        Args:
            movies (pd.DataFrame): Movie metadata with columns like title, genres,
                                 budget, revenue, release_year
            credits (pd.DataFrame): Cast and crew data with lead_actors, directors,
                                  cast_size columns
            keywords (pd.DataFrame): Movie keywords and themes
            ratings (pd.DataFrame): User ratings (already filtered for reliability)
            links (pd.DataFrame): Mapping between MovieLens IDs and TMDb IDs
        
        Returns:
            pd.DataFrame: Merged DataFrame containing all features needed for
                         recommendation modeling. Uses ratings as base table.
        
        Note:
            - Uses left joins to preserve all rating records
            - Only includes movies released after MIN_RELEASE_YEAR (default: 2000)
            - Merge columns are controlled by USE_* configuration flags
            - Handles suffix conflicts with '_credits' and '_keywords' suffixes
        """
        print("Merging datasets...")
        
        # Start with ratings as base
        merged = ratings.copy()
        
        # Merge ratings (movieId) with links (movieId) to get tmdbId
        merged = merged.merge(links[['movieId', 'tmdbId']], on='movieId', how='left')
        
        # Drop rows where tmdbId is missing (cannot link to metadata)
        initial_count = len(merged)
        merged = merged.dropna(subset=['tmdbId'])
        if len(merged) < initial_count:
            print(f"Dropped {initial_count - len(merged)} ratings with no valid link to TMDb ID")
        
        # Ensure tmdbId is integer for merging
        merged['tmdbId'] = merged['tmdbId'].astype(int)
        
        # Merge with movies using tmdbId -> id
        merged = merged.merge(movies, left_on='tmdbId', right_on='id', how='left')
        
        # Merge with credits using tmdbId -> id
        if self.config.USE_ACTORS or self.config.USE_DIRECTORS:
            merged = merged.merge(credits[['id', 'lead_actors', 'directors', 'cast_size']], 
                                left_on='tmdbId', right_on='id', how='left', suffixes=('', '_credits'))
        
        # Merge with keywords using tmdbId -> id
        if self.config.USE_KEYWORDS:
            merged = merged.merge(keywords, left_on='tmdbId', right_on='id', how='left', suffixes=('', '_keywords'))
        
        # Filter movies by release year (only movies after MIN_RELEASE_YEAR)
        print(f"Filtering movies released after {self.config.MIN_RELEASE_YEAR}...")
        initial_count = len(merged)
        merged = merged[merged['release_year'] >= self.config.MIN_RELEASE_YEAR].copy()
        filtered_count = len(merged)
        print(f"Filtered from {initial_count} to {filtered_count} ratings ({initial_count - filtered_count} removed)")
        
        print(f"Merged dataset shape: {merged.shape}")
        return merged
    
    def create_genre_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create genre-based features for movie recommendation modeling using MultiLabelBinarizer.
        
        Transforms the genres column (list of genre strings) into binary indicator
        features for each unique genre, plus additional genre-related features.
        
        Args:
            df (pd.DataFrame): Input DataFrame with 'genres' column containing
                             string representations of genre lists
        
        Returns:
            pd.DataFrame: Enhanced DataFrame with additional columns:
                - genre_{genre_name}: Binary features (0/1) for each unique genre
                - genres_list: Parsed list objects from genre strings
        
        Features Created:
            - Binary encoding for each genre (e.g., 'genre_action', 'genre_comedy')
            - Handles missing values and normalizes genre names
        
        Note:
            Only executes if USE_GENRES is True in configuration.
            Uses MultiLabelBinarizer for efficient binary encoding.
        """
        if not self.config.USE_GENRES:
            return df
            
        print("Creating genre features...")
        
        # Parse genres
        df['genres_list'] = self.parse_list_column(df['genres'])
        
        # Filter for top genres if limit is set
        if self.config.MAX_TOP_GENRES is not None:
            all_genres = []
            for genres in df['genres_list'].dropna():
                all_genres.extend(genres)
            
            genre_counts = pd.Series(all_genres).value_counts()
            top_genres = genre_counts.head(self.config.MAX_TOP_GENRES).index.tolist()
            
            # Filter genres list to only include top genres
            df['genres_list'] = df['genres_list'].apply(
                lambda x: [g for g in x if g in top_genres] if isinstance(x, list) else []
            )
        
        # Use MultiLabelBinarizer for efficient binary encoding
        mlb_genres = MultiLabelBinarizer()
        genre_binary = mlb_genres.fit_transform(df['genres_list'])
        
        # Create DataFrame with binary genre features
        genre_columns = [f'genre_{genre.lower().replace(" ", "_")}' for genre in mlb_genres.classes_]
        genre_df = pd.DataFrame(genre_binary, columns=genre_columns, index=df.index)
        
        # Merge with main DataFrame
        df = pd.concat([df, genre_df], axis=1)
        
        
        print(f"Created {len(genre_columns)} genre features")
        return df
    
    def create_cast_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create cast and director features for movie recommendation modeling using MultiLabelBinarizer.
        
        Transforms cast and director information into binary indicator features
        for the most frequent/popular actors and directors in the dataset.
        
        Args:
            df (pd.DataFrame): Input DataFrame with 'lead_actors' and 'directors' 
                             columns containing string representations of lists
        
        Returns:
            pd.DataFrame: Enhanced DataFrame with additional columns:
                - actor_{actor_name}: Binary features for top frequent actors
                - director_{director_name}: Binary features for top frequent directors  
                - cast_size_filled: Number of cast members (NaN filled with 0)
                - lead_actors_list/directors_list: Parsed list objects
        
        Features Created:
            Actor Features (if USE_ACTORS=True):
            - Binary indicators for top N most frequent actors (MAX_TOP_ACTORS)
            - Limited to first MAX_ACTORS lead actors per movie
            
            Director Features (if USE_DIRECTORS=True):
            - Binary indicators for top N most frequent directors (MAX_TOP_DIRECTORS)
            
            Cast Statistics:
            - Cast size as numerical feature
        
        Note:
            - Actor/director names are sanitized for safe column names
            - Only most frequent actors/directors become features (popularity-based)
            - Uses MultiLabelBinarizer for efficient binary encoding
            - Helps capture "star power" and directorial influence on ratings
        """
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
            
            if self.config.MAX_TOP_ACTORS is not None:
                top_actors = actor_counts.head(self.config.MAX_TOP_ACTORS).index.tolist()
                # Filter actors list to only include top actors
                df['top_actors_list'] = df['lead_actors_list'].apply(
                    lambda x: [actor for actor in x if actor in top_actors] if isinstance(x, list) else []
                )
            else:
                # Use all actors found
                df['top_actors_list'] = df['lead_actors_list']
            
            # Use MultiLabelBinarizer for efficient binary encoding
            mlb_actors = MultiLabelBinarizer()
            actor_binary = mlb_actors.fit_transform(df['top_actors_list'])
            
            # Create DataFrame with binary actor features
            actor_columns = [f'actor_{re.sub(r"[^\w]", "_", actor.lower())}' for actor in mlb_actors.classes_]
            actor_df = pd.DataFrame(actor_binary, columns=actor_columns, index=df.index)
            
            # Merge with main DataFrame
            df = pd.concat([df, actor_df], axis=1)
            print(f"Created {len(actor_columns)} actor features")
        
        if self.config.USE_DIRECTORS:
            # Parse directors
            df['directors_list'] = self.parse_list_column(df['directors'])
            
            # Get top directors based on frequency
            all_directors = []
            for directors in df['directors_list'].dropna():
                all_directors.extend(directors)
            
            director_counts = pd.Series(all_directors).value_counts()
            
            if self.config.MAX_TOP_DIRECTORS is not None:
                top_directors = director_counts.head(self.config.MAX_TOP_DIRECTORS).index.tolist()
                # Filter directors list to only include top directors
                df['top_directors_list'] = df['directors_list'].apply(
                    lambda x: [director for director in x if director in top_directors] if isinstance(x, list) else []
                )
            else:
                # Use all directors
                df['top_directors_list'] = df['directors_list']
            
            # Use MultiLabelBinarizer for efficient binary encoding
            mlb_directors = MultiLabelBinarizer()
            director_binary = mlb_directors.fit_transform(df['top_directors_list'])
            
            # Create DataFrame with binary director features
            director_columns = [f'director_{re.sub(r"[^\w]", "_", director.lower())}' for director in mlb_directors.classes_]
            director_df = pd.DataFrame(director_binary, columns=director_columns, index=df.index)
            
            # Merge with main DataFrame
            df = pd.concat([df, director_df], axis=1)
            print(f"Created {len(director_columns)} director features")
        
        # Add cast size feature
        df['cast_size_filled'] = df['cast_size'].fillna(0)
        
        return df
    
    def create_keyword_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create keyword-based features using TF-IDF vectorization.
        
        Transforms movie keywords into numerical features that capture thematic
        content and plot elements beyond what genres provide.
        
        Args:
            df (pd.DataFrame): Input DataFrame with 'keywords' column containing
                             string representations of keyword lists
        
        Returns:
            pd.DataFrame: Enhanced DataFrame with additional columns:
                - keyword_tfidf_{i}: TF-IDF weighted keyword features (i=0 to MAX_KEYWORD_FEATURES-1)
                - keywords_list: Parsed keyword list objects  
                - keywords_text: Space-separated keyword strings for TF-IDF
        
        Features Created:
            TF-IDF Features:
            - Converts keywords to TF-IDF vectors (MAX_KEYWORD_FEATURES dimensions)
            - Captures keyword importance and rarity across the dataset
            - Filters English stop words
        
        Algorithm:
            1. Parse keyword lists and limit to MAX_KEYWORDS per movie
            2. Convert to space-separated text strings
            3. Apply TF-IDF vectorization with feature limits
            4. Create DataFrame columns for each TF-IDF dimension
        
        Note:
            Only executes if USE_KEYWORDS is True in configuration.
            TF-IDF helps distinguish important/unique keywords from common ones.
        """
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
        max_features = self.config.MAX_KEYWORD_FEATURES
        tfidf = TfidfVectorizer(max_features=max_features, stop_words='english')
        keyword_tfidf = tfidf.fit_transform(df['keywords_text'])
        
        # Create DataFrame with TF-IDF features
        # Use actual words in column names for interpretability (e.g., keyword_hero)
        keyword_features = pd.DataFrame(
            keyword_tfidf.toarray(),
            columns=[f'keyword_{word}' for word in tfidf.get_feature_names_out()],
            index=df.index
        )
        
        # Merge with main DataFrame
        df = pd.concat([df, keyword_features], axis=1)
        
        
        return df
    
    def create_movie_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive movie-specific features for recommendation modeling.
        
        Transforms raw movie attributes into engineered features including
        financial metrics, temporal features, quality indicators, and
        categorical encodings.
        
        Args:
            df (pd.DataFrame): Input DataFrame with movie metadata columns:
                - release_year, runtime, budget, revenue, vote_average, etc.
        
        Returns:
            pd.DataFrame: Enhanced DataFrame with additional engineered columns:
                
                Basic Features (filled):
                - release_year_filled, runtime_filled, budget_filled, etc.
                
                Log-Transformed Features:
                - budget_log, revenue_log, vote_count_log, profit_log
                
                Derived Financial Features:  
                - movie_age, profit, roi (return on investment)
                
                Encoded Categorical Features:
                - original_language_encoded, primary_genre_encoded
        
        Feature Engineering Details:
            Financial Features:
            - Log transformations for skewed distributions (budget, revenue)
            - Profit calculation and log-transformed profit
            - ROI with division-by-zero protection
            
            Temporal Features:
            - Movie age (current year - release year)
            - Release year as continuous feature
            
            Quality Indicators:
            - Vote average, vote count (popularity metrics)
            - Popularity score from metadata
            
            Missing Value Handling:
            - Medians for continuous features (runtime, vote_average)
            - Zeros for financial features (budget, revenue)
        
        Note:
            Only executes if USE_MOVIE_FEATURES is True in configuration.
            Label encoders are stored for potential inverse transformations.
        """
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
        
        # Adult content flag
        if 'adult' in df.columns:
            df['adult_encoded'] = df['adult'].fillna(False).astype(int)
        
        return df
    
    def create_user_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create user-specific features to capture individual rating behaviors.
        
        Calculates aggregate statistics for each user:
        - Average rating given
        - Standard deviation of ratings (volatility)
        - Total number of ratings (activity level)
        - Activity span (time between first and last rating)
        
        Args:
            df (pd.DataFrame): Input DataFrame with 'userId', 'rating', 'timestamp'
            
        Returns:
            pd.DataFrame: DataFrame with added user features
        """
        if not self.config.USE_USER_FEATURES:
            return df
            
        print("Creating user features...")
        
        # Calculate user stats
        user_stats = df.groupby('userId').agg({
            'rating': ['mean', 'std', 'count'],
            'timestamp': lambda x: (x.max() - x.min()) / (24 * 3600 * 365)  # Span in years
        })
        
        # Flatten column names
        user_stats.columns = ['user_avg_rating', 'user_rating_std', 'user_rating_count', 'user_activity_span']
        
        # Fill missing std (for users with 1 rating) with 0
        user_stats['user_rating_std'] = user_stats['user_rating_std'].fillna(0)
        
        # Merge back to main dataframe
        df = df.merge(user_stats, on='userId', how='left')
        
        return df

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create temporal features from rating timestamps.
        
        Extracts time-based information to capture temporal patterns:
        - Rating year (and month/day if timestamp available)
        - Time delay between movie release and rating
        
        Args:
            df (pd.DataFrame): Input DataFrame with 'rating_year' and 'release_year'
                              (and optionally 'timestamp')
            
        Returns:
            pd.DataFrame: DataFrame with added temporal features
        """
        if not self.config.USE_TEMPORAL_FEATURES:
            return df
            
        print("Creating temporal features...")
        
        # Check if we have full timestamp
        if 'timestamp' in df.columns:
            # Convert timestamp to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                # Check if timestamp is in seconds (Unix timestamp)
                df['rating_date'] = pd.to_datetime(df['timestamp'], unit='s')
            else:
                df['rating_date'] = df['timestamp']
                
            # Extract basic temporal components
            df['rating_year'] = df['rating_date'].dt.year
            df['rating_month'] = df['rating_date'].dt.month
            df['rating_day_of_week'] = df['rating_date'].dt.dayofweek
            df['is_weekend'] = df['rating_day_of_week'].isin([5, 6]).astype(int)
        elif 'rating_year' in df.columns:
            # We only have the year
            print("Full timestamp not available, using rating_year only")
            # Ensure rating_year is numeric
            df['rating_year'] = pd.to_numeric(df['rating_year'], errors='coerce')
        else:
            print("Warning: No temporal information found (timestamp or rating_year)")
            return df
        
        # Calculate delay between release and rating (in years)
        # Ensure release_year is numeric and handle missing values
        df['release_year_numeric'] = pd.to_numeric(df['release_year'], errors='coerce')
        
        # Calculate delay
        if 'rating_year' in df.columns:
            df['rating_delay'] = df['rating_year'] - df['release_year_numeric']
            
            # Handle potential negative delays (rating before release - data error or pre-release)
            # Clip to 0
            df['rating_delay'] = df['rating_delay'].clip(lower=0)
            
            # Fill missing delays with median
            df['rating_delay'] = df['rating_delay'].fillna(df['rating_delay'].median())
        
        return df
    


class RandomForestRecommender:
    """Random Forest Movie Recommendation Model"""
    
    def __init__(self, config: MovieRecommenderConfig):
        self.config = config
        self.model = RandomForestClassifier(
            n_estimators=config.N_ESTIMATORS,
            max_depth=config.MAX_DEPTH,
            min_samples_split=config.MIN_SAMPLES_SPLIT,
            min_samples_leaf=config.MIN_SAMPLES_LEAF,
            random_state=config.RANDOM_STATE,
            n_jobs=config.N_JOBS
        )
        self.feature_columns = None
        self.processor = DataProcessor(config)
        self.rating_encoder = LabelEncoder()
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select and prepare the final feature set for Random Forest training.
        
        Consolidates all engineered features into the final feature matrix
        based on configuration flags. Ensures only valid features are included
        and stores the feature list for consistent prediction usage.
        
        Args:
            df (pd.DataFrame): Fully processed DataFrame with all engineered features
        
        Returns:
            pd.DataFrame: Feature matrix (X) ready for model training/prediction
                         containing only the selected features
        
        Feature Categories Included:
            Movie Features (if USE_MOVIE_FEATURES=True):
            - Basic: release_year_filled, runtime_filled, budget_log, etc.
            - Encoded: original_language_encoded, primary_genre_encoded
            
            User Features (if USE_USER_FEATURES=True):
            - Behavioral: user_avg_rating, user_rating_std, user_rating_count
            - Temporal: user_activity_span
            
            Cast Features (if USE_ACTORS/USE_DIRECTORS=True):
            - Binary actor/director indicators
            - Cast size numerical feature
            
            Genre Features (if USE_GENRES=True):
            - Binary genre indicators for each unique genre
            
            Keyword Features (if USE_KEYWORDS=True):
            - TF-IDF keyword vectors
        
        Side Effects:
            - Sets self.feature_columns for later use in predictions
            - Validates feature availability in DataFrame
        
        Note:
            This method determines the exact features used by the model.
            Feature selection is controlled by USE_* configuration flags.
        """
        print("Preparing features for model...")
        
        # Define feature columns to use
        feature_columns = []
        
        # Basic movie features
        if self.config.USE_MOVIE_FEATURES:
            basic_features = [
                'release_year_filled', 'runtime_filled', 'budget_log', 'revenue_log',
                'vote_average_filled', 'vote_count_log', 'popularity_filled',
                'movie_age', 'profit_log', 'roi'
            ]
            feature_columns.extend([col for col in basic_features if col in df.columns])
            
            # Encoded categorical features
            encoded_features = ['original_language_encoded', 'primary_genre_encoded', 'adult_encoded']
            feature_columns.extend([col for col in encoded_features if col in df.columns])
        
        # User features
        if self.config.USE_USER_FEATURES:
            user_features = [
                'user_avg_rating', 'user_rating_std', 'user_rating_count',
                'user_activity_span'
            ]
            feature_columns.extend([col for col in user_features if col in df.columns])
            
        # Temporal features
        if self.config.USE_TEMPORAL_FEATURES:
            temporal_features = [
                'rating_year', 'rating_month', 'rating_day_of_week', 
                'is_weekend', 'rating_delay'
            ]
            feature_columns.extend([col for col in temporal_features if col in df.columns])
        
        # Cast features
        if self.config.USE_ACTORS or self.config.USE_DIRECTORS:
            cast_features = [col for col in df.columns if col.startswith(('actor_', 'director_'))]
            feature_columns.extend(cast_features)
            if 'cast_size_filled' in df.columns:
                feature_columns.append('cast_size_filled')
        
        # Genre features
        if self.config.USE_GENRES:
            # Exclude metadata columns that start with genre_
            genre_features = [
                col for col in df.columns 
                if col.startswith('genre_') 
                and col not in ['genre_count']
            ]
            feature_columns.extend(genre_features)
        
        # Keyword features
        if self.config.USE_KEYWORDS:
            # Exclude metadata columns that start with keyword_
            keyword_features = [
                col for col in df.columns 
                if col.startswith('keyword_') 
                and col not in ['keyword_count']
            ]
            feature_columns.extend(keyword_features)
        
        # Store feature columns for later use
        self.feature_columns = feature_columns
        
        print(f"Selected {len(feature_columns)} features for training")
        return df[feature_columns]
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Train the Random Forest model and evaluate performance.
        
        Performs complete model training including data splitting, model fitting,
        prediction, and comprehensive evaluation with multiple metrics.
        
        Args:
            X (pd.DataFrame): Feature matrix with all prepared features
            y (pd.Series): Target variable (movie ratings)
        
        Returns:
            Dict: Comprehensive metrics dictionary containing:
                Training Metrics:
                - train_mse, train_mae, train_r2: Performance on training set
                
                Test Metrics:
                - test_mse, test_mae, test_r2: Performance on held-out test set
                
                Cross-Validation:
                - cv_accuracy, cv_std: Cross-validated accuracy with std dev
                
                Recommendation Metrics:
                - accuracy_within_0_5: % predictions within ±0.5 rating points
                - accuracy_within_1_0: % predictions within ±1.0 rating points  
                - binary_accuracy: % correct like/dislike classifications
                - precision_top20: Precision for top 20% movie recommendations
                - recall_top20: Recall for top 20% movie recommendations
        
        Training Process:
            1. Encode target ratings using LabelEncoder
            2. Split data into training/test sets (TEST_SIZE ratio)
            3. Fit Random Forest Classifier on training data
            4. Generate predictions for both sets and decode them
            5. Calculate regression metrics (MSE, MAE, R²) for comparison
            6. Perform cross-validation for robust evaluation (Accuracy)
            7. Calculate recommendation-specific accuracy metrics
        
        Model Configuration:
            Uses hyperparameters from config: n_estimators, max_depth,
            min_samples_split, min_samples_leaf, random_state, n_jobs
        
        Note:
            Side effect: Trains self.model which can then be used for predictions
        """
        print("Training Random Forest model...")
        
        # Encode target ratings for classification
        y_encoded = self.rating_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(
            X, y_encoded, test_size=self.config.TEST_SIZE, 
            random_state=self.config.RANDOM_STATE
        )
        
        # Train model
        self.model.fit(X_train, y_train_encoded)
        
        # Make predictions
        y_pred_train_encoded = self.model.predict(X_train)
        y_pred_test_encoded = self.model.predict(X_test)
        
        # Convert back to original rating scale for metrics calculation
        y_train_pred = self.rating_encoder.inverse_transform(y_pred_train_encoded)
        y_pred_test = self.rating_encoder.inverse_transform(y_pred_test_encoded)
        
        # Get original y values for metrics (or inverse transform the split ones)
        y_train = self.rating_encoder.inverse_transform(y_train_encoded)
        y_test = self.rating_encoder.inverse_transform(y_test_encoded)
        
        # Calculate metrics
        metrics = {
            'train_mse': mean_squared_error(y_train, y_train_pred),
            'test_mse': mean_squared_error(y_test, y_pred_test),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_r2': r2_score(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_pred_test)
        };
        
        # Cross-validation (using accuracy for classification)
        cv_scores = cross_val_score(self.model, X, y_encoded, cv=self.config.CV_FOLDS, 
                                   scoring='accuracy')
        metrics['cv_accuracy'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()
        
        # Calculate recommendation accuracy metrics
        metrics.update(self._calculate_recommendation_accuracy(pd.Series(y_test), y_pred_test))
        
        print("Training completed!")
        return metrics
    
    def _calculate_recommendation_accuracy(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict:
        """
        Calculate recommendation-specific accuracy metrics.
        
        Computes various accuracy measures that are more relevant for 
        recommendation systems than traditional regression metrics.
        
        Args:
            y_true (pd.Series): Actual ratings from users
            y_pred (np.ndarray): Predicted ratings from model
        
        Returns:
            Dict: Dictionary containing recommendation accuracy metrics:
                - accuracy_within_0_5: Fraction of predictions within ±0.5 points
                - accuracy_within_1_0: Fraction of predictions within ±1.0 points
                - binary_accuracy: Accuracy for like/dislike classification
                - precision_top20: Precision for top 20% recommendations
                - recall_top20: Recall for top 20% recommendations
        
        Metric Details:
            Threshold-Based Accuracy:
            - Within ±0.5: Very precise predictions (strict threshold)
            - Within ±1.0: Practically useful predictions (lenient threshold)
            
            Binary Classification:
            - Converts ratings to like (≥3.5) vs dislike (<3.5)
            - Measures ability to distinguish good from bad movies
            
            Precision@Top20%:
            - Of movies predicted as top 20%, what % are actually top 20%?
            - Measures recommendation quality (low false positives)
            
            Recall@Top20%:
            - Of actually top 20% movies, what % did we predict as top 20%?
            - Measures recommendation coverage (low false negatives)
        
        Use Cases:
            - accuracy_within_1_0: Best overall "practical accuracy" metric
            - binary_accuracy: Good/bad movie classification performance
            - precision_top20: Quality of high-confidence recommendations
            - recall_top20: Coverage of truly good movies
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
        
        # Calculate Confusion Matrix components
        tp = ((y_true_binary == 1) & (y_pred_binary == 1)).sum()
        tn = ((y_true_binary == 0) & (y_pred_binary == 0)).sum()
        fp = ((y_true_binary == 0) & (y_pred_binary == 1)).sum()
        fn = ((y_true_binary == 1) & (y_pred_binary == 0)).sum()
        
        # Calculate Binary Precision and Recall
        binary_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        binary_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # 4. Recommendation precision@k metrics
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
            'precision_top20': precision_top20,
            'recall_top20': recall_top20
        }
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Extract and rank feature importance from the trained Random Forest model.
        Also calculates grouped importance for binary feature categories.
        
        Returns:
            pd.DataFrame: Feature importance ranking with columns:
                - feature: Feature name (string)
                - importance: Importance score (float, 0-1 range)
                Sorted by importance in descending order
        
        Feature Importance Interpretation:
            - Values sum to 1.0 across all features
            - Higher values indicate more predictive power
            - Based on mean decrease in impurity across all trees
            - Helps identify which features drive recommendations
        
        Raises:
            ValueError: If model hasn't been trained yet or feature_columns not set
        
        Usage:
            Analyze which features are most important for predictions:
            - User behavioral features typically rank highest
            - Movie quality features (vote_average) usually important
            - Content features (genres, cast) provide nuanced preferences
        """
        if self.model is None or self.feature_columns is None:
            raise ValueError("Model must be trained first")
        
        # Get individual feature importance
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Calculate grouped importance for binary feature categories
        feature_groups = {
            'Genre Features': [col for col in self.feature_columns if col.startswith('genre_')],
            'Actor Features': [col for col in self.feature_columns if col.startswith('actor_')],
            'Director Features': [col for col in self.feature_columns if col.startswith('director_')],
            'Keyword Features': [col for col in self.feature_columns if col.startswith('keyword_')],
            'Movie Features': [col for col in self.feature_columns if col in [
                'release_year_filled', 'runtime_filled', 'budget_log', 'revenue_log',
                'vote_average_filled', 'vote_count_log', 'popularity_filled',
                'movie_age', 'profit_log', 'roi',
                'original_language_encoded', 'primary_genre_encoded', 'cast_size_filled'
            ]],
            'User Features': [col for col in self.feature_columns if col.startswith('user_')],
            'Temporal Features': [col for col in self.feature_columns if col in [
                'rating_year', 'rating_month', 'rating_day_of_week', 'is_weekend', 'rating_delay'
            ]]
        }
        
        group_importance = {}
        for group_name, features in feature_groups.items():
            if features:  # Only if group has features
                group_total = importance_df[importance_df['feature'].isin(features)]['importance'].sum()
                group_importance[group_name] = {
                    'total_importance': group_total,
                    'feature_count': len(features),
                    'avg_importance': group_total / len(features) if features else 0
                }
        
        # Store group importance for later use
        self.feature_group_importance = group_importance
        
        return importance_df
    
    def get_feature_group_analysis(self) -> pd.DataFrame:
        """
        Get detailed analysis of feature importance by groups.
        
        Returns:
            pd.DataFrame: Group analysis with columns:
                - group: Feature group name
                - total_importance: Sum of importance for all features in group
                - feature_count: Number of features in group
                - avg_importance: Average importance per feature in group
                - importance_percentage: Percentage of total model importance
        
        Note:
            Must call get_feature_importance() first to populate group data.
        """
        if not hasattr(self, 'feature_group_importance'):
            raise ValueError("Must call get_feature_importance() first to calculate group importance")
        
        group_data = []
        for group_name, stats in self.feature_group_importance.items():
            group_data.append({
                'group': group_name,
                'total_importance': stats['total_importance'],
                'feature_count': stats['feature_count'],
                'avg_importance': stats['avg_importance'],
                'importance_percentage': stats['total_importance'] * 100
            })
        
        group_df = pd.DataFrame(group_data)
        return group_df.sort_values('total_importance', ascending=False)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate rating predictions using the trained Random Forest model.
        
        Args:
            X (pd.DataFrame): Feature matrix with same columns as training data.
                             Must contain all features specified in self.feature_columns
        
        Returns:
            np.ndarray: Predicted ratings (discrete values, e.g. 0.5, 1.0, ..., 5.0)
        
        Raises:
            ValueError: If model hasn't been trained yet
        
        Note:
            Predictions represent expected user ratings for movies.
            Use for generating recommendations by ranking predicted ratings.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
            
        # Get encoded predictions
        y_pred_encoded = self.model.predict(X)
        
        # Convert back to original rating scale
        return self.rating_encoder.inverse_transform(y_pred_encoded)
    
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
                pd.Series(y_true_user), y_pred_user
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
        
        Analyzes genre preferences by calculating average ratings per genre
        for movies the user has rated, identifying genres they consistently rate highly.
        
        Args:
            user_movies (pd.DataFrame): User's movie rating history with 'genres' 
                                      and 'rating' columns
            top_n (int): Number of top genres to return. Defaults to 3.
        
        Returns:
            List[str]: List of favorite genre names, ordered by preference
                      (highest average rating first)
        
        Algorithm:
            1. Parse genres from each movie the user rated
            2. Group ratings by genre and calculate average rating per genre
            3. Filter genres with at least 2 movies (minimum statistical significance)
            4. Sort by average rating and return top N genres
        
        Example:
            User rates: Action movies avg 4.2, Comedy avg 3.8, Horror avg 2.1
            Returns: ['Action', 'Comedy'] (if top_n=2, Horror excluded for low rating)
        
        Use Cases:
            - User profiling and preference analysis
            - Genre-based recommendation filtering
            - Understanding user taste patterns
        
        Note:
            Requires at least 2 movies per genre for statistical reliability.
            Handles malformed genre data gracefully.
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
    temporal_features = ['rating_year', 'rating_month', 'rating_day_of_week', 'is_weekend', 'rating_delay']
    
    grouped_counts = {k: 0 for k in groups}
    grouped_counts['Temporal'] = 0
    other_features = []
    
    for feature in features:
        matched = False
        for group, prefix in groups.items():
            if feature.startswith(prefix):
                grouped_counts[group] += 1
                matched = True
                break
        
        if not matched:
            if feature in temporal_features:
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


def main():
    """
    Main execution function for the Random Forest Movie Recommender System.
    
    Orchestrates the complete machine learning pipeline including data loading,
    preprocessing, feature engineering, model training, evaluation, and 
    demonstration of personalized recommendations.
    
    Returns:
        Tuple[RandomForestRecommender, pd.DataFrame, Dict]: 
            - recommender: Trained model instance
            - merged_data: Processed dataset with all features
            - metrics: Training and evaluation metrics
    
    Pipeline Steps:
        1. Data Loading: Load movies, credits, keywords, and ratings datasets
        2. Data Filtering: Apply reliability filters (user/movie activity thresholds)
        3. Dataset Merging: Combine all data sources with year filtering (post-2000)
        4. Feature Engineering: Create all feature types (genres, cast, keywords, etc.)
        5. Model Training: Train Random Forest with comprehensive evaluation
        6. Results Display: Show metrics, feature importance, and user demos
    
    Output Sections:
        - Training Results: MSE, MAE, R² for train/test sets
        - Recommendation Accuracy: Threshold-based and classification accuracy
        - Feature Importance: Top 20 most predictive features
        - User Demo: Individual user recommendation examples
    
    Configuration:
        All parameters controlled via MovieRecommenderConfig class including:
        - Data filtering thresholds
        - Feature engineering options  
        - Random Forest hyperparameters
        - Evaluation settings
    """
    print("=== Random Forest Movie Recommender ===\n")
    
    # Initialize configuration
    config = MovieRecommenderConfig()
    processor = DataProcessor(config)
    
    # Load and process data
    movies, credits, keywords, ratings, links = processor.load_data()
    
    # Filter reliable data
    ratings_filtered = processor.filter_reliable_data(ratings)
    
    # Merge datasets
    merged_data = processor.merge_datasets(movies, credits, keywords, ratings_filtered, links)
    
    # Apply feature engineering
    print("\n=== Feature Engineering ===")
    merged_data = processor.create_genre_features(merged_data)
    merged_data = processor.create_cast_features(merged_data)
    merged_data = processor.create_keyword_features(merged_data)
    merged_data = processor.create_movie_features(merged_data)
    merged_data = processor.create_user_features(merged_data)
    merged_data = processor.create_temporal_features(merged_data)
    
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
    print(f"CV Accuracy: {metrics['cv_accuracy']:.4f} (±{metrics['cv_std']:.4f})")
    
    # Display recommendation accuracy metrics
    print("\n=== Recommendation Accuracy Metrics ===")
    print(f"Accuracy within ±0.5 points: {metrics['accuracy_within_0_5']:.4f} ({metrics['accuracy_within_0_5']*100:.1f}%)")
    print(f"Accuracy within ±1.0 points: {metrics['accuracy_within_1_0']:.4f} ({metrics['accuracy_within_1_0']*100:.1f}%)")
    print(f"Binary classification accuracy: {metrics['binary_accuracy']:.4f} ({metrics['binary_accuracy']*100:.1f}%)")
    
    # Display Confusion Matrix
    total_predictions = metrics['tp'] + metrics['tn'] + metrics['fp'] + metrics['fn']
    print(f"\nConfusion Matrix (Threshold=3.5):")
    print(f"True Positives (TP): {metrics['tp']} ({metrics['tp']/total_predictions*100:.1f}%)")
    print(f"False Positives (FP): {metrics['fp']} ({metrics['fp']/total_predictions*100:.1f}%)")
    print(f"True Negatives (TN): {metrics['tn']} ({metrics['tn']/total_predictions*100:.1f}%)")
    print(f"False Negatives (FN): {metrics['fn']} ({metrics['fn']/total_predictions*100:.1f}%)")
    
    print(f"\nBinary Precision (Threshold=3.5): {metrics['binary_precision']:.4f} ({metrics['binary_precision']*100:.1f}%)")
    print(f"Binary Recall (Threshold=3.5): {metrics['binary_recall']:.4f} ({metrics['binary_recall']*100:.1f}%)")
    
    print(f"\nPrecision@Top20%: {metrics['precision_top20']:.4f} ({metrics['precision_top20']*100:.1f}%)")
    print(f"Recall@Top20%: {metrics['recall_top20']:.4f} ({metrics['recall_top20']*100:.1f}%)")
    
    # Display feature importance analysis
    print("\n=== Feature Importance Analysis ===")
    
    # Get feature importance (MDI)
    feature_importance = recommender.get_feature_importance()
    
    # Format for display (match Naive Bayes visual style)
    feature_importance['importance_pct'] = feature_importance['importance'] * 100
    
    print("\nTop 15 Features (MDI Importance %):")
    print(feature_importance.head(15)[['feature', 'importance_pct']].to_string(index=False, float_format='%.2f%%'))
    
    # Get group analysis
    group_df = recommender.get_feature_group_analysis()
    
    # Format group stats to match Naive Bayes columns
    # Naive Bayes: ['Feature Group', 'Total Importance %', 'Feature Count']
    group_stats = group_df[['group', 'importance_percentage', 'feature_count']].copy()
    group_stats.columns = ['Feature Group', 'Total Importance %', 'Feature Count']
    
    print("\n=== Feature Group Importance (MDI) ===")
    print(group_stats.to_string(index=False, float_format='%.2f%%'))
    
    # Display standardized feature summary
    print_feature_summary(recommender.feature_columns)
    
    # Save feature list to file
    with open('random_forest_features.txt', 'w') as f:
        for feature in recommender.feature_columns:
            f.write(f"{feature}\n")
    print(f"Feature list saved to random_forest_features.txt")
    
    # Demonstrate user recommendations
    print("\n=== User Recommendation Demo ===")
    demo_user_recommendations(recommender, merged_data)
    
    # Print feature summary
    print_feature_summary(recommender.feature_columns)
    
    return recommender, merged_data, metrics


def demo_user_recommendations(recommender: RandomForestRecommender, data: pd.DataFrame, 
                            num_users: int = 3):
    """
    Demonstrate the recommendation system with real user examples.
    
    Selects random users from the dataset and generates personalized recommendations,
    showing how the model performs on individual users with detailed analysis.
    
    Args:
        recommender (RandomForestRecommender): Trained recommendation model
        data (pd.DataFrame): Complete dataset with user ratings and features
        num_users (int): Number of users to demonstrate. Defaults to 3.
    
    Demo Output for Each User:
        User Profile:
        - Total movies rated, average rating, rating variability
        - Favorite genres based on rating patterns
        
        Movie History:
        - Top 5 highest-rated movies in their training set
        - Shows user's demonstrated preferences
        
        Prediction Performance:
        - MAE and accuracy metrics for this specific user
        - How well the model predicts their ratings
        
        Specific Predictions:
        - Side-by-side comparison of predicted vs actual ratings
        - Shows model performance on individual movies
    
    User Selection Criteria:
        - Only includes users with ≥10 ratings for statistical significance
        - Random selection ensures diverse user types
        - Handles edge cases gracefully
    
    Use Cases:
        - Model validation with real examples
        - Understanding model behavior for different user types
        - Demonstrating recommendation quality to stakeholders
        - Identifying model strengths and weaknesses
    
    Note:
        Uses 70/30 split: 70% of user's movies for "history", 30% for evaluation.
        Provides concrete examples of how the system would work in practice.
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


if __name__ == "__main__":
    recommender, data, metrics = main()
