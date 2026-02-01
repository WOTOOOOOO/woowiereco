"""
Content-Based Recommender System
A modular implementation for movie recommendations using content features
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
import ast
import warnings
from typing import Dict, List, Tuple

warnings.filterwarnings('ignore')

# Configuration Constants
class Config:
    """Configuration class for all hyperparameters and settings"""
    
    # Data paths
    DATA_DIR = "cleaned_data/"
    MOVIES_FILE = "movies_metadata_cleaned.csv"
    CREDITS_FILE = "credits_cleaned.csv"
    KEYWORDS_FILE = "keywords_cleaned.csv"
    RATINGS_FILE = "ratings_cleaned.csv"
    LINKS_FILE = "links.csv"
    
    # Splitting parameters
    TRAIN_RATIO = 0.8  # 80% of user ratings for training
    MIN_USER_RATINGS = 10  # Minimum ratings per user to include
    
    # User sampling parameters
    MAX_USERS = 10000  # Maximum number of users to process
    USER_SAMPLE_SEED = 42  # Random seed for user sampling
    
    # Feature engineering parameters
    MIN_RELEASE_YEAR = 2000 # Only include movies released after this year
    # Feature counts (Set to None to use all available features)
    # Set to 0 to effectively disable generation of these features
    MAX_GENRES = 15
    MAX_ACTORS = 20
    MAX_DIRECTORS = 10
    MAX_KEYWORDS = 25
    MAX_TFIDF_FEATURES = 50
    MAX_COMPANIES = 0
    MAX_STUDIOS = 0
    MIN_DF = 5
    MAX_DF = 0.80
    
    # Content features weights
    # Set any weight to 0.0 to completely disable the feature
    GENRE_WEIGHT = 0.25
    KEYWORD_WEIGHT = 0.20
    CAST_WEIGHT = 0.15
    DIRECTOR_WEIGHT = 0.10
    PRODUCTION_COMPANIES_WEIGHT = 0
    MAIN_STUDIO_WEIGHT = 0
    LANGUAGE_WEIGHT = 0.02
    NUMERICAL_WEIGHT = 0.15
    OVERVIEW_WEIGHT = 0.15
    
    # Recommendation parameters
    TOP_K_RECOMMENDATIONS = 15
    SIMILARITY_THRESHOLD = 0.01  # Minimum similarity to consider
    
    # Evaluation parameters
    RATING_THRESHOLD = 3.5  # Min rating to count as "positive" for enjoyment metrics
    EVAL_PROGRESS_INTERVAL = 100  # Print progress every N users
    
    @classmethod
    def print_config(cls):
        """Print current configuration settings"""
        print("\nCurrent Configuration:")
        print("=" * 30)
        print(f"Max Users: {cls.MAX_USERS if cls.MAX_USERS else 'All users'}")
        print(f"Min User Ratings: {cls.MIN_USER_RATINGS}")
        print(f"Train Ratio: {cls.TRAIN_RATIO}")
        print(f"Feature Weights: Genre={cls.GENRE_WEIGHT}, Keyword={cls.KEYWORD_WEIGHT}, "
              f"Cast={cls.CAST_WEIGHT}, Director={cls.DIRECTOR_WEIGHT}")
        print(f"                 Production={cls.PRODUCTION_COMPANIES_WEIGHT}, Studio={cls.MAIN_STUDIO_WEIGHT}, "
              f"Language={cls.LANGUAGE_WEIGHT}, "
              f"Numerical={cls.NUMERICAL_WEIGHT}, Overview={cls.OVERVIEW_WEIGHT}")
        print(f"Top K Recommendations: {cls.TOP_K_RECOMMENDATIONS}")
        print(f"Evaluation: Threshold={cls.RATING_THRESHOLD}")
        print()

class DataLoader:
    """
    Handles loading, preprocessing, and merging of movie metadata, ratings, and feature datasets.
    """
    
    @staticmethod
    def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all required datasets from CSV files defined in Config.
        
        Reads the following files:
        - Movies Metadata (movies_metadata_cleaned.csv)
        - Credits (credits_cleaned.csv)
        - Keywords (keywords_cleaned.csv)
        - Ratings (ratings_cleaned.csv)
        - Links (links.csv) for ID mapping
        
        Returns:
            Tuple containing five DataFrames (movies, credits, keywords, ratings, links).
        """
        print("Loading datasets...")
        
        movies_df = pd.read_csv(f"{Config.DATA_DIR}{Config.MOVIES_FILE}")
        credits_df = pd.read_csv(f"{Config.DATA_DIR}{Config.CREDITS_FILE}")
        keywords_df = pd.read_csv(f"{Config.DATA_DIR}{Config.KEYWORDS_FILE}")
        ratings_df = pd.read_csv(f"{Config.DATA_DIR}{Config.RATINGS_FILE}")
        links_df = pd.read_csv(f"{Config.DATA_DIR}{Config.LINKS_FILE}")
        
        print(f"Loaded {len(movies_df)} movies, {len(credits_df)} credit records, "
              f"{len(keywords_df)} keyword records, {len(ratings_df)} ratings, {len(links_df)} links")
        
        return movies_df, credits_df, keywords_df, ratings_df, links_df
    
    @staticmethod
    def merge_movie_features(movies_df: pd.DataFrame, credits_df: pd.DataFrame, 
                           keywords_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge secondary feature datasets (credits, keywords) into the main movie DataFrame.
        
        Performs left joins on 'id', handles missing values for list-like columns 
        (keywords, actors, directors) by filling with empty list strings, and 
        applies the release year filter.
        
        Args:
            movies_df (pd.DataFrame): Main movies metadata.
            credits_df (pd.DataFrame): Credits data containing cast and crew.
            keywords_df (pd.DataFrame): Keywords data.
            
        Returns:
            pd.DataFrame: A single consolidated DataFrame with all movie features.
        """
        print("Merging movie features...")
        
        # Merge on movie id
        merged_df = movies_df.merge(credits_df, on='id', how='left')
        merged_df = merged_df.merge(keywords_df, on='id', how='left')
        
        # Fill missing values
        merged_df['keywords'] = merged_df['keywords'].fillna('[]')
        merged_df['actors'] = merged_df['actors'].fillna('[]')
        merged_df['directors'] = merged_df['directors'].fillna('[]')
        merged_df['overview'] = merged_df['overview'].fillna('')
        
        # Filter by release year if configured
        if hasattr(Config, 'MIN_RELEASE_YEAR'):
            print(f"Filtering movies released after {Config.MIN_RELEASE_YEAR}...")
            initial_count = len(merged_df)
            # Ensure release_year is numeric
            merged_df['release_year'] = pd.to_numeric(merged_df['release_year'], errors='coerce').fillna(0)
            merged_df = merged_df[merged_df['release_year'] >= Config.MIN_RELEASE_YEAR].copy()
            print(f"Filtered from {initial_count} to {len(merged_df)} movies")
        
        print(f"Merged dataset has {len(merged_df)} movies with features")
        return merged_df

class DataSplitter:
    """
    Manages user sampling, filtering, and train/test splitting of rating data.
    """
    
    @staticmethod
    def sample_users(ratings_df: pd.DataFrame) -> pd.DataFrame:
        """
        Randomly sample a subset of users to reduce dataset size for development/testing.
        
        If Config.MAX_USERS is set, randomly selects that many unique users from the 
        dataset. If None or if dataset is smaller than the limit, returns all users.
        
        Args:
            ratings_df (pd.DataFrame): The full ratings DataFrame.
            
        Returns:
            pd.DataFrame: Ratings DataFrame containing only the sampled users.
        """
        if Config.MAX_USERS is None:
            print("Processing all users (no limit specified)")
            return ratings_df
        
        unique_users = ratings_df['userId'].unique()
        total_users = len(unique_users)
        
        if total_users <= Config.MAX_USERS:
            print(f"Dataset has {total_users} users, which is <= {Config.MAX_USERS}. Using all users.")
            return ratings_df
        
        # Sample users randomly
        np.random.seed(Config.USER_SAMPLE_SEED)
        sampled_users = np.random.choice(unique_users, size=Config.MAX_USERS, replace=False)
        
        sampled_ratings = ratings_df[ratings_df['userId'].isin(sampled_users)]
        
        print(f"Sampled {Config.MAX_USERS} users from {total_users} total users")
        print(f"Sampled ratings: {len(sampled_ratings)} from {len(ratings_df)} total ratings")
        
        return sampled_ratings
    
    @staticmethod
    def filter_users(ratings_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter out users who have too few ratings.
        
        Removes users with fewer than Config.MIN_USER_RATINGS interactions to ensure
        sufficient data for building reliable user profiles.
        
        Args:
            ratings_df (pd.DataFrame): The ratings DataFrame (potentially sampled).
            
        Returns:
            pd.DataFrame: Filtered ratings DataFrame.
        """
        user_counts = ratings_df['userId'].value_counts()
        valid_users = user_counts[user_counts >= Config.MIN_USER_RATINGS].index
        
        filtered_ratings = ratings_df[ratings_df['userId'].isin(valid_users)]
        
        print(f"Filtered to {len(valid_users)} users with >= {Config.MIN_USER_RATINGS} ratings")
        print(f"Remaining ratings: {len(filtered_ratings)}")
        
        return filtered_ratings
    
    @staticmethod
    def split_user_ratings(ratings_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split each user's rating history into training and testing sets.
        
        Uses a temporal split if 'rating_year' is available:
        - Sorts user's ratings by year.
        - First (Traing_Ratio)% (e.g. 80%) -> Training (Past).
        - Last (1-Train_Ratio)% (e.g. 20%) -> Testing (Future).
        
        Args:
            ratings_df (pd.DataFrame): Filtered ratings DataFrame.
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (Train DataFrame, Test DataFrame).
        """
        print("Splitting user ratings (Temporal Split)...")
        
        train_list = []
        test_list = []
        
        # Ensure data is sorted for consistent splitting
        # Although we sort per user below, global sort helps debug stability
        if 'rating_year' in ratings_df.columns:
            print("Using time-based splitting based on 'rating_year'.")
        else:
            print("Warning: 'rating_year' not found. Falling back to simple random split.")
            
        for user_id in ratings_df['userId'].unique():
            user_ratings = ratings_df[ratings_df['userId'] == user_id]
            
            if 'rating_year' in user_ratings.columns:
                # TEMPORAL SPLIT: Sort by year, then random shuffle within same year to break ties
                # (Since we only have 'year', not exact timestamp, this is the best approximation)
                user_ratings = user_ratings.sort_values(by='rating_year', kind='mergesort')
                
                split_idx = int(len(user_ratings) * Config.TRAIN_RATIO)
                
                # Take first N as training (older ratings)
                train_ratings = user_ratings.iloc[:split_idx]
                # Take last M as testing (newer ratings)
                test_ratings = user_ratings.iloc[split_idx:]
            else:
                # RANDOM SPLIT (Fallback)
                user_ratings = user_ratings.sample(frac=1, random_state=42)
                split_idx = int(len(user_ratings) * Config.TRAIN_RATIO)
                train_ratings = user_ratings.iloc[:split_idx]
                test_ratings = user_ratings.iloc[split_idx:]
            
            train_list.append(train_ratings)
            test_list.append(test_ratings)
        
        train_df = pd.concat(train_list, ignore_index=True)
        test_df = pd.concat(test_list, ignore_index=True)
        
        print(f"Train set: {len(train_df)} ratings, Test set: {len(test_df)} ratings")
        
        return train_df, test_df

class FeatureEngineer:
    """
    Responsible for transforming raw movie metadata into a numerical feature matrix.
    Uses One-Hot/Multi-Hot encoding for categorical data and TF-IDF for text.
    """
    
    def __init__(self):
        self.genre_encoder = MultiLabelBinarizer()
        self.keyword_encoder = MultiLabelBinarizer()
        self.cast_encoder = MultiLabelBinarizer()
        self.director_encoder = MultiLabelBinarizer()
        self.production_companies_encoder = MultiLabelBinarizer()
        self.main_studio_encoder = MultiLabelBinarizer()
        self.language_encoder = MultiLabelBinarizer()
        
        if Config.OVERVIEW_WEIGHT > 0 and (Config.MAX_TFIDF_FEATURES is None or Config.MAX_TFIDF_FEATURES > 0):
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=Config.MAX_TFIDF_FEATURES,
                min_df=Config.MIN_DF,
                max_df=Config.MAX_DF,
                stop_words='english'
            )
        else:
            self.tfidf_vectorizer = None
            
        self.scaler = StandardScaler()
        self.feature_names = []
    
    def safe_eval(self, x):
        """Safely evaluate string representations of lists"""
        if pd.isna(x) or x == '[]':
            return []
        try:
            return ast.literal_eval(x)
        except:
            return []
    
    def create_feature_vectors(self, movies_df: pd.DataFrame) -> np.ndarray:
        """
        Generate the final feature matrix for all movies.
        
        Processes enabled features independently (Genres, Keywords, Cast, Director, etc.)
        using MultiLabelBinarizer or OneHotEncoder, and scales numerical features.
        Concatenates all partial feature matrices into one large dense/sparse matrix.
        
        Args:
            movies_df (pd.DataFrame): DataFrame containing prepared movie data.
            
        Returns:
            np.ndarray: A 2D array where rows are movies and columns are features.
        """
        print("\n=== Feature Engineering ===")
        
        feature_vectors = []
        
        # 1. GENRE FEATURES (Multi-hot encoding)
        if Config.GENRE_WEIGHT > 0:
            print("Processing genres...")
            movies_df['genre_list'] = movies_df['genres'].apply(self.safe_eval)
            
            # Filter to common genres to reduce noise
            all_genres = [genre for genres in movies_df['genre_list'] for genre in genres]
            genre_counts = pd.Series(all_genres).value_counts()
            
            if Config.MAX_GENRES is not None and Config.MAX_GENRES > 0:
                top_genres = genre_counts.head(Config.MAX_GENRES).index.tolist()
            elif Config.MAX_GENRES == 0:
                top_genres = []
            else:
                top_genres = genre_counts.index.tolist()
            
            movies_df['filtered_genres'] = movies_df['genre_list'].apply(
                lambda x: [g for g in x if g in top_genres]
            )
            
            genre_features = self.genre_encoder.fit_transform(movies_df['filtered_genres'])
            feature_vectors.append(genre_features * Config.GENRE_WEIGHT)
            print(f"Genre features: {genre_features.shape[1]} categories")
        
        # 2. KEYWORD FEATURES (Multi-hot encoding)
        if Config.KEYWORD_WEIGHT > 0:
            print("Processing keywords...")
            movies_df['keyword_list'] = movies_df['keywords'].apply(self.safe_eval)
            
            # Select discriminative keywords
            all_keywords = [kw for keywords in movies_df['keyword_list'] for kw in keywords]
            keyword_counts = pd.Series(all_keywords).value_counts()
            
            # Use top N keywords as per NB config
            if Config.MAX_KEYWORDS is not None and Config.MAX_KEYWORDS > 0:
                good_keywords = keyword_counts.head(Config.MAX_KEYWORDS).index.tolist()
            elif Config.MAX_KEYWORDS == 0:
                 good_keywords = []
            else:
                good_keywords = keyword_counts.index.tolist()
            
            movies_df['filtered_keywords'] = movies_df['keyword_list'].apply(
                lambda x: [kw for kw in x if kw in good_keywords]
            )
            
            keyword_features = self.keyword_encoder.fit_transform(movies_df['filtered_keywords'])
            feature_vectors.append(keyword_features * Config.KEYWORD_WEIGHT)
            print(f"Keyword features: {keyword_features.shape[1]} categories")
        
        # 3. CAST FEATURES (Multi-hot encoding for top actors)
        if Config.CAST_WEIGHT > 0:
            print("Processing cast...")
            movies_df['actor_list'] = movies_df['actors'].apply(self.safe_eval)
            movies_df['top_actors'] = movies_df['actor_list'].apply(lambda x: x[:3])  # Top 3 actors
            
            # Select popular actors (Top N)
            all_actors = [actor for actors in movies_df['top_actors'] for actor in actors]
            actor_counts = pd.Series(all_actors).value_counts()
            
            if Config.MAX_ACTORS is not None and Config.MAX_ACTORS > 0:
                popular_actors = actor_counts.head(Config.MAX_ACTORS).index.tolist()
            elif Config.MAX_ACTORS == 0:
                popular_actors = []
            else:
                popular_actors = actor_counts.index.tolist()
            
            movies_df['filtered_actors'] = movies_df['top_actors'].apply(
                lambda x: [actor for actor in x if actor in popular_actors]
            )
            
            cast_features = self.cast_encoder.fit_transform(movies_df['filtered_actors'])
            feature_vectors.append(cast_features * Config.CAST_WEIGHT)
            print(f"Cast features: {cast_features.shape[1]} categories")
        
        # 4. DIRECTOR FEATURES (Multi-hot encoding)
        if Config.DIRECTOR_WEIGHT > 0:
            print("Processing directors...")
            movies_df['director_list'] = movies_df['directors'].apply(self.safe_eval)
            
            # Select directors (Top N)
            all_directors = [director for directors in movies_df['director_list'] for director in directors]
            director_counts = pd.Series(all_directors).value_counts()
            
            if Config.MAX_DIRECTORS is not None and Config.MAX_DIRECTORS > 0:
                prolific_directors = director_counts.head(Config.MAX_DIRECTORS).index.tolist()
            elif Config.MAX_DIRECTORS == 0:
                prolific_directors = []
            else:
                prolific_directors = director_counts.index.tolist()
            
            movies_df['filtered_directors'] = movies_df['director_list'].apply(
                lambda x: [director for director in x if director in prolific_directors]
            )
            
            director_features = self.director_encoder.fit_transform(movies_df['filtered_directors'])
            feature_vectors.append(director_features * Config.DIRECTOR_WEIGHT)
            print(f"Director features: {director_features.shape[1]} categories")
        
        # 5. PRODUCTION COMPANIES (Multi-hot encoding)
        if Config.PRODUCTION_COMPANIES_WEIGHT > 0:
            print("Processing production companies...")
            movies_df['production_companies_list'] = movies_df['production_companies'].apply(self.safe_eval)
            movies_df['top_companies'] = movies_df['production_companies_list'].apply(lambda x: x[:2])  # Top 2 companies
            
            # Select companies with multiple movies
            all_companies = [company for companies in movies_df['top_companies'] for company in companies]
            company_counts = pd.Series(all_companies).value_counts()
            
            if Config.MAX_COMPANIES is not None and Config.MAX_COMPANIES > 0:
                major_companies = company_counts[company_counts >= 3].head(Config.MAX_COMPANIES).index.tolist()
            elif Config.MAX_COMPANIES == 0:
                major_companies = []
            else:
                major_companies = company_counts[company_counts >= 3].index.tolist()
                
            movies_df['filtered_companies'] = movies_df['top_companies'].apply(
                lambda x: [company for company in x if company in major_companies]
            )
            
            company_features = self.production_companies_encoder.fit_transform(movies_df['filtered_companies'])
            feature_vectors.append(company_features * Config.PRODUCTION_COMPANIES_WEIGHT)
            print(f"Production company features: {company_features.shape[1]} categories")
        
        # 6. MAIN STUDIO (One-hot encoding)
        if Config.MAIN_STUDIO_WEIGHT > 0:
            print("Processing main studio...")
            main_studios = movies_df['main_studio'].fillna('Unknown')
            
            # Select studios with multiple movies
            studio_counts = main_studios.value_counts()
            
            if Config.MAX_STUDIOS is not None and Config.MAX_STUDIOS > 0:
                major_studios = studio_counts[studio_counts >= 5].head(Config.MAX_STUDIOS).index.tolist()
            elif Config.MAX_STUDIOS == 0:
                major_studios = []
            else:
                major_studios = studio_counts[studio_counts >= 5].index.tolist()
                
            movies_df['filtered_studio'] = main_studios.apply(
                lambda x: [x] if x in major_studios else ['Other']
            )
            
            studio_features = self.main_studio_encoder.fit_transform(movies_df['filtered_studio'])
            feature_vectors.append(studio_features * Config.MAIN_STUDIO_WEIGHT)
            print(f"Studio features: {studio_features.shape[1]} categories")
        
        # 7. LANGUAGE (Multi-hot encoding)
        if Config.LANGUAGE_WEIGHT > 0:
            print("Processing languages...")
            # Just encode all languages as requested
            movies_df['language_list'] = movies_df['original_language'].fillna('unknown').apply(lambda x: [x])
            
            language_features = self.language_encoder.fit_transform(movies_df['language_list'])
            feature_vectors.append(language_features * Config.LANGUAGE_WEIGHT)
            print(f"Language features: {language_features.shape[1]} categories")
        
        # 8. TEXT FEATURES (TF-IDF on Overview)
        if Config.OVERVIEW_WEIGHT > 0 and self.tfidf_vectorizer is not None:
            print("Processing overview text...")
            movies_df['overview'] = movies_df['overview'].fillna('')
            tfidf_features = self.tfidf_vectorizer.fit_transform(movies_df['overview']).toarray()
            feature_vectors.append(tfidf_features * Config.OVERVIEW_WEIGHT)
            print(f"TF-IDF features: {tfidf_features.shape[1]} terms")

        # 9. NUMERICAL FEATURES
        if Config.NUMERICAL_WEIGHT > 0:
            print("Processing numerical features...")
            numerical_features = []
            
            # Basic descriptive numerical features (No complex log features)
            if 'vote_average' in movies_df.columns: 
                vote_avg = movies_df['vote_average'].fillna(movies_df['vote_average'].mean()).values
                numerical_features.append(vote_avg)
                self.feature_names.append("vote_average")
            
            if 'popularity' in movies_df.columns: 
                popularity = movies_df['popularity'].fillna(0).values
                numerical_features.append(popularity)
                self.feature_names.append("popularity")
            
            if 'release_year' in movies_df.columns: 
                numerical_features.append(pd.to_numeric(movies_df['release_year'], errors='coerce').fillna(2000).values)
                self.feature_names.append("release_year")
            
            if 'runtime' in movies_df.columns: 
                runtime = movies_df['runtime'].fillna(movies_df['runtime'].mean()).values
                numerical_features.append(runtime)
                self.feature_names.append("runtime")
            
            # Adult flag
            adult_flag = movies_df['adult'].fillna(False).astype(int).values
            numerical_features.append(adult_flag)
            self.feature_names.append("adult")
            
            if numerical_features:
                numerical_matrix = np.column_stack(numerical_features)
                
                # Replace NaNs and Infs
                numerical_matrix = np.nan_to_num(numerical_matrix, nan=0.0, posinf=0.0, neginf=0.0)
                
                numerical_matrix = self.scaler.fit_transform(numerical_matrix)
                feature_vectors.append(numerical_matrix * Config.NUMERICAL_WEIGHT) 
                print(f"Specific features added: {len(self.feature_names) - len(prefix_features) if 'prefix_features' in locals() else 'many'}")
        
        # Combine all features
        if not feature_vectors:
             raise ValueError("No features selected! Please enable at least one feature type with weight > 0.")
             
        combined_features = np.hstack(feature_vectors)
        
        # Collect feature names in correct order - categorical
        prefix_features = []
        if Config.GENRE_WEIGHT > 0 and hasattr(self.genre_encoder, 'classes_'):
             prefix_features.extend([f"Genre: {c}" for c in self.genre_encoder.classes_])
        if Config.KEYWORD_WEIGHT > 0 and hasattr(self.keyword_encoder, 'classes_'):
             prefix_features.extend([f"Keyword: {c}" for c in self.keyword_encoder.classes_])
        if Config.CAST_WEIGHT > 0 and hasattr(self.cast_encoder, 'classes_'):
             prefix_features.extend([f"Cast: {c}" for c in self.cast_encoder.classes_])
        if Config.DIRECTOR_WEIGHT > 0 and hasattr(self.director_encoder, 'classes_'):
             prefix_features.extend([f"Director: {c}" for c in self.director_encoder.classes_])
        if Config.PRODUCTION_COMPANIES_WEIGHT > 0 and hasattr(self.production_companies_encoder, 'classes_'):
             prefix_features.extend([f"Company: {c}" for c in self.production_companies_encoder.classes_])
        if Config.MAIN_STUDIO_WEIGHT > 0 and hasattr(self.main_studio_encoder, 'classes_'):
             prefix_features.extend([f"Studio: {c}" for c in self.main_studio_encoder.classes_])
        if Config.LANGUAGE_WEIGHT > 0 and hasattr(self.language_encoder, 'classes_'):
             prefix_features.extend([f"Language: {c}" for c in self.language_encoder.classes_])
             
        # Prepend categorical to numerical/specific names
        self.feature_names = prefix_features + self.feature_names

        print(f"Final dataset shape: {combined_features.shape}")
        
        return combined_features

    def print_feature_list(self):
        """Print grouped feature summary matching the requested format"""
        print("\n=== Feature Summary ===")
        
        # Group features by prefix
        groups = {}
        specific_features = []
        
        for name in self.feature_names:
            if ": " in name:
                prefix, _ = name.split(": ", 1)
                groups[prefix] = groups.get(prefix, 0) + 1
            else:
                specific_features.append(name)
        
        print("Grouped Features:")
        for prefix, count in groups.items():
            print(f"  - {count} {prefix} features")
            
        print("\nSpecific Features:")
        for name in specific_features:
            print(f"  - {name}")
            
        print(f"\nTotal Features: {len(self.feature_names)}")
        

class UserProfiler:
    """
    Builds vector representations of user preferences based on their watch history.
    """
    
    @staticmethod
    def create_user_profiles(train_ratings: pd.DataFrame, 
                           movie_features: np.ndarray,
                           movie_id_to_idx: Dict[int, int]) -> Dict[int, np.ndarray]:
        """
        Generate a preference profile vector for each user.
        
        The profile is a weighted average of the feature vectors of movies the user has rated.
        Weights are centered around the neutral rating (3.0):
        - Rating 5.0 -> Weight +2.0 (Move profile towards this movie's features)
        - Rating 3.0 -> Weight  0.0 (Neutral, no impact)
        - Rating 1.0 -> Weight -2.0 (Move profile away from this movie's features)
        
        Args:
            train_ratings (pd.DataFrame): Training set ratings.
            movie_features (np.ndarray): The global movie feature matrix.
            movie_id_to_idx (Dict[int, int]): Mapping from real Movie ID to matrix row index.
            
        Returns:
            Dict[int, np.ndarray]: Dictionary mapping User ID to their 1D profile vector.
        """
        print("Creating user profiles...")
        
        user_profiles = {}
        
        # Group ratings by user
        user_groups = train_ratings.groupby('userId')
        
        for user_id, params in user_groups:
            # Filter for movies that exist in our feature matrix
            valid_interactions = []
            
            for _, row in params.iterrows():
                mid = row['movieId']
                rating = row['rating']
                if mid in movie_id_to_idx:
                    valid_interactions.append((movie_id_to_idx[mid], rating))
            
            if not valid_interactions:
                continue
                
            # Separate indices and ratings
            indices = [x[0] for x in valid_interactions]
            ratings = np.array([x[1] for x in valid_interactions])
            
            # Weighting Strategy: Center the ratings around neutral (3.0)
            # 5.0 -> +2.0 (Strong Positive)
            # 3.0 ->  0.0 (Neutral)
            # 1.0 -> -2.0 (Strong Negative)
            weights = ratings - 3.0
            
            # Reshape weights for broadcasting: (n_samples, 1)
            weights = weights[:, np.newaxis]
            
            # Get feature vectors: (n_samples, n_features)
            vectors = movie_features[indices]
            
            # Compute weighted sum
            # Negative weights will flip the feature vector direction, effectively
            # "repelling" the profile from features of hated movies.
            weighted_vectors = vectors * weights
            user_profile = np.sum(weighted_vectors, axis=0)
            
            # Normalize by sum of absolute weights to keep scale reasonable
            # (avoid profile vector growing infinitely with more ratings)
            weight_sum = np.sum(np.abs(weights))
            if weight_sum > 0:
                user_profile = user_profile / weight_sum
                
            user_profiles[user_id] = user_profile
        
        print(f"Created profiles for {len(user_profiles)} users")
        
        return user_profiles

class RecommenderEngine:
    """
    Generates ranked lists of movie recommendations using Cosine Similarity.
    """
    
    @staticmethod
    def generate_recommendations(user_profiles: Dict[int, np.ndarray],
                               movie_features: np.ndarray,
                               movie_idx_to_id: Dict[int, int],
                               train_ratings: pd.DataFrame,
                               user_id: int) -> List[int]:
        """
        Produce a list of recommended movie IDs for a single user.
        
        Calculates the cosine similarity between the user's profile vector and 
        all movie feature vectors. Excludes movies the user has already seen (in training set).
        
        Args:
            user_profiles (Dict[int, np.ndarray]): Helper dictionary of user vectors.
            movie_features (np.ndarray): Global movie feature matrix.
            movie_idx_to_id (Dict[int, int]): Mapping from matrix row index to Movie ID.
            train_ratings (pd.DataFrame): Training data (to look up watched status).
            user_id (int): The ID of the target user.
            
        Returns:
            List[int]: Top K movie IDs ordered by similarity score (descending).
        """
        
        if user_id not in user_profiles:
            return []
        
        user_profile = user_profiles[user_id]
        
        # Get movies already watched by user
        watched_movies = set(train_ratings[train_ratings['userId'] == user_id]['movieId'].values)
        
        # Calculate similarities with all movies
        similarities = cosine_similarity([user_profile], movie_features)[0]
        
        # Create list of (movie_id, similarity) for unwatched movies
        recommendations = []
        for idx, similarity in enumerate(similarities):
            movie_id = movie_idx_to_id[idx]
            if movie_id not in watched_movies and similarity > Config.SIMILARITY_THRESHOLD:
                recommendations.append((movie_id, similarity))
        
        # Sort by similarity and return top K
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return [movie_id for movie_id, _ in recommendations[:Config.TOP_K_RECOMMENDATIONS]]

class Evaluator:
    """
    Computes performance metrics by comparing recommendations against held-out test data.
    """
    
    @staticmethod
    def evaluate_recommendations(user_profiles: Dict[int, np.ndarray],
                               movie_features: np.ndarray,
                               movie_idx_to_id: Dict[int, int],
                               train_ratings: pd.DataFrame,
                               test_ratings: pd.DataFrame) -> Dict[str, float]:
        """
        Run the full evaluation pipeline across all test users.
        
        Calculates:
        - Precision, Recall, Hit Rate (Standard consumption metrics).
        - Enjoyment Precision/Recall (Metrics weighted by whether user liked the item).
        - Confusion Matrix (TP, FP, FN, TN) for both consumption and enjoyment.
        
        Args:
            user_profiles (Dict[int, np.ndarray]): User preference vectors.
            movie_features (np.ndarray): Movie feature matrix.
            movie_idx_to_id (Dict[int, int]): Index to ID mapping.
            train_ratings (pd.DataFrame): Training set (for exclusion).
            test_ratings (pd.DataFrame): Test set (Ground Truth).
            
        Returns:
            Dict[str, float]: Dictionary of aggregated average metrics.
        """
        print("\n=== Evaluation ===")
        
        all_metrics = []
        test_users = test_ratings['userId'].unique()
        total_users = len(test_users)
        evaluated_users = 0
        total_available_movies = len(movie_idx_to_id) # The Universe size
        
        for i, user_id in enumerate(test_users):
            if i % Config.EVAL_PROGRESS_INTERVAL == 0:
                print(f"Evaluated {i}/{total_users} users...")
                
            # Get Ground Truth (Movies watched in "future")
            test_user_ratings = test_ratings[test_ratings['userId'] == user_id]
            actual_movies = set(test_user_ratings['movieId'].values)
            actual_ratings_map = dict(zip(test_user_ratings['movieId'], test_user_ratings['rating']))
            
            if not actual_movies:
                continue
                
            # Get Predictions
            recommendations = RecommenderEngine.generate_recommendations(
                user_profiles, movie_features, movie_idx_to_id, train_ratings, user_id
            )
            
            if not recommendations:
                continue
                
            evaluated_users += 1
            
            # --- Confusion Matrix Calculation ---
            rec_set = set(recommendations)
            
            # True Positives (Hits): Recommended and Watched
            tp = len(rec_set.intersection(actual_movies))
            
            # False Positives: Recommended but NOT Watched
            fp = len(rec_set) - tp
            
            # False Negatives: Watched but NOT Recommended
            fn = len(actual_movies) - tp
            
            # True Negatives: Not Recommended and NOT Watched
            # TN = Universe - (TP + FP + FN)
            tn = total_available_movies - (tp + fp + fn)
            
            # Enjoyment (Hits that were also rated positively)
            enjoyment_hits = sum(1 for m in rec_set if m in actual_ratings_map and actual_ratings_map[m] >= Config.RATING_THRESHOLD)
            
            # --- Enjoyment Confusion Matrix ---
            # Ground truth for enjoyment: Movies watched AND rated >= threshold
            actual_liked = {m for m in actual_movies if actual_ratings_map[m] >= Config.RATING_THRESHOLD}
            
            # Enjoyment TP: Recommended AND Liked
            e_tp = len(rec_set.intersection(actual_liked))
            
            # Enjoyment FP: Recommended AND (Not Watched OR Not Liked)
            e_fp = len(rec_set) - e_tp
            
            # Enjoyment FN: Liked BUT Not Recommended
            e_fn = len(actual_liked) - e_tp
            
            # Enjoyment TN: Not Recommended AND (Not Watched OR Not Liked)
            e_tn = total_available_movies - (e_tp + e_fp + e_fn)
            
            # Standard Metrics
            precision = tp / len(recommendations)
            recall = tp / len(actual_movies)
            hit_rate = 1.0 if tp > 0 else 0.0
            accuracy = (tp + tn) / total_available_movies if total_available_movies > 0 else 0.0
            
            # Enjoyment Metrics
            enjoyment_precision = enjoyment_hits / len(recommendations)
            enjoyment_recall = e_tp / len(actual_liked) if actual_liked else 0.0
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'hit_rate': hit_rate,
                'enjoyment_precision': enjoyment_precision,
                'enjoyment_recall': enjoyment_recall,
                'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
                'e_tp': e_tp, 'e_fp': e_fp, 'e_fn': e_fn, 'e_tn': e_tn
            }
            all_metrics.append(metrics)
            
        print(f"Evaluation completed: {evaluated_users} users evaluated")
        
        if not all_metrics:
            return {
                'accuracy': 0.0,
                'precision': 0.0, 'recall': 0.0, 'hit_rate': 0.0,
                'enjoyment_precision': 0.0, 'enjoyment_recall': 0.0,
                'evaluated_users': 0, 'total_test_users': total_users,
                'tp': 0.0, 'fp': 0.0, 'fn': 0.0, 'tn': 0.0,
                'e_tp': 0.0, 'e_fp': 0.0, 'e_fn': 0.0, 'e_tn': 0.0
            }
            
        # Calculate Averages
        avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0].keys()}
        
        avg_metrics['evaluated_users'] = evaluated_users
        avg_metrics['total_test_users'] = total_users
        
        return avg_metrics
def main():
    """Main execution function"""
    print("\n=== Content-Based Recommender System ===\n")
    
    # Load data
    movies_df, credits_df, keywords_df, ratings_df, links_df = DataLoader.load_data()
    
    # OPTIMIZATION: Sample users early to reduce memory usage during processing
    # Processing 26M ratings in the merge/groupby steps causes OOM crashes
    print("\nSampling users early to reduce memory usage...")
    ratings_df = DataSplitter.sample_users(ratings_df)
    import gc
    gc.collect()
    
    # --- FIX 1: Map MovieLens IDs (ratings) to TMDB IDs (features) ---
    print("\nMapping MovieLens IDs to TMDB IDs...")
    # links_df maps movieId (MovieLens) -> tmdbId (TMDB)
    # We need to filter out ratings that don't have a TMDB match
    
    # Ensure IDs are numeric
    links_df['tmdbId'] = pd.to_numeric(links_df['tmdbId'], errors='coerce')
    links_df = links_df.dropna(subset=['tmdbId', 'movieId'])
    links_df['tmdbId'] = links_df['tmdbId'].astype(int)
    links_df['movieId'] = links_df['movieId'].astype(int)
    
    # Merge ratings with links
    initial_ratings = len(ratings_df)
    
    # Consistent Logic with Collaborative Filter:
    # 1. Merge Left (keep all ratings, try to find match)
    ratings_df = ratings_df.merge(links_df[['movieId', 'tmdbId']], on='movieId', how='left')
    
    # 2. Drop ratings that couldn't be mapped
    ratings_df = ratings_df.dropna(subset=['tmdbId'])
    ratings_df['tmdbId'] = ratings_df['tmdbId'].astype(int)
    
    # 3. Rename tmdbId -> movieId
    ratings_df = ratings_df.drop('movieId', axis=1).rename(columns={'tmdbId': 'movieId'})
    
    # 4. Handle Duplicates (Multiple ML IDs -> Single TMDB ID)
    # Takes max of rating/year if duplicates exist for same user+movie
    ratings_df = ratings_df.groupby(['userId', 'movieId'], as_index=False).max()
    
    print(f"Mapped {len(ratings_df)} ratings to TMDB IDs (dropped {initial_ratings - len(ratings_df)} unmapped)")
    
    # Merge movie features
    movies_with_features = DataLoader.merge_movie_features(movies_df, credits_df, keywords_df)
    
    # Ensure movie features index is strictly int
    movies_with_features['id'] = pd.to_numeric(movies_with_features['id'], errors='coerce').fillna(-1).astype(int)
    movies_with_features = movies_with_features[movies_with_features['id'] > 0]
    
    # Filter ratings (Users were already sampled at the start)
    filtered_ratings = DataSplitter.filter_users(ratings_df)
    train_ratings, test_ratings = DataSplitter.split_user_ratings(filtered_ratings)
    
    # Get movies that appear in ratings
    rated_movie_ids = set(filtered_ratings['movieId'].unique())
    movies_with_features = movies_with_features[movies_with_features['id'].isin(rated_movie_ids)].copy()

    # Reset index to ensure contiguous indices for feature matrix
    movies_with_features = movies_with_features.reset_index(drop=True)

    # Create mappings AFTER resetting index
    movie_id_to_idx = {movie_id: idx for idx, movie_id in enumerate(movies_with_features['id'].values)}
    movie_idx_to_id = {idx: movie_id for idx, movie_id in enumerate(movies_with_features['id'].values)}

    # Engineer features
    feature_engineer = FeatureEngineer()
    movie_features = feature_engineer.create_feature_vectors(movies_with_features)
    feature_engineer.print_feature_list()
    
    # Create user profiles
    user_profiles = UserProfiler.create_user_profiles(
        train_ratings, movie_features, movie_id_to_idx
    )
    
    # Run Evaluation
    metrics = Evaluator.evaluate_recommendations(
        user_profiles, movie_features, movie_idx_to_id, 
        train_ratings, test_ratings
    )

    # Print results
    print("\n=== Recommendation Metrics ===")
    print(f"Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"Precision: {metrics['precision']*100:.2f}%")
    print(f"Recall:    {metrics['recall']*100:.2f}%")
    print(f"Hit Rate:  {metrics['hit_rate']*100:.2f}%")
    print(f"Enjoyment Precision: {metrics['enjoyment_precision']*100:.2f}%")
    print(f"Enjoyment Recall:    {metrics['enjoyment_recall']*100:.2f}%")
    print(f"Evaluated:    {metrics['evaluated_users']}/{metrics['total_test_users']} users")
    
    print("\n=== Confusion Matrix (Avg per User) ===")
    print(f"True Positives (Hits):  {metrics['tp']:.2f}")
    print(f"False Positives (Miss): {metrics['fp']:.2f}")
    print(f"False Negatives (Lost): {metrics['fn']:.2f}")
    print(f"True Negatives (Ignore):{metrics['tn']:.2f}")
    
    if 'e_tp' in metrics:
        print("\n=== Enjoyment Confusion Matrix (Rated >= 3.5) ===")
        print(f"E-TP (Liked & Rec'd):   {metrics['e_tp']:.2f}")
        print(f"E-FP (Bad Rec):         {metrics['e_fp']:.2f}")
        print(f"E-FN (Missed gem):      {metrics['e_fn']:.2f}")
        print(f"E-TN (Correctly skip):  {metrics['e_tn']:.2f}")

    # Print configuration summary
    print("\n=== Configuration Summary ===")
    print(f"Max Users: {Config.MAX_USERS if Config.MAX_USERS else 'All'}")
    print(f"Min Ratings: {Config.MIN_USER_RATINGS}")
    print(f"Train Ratio: {Config.TRAIN_RATIO}")

if __name__ == "__main__":
    main()
