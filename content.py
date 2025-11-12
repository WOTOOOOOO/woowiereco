"""
Content-Based Recommender System
A modular implementation for movie recommendations using content features
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, ndcg_score
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
    
    # Splitting parameters
    TRAIN_RATIO = 0.8  # 80% of user ratings for training
    MIN_USER_RATINGS = 10  # Minimum ratings per user to include
    
    # User sampling parameters
    MAX_USERS = None  # Maximum number of users to process (None = all users)
    # MAX_USERS = 40000  # Uncomment to limit to 40k users for faster processing
    USER_SAMPLE_SEED = 42  # Random seed for user sampling
    
    # Feature engineering parameters
    MAX_TFIDF_FEATURES = 5000  # Maximum features for TF-IDF vectorization
    MIN_DF = 1  # Minimum document frequency for TF-IDF
    MAX_DF = 0.95  # Maximum document frequency for TF-IDF
    
    # Content features weights
    GENRE_WEIGHT = 0.25
    KEYWORD_WEIGHT = 0.20
    CAST_WEIGHT = 0.15
    DIRECTOR_WEIGHT = 0.10
    PRODUCTION_COMPANIES_WEIGHT = 0.10
    MAIN_STUDIO_WEIGHT = 0.08
    PRIMARY_GENRE_WEIGHT = 0.05
    COLLECTION_WEIGHT = 0.03
    LANGUAGE_WEIGHT = 0.02
    ADULT_WEIGHT = 0.02
    
    # Recommendation parameters
    TOP_K_RECOMMENDATIONS = 10
    SIMILARITY_THRESHOLD = 0.01  # Minimum similarity to consider
    
    # Evaluation parameters
    PRECISION_K = 5
    RECALL_K = 10
    NDCG_K = 10
    MAX_EVAL_USERS = None  # Limit evaluation to N users for faster testing (None = all test users)
    EVAL_PROGRESS_INTERVAL = 1000  # Print progress every N users
    
    @classmethod
    def enable_user_sampling(cls, max_users: int = 20000):
        """Enable user sampling with specified maximum users"""
        cls.MAX_USERS = max_users
        print(f"User sampling enabled: max {max_users} users")
    
    @classmethod
    def disable_user_sampling(cls):
        """Disable user sampling - process all users"""
        cls.MAX_USERS = None
        print("User sampling disabled: processing all users")
    
    @classmethod
    def enable_fast_evaluation(cls, max_eval_users: int = 1000):
        """Enable fast evaluation mode with limited users"""
        cls.MAX_EVAL_USERS = max_eval_users
        print(f"Fast evaluation enabled: max {max_eval_users} test users")
    
    @classmethod
    def disable_fast_evaluation(cls):
        """Disable fast evaluation - evaluate all test users"""
        cls.MAX_EVAL_USERS = None
        print("Fast evaluation disabled: evaluating all test users")
    
    @classmethod
    def print_config(cls):
        """Print current configuration settings"""
        print("\nCurrent Configuration:")
        print("=" * 30)
        print(f"Max Users: {cls.MAX_USERS if cls.MAX_USERS else 'All users'}")
        print(f"Max Eval Users: {cls.MAX_EVAL_USERS if cls.MAX_EVAL_USERS else 'All test users'}")
        print(f"Min User Ratings: {cls.MIN_USER_RATINGS}")
        print(f"Train Ratio: {cls.TRAIN_RATIO}")
        print(f"Feature Weights: Genre={cls.GENRE_WEIGHT}, Keyword={cls.KEYWORD_WEIGHT}, "
              f"Cast={cls.CAST_WEIGHT}, Director={cls.DIRECTOR_WEIGHT}")
        print(f"                 Production={cls.PRODUCTION_COMPANIES_WEIGHT}, Studio={cls.MAIN_STUDIO_WEIGHT}, "
              f"PrimaryGenre={cls.PRIMARY_GENRE_WEIGHT}, Collection={cls.COLLECTION_WEIGHT}")
        print(f"                 Language={cls.LANGUAGE_WEIGHT}, Adult={cls.ADULT_WEIGHT}")
        print(f"Top K Recommendations: {cls.TOP_K_RECOMMENDATIONS}")
        print(f"Evaluation: P@{cls.PRECISION_K}, R@{cls.RECALL_K}, NDCG@{cls.NDCG_K}")
        print()

class DataLoader:
    """Handle data loading and preprocessing"""
    
    @staticmethod
    def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load all required datasets"""
        print("Loading datasets...")
        
        movies_df = pd.read_csv(f"{Config.DATA_DIR}{Config.MOVIES_FILE}")
        credits_df = pd.read_csv(f"{Config.DATA_DIR}{Config.CREDITS_FILE}")
        keywords_df = pd.read_csv(f"{Config.DATA_DIR}{Config.KEYWORDS_FILE}")
        ratings_df = pd.read_csv(f"{Config.DATA_DIR}{Config.RATINGS_FILE}")
        
        print(f"Loaded {len(movies_df)} movies, {len(credits_df)} credit records, "
              f"{len(keywords_df)} keyword records, {len(ratings_df)} ratings")
        
        return movies_df, credits_df, keywords_df, ratings_df
    
    @staticmethod
    def merge_movie_features(movies_df: pd.DataFrame, credits_df: pd.DataFrame, 
                           keywords_df: pd.DataFrame) -> pd.DataFrame:
        """Merge all movie feature datasets"""
        print("Merging movie features...")
        
        # Merge on movie id
        merged_df = movies_df.merge(credits_df, on='id', how='left')
        merged_df = merged_df.merge(keywords_df, on='id', how='left')
        
        # Fill missing values
        merged_df['keywords'] = merged_df['keywords'].fillna('[]')
        merged_df['actors'] = merged_df['actors'].fillna('[]')
        merged_df['directors'] = merged_df['directors'].fillna('[]')
        merged_df['overview'] = merged_df['overview'].fillna('')
        
        print(f"Merged dataset has {len(merged_df)} movies with features")
        return merged_df

class DataSplitter:
    """Handle train/test splitting for users"""
    
    @staticmethod
    def sample_users(ratings_df: pd.DataFrame) -> pd.DataFrame:
        """Sample a subset of users if MAX_USERS is specified"""
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
        """Filter users with minimum required ratings"""
        user_counts = ratings_df['userId'].value_counts()
        valid_users = user_counts[user_counts >= Config.MIN_USER_RATINGS].index
        
        filtered_ratings = ratings_df[ratings_df['userId'].isin(valid_users)]
        
        print(f"Filtered to {len(valid_users)} users with >= {Config.MIN_USER_RATINGS} ratings")
        print(f"Remaining ratings: {len(filtered_ratings)}")
        
        return filtered_ratings
    
    @staticmethod
    def split_user_ratings(ratings_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split each user's ratings into train/test sets"""
        print("Splitting user ratings...")
        
        train_list = []
        test_list = []
        
        for user_id in ratings_df['userId'].unique():
            user_ratings = ratings_df[ratings_df['userId'] == user_id]
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
    """Extract and engineer content features from movies"""
    
    def __init__(self):
        self.genre_vectorizer = None
        self.keyword_vectorizer = None
        self.cast_vectorizer = None
        self.director_vectorizer = None
        self.production_companies_vectorizer = None
        self.main_studio_vectorizer = None
        self.primary_genre_vectorizer = None
        self.collection_vectorizer = None
        self.language_vectorizer = None
        self.adult_vectorizer = None
        self.scaler = StandardScaler()
    
    def safe_eval(self, x):
        """Safely evaluate string representations of lists"""
        if pd.isna(x) or x == '[]':
            return []
        try:
            return ast.literal_eval(x)
        except:
            return []
    
    def prepare_text_features(self, movies_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare text features for vectorization"""
        print("Preparing text features...")
        
        # Process genres
        movies_df['genre_list'] = movies_df['genres'].apply(self.safe_eval)
        movies_df['genre_text'] = movies_df['genre_list'].apply(lambda x: ' '.join(x))
        
        # Process keywords
        movies_df['keyword_list'] = movies_df['keywords'].apply(self.safe_eval)
        movies_df['keyword_text'] = movies_df['keyword_list'].apply(lambda x: ' '.join(x))
        
        # Process cast (limit to top actors)
        movies_df['actor_list'] = movies_df['actors'].apply(self.safe_eval)
        movies_df['top_actors'] = movies_df['actor_list'].apply(lambda x: x[:5] if x else [])
        movies_df['actor_text'] = movies_df['top_actors'].apply(lambda x: ' '.join(x))
        
        # Process directors
        movies_df['director_list'] = movies_df['directors'].apply(self.safe_eval)
        movies_df['director_text'] = movies_df['director_list'].apply(lambda x: ' '.join(x))
        
        # Process production companies
        movies_df['production_companies_list'] = movies_df['production_companies'].apply(self.safe_eval)
        movies_df['production_companies_text'] = movies_df['production_companies_list'].apply(lambda x: ' '.join(x))
        
        # Process main studio (single value, not a list)
        movies_df['main_studio_text'] = movies_df['main_studio'].fillna('').astype(str)
        
        # Process primary genre (single value, not a list)
        movies_df['primary_genre_text'] = movies_df['primary_genre'].fillna('').astype(str)
        
        # Process collection name (single value, not a list)
        movies_df['collection_name_text'] = movies_df['collection_name'].fillna('').astype(str)
        
        # Process original language (single value, not a list)
        movies_df['original_language_text'] = movies_df['original_language'].fillna('').astype(str)
        
        # Process adult flag (convert boolean to text)
        movies_df['adult_text'] = movies_df['adult'].fillna(False).astype(str)
        
        return movies_df
    
    def create_feature_vectors(self, movies_df: pd.DataFrame) -> np.ndarray:
        """Create combined feature vectors for all movies"""
        print("Creating feature vectors...")
        
        # Prepare text features
        movies_df = self.prepare_text_features(movies_df)
        
        # Create TF-IDF vectors for different features
        feature_vectors = []
        
        # Genre features
        self.genre_vectorizer = TfidfVectorizer(
            max_features=1000, min_df=Config.MIN_DF, max_df=Config.MAX_DF
        )
        genre_features = self.genre_vectorizer.fit_transform(movies_df['genre_text'])
        feature_vectors.append(genre_features.toarray() * Config.GENRE_WEIGHT)
        
        # Keyword features
        self.keyword_vectorizer = TfidfVectorizer(
            max_features=Config.MAX_TFIDF_FEATURES, min_df=Config.MIN_DF, max_df=Config.MAX_DF
        )
        keyword_features = self.keyword_vectorizer.fit_transform(movies_df['keyword_text'])
        feature_vectors.append(keyword_features.toarray() * Config.KEYWORD_WEIGHT)
        
        # Cast features
        self.cast_vectorizer = TfidfVectorizer(
            max_features=2000, min_df=Config.MIN_DF, max_df=Config.MAX_DF
        )
        cast_features = self.cast_vectorizer.fit_transform(movies_df['actor_text'])
        feature_vectors.append(cast_features.toarray() * Config.CAST_WEIGHT)
        
        # Director features
        self.director_vectorizer = TfidfVectorizer(
            max_features=500, min_df=Config.MIN_DF, max_df=Config.MAX_DF
        )
        director_features = self.director_vectorizer.fit_transform(movies_df['director_text'])
        feature_vectors.append(director_features.toarray() * Config.DIRECTOR_WEIGHT)
        
        # Production companies features
        self.production_companies_vectorizer = TfidfVectorizer(
            max_features=1000, min_df=Config.MIN_DF, max_df=Config.MAX_DF
        )
        production_companies_features = self.production_companies_vectorizer.fit_transform(movies_df['production_companies_text'])
        feature_vectors.append(production_companies_features.toarray() * Config.PRODUCTION_COMPANIES_WEIGHT)
        
        # Main studio features
        self.main_studio_vectorizer = TfidfVectorizer(
            max_features=300, min_df=Config.MIN_DF, max_df=Config.MAX_DF
        )
        main_studio_features = self.main_studio_vectorizer.fit_transform(movies_df['main_studio_text'])
        feature_vectors.append(main_studio_features.toarray() * Config.MAIN_STUDIO_WEIGHT)
        
        # Primary genre features
        self.primary_genre_vectorizer = TfidfVectorizer(
            max_features=50, min_df=Config.MIN_DF, max_df=Config.MAX_DF
        )
        primary_genre_features = self.primary_genre_vectorizer.fit_transform(movies_df['primary_genre_text'])
        feature_vectors.append(primary_genre_features.toarray() * Config.PRIMARY_GENRE_WEIGHT)
        
        # Collection features
        self.collection_vectorizer = TfidfVectorizer(
            max_features=200, min_df=Config.MIN_DF, max_df=Config.MAX_DF
        )
        collection_features = self.collection_vectorizer.fit_transform(movies_df['collection_name_text'])
        feature_vectors.append(collection_features.toarray() * Config.COLLECTION_WEIGHT)
        
        # Language features
        self.language_vectorizer = TfidfVectorizer(
            max_features=50, min_df=Config.MIN_DF, max_df=Config.MAX_DF
        )
        language_features = self.language_vectorizer.fit_transform(movies_df['original_language_text'])
        feature_vectors.append(language_features.toarray() * Config.LANGUAGE_WEIGHT)
        
        # Adult features
        self.adult_vectorizer = TfidfVectorizer(
            max_features=2, min_df=1, max_df=1.0  # Only True/False values
        )
        adult_features = self.adult_vectorizer.fit_transform(movies_df['adult_text'])
        feature_vectors.append(adult_features.toarray() * Config.ADULT_WEIGHT)
        
        # Combine all features
        combined_features = np.hstack(feature_vectors)
        
        # Normalize features
        combined_features = self.scaler.fit_transform(combined_features)
        
        print(f"Created feature matrix of shape: {combined_features.shape}")
        
        return combined_features

class UserProfiler:
    """Create user profiles based on their watch history"""
    
    @staticmethod
    def create_user_profiles(train_ratings: pd.DataFrame, 
                           movie_features: np.ndarray,
                           movie_id_to_idx: Dict[int, int]) -> Dict[int, np.ndarray]:
        """Create user preference profiles from training data"""
        print("Creating user profiles...")
        
        user_profiles = {}
        
        for user_id in train_ratings['userId'].unique():
            user_movies = train_ratings[train_ratings['userId'] == user_id]['movieId'].values
            
            # Get feature vectors for user's watched movies
            movie_indices = [movie_id_to_idx[mid] for mid in user_movies if mid in movie_id_to_idx]
            
            if movie_indices:
                user_movie_features = movie_features[movie_indices]
                # Average the features (could also weight by ratings)
                user_profile = np.mean(user_movie_features, axis=0)
                user_profiles[user_id] = user_profile
        
        print(f"Created profiles for {len(user_profiles)} users")
        
        return user_profiles

class RecommenderEngine:
    """Generate recommendations using content-based filtering"""
    
    @staticmethod
    def generate_recommendations(user_profiles: Dict[int, np.ndarray],
                               movie_features: np.ndarray,
                               movie_idx_to_id: Dict[int, int],
                               train_ratings: pd.DataFrame,
                               user_id: int) -> List[int]:
        """Generate recommendations for a specific user"""
        
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
    """Evaluate recommendation performance"""
    
    @staticmethod
    def evaluate_recommendations(user_profiles: Dict[int, np.ndarray],
                               movie_features: np.ndarray,
                               movie_idx_to_id: Dict[int, int],
                               train_ratings: pd.DataFrame,
                               test_ratings: pd.DataFrame) -> Dict[str, float]:
        """Evaluate the recommender system"""
        print("Evaluating recommendations...")
        
        precision_scores = []
        recall_scores = []
        ndcg_scores = []
        
        test_users = test_ratings['userId'].unique()
        
        # Limit evaluation users if specified
        if Config.MAX_EVAL_USERS is not None:
            test_users = test_users[:Config.MAX_EVAL_USERS]
            print(f"Limiting evaluation to {len(test_users)} users for faster testing")
        
        total_users = len(test_users)
        evaluated_users = 0
        
        for i, user_id in enumerate(test_users):
            if user_id not in user_profiles:
                continue
            
            # Progress tracking
            if (i + 1) % Config.EVAL_PROGRESS_INTERVAL == 0:
                print(f"Evaluated {i + 1}/{total_users} users ({(i + 1)/total_users*100:.1f}%)")
            
            evaluated_users += 1
            
            # Generate recommendations
            recommendations = RecommenderEngine.generate_recommendations(
                user_profiles, movie_features, movie_idx_to_id, 
                train_ratings, user_id
            )
            
            # Get actual test movies for this user
            actual_movies = set(test_ratings[test_ratings['userId'] == user_id]['movieId'].values)
            
            if not recommendations or not actual_movies:
                continue
            
            # Calculate metrics
            recommended_set = set(recommendations[:Config.PRECISION_K])
            precision = len(recommended_set & actual_movies) / len(recommended_set)
            precision_scores.append(precision)
            
            recommended_set_recall = set(recommendations[:Config.RECALL_K])
            recall = len(recommended_set_recall & actual_movies) / len(actual_movies)
            recall_scores.append(recall)
            
            # NDCG calculation (simplified)
            relevant = [1 if rec in actual_movies else 0 for rec in recommendations[:Config.NDCG_K]]
            # NDCG requires at least 2 documents, and at least 1 relevant item
            if sum(relevant) > 0 and len(actual_movies) > 1:
                try:
                    # Create ideal ranking (all relevant items first)
                    ideal = sorted(relevant, reverse=True)
                    ndcg = ndcg_score([ideal], [relevant])
                    ndcg_scores.append(ndcg)
                except ValueError:
                    # Skip if NDCG calculation fails
                    continue
        
        print(f"Evaluation completed: {evaluated_users} users evaluated")
        print(f"Metric calculations: {len(precision_scores)} precision, {len(recall_scores)} recall, {len(ndcg_scores)} NDCG scores")
        
        metrics = {
            'precision': np.mean(precision_scores) if precision_scores else 0,
            'recall': np.mean(recall_scores) if recall_scores else 0,
            'ndcg': np.mean(ndcg_scores) if ndcg_scores else 0,
            'coverage': evaluated_users / len(user_profiles) if user_profiles else 0,
            'evaluated_users': evaluated_users,
            'total_test_users': len(test_users)
        }
        
        return metrics

def main():
    """Main execution function"""
    print("Starting Content-Based Recommender System")
    print("=" * 50)
    
    # Print current configuration
    Config.print_config()
    
    # Load data
    movies_df, credits_df, keywords_df, ratings_df = DataLoader.load_data()
    
    # Merge movie features
    movies_with_features = DataLoader.merge_movie_features(movies_df, credits_df, keywords_df)
    
    # Sample users (if specified) and filter ratings
    sampled_ratings = DataSplitter.sample_users(ratings_df)
    filtered_ratings = DataSplitter.filter_users(sampled_ratings)
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
    
    # Create user profiles
    user_profiles = UserProfiler.create_user_profiles(
        train_ratings, movie_features, movie_id_to_idx
    )
    
    # Evaluate system
    metrics = Evaluator.evaluate_recommendations(
        user_profiles, movie_features, movie_idx_to_id, 
        train_ratings, test_ratings
    )
    
    # Print results
    print("\nEvaluation Results:")
    print("=" * 30)
    print(f"Precision@{Config.PRECISION_K}: {metrics['precision']:.4f}")
    print(f"Recall@{Config.RECALL_K}: {metrics['recall']:.4f}")
    print(f"NDCG@{Config.NDCG_K}: {metrics['ndcg']:.4f}")
    print(f"Coverage: {metrics['coverage']:.4f}")
    print(f"Users Evaluated: {metrics['evaluated_users']}/{metrics['total_test_users']}")
    
    return user_profiles, movie_features, movie_idx_to_id, movies_with_features, train_ratings

def run_with_user_sampling(max_users: int = 20000):
    """Convenience function to run with user sampling enabled"""
    Config.enable_user_sampling(max_users)
    return main()

def run_full_dataset():
    """Convenience function to run with all users"""
    Config.disable_user_sampling()
    return main()

def run_quick_test(max_users: int = 5000, max_eval_users: int = 500):
    """Convenience function for quick testing with limited users and evaluation"""
    Config.enable_user_sampling(max_users)
    Config.enable_fast_evaluation(max_eval_users)
    return main()

if __name__ == "__main__":
    # Example usage:
    # For faster testing with limited users:
    result = run_with_user_sampling(5000)
    # For full dataset (default):
    # Config.disable_user_sampling()
    