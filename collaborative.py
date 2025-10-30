"""
Collaborative Filtering Algorithm for Movie Recommendations
=========================================================

Pure algorithmic approach without ML models or training.
Uses user rating similarities to predict movie preferences.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set
import logging
from collections import defaultdict, Counter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CollaborativeFilteringConfig:
    """Configuration parameters for the collaborative filtering algorithm."""
    
    def __init__(self):
        # Data split parameters
        self.train_user_percentage = 0.70
        self.known_history_percentage = 0.80
        
        # Rating and similarity parameters  
        self.rating_threshold = 4.0
        self.min_shared_movies = 3  # Minimum movies to consider users similar
        
        # Similarity method: 'shared_count', 'weighted_shared', 'jaccard'
        self.similarity_method = 'weighted_shared'
        self.rating_agreement_bonus = 0.5  # Bonus for similar rating patterns
        
        # Recommendation scoring weights
        self.score_weights = {
            'high_frequency_good_rating': 1.0,      # Many watchers + good ratings
            'low_frequency_excellent_rating': 0.8,   # Few watchers + excellent ratings
            'high_frequency_mediocre_rating': 0.6,   # Many watchers + mediocre ratings
            'base_score': 0.4                        # Everything else
        }
        
        # Scoring thresholds
        self.high_frequency_threshold = 50  # Movies watched by 50+ users
        self.excellent_rating_threshold = 4.5
        self.good_rating_threshold = 4.0
        
        # Prediction parameters
        self.max_similar_users = 100  # Limit similar users for performance
        self.top_n_recommendations = 50  # Top N movies to consider for prediction


class CollaborativeFiltering:
    """Collaborative filtering recommender system."""
    
    def __init__(self, config: CollaborativeFilteringConfig = None):
        self.config = config or CollaborativeFilteringConfig()
        self.ratings_data = None
        self.train_users = None
        self.test_users = None
        self.user_profiles = {}  # Cache user rating profiles
        self.movie_stats = {}    # Cache movie statistics
        
    def load_data(self, ratings_file: str) -> None:
        """Load and prepare ratings data."""
        logger.info(f"Loading ratings data from {ratings_file}")
        
        self.ratings_data = pd.read_csv(ratings_file)
        logger.info(f"Loaded {len(self.ratings_data)} ratings")
        logger.info(f"Unique users: {self.ratings_data['userId'].nunique()}")
        logger.info(f"Unique movies: {self.ratings_data['movieId'].nunique()}")
        
        # Precompute movie statistics
        self._compute_movie_stats()
        
    def split_data(self) -> Tuple[Set[int], Set[int]]:
        """Split users into training (70%) and test (30%) sets."""
        logger.info("Splitting data into train/test sets")
        
        unique_users = self.ratings_data['userId'].unique()
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(unique_users)
        
        train_size = int(len(unique_users) * self.config.train_user_percentage)
        
        self.train_users = set(unique_users[:train_size])
        self.test_users = set(unique_users[train_size:])
        
        logger.info(f"Training users: {len(self.train_users)}")
        logger.info(f"Test users: {len(self.test_users)}")
        
        return self.train_users, self.test_users
    
    def split_user_history(self, user_id: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split individual user's ratings into known (80%) and hidden (20%) history."""
        user_ratings = self.ratings_data[self.ratings_data['userId'] == user_id].copy()
        
        # Sort by timestamp to maintain chronological order
        user_ratings = user_ratings.sort_values('rating_year')
        
        split_point = int(len(user_ratings) * self.config.known_history_percentage)
        
        known_history = user_ratings.iloc[:split_point]
        hidden_history = user_ratings.iloc[split_point:]
        
        return known_history, hidden_history
    
    def _compute_movie_stats(self) -> None:
        """Precompute movie statistics for scoring."""
        logger.info("Computing movie statistics")
        
        movie_groups = self.ratings_data.groupby('movieId')
        
        for movie_id, group in movie_groups:
            self.movie_stats[movie_id] = {
                'avg_rating': group['rating'].mean(),
                'rating_count': len(group),
                'high_rating_count': (group['rating'] >= self.config.rating_threshold).sum()
            }
    
    def get_user_profile(self, user_id: int, ratings_df: pd.DataFrame = None) -> Dict:
        """Get user's rating profile (cached for performance)."""
        cache_key = f"{user_id}_{len(ratings_df) if ratings_df is not None else 'full'}"
        
        if cache_key in self.user_profiles:
            return self.user_profiles[cache_key]
        
        if ratings_df is not None:
            user_ratings = ratings_df
        else:
            user_ratings = self.ratings_data[self.ratings_data['userId'] == user_id]
        
        profile = {
            'high_rated_movies': set(user_ratings[user_ratings['rating'] >= self.config.rating_threshold]['movieId']),
            'all_movies': set(user_ratings['movieId']),
            'avg_rating': user_ratings['rating'].mean(),
            'rating_dict': dict(zip(user_ratings['movieId'], user_ratings['rating']))
        }
        
        self.user_profiles[cache_key] = profile
        return profile
    
    def calculate_user_similarity(self, test_user_profile: Dict, train_user_id: int) -> float:
        """Calculate similarity between test user and training user."""
        train_user_profile = self.get_user_profile(train_user_id)
        
        # Find shared high-rated movies
        shared_high_rated = test_user_profile['high_rated_movies'].intersection(
            train_user_profile['high_rated_movies']
        )
        
        if len(shared_high_rated) < self.config.min_shared_movies:
            return 0.0
        
        if self.config.similarity_method == 'shared_count':
            return len(shared_high_rated)
        
        elif self.config.similarity_method == 'weighted_shared':
            # Base similarity from shared movies
            similarity = len(shared_high_rated)
            
            # Add bonus for rating agreement
            rating_agreement = 0
            shared_all = test_user_profile['all_movies'].intersection(train_user_profile['all_movies'])
            
            for movie_id in shared_all:
                test_rating = test_user_profile['rating_dict'][movie_id]
                train_rating = train_user_profile['rating_dict'][movie_id]
                # Bonus for similar ratings (within 1 star)
                if abs(test_rating - train_rating) <= 1.0:
                    rating_agreement += self.config.rating_agreement_bonus
            
            return similarity + rating_agreement
        
        elif self.config.similarity_method == 'jaccard':
            shared_movies = test_user_profile['all_movies'].intersection(train_user_profile['all_movies'])
            union_movies = test_user_profile['all_movies'].union(train_user_profile['all_movies'])
            return len(shared_movies) / len(union_movies) if union_movies else 0.0
        
        return 0.0
    
    def find_similar_users(self, test_user_profile: Dict) -> List[Tuple[int, float]]:
        """Find similar users from training set."""
        similarities = []
        
        train_user_data = self.ratings_data[self.ratings_data['userId'].isin(self.train_users)]
        
        for train_user_id in self.train_users:
            similarity = self.calculate_user_similarity(test_user_profile, train_user_id)
            
            if similarity > 0:
                similarities.append((train_user_id, similarity))
        
        # Sort by similarity (descending) and limit for performance
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:self.config.max_similar_users]
    
    def score_movie_recommendation(self, movie_id: int, similar_users: List[Tuple[int, float]]) -> float:
        """Score a movie recommendation based on similar users' ratings."""
        if movie_id not in self.movie_stats:
            return 0.0
        
        movie_stats = self.movie_stats[movie_id]
        
        # Get ratings from similar users
        similar_user_ratings = []
        total_similarity = 0
        
        for user_id, similarity in similar_users:
            user_profile = self.get_user_profile(user_id)
            if movie_id in user_profile['rating_dict']:
                rating = user_profile['rating_dict'][movie_id]
                similar_user_ratings.append(rating * similarity)  # Weight by similarity
                total_similarity += similarity
        
        if not similar_user_ratings or total_similarity == 0:
            return 0.0
        
        # Weighted average rating from similar users
        weighted_avg_rating = sum(similar_user_ratings) / total_similarity
        
        # Determine score category and weight
        freq_count = movie_stats['rating_count']
        avg_rating = movie_stats['avg_rating']
        
        if (freq_count >= self.config.high_frequency_threshold and 
            avg_rating >= self.config.good_rating_threshold):
            weight = self.config.score_weights['high_frequency_good_rating']
        elif (freq_count < self.config.high_frequency_threshold and 
              avg_rating >= self.config.excellent_rating_threshold):
            weight = self.config.score_weights['low_frequency_excellent_rating']
        elif freq_count >= self.config.high_frequency_threshold:
            weight = self.config.score_weights['high_frequency_mediocre_rating']
        else:
            weight = self.config.score_weights['base_score']
        
        # Final score: weighted rating * category weight * frequency boost
        frequency_boost = min(2.0, 1.0 + (freq_count / 1000))  # Cap at 2x boost
        
        return weighted_avg_rating * weight * frequency_boost
    
    def generate_recommendations(self, test_user_profile: Dict, 
                               similar_users: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        """Generate movie recommendations for a test user."""
        candidate_movies = set()
        
        # Collect all movies watched by similar users
        for user_id, similarity in similar_users:
            user_profile = self.get_user_profile(user_id)
            candidate_movies.update(user_profile['high_rated_movies'])
        
        # Remove movies already seen by test user
        candidate_movies -= test_user_profile['all_movies']
        
        # Score each candidate movie
        scored_recommendations = []
        for movie_id in candidate_movies:
            score = self.score_movie_recommendation(movie_id, similar_users)
            if score > 0:
                scored_recommendations.append((movie_id, score))
        
        # Sort by score (descending)
        scored_recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return scored_recommendations[:self.config.top_n_recommendations]
    
    def predict_for_user(self, test_user_id: int) -> Tuple[List[int], List[int], Dict[int, float]]:
        """Make predictions for a single test user."""
        # Split user's history
        known_history, hidden_history = self.split_user_history(test_user_id)
        
        if len(known_history) == 0 or len(hidden_history) == 0:
            return [], list(hidden_history['movieId']), {}
        
        # Get user profile from known history
        test_user_profile = self.get_user_profile(test_user_id, known_history)
        
        # Find similar users
        similar_users = self.find_similar_users(test_user_profile)
        
        if not similar_users:
            return [], list(hidden_history['movieId']), {}
        
        # Generate recommendations
        recommendations = self.generate_recommendations(test_user_profile, similar_users)
        
        # Extract just the movie IDs as predictions
        predictions = [movie_id for movie_id, score in recommendations]
        actual_movies = list(hidden_history['movieId'])
        
        # Create ratings dictionary for actual movies
        actual_ratings_dict = dict(zip(hidden_history['movieId'], hidden_history['rating']))
        
        return predictions, actual_movies, actual_ratings_dict
    
    def evaluate_predictions(self, predictions: List[int], actual_movies: List[int], 
                           actual_ratings_dict: Dict[int, float]) -> Dict[str, float]:
        """Evaluate prediction accuracy considering both overlap and user enjoyment."""
        if not actual_movies:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'hit_rate': 0.0, 
                   'enjoyment_precision': 0.0, 'enjoyment_recall': 0.0, 'enjoyment_f1': 0.0}
        
        predictions_set = set(predictions)
        actual_set = set(actual_movies)
        
        # Calculate basic hits (movie overlap)
        basic_hits = len(predictions_set.intersection(actual_set))
        
        # Calculate enjoyment hits (movie overlap + user enjoyed it)
        enjoyment_hits = 0
        for movie_id in predictions_set.intersection(actual_set):
            if movie_id in actual_ratings_dict:
                # User enjoyed if they rated >= threshold
                if actual_ratings_dict[movie_id] >= self.config.rating_threshold:
                    enjoyment_hits += 1
        
        # Calculate basic metrics
        precision = basic_hits / len(predictions) if predictions else 0.0
        recall = basic_hits / len(actual_movies) if actual_movies else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        hit_rate = 1.0 if basic_hits > 0 else 0.0
        
        # Calculate enjoyment metrics
        enjoyment_precision = enjoyment_hits / len(predictions) if predictions else 0.0
        enjoyment_recall = enjoyment_hits / len(actual_movies) if actual_movies else 0.0
        enjoyment_f1 = 2 * enjoyment_precision * enjoyment_recall / (enjoyment_precision + enjoyment_recall) if (enjoyment_precision + enjoyment_recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall, 
            'f1': f1,
            'hit_rate': hit_rate,
            'hits': basic_hits,
            'enjoyment_precision': enjoyment_precision,
            'enjoyment_recall': enjoyment_recall,
            'enjoyment_f1': enjoyment_f1,
            'enjoyment_hits': enjoyment_hits,
            'total_predictions': len(predictions),
            'total_actual': len(actual_movies)
        }
    
    def run_evaluation(self) -> Dict[str, float]:
        """Run full evaluation on test users."""
        logger.info("Starting collaborative filtering evaluation")
        
        # Split data
        self.split_data()
        
        all_metrics = []
        successful_predictions = 0
        
        for i, test_user_id in enumerate(self.test_users):
            if i % 50 == 0:
                logger.info(f"Processing test user {i+1}/{len(self.test_users)}")
            
            try:
                predictions, actual_movies, actual_ratings_dict = self.predict_for_user(test_user_id)
                
                if predictions and actual_movies:
                    metrics = self.evaluate_predictions(predictions, actual_movies, actual_ratings_dict)
                    all_metrics.append(metrics)
                    successful_predictions += 1
                    
            except Exception as e:
                logger.warning(f"Error predicting for user {test_user_id}: {e}")
                continue
        
        if not all_metrics:
            logger.error("No successful predictions made!")
            return {}
        
        # Aggregate metrics
        avg_metrics = {
            'avg_precision': np.mean([m['precision'] for m in all_metrics]),
            'avg_recall': np.mean([m['recall'] for m in all_metrics]),
            'avg_f1': np.mean([m['f1'] for m in all_metrics]),
            'avg_hit_rate': np.mean([m['hit_rate'] for m in all_metrics]),
            'avg_enjoyment_precision': np.mean([m['enjoyment_precision'] for m in all_metrics]),
            'avg_enjoyment_recall': np.mean([m['enjoyment_recall'] for m in all_metrics]),
            'avg_enjoyment_f1': np.mean([m['enjoyment_f1'] for m in all_metrics]),
            'total_users_evaluated': len(all_metrics),
            'successful_prediction_rate': successful_predictions / len(self.test_users)
        }
        
        logger.info("=== COLLABORATIVE FILTERING RESULTS ===")
        for metric, value in avg_metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        return avg_metrics


def main():
    """Main execution function."""
    # Initialize collaborative filtering
    config = CollaborativeFilteringConfig()
    cf = CollaborativeFiltering(config)
    
    # Load data
    ratings_file = "/home/woto/Desktop/personal/python/thesis/cleaned_data/ratings_cleaned.csv"
    cf.load_data(ratings_file)
    
    # Run evaluation
    results = cf.run_evaluation()
    
    print("\nCollaborative Filtering Evaluation Complete!")
    print("Results:", results)


if __name__ == "__main__":
    main()
