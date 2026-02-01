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
import time
import random
from multiprocessing import Pool, cpu_count


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CollaborativeFilteringConfig:
    """Configuration parameters for the collaborative filtering algorithm."""
    
    def __init__(self):
        # Data & Split
        self.train_user_percentage = 0.80
        self.known_history_percentage = 0.80
        self.benchmark_max_total_users = 10000
        
        # Algorithm Hyperparameters
        self.min_common_users_for_item_sim = 5   # Min co-ratings for similarity calc
        self.max_similar_items = 20              # Max neighbors to store per item (memory opt)
        self.similarity_threshold = 0.01         # Min cosine score to retain
        self.rating_threshold = 3.5              # Min rating to count as "positive"
        
        # Recommendations & Fallback
        self.enable_popularity_fallback = True
        self.top_n_recommendations = 15
        
        # Execution
        self.use_parallel = False  
        self.num_processes = None
        self.min_release_year = 2000


class CollaborativeFiltering:
    """Item-Based Collaborative Filtering (IBCF) engine with Popularity Fallback."""
    
    def __init__(self, config: CollaborativeFilteringConfig = None):
        """Initialize the CF engine with configuration and empty caches."""
        self.config = config or CollaborativeFilteringConfig()
        self.ratings_data = None
        self.train_users = set()
        self.test_users = set()
        
        # State
        self.item_similarity = {}         # {movie_id: [(sim_id, score), ...]}
        self.global_movie_stats = {}      # {movie_id: {'rating_avg': ..., 'rating_count': ...}}
        self.popular_movies = []          # [movie_id, ...] sorted by weighted score
        self.movie_titles = {}            # {movie_id: title}
        self.total_movies = 0             # Total number of unique movies in dataset
        
    def load_data(self, ratings_file: str) -> None:
        """
        Load ratings and metadata datasets.
        
        Args:
            ratings_file (str): Path to the ratings CSV file.
            
        Side Effects:
            - Populates self.movie_titles with movie ID -> Title mapping.
            - Populates self.ratings_data with filtered ratings dataframe.
            - Filters movies by release year (Config.min_release_year).
        """
        # 1. Load Metadata (Titles & Year Filter)
        meta_file = ratings_file.replace('ratings_cleaned.csv', 'movies_metadata_cleaned.csv')
        logger.info(f"Loading metadata from {meta_file}...")
        
        allowed_movies = set()
        try:
            meta_df = pd.read_csv(meta_file, usecols=['id', 'release_year', 'title'])
            meta_df['id'] = pd.to_numeric(meta_df['id'], errors='coerce')
            meta_df = meta_df.dropna(subset=['id', 'release_year'])
            
            # Cache titles and filter
            self.movie_titles = dict(zip(meta_df['id'].astype(int), meta_df['title']))
            allowed_movies = set(meta_df[meta_df['release_year'] >= self.config.min_release_year]['id'].astype(int))
            logger.info(f"Movies >= {self.config.min_release_year}: {len(allowed_movies)}")
            
        except Exception as e:
            logger.error(f"Metadata load failed: {e}")

        # 2. Load Ratings
        logger.info(f"Loading ratings from {ratings_file}")
        self.ratings_data = pd.read_csv(ratings_file, usecols=['userId', 'movieId', 'rating', 'rating_year'])
        
        # Optimization: Sample users early to avoid OOM during merges
        if self.config.benchmark_max_total_users:
            unique_users = self.ratings_data['userId'].unique()
            if len(unique_users) > self.config.benchmark_max_total_users:
                logger.info(f"Sampling {self.config.benchmark_max_total_users} users early to save memory...")
                np.random.seed(42)
                sampled_users = np.random.choice(unique_users, size=self.config.benchmark_max_total_users, replace=False)
                self.ratings_data = self.ratings_data[self.ratings_data['userId'].isin(sampled_users)]
                import gc
                gc.collect()
        
        # 2.5 Map MovieLens IDs to TMDB IDs
        links_file = ratings_file.replace('ratings_cleaned.csv', 'links.csv')
        try:
            logger.info(f"Loading ID mapping from {links_file}...")
            links_df = pd.read_csv(links_file)
            
            # [Matched Naive Bayes Logic]
            # Merge ratings (movieId) with links (movieId) to get tmdbId
            orig_count = len(self.ratings_data)
            self.ratings_data = self.ratings_data.merge(links_df[['movieId', 'tmdbId']], on='movieId', how='left')
            
            # Drop rows where tmdbId is missing (cannot link to metadata)
            self.ratings_data = self.ratings_data.dropna(subset=['tmdbId'])
            
            # Ensure tmdbId is integer
            self.ratings_data['tmdbId'] = self.ratings_data['tmdbId'].astype(int)
            
            # [Collaborative Filtering Specific]
            # Replace movieId with tmdbId as the primary key
            self.ratings_data = self.ratings_data.drop('movieId', axis=1).rename(columns={'tmdbId': 'movieId'})
            
            # Handle potential N-to-1 mappings (Multiple ML IDs -> Single TMDb ID)
            # We keep the highest rating if a user rated different versions of the same movie (e.g. Extended vs Theatrical)
            self.ratings_data = self.ratings_data.groupby(['userId', 'movieId'], as_index=False).max()
            
            logger.info(f"Mapped IDs: {orig_count} -> {len(self.ratings_data)} unique ratings")
            
        except Exception as e:
            logger.warning(f"ID Mapping failed (continuing without mapping): {e}")

        # 3. Apply Movie Filter
        if allowed_movies:
            orig_len = len(self.ratings_data)
            self.ratings_data = self.ratings_data[self.ratings_data['movieId'].isin(allowed_movies)]
            logger.info(f"Filtered ratings: {orig_len} -> {len(self.ratings_data)}")
        
        self.total_movies = self.ratings_data['movieId'].nunique()
        logger.info(f"Users: {self.ratings_data['userId'].nunique()}, Movies: {self.total_movies}")
        
    def split_data(self) -> None:
        """
        Split unique users into Training and Testing sets.
        
        Logic:
            1. Shuffles all unique user IDs.
            2. Caps total users at config.benchmark_max_total_users to manage resource usage.
            3. Assigns first config.train_user_percentage (80%) to Train set.
            4. Assigns remaining to Test set.
            
        Side Effects:
            - Populates self.train_users (Set[int])
            - Populates self.test_users (Set[int])
            - Triggers self.train_model() immediately after splitting.
        """
        logger.info("Splitting train/test users")
        
        users = self.ratings_data['userId'].unique()
        np.random.seed(42)
        np.random.shuffle(users)
        
        # Enforce user limit
        if self.config.benchmark_max_total_users and len(users) > self.config.benchmark_max_total_users:
            users = users[:self.config.benchmark_max_total_users]
            logger.info(f"Capping dataset at {len(users)} users")
        
        split_idx = int(len(users) * self.config.train_user_percentage)
        self.train_users = set(users[:split_idx])
        self.test_users = set(users[split_idx:])
        
        logger.info(f"Train Users: {len(self.train_users)}, Test Users: {len(self.test_users)}")
        self.train_model()

    def inspect_sample_recommendations(self, num_users: int = 5) -> None:
        """
        Print detailed breakdown of recommendations for random test users.
        Shows History (Know), Prediction, and Future (Hidden) for validation.
        """
        if not self.test_users:
            return

        print(f"\n=== INSPECTING RECOMMENDATIONS ({num_users} Users) ===")
        sample_users = random.sample(list(self.test_users), min(num_users, len(self.test_users)))
        
        for user_id in sample_users:
            print(f"\nUser {user_id}:")
            known, hidden = self.split_user_history(user_id)
            recs, actual_ids, actual_ratings = self.predict_for_user(user_id)
            
            # 1. History (Likes)
            print("  [HISTORY] Favorites:")
            liked = known[known['rating'] >= 3.5].sort_values('rating', ascending=False).head(5)
            for _, row in liked.iterrows():
                title = self.movie_titles.get(int(row['movieId']), f"ID:{row['movieId']}")
                print(f"    - {title} ({row['rating']})")
                
            # 2. Recommendations
            print("  [PREDICTION] Suggested:")
            for mid in recs[:5]:
                print(f"    - {self.movie_titles.get(mid, f'ID:{mid}')}")
                
            # 3. Validation
            print("  [ACTUAL] Actually Watched:")
            for mid in actual_ids:
                rating = actual_ratings.get(mid, 0.0)
                print(f"    - {self.movie_titles.get(mid, f'ID:{mid}')} ({rating})")

    def train_model(self) -> None:
        """
        Core Training Pipeline: Builds the item-item similarity model from training users.
        
        Steps:
        1.  **Global Stats**: Calculates Weighted Ratings (IMDB formula) for all movies
            to serve as a popularity fallback and quality baseline.
        2.  **Activity Filter**: Identifies "active" movies (min ratings) to reduce matrix sparsity.
        3.  **Co-occurrence Matrix**: Computes dot products between all movie pairs rated by same users.
        4.  **Cosine Similarity**: Normalizes dot products by Euclidean norms.
        5.  **Popularity Penalty**: Divides similarity by log(popularity) to dampen generic blockbuster correlations.
        6.  **Pruning**: Stores only Top-K (config.max_similar_items) neighbors per movie to optimize memory.
        
        Side Effects:
            - Populates self.global_movie_stats
            - Populates self.popular_movies (Fallback list)
            - Populates self.item_similarity (The Model)
        """
        logger.info("Training: Computing Stats & Similarity Matrix...")
        start_time = time.time()
        
        # 1. Filter to Train Set
        train_df = self.ratings_data[self.ratings_data['userId'].isin(self.train_users)]
        
        # 2. Global Stats (Weighted Rating / IMDB Formula)
        stats = train_df.groupby('movieId').agg(
            rating_count=('rating', 'count'), 
            rating_avg=('rating', 'mean')
        )
        C = stats['rating_avg'].mean()
        m = stats['rating_count'].quantile(0.90)
        
        # WR = (v / (v+m)) * R + (m / (v+m)) * C
        stats['score'] = stats.apply(
            lambda x: (x['rating_count']/(x['rating_count']+m) * x['rating_avg']) + (m/(x['rating_count']+m) * C), 
            axis=1
        )
        self.popular_movies = stats.sort_values('score', ascending=False).index.tolist()
        self.global_movie_stats = stats.to_dict('index')
        
        # 3. Item-Item Similarity
        # Filter sparse movies to speed up matrix calc
        active_movies = stats[stats['rating_count'] >= self.config.min_common_users_for_item_sim].index
        filtered_train = train_df[train_df['movieId'].isin(active_movies)]
        logger.info(f"Computing similarity for {len(active_movies)} active movies...")
        
        # Precompute Norms |A| (Euclidean length of rating vector)
        movie_norms = {}
        for mid in active_movies:
             ratings = filtered_train[filtered_train['movieId'] == mid]['rating']
             movie_norms[mid] = np.sqrt(np.sum(ratings ** 2))

        # Compute Dot Products (A . B) via inverted index
        # User -> [MovieA, MovieB] implies co-occurrence
        co_occur = defaultdict(dict)
        for user_id, group in filtered_train.groupby('userId'):
            user_ratings = list(zip(group['movieId'], group['rating']))
            
            # Pairwise combinations within user history
            for i in range(len(user_ratings)):
                m1, r1 = user_ratings[i]
                for j in range(i + 1, len(user_ratings)):
                    m2, r2 = user_ratings[j]
                    
                    prod = r1 * r2
                    if m2 not in co_occur[m1]: co_occur[m1][m2] = 0.0
                    co_occur[m1][m2] += prod
                    if m1 not in co_occur[m2]: co_occur[m2][m1] = 0.0
                    co_occur[m2][m1] += prod

        # Compute Final Cosine Score with Penalty
        self.item_similarity = {}
        for m1, neighbors in co_occur.items():
            scores = []
            norm1 = movie_norms.get(m1, 0)
            if norm1 == 0: continue
            
            for m2, dot_prod in neighbors.items():
                norm2 = movie_norms.get(m2, 0)
                if norm2 == 0: continue
                
                # Cosine = (A . B) / (|A| * |B|)
                cosine = dot_prod / (norm1 * norm2)
                
                # Popularity Penalty: Divide by log(count) to down-weight ubiquitous movies
                count_m2 = self.global_movie_stats.get(m2, {}).get('rating_count', 0)
                cosine /= np.log(count_m2 + 2)
                
                if cosine >= self.config.similarity_threshold:
                    scores.append((m2, cosine))
            
            # Keep Top-K neighbors
            scores.sort(key=lambda x: x[1], reverse=True)
            self.item_similarity[m1] = scores[:self.config.max_similar_items]
            
        logger.info(f"Training finished in {time.time() - start_time:.2f}s")

    def split_user_history(self, user_id: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Timeline Split:
        - Sort ratings by year.
        - First 80% = Known History (Input).
        - Last 20% = Hidden Future (Target).
        """
        user_ratings = self.ratings_data[self.ratings_data['userId'] == user_id].copy()
        user_ratings = user_ratings.sort_values('rating_year')
        
        split_point = int(len(user_ratings) * self.config.known_history_percentage)
        
        # Ensure at least 1 hidden item for testing if possible
        if len(user_ratings) >= 5:
            known = user_ratings.iloc[:split_point]
            hidden = user_ratings.iloc[split_point:]
        else:
            known = user_ratings.iloc[:-1] if len(user_ratings) > 1 else user_ratings
            hidden = user_ratings.iloc[-1:] if len(user_ratings) > 1 else pd.DataFrame()
            
        return known, hidden
    
    def predict_for_user(self, user_id: int) -> Tuple[List[int], List[int], Dict[int, float]]:
        """
        Generate recommendations for a specific user based on their known history.
        
        Args:
            user_id (int): ID of the user to predict for.
            
        Returns:
            Tuple containing:
            1. recommended_ids (List[int]): Top N movie IDs recommended by the system.
            2. actual_movies (List[int]): The hidden "future" movie IDs actually watched (Ground Truth).
            3. actual_ratings (Dict[int, float]): Dictionary of {movie_id: rating} for the hidden movies.
               Returns ([], [], {}) if user has insufficient history.
               
        Logic:
            - Uses positive ratings (>= threshold) from 'known' history as seeds.
            - Accumulates scores from similar items (Score = Sim * HistoryRating).
            - Fills remaining slots with Global Popular movies if needed (Fallback).
        """
        known, hidden = self.split_user_history(user_id)
        if len(hidden) == 0: return [], [], {}

        # 1. Gather Positive Feedback
        user_history = {
            m: r for m, r in zip(known['movieId'], known['rating']) 
            if r >= self.config.rating_threshold
        }
        
        # 2. Score Candidates (Weighted Sum of Similarities)
        candidates = defaultdict(float) 
        
        for movie_id, rating in user_history.items():
            if movie_id in self.item_similarity:
                for similar_movie, sim_score in self.item_similarity[movie_id]:
                    if similar_movie in user_history: continue
                        
                    # Score = Similarity * UserRating
                    # We sum huge scores if multiple watched movies point to the same candidate
                    candidates[similar_movie] += (sim_score * rating)
        
        # 3. Rank & Select Top N
        final_scores = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        recommended_ids = [m for m, s in final_scores[:self.config.top_n_recommendations]]
        
        # 4. Popularity Fallback (Fill remainder)
        if len(recommended_ids) < self.config.top_n_recommendations and self.config.enable_popularity_fallback:
            for pop_movie in self.popular_movies:
                if pop_movie not in user_history and pop_movie not in recommended_ids:
                    recommended_ids.append(pop_movie)
                    if len(recommended_ids) >= self.config.top_n_recommendations: break
        
        actual_movies = hidden['movieId'].tolist()
        actual_ratings = dict(zip(hidden['movieId'], hidden['rating']))
        
        return recommended_ids, actual_movies, actual_ratings
    
    def evaluate_predictions(self, predictions: List[int], actual_movies: List[int], 
                           actual_ratings_dict: Dict[int, float]) -> Dict[str, float]:
        """
        Calculate accuracy metrics for a single user by comparing predictions to ground truth.
        
        Args:
            predictions (List[int]): The list of recommended movie IDs.
            actual_movies (List[int]): The list of movies the user actually watched (hidden future).
            actual_ratings_dict (Dict[int, float]): The ratings user gave to the hidden movies.
            
        Returns:
            Dict[str, float]: dictionary including:
                - precision, recall, f1, hit_rate (Binary Match)
                - enjoyment_precision, enjoyment_recall, enjoyment_f1 (Rated >= threshold)
                - hits, enjoyment_hits, total_predictions, total_actual (Raw counts)
        """
        if not actual_movies:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'hit_rate': 0.0, 
                   'enjoyment_precision': 0.0, 'enjoyment_recall': 0.0, 'enjoyment_f1': 0.0}
        
        preds_set, actual_set = set(predictions), set(actual_movies)
        
        # Overlap (Hits)
        hits = preds_set.intersection(actual_set)
        basic_hits = len(hits)
        
        # Confusion Matrix (Standard)
        tp = basic_hits
        fp = len(predictions) - tp
        fn = len(actual_movies) - tp
        tn = self.total_movies - (tp + fp + fn)

        # Enjoyment (Hits that were also rated positively)
        # Ground truth for enjoyment: Movies watched AND rated >= threshold
        actual_liked = {m for m in actual_movies if actual_ratings_dict.get(m, 0) >= self.config.rating_threshold}
        
        # Enjoyment TP: Recommended AND Liked
        enjoyment_hits = len(preds_set.intersection(actual_liked))
        e_tp = enjoyment_hits
        
        # Enjoyment FP: Recommended AND (Not Watched OR Not Liked)
        e_fp = len(predictions) - e_tp
        
        # Enjoyment FN: Liked BUT Not Recommended
        e_fn = len(actual_liked) - e_tp
        
        # Enjoyment TN: Not Recommended AND (Not Watched OR Not Liked)
        e_tn = self.total_movies - (e_tp + e_fp + e_fn)

        # Basic Metrics
        prec = basic_hits / len(predictions) if predictions else 0.0
        rec = basic_hits / len(actual_movies)
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        acc = (tp + tn) / self.total_movies if self.total_movies > 0 else 0.0
        
        # Enjoyment Metrics
        e_prec = enjoyment_hits / len(predictions) if predictions else 0.0
        e_rec = enjoyment_hits / len(actual_liked) if actual_liked else 0.0
        e_f1 = 2 * e_prec * e_rec / (e_prec + e_rec) if (e_prec + e_rec) > 0 else 0.0
        
        return {
            'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'hit_rate': 1.0 if basic_hits > 0 else 0.0, 'hits': basic_hits,
            'enjoyment_precision': e_prec, 'enjoyment_recall': e_rec, 'enjoyment_f1': e_f1, 'enjoyment_hits': enjoyment_hits,
            'total_predictions': len(predictions), 'total_actual': len(actual_movies),
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'e_tp': e_tp, 'e_fp': e_fp, 'e_fn': e_fn, 'e_tn': e_tn
        }
    
    def _eval_user_wrapper(self, user_id: int) -> Dict[str, float]:
        """Helper for multiprocessing evaluation."""
        try:
            predictions, actual_movies, actual_ratings_dict = self.predict_for_user(user_id)
            if predictions and actual_movies:
                return self.evaluate_predictions(predictions, actual_movies, actual_ratings_dict)
        except Exception:
            return None
        return None
    
    def run_evaluation(self, max_test_users: int = None) -> Dict[str, float]:
        """
        Execute the full evaluation pipeline on the Test User set.
        
        Args:
            max_test_users (int, optional): Limit number of users to evaluate (for fast debugging).
            
        Returns:
            Dict[str, float]: Aggregated average metrics across all evaluated users (Precision, Recall, Hit Rate, etc.).
            
        Execution Mode:
            - Sequential: If users < 100 or parallelism disabled.
            - Parallel: Uses multiprocessing Pool if users > 100 and enabled, to speed up CPU-bound evaluation.
        """
        logger.info("Starting Evaluation...")
        if not self.train_users or not self.test_users: self.split_data()
        
        test_users_list = list(self.test_users)
        if max_test_users: test_users_list = test_users_list[:max_test_users]
        
        all_metrics = []
        successful = 0
        
        # Parallel Execution
        if self.config.use_parallel and len(test_users_list) > 100:
            num_proc = self.config.num_processes or max(1, cpu_count() - 1)
            logger.info(f"Parallel Eval: {num_proc} cores")
            
            with Pool(processes=num_proc) as pool:
                chunk = max(1, len(test_users_list) // (num_proc * 4))
                for i, m in enumerate(pool.imap_unordered(self._eval_user_wrapper, test_users_list, chunksize=chunk)):
                    if i % 100 == 0: logger.info(f"Evaluated {i}/{len(test_users_list)}")
                    if m: 
                        all_metrics.append(m)
                        successful += 1
                        
        # Sequential Execution
        else:
            for i, uid in enumerate(test_users_list):
                if i % 100 == 0: logger.info(f"Evaluated {i}/{len(test_users_list)}")
                m = self._eval_user_wrapper(uid)
                if m:
                    all_metrics.append(m)
                    successful += 1
        
        if not all_metrics:
            logger.error("No predictions made.")
            return {}
        
        # Averages
        avg_metrics = {
            k: float(np.mean([m[k.replace('avg_', '')] for m in all_metrics]))
            for k in ['avg_accuracy', 'avg_precision', 'avg_recall', 'avg_f1', 'avg_hit_rate', 
                      'avg_enjoyment_precision', 'avg_enjoyment_recall', 'avg_enjoyment_f1']
        }
        
        # Confusion Matrix Averages
        cm_metrics = {
            k: float(np.mean([m[k.replace('avg_', '')] for m in all_metrics]))
            for k in ['avg_tp', 'avg_fp', 'avg_fn', 'avg_tn',
                      'avg_e_tp', 'avg_e_fp', 'avg_e_fn', 'avg_e_tn']
        }
        avg_metrics.update(cm_metrics)
        
        avg_metrics.update({'total_users_evaluated': len(all_metrics), 'successful_prediction_rate': successful / len(test_users_list)})
        
        return avg_metrics
    
    @staticmethod
    def _print_results_summary(results: Dict[str, float], title: str = "RESULTS") -> None:
        """Output formatted metrics to console."""
        if not results: return
        print(f"\n=== {title} ===")
        print(f"Evaluated: {results['total_users_evaluated']} users (Success: {results['successful_prediction_rate']:.1%})")
        print("\n--- Match Accuracy ---")
        print(f"  Accuracy:  {results['avg_accuracy']:.2%}")
        print(f"  Precision: {results['avg_precision']:.1%}")
        print(f"  Recall:    {results['avg_recall']:.1%}")
        print(f"  Hit Rate:  {results['avg_hit_rate']:.1%}")
        print("\n--- Enjoyment Accuracy (User Liked It) ---")
        print(f"  Precision: {results['avg_enjoyment_precision']:.1%}")
        print(f"  Recall:    {results['avg_enjoyment_recall']:.1%}")
        
        print("\n--- Confusion Matrix (Avg per User) ---")
        print(f"  TP (Hits):    {results['avg_tp']:.2f}")
        print(f"  FP (Miss):    {results['avg_fp']:.2f}")
        print(f"  FN (Lost):    {results['avg_fn']:.2f}")
        print(f"  TN (Ignore):  {results['avg_tn']:.2f}")
        
        print("\n--- Enjoyment Confusion Matrix (Rated >= 3.5) ---")
        print(f"  E-TP (Liked & Rec'd):   {results['avg_e_tp']:.2f}")
        print(f"  E-FP (Bad Rec):         {results['avg_e_fp']:.2f}")
        print(f"  E-FN (Missed gem):      {results['avg_e_fn']:.2f}")
        print(f"  E-TN (Correctly skip):  {results['avg_e_tn']:.2f}")


def main():
    """
    Entry point.
    1. Initialize and Load Data.
    2. Run Full Evaluation.
    3. Print Metrics and Sample Recommendations.
    """
    config = CollaborativeFilteringConfig()
    cf = CollaborativeFiltering(config)
    
    ratings_file = "/home/woto/Desktop/personal/python/thesis/cleaned_data/ratings_cleaned.csv"
    cf.load_data(ratings_file)
    
    # Run full evaluation
    print("\n=== RUNNING EVALUATION ===")
    results = cf.run_evaluation()
    
    cf._print_results_summary(results, title="EVALUATION")
    cf.inspect_sample_recommendations(num_users=3)


if __name__ == "__main__":
    main()
