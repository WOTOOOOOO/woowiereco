"""
Data Cleaning Pipelines for Movie Dataset
=========================================

This module contains separate, modular cleaning pipelines for each CSV file in the dataset.
Each pipeline is designed to handle specific data quality issues while being reusable and maintainable.
"""

import pandas as pd
import numpy as np
import json
import ast
from typing import List, Dict, Any, Optional
import warnings
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class DataValidator:
    """Utility class for common data validation operations."""
    
    @staticmethod
    def validate_json_field(value: Any) -> bool:
        """Check if a field contains valid JSON data."""
        if pd.isna(value) or value == '':
            return False
        try:
            if isinstance(value, str):
                json.loads(value)
                return True
            return False
        except (json.JSONDecodeError, TypeError):
            return False
    
    @staticmethod
    def safe_json_parse(value: Any, default: Any = None) -> Any:
        """Safely parse JSON with fallback to default value."""
        if pd.isna(value) or value == '':
            return default
        try:
            if isinstance(value, str):
                return json.loads(value)
            return value
        except (json.JSONDecodeError, TypeError, ValueError):
            try:
                # Try ast.literal_eval as fallback for malformed JSON
                return ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return default
    
    @staticmethod
    def validate_numeric_value(series: pd.Series) -> pd.Series:
        """Validate that values are finite numbers (not NaN, not inf, not negative where not allowed)."""
        # Accept only finite, non-negative numbers (for most movie fields)
        return series.apply(lambda x: pd.notna(x) and np.isfinite(x) and x >= 0)


class CreditsDataCleaner:
    """Cleaning pipeline for credits.csv data."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.validator = DataValidator()
    
    def load_data(self) -> pd.DataFrame:
        """Load credits data with proper error handling."""
        logger.info(f"Loading credits data from {self.file_path}")
        try:
            df = pd.read_csv(self.file_path)
            logger.info(f"Loaded {len(df)} records from credits.csv")
            return df
        except Exception as e:
            logger.error(f"Error loading credits data: {e}")
            raise
    
    def clean_cast_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate cast information - keep only recommendation-relevant data."""
        logger.info("Cleaning cast data")
        
        # Parse cast JSON safely
        df['cast_parsed'] = df['cast'].apply(
            lambda x: self.validator.safe_json_parse(x, [])
        )
        
        # Extract all actor names (for collaborative filtering and content-based recommendations)
        df['actors'] = df['cast_parsed'].apply(
            lambda x: [member.get('name', '') for member in x 
                      if isinstance(member, dict) and member.get('name')] if isinstance(x, list) else []
        )
        
        # Extract main actors (top 5 by billing order for lead actor analysis)
        df['lead_actors'] = df['cast_parsed'].apply(
            lambda x: [member.get('name', '') for member in sorted(x, key=lambda m: m.get('order', 999))[:5] 
                      if isinstance(member, dict) and member.get('name')] if isinstance(x, list) else []
        )
        
        # Actor count (useful feature for recommendation)
        df['cast_size'] = df['actors'].apply(len)
        
        return df
    
    def clean_crew_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate crew information - keep only key roles for recommendations."""
        logger.info("Cleaning crew data")
        
        # Parse crew JSON safely
        df['crew_parsed'] = df['crew'].apply(
            lambda x: self.validator.safe_json_parse(x, [])
        )
        
        # Extract key roles that matter for movie recommendations
        df['directors'] = df['crew_parsed'].apply(
            lambda x: [member.get('name', '') for member in x 
                      if isinstance(member, dict) and member.get('job') == 'Director'] if isinstance(x, list) else []
        )
        
        df['producers'] = df['crew_parsed'].apply(
            lambda x: [member.get('name', '') for member in x 
                      if isinstance(member, dict) and 'Producer' in member.get('job', '')] if isinstance(x, list) else []
        )
        
        df['writers'] = df['crew_parsed'].apply(
            lambda x: [member.get('name', '') for member in x 
                      if isinstance(member, dict) and member.get('department') == 'Writing'] if isinstance(x, list) else []
        )
        
        # Music composers (important for some recommendation algorithms)
        df['composers'] = df['crew_parsed'].apply(
            lambda x: [member.get('name', '') for member in x 
                      if isinstance(member, dict) and 'Music' in member.get('job', '') 
                      and 'Composer' in member.get('job', '')] if isinstance(x, list) else []
        )
        
        return df
    
    def validate_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate movie IDs."""
        logger.info("Validating movie IDs in credits data")
        
        # Ensure ID is numeric and positive
        df['id'] = pd.to_numeric(df['id'], errors='coerce')
        
        # Mark invalid IDs
        df['valid_id'] = (df['id'] > 0) & df['id'].notna()
        
        # Log validation results
        invalid_count = (~df['valid_id']).sum()
        if invalid_count > 0:
            logger.warning(f"Found {invalid_count} records with invalid IDs")
        
        return df
    
    def clean_data(self) -> pd.DataFrame:
        """Execute full cleaning pipeline for credits data."""
        logger.info("Starting credits data cleaning pipeline")
        
        df = self.load_data()
        df = self.clean_cast_data(df)
        df = self.clean_crew_data(df)
        df = self.validate_ids(df)
        
        # Keep only recommendation-relevant columns
        recommendation_columns = [
            'id', 'actors', 'lead_actors', 'cast_size', 
            'directors', 'producers', 'writers', 'composers'
        ]
        
        # Filter to keep only useful columns
        df_clean = df[recommendation_columns].copy()
        
        # Create summary statistics
        logger.info(f"Credits cleaning complete. Final dataset: {len(df_clean)} records")
        logger.info(f"Total unique actors extracted: {len(set([actor for actors in df_clean['actors'] for actor in actors]))}")
        logger.info(f"Average actors per movie: {df_clean['cast_size'].mean():.2f}")
        
        return df_clean


class KeywordsDataCleaner:
    """Cleaning pipeline for keywords.csv data."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.validator = DataValidator()
    
    def load_data(self) -> pd.DataFrame:
        """Load keywords data with proper error handling."""
        logger.info(f"Loading keywords data from {self.file_path}")
        try:
            df = pd.read_csv(self.file_path)
            logger.info(f"Loaded {len(df)} records from keywords.csv")
            return df
        except Exception as e:
            logger.error(f"Error loading keywords data: {e}")
            raise
    
    def clean_keywords(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate keywords data - keep only keyword names for recommendations."""
        logger.info("Cleaning keywords data")
        
        # Parse keywords JSON safely
        df['keywords_parsed'] = df['keywords'].apply(
            lambda x: self.validator.safe_json_parse(x, [])
        )
        
        # Extract only keyword names (the most important part for content-based recommendations)
        df['keywords'] = df['keywords_parsed'].apply(
            lambda x: [kw.get('name', '').lower() for kw in x if isinstance(kw, dict) and kw.get('name')] if isinstance(x, list) else []
        )
        
        # Keyword count (useful feature)
        df['keyword_count'] = df['keywords'].apply(len)
        
        return df
    
    def validate_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate movie IDs."""
        logger.info("Validating movie IDs in keywords data")
        
        # Ensure ID is numeric and positive
        df['id'] = pd.to_numeric(df['id'], errors='coerce')
        
        # Mark invalid IDs
        df['valid_id'] = (df['id'] > 0) & df['id'].notna()
        
        # Log validation results
        invalid_count = (~df['valid_id']).sum()
        if invalid_count > 0:
            logger.warning(f"Found {invalid_count} records with invalid IDs")
        
        return df
    
    def clean_data(self) -> pd.DataFrame:
        """Execute full cleaning pipeline for keywords data."""
        logger.info("Starting keywords data cleaning pipeline")
        
        df = self.load_data()
        df = self.clean_keywords(df)
        df = self.validate_ids(df)
        
        # Keep only recommendation-relevant columns
        recommendation_columns = ['id', 'keywords', 'keyword_count']
        df_clean = df[recommendation_columns].copy()
        
        # Create summary statistics
        logger.info(f"Keywords cleaning complete. Final dataset: {len(df_clean)} records")
        logger.info(f"Average keywords per movie: {df_clean['keyword_count'].mean():.2f}")
        
        return df_clean



class MoviesMetadataCleaner:
    """Cleaning pipeline for movies_metadata.csv data."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.validator = DataValidator()
    
    def load_data(self) -> pd.DataFrame:
        """Load movies metadata with proper error handling."""
        logger.info(f"Loading movies metadata from {self.file_path}")
        try:
            # Use low_memory=False to avoid dtype warnings for mixed types
            df = pd.read_csv(self.file_path, low_memory=False)
            logger.info(f"Loaded {len(df)} records from movies_metadata.csv")
            return df
        except Exception as e:
            logger.error(f"Error loading movies metadata: {e}")
            raise
    
    def clean_numeric_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean numeric fields like budget, revenue, runtime, etc."""
        logger.info("Cleaning numeric fields")
        
        # Clean budget and revenue
        numeric_fields = ['budget', 'revenue', 'runtime', 'vote_average', 'vote_count', 'popularity']
        
        for field in numeric_fields:
            if field in df.columns:
                # Convert to numeric, replacing invalid values with NaN
                df[field] = pd.to_numeric(df[field], errors='coerce')
                
                # Validate only that the value is a valid number (not outlier filtering)
                df[f'{field}_valid'] = self.validator.validate_numeric_value(df[field])
            else:
                    # If field is missing, add it as NaN and mark all as invalid
                    df[field] = np.nan
                    df[f'{field}_valid'] = False
        
        return df
    
    def clean_date_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean date fields."""
        logger.info("Cleaning date fields")
        
        if 'release_date' in df.columns:
            # Convert to datetime
            df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
            # Validate reasonable date range (movies from 1900 to current year + 5)
            current_year = datetime.now().year
            df['release_date_valid'] = (
                df['release_date'].notna() & 
                (df['release_date'].dt.year <= current_year)
            )
            # Extract useful date components
            df['release_year'] = df['release_date'].dt.year
            df['release_month'] = df['release_date'].dt.month
            df['release_decade'] = (df['release_year'] // 10) * 10
        else:
            # If field is missing, add it as NaT and mark all as invalid
            df['release_date'] = pd.NaT
            df['release_date_valid'] = False
            df['release_year'] = np.nan
            df['release_month'] = np.nan
            df['release_decade'] = np.nan
        
        return df
    
    def clean_json_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean JSON fields - keep only recommendation-relevant data."""
        logger.info("Cleaning JSON fields")
        
        # Only process genres (most important for recommendations)
        if 'genres' in df.columns:
            df['genres_parsed'] = df['genres'].apply(
                lambda x: self.validator.safe_json_parse(x, [])
            )
            df['genres'] = df['genres_parsed'].apply(
                lambda x: [g.get('name', '') for g in x if isinstance(g, dict)] if isinstance(x, list) else []
            )
            df['genre_count'] = df['genres'].apply(len)
            df['primary_genre'] = df['genres'].apply(lambda x: x[0] if x else None)
        else:
            df['genres'] = [[] for _ in range(len(df))]
            df['genre_count'] = 0
            df['primary_genre'] = None
        
        # Production companies (can be useful for studio-based recommendations)
        if 'production_companies' in df.columns:
            df['production_companies_parsed'] = df['production_companies'].apply(
                lambda x: self.validator.safe_json_parse(x, [])
            )
            df['production_companies'] = df['production_companies_parsed'].apply(
                lambda x: [c.get('name', '') for c in x if isinstance(c, dict)] if isinstance(x, list) else []
            )
            df['main_studio'] = df['production_companies'].apply(lambda x: x[0] if x else None)
        else:
            df['production_companies'] = [[] for _ in range(len(df))]
            df['main_studio'] = None
        
        # Collections (useful for franchise-based recommendations)
        if 'belongs_to_collection' in df.columns:
            df['collection_parsed'] = df['belongs_to_collection'].apply(
                lambda x: self.validator.safe_json_parse(x, None)
            )
            df['collection_name'] = df['collection_parsed'].apply(
                lambda x: x.get('name', '') if isinstance(x, dict) else ''
            )
            df['is_part_of_collection'] = df['collection_name'] != ''
        else:
            df['collection_name'] = ''
            df['is_part_of_collection'] = False
        
        return df
    
    def clean_categorical_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean categorical fields - keep only recommendation-relevant ones."""
        logger.info("Cleaning categorical fields")
        
        # Clean adult field (important for content filtering in recommendations)
        if 'adult' in df.columns:
            df['adult'] = df['adult'].map({'False': False, 'True': True, False: False, True: True})
            df['adult'] = df['adult'].fillna(False).astype(bool)
        else:
            df['adult'] = False
            
        # Clean language codes (important for recommendations)
        if 'original_language' in df.columns:
            df['original_language'] = df['original_language'].str.lower().str.strip()
            df['is_english'] = df['original_language'] == 'en'
        else:
            df['original_language'] = np.nan
            df['is_english'] = False

        return df
    
    def validate_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate movie IDs."""
        logger.info("Validating movie IDs in metadata")
        
        # Clean main ID
        if 'id' in df.columns:
            df['id'] = pd.to_numeric(df['id'], errors='coerce')
            df['valid_id'] = (df['id'] > 0) & df['id'].notna()
        
        return df
    
    def clean_data(self) -> pd.DataFrame:
        """Execute full cleaning pipeline for movies metadata."""
        logger.info("Starting movies metadata cleaning pipeline")
        
        df = self.load_data()
        df = self.clean_numeric_fields(df)
        df = self.clean_date_fields(df)
        df = self.clean_json_fields(df)
        df = self.clean_categorical_fields(df)
        df = self.validate_ids(df)
        
        # Keep only recommendation-relevant columns
        recommendation_columns = [
            'id', 'title', 'original_title', 'overview',
            'genres', 'genre_count', 'primary_genre',
            'production_companies', 'main_studio', 
            'collection_name', 'is_part_of_collection',
            'release_date', 'release_year', 'release_decade',
            'budget', 'revenue', 'runtime', 
            'original_language', 'is_english', 'adult',
            'vote_average', 'vote_count', 'popularity'
        ]
        
        # Filter to keep only useful columns that exist in the dataframe
        available_columns = [col for col in recommendation_columns if col in df.columns]
        df_clean = df[available_columns].copy()
        
        # Create summary statistics
        logger.info(f"Movies metadata cleaning complete. Final dataset: {len(df_clean)} records")
        if 'release_date' in df_clean.columns:
            logger.info(f"Records with valid release dates: {df_clean['release_date'].notna().sum()}")
        if 'budget' in df_clean.columns:
            logger.info(f"Records with valid budgets: {(df_clean['budget'] > 0).sum()}")
        
        return df_clean


class RatingsDataCleaner:
    """Cleaning pipeline for ratings.csv data."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.validator = DataValidator()
    
    def load_data(self, chunk_size: int = 100000) -> pd.DataFrame:
        """Load ratings data in chunks due to large size."""
        logger.info(f"Loading ratings data from {self.file_path}")
        try:
            # For very large files, consider loading in chunks
            df = pd.read_csv(self.file_path)
            logger.info(f"Loaded {len(df)} records from ratings.csv")
            return df
        except Exception as e:
            logger.error(f"Error loading ratings data: {e}")
            raise
    
    def clean_ratings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate rating values."""
        logger.info("Cleaning rating values")
        
        # Convert rating to numeric
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        
        # Validate rating range (typically 0.5 to 5.0 in 0.5 increments)
        valid_ratings = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        df['rating_valid'] = df['rating'].isin(valid_ratings)
        
        # Mark invalid ratings
        invalid_ratings = (~df['rating_valid']) & df['rating'].notna()
        if invalid_ratings.sum() > 0:
            logger.warning(f"Found {invalid_ratings.sum()} invalid rating values")
        
        return df
    
    def clean_user_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate user IDs."""
        logger.info("Cleaning user IDs")
        
        # Convert to numeric
        df['userId'] = pd.to_numeric(df['userId'], errors='coerce')
        
        # Validate positive integers
        df['userId_valid'] = (df['userId'] > 0) & df['userId'].notna()
        
        return df
    
    def clean_movie_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate movie IDs."""
        logger.info("Cleaning movie IDs")
        
        # Convert to numeric
        df['movieId'] = pd.to_numeric(df['movieId'], errors='coerce')
        
        # Validate positive integers
        df['movieId_valid'] = (df['movieId'] > 0) & df['movieId'].notna()
        
        return df
    
    def clean_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate timestamps - keep simple for recommendations."""
        logger.info("Cleaning timestamps")
        
        # Convert to numeric first
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
        
        # Convert to datetime for basic validation
        df['rating_datetime'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
        
        # Validate reasonable date range
        current_year = datetime.now().year
        df['valid_timestamp'] = (
            df['rating_datetime'].notna() & 
            (df['rating_datetime'].dt.year <= current_year)
        )
        
        # Keep only the year for temporal recommendations
        df['rating_year'] = df['rating_datetime'].dt.year
        
        return df
    
    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect basic anomalies - simplified for recommendation system."""
        logger.info("Detecting basic rating anomalies")
        
        # Check for movies with very few ratings (less useful for recommendations)
        movie_rating_counts = df.groupby('movieId')['rating'].count()
        df['movie_rating_count'] = df['movieId'].map(movie_rating_counts)
        df['reliable_movie'] = df['movie_rating_count'] >= 5  # At least 5 ratings
        
        return df
    
    def clean_data(self) -> pd.DataFrame:
        """Execute full cleaning pipeline for ratings data."""
        logger.info("Starting ratings data cleaning pipeline")
        
        df = self.load_data()
        df = self.clean_ratings(df)
        df = self.clean_user_ids(df)
        df = self.clean_movie_ids(df)
        df = self.clean_timestamps(df)
        df = self.detect_anomalies(df)
        
        # Keep only essential recommendation columns
        recommendation_columns = [
            'userId', 'movieId', 'rating', 'rating_year', 
            'movie_rating_count', 'reliable_movie'
        ]
        
        # Filter to valid records and useful columns
        valid_records = (
            df['rating_valid'] & 
            df['userId_valid'] & 
            df['movieId_valid'] & 
            df['valid_timestamp']
        )
        
        df_clean = df[valid_records][recommendation_columns].copy()
        
        logger.info(f"Ratings cleaning complete. Final dataset: {len(df_clean)} records")
        logger.info(f"Unique users: {df_clean['userId'].nunique()}")
        logger.info(f"Unique movies: {df_clean['movieId'].nunique()}")
        logger.info(f"Reliable movies (5+ ratings): {df_clean['reliable_movie'].sum()}")
        
        return df_clean


# Main execution functions
def clean_all_datasets(data_dir: str = './data') -> Dict[str, pd.DataFrame]:
    """
    Execute all cleaning pipelines and return cleaned datasets optimized for movie recommendations.
    
    Args:
        data_dir: Directory containing the CSV files
        
    Returns:
        Dictionary of cleaned DataFrames containing only recommendation-relevant data
    """
    import os
    
    cleaned_data = {}
    
    # Define file paths - only include files useful for recommendations
    files = {
        'credits': os.path.join(data_dir, 'credits.csv'),
        'keywords': os.path.join(data_dir, 'keywords.csv'),
        'movies_metadata': os.path.join(data_dir, 'movies_metadata.csv'),
        'ratings': os.path.join(data_dir, 'ratings.csv')  # Use small file for faster processing
    }
    
    # Execute cleaning pipelines
    try:
        logger.info("Starting recommendation-focused data cleaning process")
        
        cleaned_data['credits'] = CreditsDataCleaner(files['credits']).clean_data()
        cleaned_data['keywords'] = KeywordsDataCleaner(files['keywords']).clean_data()
        cleaned_data['movies_metadata'] = MoviesMetadataCleaner(files['movies_metadata']).clean_data()
        cleaned_data['ratings'] = RatingsDataCleaner(files['ratings']).clean_data()
        
        logger.info("All recommendation datasets cleaned successfully!")
        
    except Exception as e:
        logger.error(f"Error in cleaning process: {e}")
        raise
    
    return cleaned_data


def save_cleaned_datasets(cleaned_data: Dict[str, pd.DataFrame], output_dir: str = './cleaned_data'):
    """
    Save cleaned datasets to specified directory.
    
    Args:
        cleaned_data: Dictionary of cleaned DataFrames
        output_dir: Output directory for cleaned files
    """
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for dataset_name, df in cleaned_data.items():
        output_path = os.path.join(output_dir, f'{dataset_name}_cleaned.csv')
        df.to_csv(output_path, index=False)
        logger.info(f"Saved cleaned {dataset_name} data to {output_path}")


if __name__ == "__main__":
    # Example usage
    data_directory = "/home/woto/Desktop/personal/python/thesis/data"
    # Clean all datasets
    cleaned_datasets = clean_all_datasets(data_directory)
    
    # Save cleaned datasets
    save_cleaned_datasets(cleaned_datasets)
    
    print("Data cleaning completed successfully!")
