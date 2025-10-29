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
        """Clean and validate cast information."""
        logger.info("Cleaning cast data")
        
        # Parse cast JSON safely
        df['cast_parsed'] = df['cast'].apply(
            lambda x: self.validator.safe_json_parse(x, [])
        )
        
        # Extract useful cast metrics
        df['num_cast_members'] = df['cast_parsed'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        df['has_cast_data'] = df['cast_parsed'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)
        
        # Extract all actor names from cast
        df['actor_names'] = df['cast_parsed'].apply(
            lambda x: [member.get('name', '') for member in x 
                      if isinstance(member, dict) and member.get('name')] if isinstance(x, list) else []
        )
        
        # Extract character names
        df['character_names'] = df['cast_parsed'].apply(
            lambda x: [member.get('character', '') for member in x 
                      if isinstance(member, dict) and member.get('character')] if isinstance(x, list) else []
        )
        
        # Extract main cast (first 10 members by order) with detailed info
        df['main_cast'] = df['cast_parsed'].apply(
            lambda x: [{'name': member.get('name', ''), 
                       'character': member.get('character', ''),
                       'order': member.get('order', 999)} 
                      for member in x if isinstance(member, dict) and member.get('order', 999) < 10] if isinstance(x, list) else []
        )
        
        # Create concatenated string of all actor names for easy text analysis
        df['actors_text'] = df['actor_names'].apply(
            lambda x: ', '.join(x) if x else ''
        )
        
        return df
    
    def clean_crew_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate crew information."""
        logger.info("Cleaning crew data")
        
        # Parse crew JSON safely
        df['crew_parsed'] = df['crew'].apply(
            lambda x: self.validator.safe_json_parse(x, [])
        )
        
        # Extract crew metrics
        df['num_crew_members'] = df['crew_parsed'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        df['has_crew_data'] = df['crew_parsed'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)
        
        # Extract all crew members with their roles/jobs and departments, sorted by department, job, name
        df['crew_members'] = df['crew_parsed'].apply(
            lambda x: sorted(
                [
                    {
                        'name': member.get('name', ''),
                        'job': member.get('job', ''),
                        'department': member.get('department', '')
                    }
                    for member in x if isinstance(member, dict) and member.get('name')
                ],
                key=lambda m: (m['job'] or '', m['department'] or '', m['name'] or '')
            ) if isinstance(x, list) else []
        )
        
        # Extract specific key roles for backward compatibility and easy access
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
        
        # Create text representations for analysis
        df['crew_text'] = df['crew_members'].apply(
            lambda x: ', '.join([f"{member['name']} ({member['job']})" for member in x]) if x else ''
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
        
        # Create summary statistics
        logger.info(f"Credits cleaning complete. Final dataset: {len(df)} records")
        logger.info(f"Records with cast data: {df['has_cast_data'].sum()}")
        logger.info(f"Records with crew data: {df['has_crew_data'].sum()}")
        logger.info(f"Total unique actors extracted: {len(set([actor for actors in df['actor_names'] for actor in actors]))}")
        logger.info(f"Average actors per movie: {df['num_cast_members'].mean():.2f}")
        logger.info(f"Average crew members per movie: {df['num_crew_members'].mean():.2f}")
        
        return df


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
        """Clean and validate keywords data."""
        logger.info("Cleaning keywords data")
        
        # Parse keywords JSON safely
        df['keywords_parsed'] = df['keywords'].apply(
            lambda x: self.validator.safe_json_parse(x, [])
        )
        
        # Extract keyword metrics
        df['num_keywords'] = df['keywords_parsed'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        df['has_keywords'] = df['keywords_parsed'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)
        
        # Extract keyword names as list
        df['keyword_names'] = df['keywords_parsed'].apply(
            lambda x: [kw.get('name', '') for kw in x if isinstance(kw, dict) and kw.get('name')] if isinstance(x, list) else []
        )
        
        # Create keyword string for text analysis
        df['keywords_text'] = df['keyword_names'].apply(
            lambda x: ', '.join(x) if x else ''
        )
        
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
        
        # Create summary statistics
        logger.info(f"Keywords cleaning complete. Final dataset: {len(df)} records")
        logger.info(f"Records with keywords: {df['has_keywords'].sum()}")
        logger.info(f"Average keywords per movie: {df['num_keywords'].mean():.2f}")
        
        return df


class LinksDataCleaner:
    """Cleaning pipeline for links.csv data."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.validator = DataValidator()
    
    def load_data(self) -> pd.DataFrame:
        """Load links data with proper error handling."""
        logger.info(f"Loading links data from {self.file_path}")
        try:
            df = pd.read_csv(self.file_path)
            logger.info(f"Loaded {len(df)} records from links.csv")
            return df
        except Exception as e:
            logger.error(f"Error loading links data: {e}")
            raise
    
    def clean_movie_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate movie ID mappings."""
        logger.info("Cleaning movie ID mappings")
        
        # Convert all ID columns to numeric
        id_columns = ['movieId', 'tmdbId']
        for col in id_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle IMDB IDs (they have leading zeros and should be strings)
        if 'imdbId' in df.columns:
            df['imdbId'] = df['imdbId'].astype(str).str.zfill(7)  # Pad with zeros
            df['imdbId_clean'] = 'tt' + df['imdbId']  # Standard IMDB format
        
        # Validate all IDs are present and valid
        df['has_movieId'] = df['movieId'].notna() & (df['movieId'] > 0)
        df['has_imdbId'] = df['imdbId'].notna() & (df['imdbId'] != '0000000')
        df['has_tmdbId'] = df['tmdbId'].notna() & (df['tmdbId'] > 0)
        
        # Mark complete records
        df['complete_mapping'] = df['has_movieId'] & df['has_imdbId'] & df['has_tmdbId']
        
        return df
    
    def detect_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and flag duplicate ID mappings."""
        logger.info("Detecting duplicate mappings")
        
        # Check for duplicate movieIds
        df['duplicate_movieId'] = df.duplicated(subset=['movieId'], keep=False)
        
        # Check for duplicate IMDb IDs
        if 'imdbId' in df.columns:
            df['duplicate_imdbId'] = df.duplicated(subset=['imdbId'], keep=False)
        
        # Check for duplicate TMDB IDs
        if 'tmdbId' in df.columns:
            df['duplicate_tmdbId'] = df.duplicated(subset=['tmdbId'], keep=False)
        
        return df
    
    def clean_data(self) -> pd.DataFrame:
        """Execute full cleaning pipeline for links data."""
        logger.info("Starting links data cleaning pipeline")
        
        df = self.load_data()
        df = self.clean_movie_ids(df)
        df = self.detect_duplicates(df)
        
        # Create summary statistics
        complete_mappings = df['complete_mapping'].sum()
        logger.info(f"Links cleaning complete. Final dataset: {len(df)} records")
        logger.info(f"Complete mappings: {complete_mappings} ({complete_mappings/len(df)*100:.1f}%)")
        
        return df


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
        """Clean JSON fields like genres, production_companies, etc."""
        logger.info("Cleaning JSON fields")
        
        json_fields = ['genres', 'production_companies', 'production_countries', 
                      'spoken_languages', 'belongs_to_collection']
        
        for field in json_fields:
            if field in df.columns:
                # Parse JSON safely
                df[f'{field}_parsed'] = df[field].apply(
                    lambda x: self.validator.safe_json_parse(x, [])
                )
                # Extract names/values from parsed data
                if field == 'genres':
                    df['genre_names'] = df[f'{field}_parsed'].apply(
                        lambda x: [g.get('name', '') for g in x if isinstance(g, dict)] if isinstance(x, list) else []
                    )
                    df['num_genres'] = df['genre_names'].apply(len)
                    df['primary_genre'] = df['genre_names'].apply(lambda x: x[0] if x else None)
                elif field == 'production_companies':
                    df['production_company_names'] = df[f'{field}_parsed'].apply(
                        lambda x: [c.get('name', '') for c in x if isinstance(c, dict)] if isinstance(x, list) else []
                    )
                    df['num_production_companies'] = df['production_company_names'].apply(len)
                elif field == 'production_countries':
                    df['production_country_codes'] = df[f'{field}_parsed'].apply(
                        lambda x: [c.get('iso_3166_1', '') for c in x if isinstance(c, dict)] if isinstance(x, list) else []
                    )
                    df['num_production_countries'] = df['production_country_codes'].apply(len)
                elif field == 'spoken_languages':
                    df['spoken_language_codes'] = df[f'{field}_parsed'].apply(
                        lambda x: [l.get('iso_639_1', '') for l in x if isinstance(l, dict)] if isinstance(x, list) else []
                    )
                    df['num_spoken_languages'] = df['spoken_language_codes'].apply(len)
            else:
                # If field is missing, add empty/NaN columns for all expected derived columns
                df[f'{field}_parsed'] = [[] for _ in range(len(df))]
                if field == 'genres':
                    df['genre_names'] = [[] for _ in range(len(df))]
                    df['num_genres'] = np.nan
                    df['primary_genre'] = np.nan
                elif field == 'production_companies':
                    df['production_company_names'] = [[] for _ in range(len(df))]
                    df['num_production_companies'] = np.nan
                elif field == 'production_countries':
                    df['production_country_codes'] = [[] for _ in range(len(df))]
                    df['num_production_countries'] = np.nan
                elif field == 'spoken_languages':
                    df['spoken_language_codes'] = [[] for _ in range(len(df))]
                    df['num_spoken_languages'] = np.nan
        
        return df
    
    def clean_categorical_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean categorical fields."""
        logger.info("Cleaning categorical fields")
        
        # Clean adult field
        if 'adult' in df.columns:
            df['adult'] = df['adult'].map({'False': False, 'True': True, False: False, True: True})
            df['adult'] = df['adult'].fillna(False).astype(bool)
        else:
            df['adult'] = False
        # Clean video field
        if 'video' in df.columns:
            df['video'] = df['video'].map({'False': False, 'True': True, False: False, True: True})
            df['video'] = df['video'].fillna(False).astype(bool)
        else:
            df['video'] = False
        # Clean language codes
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
        
        # Create summary statistics
        logger.info(f"Movies metadata cleaning complete. Final dataset: {len(df)} records")
        if 'release_date_valid' in df.columns:
            logger.info(f"Records with valid release dates: {df['release_date_valid'].sum()}")
        if 'budget_valid' in df.columns:
            logger.info(f"Records with valid budgets: {df['budget_valid'].sum()}")
        
        return df


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
        """Clean and validate timestamps."""
        logger.info("Cleaning timestamps")
        
        # Convert to numeric first
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
        
        # Convert to datetime
        df['rating_datetime'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
        
        # Validate reasonable date range
        current_year = datetime.now().year
        df['timestamp_valid'] = (
            df['rating_datetime'].notna() & 
            (df['rating_datetime'].dt.year <= current_year)
        )
        
        # Extract useful date components
        df['rating_year'] = df['rating_datetime'].dt.year
        df['rating_month'] = df['rating_datetime'].dt.month
        df['rating_weekday'] = df['rating_datetime'].dt.dayofweek
        
        return df
    
    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect potential anomalies in rating patterns."""
        logger.info("Detecting rating anomalies")
        
        # Check for users with excessive rating counts (potential bots)
        user_rating_counts = df.groupby('userId')['rating'].count()
        excessive_threshold = user_rating_counts.quantile(0.99)  # Top 1%
        
        df['user_excessive_ratings'] = df['userId'].map(
            user_rating_counts > excessive_threshold
        ).fillna(False)
        
        # Check for movies with very few ratings
        movie_rating_counts = df.groupby('movieId')['rating'].count()
        df['movie_rating_count'] = df['movieId'].map(movie_rating_counts)
        df['movie_few_ratings'] = df['movie_rating_count'] < 5
        
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
        
        # Create data quality summary
        valid_records = (
            df['rating_valid'] & 
            df['userId_valid'] & 
            df['movieId_valid'] & 
            df['timestamp_valid']
        )
        
        logger.info(f"Ratings cleaning complete. Final dataset: {len(df)} records")
        logger.info(f"Completely valid records: {valid_records.sum()} ({valid_records.mean()*100:.1f}%)")
        logger.info(f"Unique users: {df['userId'].nunique()}")
        logger.info(f"Unique movies: {df['movieId'].nunique()}")
        
        return df


# Main execution functions
def clean_all_datasets(data_dir: str = './data') -> Dict[str, pd.DataFrame]:
    """
    Execute all cleaning pipelines and return cleaned datasets.
    
    Args:
        data_dir: Directory containing the CSV files
        
    Returns:
        Dictionary of cleaned DataFrames
    """
    import os
    
    cleaned_data = {}
    
    # Define file paths
    files = {
        'credits': os.path.join(data_dir, 'credits.csv'),
        'keywords': os.path.join(data_dir, 'keywords.csv'),
        'links': os.path.join(data_dir, 'links.csv'),
        'movies_metadata': os.path.join(data_dir, 'movies_metadata.csv'),
        'ratings': os.path.join(data_dir, 'ratings_small.csv')
    }
    
    # Execute cleaning pipelines
    try:
        logger.info("Starting comprehensive data cleaning process")
        
        # cleaned_data['credits'] = CreditsDataCleaner(files['credits']).clean_data()
        # cleaned_data['keywords'] = KeywordsDataCleaner(files['keywords']).clean_data()
        # cleaned_data['links'] = LinksDataCleaner(files['links']).clean_data()
        cleaned_data['movies_metadata'] = MoviesMetadataCleaner(files['movies_metadata']).clean_data()
        # cleaned_data['ratings'] = RatingsDataCleaner(files['ratings']).clean_data()
        
        logger.info("All datasets cleaned successfully!")
        
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
