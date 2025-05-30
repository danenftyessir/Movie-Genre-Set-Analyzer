"""
TMDB API Client Module
"""

import requests
import json
import time
from typing import List, Dict, Optional, Union
import os
from datetime import datetime
import logging
from pathlib import Path
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

class TMDBClientException(Exception):
    """Custom exception for TMDB client errors"""
    pass

class TMDBRateLimitException(TMDBClientException):
    """Exception for rate limit errors"""
    pass

class TMDBDataValidationException(TMDBClientException):
    """Exception for data validation errors"""
    pass

class FixedTMDBClient:
    """Fixed TMDB API Client with robust error handling and validation"""
    
    # API Constants
    BASE_URL = "https://api.themoviedb.org/3"
    RATE_LIMIT_DELAY = 0.25  # 4 requests per second
    MAX_RETRIES = 3
    RETRY_BACKOFF_FACTOR = 2
    REQUEST_TIMEOUT = 30
    
    def __init__(self, api_key: str, cache_dir: str = "data"):
        """
        Initialize TMDB client with enhanced configuration
        
        Args:
            api_key: TMDB API key
            cache_dir: Directory for caching data
        """
        if not api_key or not isinstance(api_key, str):
            raise TMDBClientException("Valid API key is required")
        
        self.api_key = api_key.strip()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize session with retry strategy
        self.session = self._create_robust_session()
        
        # Request tracking
        self._request_count = 0
        self._last_request_time = 0
        self._rate_limit_remaining = 40
        
        # Data validation tracking
        self._validation_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'validation_errors': 0,
            'cached_responses': 0
        }
        
        logger.info(f"Initialized FixedTMDBClient with cache dir: {cache_dir}")
        
        # Test API connection
        self._test_api_connection()
    
    def _create_robust_session(self) -> requests.Session:
        """Create session with retry strategy and proper headers"""
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=self.MAX_RETRIES,
            backoff_factor=self.RETRY_BACKOFF_FACTOR,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set headers
        session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'User-Agent': 'MovieGenreAnalyzer/1.0'
        })
        
        return session
    
    def _test_api_connection(self):
        """Test API connection and validate API key"""
        try:
            response = self._make_request_internal("/configuration", timeout=10)
            if response and 'images' in response:
                logger.info("API connection test successful")
            else:
                logger.warning("API connection test returned unexpected response")
        except Exception as e:
            logger.error(f"API connection test failed: {str(e)}")
            raise TMDBClientException(f"Failed to connect to TMDB API: {str(e)}")
    
    def _enforce_rate_limit(self):
        """Enforce rate limiting to respect TMDB API limits"""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        
        if time_since_last < self.RATE_LIMIT_DELAY:
            sleep_time = self.RATE_LIMIT_DELAY - time_since_last
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.3f}s")
            time.sleep(sleep_time)
        
        self._last_request_time = time.time()
        self._request_count += 1
    
    def _make_request_internal(self, endpoint: str, params: Optional[Dict] = None, 
                             timeout: int = None) -> Dict:
        """Internal method for making requests with full error handling"""
        if not endpoint.startswith('/'):
            endpoint = '/' + endpoint
        
        url = f"{self.BASE_URL}{endpoint}"
        timeout = timeout or self.REQUEST_TIMEOUT
        
        # Prepare parameters
        if params is None:
            params = {}
        params['api_key'] = self.api_key
        
        # Enforce rate limiting
        self._enforce_rate_limit()
        
        self._validation_stats['total_requests'] += 1
        
        try:
            logger.debug(f"Making request to: {endpoint}")
            
            response = self.session.get(
                url, 
                params=params, 
                timeout=timeout
            )
            
            # Handle rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 60))
                logger.warning(f"Rate limited. Waiting {retry_after}s")
                time.sleep(retry_after)
                raise TMDBRateLimitException(f"Rate limited, retry after {retry_after}s")
            
            # Check for other HTTP errors
            response.raise_for_status()
            
            # Parse JSON response
            try:
                data = response.json()
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON response from {endpoint}: {str(e)}")
                raise TMDBClientException(f"Invalid JSON response: {str(e)}")
            
            # Update rate limit info from headers
            self._rate_limit_remaining = int(
                response.headers.get('X-RateLimit-Remaining', self._rate_limit_remaining)
            )
            
            self._validation_stats['successful_requests'] += 1
            logger.debug(f"Successful request to {endpoint}")
            
            return data
            
        except requests.exceptions.Timeout:
            logger.error(f"Request timeout for {endpoint}")
            self._validation_stats['failed_requests'] += 1
            raise TMDBClientException(f"Request timeout for {endpoint}")
            
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error for {endpoint}: {str(e)}")
            self._validation_stats['failed_requests'] += 1
            raise TMDBClientException(f"Connection error: {str(e)}")
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error for {endpoint}: {str(e)}")
            self._validation_stats['failed_requests'] += 1
            
            if response.status_code == 401:
                raise TMDBClientException("Invalid API key")
            elif response.status_code == 404:
                raise TMDBClientException(f"Endpoint not found: {endpoint}")
            else:
                raise TMDBClientException(f"HTTP {response.status_code}: {str(e)}")
                
        except Exception as e:
            logger.error(f"Unexpected error for {endpoint}: {str(e)}")
            self._validation_stats['failed_requests'] += 1
            raise TMDBClientException(f"Unexpected error: {str(e)}")
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Public method for making requests with caching"""
        cache_key = self._generate_cache_key(endpoint, params)
        cached_data = self._get_cached_data(cache_key)
        
        if cached_data is not None:
            logger.debug(f"Using cached data for {endpoint}")
            self._validation_stats['cached_responses'] += 1
            return cached_data
        
        # Make fresh request
        data = self._make_request_internal(endpoint, params)
        
        # Cache the response
        self._cache_data(cache_key, data)
        
        return data
    
    def _generate_cache_key(self, endpoint: str, params: Optional[Dict]) -> str:
        """Generate cache key for request"""
        cache_parts = [endpoint.replace('/', '_')]
        if params:
            # Sort params for consistent cache keys
            sorted_params = sorted(params.items())
            cache_parts.extend([f"{k}={v}" for k, v in sorted_params if k != 'api_key'])
        return "_".join(cache_parts)
    
    def _get_cached_data(self, cache_key: str) -> Optional[Dict]:
        """Get cached data if available and not expired"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            # Check if cache is still valid (24 hours)
            file_age = time.time() - cache_file.stat().st_mtime
            if file_age > 24 * 3600:  # 24 hours
                logger.debug(f"Cache expired for {cache_key}")
                cache_file.unlink()  # Delete expired cache
                return None
            
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to read cache {cache_key}: {str(e)}")
            # Delete corrupted cache file
            try:
                cache_file.unlink()
            except:
                pass
            return None
    
    def _cache_data(self, cache_key: str, data: Dict):
        """Cache response data"""
        try:
            cache_file = self.cache_dir / f"{cache_key}.json"
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.debug(f"Cached data for {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to cache data for {cache_key}: {str(e)}")
    
    def get_genres(self) -> List[Dict]:
        """Fetch all movie genres from TMDB with enhanced validation"""
        try:
            logger.info("Fetching movie genres")
            response = self._make_request("/genre/movie/list")
            
            genres = response.get('genres', [])
            
            # Enhanced validation
            validated_genres = self._validate_genres_data(genres)
            
            logger.info(f"Successfully fetched {len(validated_genres)} genres")
            return validated_genres
            
        except Exception as e:
            logger.error(f"Failed to fetch genres: {str(e)}")
            raise TMDBClientException(f"Failed to fetch genres: {str(e)}")
    
    def _validate_genres_data(self, genres: List[Dict]) -> List[Dict]:
        """Enhanced validation for genres data structure"""
        if not isinstance(genres, list):
            raise TMDBDataValidationException("Genres must be a list")
        
        validated_genres = []
        
        for genre in genres:
            if not isinstance(genre, dict):
                logger.warning(f"Invalid genre format: {genre}")
                self._validation_stats['validation_errors'] += 1
                continue
            
            if 'id' not in genre or 'name' not in genre:
                logger.warning(f"Genre missing required fields: {genre}")
                self._validation_stats['validation_errors'] += 1
                continue
            
            if not isinstance(genre['id'], int) or not isinstance(genre['name'], str):
                logger.warning(f"Genre has invalid field types: {genre}")
                self._validation_stats['validation_errors'] += 1
                continue
            
            # Clean up genre name
            clean_genre = {
                'id': genre['id'],
                'name': genre['name'].strip()
            }
            
            validated_genres.append(clean_genre)
        
        return validated_genres
    
    def get_popular_movies(self, page: int = 1) -> Dict:
        """Fetch popular movies from TMDB with enhanced validation"""
        if not isinstance(page, int) or page < 1:
            raise ValueError("Page must be a positive integer")
        
        try:
            logger.info(f"Fetching popular movies page {page}")
            
            params = {
                'page': page,
                'language': 'en-US'
            }
            
            response = self._make_request("/movie/popular", params)
            
            # Enhanced validation
            validated_response = self._validate_movies_response(response, page)
            
            logger.info(f"Successfully fetched page {page} with {len(validated_response.get('results', []))} movies")
            return validated_response
            
        except Exception as e:
            logger.error(f"Failed to fetch popular movies page {page}: {str(e)}")
            raise TMDBClientException(f"Failed to fetch popular movies: {str(e)}")
    
    def _validate_movies_response(self, response: Dict, page: int) -> Dict:
        """Enhanced validation for movies response structure"""
        if not isinstance(response, dict):
            raise TMDBDataValidationException("Response must be a dictionary")
        
        required_fields = ['results', 'total_pages', 'total_results']
        for field in required_fields:
            if field not in response:
                raise TMDBDataValidationException(f"Response missing required field: {field}")
        
        # Validate results array
        results = response.get('results', [])
        if not isinstance(results, list):
            raise TMDBDataValidationException("Results must be a list")
        
        validated_results = []
        
        for movie in results:
            try:
                validated_movie = self._validate_movie_data(movie)
                validated_results.append(validated_movie)
            except TMDBDataValidationException as e:
                logger.warning(f"Invalid movie data on page {page}: {str(e)}")
                self._validation_stats['validation_errors'] += 1
                continue
        
        # Update response with validated results
        response['results'] = validated_results
        return response
    
    def _validate_movie_data(self, movie: Dict) -> Dict:
        """Enhanced validation for individual movie data"""
        if not isinstance(movie, dict):
            raise TMDBDataValidationException("Movie must be a dictionary")
        
        # Required fields
        required_fields = ['id', 'title']
        for field in required_fields:
            if field not in movie:
                raise TMDBDataValidationException(f"Movie missing required field: {field}")
        
        # Validate field types and clean data
        validated_movie = {}
        
        # Validate ID
        if not isinstance(movie['id'], int):
            raise TMDBDataValidationException("Movie ID must be an integer")
        validated_movie['id'] = movie['id']
        
        # Validate title
        title = movie['title']
        if not isinstance(title, str) or not title.strip():
            raise TMDBDataValidationException("Movie title must be a non-empty string")
        validated_movie['title'] = title.strip()
        
        # Validate and clean genre_ids
        genre_ids = movie.get('genre_ids', [])
        if not isinstance(genre_ids, list):
            logger.warning(f"Movie {movie['id']}: genre_ids is not a list, setting to empty")
            validated_movie['genre_ids'] = []
        else:
            # Clean genre IDs - only keep valid integers
            valid_genre_ids = []
            for gid in genre_ids:
                if isinstance(gid, int) and gid > 0:
                    valid_genre_ids.append(gid)
                else:
                    logger.debug(f"Movie {movie['id']}: invalid genre ID {gid}")
            
            validated_movie['genre_ids'] = valid_genre_ids
        
        # Copy other useful fields if present
        optional_fields = ['overview', 'release_date', 'vote_average', 'vote_count', 'popularity']
        for field in optional_fields:
            if field in movie:
                validated_movie[field] = movie[field]
        
        return validated_movie
    
    def get_movies_by_page_range(self, start_page: int = 1, end_page: int = 10) -> List[Dict]:
        """Fetch multiple pages of popular movies with enhanced error handling"""
        if start_page < 1 or end_page < start_page:
            raise ValueError("Invalid page range")
        
        logger.info(f"Fetching movies from page {start_page} to {end_page}")
        
        all_movies = []
        failed_pages = []
        successful_pages = 0
        
        for page in range(start_page, end_page + 1):
            try:
                logger.info(f"Fetching page {page}/{end_page}")
                response = self.get_popular_movies(page)
                
                if 'results' in response and response['results']:
                    page_movies = response['results']
                    all_movies.extend(page_movies)
                    successful_pages += 1
                    logger.info(f"Page {page}: added {len(page_movies)} movies")
                else:
                    logger.warning(f"Page {page}: no results found")
                    failed_pages.append(page)
                
                # Check if we've reached the end
                total_pages = response.get('total_pages', end_page)
                if page >= total_pages:
                    logger.info(f"Reached maximum pages ({total_pages}), stopping at page {page}")
                    break
                    
            except TMDBClientException as e:
                logger.error(f"Failed to fetch page {page}: {str(e)}")
                failed_pages.append(page)
                continue
            
            except Exception as e:
                logger.error(f"Unexpected error fetching page {page}: {str(e)}")
                failed_pages.append(page)
                continue
        
        # Enhanced result reporting
        logger.info(f"Successfully fetched {len(all_movies)} movies from {successful_pages} pages")
        if failed_pages:
            logger.warning(f"Failed to fetch {len(failed_pages)} pages: {failed_pages}")
        
        # Remove duplicates based on movie ID
        unique_movies = []
        seen_ids = set()
        
        for movie in all_movies:
            movie_id = movie.get('id')
            if movie_id not in seen_ids:
                unique_movies.append(movie)
                seen_ids.add(movie_id)
        
        if len(unique_movies) != len(all_movies):
            logger.info(f"Removed {len(all_movies) - len(unique_movies)} duplicate movies")
        
        return unique_movies
    
    def get_movie_details(self, movie_id: int) -> Dict:
        """Fetch detailed information for a specific movie"""
        if not isinstance(movie_id, int) or movie_id <= 0:
            raise ValueError("Movie ID must be a positive integer")
        
        try:
            logger.info(f"Fetching details for movie ID {movie_id}")
            response = self._make_request(f"/movie/{movie_id}")
            
            # Validate movie details
            validated_movie = self._validate_movie_data(response)
            
            logger.info(f"Successfully fetched details for '{validated_movie.get('title', 'Unknown')}'")
            return validated_movie
            
        except Exception as e:
            logger.error(f"Failed to fetch movie details for ID {movie_id}: {str(e)}")
            raise TMDBClientException(f"Failed to fetch movie details: {str(e)}")
    
    def save_data_to_file(self, data: any, filename: str, folder: str = None):
        """Save data to JSON file with enhanced error handling"""
        try:
            if folder:
                save_dir = Path(folder)
            else:
                save_dir = self.cache_dir
            
            save_dir.mkdir(exist_ok=True)
            filepath = save_dir / filename
            
            # Ensure .json extension
            if not filepath.suffix:
                filepath = filepath.with_suffix('.json')
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Data saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save data to {filename}: {str(e)}")
            raise TMDBClientException(f"Failed to save data: {str(e)}")
    
    def load_data_from_file(self, filename: str, folder: str = None) -> any:
        """Load data from JSON file with enhanced error handling"""
        try:
            if folder:
                load_dir = Path(folder)
            else:
                load_dir = self.cache_dir
            
            filepath = load_dir / filename
            
            # Try with .json extension if file not found
            if not filepath.exists() and not filepath.suffix:
                filepath = filepath.with_suffix('.json')
            
            if not filepath.exists():
                logger.warning(f"File not found: {filepath}")
                return None
            
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"Data loaded from {filepath}")
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in file {filename}: {str(e)}")
            raise TMDBClientException(f"Invalid JSON file: {str(e)}")
            
        except Exception as e:
            logger.error(f"Failed to load data from {filename}: {str(e)}")
            raise TMDBClientException(f"Failed to load data: {str(e)}")
    
    def get_client_statistics(self) -> Dict:
        """Get comprehensive client statistics"""
        return {
            'request_statistics': self._validation_stats.copy(),
            'rate_limit_remaining': self._rate_limit_remaining,
            'total_requests_made': self._request_count,
            'cache_directory': str(self.cache_dir),
            'cache_files_count': len(list(self.cache_dir.glob('*.json'))),
            'success_rate': (
                self._validation_stats['successful_requests'] / 
                max(self._validation_stats['total_requests'], 1)
            ) * 100,
            'validation_error_rate': (
                self._validation_stats['validation_errors'] / 
                max(self._validation_stats['successful_requests'], 1)
            ) * 100
        }
    
    def clear_cache(self, older_than_hours: int = 24):
        """Clear cache files older than specified hours"""
        try:
            cutoff_time = time.time() - (older_than_hours * 3600)
            cleared_count = 0
            
            for cache_file in self.cache_dir.glob('*.json'):
                if cache_file.stat().st_mtime < cutoff_time:
                    cache_file.unlink()
                    cleared_count += 1
            
            logger.info(f"Cleared {cleared_count} cache files older than {older_than_hours} hours")
            return cleared_count
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {str(e)}")
            raise TMDBClientException(f"Failed to clear cache: {str(e)}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources"""
        if hasattr(self, 'session'):
            self.session.close()
        
        # Log final statistics
        stats = self.get_client_statistics()
        logger.info(f"TMDB Client session ended. Success rate: {stats['success_rate']:.1f}%")

# Backward compatibility
TMDBClient = FixedTMDBClient
EnhancedTMDBClient = FixedTMDBClient