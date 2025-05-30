"""
Set Operations Module
Implements fundamental set operations with correct complement calculation and validation
"""

from typing import Set, List, Dict, Tuple, Optional, Union
import pandas as pd
import logging
from collections import defaultdict, Counter
import numpy as np
from functools import lru_cache
import time
from pathlib import Path

logger = logging.getLogger(__name__)

class FixedSetOperations:
    """class for performing set theory operations on movie genres"""
    
    def __init__(self, movies_data: List[Dict], genres_data: List[Dict]):
        """
        Initialize with comprehensive validation and optimization
        
        Args:
            movies_data: List of movie dictionaries
            genres_data: List of genre dictionaries
        """
        self.movies = self._validate_movies_data(movies_data)
        self.genres = self._validate_genres_data(genres_data)
        self.genre_sets = self._create_optimized_genre_sets()
        
        # Create universal set of all valid movie IDs
        self._universal_movie_set = self._create_universal_set()
        
        # Performance tracking
        self._operation_stats = {
            'total_operations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'average_operation_time': 0
        }
        
        # Enhanced validation results
        self._validation_summary = self._validate_set_integrity()
        
        logger.info(f"Initialized FixedSetOperations with {len(self.movies)} movies, {len(self.genres)} genres")
        logger.info(f"Universal set size: {len(self._universal_movie_set)} movies")
        
        if self._validation_summary['has_issues']:
            logger.warning(f"Set operations validation found {len(self._validation_summary['issues'])} issues")
        else:
            logger.info("Set operations validation passed successfully")
    
    def _validate_movies_data(self, movies_data: List[Dict]) -> List[Dict]:
        """Enhanced validation and cleaning of movie data"""
        if not isinstance(movies_data, list):
            raise TypeError("Movies data must be a list")
        
        validated_movies = []
        issues = []
        skipped_movies = 0
        duplicate_ids = set()
        
        for i, movie in enumerate(movies_data):
            if not isinstance(movie, dict):
                issues.append(f"Movie at index {i} is not a dictionary")
                skipped_movies += 1
                continue
            
            # Required fields validation
            if 'id' not in movie:
                issues.append(f"Movie at index {i} missing 'id' field")
                skipped_movies += 1
                continue
                
            if not isinstance(movie['id'], int):
                issues.append(f"Movie at index {i} has non-integer id: {movie['id']}")
                skipped_movies += 1
                continue
            
            # Check for duplicates
            movie_id = movie['id']
            if movie_id in duplicate_ids:
                logger.debug(f"Skipping duplicate movie ID: {movie_id}")
                skipped_movies += 1
                continue
            
            duplicate_ids.add(movie_id)
            
            # Clean and validate the movie data
            clean_movie = {
                'id': movie_id,
                'title': str(movie.get('title', f'Movie_{movie_id}')).strip()
            }
            
            # Clean genre_ids field
            genre_ids = movie.get('genre_ids', [])
            if not isinstance(genre_ids, list):
                logger.debug(f"Movie {movie_id}: genre_ids is not a list, setting to empty")
                clean_movie['genre_ids'] = []
            else:
                # Filter out invalid genre IDs
                valid_genre_ids = []
                for gid in genre_ids:
                    if isinstance(gid, int) and gid > 0:
                        valid_genre_ids.append(gid)
                    else:
                        logger.debug(f"Movie {movie_id}: invalid genre ID {gid}")
                
                clean_movie['genre_ids'] = valid_genre_ids
            
            # Copy other useful fields
            for field in ['overview', 'release_date', 'vote_average', 'popularity']:
                if field in movie:
                    clean_movie[field] = movie[field]
            
            validated_movies.append(clean_movie)
        
        if issues:
            logger.warning(f"Movie data validation found {len(issues)} issues, skipped {skipped_movies} movies")
            for issue in issues[:5]:  # Log first 5 issues
                logger.warning(f"  - {issue}")
        
        logger.info(f"Validated {len(validated_movies)}/{len(movies_data)} movies")
        return validated_movies
    
    def _validate_genres_data(self, genres_data: List[Dict]) -> Dict[int, str]:
        """Enhanced validation and conversion of genre data to lookup dictionary"""
        if not isinstance(genres_data, list):
            raise TypeError("Genres data must be a list")
        
        validated_genres = {}
        issues = []
        
        for i, genre in enumerate(genres_data):
            if not isinstance(genre, dict):
                issues.append(f"Genre at index {i} is not a dictionary")
                continue
            
            if 'id' not in genre or 'name' not in genre:
                issues.append(f"Genre at index {i} missing required fields")
                continue
            
            if not isinstance(genre['id'], int) or not isinstance(genre['name'], str):
                issues.append(f"Genre at index {i} has invalid field types")
                continue
            
            # Clean genre name
            genre_name = genre['name'].strip()
            if not genre_name:
                issues.append(f"Genre at index {i} has empty name")
                continue
            
            validated_genres[genre['id']] = genre_name
        
        if issues:
            logger.warning(f"Genre data validation found {len(issues)} issues")
            for issue in issues[:5]:
                logger.warning(f"  - {issue}")
        
        logger.info(f"Validated {len(validated_genres)}/{len(genres_data)} genres")
        return validated_genres
    
    def _create_universal_set(self) -> Set[int]:
        """Create the universal set of all valid movie IDs (FIXED)"""
        universal_set = set()
        
        for movie in self.movies:
            movie_id = movie.get('id')
            if isinstance(movie_id, int) and movie_id > 0:
                universal_set.add(movie_id)
        
        logger.info(f"Created universal set with {len(universal_set)} movie IDs")
        return universal_set
    
    def _create_optimized_genre_sets(self) -> Dict[str, Set[int]]:
        """Create optimized genre sets with enhanced validation"""
        genre_sets = {genre_name: set() for genre_name in self.genres.values()}
        
        # Track statistics
        total_assignments = 0
        invalid_assignments = 0
        movies_without_genres = 0
        
        for movie in self.movies:
            movie_id = movie.get('id')
            if not movie_id:
                continue
                
            genre_ids = movie.get('genre_ids', [])
            
            if not genre_ids:
                movies_without_genres += 1
            
            for genre_id in genre_ids:
                total_assignments += 1
                
                if genre_id in self.genres:
                    genre_name = self.genres[genre_id]
                    genre_sets[genre_name].add(movie_id)
                else:
                    invalid_assignments += 1
                    logger.debug(f"Movie {movie_id}: invalid genre ID {genre_id}")
        
        # Keep all genre sets (even empty ones) for consistency
        logger.info(f"Created genre sets for {len(genre_sets)} genres")
        
        if invalid_assignments > 0:
            logger.warning(f"Found {invalid_assignments}/{total_assignments} invalid genre assignments")
        
        if movies_without_genres > 0:
            logger.info(f"Found {movies_without_genres} movies without genre assignments")
        
        # Log genre set sizes
        non_empty_sets = {name: len(gset) for name, gset in genre_sets.items() if gset}
        logger.info(f"Non-empty genre sets: {len(non_empty_sets)}")
        
        return genre_sets
    
    def _validate_set_integrity(self) -> Dict:
        """Enhanced validation of mathematical properties of genre sets"""
        issues = []
        warnings = []
        
        # Validate universal set consistency
        all_movie_ids_in_genres = set()
        for genre_set in self.genre_sets.values():
            all_movie_ids_in_genres.update(genre_set)
        
        # Movies in universal set but not in any genre
        unassigned_movies = self._universal_movie_set - all_movie_ids_in_genres
        if unassigned_movies:
            logger.info(f"Found {len(unassigned_movies)} movies not assigned to any genre")
        
        # Check for basic set properties
        genre_names = list(self.genre_sets.keys())
        
        if len(genre_names) < 2:
            issues.append("Need at least 2 genres for set operations")
            return {'has_issues': True, 'issues': issues, 'warnings': warnings}
        
        # Test fundamental set properties
        test_pairs = [(genre_names[i], genre_names[i+1]) for i in range(min(3, len(genre_names)-1))]
        
        for genre_a, genre_b in test_pairs:
            set_a = self.genre_sets[genre_a]
            set_b = self.genre_sets[genre_b]
            
            # Test commutative property: A ∪ B = B ∪ A
            union_ab = len(set_a | set_b)
            union_ba = len(set_b | set_a)
            if union_ab != union_ba:
                issues.append(f"Union not commutative for {genre_a}, {genre_b}")
            
            # Test intersection bounds
            inter_ab = len(set_a & set_b)
            if inter_ab > min(len(set_a), len(set_b)):
                issues.append(f"Intersection larger than constituent sets: {genre_a}, {genre_b}")
            
            # Test union bounds
            if union_ab > len(set_a) + len(set_b):
                issues.append(f"Union violates size bounds: {genre_a}, {genre_b}")
        
        for genre_name in genre_names[:3]:  # Test first 3 genres
            genre_set = self.genre_sets[genre_name]
            complement_set = self._universal_movie_set - genre_set
            
            # Verify |A| + |A'| = |U|
            if len(genre_set) + len(complement_set) != len(self._universal_movie_set):
                issues.append(f"Complement operation error for {genre_name}: "
                           f"|{genre_name}| + |{genre_name}'| = {len(genre_set)} + {len(complement_set)} "
                           f"!= |U| = {len(self._universal_movie_set)}")
            
            # Verify A ∩ A' = ∅
            intersection_with_complement = len(genre_set & complement_set)
            if intersection_with_complement > 0:
                issues.append(f"Genre {genre_name} intersects with its complement")
        
        # Check for suspicious overlaps
        high_overlap_pairs = []
        for i, genre_a in enumerate(genre_names):
            for genre_b in genre_names[i+1:]:
                set_a = self.genre_sets[genre_a]
                set_b = self.genre_sets[genre_b]
                
                if set_a and set_b:
                    overlap = len(set_a & set_b) / len(set_a | set_b)
                    if overlap > 0.8:  # Very high overlap
                        high_overlap_pairs.append((genre_a, genre_b, overlap))
        
        if high_overlap_pairs and len(high_overlap_pairs) > len(genre_names) * 0.1:
            warnings.append(f"Found {len(high_overlap_pairs)} genre pairs with >80% overlap")
        
        return {
            'has_issues': len(issues) > 0,
            'issues': issues,
            'warnings': warnings,
            'high_overlap_pairs': high_overlap_pairs,
            'universal_set_size': len(self._universal_movie_set),
            'unassigned_movies': len(unassigned_movies),
            'total_movies_in_genres': len(all_movie_ids_in_genres),
            'genre_set_sizes': {name: len(gset) for name, gset in self.genre_sets.items()}
        }
    
    @lru_cache(maxsize=128)
    def get_genre_set(self, genre_name: str) -> Set[int]:
        """Get the set of movie IDs for a specific genre (cached)"""
        if genre_name in self.genre_sets:
            return self.genre_sets[genre_name].copy()
        else:
            logger.warning(f"Genre '{genre_name}' not found")
            return set()
    
    def _track_operation(self, operation_name: str, start_time: float):
        """Track operation performance"""
        elapsed = time.time() - start_time
        self._operation_stats['total_operations'] += 1
        
        # Update running average
        current_avg = self._operation_stats['average_operation_time']
        total_ops = self._operation_stats['total_operations']
        self._operation_stats['average_operation_time'] = (
            (current_avg * (total_ops - 1) + elapsed) / total_ops
        )
    
    def union(self, genre_a: str, genre_b: str) -> Set[int]:
        """
        Union operation with validation
        A ∪ B = {x | x ∈ A or x ∈ B}
        """
        start_time = time.time()
        
        if genre_a not in self.genre_sets or genre_b not in self.genre_sets:
            logger.warning(f"Invalid genre names in union: {genre_a}, {genre_b}")
            return set()
        
        result = self.genre_sets[genre_a] | self.genre_sets[genre_b]
        self._track_operation('union', start_time)
        
        logger.debug(f"Union({genre_a}, {genre_b}): {len(result)} movies")
        return result
    
    def intersection(self, genre_a: str, genre_b: str) -> Set[int]:
        """
        Intersection operation with validation
        A ∩ B = {x | x ∈ A and x ∈ B}
        """
        start_time = time.time()
        
        if genre_a not in self.genre_sets or genre_b not in self.genre_sets:
            logger.warning(f"Invalid genre names in intersection: {genre_a}, {genre_b}")
            return set()
        
        result = self.genre_sets[genre_a] & self.genre_sets[genre_b]
        self._track_operation('intersection', start_time)
        
        logger.debug(f"Intersection({genre_a}, {genre_b}): {len(result)} movies")
        return result
    
    def difference(self, genre_a: str, genre_b: str) -> Set[int]:
        """
        Difference operation with validation
        A \ B = {x | x ∈ A and x ∉ B}
        """
        start_time = time.time()
        
        if genre_a not in self.genre_sets or genre_b not in self.genre_sets:
            logger.warning(f"Invalid genre names in difference: {genre_a}, {genre_b}")
            return set()
        
        result = self.genre_sets[genre_a] - self.genre_sets[genre_b]
        self._track_operation('difference', start_time)
        
        logger.debug(f"Difference({genre_a}, {genre_b}): {len(result)} movies")
        return result
    
    def symmetric_difference(self, genre_a: str, genre_b: str) -> Set[int]:
        """
        Symmetric difference operation
        A △ B = (A \ B) ∪ (B \ A)
        """
        start_time = time.time()
        
        if genre_a not in self.genre_sets or genre_b not in self.genre_sets:
            logger.warning(f"Invalid genre names in symmetric difference: {genre_a}, {genre_b}")
            return set()
        
        result = self.genre_sets[genre_a] ^ self.genre_sets[genre_b]
        self._track_operation('symmetric_difference', start_time)
        
        logger.debug(f"Symmetric Difference({genre_a}, {genre_b}): {len(result)} movies")
        return result
    
    def complement(self, genre: str) -> Set[int]:
        """
        complement operation with validation
        A' = U \ A where U is the universal set of all valid movie IDs
        """
        start_time = time.time()
        
        if genre not in self.genre_sets:
            logger.warning(f"Invalid genre name in complement: {genre}")
            return self._universal_movie_set.copy()
        
        # Use the pre-computed universal set for correct complement
        genre_set = self.genre_sets[genre]
        result = self._universal_movie_set - genre_set
        self._track_operation('complement', start_time)
        
        # Enhanced logging for complement operation
        logger.debug(f"Complement({genre}):")
        logger.debug(f"  Genre set size: {len(genre_set)}")
        logger.debug(f"  Universal set size: {len(self._universal_movie_set)}")
        logger.debug(f"  Complement size: {len(result)}")
        logger.debug(f"  Verification: {len(genre_set)} + {len(result)} = {len(genre_set) + len(result)} (should equal {len(self._universal_movie_set)})")
        
        # Verify the complement calculation
        expected_size = len(self._universal_movie_set) - len(genre_set)
        if len(result) != expected_size:
            logger.error(f"COMPLEMENT CALCULATION ERROR for {genre}:")
            logger.error(f"  Expected size: {expected_size}")
            logger.error(f"  Actual size: {len(result)}")
            logger.error(f"  Genre set: {len(genre_set)} movies")
            logger.error(f"  Universal set: {len(self._universal_movie_set)} movies")
        
        return result
    
    def multi_genre_intersection(self, genres: List[str]) -> Set[int]:
        """Multi-genre intersection with enhanced validation"""
        start_time = time.time()
        
        if not genres:
            return set()
        
        # Validate all genres exist
        valid_genres = [g for g in genres if g in self.genre_sets]
        if len(valid_genres) != len(genres):
            invalid = set(genres) - set(valid_genres)
            logger.warning(f"Invalid genres in multi-intersection: {invalid}")
        
        if not valid_genres:
            return set()
        
        # Start with first genre set
        result = self.genre_sets[valid_genres[0]].copy()
        
        # Intersect with remaining genres
        for genre in valid_genres[1:]:
            result &= self.genre_sets[genre]
            
            # Early termination if result becomes empty
            if not result:
                break
        
        self._track_operation('multi_intersection', start_time)
        logger.debug(f"Multi-intersection({valid_genres}): {len(result)} movies")
        
        return result
    
    def multi_genre_union(self, genres: List[str]) -> Set[int]:
        """Multi-genre union with validation"""
        start_time = time.time()
        
        if not genres:
            return set()
        
        valid_genres = [g for g in genres if g in self.genre_sets]
        if len(valid_genres) != len(genres):
            invalid = set(genres) - set(valid_genres)
            logger.warning(f"Invalid genres in multi-union: {invalid}")
        
        result = set()
        for genre in valid_genres:
            result |= self.genre_sets[genre]
        
        self._track_operation('multi_union', start_time)
        logger.debug(f"Multi-union({valid_genres}): {len(result)} movies")
        
        return result
    
    def exclusive_genre_movies(self, genre: str) -> Set[int]:
        """Find movies that belong exclusively to one genre"""
        start_time = time.time()
        
        if genre not in self.genre_sets:
            logger.warning(f"Invalid genre name: {genre}")
            return set()
        
        genre_movies = self.genre_sets[genre]
        exclusive_movies = set()
        
        for movie_id in genre_movies:
            movie_genres = self.get_movie_genres(movie_id)
            if len(movie_genres) == 1 and genre in movie_genres:
                exclusive_movies.add(movie_id)
        
        self._track_operation('exclusive_genre', start_time)
        logger.debug(f"Exclusive {genre} movies: {len(exclusive_movies)}")
        
        return exclusive_movies
    
    def get_movie_genres(self, movie_id: int) -> Set[str]:
        """Get genre names for a specific movie with enhanced validation"""
        movie_genres = set()
        
        for movie in self.movies:
            if movie.get('id') == movie_id:
                genre_ids = movie.get('genre_ids', [])
                movie_genres = {
                    self.genres[gid] for gid in genre_ids 
                    if gid in self.genres
                }
                break
        
        return movie_genres
    
    def genre_co_occurrence_matrix(self) -> pd.DataFrame:
        """Create co-occurrence matrix with enhanced validation"""
        start_time = time.time()
        
        genre_names = list(self.genre_sets.keys())
        n = len(genre_names)
        
        if n == 0:
            logger.warning("No genres available for co-occurrence matrix")
            return pd.DataFrame()
        
        matrix = pd.DataFrame(0, index=genre_names, columns=genre_names)
        
        # Calculate co-occurrences efficiently
        for i, genre_a in enumerate(genre_names):
            for j, genre_b in enumerate(genre_names):
                if i <= j:  # Only calculate upper triangle
                    if i == j:
                        # Diagonal: genre with itself
                        co_occurrence = len(self.genre_sets[genre_a])
                    else:
                        # Off-diagonal: intersection
                        co_occurrence = len(self.genre_sets[genre_a] & self.genre_sets[genre_b])
                    
                    matrix.iloc[i, j] = co_occurrence
                    matrix.iloc[j, i] = co_occurrence  # Symmetric matrix
        
        self._track_operation('co_occurrence_matrix', start_time)
        logger.info(f"Generated {n}x{n} co-occurrence matrix")
        
        return matrix
    
    def get_operation_statistics(self) -> Dict:
        """Get comprehensive operation statistics"""
        stats = self._operation_stats.copy()
        
        # Add validation statistics
        stats.update({
            'total_movies': len(self.movies),
            'total_genres': len(self.genres),
            'active_genre_sets': len(self.genre_sets),
            'universal_set_size': len(self._universal_movie_set),
            'validation_summary': self._validation_summary,
            'cache_hit_rate': (
                stats['cache_hits'] / max(stats['cache_hits'] + stats['cache_misses'], 1)
            ) * 100
        })
        
        # Add detailed genre statistics
        genre_stats = {}
        for genre_name, genre_set in self.genre_sets.items():
            genre_stats[genre_name] = {
                'movie_count': len(genre_set),
                'percentage_of_total': (len(genre_set) / len(self._universal_movie_set)) * 100 if self._universal_movie_set else 0
            }
        stats['genre_statistics'] = genre_stats
        
        return stats
    
    def validate_operation_result(self, operation: str, result: Set[int], 
                                expected_properties: Dict = None) -> bool:
        """Enhanced validation of set operation results"""
        if not isinstance(result, set):
            logger.error(f"Operation {operation} returned non-set result")
            return False
        
        # Check if all movie IDs in result are valid
        invalid_ids = result - self._universal_movie_set
        
        if invalid_ids:
            logger.error(f"Operation {operation} returned invalid movie IDs: {list(invalid_ids)[:5]}...")
            return False
        
        # Check expected properties if provided
        if expected_properties:
            if 'max_size' in expected_properties:
                if len(result) > expected_properties['max_size']:
                    logger.error(f"Operation {operation} result exceeds maximum expected size")
                    return False
            
            if 'min_size' in expected_properties:
                if len(result) < expected_properties['min_size']:
                    logger.warning(f"Operation {operation} result below minimum expected size")
        
        return True
    
    def clear_cache(self):
        """Clear operation cache"""
        self.get_genre_set.cache_clear()
        logger.info("Operation cache cleared")
    
    def save_validation_report(self, filepath: str = None):
        """Save validation report to file"""
        if filepath is None:
            filepath = f"validation_reports/set_operations_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            # Ensure directory exists
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            report = {
                'validation_summary': self._validation_summary,
                'operation_statistics': self.get_operation_statistics(),
                'timestamp': time.time()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Set operations validation report saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save validation report: {str(e)}")
    
    def __repr__(self) -> str:
        return (f"FixedSetOperations(movies={len(self.movies)}, "
                f"genres={len(self.genre_sets)}, "
                f"universal_set={len(self._universal_movie_set)}, "
                f"operations={self._operation_stats['total_operations']})")

# Backward compatibility
SetOperations = FixedSetOperations
EnhancedSetOperations = FixedSetOperations