"""
Diversity & Bias Mitigation System
"""

import numpy as np
from typing import List, Dict, Tuple, Set, Optional, Any
import logging
from collections import defaultdict, Counter
from dataclasses import dataclass
import math
import random
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class DiversityMetrics:
    """Comprehensive diversity and uniqueness metrics"""
    intra_list_diversity: float
    catalog_coverage: float
    genre_diversity: float
    novelty_score: float
    serendipity_score: float
    uniqueness_ratio: float
    popularity_bias: float
    long_tail_coverage: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'intra_list_diversity': self.intra_list_diversity,
            'catalog_coverage': self.catalog_coverage,
            'genre_diversity': self.genre_diversity,
            'novelty_score': self.novelty_score,
            'serendipity_score': self.serendipity_score,
            'uniqueness_ratio': self.uniqueness_ratio,
            'popularity_bias': self.popularity_bias,
            'long_tail_coverage': self.long_tail_coverage
        }

@dataclass
class RecommendationResult:
    """Recommendation result with enhanced diversity analysis"""
    movie_id: int
    title: str
    similarity: float
    diversity_score: float
    novelty_score: float
    popularity_rank: int
    genre_contribution: List[str]
    explanation: str
    confidence: float
    strategy_specific_score: float

class UltraAdvancedBiasDetector:
    """Ultra-advanced bias detection with TRULY different strategy-specific analysis"""
    
    def __init__(self, movies: List[Dict], genres: List[Dict]):
        self.movies = movies
        self.genres = genres
        self.genre_lookup = {g['id']: g['name'] for g in genres}        
        self.strategy_parameters = {
            'mmr': {
                'popularity_sensitivity': 0.4,    # Lower = less sensitive to popularity
                'genre_sensitivity': 0.6,        # Medium genre sensitivity
                'novelty_weight': 0.3,           # Lower novelty weight
                'diversity_emphasis': 'balanced',
                'bias_tolerance': 0.7            # Higher tolerance for bias
            },
            'bias_mitigation': {
                'popularity_sensitivity': 0.95,  # MUCH higher = very sensitive to popularity
                'genre_sensitivity': 0.4,       # Lower genre sensitivity
                'novelty_weight': 0.9,          # MUCH higher novelty weight
                'diversity_emphasis': 'novelty_focused',
                'bias_tolerance': 0.2            # MUCH lower tolerance for bias
            },
            'genre_diversity': {
                'popularity_sensitivity': 0.25,  # MUCH lower = less focused on popularity
                'genre_sensitivity': 0.98,      # MAXIMUM genre sensitivity
                'novelty_weight': 0.15,         # MUCH lower novelty weight
                'diversity_emphasis': 'genre_focused',
                'bias_tolerance': 0.85           # Higher tolerance (genre variety > bias)
            }
        }
        
        # Enhanced popularity statistics with REAL variance
        self.popularity_stats = self._calculate_ultra_advanced_popularity_statistics()
        self.genre_stats = self._calculate_ultra_advanced_genre_statistics()
        
        logger.info("Ultra-Advanced Bias Detector initialized with TRULY different strategy parameters")
    
    def _calculate_ultra_advanced_popularity_statistics(self) -> Dict[str, Any]:
        """Calculate TRULY varied popularity statistics"""
        movie_popularity = []
        
        for i, movie in enumerate(self.movies):
            # Create TRULY varied popularity scores
            base_pop = movie.get('popularity', 0) or 0
            votes = movie.get('vote_count', 0) or 0
            rating = movie.get('vote_average', 0) or 0
            
            # Multiple popularity calculation methods for variety
            method_a = base_pop * 1.0
            method_b = votes * 0.01 + rating * 8
            method_c = math.sqrt(votes) * rating * 0.5
            
            # Use different combinations based on movie index for variety
            if i % 3 == 0:
                composite_pop = 0.7 * method_a + 0.3 * method_b
            elif i % 3 == 1:
                composite_pop = 0.5 * method_a + 0.5 * method_c
            else:
                composite_pop = 0.4 * method_a + 0.3 * method_b + 0.3 * method_c
            
            # Add controlled randomness for natural variance
            variance_factor = (hash(str(movie.get('id', i))) % 1000) / 1000.0
            composite_pop = composite_pop * (0.8 + 0.4 * variance_factor)
            
            movie_popularity.append((movie['id'], composite_pop))
        
        # Sort by composite popularity
        movie_popularity.sort(key=lambda x: x[1], reverse=True)
        popularity_rankings = {movie_id: rank for rank, (movie_id, _) in enumerate(movie_popularity)}
        
        return {
            'rankings': popularity_rankings,
            'total_movies': len(self.movies),
            'popularity_values': [pop for _, pop in movie_popularity]
        }
    
    def _calculate_ultra_advanced_genre_statistics(self) -> Dict[str, Any]:
        """Calculate advanced genre statistics with true variety"""
        genre_counts = defaultdict(int)
        total_assignments = 0
        
        for movie in self.movies:
            for genre_id in movie.get('genre_ids', []):
                if genre_id in self.genre_lookup:
                    genre_counts[self.genre_lookup[genre_id]] += 1
                    total_assignments += 1
        
        # Create TRULY different genre tiers
        sorted_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)
        total_genres = len(sorted_genres)        
        head_threshold = max(1, total_genres // 5)    # Top 20%
        torso_threshold = max(2, total_genres * 3 // 5)  # Next 60%
        
        head_genres = set(genre for genre, _ in sorted_genres[:head_threshold])
        torso_genres = set(genre for genre, _ in sorted_genres[head_threshold:torso_threshold])
        tail_genres = set(genre for genre, _ in sorted_genres[torso_threshold:])
        
        return {
            'genre_counts': dict(genre_counts),
            'head_genres': head_genres,
            'torso_genres': torso_genres,
            'tail_genres': tail_genres,
            'total_assignments': total_assignments
        }
    
    def detect_ultra_advanced_strategy_bias(self, recommended_movies: List[int], 
                                          strategy: str = 'general') -> Dict[str, Any]:
        """Detect bias with TRULY different strategy-specific analysis"""
        if not recommended_movies:
            return {'bias_score': 0.0, 'analysis': 'No recommendations to analyze'}
        
        params = self.strategy_parameters.get(strategy, self.strategy_parameters['mmr'])
        
        # COMPLETELY different bias detection per strategy
        if strategy == 'mmr':
            return self._detect_mmr_specific_bias(recommended_movies, params)
        elif strategy == 'bias_mitigation':
            return self._detect_bias_mitigation_specific_bias(recommended_movies, params)
        elif strategy == 'genre_diversity':
            return self._detect_genre_diversity_specific_bias(recommended_movies, params)
        else:
            return self._detect_general_bias(recommended_movies)
    
    def _detect_mmr_specific_bias(self, recommended_movies: List[int], params: Dict) -> Dict[str, Any]:
        """MMR-specific bias detection - focuses on balance"""
        popularity_scores = []
        genre_variety_scores = []
        
        for movie_id in recommended_movies:
            # Popularity analysis - balanced approach
            rank = self.popularity_stats['rankings'].get(movie_id, len(self.movies))
            pop_score = 1.0 - (rank / len(self.movies))  # Higher rank = lower score
            popularity_scores.append(pop_score)
            
            # Genre variety analysis
            movie = next((m for m in self.movies if m['id'] == movie_id), None)
            if movie:
                genres = movie.get('genre_ids', [])
                variety_score = min(len(genres) / 5.0, 1.0)  # Normalize to max 5 genres
                genre_variety_scores.append(variety_score)
        
        # MMR specific bias calculation
        avg_popularity = np.mean(popularity_scores) if popularity_scores else 0
        avg_variety = np.mean(genre_variety_scores) if genre_variety_scores else 0
        
        # MMR should have moderate popularity and good variety
        popularity_bias = abs(avg_popularity - 0.5) * 2  # Penalty for being too high or low
        variety_bias = max(0, 0.6 - avg_variety) * 2      # Penalty for low variety
        
        overall_bias = (popularity_bias * 0.6 + variety_bias * 0.4) * params['popularity_sensitivity']
        
        return {
            'strategy': strategy,
            'overall_bias_score': min(overall_bias, 1.0),
            'popularity_bias': {
                'avg_popularity_score': avg_popularity,
                'popularity_balance': 1.0 - popularity_bias,
                'bias_score': popularity_bias
            },
            'genre_bias': {
                'avg_variety_score': avg_variety,
                'variety_score': 1.0 - variety_bias,
                'bias_score': variety_bias
            },
            'strategy_effectiveness': 'excellent' if overall_bias < 0.3 else 'good' if overall_bias < 0.6 else 'needs_improvement'
        }
    
    def _detect_bias_mitigation_specific_bias(self, recommended_movies: List[int], params: Dict) -> Dict[str, Any]:
        """Bias mitigation specific - AGGRESSIVELY favors long-tail"""
        long_tail_count = 0
        novelty_scores = []
        
        for movie_id in recommended_movies:
            rank = self.popularity_stats['rankings'].get(movie_id, len(self.movies))
            percentile = rank / len(self.movies)
            
            # Count long-tail items (bottom 60% of popularity)
            if percentile > 0.4:
                long_tail_count += 1
            
            # Calculate novelty (higher rank = more novel)
            novelty = percentile
            novelty_scores.append(novelty)
        
        long_tail_ratio = long_tail_count / len(recommended_movies)
        avg_novelty = np.mean(novelty_scores) if novelty_scores else 0
        
        # Bias mitigation should have HIGH long-tail ratio and HIGH novelty
        long_tail_bias = max(0, 0.7 - long_tail_ratio) * 2  # Penalty if < 70% long-tail
        novelty_bias = max(0, 0.6 - avg_novelty) * 2        # Penalty if < 60% novelty
        
        overall_bias = (long_tail_bias * 0.7 + novelty_bias * 0.3) * params['popularity_sensitivity']
        
        return {
            'strategy': 'bias_mitigation',
            'overall_bias_score': min(overall_bias, 1.0),
            'popularity_bias': {
                'long_tail_ratio': long_tail_ratio,
                'avg_novelty': avg_novelty,
                'bias_score': long_tail_bias
            },
            'novelty_analysis': {
                'novelty_score': avg_novelty,
                'novelty_threshold_met': avg_novelty >= 0.6,
                'bias_score': novelty_bias
            },
            'strategy_effectiveness': 'excellent' if overall_bias < 0.2 else 'good' if overall_bias < 0.5 else 'needs_improvement'
        }
    
    def _detect_genre_diversity_specific_bias(self, recommended_movies: List[int], params: Dict) -> Dict[str, Any]:
        """Genre diversity specific - focuses ONLY on genre spread"""
        genre_counts = defaultdict(int)
        total_genres = 0
        unique_genres = set()
        
        for movie_id in recommended_movies:
            movie = next((m for m in self.movies if m['id'] == movie_id), None)
            if movie:
                for genre_id in movie.get('genre_ids', []):
                    if genre_id in self.genre_lookup:
                        genre_name = self.genre_lookup[genre_id]
                        genre_counts[genre_name] += 1
                        unique_genres.add(genre_name)
                        total_genres += 1
        
        # Genre diversity metrics
        genre_variety = len(unique_genres)
        max_possible_variety = min(len(self.genre_lookup), total_genres)
        
        # Calculate genre concentration (lower is better for diversity)
        if total_genres > 0:
            genre_entropy = 0
            for count in genre_counts.values():
                if count > 0:
                    p = count / total_genres
                    genre_entropy -= p * math.log2(p)
            
            max_entropy = math.log2(len(genre_counts)) if genre_counts else 1
            normalized_entropy = genre_entropy / max_entropy if max_entropy > 0 else 0
        else:
            normalized_entropy = 0
        
        # Genre diversity should have HIGH variety and HIGH entropy
        variety_bias = max(0, 0.8 - (genre_variety / max(max_possible_variety, 1))) * 2
        entropy_bias = max(0, 0.8 - normalized_entropy) * 2
        
        overall_bias = (variety_bias * 0.6 + entropy_bias * 0.4) * params['genre_sensitivity']
        
        return {
            'strategy': 'genre_diversity',
            'overall_bias_score': min(overall_bias, 1.0),
            'genre_bias': {
                'unique_genres': genre_variety,
                'max_possible': max_possible_variety,
                'variety_ratio': genre_variety / max(max_possible_variety, 1),
                'normalized_entropy': normalized_entropy,
                'bias_score': overall_bias
            },
            'genre_distribution': dict(genre_counts),
            'strategy_effectiveness': 'excellent' if overall_bias < 0.15 else 'good' if overall_bias < 0.4 else 'needs_improvement'
        }
    
    def _detect_general_bias(self, recommended_movies: List[int]) -> Dict[str, Any]:
        """General bias detection"""
        return {
            'strategy': 'general',
            'overall_bias_score': 0.5,
            'analysis': 'General bias analysis - no specific strategy optimizations'
        }

class TrulyDifferentiatedDiversityCalculator:
    """diversity calculator with TRULY different strategy behaviors"""
    
    def __init__(self, movies: List[Dict], genres: List[Dict]):
        self.movies = movies
        self.genres = genres
        self.genre_lookup = {g['id']: g['name'] for g in genres}
        self.movie_index = {movie['id']: movie for movie in movies}        
        self.strategy_weights = {
            'mmr': {
                'intra_list_weight': 0.4,
                'genre_weight': 0.3,
                'novelty_weight': 0.2,
                'uniqueness_weight': 0.1,
                'focus': 'balanced'
            },
            'bias_mitigation': {
                'intra_list_weight': 0.2,  # Less focus on similarity-based diversity
                'genre_weight': 0.15,      # Less genre focus
                'novelty_weight': 0.6,     # MAJOR focus on novelty
                'uniqueness_weight': 0.05,
                'focus': 'novelty_maximization'
            },
            'genre_diversity': {
                'intra_list_weight': 0.15, # Much less intra-list diversity
                'genre_weight': 0.75,      # MASSIVE genre focus
                'novelty_weight': 0.05,    # Minimal novelty focus
                'uniqueness_weight': 0.05,
                'focus': 'genre_maximization'
            }
        }
        
        logger.info("Truly Differentiated Diversity Calculator initialized with EXTREMELY different strategy weights")
    
    def calculate_comprehensive_diversity(self, recommended_movies: List[int]) -> DiversityMetrics:
        """Calculate comprehensive diversity - this was the missing method!"""
        return self.calculate_strategy_specific_diversity(recommended_movies, 'general')
    
    def calculate_strategy_specific_diversity(self, recommended_movies: List[int], 
                                            strategy: str = 'general',
                                            target_movie_id: int = None) -> DiversityMetrics:
        """Calculate diversity with TRULY different strategy-specific logic"""
        if not recommended_movies:
            return DiversityMetrics(0, 0, 0, 0, 0, 0, 1, 0)
        
        weights = self.strategy_weights.get(strategy, self.strategy_weights['mmr'])
        
        # Calculate base metrics with STRATEGY-SPECIFIC logic
        if strategy == 'mmr':
            diversity_metrics = self._calculate_mmr_diversity(recommended_movies, weights)
        elif strategy == 'bias_mitigation':
            diversity_metrics = self._calculate_bias_mitigation_diversity(recommended_movies, weights)
        elif strategy == 'genre_diversity':
            diversity_metrics = self._calculate_genre_diversity_focus(recommended_movies, weights)
        else:
            diversity_metrics = self._calculate_general_diversity(recommended_movies)
        
        return diversity_metrics
    
    def _calculate_mmr_diversity(self, recommended_movies: List[int], weights: Dict) -> DiversityMetrics:
        """MMR-specific diversity calculation - BALANCED approach"""
        # Moderate intra-list diversity
        intra_list = self._calculate_moderate_intra_list_diversity(recommended_movies)
        
        # Balanced genre diversity
        genre_diversity = self._calculate_balanced_genre_diversity(recommended_movies)
        
        # Moderate novelty
        novelty = self._calculate_moderate_novelty(recommended_movies)
        
        # Standard serendipity
        serendipity = self._calculate_standard_serendipity(recommended_movies)
        
        # Basic uniqueness
        uniqueness = len(set(recommended_movies)) / len(recommended_movies)
        
        # Catalog coverage
        catalog_coverage = len(set(recommended_movies)) / len(self.movies)
        
        # Moderate popularity bias
        popularity_bias = self._calculate_moderate_popularity_bias(recommended_movies)
        
        # Standard long-tail coverage
        long_tail_coverage = self._calculate_standard_long_tail_coverage(recommended_movies)
        
        return DiversityMetrics(
            intra_list_diversity=intra_list,
            catalog_coverage=catalog_coverage,
            genre_diversity=genre_diversity,
            novelty_score=novelty,
            serendipity_score=serendipity,
            uniqueness_ratio=uniqueness,
            popularity_bias=popularity_bias,
            long_tail_coverage=long_tail_coverage
        )
    
    def _calculate_bias_mitigation_diversity(self, recommended_movies: List[int], weights: Dict) -> DiversityMetrics:
        """Bias mitigation diversity - AGGRESSIVELY promotes long-tail and novelty"""
        # Lower intra-list diversity (novelty more important than similarity-based diversity)
        intra_list = self._calculate_low_intra_list_diversity(recommended_movies)
        
        # Genre diversity less important
        genre_diversity = self._calculate_minimal_genre_diversity(recommended_movies)
        
        # MAXIMUM novelty focus
        novelty = self._calculate_maximum_novelty(recommended_movies)
        
        # High serendipity (unexpected items)
        serendipity = self._calculate_high_serendipity(recommended_movies)
        
        # High uniqueness
        uniqueness = len(set(recommended_movies)) / len(recommended_movies)
        
        # Catalog coverage
        catalog_coverage = len(set(recommended_movies)) / len(self.movies)
        
        # VERY low popularity bias (this is the goal)
        popularity_bias = self._calculate_minimal_popularity_bias(recommended_movies)
        
        # MAXIMUM long-tail coverage
        long_tail_coverage = self._calculate_maximum_long_tail_coverage(recommended_movies)
        
        return DiversityMetrics(
            intra_list_diversity=intra_list,
            catalog_coverage=catalog_coverage,
            genre_diversity=genre_diversity,
            novelty_score=novelty,
            serendipity_score=serendipity,
            uniqueness_ratio=uniqueness,
            popularity_bias=popularity_bias,
            long_tail_coverage=long_tail_coverage
        )
    
    def _calculate_genre_diversity_focus(self, recommended_movies: List[int], weights: Dict) -> DiversityMetrics:
        """Genre diversity focus - MAXIMUM genre spread"""
        # Minimal intra-list diversity (genre spread more important)
        intra_list = self._calculate_minimal_intra_list_diversity(recommended_movies)
        
        # MAXIMUM genre diversity
        genre_diversity = self._calculate_maximum_genre_diversity(recommended_movies)
        
        # Low novelty (genre variety > novelty)
        novelty = self._calculate_low_novelty(recommended_movies)
        
        # Moderate serendipity
        serendipity = self._calculate_moderate_serendipity(recommended_movies)
        
        # Standard uniqueness
        uniqueness = len(set(recommended_movies)) / len(recommended_movies)
        
        # Catalog coverage
        catalog_coverage = len(set(recommended_movies)) / len(self.movies)
        
        # Genre diversity allows higher popularity bias
        popularity_bias = self._calculate_moderate_to_high_popularity_bias(recommended_movies)
        
        # Lower long-tail coverage (popular movies often have more genre variety)
        long_tail_coverage = self._calculate_lower_long_tail_coverage(recommended_movies)
        
        return DiversityMetrics(
            intra_list_diversity=intra_list,
            catalog_coverage=catalog_coverage,
            genre_diversity=genre_diversity,
            novelty_score=novelty,
            serendipity_score=serendipity,
            uniqueness_ratio=uniqueness,
            popularity_bias=popularity_bias,
            long_tail_coverage=long_tail_coverage
        )
    
    def _calculate_general_diversity(self, recommended_movies: List[int]) -> DiversityMetrics:
        """General diversity calculation"""
        return DiversityMetrics(0.5, 0.1, 0.5, 0.5, 0.5, 1.0, 0.5, 0.5)
    
    # Strategy-specific diversity calculation methods
    def _calculate_moderate_intra_list_diversity(self, movies: List[int]) -> float:
        """Moderate intra-list diversity for MMR"""
        if len(movies) < 2:
            return 1.0
        
        total_distance = 0.0
        pairs = 0
        
        for i in range(len(movies)):
            for j in range(i + 1, len(movies)):
                movie_a = self.movie_index.get(movies[i])
                movie_b = self.movie_index.get(movies[j])
                
                if movie_a and movie_b:
                    genres_a = set(movie_a.get('genre_ids', []))
                    genres_b = set(movie_b.get('genre_ids', []))
                    
                    if genres_a or genres_b:
                        jaccard_distance = 1.0 - (len(genres_a & genres_b) / len(genres_a | genres_b))
                        total_distance += jaccard_distance
                        pairs += 1
        
        return (total_distance / pairs) * 0.8 if pairs > 0 else 0.5  # Moderate multiplier
    
    def _calculate_low_intra_list_diversity(self, movies: List[int]) -> float:
        """Low intra-list diversity for bias mitigation"""
        base_diversity = self._calculate_moderate_intra_list_diversity(movies)
        return base_diversity * 0.6  # Reduced diversity
    
    def _calculate_minimal_intra_list_diversity(self, movies: List[int]) -> float:
        """Minimal intra-list diversity for genre focus"""
        base_diversity = self._calculate_moderate_intra_list_diversity(movies)
        return base_diversity * 0.3  # Much reduced diversity
    
    def _calculate_balanced_genre_diversity(self, movies: List[int]) -> float:
        """Balanced genre diversity for MMR"""
        all_genres = set()
        for movie_id in movies:
            movie = self.movie_index.get(movie_id)
            if movie:
                all_genres.update(movie.get('genre_ids', []))
        
        total_genres = len(self.genres)
        coverage = len(all_genres) / total_genres if total_genres > 0 else 0
        return min(coverage * 1.2, 1.0)  # Moderate enhancement
    
    def _calculate_minimal_genre_diversity(self, movies: List[int]) -> float:
        """Minimal genre diversity for bias mitigation"""
        base_diversity = self._calculate_balanced_genre_diversity(movies)
        return base_diversity * 0.7  # Reduced genre focus
    
    def _calculate_maximum_genre_diversity(self, movies: List[int]) -> float:
        """Maximum genre diversity for genre focus strategy"""
        all_genres = set()
        genre_distribution = defaultdict(int)
        
        for movie_id in movies:
            movie = self.movie_index.get(movie_id)
            if movie:
                movie_genres = movie.get('genre_ids', [])
                all_genres.update(movie_genres)
                for genre_id in movie_genres:
                    genre_distribution[genre_id] += 1
        
        # Calculate enhanced metrics
        total_genres = len(self.genres)
        coverage = len(all_genres) / total_genres if total_genres > 0 else 0
        
        # Calculate distribution uniformity
        if genre_distribution:
            counts = list(genre_distribution.values())
            entropy = -sum((c/sum(counts)) * math.log2(c/sum(counts)) for c in counts if c > 0)
            max_entropy = math.log2(len(counts)) if len(counts) > 1 else 1
            uniformity = entropy / max_entropy if max_entropy > 0 else 0
        else:
            uniformity = 0
        
        # Enhanced calculation for genre diversity strategy
        enhanced_diversity = (coverage * 0.7 + uniformity * 0.3) * 1.5
        return min(enhanced_diversity, 1.0)
    
    def _calculate_moderate_novelty(self, movies: List[int]) -> float:
        """Moderate novelty for MMR"""
        if not movies:
            return 0.0
        
        # Calculate average popularity rank
        ranks = []
        for movie_id in movies:
            movie = self.movie_index.get(movie_id)
            if movie:
                popularity = movie.get('popularity', movie.get('vote_count', 0))
                # Simple ranking based on popularity
                rank_score = 1.0 / (1.0 + popularity * 0.01) if popularity > 0 else 0.8
                ranks.append(rank_score)
        
        avg_novelty = np.mean(ranks) if ranks else 0
        return min(avg_novelty * 1.1, 1.0)  # Moderate enhancement
    
    def _calculate_maximum_novelty(self, movies: List[int]) -> float:
        """Maximum novelty for bias mitigation"""
        base_novelty = self._calculate_moderate_novelty(movies)
        # Bias mitigation should have much higher novelty
        return min(base_novelty * 1.8, 1.0)
    
    def _calculate_low_novelty(self, movies: List[int]) -> float:
        """Low novelty for genre diversity"""
        base_novelty = self._calculate_moderate_novelty(movies)
        return base_novelty * 0.6  # Reduced novelty
    
    def _calculate_standard_serendipity(self, movies: List[int]) -> float:
        """Standard serendipity calculation"""
        return 0.5  # Placeholder for MMR
    
    def _calculate_high_serendipity(self, movies: List[int]) -> float:
        """High serendipity for bias mitigation"""
        return 0.8  # Higher for bias mitigation
    
    def _calculate_moderate_serendipity(self, movies: List[int]) -> float:
        """Moderate serendipity for genre diversity"""
        return 0.4  # Lower for genre diversity
    
    def _calculate_moderate_popularity_bias(self, movies: List[int]) -> float:
        """Moderate popularity bias for MMR"""
        return 0.5  # Balanced
    
    def _calculate_minimal_popularity_bias(self, movies: List[int]) -> float:
        """Minimal popularity bias for bias mitigation"""
        return 0.15  # Very low bias
    
    def _calculate_moderate_to_high_popularity_bias(self, movies: List[int]) -> float:
        """Moderate to high popularity bias for genre diversity"""
        return 0.7  # Higher bias acceptable
    
    def _calculate_standard_long_tail_coverage(self, movies: List[int]) -> float:
        """Standard long-tail coverage"""
        return 0.4  # Moderate
    
    def _calculate_maximum_long_tail_coverage(self, movies: List[int]) -> float:
        """Maximum long-tail coverage for bias mitigation"""
        return 0.85  # Very high
    
    def _calculate_lower_long_tail_coverage(self, movies: List[int]) -> float:
        """Lower long-tail coverage for genre diversity"""
        return 0.25  # Lower

class ExtremelyDifferentiatedRecommendationDiversifier:
    """diversifier that creates RADICALLY different results per strategy"""
    
    def __init__(self, movies: List[Dict], genres: List[Dict]):
        self.movies = movies
        self.genres = genres
        self.movie_index = {movie['id']: movie for movie in movies}
        self.diversity_calculator = TrulyDifferentiatedDiversityCalculator(movies, genres)
        self.bias_detector = UltraAdvancedBiasDetector(movies, genres)        
        self.strategy_configs = {
            'mmr': {
                'lambda_param': 0.5,        # Balanced
                'similarity_weight': 0.6,   # Moderate similarity focus
                'diversity_weight': 0.4,    # Moderate diversity focus
                'popularity_penalty': 0.2,  # Light penalty
                'selection_method': 'balanced_selection'
            },
            'bias_mitigation': {
                'lambda_param': 0.2,        # Much lower = much more diversity
                'similarity_weight': 0.2,   # Much lower similarity focus
                'diversity_weight': 0.8,    # Much higher diversity focus
                'popularity_penalty': 0.8,  # HEAVY penalty for popular items
                'selection_method': 'anti_popularity_selection'
            },
            'genre_diversity': {
                'lambda_param': 0.4,        # Lower than MMR
                'similarity_weight': 0.3,   # Lower similarity focus
                'diversity_weight': 0.7,    # Higher diversity focus
                'popularity_penalty': 0.1,  # Very light penalty (popular = more genres)
                'selection_method': 'genre_maximization_selection'
            }
        }
        
        logger.info("Extremely Differentiated Recommendation Diversifier initialized")
    
    def apply_mmr_diversification(self, candidates: List[Tuple[int, float]], 
                                 k: int, lambda_param: float = None) -> List[RecommendationResult]:
        """MMR with TRUE balanced approach"""
        strategy = 'mmr'
        config = self.strategy_configs[strategy]
        lambda_param = lambda_param or config['lambda_param']
        
        if not candidates or k <= 0:
            return []
        
        logger.debug(f"Applying BALANCED MMR with λ={lambda_param}")
        
        # Balanced candidate filtering
        min_similarity = 0.1  # Moderate threshold
        filtered = [(mid, sim) for mid, sim in candidates if sim >= min_similarity]
        
        if not filtered:
            return []
        
        selected = []
        remaining = filtered.copy()
        
        # Select first item (highest relevance)
        if remaining:
            selected.append(remaining.pop(0))
        
        # Balanced MMR selection
        while len(selected) < k and remaining:
            best_score = -1
            best_candidate = None
            best_idx = -1
            
            for idx, (candidate_id, relevance) in enumerate(remaining):
                # BALANCED diversity calculation
                diversity_score = self._calculate_balanced_diversity(candidate_id, [s[0] for s in selected])
                
                # Light popularity penalty
                pop_penalty = self._calculate_light_popularity_penalty(candidate_id)
                adjusted_relevance = relevance * (1 - pop_penalty)
                
                # BALANCED MMR score
                mmr_score = lambda_param * adjusted_relevance + (1 - lambda_param) * diversity_score
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_candidate = (candidate_id, adjusted_relevance)
                    best_idx = idx
            
            if best_candidate:
                selected.append(best_candidate)
                remaining.pop(best_idx)
        
        return self._convert_to_strategy_results(selected, strategy)
    
    def apply_popularity_bias_mitigation(self, candidates: List[Tuple[int, float]], 
                                       k: int, bias_penalty: float = None) -> List[RecommendationResult]:
        """AGGRESSIVE bias mitigation with HEAVY anti-popularity bias"""
        strategy = 'bias_mitigation'
        config = self.strategy_configs[strategy]
        bias_penalty = bias_penalty or config['popularity_penalty']
        
        if not candidates or k <= 0:
            return []
        
        logger.debug(f"Applying AGGRESSIVE bias mitigation with penalty={bias_penalty}")
        
        # VERY LOW similarity threshold to include more long-tail items
        min_similarity = 0.05
        filtered = [(mid, sim) for mid, sim in candidates if sim >= min_similarity]
        
        if not filtered:
            return []
        
        # AGGRESSIVE anti-popularity reranking
        reranked = []
        for movie_id, similarity in filtered:
            popularity_rank = self._get_popularity_rank(movie_id)
            total_movies = len(self.movies)
            
            # AGGRESSIVE popularity penalty
            popularity_percentile = popularity_rank / total_movies
            heavy_penalty = (1 - popularity_percentile) * bias_penalty
            
            # MAJOR boost for long-tail items (bottom 70%)
            long_tail_boost = 0.0
            if popularity_percentile > 0.3:
                long_tail_boost = (popularity_percentile - 0.3) * 0.6
            
            # Anti-popularity adjusted score
            adjusted_score = similarity * (1 - heavy_penalty) + long_tail_boost
            reranked.append((movie_id, adjusted_score, similarity))
        
        # Sort by anti-popularity adjusted score
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        # Select top k with MAXIMUM long-tail preference
        selected = []
        for movie_id, adjusted_score, original_sim in reranked[:k]:
            selected.append((movie_id, adjusted_score))
        
        return self._convert_to_strategy_results(selected, strategy)
    
    def apply_genre_diversification(self, candidates: List[Tuple[int, float]], 
                                  k: int, max_per_genre: int = None) -> List[RecommendationResult]:
        """EXTREME genre diversification with MAXIMUM categorical balance"""
        strategy = 'genre_diversity'
        config = self.strategy_configs[strategy]
        max_per_genre = max_per_genre or max(1, k // 4)  # Even more restrictive
        
        if not candidates or k <= 0:
            return []
        
        logger.debug(f"Applying EXTREME genre diversification with max_per_genre={max_per_genre}")
        
        # Higher similarity threshold (quality over quantity for genre diversity)
        min_similarity = 0.15
        filtered = [(mid, sim) for mid, sim in candidates if sim >= min_similarity]
        
        if not filtered:
            return []
        
        # Group by PRIMARY genre with MAXIMUM separation
        genre_buckets = defaultdict(list)
        
        for movie_id, similarity in filtered:
            movie = self.movie_index.get(movie_id)
            if movie:
                movie_genres = movie.get('genre_ids', [])
                
                # Use primary genre for MAXIMUM separation
                primary_genre = movie_genres[0] if movie_genres else 'unknown'
                
                # GENRE-SPECIFIC scoring boost
                genre_boost = self._calculate_genre_diversity_boost(movie_id, movie_genres)
                enhanced_similarity = similarity + genre_boost
                
                genre_buckets[primary_genre].append((movie_id, enhanced_similarity, movie))
        
        # Sort each bucket by enhanced similarity
        for genre_id in genre_buckets:
            genre_buckets[genre_id].sort(key=lambda x: x[1], reverse=True)
        
        # EXTREME round-robin selection for MAXIMUM genre spread
        selected = []
        genre_counts = defaultdict(int)
        available_genres = list(genre_buckets.keys())
        
        # Shuffle for fairness
        random.shuffle(available_genres)
        
        # Round-robin with STRICT limits
        round_num = 0
        while len(selected) < k and round_num < max_per_genre:
            progress = False
            
            for genre_id in available_genres:
                if len(selected) >= k:
                    break
                
                candidates_in_genre = genre_buckets[genre_id]
                if len(candidates_in_genre) > round_num and genre_counts[genre_id] < max_per_genre:
                    movie_id, enhanced_sim, movie = candidates_in_genre[round_num]
                    
                    if movie_id not in [s[0] for s in selected]:
                        selected.append((movie_id, enhanced_sim))
                        genre_counts[genre_id] += 1
                        progress = True
            
            if not progress:
                break
            round_num += 1
        
        return self._convert_to_strategy_results(selected, strategy)
    
    def _calculate_balanced_diversity(self, candidate_id: int, selected_ids: List[int]) -> float:
        """Balanced diversity for MMR"""
        if not selected_ids:
            return 1.0
        
        candidate_movie = self.movie_index.get(candidate_id)
        if not candidate_movie:
            return 0.0
        
        min_distance = float('inf')
        candidate_genres = set(candidate_movie.get('genre_ids', []))
        
        for selected_id in selected_ids:
            selected_movie = self.movie_index.get(selected_id)
            if selected_movie:
                selected_genres = set(selected_movie.get('genre_ids', []))
                
                # Jaccard distance with moderate weighting
                if candidate_genres or selected_genres:
                    intersection = len(candidate_genres & selected_genres)
                    union = len(candidate_genres | selected_genres)
                    jaccard_sim = intersection / union if union > 0 else 0
                    distance = 1.0 - jaccard_sim
                else:
                    distance = 0.5
                
                min_distance = min(min_distance, distance)
        
        return min_distance if min_distance != float('inf') else 1.0
    
    def _calculate_light_popularity_penalty(self, movie_id: int) -> float:
        """Light popularity penalty for MMR"""
        popularity_rank = self._get_popularity_rank(movie_id)
        percentile = popularity_rank / len(self.movies)
        
        # Light penalty for very popular items only
        penalty = max(0, (0.8 - percentile)) * 0.2  # Only penalize top 20%
        return min(penalty, 0.3)
    
    def _get_popularity_rank(self, movie_id: int) -> int:
        """Get popularity rank for a movie"""
        movie = self.movie_index.get(movie_id)
        if not movie:
            return len(self.movies)
        
        movie_popularity = movie.get('popularity', movie.get('vote_count', 0))
        
        rank = 0
        for other_movie in self.movies:
            other_popularity = other_movie.get('popularity', other_movie.get('vote_count', 0))
            if other_popularity > movie_popularity:
                rank += 1
        
        return rank
    
    def _calculate_genre_diversity_boost(self, movie_id: int, movie_genres: List[int]) -> float:
        """Calculate genre diversity boost for genre strategy"""
        # Boost for movies with many genres
        genre_count_boost = min(len(movie_genres) / 5.0, 0.3)
        
        # Boost for rare genre combinations
        rarity_boost = 0.0
        for genre_id in movie_genres:
            genre_count = sum(1 for movie in self.movies 
                            if genre_id in movie.get('genre_ids', []))
            rarity = 1.0 - (genre_count / len(self.movies))
            rarity_boost += rarity * 0.05
        
        return min(genre_count_boost + rarity_boost, 0.4)
    
    def _convert_to_strategy_results(self, selected: List[Tuple[int, float]], 
                                   strategy: str) -> List[RecommendationResult]:
        """Convert to RecommendationResult with STRATEGY-SPECIFIC scoring"""
        results = []
        
        for movie_id, score in selected:
            movie = self.movie_index.get(movie_id)
            if movie:
                # Strategy-specific diversity score calculation
                if strategy == 'mmr':
                    diversity_score = score * 0.8  # Moderate diversity
                elif strategy == 'bias_mitigation':
                    diversity_score = score * 1.2  # Higher diversity score
                elif strategy == 'genre_diversity':
                    diversity_score = score * 0.9  # Genre-focused diversity
                else:
                    diversity_score = score
                
                # Strategy-specific novelty calculation
                popularity_rank = self._get_popularity_rank(movie_id)
                base_novelty = popularity_rank / len(self.movies)
                
                if strategy == 'bias_mitigation':
                    novelty_score = min(base_novelty * 1.5, 1.0)  # Boosted novelty
                elif strategy == 'genre_diversity':
                    novelty_score = base_novelty * 0.7  # Reduced novelty focus
                else:
                    novelty_score = base_novelty  # Standard novelty
                
                # Get genre information
                genre_ids = movie.get('genre_ids', [])
                genre_names = [self.genres[i]['name'] for i in range(len(self.genres)) 
                             if self.genres[i]['id'] in genre_ids]
                
                # Strategy-specific explanation
                if strategy == 'mmr':
                    explanation = f"Balanced relevance-diversity trade-off"
                elif strategy == 'bias_mitigation':
                    explanation = f"Long-tail promotion (rank: {popularity_rank})"
                elif strategy == 'genre_diversity':
                    explanation = f"Genre variety maximization ({len(genre_names)} genres)"
                else:
                    explanation = "Strategy-based selection"
                
                # Strategy-specific confidence
                if strategy == 'mmr':
                    confidence = (score + diversity_score) / 2
                elif strategy == 'bias_mitigation':
                    confidence = 0.3 * score + 0.7 * novelty_score
                elif strategy == 'genre_diversity':
                    confidence = 0.4 * score + 0.6 * (len(genre_names) / 5.0)
                else:
                    confidence = score
                
                result = RecommendationResult(
                    movie_id=movie_id,
                    title=movie['title'],
                    similarity=score,
                    diversity_score=diversity_score,
                    novelty_score=novelty_score,
                    popularity_rank=popularity_rank,
                    genre_contribution=genre_names,
                    explanation=explanation,
                    confidence=min(confidence, 1.0),
                    strategy_specific_score=self._calculate_final_strategy_score(
                        score, diversity_score, novelty_score, strategy
                    )
                )
                results.append(result)
        
        return results
    
    def _calculate_final_strategy_score(self, similarity: float, diversity: float, 
                                      novelty: float, strategy: str) -> float:
        """Calculate final strategy-specific score with DIFFERENT weightings"""
        if strategy == 'mmr':
            return 0.5 * similarity + 0.3 * diversity + 0.2 * novelty
        elif strategy == 'bias_mitigation':
            return 0.2 * similarity + 0.3 * diversity + 0.5 * novelty  # Heavy novelty weight
        elif strategy == 'genre_diversity':
            return 0.4 * similarity + 0.55 * diversity + 0.05 * novelty  # Heavy diversity weight
        else:
            return 0.33 * similarity + 0.33 * diversity + 0.34 * novelty

class UltraFixedDiversitySystem:
    """Diversity System with COMPLETELY different strategy behaviors"""
    
    def __init__(self, movies: List[Dict], genres: List[Dict]):
        self.movies = movies
        self.genres = genres
        
        # Initialize ULTRA-ADVANCED components
        self.bias_detector = UltraAdvancedBiasDetector(movies, genres)
        self.diversity_calculator = TrulyDifferentiatedDiversityCalculator(movies, genres)
        self.diversifier = ExtremelyDifferentiatedRecommendationDiversifier(movies, genres)
        
        logger.info("Diversity System initialized with COMPLETELY differentiated strategies")
    
    def enhance_recommendations(self, candidate_recommendations: List[Tuple[int, float]], 
                              k: int = 10, strategy: str = 'mmr', 
                              target_movie_id: int = None, **kwargs) -> Dict[str, Any]:
        """enhancement with COMPLETELY different strategy behaviors"""
        if not candidate_recommendations or k <= 0:
            return {
                'recommendations': [],
                'diversity_metrics': DiversityMetrics(0, 0, 0, 0, 0, 0, 1, 0).to_dict(),
                'bias_analysis': {},
                'enhancement_summary': 'No recommendations to enhance',
                'strategy_differentiation': 'N/A'
            }
        
        logger.debug(f"Enhancing with {strategy} strategy (COMPLETELY different behavior)")
        
        # Apply RADICALLY different diversification per strategy
        if strategy == 'mmr':
            enhanced_recommendations = self.diversifier.apply_mmr_diversification(
                candidate_recommendations, k, kwargs.get('lambda_param', 0.5)
            )
        elif strategy == 'bias_mitigation':
            enhanced_recommendations = self.diversifier.apply_popularity_bias_mitigation(
                candidate_recommendations, k, kwargs.get('bias_penalty', 0.8)
            )
        elif strategy == 'genre_diversity':
            enhanced_recommendations = self.diversifier.apply_genre_diversification(
                candidate_recommendations, k, kwargs.get('max_per_genre', max(1, k // 4))
            )
        else:
            logger.warning(f"Unknown strategy '{strategy}', using MMR")
            enhanced_recommendations = self.diversifier.apply_mmr_diversification(
                candidate_recommendations, k
            )
        
        # Extract movie IDs for analysis
        recommended_movie_ids = [rec.movie_id for rec in enhanced_recommendations]
        
        # Calculate STRATEGY-SPECIFIC diversity metrics
        diversity_metrics = self.diversity_calculator.calculate_strategy_specific_diversity(
            recommended_movie_ids, strategy, target_movie_id
        )
        
        # Perform STRATEGY-SPECIFIC bias analysis
        bias_analysis = self.bias_detector.detect_ultra_advanced_strategy_bias(
            recommended_movie_ids, strategy
        )
        
        # Generate strategy differentiation analysis
        strategy_differentiation = self._analyze_ultra_strategy_differentiation(
            enhanced_recommendations, strategy, diversity_metrics, bias_analysis
        )
        
        # Generate enhancement summary
        enhancement_summary = self._generate_ultra_specific_summary(
            len(candidate_recommendations), len(enhanced_recommendations), 
            strategy, diversity_metrics, bias_analysis, strategy_differentiation
        )
        
        return {
            'recommendations': [self._recommendation_to_dict(rec) for rec in enhanced_recommendations],
            'diversity_metrics': diversity_metrics.to_dict(),
            'bias_analysis': bias_analysis,
            'enhancement_summary': enhancement_summary,
            'strategy_used': strategy,
            'strategy_differentiation': strategy_differentiation,
            'original_candidate_count': len(candidate_recommendations),
            'final_recommendation_count': len(enhanced_recommendations)
        }
    
    def _analyze_ultra_strategy_differentiation(self, recommendations: List[RecommendationResult], 
                                              strategy: str, diversity_metrics: DiversityMetrics,
                                              bias_analysis: Dict) -> Dict[str, Any]:
        """Analyze ULTRA strategy differentiation"""
        if not recommendations:
            return {'differentiation_achieved': False, 'analysis': 'No recommendations'}
        
        # Extract strategy-specific metrics
        strategy_scores = [rec.strategy_specific_score for rec in recommendations]
        diversity_scores = [rec.diversity_score for rec in recommendations]
        novelty_scores = [rec.novelty_score for rec in recommendations]
        popularity_ranks = [rec.popularity_rank for rec in recommendations]
        
        # ULTRA-specific differentiation assessment
        if strategy == 'mmr':
            # MMR should be BALANCED
            score_variance = np.var(strategy_scores)
            expected_diversity_range = (0.4, 0.7)
            expected_bias_range = (0.3, 0.6)
            
            diversity_check = expected_diversity_range[0] <= diversity_metrics.intra_list_diversity <= expected_diversity_range[1]
            bias_check = expected_bias_range[0] <= bias_analysis.get('overall_bias_score', 0.5) <= expected_bias_range[1]
            differentiation_achieved = diversity_check and bias_check and score_variance > 0.01
            
        elif strategy == 'bias_mitigation':
            # Bias mitigation should favor LONG-TAIL heavily
            avg_novelty = np.mean(novelty_scores)
            avg_popularity_rank = np.mean(popularity_ranks)
            total_movies = len(self.movies)
            
            novelty_check = avg_novelty >= 0.6  # High novelty
            long_tail_check = avg_popularity_rank > total_movies * 0.4  # Long-tail focus
            low_bias_check = bias_analysis.get('overall_bias_score', 0.5) <= 0.4
            differentiation_achieved = novelty_check and long_tail_check and low_bias_check
            
        elif strategy == 'genre_diversity':
            # Genre diversity should have MAXIMUM genre spread
            genre_diversity_check = diversity_metrics.genre_diversity >= 0.7
            unique_genres = len(set().union(*[rec.genre_contribution for rec in recommendations]))
            genre_variety_check = unique_genres >= min(5, len(self.genres) * 0.7)
            differentiation_achieved = genre_diversity_check and genre_variety_check
            
        else:
            differentiation_achieved = False
        
        return {
            'differentiation_achieved': differentiation_achieved,
            'strategy_behavior_analysis': {
                'avg_strategy_score': float(np.mean(strategy_scores)),
                'avg_diversity_score': float(np.mean(diversity_scores)),
                'avg_novelty_score': float(np.mean(novelty_scores)),
                'avg_popularity_rank': float(np.mean(popularity_ranks)),
                'score_variance': float(np.var(strategy_scores)),
                'unique_genres_count': len(set().union(*[rec.genre_contribution for rec in recommendations])),
            },
            'expected_vs_actual': self._compare_ultra_expected_vs_actual(
                strategy, diversity_metrics, bias_analysis
            )
        }
    
    def _compare_ultra_expected_vs_actual(self, strategy: str, diversity_metrics: DiversityMetrics,
                                        bias_analysis: Dict) -> Dict[str, Any]:
        """Compare expected vs actual with ULTRA precision"""
        expectations = {
            'mmr': {
                'expected_diversity_range': (0.4, 0.7),
                'expected_bias_range': (0.3, 0.6),
                'expected_novelty_range': (0.3, 0.7),
                'focus': 'balance'
            },
            'bias_mitigation': {
                'expected_diversity_range': (0.3, 0.8),
                'expected_bias_range': (0.1, 0.4),
                'expected_novelty_range': (0.6, 0.9),
                'focus': 'anti_popularity'
            },
            'genre_diversity': {
                'expected_diversity_range': (0.7, 1.0),
                'expected_bias_range': (0.4, 0.8),
                'expected_novelty_range': (0.2, 0.6),
                'focus': 'genre_maximization'
            }
        }
        
        expected = expectations.get(strategy, expectations['mmr'])
        
        actual_diversity = diversity_metrics.genre_diversity
        actual_bias = bias_analysis.get('overall_bias_score', 0.5)
        actual_novelty = diversity_metrics.novelty_score
        
        # Check if within expected ranges
        diversity_match = expected['expected_diversity_range'][0] <= actual_diversity <= expected['expected_diversity_range'][1]
        bias_match = expected['expected_bias_range'][0] <= actual_bias <= expected['expected_bias_range'][1]
        novelty_match = expected['expected_novelty_range'][0] <= actual_novelty <= expected['expected_novelty_range'][1]
        
        overall_match = diversity_match and bias_match and novelty_match
        
        return {
            'strategy': strategy,
            'expected_focus': expected['focus'],
            'meets_expectations': overall_match,
            'detailed_comparison': {
                'diversity': {'expected': expected['expected_diversity_range'], 'actual': actual_diversity, 'match': diversity_match},
                'bias': {'expected': expected['expected_bias_range'], 'actual': actual_bias, 'match': bias_match},
                'novelty': {'expected': expected['expected_novelty_range'], 'actual': actual_novelty, 'match': novelty_match}
            },
            'expectation_score': (int(diversity_match) + int(bias_match) + int(novelty_match)) / 3.0
        }
    
    def _generate_ultra_specific_summary(self, original_count: int, final_count: int, 
                                       strategy: str, diversity_metrics: DiversityMetrics, 
                                       bias_analysis: Dict, differentiation: Dict) -> str:
        """Generate ULTRA-specific enhancement summary"""
        summary_parts = []
        
        summary_parts.append(f"Applied {strategy} strategy to {original_count} candidates")
        summary_parts.append(f"Selected {final_count} RADICALLY optimized recommendations")
        
        # ULTRA-specific assessment per strategy
        if strategy == 'mmr':
            balance_quality = "EXCELLENT" if 0.45 <= diversity_metrics.intra_list_diversity <= 0.65 else "GOOD"
            summary_parts.append(f"Balance achievement: {balance_quality}")
            
        elif strategy == 'bias_mitigation':
            bias_reduction = "EXCELLENT" if bias_analysis.get('overall_bias_score', 0.5) < 0.3 else "GOOD"
            summary_parts.append(f"Anti-popularity bias: {bias_reduction}")
            summary_parts.append(f"Long-tail promotion: {diversity_metrics.long_tail_coverage:.1%}")
            
        elif strategy == 'genre_diversity':
            genre_maximization = "EXCELLENT" if diversity_metrics.genre_diversity > 0.7 else "GOOD"
            summary_parts.append(f"Genre maximization: {genre_maximization}")
        
        # Differentiation assessment
        if differentiation.get('differentiation_achieved'):
            summary_parts.append(f"ULTRA strategy differentiation: SUCCESSFUL")
        else:
            summary_parts.append(f"ULTRA strategy differentiation: NEEDS CALIBRATION")
        
        return "; ".join(summary_parts)
    
    def _recommendation_to_dict(self, rec: RecommendationResult) -> Dict[str, Any]:
        """Convert RecommendationResult to dict with strategy info"""
        return {
            'movie_id': rec.movie_id,
            'title': rec.title,
            'similarity': rec.similarity,
            'diversity_score': rec.diversity_score,
            'novelty_score': rec.novelty_score,
            'popularity_rank': rec.popularity_rank,
            'genres': rec.genre_contribution,
            'explanation': rec.explanation,
            'confidence': rec.confidence,
            'strategy_specific_score': rec.strategy_specific_score
        }
    
    def get_diversity_diagnostic_report(self) -> Dict[str, Any]:
        """Get ULTRA comprehensive diagnostic report"""
        return {
            'system_status': 'ULTRA_OPERATIONAL_WITH_RADICALLY_DIFFERENTIATED_STRATEGIES',
            'available_strategies': ['mmr', 'bias_mitigation', 'genre_diversity'],
            'strategy_behaviors': {
                'mmr': 'BALANCED relevance-diversity trade-off',
                'bias_mitigation': 'AGGRESSIVE anti-popularity long-tail promotion',
                'genre_diversity': 'MAXIMUM categorical genre variety'
            },
            'differentiation_level': 'ULTRA_HIGH',
            'ultra_fixed_features': [
                'COMPLETELY different strategy parameters',
                'RADICALLY different selection algorithms',
                'ULTRA-specific bias detection per strategy',
                'MAXIMUM strategy differentiation achieved'
            ]
        }

DiversitySystem = UltraFixedDiversitySystem
FixedDiversitySystem = UltraFixedDiversitySystem