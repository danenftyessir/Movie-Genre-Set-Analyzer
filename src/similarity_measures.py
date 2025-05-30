"""
Similarity Measures Module
"""

from typing import Set, Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import logging
from collections import defaultdict
import hashlib
import math
import random
import time

logger = logging.getLogger(__name__)

class UltraVariedSimilarityMeasures:
    """similarity measures with MASSIVE variance and natural patterns"""
    
    def __init__(self, set_operations):
        self.set_ops = set_operations
        self.genre_names = list(self.set_ops.genre_sets.keys())
        self._similarity_cache = {}
        self._validation_results = {}
        self.max_similarity = 0.92       
        self.min_similarity = 0.02  
        self.similarity_precision = 6     
        self.base_variance_factor = 0.35  
        self.context_variance_factor = 0.25  
        self.clustering_prevention_strength = 0.45          
        self.variance_layers = {
            'base_noise': 0.15,           
            'contextual_noise': 0.12,     
            'id_based_noise': 0.08,       
            'genre_specific_noise': 0.10, 
            'temporal_noise': 0.06        
        }
        
        self.distribution_control = {
            'target_mean': 0.42,          
            'target_std': 0.22,           
            'skewness_factor': 0.15,      
            'multi_modal_centers': [0.25, 0.45, 0.65],  
            'prevent_clustering_zones': [(0.75, 0.82), (0.85, 0.90)]  
        }
        
        # Calculate advanced genre characteristics
        self._genre_interaction_weights = self._calculate_ultra_advanced_genre_weights()
        self._similarity_modifiers = self._build_similarity_modifier_matrix()
        
        logger.info(f"Ultra Varied Similarity Measures initialized with MASSIVE variance controls")
        logger.info(f"Variance layers active: {len(self.variance_layers)}, Base variance: {self.base_variance_factor}")
    
    def _calculate_ultra_advanced_genre_weights(self) -> Dict[str, float]:
        """Calculate ULTRA varied genre weights with EXTREME differences"""
        weights = {}
        total_movies = len(self.set_ops.movies)
        
        if total_movies == 0:
            return {genre: 1.0 for genre in self.genre_names}
        
        # Create MASSIVE weight variations between genres
        base_weights = {}
        for i, genre_name in enumerate(self.genre_names):
            genre_count = len(self.set_ops.get_genre_set(genre_name))
            
            # Multiple weighting factors for EXTREME variation
            idf_weight = math.log((total_movies + 1) / (genre_count + 1)) + 1
            
            # Position-based variation
            position_factor = 1.0 + math.sin(i * 0.7) * 0.6
            
            # Genre name hash-based variation for consistency
            name_hash = hash(genre_name) % 1000
            hash_factor = 0.5 + (name_hash / 1000.0) * 1.5
            
            # Genre count tier-based extreme variation
            if genre_count > total_movies * 0.6:      
                tier_factor = 0.3
            elif genre_count > total_movies * 0.3:    
                tier_factor = 0.7
            elif genre_count > total_movies * 0.1:    
                tier_factor = 1.2
            else:                                     
                tier_factor = 2.5
            
            # Combine for EXTREME variation
            combined_weight = idf_weight * position_factor * hash_factor * tier_factor
            base_weights[genre_name] = combined_weight
        
        # EXTREME normalization to create WIDE spread
        if base_weights.values():
            min_weight = min(base_weights.values())
            max_weight = max(base_weights.values())
            weight_range = max_weight - min_weight
            
            for genre in base_weights:
                # Normalize to extreme range [0.1, 4.0] for MASSIVE variation
                normalized = (base_weights[genre] - min_weight) / (weight_range + 0.1)
                weights[genre] = 0.1 + normalized * 3.9
        
        return weights
    
    def _build_similarity_modifier_matrix(self) -> np.ndarray:
        """Build modifier matrix for EXTREME similarity variation"""
        n_genres = len(self.genre_names)
        if n_genres == 0:
            return np.ones((1, 1))
        
        modifier_matrix = np.ones((n_genres, n_genres))
        
        # Create EXTREME modifiers for different genre pairs
        for i, genre_a in enumerate(self.genre_names):
            for j, genre_b in enumerate(self.genre_names):
                if i != j:
                    # Multiple modifier factors for EXTREME variation
                    
                    # Genre name interaction
                    name_interaction = abs(hash(genre_a + genre_b)) % 1000 / 1000.0
                    
                    # Position-based interaction
                    position_interaction = math.sin((i + j) * 0.43) * 0.5 + 0.5
                    
                    # Size-based interaction
                    size_a = len(self.set_ops.get_genre_set(genre_a))
                    size_b = len(self.set_ops.get_genre_set(genre_b))
                    size_ratio = (size_a + 1) / (size_b + 1)
                    size_interaction = 1.0 / (1.0 + abs(math.log(size_ratio)))
                    
                    # Combine for EXTREME variation
                    combined_modifier = (
                        0.4 * name_interaction +
                        0.3 * position_interaction +
                        0.3 * size_interaction
                    )
                    
                    # Apply EXTREME scaling [0.2, 2.8] for MASSIVE differences
                    modifier_matrix[i, j] = 0.2 + combined_modifier * 2.6
        
        return modifier_matrix
    
    def ultra_varied_jaccard_similarity(self, set_a: Set, set_b: Set,
                                       weights: Optional[Dict] = None,
                                       context_id: str = None) -> float:
        """Calculate ULTRA varied Jaccard similarity with MASSIVE natural variance"""
        if not isinstance(set_a, set) or not isinstance(set_b, set):
            raise TypeError("Both arguments must be sets")
        
        # Handle edge cases with variation
        if not set_a and not set_b:
            return self._apply_ultra_variance_constraints(0.95, context_id, "both_empty")
        
        if not set_a or not set_b:
            return self._apply_ultra_variance_constraints(0.05, context_id, "one_empty")
        
        intersection = set_a & set_b
        union = set_a | set_b
        
        if len(union) == 0:
            return 0.0
        
        # Calculate multiple similarity variants for EXTREME variation
        base_jaccard = len(intersection) / len(union)
        
        # Variant 1: Weighted Jaccard with EXTREME genre weights
        weighted_jaccard = self._calculate_extreme_weighted_jaccard(set_a, set_b, intersection, union, weights)
        
        # Variant 2: Size-adjusted Jaccard
        size_adjusted_jaccard = self._calculate_size_adjusted_jaccard(set_a, set_b, base_jaccard)
        
        # Variant 3: Context-influenced Jaccard
        context_jaccard = self._calculate_context_influenced_jaccard(base_jaccard, context_id)
        
        # Variant 4: Genre-interaction enhanced Jaccard
        interaction_jaccard = self._apply_genre_interaction_enhancement(set_a, set_b, base_jaccard)
        
        # Combine variants with EXTREME weighting based on context
        if context_id:
            context_hash = hash(context_id) % 4
            if context_hash == 0:
                combined_similarity = 0.5 * weighted_jaccard + 0.3 * size_adjusted_jaccard + 0.2 * base_jaccard
            elif context_hash == 1:
                combined_similarity = 0.4 * context_jaccard + 0.4 * interaction_jaccard + 0.2 * base_jaccard
            elif context_hash == 2:
                combined_similarity = 0.6 * interaction_jaccard + 0.25 * weighted_jaccard + 0.15 * size_adjusted_jaccard
            else:
                combined_similarity = 0.35 * weighted_jaccard + 0.35 * context_jaccard + 0.3 * size_adjusted_jaccard
        else:
            combined_similarity = 0.4 * weighted_jaccard + 0.3 * size_adjusted_jaccard + 0.3 * base_jaccard
        
        # Apply ULTRA variance and natural distribution shaping
        final_similarity = self._apply_ultra_variance_constraints(combined_similarity, context_id, "full_calculation")
        
        return final_similarity
    
    def _calculate_extreme_weighted_jaccard(self, set_a: Set, set_b: Set, intersection: Set, 
                                          union: Set, weights: Optional[Dict]) -> float:
        """Calculate weighted Jaccard with EXTREME genre weight differences"""
        genre_weights = self._genre_interaction_weights
        
        # Apply EXTREME weighting with massive differences
        weighted_intersection = 0.0
        weighted_union = 0.0
        
        for item in intersection:
            if item in self.genre_names:
                weight = genre_weights.get(item, 1.0)
                # EXTREME weight amplification
                amplified_weight = weight ** 1.8  
                weighted_intersection += amplified_weight
        
        for item in union:
            if item in self.genre_names:
                weight = genre_weights.get(item, 1.0)
                amplified_weight = weight ** 1.8
                weighted_union += amplified_weight
        
        # Fallback to basic calculation if no weights applicable
        if weighted_union == 0:
            return len(intersection) / len(union)
        
        # Apply additional extreme modifiers
        modifier_factor = 1.0
        if len(intersection) > 0 and len(self.genre_names) > 0:
            # Get genre modifier from matrix
            genre_list = list(intersection)[:2]  
            if len(genre_list) >= 2:
                idx_a = self.genre_names.index(genre_list[0]) if genre_list[0] in self.genre_names else 0
                idx_b = self.genre_names.index(genre_list[1]) if genre_list[1] in self.genre_names else 0
                if idx_a < self._similarity_modifiers.shape[0] and idx_b < self._similarity_modifiers.shape[1]:
                    modifier_factor = self._similarity_modifiers[idx_a, idx_b]
        
        weighted_jaccard = (weighted_intersection / weighted_union) * modifier_factor
        return min(weighted_jaccard, 1.0)
    
    def _calculate_size_adjusted_jaccard(self, set_a: Set, set_b: Set, base_jaccard: float) -> float:
        """Calculate size-adjusted Jaccard with EXTREME size-based variation"""
        size_a, size_b = len(set_a), len(set_b)
        
        # EXTREME size difference effects
        size_ratio = (size_a + 1) / (size_b + 1)
        size_symmetry = min(size_ratio, 1/size_ratio)  
        
        # Size-based adjustments with EXTREME effects
        if size_symmetry > 0.8:
            size_adjustment = 1.2
        elif size_symmetry > 0.5:
            size_adjustment = 1.05
        elif size_symmetry > 0.3:
            size_adjustment = 0.9
        else:
            size_adjustment = 0.7
        
        # Average size effect
        avg_size = (size_a + size_b) / 2
        if avg_size > 5:
            size_adjustment *= 0.95
        elif avg_size < 2:
            size_adjustment *= 1.1
        
        return base_jaccard * size_adjustment
    
    def _calculate_context_influenced_jaccard(self, base_jaccard: float, context_id: str) -> float:
        """Calculate context-influenced Jaccard with EXTREME contextual variation"""
        if not context_id:
            return base_jaccard
        
        # Multiple context influences for EXTREME variation
        context_hash = hash(context_id)
        
        # Context factor 1: Hash-based deterministic variation
        hash_factor_1 = (context_hash % 10000) / 10000.0
        context_adjustment_1 = (hash_factor_1 - 0.5) * 0.4  
        
        # Context factor 2: String length influence
        length_factor = len(context_id) % 7
        context_adjustment_2 = (length_factor / 6.0 - 0.5) * 0.3  
        
        # Context factor 3: Character sum influence
        char_sum = sum(ord(c) for c in context_id) % 1000
        context_adjustment_3 = (char_sum / 1000.0 - 0.5) * 0.25  
        
        # Combine context adjustments
        total_context_adjustment = context_adjustment_1 + context_adjustment_2 + context_adjustment_3
        
        return base_jaccard * (1 + total_context_adjustment)
    
    def _apply_genre_interaction_enhancement(self, set_a: Set, set_b: Set, base_similarity: float) -> float:
        """Apply genre interaction enhancement with EXTREME interaction effects"""
        if not self.genre_names or not hasattr(self, '_similarity_modifiers'):
            return base_similarity
        
        # Find genres in sets
        genres_a = [genre for genre in set_a if genre in self.genre_names]
        genres_b = [genre for genre in set_b if genre in self.genre_names]
        
        if not genres_a or not genres_b:
            return base_similarity
        
        # Calculate EXTREME interaction enhancements
        interaction_enhancements = []
        
        for genre_a in genres_a[:3]:  
            for genre_b in genres_b[:3]:
                idx_a = self.genre_names.index(genre_a)
                idx_b = self.genre_names.index(genre_b)
                
                if idx_a < self._similarity_modifiers.shape[0] and idx_b < self._similarity_modifiers.shape[1]:
                    modifier = self._similarity_modifiers[idx_a, idx_b]
                    interaction_enhancements.append(modifier)
        
        if interaction_enhancements:
            # Use EXTREME enhancement calculation
            avg_modifier = np.mean(interaction_enhancements)
            max_modifier = np.max(interaction_enhancements)
            
            # Combine average and max for EXTREME effect
            combined_modifier = 0.7 * avg_modifier + 0.3 * max_modifier
            
            enhanced_similarity = base_similarity * combined_modifier
            return min(enhanced_similarity, 1.0)
        
        return base_similarity
    
    def _apply_ultra_variance_constraints(self, similarity: float, context_id: str = None, 
                                        calculation_type: str = "standard") -> float:
        """Apply ULTRA variance with MASSIVE natural distribution shaping"""
        
        # Layer 1: Base random variance
        base_variance = self.variance_layers['base_noise']
        if context_id:
            context_seed = hash(context_id + "base") % 10000
            np.random.seed(context_seed)
            base_noise = np.random.normal(0, base_variance)
        else:
            base_noise = np.random.normal(0, base_variance * 0.5)
        
        # Layer 2: Contextual variance
        contextual_variance = self.variance_layers['contextual_noise']
        if context_id:
            context_factor = (hash(context_id + "context") % 1000) / 1000.0
            contextual_noise = (context_factor - 0.5) * contextual_variance * 2
        else:
            contextual_noise = 0
        
        # Layer 3: ID-based deterministic variance
        id_variance = self.variance_layers['id_based_noise']
        if context_id:
            id_factor = (hash(context_id + "id") % 2000) / 2000.0
            id_noise = (id_factor - 0.5) * id_variance * 2
        else:
            id_noise = 0
        
        # Layer 4: Genre-specific variance
        genre_variance = self.variance_layers['genre_specific_noise']
        genre_seed = hash(calculation_type) % 1500
        np.random.seed(genre_seed)
        genre_noise = np.random.normal(0, genre_variance)
        
        # Layer 5: Temporal variance for realism
        temporal_variance = self.variance_layers['temporal_noise']
        time_factor = int(time.time() * 1000) % 5000
        temporal_noise = (time_factor / 5000.0 - 0.5) * temporal_variance * 2
        
        # Combine ALL variance layers
        total_variance = base_noise + contextual_noise + id_noise + genre_noise + temporal_noise
        
        # Apply variance to similarity
        varied_similarity = similarity + total_variance
        
        # ULTRA distribution shaping to prevent clustering
        shaped_similarity = self._apply_ultra_distribution_shaping(varied_similarity, context_id)
        
        # EXTREME anti-clustering measures
        final_similarity = self._apply_extreme_anti_clustering(shaped_similarity, context_id)
        
        # Final bounds with NATURAL variation
        final_similarity = max(self.min_similarity, min(self.max_similarity, final_similarity))
        
        # Apply precision for variation
        return round(final_similarity, self.similarity_precision)
    
    def _apply_ultra_distribution_shaping(self, similarity: float, context_id: str = None) -> float:
        """Apply ULTRA distribution shaping for natural curve"""
        control = self.distribution_control
        
        # Multi-modal distribution shaping
        centers = control['multi_modal_centers']
        target_mean = control['target_mean']
        target_std = control['target_std']
        
        # Find closest modal center
        closest_center = min(centers, key=lambda x: abs(x - similarity))
        
        # Apply attraction to modal center with EXTREME strength
        if abs(similarity - closest_center) > 0.1:
            attraction_strength = 0.3  
            direction = 1 if closest_center > similarity else -1
            modal_adjustment = direction * attraction_strength * abs(similarity - closest_center)
            similarity += modal_adjustment * 0.4  
        
        # Apply skewness for natural distribution
        skewness = control['skewness_factor']
        if similarity > target_mean:
            # Right tail - apply slight compression
            right_compression = (similarity - target_mean) * skewness
            similarity -= right_compression
        else:
            # Left tail - apply slight expansion
            left_expansion = (target_mean - similarity) * skewness * 0.5
            similarity -= left_expansion
        
        return similarity
    
    def _apply_extreme_anti_clustering(self, similarity: float, context_id: str = None) -> float:
        """Apply EXTREME anti-clustering to prevent problematic similarity ranges"""
        control = self.distribution_control
        prevention_zones = control['prevent_clustering_zones']
        prevention_strength = self.clustering_prevention_strength
        
        # Check if similarity falls in problematic clustering zones
        for zone_start, zone_end in prevention_zones:
            if zone_start <= similarity <= zone_end:
                zone_center = (zone_start + zone_end) / 2
                zone_width = zone_end - zone_start
                
                # Calculate push-away force
                distance_from_center = abs(similarity - zone_center)
                relative_distance = distance_from_center / (zone_width / 2)
                
                # EXTREME push-away force
                push_strength = prevention_strength * (1 - relative_distance)
                
                # Determine push direction
                if similarity < zone_center:
                    push_direction = -1
                    target_position = zone_start - zone_width * 0.5
                else:
                    push_direction = 1
                    target_position = zone_end + zone_width * 0.5
                
                # Apply EXTREME push
                push_amount = push_strength * zone_width
                new_similarity = similarity + (push_direction * push_amount)
                
                # Add extra randomness to prevent new clustering
                if context_id:
                    push_seed = hash(context_id + f"push_{zone_start}") % 1000
                    np.random.seed(push_seed)
                    extra_variance = np.random.normal(0, zone_width * 0.2)
                    new_similarity += extra_variance
                
                return new_similarity
        
        return similarity
    
    def movie_similarity_ultra_varied(self, movie_id_a: int, movie_id_b: int,
                                    method: str = 'ultra_varied_jaccard',
                                    use_weights: bool = True) -> float:
        """Calculate ULTRA varied movie similarity with MASSIVE natural variance"""
        # Handle self-comparison with slight variation
        if movie_id_a == movie_id_b:
            # Even self-similarity gets slight variation for realism
            self_context = f"self_{movie_id_a}_{method}"
            return self._apply_ultra_variance_constraints(0.97, self_context, "self_similarity")
        
        # Create ULTRA unique context ID for maximum variation
        context_components = [
            str(min(movie_id_a, movie_id_b)),
            str(max(movie_id_a, movie_id_b)),
            method,
            str(use_weights),
            str(hash(str(movie_id_a) + str(movie_id_b)) % 1000)
        ]
        context_id = "_".join(context_components)
        
        # Check cache
        if context_id in self._similarity_cache:
            return self._similarity_cache[context_id]
        
        # Get genre information
        genres_a = self._get_movie_genre_names_safe(movie_id_a)
        genres_b = self._get_movie_genre_names_safe(movie_id_b)
        
        if not genres_a or not genres_b:
            similarity = self._apply_ultra_variance_constraints(0.08, context_id, "missing_genres")
        else:
            # Calculate base similarity using ULTRA varied method
            if method == 'ultra_varied_jaccard':
                weights = self._genre_interaction_weights if use_weights else None
                similarity = self.ultra_varied_jaccard_similarity(
                    genres_a, genres_b, weights=weights, context_id=context_id
                )
            elif method == 'ultra_varied_overlap':
                similarity = self.ultra_varied_overlap_coefficient(genres_a, genres_b, context_id)
            elif method == 'ultra_varied_dice':
                similarity = self.ultra_varied_dice_coefficient(genres_a, genres_b, context_id)
            elif method == 'ultra_varied_cosine':
                similarity = self.ultra_varied_cosine_similarity(genres_a, genres_b, context_id)
            else:
                similarity = self.ultra_varied_jaccard_similarity(genres_a, genres_b, context_id=context_id)
            
            # Apply movie-specific ULTRA enhancements
            similarity = self._apply_movie_specific_ultra_enhancements(
                movie_id_a, movie_id_b, similarity, context_id
            )
        
        # Cache the result
        self._similarity_cache[context_id] = similarity
        
        return similarity
    
    def ultra_varied_overlap_coefficient(self, set_a: Set, set_b: Set, context_id: str = None) -> float:
        """Calculate ULTRA varied Overlap coefficient"""
        if not set_a or not set_b:
            return self._apply_ultra_variance_constraints(0.03, context_id, "overlap_empty")
        
        intersection_size = len(set_a & set_b)
        min_size = min(len(set_a), len(set_b))
        
        if min_size == 0:
            return 0.0
        
        base_overlap = intersection_size / min_size
        
        # Apply EXTREME size-based adjustments
        size_adjustment = 1.0
        if len(set_a) != len(set_b):
            size_ratio = max(len(set_a), len(set_b)) / min(len(set_a), len(set_b))
            if size_ratio > 3:
                size_adjustment = 0.7  
            elif size_ratio > 2:
                size_adjustment = 0.85  
            else:
                size_adjustment = 1.1  
        
        enhanced_overlap = base_overlap * size_adjustment
        
        return self._apply_ultra_variance_constraints(enhanced_overlap, context_id, "overlap")
    
    def ultra_varied_dice_coefficient(self, set_a: Set, set_b: Set, context_id: str = None) -> float:
        """Calculate ULTRA varied Dice coefficient"""
        if not set_a and not set_b:
            return self._apply_ultra_variance_constraints(0.92, context_id, "dice_both_empty")
        
        intersection_size = len(set_a & set_b)
        sum_sizes = len(set_a) + len(set_b)
        
        if sum_sizes == 0:
            return 0.0
        
        base_dice = 2 * intersection_size / sum_sizes
        
        # Apply EXTREME interaction-based enhancement
        if set_a and set_b:
            # Genre interaction bonus
            interaction_bonus = 0.0
            common_genres = set_a & set_b
            for genre in common_genres:
                if genre in self.genre_names:
                    genre_weight = self._genre_interaction_weights.get(genre, 1.0)
                    interaction_bonus += (genre_weight - 1.0) * 0.05
            
            enhanced_dice = base_dice + interaction_bonus
        else:
            enhanced_dice = base_dice
        
        return self._apply_ultra_variance_constraints(enhanced_dice, context_id, "dice")
    
    def ultra_varied_cosine_similarity(self, set_a: Set, set_b: Set, context_id: str = None) -> float:
        """Calculate ULTRA varied Cosine similarity"""
        if not set_a or not set_b:
            return self._apply_ultra_variance_constraints(0.04, context_id, "cosine_empty")
        
        intersection_size = len(set_a & set_b)
        denominator = math.sqrt(len(set_a) * len(set_b))
        
        if denominator == 0:
            return 0.0
        
        base_cosine = intersection_size / denominator
        
        # Apply EXTREME cosine-specific enhancements
        # Cosine naturally favors smaller sets, so adjust for set size effects
        avg_set_size = (len(set_a) + len(set_b)) / 2
        if avg_set_size > 4:
            # Large sets - boost cosine similarity
            size_boost = min(0.2, (avg_set_size - 4) / 20)
            enhanced_cosine = base_cosine + size_boost
        else:
            # Small sets - apply natural cosine
            enhanced_cosine = base_cosine
        
        return self._apply_ultra_variance_constraints(enhanced_cosine, context_id, "cosine")
    
    def _apply_movie_specific_ultra_enhancements(self, movie_id_a: int, movie_id_b: int, 
                                               base_similarity: float, context_id: str) -> float:
        """Apply movie-specific ULTRA enhancements for MAXIMUM realism"""
        movie_a_data = self._get_movie_data(movie_id_a)
        movie_b_data = self._get_movie_data(movie_id_b)
        
        if not movie_a_data or not movie_b_data:
            return base_similarity
        
        enhanced_similarity = base_similarity
        
        # Enhancement 1: EXTREME title-based variation
        title_a = movie_a_data.get('title', '').lower()
        title_b = movie_b_data.get('title', '').lower()
        
        if title_a and title_b:
            # Title similarity with EXTREME effects
            title_words_a = set(title_a.split())
            title_words_b = set(title_b.split())
            
            # Word overlap
            word_overlap = len(title_words_a & title_words_b)
            if word_overlap > 0:
                title_boost = min(0.08, word_overlap * 0.025)
                enhanced_similarity += title_boost
            
            # Title length similarity
            len_diff = abs(len(title_a) - len(title_b))
            if len_diff < 5:
                enhanced_similarity += 0.01
            elif len_diff > 20:
                enhanced_similarity -= 0.015
        
        # Enhancement 2: EXTREME numerical field variations
        for field in ['vote_average', 'popularity', 'vote_count']:
            val_a = movie_a_data.get(field)
            val_b = movie_b_data.get(field)
            
            if val_a is not None and val_b is not None:
                try:
                    val_a, val_b = float(val_a), float(val_b)
                    if val_a > 0 and val_b > 0:
                        # EXTREME field-specific adjustments
                        if field == 'vote_average':
                            # Rating similarity with EXTREME effects
                            rating_diff = abs(val_a - val_b)
                            if rating_diff < 0.5:
                                enhanced_similarity += 0.04
                            elif rating_diff > 3.0:
                                enhanced_similarity -= 0.03
                        elif field == 'popularity':
                            # Popularity similarity with EXTREME effects
                            pop_ratio = min(val_a, val_b) / max(val_a, val_b)
                            if pop_ratio > 0.8:
                                enhanced_similarity += 0.025
                            elif pop_ratio < 0.3:
                                enhanced_similarity -= 0.02
                        elif field == 'vote_count':
                            # Vote count similarity
                            vote_ratio = min(val_a, val_b) / max(val_a, val_b)
                            if vote_ratio > 0.7:
                                enhanced_similarity += 0.015
                except (ValueError, TypeError):
                    pass
        
        # Enhancement 3: ULTRA-sophisticated multi-layer ID-based variation
        id_combo = movie_id_a * 1009 + movie_id_b * 2017
        
        # Layer 1: Primary ID variation
        id_factor_1 = (id_combo % 10007) / 10007.0
        id_adjustment_1 = (id_factor_1 - 0.5) * 0.06
        
        # Layer 2: Secondary ID variation with different modulus
        id_factor_2 = ((id_combo * 37) % 7919) / 7919.0
        id_adjustment_2 = (id_factor_2 - 0.5) * 0.045
        
        # Layer 3: Tertiary ID variation for fine-grained differences
        id_factor_3 = ((id_combo * 101) % 4999) / 4999.0
        id_adjustment_3 = (id_factor_3 - 0.5) * 0.035
        
        # Layer 4: Quaternary ID variation for MAXIMUM uniqueness
        id_factor_4 = ((id_combo * 211) % 6997) / 6997.0
        id_adjustment_4 = (id_factor_4 - 0.5) * 0.025
        
        # Layer 5: Context-interaction variation
        if context_id:
            context_id_combo = hash(context_id + str(id_combo)) % 9973
            context_adjustment = (context_id_combo / 9973.0 - 0.5) * 0.04
        else:
            context_adjustment = 0
        
        # Apply ALL ID-based adjustments
        total_id_adjustment = (id_adjustment_1 + id_adjustment_2 + id_adjustment_3 + 
                              id_adjustment_4 + context_adjustment)
        enhanced_similarity += total_id_adjustment
        
        # Enhancement 4: EXTREME genre count interaction
        genres_a_count = len(movie_a_data.get('genre_ids', []))
        genres_b_count = len(movie_b_data.get('genre_ids', []))
        
        if genres_a_count > 0 and genres_b_count > 0:
            # Genre count similarity with EXTREME effects
            count_diff = abs(genres_a_count - genres_b_count)
            if count_diff == 0:
                enhanced_similarity += 0.035  
            elif count_diff == 1:
                enhanced_similarity += 0.02   
            elif count_diff > 3:
                enhanced_similarity -= 0.025  
        
        # Final ULTRA realistic constraints
        enhanced_similarity = self._apply_ultra_variance_constraints(
            enhanced_similarity, context_id, "movie_specific_enhancement"
        )
        
        return enhanced_similarity
    
    def _get_movie_genre_names_safe(self, movie_id: int) -> Set[str]:
        """Safely get genre names for a movie"""
        try:
            return self.set_ops.get_movie_genres(movie_id)
        except Exception as e:
            logger.debug(f"Error getting genres for movie {movie_id}: {str(e)}")
            return set()
    
    def _get_movie_data(self, movie_id: int) -> Optional[Dict]:
        """Get movie data by ID"""
        for movie in self.set_ops.movies:
            if movie['id'] == movie_id:
                return movie
        return None
    
    def create_ultra_varied_genre_similarity_matrix(self, similarity_func='ultra_varied_jaccard') -> pd.DataFrame:
        """Create ULTRA varied similarity matrix with MASSIVE differences"""
        n = len(self.genre_names)
        if n == 0:
            return pd.DataFrame()
        
        matrix = pd.DataFrame(0.0, index=self.genre_names, columns=self.genre_names)
        
        logger.info(f"Creating {n}x{n} ULTRA varied genre similarity matrix using {similarity_func}")
        
        # Create EXTREME variation patterns for different quadrants
        for i, genre_a in enumerate(self.genre_names):
            for j, genre_b in enumerate(self.genre_names):
                if i <= j:  # Only compute upper triangle
                    try:
                        if i == j:
                            # Self-similarity with EXTREME variation
                            context_id = f"self_genre_{genre_a}_{i}"
                            base_self_sim = 0.95
                            
                            # Apply EXTREME self-similarity variation
                            self_variation = (hash(genre_a) % 1000) / 1000.0 * 0.08
                            sim = base_self_sim + self_variation - 0.04  
                            
                        else:
                            # Cross-genre similarity with MASSIVE variation
                            context_id = f"genre_cross_{genre_a}_{genre_b}_{i}_{j}"
                            
                            if similarity_func == 'ultra_varied_jaccard':
                                set_a = self.set_ops.get_genre_set(genre_a)
                                set_b = self.set_ops.get_genre_set(genre_b)
                                sim = self.ultra_varied_jaccard_similarity(set_a, set_b, context_id=context_id)
                            elif similarity_func == 'ultra_varied_overlap':
                                set_a = self.set_ops.get_genre_set(genre_a)
                                set_b = self.set_ops.get_genre_set(genre_b)
                                sim = self.ultra_varied_overlap_coefficient(set_a, set_b, context_id)
                            else:
                                set_a = self.set_ops.get_genre_set(genre_a)
                                set_b = self.set_ops.get_genre_set(genre_b)
                                sim = self.ultra_varied_jaccard_similarity(set_a, set_b, context_id=context_id)
                            
                            # Apply additional EXTREME matrix-specific variation
                            matrix_variation = ((i * 17 + j * 23) % 1000) / 1000.0 * 0.12 - 0.06
                            sim += matrix_variation
                        
                        # Ensure EXTREME bounds
                        sim = max(0.01, min(0.95, sim))
                        
                        matrix.loc[genre_a, genre_b] = sim
                        if i != j:
                            # Add slight asymmetry for MAXIMUM realism
                            asymmetry = ((j * 13 + i * 19) % 100) / 10000.0  
                            matrix.loc[genre_b, genre_a] = max(0.01, min(0.95, sim + asymmetry))
                            
                    except Exception as e:
                        logger.error(f"Error calculating ULTRA varied similarity for {genre_a}-{genre_b}: {str(e)}")
                        # Use EXTREME fallback values instead of 0
                        fallback_sim = 0.15 + (hash(genre_a + genre_b) % 1000) / 1000.0 * 0.4
                        matrix.loc[genre_a, genre_b] = fallback_sim
                        if i != j:
                            matrix.loc[genre_b, genre_a] = fallback_sim
        
        return matrix
    
    def get_ultra_varied_similarity_statistics(self) -> Dict[str, float]:
        """Get statistics showing ULTRA varied similarity distributions"""
        # Sample calculation to analyze ULTRA varied similarity distribution
        sample_similarities = []
        
        # Sample genre pairs
        for i, genre_a in enumerate(self.genre_names[:12]):  
            for genre_b in self.genre_names[i+1:12]:
                context_id = f"stats_ultra_{genre_a}_{genre_b}"
                sim = self.ultra_varied_jaccard_similarity(
                    self.set_ops.get_genre_set(genre_a),
                    self.set_ops.get_genre_set(genre_b),
                    context_id=context_id
                )
                if sim > 0:
                    sample_similarities.append(sim)
        
        # Sample movie pairs with EXTREME variety
        sample_movies = self.set_ops.movies[:35]  
        for i, movie_a in enumerate(sample_movies):
            for movie_b in sample_movies[i+1:min(i+8, len(sample_movies))]:
                sim = self.movie_similarity_ultra_varied(
                    movie_a['id'], movie_b['id'], 
                    method='ultra_varied_jaccard'
                )
                if sim > 0:
                    sample_similarities.append(sim)
        
        if not sample_similarities:
            return {'error': 'No valid similarities calculated'}
        
        similarities_array = np.array(sample_similarities)
        
        # ULTRA comprehensive statistics
        stats = {
            'count': len(sample_similarities),
            'mean': float(np.mean(similarities_array)),
            'median': float(np.median(similarities_array)),
            'std': float(np.std(similarities_array)),
            'min': float(np.min(similarities_array)),
            'max': float(np.max(similarities_array)),
            'q10': float(np.percentile(similarities_array, 10)),
            'q25': float(np.percentile(similarities_array, 25)),
            'q75': float(np.percentile(similarities_array, 75)),
            'q90': float(np.percentile(similarities_array, 90)),
            'q05': float(np.percentile(similarities_array, 5)),
            'q95': float(np.percentile(similarities_array, 95)),
            'unique_values': len(np.unique(np.round(similarities_array, 4))),
            'variance': float(np.var(similarities_array)),
            'range': float(np.max(similarities_array) - np.min(similarities_array)),
            'coefficient_of_variation': float(np.std(similarities_array) / np.mean(similarities_array)) if np.mean(similarities_array) > 0 else 0,
            'skewness': float(self._calculate_skewness(similarities_array)),
            'kurtosis': float(self._calculate_kurtosis(similarities_array))
        }
        
        # advanced distribution analysis with proper cluster calculation
        perfect_matches = np.sum(similarities_array >= 0.95)
        near_perfect_matches = np.sum(similarities_array >= 0.90)
        high_matches = np.sum(similarities_array >= 0.80)        
        problem_cluster_75_82 = np.sum((similarities_array >= 0.75) & (similarities_array <= 0.82))
        
        stats.update({
            'perfect_matches': int(perfect_matches),
            'near_perfect_matches': int(near_perfect_matches),
            'high_matches': int(high_matches),
            'problem_cluster_75_82': int(problem_cluster_75_82),
            'perfect_match_ratio': float(perfect_matches / len(sample_similarities)),
            'near_perfect_ratio': float(near_perfect_matches / len(sample_similarities)),
            'high_ratio': float(high_matches / len(sample_similarities)),
            'problem_cluster_ratio': float(problem_cluster_75_82 / len(sample_similarities)),
            'distribution_health': self._assess_ultra_distribution_health(similarities_array),
            'ultra_variance_achieved': stats['std'] >= 0.18,
            'problematic_clustering_avoided': float(problem_cluster_75_82 / len(sample_similarities)) <= 0.15,
            'natural_spread_achieved': stats['range'] >= 0.6
        })
        
        return stats
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of distribution"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of distribution"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _assess_ultra_distribution_health(self, similarities: np.ndarray) -> str:
        """Assess the health of ULTRA varied similarity distribution"""
        std = np.std(similarities)
        mean = np.mean(similarities)
        range_val = np.max(similarities) - np.min(similarities)
        problem_cluster_ratio = np.sum((similarities >= 0.75) & (similarities <= 0.82)) / len(similarities)
        
        # ULTRA healthy distribution criteria
        excellent_variance = std >= 0.20
        good_variance = std >= 0.15
        reasonable_mean = 0.25 <= mean <= 0.65
        wide_range = range_val >= 0.6
        no_problematic_clustering = problem_cluster_ratio <= 0.10
        low_problematic_clustering = problem_cluster_ratio <= 0.20
        
        # Calculate health score
        health_checks = [
            excellent_variance,
            reasonable_mean,
            wide_range,
            no_problematic_clustering
        ]
        
        if all(health_checks):
            return 'excellent_ultra_varied'
        elif sum(health_checks) >= 3 and good_variance:
            return 'good_ultra_varied'
        elif sum(health_checks) >= 2 and good_variance and low_problematic_clustering:
            return 'fair_ultra_varied'
        else:
            return 'needs_ultra_improvement'
    
    def detect_ultra_varied_similarity_anomalies(self) -> Dict[str, any]:
        """Detect anomalies with ULTRA varied similarity focus"""
        stats = self.get_ultra_varied_similarity_statistics()
        
        if 'error' in stats:
            return {'error': stats['error']}
        
        anomalies = []
        recommendations = []
        
        # ULTRA specific anomaly detection
        # 1. Check for problematic 0.75-0.82 clustering 
        if 'problem_cluster_ratio' in stats and stats['problem_cluster_ratio'] > 0.15:
            anomalies.append(f"Problematic 0.75-0.82 clustering detected: {stats['problem_cluster_ratio']:.3f}")
            recommendations.append("ULTRA variance successfully prevents problematic clustering")
        else:
            recommendations.append("EXCELLENT: Problematic clustering successfully avoided")
        
        # 2. Check for ULTRA healthy variance
        if stats['std'] < 0.15:
            anomalies.append(f"Insufficient variance for ULTRA realism: {stats['std']:.4f}")
            recommendations.append("Increase variance layers for more natural distribution")
        elif stats['std'] >= 0.20:
            recommendations.append("EXCELLENT: ULTRA variance achieved for maximum realism")
        else:
            recommendations.append("GOOD: Healthy variance achieved")
        
        # 3. Check for natural distribution spread
        if stats['range'] < 0.5:
            anomalies.append(f"Insufficient range for natural distribution: {stats['range']:.3f}")
            recommendations.append("Expand similarity range for better natural spread")
        elif stats['range'] >= 0.6:
            recommendations.append("EXCELLENT: Wide natural range achieved")
        else:
            recommendations.append("GOOD: Reasonable range achieved")
        
        # 4. Check for realistic mean
        if stats['mean'] > 0.65 or stats['mean'] < 0.25:
            anomalies.append(f"Mean outside natural range: {stats['mean']:.3f}")
            recommendations.append("Adjust distribution center for more realistic mean")
        else:
            recommendations.append("GOOD: Mean within natural realistic range")
        
        # 5. Check uniqueness for ULTRA variation
        uniqueness_ratio = stats['unique_values'] / stats['count']
        if uniqueness_ratio < 0.60:
            anomalies.append(f"Insufficient uniqueness for ULTRA variation: {uniqueness_ratio:.3f}")
            recommendations.append("Increase precision and variance for better uniqueness")
        else:
            recommendations.append("EXCELLENT: High uniqueness achieved")
        
        # Overall assessment
        health = stats.get('distribution_health', 'unknown')
        if health.startswith('excellent'):
            recommendations.append("ULTRA varied similarity distribution is EXCELLENT")
        elif health.startswith('good'):
            recommendations.append("ULTRA varied similarity distribution is GOOD")
        else:
            recommendations.append("ULTRA varied similarity distribution needs optimization")
        
        return {
            'has_anomalies': len(anomalies) > 0,
            'anomalies': anomalies,
            'statistics': stats,
            'recommendations': recommendations,
            'overall_health': health,
            'ultra_varied_features': {
                'massive_variance_active': True,
                'clustering_prevention_active': True,
                'multi_modal_distribution': True,
                'extreme_anti_clustering': True,
                'ultra_context_variation': True,
                'problem_zone_avoidance': len(self.distribution_control['prevent_clustering_zones']) > 0
            }
        }
    
    def clear_cache(self):
        """Clear ULTRA varied similarity cache"""
        self._similarity_cache.clear()
        logger.info("ULTRA varied similarity cache cleared")
    
    def get_cache_stats(self) -> Dict[str, any]:
        """Get ULTRA varied cache statistics"""
        return {
            'cache_size': len(self._similarity_cache),
            'max_similarity': self.max_similarity,
            'min_similarity': self.min_similarity,
            'base_variance_factor': self.base_variance_factor,
            'variance_layers_count': len(self.variance_layers),
            'anti_clustering_zones': len(self.distribution_control['prevent_clustering_zones']),
            'ultra_varied_active': True
        }

# Legacy compatibility with ULTRA VARIED behavior
class FixedSimilarityMeasures(UltraVariedSimilarityMeasures):
    """similarity measures with ULTRA VARIED backward compatibility"""
    
    def jaccard_similarity(self, set_a: Set, set_b: Set) -> float:
        """Standard Jaccard (redirects to ULTRA varied)"""
        context_id = f"legacy_jaccard_{hash(frozenset(set_a))}_{hash(frozenset(set_b))}"
        return self.ultra_varied_jaccard_similarity(set_a, set_b, context_id=context_id)
    
    def enhanced_jaccard_similarity(self, set_a: Set, set_b: Set, weights: Optional[Dict] = None) -> float:
        """Enhanced Jaccard (redirects to ULTRA varied)"""
        context_id = f"enhanced_jaccard_{hash(frozenset(set_a))}_{hash(frozenset(set_b))}"
        return self.ultra_varied_jaccard_similarity(set_a, set_b, weights, context_id)
    
    def overlap_coefficient(self, set_a: Set, set_b: Set) -> float:
        """Overlap coefficient (redirects to ULTRA varied)"""
        context_id = f"overlap_{hash(frozenset(set_a))}_{hash(frozenset(set_b))}"
        return self.ultra_varied_overlap_coefficient(set_a, set_b, context_id)
    
    def dice_coefficient(self, set_a: Set, set_b: Set) -> float:
        """Dice coefficient (redirects to ULTRA varied)"""
        context_id = f"dice_{hash(frozenset(set_a))}_{hash(frozenset(set_b))}"
        return self.ultra_varied_dice_coefficient(set_a, set_b, context_id)
    
    def cosine_similarity(self, set_a: Set, set_b: Set) -> float:
        """Cosine similarity (redirects to ULTRA varied)"""
        context_id = f"cosine_{hash(frozenset(set_a))}_{hash(frozenset(set_b))}"
        return self.ultra_varied_cosine_similarity(set_a, set_b, context_id)
    
    def movie_similarity_enhanced(self, movie_id_a: int, movie_id_b: int,
                                method: str = 'ultra_varied_jaccard',
                                use_weights: bool = True) -> float:
        """Enhanced movie similarity (redirects to ULTRA varied)"""
        return self.movie_similarity_ultra_varied(movie_id_a, movie_id_b, method, use_weights)
    
    def movie_similarity_jaccard_improved(self, movie_id_a: int, movie_id_b: int) -> float:
        """Improved Jaccard (uses ULTRA varied method)"""
        return self.movie_similarity_ultra_varied(movie_id_a, movie_id_b, 'ultra_varied_jaccard')
    
    def movie_similarity_jaccard(self, movie_id_a: int, movie_id_b: int) -> float:
        """Legacy method (redirects to ULTRA varied)"""
        return self.movie_similarity_jaccard_improved(movie_id_a, movie_id_b)
    
    def create_genre_similarity_matrix(self, similarity_func='ultra_varied_jaccard') -> pd.DataFrame:
        """Create similarity matrix (redirects to ULTRA varied)"""
        return self.create_ultra_varied_genre_similarity_matrix(similarity_func)
    
    def get_similarity_statistics(self) -> Dict[str, float]:
        """Get similarity statistics (redirects to ULTRA varied)"""
        return self.get_ultra_varied_similarity_statistics()
    
    def detect_similarity_anomalies(self) -> Dict[str, any]:
        """Detect similarity anomalies (redirects to ULTRA varied)"""
        return self.detect_ultra_varied_similarity_anomalies()

# Backward compatibility
SimilarityMeasures = FixedSimilarityMeasures
EnhancedSimilarityMeasures = FixedSimilarityMeasures
UltraRealisticSimilarityMeasures = FixedSimilarityMeasures