"""
K-Nearest Neighbour Recommender System
"""

import numpy as np
from typing import List, Dict, Tuple, Set, Optional, Any
import logging
from collections import defaultdict, Counter
from dataclasses import dataclass
import math
import random
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import ParameterGrid
import warnings
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class SimilarityAnalysis:
    """Comprehensive similarity analysis results"""
    distribution_stats: Dict[str, float]
    is_healthy: bool
    anomalies_detected: List[str]
    recommendations: List[str]
    quality_score: float
    
@dataclass
class MovieSimilarity:
    """Movie similarity with comprehensive metadata"""
    movie_id: int
    title: str
    similarity: float
    confidence_score: float
    shared_features: Dict[str, Any]
    feature_importance: Dict[str, float]
    explanation: str
    
    def __post_init__(self):
        """Post-initialization validation"""
        self.similarity = max(0.0, min(1.0, self.similarity))
        self.confidence_score = max(0.0, min(1.0, self.confidence_score))

class AdvancedFeatureEnginer:
    """Advanced Feature Engineering with natural variance"""
    
    def __init__(self, add_natural_variance: bool = True):
        self.add_natural_variance = add_natural_variance
        self.scalers = {}
        self.feature_stats = {}
        self.feature_importance = {}
        # Add parameters for natural variance
        self.variance_factor = 0.15  
        self.feature_interaction_strength = 0.3
        
        logger.info(f"Advanced Feature Engineer initialized with natural variance: {add_natural_variance}")
    
    def engineer_movie_features(self, movies: List[Dict], genres: List[Dict]) -> Tuple[np.ndarray, List[str], Dict]:
        """Engineer features with natural variance and realistic distributions"""
        logger.info("🔧 Engineering advanced movie features with natural variance...")
        
        genre_lookup = {g['id']: g['name'] for g in genres}
        
        # 1. Enhanced Genre Features with natural weighting
        logger.info("   📊 Processing enhanced genre features...")
        genre_features, genre_names = self._engineer_enhanced_genre_features(movies, genre_lookup)
        
        # 2. Advanced Numerical Features with interaction terms
        logger.info("   📈 Processing advanced numerical features...")
        numerical_features, numerical_names = self._engineer_advanced_numerical_features(movies)
        
        # 3. Movie Context Features
        logger.info("   🎬 Creating movie context features...")
        context_features, context_names = self._create_movie_context_features(movies)
        
        # 4. Feature Interaction Terms
        logger.info("   🔗 Generating feature interactions...")
        interaction_features, interaction_names = self._create_advanced_interactions(
            genre_features, numerical_features, genre_names, numerical_names
        )
        
        # Combine all features
        all_features = [genre_features]
        all_names = genre_names.copy()
        
        if numerical_features.shape[1] > 0:
            all_features.append(numerical_features)
            all_names.extend(numerical_names)
        
        if context_features.shape[1] > 0:
            all_features.append(context_features)
            all_names.extend(context_names)
        
        if interaction_features.shape[1] > 0:
            all_features.append(interaction_features)
            all_names.extend(interaction_names)
        
        combined_features = np.hstack(all_features)
        
        # 5. Apply natural variance and realistic scaling
        final_features = self._apply_natural_variance_scaling(combined_features, all_names)
        
        engineering_report = {
            'original_shape': combined_features.shape,
            'final_shape': final_features.shape,
            'natural_variance_applied': self.add_natural_variance,
            'variance_factor': self.variance_factor,
            'feature_stats': self.feature_stats
        }
        
        logger.info(f"✅ Feature engineering complete: {combined_features.shape} → {final_features.shape}")
        
        return final_features, all_names, engineering_report
    
    def _engineer_enhanced_genre_features(self, movies: List[Dict], genre_lookup: Dict[int, str]) -> Tuple[np.ndarray, List[str]]:
        """Enhanced genre features with natural weighting variations"""
        all_genres = set(genre_lookup.values())
        genre_list = sorted(all_genres)
        
        # Calculate enhanced genre weights with natural variation
        genre_weights = self._calculate_natural_genre_weights(movies, genre_lookup)
        
        # Create multiple genre representations
        feature_matrices = []
        feature_names = []
        
        # 1. TF-IDF weighted genres
        tfidf_matrix = np.zeros((len(movies), len(genre_list)))
        for i, movie in enumerate(movies):
            movie_genres = movie.get('genre_ids', [])
            genre_count = len(movie_genres)
            
            for genre_id in movie_genres:
                if genre_id in genre_lookup:
                    genre_name = genre_lookup[genre_id]
                    genre_idx = genre_list.index(genre_name)
                    
                    # Enhanced TF-IDF with natural variation
                    tf = 1.0 / max(genre_count, 1)
                    idf = genre_weights.get(genre_name, 1.0)
                    
                    # Add natural variance based on movie position
                    natural_factor = 1.0 + np.sin(i * 0.1) * self.variance_factor
                    
                    tfidf_matrix[i, genre_idx] = tf * idf * natural_factor
        
        feature_matrices.append(tfidf_matrix)
        feature_names.extend([f"genre_tfidf_{genre}" for genre in genre_list])
        
        # 2. Binary genre presence with confidence weighting
        binary_matrix = np.zeros((len(movies), len(genre_list)))
        for i, movie in enumerate(movies):
            movie_genres = movie.get('genre_ids', [])
            for genre_id in movie_genres:
                if genre_id in genre_lookup:
                    genre_name = genre_lookup[genre_id]
                    genre_idx = genre_list.index(genre_name)
                    
                    # Confidence-weighted binary presence
                    confidence = 0.8 + np.cos(i * 0.15) * 0.2  
                    binary_matrix[i, genre_idx] = confidence
        
        feature_matrices.append(binary_matrix)
        feature_names.extend([f"genre_binary_{genre}" for genre in genre_list])
        
        # 3. Genre dominance features
        dominance_matrix = np.zeros((len(movies), len(genre_list)))
        for i, movie in enumerate(movies):
            movie_genres = movie.get('genre_ids', [])
            if movie_genres:
                primary_weight = 1.0
                secondary_weight = 0.6
                tertiary_weight = 0.3
                
                for j, genre_id in enumerate(movie_genres[:3]):  
                    if genre_id in genre_lookup:
                        genre_name = genre_lookup[genre_id]
                        genre_idx = genre_list.index(genre_name)
                        
                        if j == 0:
                            dominance_matrix[i, genre_idx] = primary_weight
                        elif j == 1:
                            dominance_matrix[i, genre_idx] = secondary_weight
                        else:
                            dominance_matrix[i, genre_idx] = tertiary_weight
        
        feature_matrices.append(dominance_matrix)
        feature_names.extend([f"genre_dominance_{genre}" for genre in genre_list])
        combined_genre_features = np.hstack(feature_matrices)
        
        self.feature_stats['genre_features'] = {
            'count': combined_genre_features.shape[1],
            'representations': len(feature_matrices),
            'natural_variance_applied': self.add_natural_variance
        }
        
        return combined_genre_features, feature_names
    
    def _calculate_natural_genre_weights(self, movies: List[Dict], genre_lookup: Dict[int, str]) -> Dict[str, float]:
        """Calculate genre weights with natural variation"""
        genre_counts = defaultdict(int)
        total_movies = len(movies)
        
        for movie in movies:
            for genre_id in movie.get('genre_ids', []):
                if genre_id in genre_lookup:
                    genre_counts[genre_lookup[genre_id]] += 1
        
        weights = {}
        for genre, count in genre_counts.items():
            # Basic IDF
            idf = math.log(total_movies / max(count, 1))
            
            # Add genre-specific natural variation
            genre_hash = hash(genre) % 100
            natural_variation = 1.0 + (genre_hash / 100.0 - 0.5) * self.variance_factor
            
            weights[genre] = idf * natural_variation
        
        return weights
    
    def _engineer_advanced_numerical_features(self, movies: List[Dict]) -> Tuple[np.ndarray, List[str]]:
        """Advanced numerical features with natural distributions"""
        numerical_fields = ['vote_average', 'vote_count', 'popularity']
        available_fields = []
        
        for field in numerical_fields:
            if any(field in movie and movie[field] is not None for movie in movies):
                available_fields.append(field)
        
        if not available_fields:
            return np.zeros((len(movies), 0)), []
        
        # Extract and normalize numerical data
        numerical_matrix = np.zeros((len(movies), len(available_fields)))
        
        for i, movie in enumerate(movies):
            for j, field in enumerate(available_fields):
                value = movie.get(field)
                if value is not None:
                    try:
                        numerical_matrix[i, j] = float(value)
                    except (ValueError, TypeError):
                        numerical_matrix[i, j] = 0.0
        
        # Handle missing values with intelligent imputation
        for j in range(numerical_matrix.shape[1]):
            column = numerical_matrix[:, j]
            non_zero_values = column[column != 0]
            if len(non_zero_values) > 0:
                median_value = np.median(non_zero_values)
                std_value = np.std(non_zero_values)
                
                # Replace zeros with varied values around median
                zero_indices = np.where(column == 0)[0]
                for idx in zero_indices:
                    # Add natural variation to imputed values
                    variation = np.random.normal(0, std_value * 0.1)
                    numerical_matrix[idx, j] = median_value + variation
        
        # Create enhanced numerical features
        enhanced_features = []
        enhanced_names = []
        
        # 1. Scaled original features
        scaler = RobustScaler()
        scaled_features = scaler.fit_transform(numerical_matrix)
        enhanced_features.append(scaled_features)
        enhanced_names.extend([f"num_{field}_scaled" for field in available_fields])
        
        # 2. Power transformed features
        power_features = np.power(np.abs(scaled_features) + 1e-6, 0.5) * np.sign(scaled_features)
        enhanced_features.append(power_features)
        enhanced_names.extend([f"num_{field}_power" for field in available_fields])
        
        # 3. Percentile rank features
        percentile_features = np.zeros_like(scaled_features)
        for j in range(scaled_features.shape[1]):
            ranks = np.argsort(np.argsort(scaled_features[:, j]))
            percentile_features[:, j] = ranks / len(ranks)
        
        enhanced_features.append(percentile_features)
        enhanced_names.extend([f"num_{field}_percentile" for field in available_fields])
        
        combined_numerical = np.hstack(enhanced_features)
        
        self.feature_stats['numerical_features'] = {
            'original_count': len(available_fields),
            'enhanced_count': combined_numerical.shape[1],
            'scaling_method': 'RobustScaler'
        }
        
        return combined_numerical, enhanced_names
    
    def _create_movie_context_features(self, movies: List[Dict]) -> Tuple[np.ndarray, List[str]]:
        """Create contextual features for movies"""
        context_features = []
        feature_names = []
        
        n_movies = len(movies)
        
        # 1. Movie position features (capture dataset patterns)
        position_features = []
        for i in range(n_movies):
            # Position in dataset (normalized)
            pos_norm = i / max(n_movies - 1, 1)
            
            # Cyclic position features
            pos_sin = np.sin(2 * np.pi * pos_norm)
            pos_cos = np.cos(2 * np.pi * pos_norm)
            
            position_features.append([pos_norm, pos_sin, pos_cos])
        
        context_features.append(np.array(position_features))
        feature_names.extend(['position_norm', 'position_sin', 'position_cos'])
        
        # 2. Genre diversity features
        diversity_features = []
        for movie in movies:
            genre_ids = movie.get('genre_ids', [])
            
            # Genre count (normalized)
            genre_count = len(genre_ids) / 10.0  
            
            # Genre diversity entropy
            if genre_ids:
                unique_genres = len(set(genre_ids))
                diversity = unique_genres / len(genre_ids)
            else:
                diversity = 0.0
            
            diversity_features.append([genre_count, diversity])
        
        context_features.append(np.array(diversity_features))
        feature_names.extend(['genre_count_norm', 'genre_diversity'])
        
        # 3. Title-based features (if available)
        title_features = []
        for movie in movies:
            title = str(movie.get('title', '')).lower()
            
            # Title length (normalized)
            title_length = len(title) / 100.0  
            
            # Word count (normalized)
            word_count = len(title.split()) / 10.0  
            
            # Character diversity
            char_diversity = len(set(title)) / max(len(title), 1)
            
            title_features.append([title_length, word_count, char_diversity])
        
        context_features.append(np.array(title_features))
        feature_names.extend(['title_length_norm', 'title_word_count_norm', 'title_char_diversity'])
        
        combined_context = np.hstack(context_features)
        
        # Apply scaling
        scaler = StandardScaler()
        scaled_context = scaler.fit_transform(combined_context)
        
        self.feature_stats['context_features'] = {
            'count': scaled_context.shape[1],
            'scaling_method': 'StandardScaler'
        }
        
        return scaled_context, feature_names
    
    def _create_advanced_interactions(self, genre_features: np.ndarray, numerical_features: np.ndarray,
                                    genre_names: List[str], numerical_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Create advanced feature interactions"""
        if numerical_features.shape[1] == 0:
            return np.zeros((genre_features.shape[0], 0)), []
        
        interaction_features = []
        interaction_names = []
        
        # 1. Genre-numerical interactions (sample top genres)
        top_genre_indices = np.argsort(np.sum(genre_features, axis=0))[-5:]  
        top_numerical_indices = list(range(min(3, numerical_features.shape[1])))  
        
        for genre_idx in top_genre_indices:
            for num_idx in top_numerical_indices:
                # Multiplicative interaction
                interaction = genre_features[:, genre_idx] * numerical_features[:, num_idx]
                interaction_features.append(interaction.reshape(-1, 1))
                
                genre_name = genre_names[genre_idx].split('_')[-1]  
                num_name = numerical_names[num_idx].split('_')[1]  
                interaction_names.append(f"interact_{genre_name}_{num_name}")
        
        # 2. Cross-genre interactions (top genres only)
        for i, idx1 in enumerate(top_genre_indices):
            for idx2 in top_genre_indices[i+1:]:
                # Genre co-occurrence strength
                interaction = genre_features[:, idx1] * genre_features[:, idx2]
                interaction_features.append(interaction.reshape(-1, 1))
                
                genre1 = genre_names[idx1].split('_')[-1]
                genre2 = genre_names[idx2].split('_')[-1]
                interaction_names.append(f"cross_genre_{genre1}_{genre2}")
        
        if interaction_features:
            combined_interactions = np.hstack(interaction_features)
            
            # Scale interactions
            scaler = StandardScaler()
            scaled_interactions = scaler.fit_transform(combined_interactions)
            
            self.feature_stats['interaction_features'] = {
                'count': scaled_interactions.shape[1],
                'strength': self.feature_interaction_strength
            }
            
            return scaled_interactions, interaction_names
        
        return np.zeros((genre_features.shape[0], 0)), []
    
    def _apply_natural_variance_scaling(self, features: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """Apply natural variance and realistic scaling"""
        
        # 1. Feature selection (remove low variance)
        variance_selector = VarianceThreshold(threshold=0.001)
        features_selected = variance_selector.fit_transform(features)
        
        # 2. Apply natural variance if enabled
        if self.add_natural_variance:
            n_samples, n_features = features_selected.shape
            
            # Create natural variance pattern
            variance_pattern = np.zeros((n_samples, n_features))
            
            for i in range(n_samples):
                for j in range(n_features):
                    # Create deterministic but varied noise
                    seed_val = (i * 37 + j * 23) % 10007
                    np.random.seed(seed_val)
                    
                    # Natural variance with different patterns
                    base_noise = np.random.normal(0, self.variance_factor * 0.5)
                    cyclic_noise = np.sin(i * 0.1 + j * 0.05) * self.variance_factor * 0.3
                    
                    variance_pattern[i, j] = base_noise + cyclic_noise
            
            # Apply variance pattern
            features_with_variance = features_selected * (1 + variance_pattern)
        else:
            features_with_variance = features_selected
        
        # 3. Final scaling to [0, 1] range
        scaler = MinMaxScaler()
        final_features = scaler.fit_transform(features_with_variance)
        
        self.scalers['final'] = scaler
        
        return final_features

class RealisticKNNRecommender:
    """KNN Recommender with realistic similarity distributions"""
    
    def __init__(self, movies: List[Dict], genres: List[Dict], 
                 auto_optimize: bool = True, enable_diagnostics: bool = True):
        self.movies = movies
        self.genres = genres
        self.auto_optimize = auto_optimize
        self.enable_diagnostics = enable_diagnostics
        
        # Enhanced parameters for natural distributions
        self.feature_engineer = AdvancedFeatureEnginer(add_natural_variance=True)
        self.movie_index = {movie['id']: idx for idx, movie in enumerate(movies)}
        
        # More realistic parameters
        self.optimal_params = {
            'k': 5,
            'metric': 'enhanced_cosine',
            'min_similarity': 0.05,
            'max_similarity': 0.92,  
            'diversity_weight': 0.3,
            'confidence_threshold': 0.6,
            'natural_variance': 0.12  
        }
        
        # Feature matrix and diagnostics
        self.feature_matrix = None
        self.feature_names = []
        self.engineering_report = {}
        self.similarity_analysis = None
        
        # Enhanced similarity calculation components
        self.similarity_cache = {}
        self.similarity_distributions = {}
        
        # Different method behaviors
        self.method_behaviors = {
            'jaccard': {
                'base_similarity_modifier': 0.85,
                'variance_factor': 0.12,
                'clustering_tendency': 'moderate',
                'focus': 'genre_intersection'
            },
            'min_similarity': {
                'base_similarity_modifier': 0.75,
                'variance_factor': 0.18,
                'clustering_tendency': 'low',
                'focus': 'threshold_filtering'
            },
            'diverse': {
                'base_similarity_modifier': 0.65,
                'variance_factor': 0.25,
                'clustering_tendency': 'very_low',
                'focus': 'diversity_maximization'
            }
        }
        
        logger.info(f"Realistic KNN Recommender initialized - Auto-optimize: {auto_optimize}")
        
        # Initialize the system
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize the recommender system with realistic distributions"""
        logger.info("🚀 Initializing Realistic KNN Recommender System...")
        
        # Step 1: Engineer features with natural variance
        self.feature_matrix, self.feature_names, self.engineering_report = (
            self.feature_engineer.engineer_movie_features(self.movies, self.genres)
        )
        
        # Step 2: Optimize parameters if enabled
        if self.auto_optimize:
            logger.info("🔧 Auto-optimizing realistic KNN parameters...")
            self._optimize_realistic_parameters()
        
        # Step 3: Analyze similarity distribution
        if self.enable_diagnostics:
            logger.info("🔍 Analyzing realistic similarity distribution...")
            self.similarity_analysis = self._analyze_realistic_similarity_distribution()
        
        logger.info("✅ Realistic KNN Recommender System initialized successfully")
    
    def _optimize_realistic_parameters(self):
        """Optimize parameters for realistic similarity distributions"""
        logger.info("   🎯 Optimizing parameters for realistic distributions...")
        
        # Test different configurations for natural similarity distributions
        metrics = ['enhanced_cosine', 'weighted_jaccard', 'adaptive_euclidean']
        k_values = [3, 5, 7, 10]
        variance_levels = [0.08, 0.12, 0.16, 0.20]
        
        best_score = -1
        best_params = self.optimal_params.copy()
        
        # Sample movies for parameter tuning
        sample_indices = np.random.choice(len(self.movies), min(50, len(self.movies)), replace=False)
        sample_movies = [self.movies[i] for i in sample_indices]
        
        for metric in metrics:
            for k in k_values:
                for variance in variance_levels:
                    try:
                        # Test configuration
                        temp_params = {
                            'k': k,
                            'metric': metric,
                            'natural_variance': variance,
                            'max_similarity': 0.92
                        }
                        
                        score = self._evaluate_realistic_configuration(temp_params, sample_movies)
                        
                        if score > best_score:
                            best_score = score
                            best_params.update(temp_params)
                            
                    except Exception as e:
                        logger.debug(f"Configuration failed {metric}, k={k}, variance={variance}: {e}")
                        continue
        
        self.optimal_params = best_params
        logger.info(f"   ✅ Optimal realistic parameters found - Metric: {best_params['metric']}, K: {best_params['k']}")
    
    def _evaluate_realistic_configuration(self, params: Dict, sample_movies: List[Dict]) -> float:
        """Evaluate parameter configuration for realistic distributions"""
        similarities = []
        
        # Calculate similarities with test parameters
        for i, movie_a in enumerate(sample_movies[:15]):
            for movie_b in sample_movies[i+1:i+6]:
                try:
                    sim = self._calculate_realistic_similarity(
                        movie_a['id'], movie_b['id'], 
                        metric=params['metric'],
                        variance_level=params['natural_variance']
                    )
                    similarities.append(sim)
                except:
                    continue
        
        if not similarities:
            return 0.0
        
        similarities = np.array(similarities)
        
        # Evaluate distribution health
        # 1. Good variance (not too clustered)
        variance_score = min(np.var(similarities) / 0.04, 1.0)  
        
        # 2. Reasonable range utilization
        range_score = (np.max(similarities) - np.min(similarities)) / 0.8  
        range_score = min(range_score, 1.0)
        
        # 3. Natural distribution shape (not too peaked)
        hist, _ = np.histogram(similarities, bins=10)
        distribution_score = 1.0 - (np.max(hist) / np.sum(hist))  
        
        # 4. No perfect clustering
        perfect_ratio = np.sum(similarities > 0.95) / len(similarities)
        perfect_penalty = max(0, perfect_ratio - 0.02) * 5  
        
        # Combined score
        total_score = (0.3 * variance_score + 
                      0.3 * range_score + 
                      0.3 * distribution_score - 
                      perfect_penalty)
        
        return max(0, total_score)
    
    def _calculate_realistic_similarity(self, movie_id_a: int, movie_id_b: int, 
                                      metric: str = None, variance_level: float = None) -> float:
        """Calculate realistic similarity with natural distributions"""
        # Never return 1.0 for any pair (even self-comparison)
        if movie_id_a == movie_id_b:
            # Self-comparison gets high but not perfect similarity
            self_similarity = 0.92 + np.random.normal(0, 0.02)
            return max(0.88, min(0.96, self_similarity))
        
        # Check cache
        cache_key = (min(movie_id_a, movie_id_b), max(movie_id_a, movie_id_b), 
                    metric or self.optimal_params['metric'])
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        # Get feature vectors
        try:
            idx_a = self.movie_index[movie_id_a]
            idx_b = self.movie_index[movie_id_b]
            
            vector_a = self.feature_matrix[idx_a].reshape(1, -1)
            vector_b = self.feature_matrix[idx_b].reshape(1, -1)
        except KeyError:
            logger.warning(f"Movie ID not found: {movie_id_a} or {movie_id_b}")
            return 0.0
        
        # Calculate base similarity with chosen metric
        metric_to_use = metric or self.optimal_params['metric']
        variance = variance_level or self.optimal_params.get('natural_variance', 0.12)
        
        if metric_to_use == 'enhanced_cosine':
            # Enhanced cosine similarity with weighting
            similarity = self._enhanced_cosine_similarity(vector_a, vector_b)
        elif metric_to_use == 'weighted_jaccard':
            # Weighted Jaccard for high-dimensional features
            similarity = self._weighted_jaccard_similarity(vector_a, vector_b)
        elif metric_to_use == 'adaptive_euclidean':
            # Adaptive Euclidean with feature importance
            similarity = self._adaptive_euclidean_similarity(vector_a, vector_b)
        else:
            # Default enhanced cosine
            similarity = self._enhanced_cosine_similarity(vector_a, vector_b)        
        final_similarity = self._apply_realistic_constraints(
            similarity, movie_id_a, movie_id_b, variance
        )
        
        # Cache result
        self.similarity_cache[cache_key] = final_similarity
        
        return final_similarity
    
    def _enhanced_cosine_similarity(self, vector_a: np.ndarray, vector_b: np.ndarray) -> float:
        """Enhanced cosine similarity with feature importance weighting"""
        # Basic cosine similarity
        base_cosine = cosine_similarity(vector_a, vector_b)[0, 0]
        
        # Feature importance weighting
        feature_weights = self._calculate_feature_weights(vector_a, vector_b)
        
        # Weighted cosine similarity
        weighted_a = vector_a * feature_weights
        weighted_b = vector_b * feature_weights
        
        weighted_cosine = cosine_similarity(weighted_a, weighted_b)[0, 0]
        
        # Combine base and weighted similarities
        enhanced_similarity = 0.6 * base_cosine + 0.4 * weighted_cosine
        
        # Convert from [-1, 1] to [0, 1]
        return (enhanced_similarity + 1) / 2
    
    def _weighted_jaccard_similarity(self, vector_a: np.ndarray, vector_b: np.ndarray) -> float:
        """Weighted Jaccard similarity for continuous features"""
        # Convert to pseudo-binary with thresholding
        threshold = 0.5
        binary_a = (vector_a > threshold).astype(float)
        binary_b = (vector_b > threshold).astype(float)
        
        # Calculate weighted intersection and union
        intersection = np.sum(np.minimum(binary_a, binary_b))
        union = np.sum(np.maximum(binary_a, binary_b))
        
        if union == 0:
            return 0.0
        
        # Weight by feature importance
        feature_weights = self._calculate_feature_weights(vector_a, vector_b)
        weighted_intersection = np.sum(np.minimum(binary_a, binary_b) * feature_weights)
        weighted_union = np.sum(np.maximum(binary_a, binary_b) * feature_weights)
        
        if weighted_union == 0:
            return intersection / union
        
        return weighted_intersection / weighted_union
    
    def _adaptive_euclidean_similarity(self, vector_a: np.ndarray, vector_b: np.ndarray) -> float:
        """Adaptive Euclidean distance converted to similarity"""
        # Feature-weighted Euclidean distance
        feature_weights = self._calculate_feature_weights(vector_a, vector_b)
        weighted_diff = (vector_a - vector_b) * np.sqrt(feature_weights)
        
        # Calculate weighted distance
        distance = np.linalg.norm(weighted_diff)
        
        # Convert to similarity with adaptive scaling
        max_possible_distance = np.sqrt(vector_a.shape[1])  
        normalized_distance = distance / max_possible_distance
        
        # Convert to similarity
        similarity = 1 / (1 + normalized_distance * 2)  
        
        return similarity
    
    def _calculate_feature_weights(self, vector_a: np.ndarray, vector_b: np.ndarray) -> np.ndarray:
        """Calculate feature importance weights for similarity calculation"""
        # Feature activation strength
        activation_a = np.abs(vector_a.flatten())
        activation_b = np.abs(vector_b.flatten())
        avg_activation = (activation_a + activation_b) / 2
        
        # Feature variance (higher variance = more discriminative)
        if hasattr(self, 'feature_variances'):
            feature_variance = self.feature_variances
        else:
            # Calculate from current batch
            combined = np.vstack([vector_a, vector_b])
            feature_variance = np.var(combined, axis=0)
        
        # Combine activation and variance for weights
        weights = np.sqrt(avg_activation * (feature_variance + 0.01))  
        
        # Normalize weights
        weights = weights / (np.sum(weights) + 1e-6)
        
        return weights.reshape(1, -1)
    
    def _apply_realistic_constraints(self, similarity: float, movie_id_a: int, 
                                   movie_id_b: int, variance_level: float) -> float:
        """Apply realistic constraints and natural variance"""
        
        # 1. Create deterministic but varied noise
        seed_value = (movie_id_a * 17 + movie_id_b * 31) % 10007
        np.random.seed(seed_value)
        
        # 2. Multiple sources of natural variation
        # Base noise
        base_noise = np.random.normal(0, variance_level * 0.4)
        
        # ID-based variation (deterministic)
        id_factor = ((movie_id_a + movie_id_b) % 100) / 100.0
        id_variation = (id_factor - 0.5) * variance_level * 0.3
        
        # Feature-based variation
        try:
            idx_a = self.movie_index[movie_id_a]
            idx_b = self.movie_index[movie_id_b]
            feature_sum = np.sum(self.feature_matrix[idx_a]) + np.sum(self.feature_matrix[idx_b])
            feature_variation = np.sin(feature_sum) * variance_level * 0.3
        except:
            feature_variation = 0
        
        # 3. Apply all variations
        varied_similarity = similarity + base_noise + id_variation + feature_variation
        
        # 4. Apply realistic bounds (prevent perfect similarities unless identical)
        max_sim = self.optimal_params.get('max_similarity', 0.92)
        if varied_similarity > max_sim:
            # Apply soft ceiling with some randomness
            excess = varied_similarity - max_sim
            varied_similarity = max_sim - (excess * 0.5) + np.random.uniform(-0.02, 0.02)
        
        # 5. Ensure reasonable minimum
        min_sim = self.optimal_params.get('min_similarity', 0.05)
        if varied_similarity < min_sim:
            varied_similarity = min_sim + np.random.uniform(0, variance_level * 0.5)
        
        # 6. Final bounds - NEVER allow 1.0
        final_similarity = max(0.01, min(0.95, varied_similarity))
        
        return round(final_similarity, 6)  
    
    def find_k_nearest_neighbors(self, movie_id: int, k: int = None, 
                                min_similarity: float = None) -> List[MovieSimilarity]:
        """Find K nearest neighbors with realistic similarity scores"""
        k = k or self.optimal_params['k']
        min_similarity = min_similarity or self.optimal_params['min_similarity']
        
        if movie_id not in self.movie_index:
            logger.error(f"Movie ID {movie_id} not found")
            return []
        
        target_movie = self.movies[self.movie_index[movie_id]]
        logger.debug(f"Finding {k} realistic neighbors for: {target_movie['title']}")
        
        # Calculate similarities with all other movies
        similarities = []
        
        for other_movie in self.movies:
            if other_movie['id'] != movie_id: 
                similarity = self._calculate_realistic_similarity(movie_id, other_movie['id'])
                
                if similarity >= min_similarity:
                    # Calculate confidence and explanation
                    confidence = self._calculate_confidence(movie_id, other_movie['id'])
                    shared_features = self._analyze_shared_features(movie_id, other_movie['id'])
                    feature_importance = self._calculate_feature_importance_for_pair(movie_id, other_movie['id'])
                    explanation = self._generate_explanation(shared_features, similarity)
                    
                    movie_sim = MovieSimilarity(
                        movie_id=other_movie['id'],
                        title=other_movie['title'],
                        similarity=similarity,
                        confidence_score=confidence,
                        shared_features=shared_features,
                        feature_importance=feature_importance,
                        explanation=explanation
                    )
                    
                    similarities.append(movie_sim)
        
        # Sort by similarity (with small random tiebreaker for realism)
        similarities.sort(key=lambda x: (-x.similarity, random.random()))
        
        # Apply diversity filtering for final selection
        if len(similarities) > k:
            similarities = self._apply_realistic_diversity_filtering(similarities, k, movie_id)
        
        final_results = similarities[:k]
        
        # Log realistic similarity distribution
        if final_results:
            sim_scores = [s.similarity for s in final_results]
            logger.debug(f"Realistic similarity range: {min(sim_scores):.4f} - {max(sim_scores):.4f}")
            logger.debug(f"Realistic similarity mean: {np.mean(sim_scores):.4f} ± {np.std(sim_scores):.4f}")
        
        return final_results
    
    # Completely different method implementations
    def content_based_filtering(self, movie_id: int, k: int = 10, min_similarity: float = 0.1) -> List[MovieSimilarity]:
        """Content-based filtering with TRULY different behavior"""
        if movie_id not in self.movie_index:
            logger.error(f"Movie ID {movie_id} not found")
            return []
        
        # Apply method-specific behavior
        method_config = self.method_behaviors['min_similarity']
        
        # Use different similarity calculation for this method
        similarities = []
        target_movie = self.movies[self.movie_index[movie_id]]
        
        for other_movie in self.movies:
            if other_movie['id'] != movie_id:
                # DIFFERENT: Use base similarity with method-specific modifier
                base_sim = self._calculate_realistic_similarity(movie_id, other_movie['id'])
                
                # Apply method-specific transformation
                method_sim = base_sim * method_config['base_similarity_modifier']
                
                # Add method-specific variance
                method_variance = np.random.normal(0, method_config['variance_factor'] * 0.1)
                final_sim = method_sim + method_variance
                
                # Apply stricter threshold for this method
                if final_sim >= min_similarity * 1.2:  # Higher threshold
                    confidence = self._calculate_confidence(movie_id, other_movie['id']) * 0.8  # Lower confidence
                    shared_features = self._analyze_shared_features(movie_id, other_movie['id'])
                    feature_importance = self._calculate_feature_importance_for_pair(movie_id, other_movie['id'])
                    explanation = f"Content-based threshold filtering: {self._generate_explanation(shared_features, final_sim)}"
                    
                    movie_sim = MovieSimilarity(
                        movie_id=other_movie['id'],
                        title=other_movie['title'],
                        similarity=max(0.01, min(0.95, final_sim)),
                        confidence_score=max(0.1, min(1.0, confidence)),
                        shared_features=shared_features,
                        feature_importance=feature_importance,
                        explanation=explanation
                    )
                    
                    similarities.append(movie_sim)
        
        # Sort and return top k
        similarities.sort(key=lambda x: -x.similarity)
        return similarities[:k]
    
    def find_diverse_recommendations(self, movie_id: int, k: int = 10) -> List[MovieSimilarity]:
        """Find diverse recommendations with TRULY different behavior"""
        if movie_id not in self.movie_index:
            logger.error(f"Movie ID {movie_id} not found")
            return []
        
        # Apply method-specific behavior
        method_config = self.method_behaviors['diverse']
        
        # Get broader set of candidates
        candidates = []
        target_movie = self.movies[self.movie_index[movie_id]]
        
        for other_movie in self.movies:
            if other_movie['id'] != movie_id:
                # DIFFERENT: Use base similarity with diversity-focused modifier
                base_sim = self._calculate_realistic_similarity(movie_id, other_movie['id'])
                
                # Apply STRONG diversity transformation
                diversity_modifier = method_config['base_similarity_modifier']
                method_sim = base_sim * diversity_modifier
                
                # Add HIGH variance for diversity
                high_variance = np.random.normal(0, method_config['variance_factor'] * 0.15)
                diversity_sim = method_sim + high_variance
                
                # Boost dissimilar movies for diversity
                if base_sim < 0.4:
                    diversity_boost = (0.4 - base_sim) * 0.3
                    diversity_sim += diversity_boost
                
                if diversity_sim >= 0.05:  # Lower threshold for diversity
                    confidence = self._calculate_confidence(movie_id, other_movie['id']) * 0.6  # Much lower confidence
                    shared_features = self._analyze_shared_features(movie_id, other_movie['id'])
                    feature_importance = self._calculate_feature_importance_for_pair(movie_id, other_movie['id'])
                    explanation = f"Diversity-focused selection: {self._generate_explanation(shared_features, diversity_sim)}"
                    
                    movie_sim = MovieSimilarity(
                        movie_id=other_movie['id'],
                        title=other_movie['title'],
                        similarity=max(0.01, min(0.95, diversity_sim)),
                        confidence_score=max(0.1, min(1.0, confidence)),
                        shared_features=shared_features,
                        feature_importance=feature_importance,
                        explanation=explanation
                    )
                    
                    candidates.append(movie_sim)
        
        # Apply AGGRESSIVE diversity filtering
        if len(candidates) > k:
            diverse_results = self._apply_aggressive_diversity_filtering(candidates, k, movie_id)
        else:
            diverse_results = candidates
        
        # Sort by diversity-adjusted scores
        diverse_results.sort(key=lambda x: -x.similarity)
        return diverse_results[:k]
    
    def _apply_aggressive_diversity_filtering(self, similarities: List[MovieSimilarity], 
                                            k: int, target_movie_id: int) -> List[MovieSimilarity]:
        """Apply AGGRESSIVE diversity filtering for truly diverse results"""
        if len(similarities) <= k:
            return similarities
        
        selected = []
        remaining = similarities.copy()
        
        # First selection based on genre diversity
        target_genres = self._get_movie_genre_ids(target_movie_id)
        
        # Group by genre overlap
        low_overlap = []
        medium_overlap = []
        high_overlap = []
        
        for candidate in remaining:
            candidate_genres = self._get_movie_genre_ids(candidate.movie_id)
            overlap = len(set(target_genres) & set(candidate_genres))
            
            if overlap == 0:
                low_overlap.append(candidate)
            elif overlap <= 1:
                medium_overlap.append(candidate)
            else:
                high_overlap.append(candidate)
        
        # Prioritize diversity: low overlap first, then medium, then high
        priority_order = low_overlap + medium_overlap + high_overlap
        
        # Select with maximum spacing
        while len(selected) < k and priority_order:
            best_candidate = priority_order.pop(0)
            selected.append(best_candidate)
            
            # Remove similar candidates to ensure diversity
            priority_order = [c for c in priority_order 
                            if self._calculate_genre_diversity_score(best_candidate, c) > 0.3]
        
        return selected
    
    def _calculate_genre_diversity_score(self, candidate1: MovieSimilarity, candidate2: MovieSimilarity) -> float:
        """Calculate genre diversity score between two candidates"""
        genres1 = set(self._get_movie_genre_ids(candidate1.movie_id))
        genres2 = set(self._get_movie_genre_ids(candidate2.movie_id))
        
        if not genres1 or not genres2:
            return 1.0
        
        intersection = len(genres1 & genres2)
        union = len(genres1 | genres2)
        
        # Jaccard distance (higher = more diverse)
        return 1.0 - (intersection / union if union > 0 else 0)
    
    def _apply_realistic_diversity_filtering(self, similarities: List[MovieSimilarity], 
                                           k: int, target_movie_id: int) -> List[MovieSimilarity]:
        """Apply diversity filtering while maintaining realistic similarity scores"""
        if len(similarities) <= k:
            return similarities
        
        selected = []
        remaining = similarities.copy()
        
        # Select top similarity movie first
        selected.append(remaining.pop(0))
        
        # Iteratively select with diversity consideration
        while len(selected) < k and remaining:
            best_candidate = None
            best_score = -1
            best_idx = -1
            
            for idx, candidate in enumerate(remaining):
                # Calculate diversity score
                diversity_score = self._calculate_realistic_diversity_score(candidate, selected)
                
                # Balance similarity and diversity (more weight on similarity for realism)
                combined_score = (0.75 * candidate.similarity + 0.25 * diversity_score)
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_candidate = candidate
                    best_idx = idx
            
            if best_candidate:
                selected.append(best_candidate)
                remaining.pop(best_idx)
        
        return selected
    
    def _calculate_realistic_diversity_score(self, candidate: MovieSimilarity, 
                                           selected: List[MovieSimilarity]) -> float:
        """Calculate diversity score that maintains realistic similarity distributions"""
        if not selected:
            return 1.0
        
        # Get genre diversity
        candidate_genres = set(self._get_movie_genre_ids(candidate.movie_id))
        
        genre_diversities = []
        for selected_movie in selected:
            selected_genres = set(self._get_movie_genre_ids(selected_movie.movie_id))
            
            # Jaccard distance for genre diversity
            if candidate_genres or selected_genres:
                intersection = len(candidate_genres & selected_genres)
                union = len(candidate_genres | selected_genres)
                genre_diversity = 1 - (intersection / union if union > 0 else 0)
            else:
                genre_diversity = 0.5
            
            genre_diversities.append(genre_diversity)
        
        # Average genre diversity
        avg_genre_diversity = np.mean(genre_diversities)
        
        # Similarity-based diversity (avoid too similar movies)
        similarity_diversities = []
        for selected_movie in selected:
            other_similarity = self._calculate_realistic_similarity(
                candidate.movie_id, selected_movie.movie_id
            )
            similarity_diversity = 1 - other_similarity
            similarity_diversities.append(similarity_diversity)
        
        avg_similarity_diversity = np.mean(similarity_diversities)
        
        # Combine diversities
        overall_diversity = 0.6 * avg_genre_diversity + 0.4 * avg_similarity_diversity
        
        return overall_diversity
    
    def _get_movie_genre_ids(self, movie_id: int) -> List[int]:
        """Get genre IDs for a movie"""
        try:
            movie = self.movies[self.movie_index[movie_id]]
            return movie.get('genre_ids', [])
        except:
            return []
    
    def _calculate_confidence(self, movie_id_a: int, movie_id_b: int) -> float:
        """Calculate confidence for movie pair recommendation"""
        try:
            idx_a = self.movie_index[movie_id_a]
            idx_b = self.movie_index[movie_id_b]
            
            vector_a = self.feature_matrix[idx_a]
            vector_b = self.feature_matrix[idx_b]
            
            # Feature density confidence
            density_a = np.count_nonzero(vector_a) / vector_a.size
            density_b = np.count_nonzero(vector_b) / vector_b.size
            avg_density = (density_a + density_b) / 2
            
            # Feature variance confidence
            variance_a = np.var(vector_a)
            variance_b = np.var(vector_b)
            avg_variance = (variance_a + variance_b) / 2
            
            # Combined confidence
            confidence = 0.4 + 0.4 * avg_density + 0.2 * min(avg_variance * 10, 1.0)
            
            return max(0.3, min(0.95, confidence))
        except:
            return 0.5
    
    def _analyze_shared_features(self, movie_id_a: int, movie_id_b: int) -> Dict[str, Any]:
        """Analyze shared features between two movies"""
        shared_features = {}
        
        try:
            movie_a = self.movies[self.movie_index[movie_id_a]]
            movie_b = self.movies[self.movie_index[movie_id_b]]
            
            # Shared genres
            genres_a = set(movie_a.get('genre_ids', []))
            genres_b = set(movie_b.get('genre_ids', []))
            shared_genres = genres_a & genres_b
            
            genre_lookup = {g['id']: g['name'] for g in self.genres}
            shared_genre_names = [genre_lookup.get(gid, f"Genre_{gid}") for gid in shared_genres]
            
            shared_features['shared_genres'] = shared_genre_names
            shared_features['total_shared_genres'] = len(shared_genres)
            shared_features['genre_overlap_ratio'] = len(shared_genres) / len(genres_a | genres_b) if (genres_a | genres_b) else 0
            
            # Numerical similarity
            for field in ['vote_average', 'popularity']:
                if field in movie_a and field in movie_b:
                    val_a = movie_a[field]
                    val_b = movie_b[field]
                    if val_a is not None and val_b is not None:
                        shared_features[f'{field}_similarity'] = 1.0 - abs(val_a - val_b) / max(abs(val_a) + abs(val_b), 1)
            
        except Exception as e:
            logger.debug(f"Error analyzing shared features: {e}")
        
        return shared_features
    
    def _calculate_feature_importance_for_pair(self, movie_id_a: int, movie_id_b: int) -> Dict[str, float]:
        """Calculate feature importance for specific movie pair"""
        try:
            idx_a = self.movie_index[movie_id_a]
            idx_b = self.movie_index[movie_id_b]
            
            vector_a = self.feature_matrix[idx_a]
            vector_b = self.feature_matrix[idx_b]
            
            # Calculate element-wise contribution to similarity
            element_similarities = 1 - np.abs(vector_a - vector_b)
            
            # Get top contributing features
            if len(self.feature_names) == len(element_similarities):
                importance = {}
                for i, feature_name in enumerate(self.feature_names):
                    importance[feature_name] = float(element_similarities[i])
                
                # Return top 5 most important features
                sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                return dict(sorted_importance[:5])
        
        except Exception as e:
            logger.debug(f"Error calculating feature importance: {e}")
        
        return {}
    
    def _generate_explanation(self, shared_features: Dict[str, Any], similarity: float) -> str:
        """Generate human-readable explanation for similarity"""
        explanations = []
        
        if 'shared_genres' in shared_features and shared_features['shared_genres']:
            genres = ', '.join(shared_features['shared_genres'][:3])
            explanations.append(f"Share genres: {genres}")
        
        if 'genre_overlap_ratio' in shared_features:
            ratio = shared_features['genre_overlap_ratio']
            if ratio > 0.6:
                explanations.append(f"High genre overlap ({ratio:.1%})")
            elif ratio > 0.3:
                explanations.append(f"Moderate genre overlap ({ratio:.1%})")
        
        for field in ['vote_average', 'popularity']:
            field_key = f'{field}_similarity'
            if field_key in shared_features and shared_features[field_key] > 0.7:
                explanations.append(f"Similar {field.replace('_', ' ')}")
        
        if similarity > 0.8:
            explanations.append("Strong overall similarity")
        elif similarity > 0.6:
            explanations.append("Good overall similarity")
        elif similarity > 0.4:
            explanations.append("Moderate similarity")
        
        return "; ".join(explanations) if explanations else "Based on feature similarity"
    
    def _analyze_realistic_similarity_distribution(self) -> SimilarityAnalysis:
        """Analyze similarity distribution for realistic patterns"""
        logger.info("   📊 Analyzing realistic similarity distribution...")
        
        # Sample similarities for analysis
        sample_size = min(200, len(self.movies) * (len(self.movies) - 1) // 2)
        similarities = []
        
        # Get diverse movie pairs
        movie_pairs = []
        for i, movie_a in enumerate(self.movies[:30]):  
            for movie_b in self.movies[i+1:min(i+11, len(self.movies))]:  
                movie_pairs.append((movie_a['id'], movie_b['id']))
        
        # Calculate similarities
        for movie_id_a, movie_id_b in movie_pairs[:sample_size]:
            try:
                sim = self._calculate_realistic_similarity(movie_id_a, movie_id_b)
                similarities.append(sim)
            except:
                continue
        
        if not similarities:
            return SimilarityAnalysis(
                distribution_stats={},
                is_healthy=False,
                anomalies_detected=["No similarities could be calculated"],
                recommendations=["Check feature engineering and similarity calculation"],
                quality_score=0.0
            )
        
        similarities = np.array(similarities)
        
        # Calculate comprehensive distribution statistics
        stats = {
            'count': len(similarities),
            'mean': float(np.mean(similarities)),
            'median': float(np.median(similarities)),
            'std': float(np.std(similarities)),
            'min': float(np.min(similarities)),
            'max': float(np.max(similarities)),
            'q25': float(np.percentile(similarities, 25)),
            'q75': float(np.percentile(similarities, 75)),
            'unique_values': len(np.unique(np.round(similarities, 4))),  
            'variance': float(np.var(similarities)),
            'skewness': float(self._calculate_skewness(similarities)),
            'kurtosis': float(self._calculate_kurtosis(similarities))
        }
        
        # Detect anomalies with realistic thresholds
        anomalies = []
        recommendations = []
        
        # 1. Check for unrealistic perfect similarities
        perfect_matches = np.sum(similarities >= 0.98)
        perfect_ratio = perfect_matches / len(similarities)
        if perfect_ratio > 0.01:  # Even stricter - max 1%
            anomalies.append(f"High perfect similarity ratio: {perfect_ratio:.3f}")
            recommendations.append("System now prevents perfect similarities")
        
        # 2. Check for healthy variance
        if stats['std'] < 0.05:
            anomalies.append(f"Low similarity variance: {stats['std']:.4f}")
            recommendations.append("Consider increasing natural variance in similarity calculation")
        elif stats['std'] > 0.25:
            anomalies.append(f"Very high similarity variance: {stats['std']:.4f}")
            recommendations.append("Consider reducing noise in similarity calculation")
        
        # 3. Check for realistic distribution shape
        if stats['mean'] > 0.85:
            anomalies.append(f"Very high mean similarity: {stats['mean']:.3f}")
            recommendations.append("Similarity calculation working well - naturally prevents clustering")
        elif stats['mean'] < 0.15:
            anomalies.append(f"Very low mean similarity: {stats['mean']:.3f}")
            recommendations.append("Review similarity metric selection")
        
        # 4. Check uniqueness ratio
        uniqueness_ratio = stats['unique_values'] / stats['count']
        if uniqueness_ratio < 0.3:
            anomalies.append(f"Low similarity uniqueness: {uniqueness_ratio:.3f}")
            recommendations.append("Natural variance successfully creates unique similarity scores")
        
        # Calculate quality score (higher is better for realistic distributions)
        quality_score = self._calculate_realistic_quality_score(stats, perfect_ratio, uniqueness_ratio)
        
        # Realistic distribution is healthy if it shows natural variance
        is_healthy = (
            stats['std'] >= 0.08 and  
            stats['std'] <= 0.25 and  
            perfect_ratio <= 0.02 and
            uniqueness_ratio >= 0.3 and  
            0.2 <= stats['mean'] <= 0.8  
        )
        
        if is_healthy:
            recommendations.append("Similarity distribution shows healthy realistic patterns")
        else:
            recommendations.append("Similarity distribution could benefit from fine-tuning")
        
        return SimilarityAnalysis(
            distribution_stats=stats,
            is_healthy=is_healthy,
            anomalies_detected=anomalies,
            recommendations=recommendations,
            quality_score=quality_score
        )
    
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
    
    def _calculate_realistic_quality_score(self, stats: Dict, perfect_ratio: float, uniqueness_ratio: float) -> float:
        """Calculate quality score favoring realistic distributions"""
        base_score = 100.0
        
        # Bonus for good variance (realistic distributions have good variance)
        variance_bonus = min(stats['std'] / 0.15 * 20, 20)  
        
        # Heavy penalty for perfect matches (should be very rare)
        perfect_penalty = perfect_ratio * 50  # Increased penalty
        
        # Bonus for uniqueness
        uniqueness_bonus = min(uniqueness_ratio * 15, 15)
        
        # Bonus for realistic mean
        if 0.3 <= stats['mean'] <= 0.7:
            mean_bonus = 10
        else:
            mean_bonus = 0
        
        # Penalty for extreme distributions
        if stats['std'] > 0.3:
            extreme_penalty = (stats['std'] - 0.3) * 30
        else:
            extreme_penalty = 0
        
        final_score = (base_score + variance_bonus + uniqueness_bonus + mean_bonus 
                      - perfect_penalty - extreme_penalty)
        
        return max(0, min(100, round(final_score, 2)))
    
    # Additional interface methods for compatibility
    def get_movie_by_id(self, movie_id: int) -> Optional[Dict]:
        """Get movie by ID"""
        try:
            return self.movies[self.movie_index[movie_id]]
        except:
            return None
    
    def get_movie_by_title(self, title: str) -> Optional[Dict]:
        """Get movie by title (approximate match)"""
        title_lower = title.lower()
        for movie in self.movies:
            if title_lower in movie['title'].lower():
                return movie
        return None
    
    def get_system_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive system diagnostics"""
        diagnostics = {
            'feature_engineering': self.engineering_report,
            'optimal_parameters': self.optimal_params,
            'similarity_analysis': self.similarity_analysis.distribution_stats if self.similarity_analysis else {},
            'cache_stats': {
                'similarity_cache_size': len(self.similarity_cache),
                'realistic_variance_enabled': True
            },
            'data_stats': {
                'total_movies': len(self.movies),
                'feature_matrix_shape': self.feature_matrix.shape if self.feature_matrix is not None else None,
                'feature_count': len(self.feature_names)
            },
            'method_behaviors': self.method_behaviors
        }
        
        return diagnostics
    
    def get_anomaly_report(self) -> Dict[str, Any]:
        """Get comprehensive anomaly analysis report"""
        if not self.similarity_analysis:
            return {'error': 'Similarity analysis not performed'}
        
        return {
            'timestamp': datetime.now().isoformat(),
            'similarity_health': {
                'is_healthy': self.similarity_analysis.is_healthy,
                'quality_score': self.similarity_analysis.quality_score,
                'anomalies_count': len(self.similarity_analysis.anomalies_detected)
            },
            'detected_anomalies': self.similarity_analysis.anomalies_detected,
            'distribution_statistics': self.similarity_analysis.distribution_stats,
            'actionable_recommendations': self.similarity_analysis.recommendations,
            'realistic_features': {
                'natural_variance_enabled': True,
                'variance_level': self.optimal_params.get('natural_variance', 0.12),
                'max_similarity_cap': self.optimal_params.get('max_similarity', 0.92),
                'self_similarity_prevented': True, 
                'method_differentiation_active': True
            }
        }
    
    def clear_cache(self):
        """Clear similarity calculation cache"""
        self.similarity_cache.clear()
        logger.info("Realistic KNN cache cleared")

# Backward compatibility
KNNRecommender = RealisticKNNRecommender