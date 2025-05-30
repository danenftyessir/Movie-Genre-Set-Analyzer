"""
Statistical Analysis Module
Performs statistical analysis on movie genre data
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from collections import Counter
import scipy.stats as stats

class StatisticalAnalysis:
    """Class for performing statistical analysis on movie genre data"""
    
    def __init__(self, set_operations):
        self.set_ops = set_operations
        self.movies = set_operations.movies
        self.genres = set_operations.genres
    
    def genre_distribution(self) -> Dict[str, int]:
        """Calculate the distribution of movies across genres"""
        distribution = {}
        for genre_name in self.set_ops.genre_sets:
            distribution[genre_name] = len(self.set_ops.get_genre_set(genre_name))
        # Sort by count in descending order
        return dict(sorted(distribution.items(), key=lambda x: x[1], reverse=True))
    
    def genre_popularity_percentages(self) -> Dict[str, float]:
        """Calculate percentage of movies in each genre"""
        total_movies = len(self.movies)
        distribution = self.genre_distribution()
        percentages = {}
        for genre, count in distribution.items():
            percentages[genre] = (count / total_movies) * 100
        return percentages
    
    def average_genres_per_movie(self) -> float:
        """Calculate the average number of genres per movie"""
        total_genres = sum(len(movie.get('genre_ids', [])) for movie in self.movies)
        return total_genres / len(self.movies) if self.movies else 0
    
    def genre_combination_frequency(self, min_count: int = 5) -> List[Tuple[frozenset, int]]:
        """Find the most common genre combinations"""
        combinations = []
        
        for movie in self.movies:
            genre_ids = movie.get('genre_ids', [])
            if genre_ids:
                # Convert genre IDs to names
                genre_names = [self.genres[gid] for gid in genre_ids if gid in self.genres]
                if genre_names:
                    combinations.append(frozenset(genre_names))
        
        # Count occurrences
        combo_counts = Counter(combinations)
        
        # Filter by minimum count and sort
        frequent_combos = [(combo, count) for combo, count in combo_counts.items() 
                          if count >= min_count]
        frequent_combos.sort(key=lambda x: x[1], reverse=True)
        
        return frequent_combos
    
    def genre_correlation_matrix(self) -> pd.DataFrame:
        """
        Create a correlation matrix showing how likely genres appear together
        Uses Phi coefficient for binary variables
        """
        genre_names = list(self.set_ops.genre_sets.keys())
        n = len(genre_names)
        
        # Create binary matrix (movie x genre)
        movie_genre_matrix = []
        
        for movie in self.movies:
            movie_genres = self.set_ops.get_movie_genres(movie['id'])
            row = [1 if genre in movie_genres else 0 for genre in genre_names]
            movie_genre_matrix.append(row)
        
        # Convert to DataFrame
        df = pd.DataFrame(movie_genre_matrix, columns=genre_names)
        
        # Calculate correlation matrix
        correlation_matrix = df.corr()
        
        return correlation_matrix
    
    def genre_entropy(self) -> float:
        """
        Calculate Shannon entropy of genre distribution
        Higher entropy indicates more uniform distribution
        """
        distribution = self.genre_distribution()
        total_movies = sum(distribution.values())
        
        if total_movies == 0:
            return 0.0
        
        entropy = 0
        for count in distribution.values():
            if count > 0:
                p = count / total_movies
                entropy -= p * np.log2(p)
        
        return entropy
    
    def genre_gini_coefficient(self) -> float:
        """
        Calculate Gini coefficient for genre distribution
        Measures inequality in genre distribution (0=perfect equality, 1=perfect inequality)
        """
        distribution = list(self.genre_distribution().values())
        distribution.sort()
        
        n = len(distribution)
        index = np.arange(1, n + 1)
        
        return (2 * np.sum(index * distribution)) / (n * np.sum(distribution)) - (n + 1) / n
    
    def genre_statistics_summary(self) -> Dict:
        """Generate comprehensive statistics summary"""
        distribution = self.genre_distribution()
        counts = list(distribution.values())
        
        summary = {
            'total_movies': len(self.movies),
            'total_genres': len(self.genres),
            'average_genres_per_movie': self.average_genres_per_movie(),
            'genre_entropy': self.genre_entropy(),
            'genre_gini_coefficient': self.genre_gini_coefficient(),
            'most_popular_genre': max(distribution, key=distribution.get) if distribution else None,
            'least_popular_genre': min(distribution, key=distribution.get) if distribution else None,
            'genre_count_statistics': {
                'mean': np.mean(counts) if counts else 0,
                'median': np.median(counts) if counts else 0,
                'std': np.std(counts) if counts else 0,
                'min': min(counts) if counts else 0,
                'max': max(counts) if counts else 0
            }
        }
        
        return summary
    
    def chi_square_independence_test(self, genre_a: str, genre_b: str) -> Dict:
        """
        Perform chi-square test for independence between two genres
        Tests whether the occurrence of one genre is independent of another
        """
        # Create contingency table
        set_a = self.set_ops.get_genre_set(genre_a)
        set_b = self.set_ops.get_genre_set(genre_b)
        all_movies = set(movie['id'] for movie in self.movies)
        
        # Count occurrences
        both = len(set_a & set_b)
        only_a = len(set_a - set_b)
        only_b = len(set_b - set_a)
        neither = len(all_movies - (set_a | set_b))
        
        # Create contingency table
        contingency_table = np.array([[both, only_b], [only_a, neither]])
        
        # Perform chi-square test
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        return {
            'chi2_statistic': chi2,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'expected_frequencies': expected.tolist(),
            'observed_frequencies': contingency_table.tolist(),
            'independent': p_value > 0.05  # Common significance level
        }
    
    def genre_exclusive_probability(self) -> Dict[str, float]:
        """
        Calculate probability that a movie belongs exclusively to each genre
        """
        probabilities = {}
        
        for genre_name in self.set_ops.genre_sets:
            exclusive_movies = self.set_ops.exclusive_genre_movies(genre_name)
            total_in_genre = len(self.set_ops.get_genre_set(genre_name))
            
            if total_in_genre > 0:
                probabilities[genre_name] = len(exclusive_movies) / total_in_genre
            else:
                probabilities[genre_name] = 0.0
        
        return dict(sorted(probabilities.items(), key=lambda x: x[1], reverse=True))