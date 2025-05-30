"""
Visualization Module
Creates various charts and plots for movie genre analysis with focus on KNN
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set
import plotly.graph_objects as go
import plotly.express as px
from matplotlib_venn import venn2, venn3
import os

class Visualizer:
    """Class for creating visualizations of movie genre data"""
    
    def __init__(self, output_dir: str = 'results/plots'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
    
    def save_plot(self, filename: str, dpi: int = 300):
        """Save the current plot to file"""
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to {filepath}")
    
    def plot_genre_distribution(self, distribution: Dict[str, int], top_n: int = 15):
        """Create a bar plot of genre distribution"""
        # Get top N genres
        sorted_genres = list(distribution.items())[:top_n]
        genres, counts = zip(*sorted_genres)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create bars
        bars = ax.bar(range(len(genres)), counts)
        
        # Color gradient
        colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Customize plot
        ax.set_xticks(range(len(genres)))
        ax.set_xticklabels(genres, rotation=45, ha='right')
        ax.set_xlabel('Genre', fontsize=12)
        ax.set_ylabel('Number of Movies', fontsize=12)
        ax.set_title('Movie Distribution by Genre', fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for i, (genre, count) in enumerate(zip(genres, counts)):
            ax.text(i, count + 5, str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        self.save_plot('genre_distribution.png')
    
    def plot_similarity_heatmap(self, similarity_matrix: pd.DataFrame, 
                                title: str = "Genre Similarity Heatmap"):
        """Create a heatmap of genre similarities"""
        plt.figure(figsize=(14, 12))
        
        # Create heatmap
        sns.heatmap(similarity_matrix, 
                    annot=True, 
                    fmt='.2f', 
                    cmap='YlOrRd',
                    square=True,
                    cbar_kws={'label': 'Similarity Score'},
                    annot_kws={'size': 8})
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Genre', fontsize=12)
        plt.ylabel('Genre', fontsize=12)
        
        plt.tight_layout()
        self.save_plot('similarity_heatmap.png')
    
    def plot_venn_diagram(self, set_ops, genre_a: str, genre_b: str, 
                          genre_c: str = None):
        """Create Venn diagram for 2 or 3 genres"""
        plt.figure(figsize=(10, 8))
        
        set_a = set_ops.get_genre_set(genre_a)
        set_b = set_ops.get_genre_set(genre_b)
        
        if genre_c:
            set_c = set_ops.get_genre_set(genre_c)
            venn = venn3([set_a, set_b, set_c], 
                         (genre_a, genre_b, genre_c))
            title = f"Venn Diagram: {genre_a}, {genre_b}, {genre_c}"
            filename = f"venn_diagram_3_genres.png"
        else:
            venn = venn2([set_a, set_b], 
                         (genre_a, genre_b))
            title = f"Venn Diagram: {genre_a} vs {genre_b}"
            filename = f"venn_diagram_2_genres.png"
        
        plt.title(title, fontsize=14, fontweight='bold')
        
        self.save_plot(filename)
    
    def plot_genre_correlation_matrix(self, correlation_matrix: pd.DataFrame):
        """Create a correlation matrix plot"""
        plt.figure(figsize=(16, 14))
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        # Create heatmap
        sns.heatmap(correlation_matrix,
                    mask=mask,
                    annot=True,
                    fmt='.2f',
                    cmap='coolwarm',
                    center=0,
                    square=True,
                    linewidths=0.5,
                    cbar_kws={"shrink": 0.8, 'label': 'Correlation'},
                    annot_kws={'size': 8})
        
        plt.title('Genre Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        self.save_plot('genre_correlation_matrix.png')
    
    def plot_genre_combinations(self, combinations: List[Tuple[frozenset, int]], 
                                top_n: int = 10):
        """Plot most common genre combinations"""
        # Prepare data
        combo_labels = []
        counts = []
        
        for combo, count in combinations[:top_n]:
            label = ' & '.join(sorted(combo))
            combo_labels.append(label)
            counts.append(count)
        
        # Create horizontal bar plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        y_pos = np.arange(len(combo_labels))
        bars = ax.barh(y_pos, counts)
        
        # Color gradient
        colors = plt.cm.plasma(np.linspace(0, 1, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Customize plot
        ax.set_yticks(y_pos)
        ax.set_yticklabels(combo_labels)
        ax.set_xlabel('Number of Movies', fontsize=12)
        ax.set_title('Most Common Genre Combinations', fontsize=14, fontweight='bold')
        
        # Add value labels
        for i, (label, count) in enumerate(zip(combo_labels, counts)):
            ax.text(count + 1, i, str(count), va='center')
        
        plt.tight_layout()
        self.save_plot('genre_combinations.png')
    
    def plot_knn_recommendations(self, target_movie: Dict, 
                                 recommendations: List[Tuple[Dict, float]]):
        """FIXED: Visualize KNN recommendations for a single target movie"""
        if not recommendations:
            print("⚠️  No recommendations to plot")
            return
        
        # Prepare data
        titles = [target_movie['title']] + [r[0]['title'] for r in recommendations]
        similarities = [1.0] + [r[1] for r in recommendations]
        
        # Truncate long titles
        titles = [title[:30] + '...' if len(title) > 30 else title for title in titles]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        y_pos = np.arange(len(titles))
        bars = ax.barh(y_pos, similarities)
        
        # Color the target movie differently
        bars[0].set_color('#FF6B6B')
        bars[0].set_label('Target Movie')
        
        # Color recommendations with gradient
        if len(bars) > 1:
            colors = plt.cm.Blues(np.linspace(0.4, 1, len(bars)-1))
            for i, (bar, color) in enumerate(zip(bars[1:], colors)):
                bar.set_color(color)
            bars[1].set_label('Recommendations')
        
        # Customize plot
        ax.set_yticks(y_pos)
        ax.set_yticklabels(titles)
        ax.set_xlabel('Similarity Score', fontsize=12)
        ax.set_title(f'KNN Movie Recommendations for "{target_movie["title"]}"', 
                     fontsize=14, fontweight='bold')
        ax.legend()
        ax.set_xlim(0, 1.1)
        
        # Add value labels
        for i, (title, sim) in enumerate(zip(titles, similarities)):
            ax.text(sim + 0.02, i, f'{sim:.3f}', va='center', fontweight='bold')
        
        plt.tight_layout()
        self.save_plot('knn_recommendations.png')
        print("✅ KNN recommendations plot saved")
    
    def plot_multiple_knn_examples(self, knn):
        """FIXED: Create a comparison of KNN recommendations for multiple target movies"""
        # Select diverse target movies
        target_movies = []
        target_titles = ["The Dark Knight", "Toy Story", "The Matrix", "Titanic", "Inception"]
        
        for title in target_titles:
            movie = knn.get_movie_by_title(title)
            if movie:
                target_movies.append(movie)
        
        # If we don't have enough, add some random ones
        while len(target_movies) < 3:
            if len(target_movies) * 100 < len(knn.movies):
                target_movies.append(knn.movies[len(target_movies) * 100])
            else:
                break
        
        if not target_movies:
            print("⚠️  No target movies found for multiple KNN examples")
            return
        
        # Create subplots
        fig, axes = plt.subplots(len(target_movies), 1, figsize=(14, 4*len(target_movies)))
        if len(target_movies) == 1:
            axes = [axes]
        
        fig.suptitle('KNN Recommendations for Multiple Movies', fontsize=16, fontweight='bold')
        
        for idx, (target_movie, ax) in enumerate(zip(target_movies, axes)):
            # Get recommendations
            similar_movies = knn.find_k_nearest_neighbors(target_movie['id'], k=3)
            
            if not similar_movies:
                ax.text(0.5, 0.5, f'No recommendations found for\n{target_movie["title"]}', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(f'Target: {target_movie["title"]}')
                continue
            
            # Prepare data
            titles = [target_movie['title']]
            similarities = [1.0]
            
            for movie_sim in similar_movies:
                movie = knn.get_movie_by_id(movie_sim.movie_id)
                if movie:
                    title = movie['title']
                    if len(title) > 25:
                        title = title[:25] + '...'
                    titles.append(title)
                    similarities.append(movie_sim.similarity)
            
            # Create horizontal bar plot
            y_pos = np.arange(len(titles))
            bars = ax.barh(y_pos, similarities)
            
            # Color target movie differently
            bars[0].set_color('#FF6B6B')
            
            # Color recommendations
            for bar in bars[1:]:
                bar.set_color('#4ECDC4')
            
            # Customize subplot
            ax.set_yticks(y_pos)
            ax.set_yticklabels(titles)
            ax.set_xlabel('Similarity Score')
            ax.set_title(f'Target: {target_movie["title"]}')
            ax.set_xlim(0, 1.1)
            
            # Add value labels
            for i, sim in enumerate(similarities):
                ax.text(sim + 0.02, i, f'{sim:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        self.save_plot('multiple_knn_examples.png')
        print("✅ Multiple KNN examples plot saved")
    
    def plot_knn_performance_comparison(self, knn):
        """FIXED: Compare KNN performance with different similarity measures"""
        # Get a test movie
        target_movie = knn.get_movie_by_title("The Dark Knight")
        if not target_movie:
            target_movie = knn.movies[0] if knn.movies else None
        
        if not target_movie:
            print("⚠️  No target movie found for performance comparison")
            return
        
        # Test different approaches
        methods = {}
        
        try:
            methods['Content-Based (Jaccard)'] = knn.find_k_nearest_neighbors(target_movie['id'], k=5)
        except:
            methods['Content-Based (Jaccard)'] = []
        
        try:
            methods['Content-Based (Min Similarity)'] = knn.content_based_filtering(target_movie['id'], k=5, min_similarity=0.1)
        except:
            methods['Content-Based (Min Similarity)'] = []
        
        try:
            methods['Diverse Recommendations'] = knn.find_diverse_recommendations(target_movie['id'], k=5)
        except:
            methods['Diverse Recommendations'] = []
        
        # Create comparison plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'KNN Performance Comparison for "{target_movie["title"]}"', 
                     fontsize=16, fontweight='bold')
        
        for idx, (method_name, recommendations) in enumerate(methods.items()):
            ax = axes[idx]
            
            if recommendations:
                titles = []
                similarities = []
                
                for movie_sim in recommendations:
                    movie = knn.get_movie_by_id(movie_sim.movie_id)
                    if movie:
                        title = movie['title']
                        if len(title) > 20:
                            title = title[:20] + '...'
                        titles.append(title)
                        similarities.append(movie_sim.similarity)
                
                if titles and similarities:
                    # Create bar plot
                    y_pos = np.arange(len(titles))
                    bars = ax.barh(y_pos, similarities)
                    
                    # Color gradient
                    colors = plt.cm.viridis(np.linspace(0.3, 1, len(bars)))
                    for bar, color in zip(bars, colors):
                        bar.set_color(color)
                    
                    # Customize subplot
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(titles, fontsize=9)
                    ax.set_xlabel('Similarity Score')
                    ax.set_title(method_name)
                    ax.set_xlim(0, 1.1)
                    
                    # Add value labels
                    for i, sim in enumerate(similarities):
                        ax.text(sim + 0.02, i, f'{sim:.2f}', va='center', fontsize=8)
                else:
                    ax.text(0.5, 0.5, 'No valid\nrecommendations', ha='center', va='center', 
                            transform=ax.transAxes, fontsize=12)
                    ax.set_title(method_name)
            else:
                ax.text(0.5, 0.5, 'No recommendations\nfound', ha='center', va='center', 
                        transform=ax.transAxes, fontsize=12)
                ax.set_title(method_name)
        
        plt.tight_layout()
        self.save_plot('knn_performance_comparison.png')
        print("✅ KNN performance comparison plot saved")
    
    def plot_knn_genre_analysis(self, knn, target_movie_title: str = "The Dark Knight"):
        """FIXED: Analyze genre patterns in KNN recommendations"""
        target_movie = knn.get_movie_by_title(target_movie_title)
        if not target_movie:
            target_movie = knn.movies[0] if knn.movies else None
        
        if not target_movie:
            print("⚠️  No target movie found for genre analysis")
            return
        
        # Get recommendations
        recommendations = knn.find_k_nearest_neighbors(target_movie['id'], k=10)
        
        if not recommendations:
            print("⚠️  No recommendations found for genre analysis")
            return
        
        # Analyze genre patterns
        target_genres = knn.set_ops.get_movie_genres(target_movie['id'])
        
        genre_coverage = {}
        shared_genre_counts = {}
        
        for movie_sim in recommendations:
            movie_genres = knn.set_ops.get_movie_genres(movie_sim.movie_id)
            
            # Count shared genres
            shared = len(target_genres & movie_genres)
            if shared in shared_genre_counts:
                shared_genre_counts[shared] += 1
            else:
                shared_genre_counts[shared] = 1
            
            # Count genre coverage
            for genre in movie_genres:
                if genre in genre_coverage:
                    genre_coverage[genre] += 1
                else:
                    genre_coverage[genre] = 1
        
        # Create subplot figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'Genre Analysis for KNN Recommendations: "{target_movie["title"]}"', 
                     fontsize=14, fontweight='bold')
        
        # Plot 1: Genre coverage in recommendations
        if genre_coverage:
            genres = list(genre_coverage.keys())
            counts = list(genre_coverage.values())
            
            # Highlight target genres
            colors = ['#FF6B6B' if genre in target_genres else '#4ECDC4' for genre in genres]
            
            bars1 = ax1.bar(range(len(genres)), counts, color=colors)
            ax1.set_xticks(range(len(genres)))
            ax1.set_xticklabels(genres, rotation=45, ha='right')
            ax1.set_ylabel('Frequency in Recommendations')
            ax1.set_title('Genre Coverage in Recommendations')
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='#FF6B6B', label='Target Movie Genres'),
                             Patch(facecolor='#4ECDC4', label='Other Genres')]
            ax1.legend(handles=legend_elements)
            
            # Add value labels
            for bar, count in zip(bars1, counts):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                         str(count), ha='center', va='bottom')
        else:
            ax1.text(0.5, 0.5, 'No genre data available', ha='center', va='center', 
                    transform=ax1.transAxes, fontsize=12)
            ax1.set_title('Genre Coverage in Recommendations')
        
        # Plot 2: Shared genre distribution
        if shared_genre_counts:
            shared_counts = sorted(shared_genre_counts.items())
            shared_nums, frequencies = zip(*shared_counts)
            
            bars2 = ax2.bar(shared_nums, frequencies, color='#95E1D3')
            ax2.set_xlabel('Number of Shared Genres')
            ax2.set_ylabel('Number of Recommended Movies')
            ax2.set_title('Distribution of Shared Genres')
            ax2.set_xticks(shared_nums)
            
            # Add value labels
            for bar, freq in zip(bars2, frequencies):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                         str(freq), ha='center', va='bottom')
        else:
            ax2.text(0.5, 0.5, 'No shared genre data', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Distribution of Shared Genres')
        
        plt.tight_layout()
        self.save_plot('knn_genre_analysis.png')
        print("✅ KNN genre analysis plot saved")
    
    def plot_knn_similarity_distribution(self, knn):
        """FIXED: Plot distribution of similarity scores in KNN results"""
        # Get several target movies
        target_movies = []
        sample_titles = ["The Dark Knight", "Toy Story", "The Matrix", "Titanic"]
        
        for title in sample_titles:
            movie = knn.get_movie_by_title(title)
            if movie:
                target_movies.append(movie)
        
        if not target_movies:
            print("⚠️  No target movies found for similarity distribution")
            return
        
        # Collect similarity scores
        all_similarities = []
        movie_similarities = {}
        
        for target_movie in target_movies:
            recommendations = knn.find_k_nearest_neighbors(target_movie['id'], k=20)
            similarities = [r.similarity for r in recommendations if r.similarity > 0]
            if similarities:
                all_similarities.extend(similarities)
                movie_similarities[target_movie['title']] = similarities
        
        if not all_similarities:
            print("⚠️  No similarity data found")
            return
        
        # Create subplot figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('KNN Similarity Score Analysis', fontsize=14, fontweight='bold')
        
        # Plot 1: Overall distribution
        ax1.hist(all_similarities, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Similarity Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Overall Distribution of Similarity Scores')
        ax1.axvline(np.mean(all_similarities), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(all_similarities):.3f}')
        ax1.legend()
        
        # Plot 2: Box plot by movie
        if movie_similarities:
            movie_names = list(movie_similarities.keys())
            similarity_data = [movie_similarities[name] for name in movie_names]
            
            bp = ax2.boxplot(similarity_data, labels=movie_names, patch_artist=True)
            
            # Color the boxes
            colors = plt.cm.Set3(np.linspace(0, 1, len(bp['boxes'])))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            ax2.set_xticklabels(movie_names, rotation=45, ha='right')
            ax2.set_ylabel('Similarity Score')
            ax2.set_title('Similarity Score Distribution by Target Movie')
        else:
            ax2.text(0.5, 0.5, 'No similarity data\navailable', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Similarity Score Distribution by Target Movie')
        
        plt.tight_layout()
        self.save_plot('knn_similarity_distribution.png')
        print("✅ KNN similarity distribution plot saved")
    
    def plot_genre_statistics_dashboard(self, stats: Dict):
        """Create a dashboard with multiple statistics plots"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Genre Statistics Dashboard', fontsize=16, fontweight='bold')
        
        # Plot 1: Genre count distribution
        ax1 = axes[0, 0]
        genre_stats = stats['genre_count_statistics']
        metrics = ['Mean', 'Median', 'Std Dev', 'Min', 'Max']
        values = [genre_stats['mean'], genre_stats['median'], 
                  genre_stats['std'], genre_stats['min'], genre_stats['max']]
        
        bars = ax1.bar(metrics, values)
        ax1.set_title('Genre Count Statistics')
        ax1.set_ylabel('Value')
        
        # Add value labels
        for bar, value in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{value:.1f}', ha='center', va='bottom')
        
        # Plot 2: Pie chart of top genres
        ax2 = axes[0, 1]
        ax2.text(0.5, 0.5, 'Genre Distribution\n(Pie Chart)', 
                 ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Top Genres Distribution')
        
        # Plot 3: Metrics
        ax3 = axes[1, 0]
        metrics = ['Total Movies', 'Total Genres', 'Avg Genres/Movie', 'Entropy']
        values = [stats['total_movies'], stats['total_genres'], 
                  stats['average_genres_per_movie'], stats['genre_entropy']]
        
        bars = ax3.bar(metrics, values)
        ax3.set_title('Key Metrics')
        ax3.set_ylabel('Value')
        
        # Rotate x labels
        ax3.set_xticklabels(metrics, rotation=45, ha='right')
        
        # Plot 4: Text summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        summary_text = f"""
        Summary Statistics:
        
        • Most Popular Genre: {stats['most_popular_genre']}
        • Least Popular Genre: {stats['least_popular_genre']}
        • Genre Gini Coefficient: {stats['genre_gini_coefficient']:.3f}
        • Average Genres per Movie: {stats['average_genres_per_movie']:.2f}
        """
        ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes,
                 fontsize=12, verticalalignment='center')
        
        plt.tight_layout()
        self.save_plot('genre_statistics_dashboard.png')