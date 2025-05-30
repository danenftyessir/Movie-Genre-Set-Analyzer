"""
Data Quality Validator & Cleaner
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Set, Tuple, Optional, Any
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
import json
import warnings
import hashlib
import re
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class EnhancedDataQualityReport:
    """Enhanced comprehensive data quality assessment report"""
    is_valid: bool
    critical_issues: List[str]
    warnings: List[str]
    statistics: Dict[str, Any]
    recommendations: List[str]
    quality_score: float
    duplicate_analysis: Dict[str, Any]
    missing_value_analysis: Dict[str, Any]
    consistency_analysis: Dict[str, Any]    
    true_resolution_status: Dict[str, Any] = None
    validation_metrics: Dict[str, float] = None
    
    def __post_init__(self):
        """Post-initialization to set default values"""
        if self.true_resolution_status is None:
            self.true_resolution_status = {
                'duplicates_resolved': False,
                'missing_values_handled': False,
                'referential_integrity_fixed': False,
                'data_consistency_achieved': False
            }
        
        if self.validation_metrics is None:
            self.validation_metrics = {
                'duplicate_ratio': 0.0,
                'missing_value_ratio': 0.0,
                'referential_integrity_score': 0.0,
                'consistency_score': 0.0
            }
    
    def to_dict(self) -> Dict:
        return {
            'validation_timestamp': datetime.now().isoformat(),
            'overall_quality_score': self.quality_score,
            'is_data_valid': self.is_valid,
            'critical_issues_count': len(self.critical_issues),
            'warnings_count': len(self.warnings),
            'critical_issues': self.critical_issues,
            'warnings': self.warnings,
            'detailed_statistics': self.statistics,
            'actionable_recommendations': self.recommendations,
            'duplicate_analysis': self.duplicate_analysis,
            'missing_value_analysis': self.missing_value_analysis,
            'consistency_analysis': self.consistency_analysis,
            'true_resolution_status': self.true_resolution_status,
            'validation_metrics': self.validation_metrics
        }

class EnhancedDataQualityValidator:
    """
    Enhanced Data Quality Validator & Cleaner
    """
    
    def __init__(self, strict_mode: bool = True, auto_fix: bool = True):
        self.strict_mode = strict_mode
        self.auto_fix = auto_fix
        self.validation_rules = self._initialize_enhanced_validation_rules()
        self.cleaning_stats = defaultdict(int)
        self.duplicate_threshold = 0.02  # 2% max duplicates (stricter)
        self.missing_value_threshold = 0.05  # 5% max missing values (stricter)
        self.consistency_threshold = 0.95  # 95% min consistency
        self.resolution_metrics = {
            'original_data_quality': 0.0,
            'final_data_quality': 0.0,
            'improvement_achieved': 0.0
        }
        
        logger.info(f"Enhanced Data Quality Validator initialized - Strict: {strict_mode}, Auto-fix: {auto_fix}")
    
    def _initialize_enhanced_validation_rules(self) -> Dict[str, Dict]:
        """Initialize enhanced validation rules with stricter criteria"""
        return {
            'movies': {
                'required_fields': ['id', 'title'],
                'field_types': {'id': int, 'title': str, 'genre_ids': list},
                'field_constraints': {
                    'id': lambda x: isinstance(x, int) and x > 0,
                    'title': lambda x: isinstance(x, str) and len(x.strip()) > 0,
                    'genre_ids': lambda x: isinstance(x, list) and all(isinstance(gid, int) and gid > 0 for gid in x)
                },
                'uniqueness_fields': ['id'],
                'max_missing_ratio': 0.05,
                'title_min_length': 1,
                'title_max_length': 500,
                'max_genres_per_movie': 10
            },
            'genres': {
                'required_fields': ['id', 'name'],
                'field_types': {'id': int, 'name': str},
                'field_constraints': {
                    'id': lambda x: isinstance(x, int) and x > 0,
                    'name': lambda x: isinstance(x, str) and len(x.strip()) > 0
                },
                'uniqueness_fields': ['id', 'name'],
                'max_missing_ratio': 0.00,  # Genres should have no missing data
                'name_min_length': 2,
                'name_max_length': 50
            }
        }
    
    def validate_and_clean_data(self, movies: List[Dict], genres: List[Dict]) -> Tuple[List[Dict], List[Dict], EnhancedDataQualityReport]:
        """
        Returns: (cleaned_movies, cleaned_genres, enhanced_quality_report)
        """
        logger.info("🔍 Starting ENHANCED comprehensive data quality validation and cleaning...")
        
        # Initialize tracking
        original_movie_count = len(movies)
        original_genre_count = len(genres)
        
        # Phase 1: Pre-cleaning Analysis
        logger.info("Phase 1: Pre-cleaning comprehensive analysis...")
        pre_cleaning_metrics = self._calculate_pre_cleaning_metrics(movies, genres)
        self.resolution_metrics['original_data_quality'] = pre_cleaning_metrics['overall_quality']
        
        # Phase 2: Enhanced Critical Issues Detection
        logger.info("Phase 2: Enhanced critical data quality issues detection...")
        critical_issues = []
        warnings_list = []
        
        # Phase 3: Deep Duplicate Analysis with Advanced Detection
        logger.info("Phase 3: Advanced duplicate analysis...")
        duplicate_report = self._analyze_duplicates_enhanced(movies, genres)
        if duplicate_report['movies']['duplicate_ratio'] > self.duplicate_threshold:
            critical_issues.append(f"Movie duplicate ratio {duplicate_report['movies']['duplicate_ratio']:.3f} exceeds strict threshold {self.duplicate_threshold}")
        if duplicate_report['genres']['duplicate_ratio'] > self.duplicate_threshold:
            critical_issues.append(f"Genre duplicate ratio {duplicate_report['genres']['duplicate_ratio']:.3f} exceeds strict threshold {self.duplicate_threshold}")
        
        # Phase 4: Enhanced Missing Value Analysis
        logger.info("Phase 4: Enhanced missing value analysis...")
        missing_report = self._analyze_missing_values_enhanced(movies, genres)
        for entity_type, analysis in missing_report.items():
            if analysis['overall_missing_ratio'] > self.missing_value_threshold:
                critical_issues.append(f"{entity_type.title()} missing value ratio {analysis['overall_missing_ratio']:.3f} exceeds strict threshold {self.missing_value_threshold}")
        
        # Phase 5: Advanced Data Consistency Analysis
        logger.info("Phase 5: Advanced data consistency analysis...")
        consistency_report = self._analyze_consistency_enhanced(movies, genres)
        if consistency_report['overall_consistency_score'] < self.consistency_threshold:
            critical_issues.append(f"Data consistency score {consistency_report['overall_consistency_score']:.3f} below strict threshold {self.consistency_threshold}")
        
        # Phase 6: CRITICAL - Advanced Data Cleaning
        cleaned_movies = movies.copy()
        cleaned_genres = genres.copy()
        
        if self.auto_fix:
            logger.info("Phase 6: ADVANCED data cleaning with comprehensive fixes...")
            cleaned_movies = self._clean_movies_enhanced(cleaned_movies, genres)
            cleaned_genres = self._clean_genres_enhanced(cleaned_genres)
            logger.info("Phase 7: Post-cleaning validation...")
            post_clean_duplicate_report = self._analyze_duplicates_enhanced(cleaned_movies, cleaned_genres)
            post_clean_missing_report = self._analyze_missing_values_enhanced(cleaned_movies, cleaned_genres)
            post_clean_consistency_report = self._analyze_consistency_enhanced(cleaned_movies, cleaned_genres)
            duplicate_resolved = (
                post_clean_duplicate_report['movies']['duplicate_ratio'] <= self.duplicate_threshold and
                post_clean_duplicate_report['genres']['duplicate_ratio'] <= self.duplicate_threshold
            )
            
            missing_resolved = all(
                analysis['overall_missing_ratio'] <= self.missing_value_threshold
                for analysis in post_clean_missing_report.values()
            )
            
            consistency_resolved = post_clean_consistency_report['overall_consistency_score'] >= self.consistency_threshold
            
        else:
            post_clean_duplicate_report = duplicate_report
            post_clean_missing_report = missing_report
            post_clean_consistency_report = consistency_report
            duplicate_resolved = False
            missing_resolved = False
            consistency_resolved = False
        
        # Phase 8: Enhanced Statistics Generation
        logger.info("Phase 8: Enhanced comprehensive statistics generation...")
        statistics = self._generate_enhanced_statistics(
            cleaned_movies, cleaned_genres, 
            original_movie_count, original_genre_count,
            duplicate_report, missing_report, consistency_report,
            post_clean_duplicate_report, post_clean_missing_report, post_clean_consistency_report
        )
        
        # Phase 9: Enhanced Quality Score Calculation
        quality_score = self._calculate_enhanced_quality_score(
            post_clean_duplicate_report, 
            post_clean_missing_report, 
            post_clean_consistency_report
        )
        
        self.resolution_metrics['final_data_quality'] = quality_score
        self.resolution_metrics['improvement_achieved'] = quality_score - self.resolution_metrics['original_data_quality']
        
        # Phase 10: Enhanced Recommendations
        recommendations = self._generate_enhanced_recommendations(
            critical_issues, warnings_list, statistics, quality_score,
            duplicate_resolved, missing_resolved, consistency_resolved
        )
        
        true_resolution_status = {
            'duplicates_resolved': duplicate_resolved,
            'missing_values_handled': missing_resolved,
            'referential_integrity_fixed': consistency_resolved,
            'data_consistency_achieved': consistency_resolved,
            'overall_resolution_achieved': duplicate_resolved and missing_resolved and consistency_resolved
        }
        
        validation_metrics = {
            'duplicate_ratio': post_clean_duplicate_report['movies']['duplicate_ratio'],
            'missing_value_ratio': np.mean([analysis['overall_missing_ratio'] for analysis in post_clean_missing_report.values()]),
            'referential_integrity_score': post_clean_consistency_report['referential_integrity_score'],
            'consistency_score': post_clean_consistency_report['overall_consistency_score']
        }
        
        is_valid = (
            len(critical_issues) == 0 and 
            quality_score >= 85.0 and
            true_resolution_status['overall_resolution_achieved']
        )
        
        report = EnhancedDataQualityReport(
            is_valid=is_valid,
            critical_issues=critical_issues,
            warnings=warnings_list,
            statistics=statistics,
            recommendations=recommendations,
            quality_score=quality_score,
            duplicate_analysis=post_clean_duplicate_report,
            missing_value_analysis=post_clean_missing_report,
            consistency_analysis=post_clean_consistency_report,
            true_resolution_status=true_resolution_status,
            validation_metrics=validation_metrics
        )
        
        logger.info(f"✅ Enhanced data quality validation complete - Score: {quality_score:.1f}/100, Valid: {is_valid}")
        logger.info(f"📊 Movies: {original_movie_count} → {len(cleaned_movies)} ({len(cleaned_movies)/original_movie_count*100:.1f}% retained)")
        logger.info(f"📊 Genres: {original_genre_count} → {len(cleaned_genres)} ({len(cleaned_genres)/original_genre_count*100:.1f}% retained)")
        logger.info(f"🎯 TRUE RESOLUTION STATUS:")
        logger.info(f"   • Duplicates: {'✅ RESOLVED' if duplicate_resolved else '❌ NOT RESOLVED'}")
        logger.info(f"   • Missing Values: {'✅ RESOLVED' if missing_resolved else '❌ NOT RESOLVED'}")
        logger.info(f"   • Consistency: {'✅ RESOLVED' if consistency_resolved else '❌ NOT RESOLVED'}")
        logger.info(f"   • Overall: {'✅ FULLY RESOLVED' if true_resolution_status['overall_resolution_achieved'] else '❌ PARTIAL RESOLUTION'}")
        
        return cleaned_movies, cleaned_genres, report
    
    def _calculate_pre_cleaning_metrics(self, movies: List[Dict], genres: List[Dict]) -> Dict[str, float]:
        """Calculate metrics before cleaning for comparison"""
        duplicate_analysis = self._analyze_duplicates_enhanced(movies, genres)
        missing_analysis = self._analyze_missing_values_enhanced(movies, genres)
        consistency_analysis = self._analyze_consistency_enhanced(movies, genres)
        
        overall_quality = self._calculate_enhanced_quality_score(
            duplicate_analysis, missing_analysis, consistency_analysis
        )
        
        return {
            'overall_quality': overall_quality,
            'duplicate_ratio': duplicate_analysis['movies']['duplicate_ratio'],
            'missing_ratio': np.mean([a['overall_missing_ratio'] for a in missing_analysis.values()]),
            'consistency_score': consistency_analysis['overall_consistency_score']
        }
    
    def _analyze_duplicates_enhanced(self, movies: List[Dict], genres: List[Dict]) -> Dict[str, Any]:
        """Enhanced comprehensive duplicate analysis"""
        analysis = {}
        
        # Enhanced movie duplicate analysis
        movie_ids = [m.get('id') for m in movies if 'id' in m and isinstance(m['id'], int)]
        movie_id_duplicates = [mid for mid, count in Counter(movie_ids).items() if count > 1]
        
        movie_titles = [m.get('title', '').strip().lower() for m in movies if 'title' in m]
        movie_title_duplicates = [title for title, count in Counter(movie_titles).items() if count > 1 and title]
        
        content_based_duplicates = self._find_content_based_duplicates(movies)
        near_duplicate_titles = self._find_near_duplicate_titles(movies)
        
        analysis['movies'] = {
            'total_count': len(movies),
            'id_duplicates': len(movie_id_duplicates),
            'title_duplicates': len(movie_title_duplicates),
            'content_based_duplicates': len(content_based_duplicates),
            'near_duplicate_titles': len(near_duplicate_titles),
            'total_duplicate_issues': len(movie_id_duplicates) + len(movie_title_duplicates) + len(content_based_duplicates),
            'duplicate_ratio': (len(movie_id_duplicates) + len(movie_title_duplicates) + len(content_based_duplicates)) / max(len(movies), 1),
            'duplicate_ids': movie_id_duplicates[:10],
            'duplicate_titles': movie_title_duplicates[:10]
        }
        
        # Enhanced genre duplicate analysis
        genre_ids = [g.get('id') for g in genres if 'id' in g and isinstance(g['id'], int)]
        genre_id_duplicates = [gid for gid, count in Counter(genre_ids).items() if count > 1]
        
        genre_names = [g.get('name', '').strip().lower() for g in genres if 'name' in g]
        genre_name_duplicates = [name for name, count in Counter(genre_names).items() if count > 1 and name]
        
        # Check for semantically similar genre names
        similar_genre_names = self._find_similar_genre_names(genres)
        
        analysis['genres'] = {
            'total_count': len(genres),
            'id_duplicates': len(genre_id_duplicates),
            'name_duplicates': len(genre_name_duplicates),
            'similar_names': len(similar_genre_names),
            'total_duplicate_issues': len(genre_id_duplicates) + len(genre_name_duplicates) + len(similar_genre_names),
            'duplicate_ratio': (len(genre_id_duplicates) + len(genre_name_duplicates) + len(similar_genre_names)) / max(len(genres), 1),
            'duplicate_ids': genre_id_duplicates,
            'duplicate_names': genre_name_duplicates
        }
        
        return analysis
    
    def _find_content_based_duplicates(self, movies: List[Dict]) -> List[Tuple[Dict, Dict]]:
        """Find movies that are duplicates based on content similarity"""
        duplicates = []
        processed_indices = set()
        
        for i, movie_a in enumerate(movies):
            if i in processed_indices:
                continue
                
            title_a = movie_a.get('title', '').strip().lower()
            genres_a = set(movie_a.get('genre_ids', []))
            
            for j, movie_b in enumerate(movies[i+1:], i+1):
                if j in processed_indices:
                    continue
                
                title_b = movie_b.get('title', '').strip().lower()
                genres_b = set(movie_b.get('genre_ids', []))
                
                # Check for high similarity
                title_similarity = self._calculate_title_similarity(title_a, title_b)
                genre_similarity = len(genres_a & genres_b) / len(genres_a | genres_b) if (genres_a | genres_b) else 0
                
                # If both title and genres are very similar, likely duplicate
                if title_similarity > 0.8 and genre_similarity > 0.8:
                    duplicates.append((movie_a, movie_b))
                    processed_indices.add(j)
        
        return duplicates
    
    def _find_near_duplicate_titles(self, movies: List[Dict]) -> List[Tuple[str, str]]:
        """Find titles that are very similar but not exact duplicates"""
        near_duplicates = []
        titles = [(movie.get('title', ''), i) for i, movie in enumerate(movies)]
        
        for i, (title_a, idx_a) in enumerate(titles):
            for title_b, idx_b in titles[i+1:]:
                if idx_a != idx_b:
                    similarity = self._calculate_title_similarity(title_a.lower(), title_b.lower())
                    if 0.8 <= similarity < 1.0:  # Very similar but not identical
                        near_duplicates.append((title_a, title_b))
        
        return near_duplicates
    
    def _find_similar_genre_names(self, genres: List[Dict]) -> List[Tuple[str, str]]:
        """Find genre names that are semantically similar"""
        similar_names = []
        names = [genre.get('name', '') for genre in genres]
        
        for i, name_a in enumerate(names):
            for name_b in names[i+1:]:
                if name_a and name_b:
                    # Check for common variations
                    if self._are_genre_names_similar(name_a, name_b):
                        similar_names.append((name_a, name_b))
        
        return similar_names
    
    def _calculate_title_similarity(self, title_a: str, title_b: str) -> float:
        """Calculate similarity between two titles"""
        if not title_a or not title_b:
            return 0.0
        
        # Simple Jaccard similarity on words
        words_a = set(title_a.split())
        words_b = set(title_b.split())
        
        if not words_a and not words_b:
            return 1.0
        if not words_a or not words_b:
            return 0.0
        
        intersection = len(words_a & words_b)
        union = len(words_a | words_b)
        
        return intersection / union if union > 0 else 0.0
    
    def _are_genre_names_similar(self, name_a: str, name_b: str) -> bool:
        """Check if two genre names are semantically similar"""
        name_a = name_a.lower().strip()
        name_b = name_b.lower().strip()
        
        # Check for common variations
        variations = {
            'sci-fi': 'science fiction',
            'scifi': 'science fiction',
            'action': 'action & adventure',
            'kids': 'children',
            'family': 'children',
        }
        
        # Normalize names
        for abbrev, full in variations.items():
            name_a = name_a.replace(abbrev, full)
            name_b = name_b.replace(abbrev, full)
        
        # Check similarity
        similarity = self._calculate_title_similarity(name_a, name_b)
        return similarity > 0.7
    
    def _analyze_missing_values_enhanced(self, movies: List[Dict], genres: List[Dict]) -> Dict[str, Any]:
        """Enhanced comprehensive missing value analysis"""
        analysis = {}
        
        # Enhanced movie missing value analysis
        movie_fields = ['id', 'title', 'genre_ids', 'overview', 'release_date', 'vote_average', 'popularity']
        movie_missing = {}
        
        for field in movie_fields:
            missing_count = 0
            invalid_count = 0
            
            for movie in movies:
                if field not in movie:
                    missing_count += 1
                elif movie.get(field) in [None, '', []]:
                    missing_count += 1
                elif field == 'genre_ids' and not isinstance(movie.get(field), list):
                    invalid_count += 1
                elif field in ['vote_average', 'popularity'] and not isinstance(movie.get(field), (int, float)):
                    invalid_count += 1
            
            movie_missing[field] = {
                'missing_count': missing_count,
                'invalid_count': invalid_count,
                'total_issues': missing_count + invalid_count,
                'ratio': (missing_count + invalid_count) / max(len(movies), 1),
                'is_critical': field in self.validation_rules['movies']['required_fields']
            }
        
        overall_movie_missing = np.mean([info['ratio'] for info in movie_missing.values()])
        
        analysis['movies'] = {
            'total_records': len(movies),
            'field_analysis': movie_missing,
            'overall_missing_ratio': overall_movie_missing,
            'critical_missing_fields': [field for field, info in movie_missing.items() 
                                      if info['is_critical'] and info['ratio'] > 0.02]
        }
        
        # Enhanced genre missing value analysis
        genre_fields = ['id', 'name']
        genre_missing = {}
        
        for field in genre_fields:
            missing_count = 0
            invalid_count = 0
            
            for genre in genres:
                if field not in genre:
                    missing_count += 1
                elif genre.get(field) in [None, '']:
                    missing_count += 1
                elif field == 'id' and not isinstance(genre.get(field), int):
                    invalid_count += 1
                elif field == 'name' and not isinstance(genre.get(field), str):
                    invalid_count += 1
            
            genre_missing[field] = {
                'missing_count': missing_count,
                'invalid_count': invalid_count,
                'total_issues': missing_count + invalid_count,
                'ratio': (missing_count + invalid_count) / max(len(genres), 1),
                'is_critical': field in self.validation_rules['genres']['required_fields']
            }
        
        overall_genre_missing = np.mean([info['ratio'] for info in genre_missing.values()])
        
        analysis['genres'] = {
            'total_records': len(genres),
            'field_analysis': genre_missing,
            'overall_missing_ratio': overall_genre_missing,
            'critical_missing_fields': [field for field, info in genre_missing.items() 
                                      if info['is_critical'] and info['ratio'] > 0.001]
        }
        
        return analysis
    
    def _analyze_consistency_enhanced(self, movies: List[Dict], genres: List[Dict]) -> Dict[str, Any]:
        """Enhanced comprehensive data consistency analysis"""
        analysis = {}
        consistency_scores = []
        
        # 1. Enhanced Referential Integrity
        valid_genre_ids = set(g.get('id') for g in genres if 'id' in g and isinstance(g['id'], int))
        invalid_references = 0
        total_references = 0
        orphaned_movies = 0
        
        for movie in movies:
            genre_ids = movie.get('genre_ids', [])
            if isinstance(genre_ids, list):
                if not genre_ids:  # Movie with no genres
                    orphaned_movies += 1
                
                for gid in genre_ids:
                    total_references += 1
                    if not isinstance(gid, int) or gid not in valid_genre_ids:
                        invalid_references += 1
        
        referential_integrity_score = 1.0 - (invalid_references / max(total_references, 1))
        consistency_scores.append(referential_integrity_score)
        
        # 2. Enhanced Data Type Consistency
        type_violations = 0
        total_checks = 0
        
        for movie in movies:
            for field, expected_type in self.validation_rules['movies']['field_types'].items():
                if field in movie:
                    total_checks += 1
                    if not isinstance(movie[field], expected_type):
                        type_violations += 1
        
        for genre in genres:
            for field, expected_type in self.validation_rules['genres']['field_types'].items():
                if field in genre:
                    total_checks += 1
                    if not isinstance(genre[field], expected_type):
                        type_violations += 1
        
        type_consistency_score = 1.0 - (type_violations / max(total_checks, 1))
        consistency_scores.append(type_consistency_score)
        
        # 3. Enhanced Format Consistency
        format_violations = 0
        format_checks = 0
        
        # Check movie ID format and uniqueness
        movie_ids_seen = set()
        for movie in movies:
            if 'id' in movie:
                format_checks += 1
                movie_id = movie['id']
                if not (isinstance(movie_id, int) and movie_id > 0):
                    format_violations += 1
                elif movie_id in movie_ids_seen:
                    format_violations += 1  # Duplicate ID
                else:
                    movie_ids_seen.add(movie_id)
        
        # Check title format
        for movie in movies:
            if 'title' in movie:
                format_checks += 1
                title = movie['title']
                if not (isinstance(title, str) and title.strip()):
                    format_violations += 1
        
        # Check genre format and uniqueness
        genre_ids_seen = set()
        genre_names_seen = set()
        for genre in genres:
            if 'id' in genre:
                format_checks += 1
                genre_id = genre['id']
                if not (isinstance(genre_id, int) and genre_id > 0):
                    format_violations += 1
                elif genre_id in genre_ids_seen:
                    format_violations += 1
                else:
                    genre_ids_seen.add(genre_id)
            
            if 'name' in genre:
                format_checks += 1
                genre_name = genre['name'].strip().lower() if isinstance(genre['name'], str) else ''
                if not genre_name:
                    format_violations += 1
                elif genre_name in genre_names_seen:
                    format_violations += 1
                else:
                    genre_names_seen.add(genre_name)
        
        format_consistency_score = 1.0 - (format_violations / max(format_checks, 1))
        consistency_scores.append(format_consistency_score)
        
        # 4. Business Logic Consistency
        business_logic_violations = 0
        business_logic_checks = 0
        
        # Check for reasonable genre assignments
        for movie in movies:
            business_logic_checks += 1
            genre_count = len(movie.get('genre_ids', []))
            if genre_count > 10:  # Too many genres
                business_logic_violations += 1
        
        business_logic_score = 1.0 - (business_logic_violations / max(business_logic_checks, 1))
        consistency_scores.append(business_logic_score)
        
        # Overall consistency score
        overall_consistency = np.mean(consistency_scores)
        
        analysis = {
            'referential_integrity_score': referential_integrity_score,
            'invalid_genre_references': invalid_references,
            'total_genre_references': total_references,
            'orphaned_movies': orphaned_movies,
            'type_consistency_score': type_consistency_score,
            'type_violations': type_violations,
            'format_consistency_score': format_consistency_score,
            'format_violations': format_violations,
            'business_logic_score': business_logic_score,
            'business_logic_violations': business_logic_violations,
            'overall_consistency_score': overall_consistency,
            'consistency_breakdown': {
                'referential_integrity': referential_integrity_score,
                'type_consistency': type_consistency_score,
                'format_consistency': format_consistency_score,
                'business_logic': business_logic_score
            }
        }
        
        return analysis
    
    def _clean_movies_enhanced(self, movies: List[Dict], genres: List[Dict]) -> List[Dict]:
        """Enhanced comprehensive movie data cleaning"""
        logger.info("🧹 Enhanced movie data cleaning...")
        
        cleaned_movies = []
        seen_ids = set()
        seen_title_hashes = set()
        
        # Build valid genre IDs for validation
        valid_genre_ids = set(g.get('id') for g in genres if 'id' in g and isinstance(g['id'], int))
        
        for movie in movies:
            # Skip if missing critical fields
            if 'id' not in movie or 'title' not in movie:
                self.cleaning_stats['movies_missing_critical_fields'] += 1
                continue
            
            # Enhanced ID validation
            movie_id = movie['id']
            if not isinstance(movie_id, int) or movie_id <= 0:
                self.cleaning_stats['movies_invalid_id'] += 1
                continue
            
            # Skip duplicate IDs
            if movie_id in seen_ids:
                self.cleaning_stats['movies_duplicate_id'] += 1
                continue
            seen_ids.add(movie_id)
            
            # Enhanced title cleaning and validation
            title = str(movie['title']).strip()
            if not title or len(title) < 1:
                self.cleaning_stats['movies_empty_title'] += 1
                continue
            
            # Enhanced duplicate title detection
            title_normalized = re.sub(r'[^\w\s]', '', title.lower())
            title_hash = hashlib.md5(title_normalized.encode()).hexdigest()
            if title_hash in seen_title_hashes:
                self.cleaning_stats['movies_duplicate_title'] += 1
                continue
            seen_title_hashes.add(title_hash)
            
            # Create enhanced cleaned movie object
            cleaned_movie = {
                'id': movie_id,
                'title': title
            }
            
            # Enhanced genre_ids cleaning with validation
            genre_ids = movie.get('genre_ids', [])
            if isinstance(genre_ids, list):
                cleaned_genre_ids = []
                for gid in genre_ids:
                    if isinstance(gid, int) and gid > 0 and gid in valid_genre_ids:
                        cleaned_genre_ids.append(gid)
                    else:
                        self.cleaning_stats['movies_invalid_genre_reference'] += 1
                
                cleaned_movie['genre_ids'] = cleaned_genre_ids
            else:
                cleaned_movie['genre_ids'] = []
                self.cleaning_stats['movies_invalid_genre_ids'] += 1
            
            # Enhanced cleaning of other fields
            for field in ['overview', 'release_date', 'vote_average', 'vote_count', 'popularity']:
                if field in movie and movie[field] is not None:
                    value = movie[field]
                    
                    # Validate numeric fields
                    if field in ['vote_average', 'vote_count', 'popularity']:
                        if isinstance(value, (int, float)) and not np.isnan(value):
                            cleaned_movie[field] = float(value)
                        else:
                            self.cleaning_stats[f'movies_invalid_{field}'] += 1
                    else:
                        cleaned_movie[field] = value
            
            cleaned_movies.append(cleaned_movie)
        
        logger.info(f"✅ Enhanced movie cleaning complete: {len(movies)} → {len(cleaned_movies)} movies")
        return cleaned_movies
    
    def _clean_genres_enhanced(self, genres: List[Dict]) -> List[Dict]:
        """Enhanced comprehensive genre data cleaning"""
        logger.info("🧹 Enhanced genre data cleaning...")
        
        cleaned_genres = []
        seen_ids = set()
        seen_names = set()
        
        for genre in genres:
            # Skip if missing critical fields
            if 'id' not in genre or 'name' not in genre:
                self.cleaning_stats['genres_missing_critical_fields'] += 1
                continue
            
            # Enhanced ID validation
            genre_id = genre['id']
            if not isinstance(genre_id, int) or genre_id <= 0:
                self.cleaning_stats['genres_invalid_id'] += 1
                continue
            
            # Skip duplicate IDs
            if genre_id in seen_ids:
                self.cleaning_stats['genres_duplicate_id'] += 1
                continue
            seen_ids.add(genre_id)
            
            # Enhanced name cleaning and validation
            name = str(genre['name']).strip()
            if not name or len(name) < 2:
                self.cleaning_stats['genres_empty_name'] += 1
                continue
            
            # Skip duplicate names (case-insensitive, normalized)
            name_normalized = re.sub(r'[^\w\s]', '', name.lower())
            if name_normalized in seen_names:
                self.cleaning_stats['genres_duplicate_name'] += 1
                continue
            seen_names.add(name_normalized)
            
            # Create enhanced clean genre object
            cleaned_genre = {
                'id': genre_id,
                'name': name
            }
            
            cleaned_genres.append(cleaned_genre)
        
        logger.info(f"✅ Enhanced genre cleaning complete: {len(genres)} → {len(cleaned_genres)} genres")
        return cleaned_genres
    
    def _generate_enhanced_statistics(self, movies: List[Dict], genres: List[Dict], 
                                    original_movie_count: int, original_genre_count: int,
                                    pre_duplicate_report: Dict, pre_missing_report: Dict, 
                                    pre_consistency_report: Dict,
                                    post_duplicate_report: Dict, post_missing_report: Dict, 
                                    post_consistency_report: Dict) -> Dict[str, Any]:
        """Generate enhanced comprehensive statistics"""
        return {
            'input_data': {
                'original_movie_count': original_movie_count,
                'original_genre_count': original_genre_count,
                'final_movie_count': len(movies),
                'final_genre_count': len(genres),
                'movie_retention_rate': len(movies) / max(original_movie_count, 1),
                'genre_retention_rate': len(genres) / max(original_genre_count, 1)
            },
            'cleaning_statistics': dict(self.cleaning_stats),
            'pre_cleaning_metrics': {
                'duplicate_ratio': pre_duplicate_report['movies']['duplicate_ratio'],
                'missing_value_ratio': pre_missing_report['movies']['overall_missing_ratio'],
                'consistency_score': pre_consistency_report['overall_consistency_score']
            },
            'post_cleaning_metrics': {
                'duplicate_ratio': post_duplicate_report['movies']['duplicate_ratio'],
                'missing_value_ratio': post_missing_report['movies']['overall_missing_ratio'],
                'consistency_score': post_consistency_report['overall_consistency_score']
            },
            'improvement_metrics': {
                'duplicate_reduction': pre_duplicate_report['movies']['duplicate_ratio'] - post_duplicate_report['movies']['duplicate_ratio'],
                'missing_value_reduction': pre_missing_report['movies']['overall_missing_ratio'] - post_missing_report['movies']['overall_missing_ratio'],
                'consistency_improvement': post_consistency_report['overall_consistency_score'] - pre_consistency_report['overall_consistency_score']
            },
            'data_distribution': {
                'movies_with_genres': sum(1 for m in movies if m.get('genre_ids')),
                'movies_without_genres': sum(1 for m in movies if not m.get('genre_ids')),
                'average_genres_per_movie': np.mean([len(m.get('genre_ids', [])) for m in movies]),
                'unique_movie_titles': len(set(m['title'].lower() for m in movies if 'title' in m))
            },
            'validation_thresholds': {
                'duplicate_threshold': self.duplicate_threshold,
                'missing_value_threshold': self.missing_value_threshold,
                'consistency_threshold': self.consistency_threshold
            },
            'resolution_metrics': self.resolution_metrics
        }
    
    def _calculate_enhanced_quality_score(self, duplicate_report: Dict, missing_report: Dict, 
                                        consistency_report: Dict) -> float:
        """Calculate enhanced overall data quality score (0-100)"""
        base_score = 100.0
        
        # Enhanced deductions for duplicates
        avg_duplicate_ratio = (duplicate_report['movies']['duplicate_ratio'] + 
                             duplicate_report['genres']['duplicate_ratio']) / 2
        duplicate_penalty = min(avg_duplicate_ratio * 40, 25)  # Max 25 points penalty
        
        # Enhanced deductions for missing values
        avg_missing_ratio = (missing_report['movies']['overall_missing_ratio'] + 
                           missing_report['genres']['overall_missing_ratio']) / 2
        missing_penalty = min(avg_missing_ratio * 35, 30)  # Max 30 points penalty
        
        # Enhanced deductions for consistency issues
        consistency_penalty = min((1.0 - consistency_report['overall_consistency_score']) * 40, 35)  # Max 35 points penalty
        
        # Additional penalties for critical issues
        critical_penalty = 0
        if missing_report['movies']['critical_missing_fields']:
            critical_penalty += 5
        if missing_report['genres']['critical_missing_fields']:
            critical_penalty += 5
        
        final_score = max(0, base_score - duplicate_penalty - missing_penalty - consistency_penalty - critical_penalty)
        
        return round(final_score, 2)
    
    def _generate_enhanced_recommendations(self, critical_issues: List[str], warnings: List[str], 
                                         statistics: Dict, quality_score: float,
                                         duplicates_resolved: bool, missing_resolved: bool,
                                         consistency_resolved: bool) -> List[str]:
        """Generate enhanced actionable recommendations"""
        recommendations = []
        
        # Resolution status based recommendations
        if duplicates_resolved and missing_resolved and consistency_resolved:
            recommendations.append("🎉 EXCELLENT: All critical data quality issues have been successfully resolved")
            recommendations.append("✅ Data quality is now at optimal level for reliable analysis")
        else:
            recommendations.append("🚨 ATTENTION: Some critical data quality issues remain unresolved")
            
            if not duplicates_resolved:
                recommendations.append("🔧 PRIORITY: Implement more aggressive duplicate detection and removal")
            if not missing_resolved:
                recommendations.append("🔧 PRIORITY: Develop comprehensive missing value handling strategy")
            if not consistency_resolved:
                recommendations.append("🔧 PRIORITY: Fix referential integrity and data consistency issues")
        
        # Quality score based recommendations
        if quality_score >= 90:
            recommendations.append("🏆 Data quality is excellent - maintain current standards")
        elif quality_score >= 80:
            recommendations.append("✅ Data quality is good - minor optimizations recommended")
        elif quality_score >= 70:
            recommendations.append("⚠️ Data quality needs improvement - implement monitoring")
        else:
            recommendations.append("🚨 Data quality is critically low - immediate intervention required")
        
        # Specific improvement recommendations
        improvement_metrics = statistics.get('improvement_metrics', {})
        
        if improvement_metrics.get('duplicate_reduction', 0) > 0:
            recommendations.append(f"📈 Duplicate reduction achieved: {improvement_metrics['duplicate_reduction']:.3f}")
        
        if improvement_metrics.get('missing_value_reduction', 0) > 0:
            recommendations.append(f"📈 Missing value reduction achieved: {improvement_metrics['missing_value_reduction']:.3f}")
        
        if improvement_metrics.get('consistency_improvement', 0) > 0:
            recommendations.append(f"📈 Consistency improvement achieved: {improvement_metrics['consistency_improvement']:.3f}")
        
        # Long-term recommendations
        recommendations.append("🔄 Implement continuous data quality monitoring")
        recommendations.append("📊 Set up automated alerts for data quality degradation")
        recommendations.append("🏗️ Establish comprehensive data governance framework")
        recommendations.append("🎯 Regularly review and update data quality thresholds")
        
        return recommendations
    
    def save_enhanced_quality_report(self, report: EnhancedDataQualityReport, filepath: str):
        """Save enhanced comprehensive data quality report"""
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
            
            logger.info(f"📋 Enhanced data quality report saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save enhanced data quality report: {str(e)}")
    
    def get_enhanced_cleaning_summary(self) -> Dict[str, Any]:
        """Get enhanced summary of data cleaning operations"""
        return {
            'cleaning_statistics': dict(self.cleaning_stats),
            'resolution_metrics': self.resolution_metrics,
            'quality_thresholds': {
                'duplicate_threshold': self.duplicate_threshold,
                'missing_value_threshold': self.missing_value_threshold,
                'consistency_threshold': self.consistency_threshold
            },
            'validation_rules': self.validation_rules
        }

# Backward compatibility
DataQualityValidator = EnhancedDataQualityValidator