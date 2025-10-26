"""
Feature extraction from resumes and job descriptions
Extracts skills, education, experience, and other relevant features
"""

import re
import logging
from typing import List, Dict, Set, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract structured features from resume and job description text"""
    
    # Common programming languages
    PROGRAMMING_LANGUAGES = {
        'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'c', 'go', 'rust',
        'ruby', 'php', 'swift', 'kotlin', 'scala', 'r', 'matlab', 'perl', 'shell',
        'bash', 'sql', 'html', 'css', 'dart', 'objective-c'
    }
    
    # Common frameworks and libraries
    FRAMEWORKS = {
        'react', 'angular', 'vue', 'django', 'flask', 'fastapi', 'spring', 'express',
        'node', 'nodejs', 'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'pandas',
        'numpy', 'scipy', 'spark', 'hadoop', 'kafka', 'docker', 'kubernetes', 'aws',
        'azure', 'gcp', 'git', 'jenkins', 'terraform', 'ansible'
    }
    
    # Common databases
    DATABASES = {
        'mysql', 'postgresql', 'mongodb', 'redis', 'cassandra', 'elasticsearch',
        'dynamodb', 'oracle', 'sql server', 'sqlite', 'mariadb', 'neo4j', 'couchdb'
    }
    
    # Cloud platforms
    CLOUD_PLATFORMS = {
        'aws', 'azure', 'gcp', 'google cloud', 'amazon web services', 'heroku',
        'digitalocean', 'ibm cloud', 'oracle cloud'
    }
    
    # Education degrees
    DEGREES = {
        'phd', 'ph.d', 'doctorate', 'masters', 'master', 'msc', 'm.sc', 'ma', 'm.a',
        'mba', 'm.b.a', 'bachelor', 'bachelors', 'bs', 'b.s', 'ba', 'b.a', 'bsc',
        'b.sc', 'associate', 'diploma'
    }
    
    # Combine all technical skills
    TECHNICAL_SKILLS = PROGRAMMING_LANGUAGES | FRAMEWORKS | DATABASES | CLOUD_PLATFORMS
    
    def __init__(self):
        """Initialize feature extractor"""
        self.tfidf_vectorizer = None
    
    def extract_skills(self, text: str, custom_skills: Optional[List[str]] = None) -> List[str]:
        """
        Extract technical skills from text
        
        Args:
            text: Input text (resume or job description)
            custom_skills: Additional custom skills to look for
            
        Returns:
            List of identified skills
        """
        text_lower = text.lower()
        found_skills = set()
        
        # Check for predefined technical skills
        for skill in self.TECHNICAL_SKILLS:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text_lower):
                found_skills.add(skill)
        
        # Check for custom skills
        if custom_skills:
            for skill in custom_skills:
                pattern = r'\b' + re.escape(skill.lower()) + r'\b'
                if re.search(pattern, text_lower):
                    found_skills.add(skill.lower())
        
        return sorted(list(found_skills))
    
    def extract_education(self, text: str) -> List[str]:
        """
        Extract education degrees from text
        
        Args:
            text: Input text
            
        Returns:
            List of found degrees
        """
        text_lower = text.lower()
        found_degrees = set()
        
        for degree in self.DEGREES:
            pattern = r'\b' + re.escape(degree) + r'\b'
            if re.search(pattern, text_lower):
                found_degrees.add(degree)
        
        return sorted(list(found_degrees))
    
    def calculate_skills_match(
        self,
        resume_skills: List[str],
        required_skills: List[str],
        preferred_skills: Optional[List[str]] = None
    ) -> Dict[str, any]:
        """
        Calculate skill match statistics
        
        Args:
            resume_skills: Skills from resume
            required_skills: Required skills from job
            preferred_skills: Preferred skills from job
            
        Returns:
            Dictionary with match statistics
        """
        resume_set = set(skill.lower() for skill in resume_skills)
        required_set = set(skill.lower() for skill in required_skills)
        preferred_set = set(skill.lower() for skill in (preferred_skills or []))
        
        # Calculate matches
        matched_required = resume_set & required_set
        matched_preferred = resume_set & preferred_set
        missing_required = required_set - resume_set
        
        # Calculate percentages
        required_match_pct = (len(matched_required) / len(required_set) * 100) if required_set else 100
        preferred_match_pct = (len(matched_preferred) / len(preferred_set) * 100) if preferred_set else 0
        
        return {
            'matched_required': list(matched_required),
            'matched_preferred': list(matched_preferred),
            'missing_required': list(missing_required),
            'required_match_percentage': required_match_pct,
            'preferred_match_percentage': preferred_match_pct,
            'total_matched': len(matched_required) + len(matched_preferred)
        }
    
    def calculate_experience_match(
        self,
        resume_experience: Optional[float],
        min_experience: Optional[float],
        max_experience: Optional[float]
    ) -> float:
        """
        Calculate experience match score
        
        Args:
            resume_experience: Years of experience from resume
            min_experience: Minimum required experience
            max_experience: Maximum preferred experience
            
        Returns:
            Match score (0-100)
        """
        if resume_experience is None:
            return 50.0  # Neutral score if unknown
        
        if min_experience is None:
            return 100.0  # No minimum requirement
        
        # Perfect match if within range
        if max_experience and min_experience <= resume_experience <= max_experience:
            return 100.0
        
        # Good match if meets minimum
        if resume_experience >= min_experience:
            # Slightly reduce score if significantly over-qualified
            if max_experience and resume_experience > max_experience * 1.5:
                return 75.0
            return 95.0
        
        # Partial match if close to minimum
        if resume_experience >= min_experience * 0.7:
            return 70.0
        
        # Low match if below minimum
        ratio = resume_experience / min_experience
        return max(30.0, ratio * 100)
    
    def fit_tfidf(self, documents: List[str], max_features: int = 1000):
        """
        Fit TF-IDF vectorizer on documents
        
        Args:
            documents: List of documents
            max_features: Maximum number of features
        """
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),  # unigrams and bigrams
            min_df=2,  # ignore terms that appear in less than 2 documents
            max_df=0.8  # ignore terms that appear in more than 80% of documents
        )
        
        self.tfidf_vectorizer.fit(documents)
        logger.info(f"Fitted TF-IDF vectorizer with {len(self.tfidf_vectorizer.vocabulary_)} features")
    
    def extract_tfidf_features(self, text: str) -> np.ndarray:
        """
        Extract TF-IDF features from text
        
        Args:
            text: Input text
            
        Returns:
            TF-IDF feature vector
        """
        if self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF vectorizer not fitted. Call fit_tfidf() first.")
        
        features = self.tfidf_vectorizer.transform([text])
        return features.toarray()[0]
    
    def extract_all_features(
        self,
        resume_text: str,
        job_text: str,
        required_skills: Optional[List[str]] = None,
        preferred_skills: Optional[List[str]] = None,
        min_experience: Optional[float] = None,
        max_experience: Optional[float] = None
    ) -> Dict[str, any]:
        """
        Extract all features for resume-job pair
        
        Args:
            resume_text: Resume text
            job_text: Job description text
            required_skills: Required skills list
            preferred_skills: Preferred skills list
            min_experience: Minimum experience required
            max_experience: Maximum experience preferred
            
        Returns:
            Dictionary of extracted features
        """
        # Extract skills
        resume_skills = self.extract_skills(resume_text)
        job_skills = self.extract_skills(job_text)
        
        # Calculate skills match
        skills_match = self.calculate_skills_match(
            resume_skills=resume_skills,
            required_skills=required_skills or job_skills,
            preferred_skills=preferred_skills
        )
        
        # Extract education
        resume_education = self.extract_education(resume_text)
        job_education = self.extract_education(job_text)
        
        # Extract experience
        from backend.ml.preprocessing.text_cleaner import TextCleaner
        resume_experience = TextCleaner.extract_years_of_experience(resume_text)
        
        # Calculate experience match
        experience_match = self.calculate_experience_match(
            resume_experience=resume_experience,
            min_experience=min_experience,
            max_experience=max_experience
        )
        
        return {
            'resume_skills': resume_skills,
            'job_skills': job_skills,
            'skills_match': skills_match,
            'resume_education': resume_education,
            'job_education': job_education,
            'resume_experience_years': resume_experience,
            'experience_match_score': experience_match
        }


# Singleton instance
_extractor_instance = None


def get_feature_extractor() -> FeatureExtractor:
    """
    Get singleton instance of FeatureExtractor
    
    Returns:
        FeatureExtractor instance
    """
    global _extractor_instance
    
    if _extractor_instance is None:
        _extractor_instance = FeatureExtractor()
    
    return _extractor_instance

