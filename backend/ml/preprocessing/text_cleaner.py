"""
Text preprocessing and cleaning utilities
Handles resume and job description text cleaning
"""

import re
import logging
from typing import List, Dict, Optional, Set
import unicodedata

logger = logging.getLogger(__name__)


class TextCleaner:
    """Text cleaning and preprocessing utilities"""
    
    # Common stop words for resume screening (minimal set)
    STOP_WORDS = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'will', 'with'
    }
    
    # Common section headers in resumes
    SECTION_HEADERS = [
        'summary', 'objective', 'experience', 'education', 'skills',
        'work experience', 'professional experience', 'employment history',
        'qualifications', 'certifications', 'projects', 'publications',
        'awards', 'languages', 'interests', 'references'
    ]
    
    def __init__(self, remove_stop_words: bool = False):
        """
        Initialize text cleaner
        
        Args:
            remove_stop_words: Whether to remove stop words
        """
        self.remove_stop_words = remove_stop_words
    
    @staticmethod
    def normalize_unicode(text: str) -> str:
        """
        Normalize unicode characters
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    
    @staticmethod
    def remove_urls(text: str) -> str:
        """Remove URLs from text"""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.sub(url_pattern, '', text)
    
    @staticmethod
    def remove_emails(text: str) -> str:
        """Remove email addresses from text"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.sub(email_pattern, '', text)
    
    @staticmethod
    def remove_phone_numbers(text: str) -> str:
        """Remove phone numbers from text"""
        phone_pattern = r'[\+]?[(]?[0-9]{1,4}[)]?[-\s\.]?[(]?[0-9]{1,4}[)]?[-\s\.]?[0-9]{1,4}[-\s\.]?[0-9]{1,4}'
        return re.sub(phone_pattern, '', text)
    
    @staticmethod
    def remove_special_characters(text: str, keep_alphanumeric: bool = True) -> str:
        """
        Remove special characters
        
        Args:
            text: Input text
            keep_alphanumeric: Keep alphanumeric characters
            
        Returns:
            Cleaned text
        """
        if keep_alphanumeric:
            # Keep letters, numbers, and basic punctuation
            text = re.sub(r'[^a-zA-Z0-9\s\.,;:\-\+\#]', ' ', text)
        else:
            # Keep only letters and spaces
            text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def remove_stopwords(self, text: str) -> str:
        """
        Remove stop words from text
        
        Args:
            text: Input text
            
        Returns:
            Text without stop words
        """
        words = text.lower().split()
        filtered_words = [word for word in words if word not in self.STOP_WORDS]
        return ' '.join(filtered_words)
    
    def clean_text(
        self,
        text: str,
        remove_personal_info: bool = True,
        lowercase: bool = True
    ) -> str:
        """
        Complete text cleaning pipeline
        
        Args:
            text: Input text
            remove_personal_info: Remove emails, phones, URLs
            lowercase: Convert to lowercase
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Normalize unicode
        text = self.normalize_unicode(text)
        
        # Remove personal information
        if remove_personal_info:
            text = self.remove_urls(text)
            text = self.remove_emails(text)
            text = self.remove_phone_numbers(text)
        
        # Remove special characters
        text = self.remove_special_characters(text)
        
        # Convert to lowercase
        if lowercase:
            text = text.lower()
        
        # Remove stop words if configured
        if self.remove_stop_words:
            text = self.remove_stopwords(text)
        
        # Final cleanup
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    @staticmethod
    def extract_sections(text: str) -> Dict[str, str]:
        """
        Extract common sections from resume text
        
        Args:
            text: Resume text
            
        Returns:
            Dictionary of section name to content
        """
        sections = {}
        text_lower = text.lower()
        
        # Find section headers
        section_positions = []
        for header in TextCleaner.SECTION_HEADERS:
            pattern = r'\b' + re.escape(header) + r'\b'
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                section_positions.append((match.start(), header))
        
        # Sort by position
        section_positions.sort()
        
        # Extract content between headers
        for i, (start, header) in enumerate(section_positions):
            if i + 1 < len(section_positions):
                end = section_positions[i + 1][0]
            else:
                end = len(text)
            
            content = text[start:end].strip()
            # Remove the header itself from content
            content = re.sub(r'^' + re.escape(header) + r'\s*:?\s*', '', content, flags=re.IGNORECASE)
            sections[header] = content
        
        return sections
    
    @staticmethod
    def extract_years_of_experience(text: str) -> Optional[float]:
        """
        Extract years of experience from text
        
        Args:
            text: Resume text
            
        Returns:
            Number of years of experience, or None if not found
        """
        patterns = [
            r'(\d+)[\+]?\s*(?:years?|yrs?)\s+(?:of\s+)?experience',
            r'experience\s*:?\s*(\d+)[\+]?\s*(?:years?|yrs?)',
            r'(\d+)[\+]?\s*(?:years?|yrs?)\s+(?:in|of)',
        ]
        
        text_lower = text.lower()
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    years = float(match.group(1))
                    return years
                except ValueError:
                    continue
        
        return None
    
    @staticmethod
    def extract_email(text: str) -> Optional[str]:
        """
        Extract email address from text
        
        Args:
            text: Input text
            
        Returns:
            Email address or None
        """
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        match = re.search(email_pattern, text)
        return match.group(0) if match else None
    
    @staticmethod
    def extract_phone(text: str) -> Optional[str]:
        """
        Extract phone number from text
        
        Args:
            text: Input text
            
        Returns:
            Phone number or None
        """
        phone_pattern = r'[\+]?[(]?[0-9]{1,4}[)]?[-\s\.]?[(]?[0-9]{1,4}[)]?[-\s\.]?[0-9]{1,4}[-\s\.]?[0-9]{1,4}'
        match = re.search(phone_pattern, text)
        return match.group(0) if match else None


# Singleton instance
_cleaner_instance = None


def get_text_cleaner(remove_stop_words: bool = False) -> TextCleaner:
    """
    Get singleton instance of TextCleaner
    
    Args:
        remove_stop_words: Whether to remove stop words
        
    Returns:
        TextCleaner instance
    """
    global _cleaner_instance
    
    if _cleaner_instance is None:
        _cleaner_instance = TextCleaner(remove_stop_words=remove_stop_words)
    
    return _cleaner_instance

