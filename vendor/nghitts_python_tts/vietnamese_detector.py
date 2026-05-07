"""
Vietnamese Language Detector
Detects if a word is Vietnamese based on diacritics, character patterns, and word structure
"""

import re
from typing import Set


class VietnameseDetector:
    """Detector for Vietnamese words based on phonetic structure."""
    
    def __init__(self):
        # 1. Vietnamese diacritics (definite Vietnamese markers)
        self.vn_accent_regex = re.compile(
            r'[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]',
            re.IGNORECASE
        )
        
        # 2. Vietnamese phonetic components
        self.vn_vowels = 'ueoaiy'
        
        # Valid Vietnamese onsets (initial consonants)
        self.vn_onsets: Set[str] = {
            'b', 'c', 'd', 'đ', 'g', 'h', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'x',
            'ch', 'gh', 'gi', 'kh', 'ng', 'nh', 'ph', 'qu', 'th', 'tr'
        }
        
        # Valid Vietnamese endings (final consonants)
        self.vn_endings: Set[str] = {'p', 't', 'c', 'm', 'n', 'ng', 'ch', 'nh'}
        
        # 3. English special characters (must be transliterated)
        self.en_special_chars = re.compile(r'[fwzj]', re.IGNORECASE)
    
    def is_vietnamese_word(self, word: str) -> bool:
        """
        Check if a word is Vietnamese.
        
        Args:
            word: The word to check
            
        Returns:
            True if the word is Vietnamese, False otherwise
        """
        if not word:
            return False
        
        w = word.lower().strip()
        
        # STEP 1: If has diacritics -> Definitely Vietnamese
        if self.vn_accent_regex.search(w):
            return True
        
        # STEP 2: If contains f, w, z, j -> Definitely English (needs transliteration)
        if self.en_special_chars.search(w):
            return False
        
        # STEP 3: Analyze word structure (no diacritics)
        # Split word into 3 parts: onset - vowel - ending
        match = re.match(r'^([^ueoaiy]*)([ueoaiy]+)([^ueoaiy]*)$', w)
        
        if not match:
            return False  # No vowel means not Vietnamese
        
        onset, vowel, ending = match.groups()
        
        # Check onset: must be empty (like 'anh') or in Vietnamese onset list
        if onset and onset not in self.vn_onsets:
            return False
        
        # Check ending: must be empty (like 'ba') or in Vietnamese ending list
        if ending and ending not in self.vn_endings:
            return False
        
        # Check vowel: Vietnamese doesn't have double vowels like 'ee', 'oo'
        # Exception: 'oa', 'oe', 'ua', 'uy' are valid Vietnamese vowels
        if re.search(r'ee|oo|ea|oa|ae|ie', vowel):
            if vowel not in ['oa', 'oe', 'ua', 'uy']:
                return False
        
        # If passes all checks, consider it Vietnamese
        return True


# Create a singleton instance
_detector = VietnameseDetector()


def is_vietnamese_word(word: str) -> bool:
    """
    Check if a word is Vietnamese.
    
    Args:
        word: The word to check
        
    Returns:
        True if the word is Vietnamese, False otherwise
    """
    return _detector.is_vietnamese_word(word)
