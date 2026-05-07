"""
Text Cleaner for TTS
Main processing pipeline that combines Vietnamese processing and transliteration
"""

import re
from typing import Optional, Dict, Set, List
from vietnamese_processor import process_vietnamese_text
from vietnamese_detector import is_vietnamese_word
from transliterator import transliterate_word


# Words to skip in transliteration (e.g. MC = Master of Ceremonies, kept as-is)
TRANSLITERATION_SKIP_WORDS: Set[str] = {'mc'}


def clean_text_for_tts(text: str) -> str:
    """
    Clean text for TTS: remove emojis, special characters, etc.
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text
    """
    if not text or not isinstance(text, str):
        return ''
    
    # Remove emojis using Unicode ranges
    emoji_pattern = (
        r'[\U0001F600-\U0001F64F]|[\U0001F300-\U0001F5FF]|[\U0001F680-\U0001F6FF]|'
        r'[\U0001F1E0-\U0001F1FF]|[\U00002600-\U000026FF]|[\U00002700-\U000027BF]|'
        r'[\U0001F900-\U0001F9FF]|[\U0001F018-\U0001F270]|[\U0000238C-\U00002454]|'
        r'[\U000020D0-\U000020FF]|[\U0000FE0F]|[\U0000200D]'
    )
    text = re.sub(emoji_pattern, '', text)
    
    # Clean up special characters
    text = text.replace('\\', '')
    text = text.replace('(', '')
    text = text.replace(')', '')
    text = text.replace('¯', '')
    text = re.sub(r'[""""]', '', text)
    text = text.replace(' —', '.')
    text = re.sub(r'\b_\b', ' ', text)
    
    # Remove dashes but preserve those between numbers (for ranges like 25-26)
    text = re.sub(r'(?<!\d)-(?!\d)', ' ', text)
    
    # Remove non-Latin characters (keep Latin, Vietnamese, numbers, punctuation, whitespace)
    text = re.sub(r'[^\u0000-\u024F\u1E00-\u1EFF]', '', text)
    
    return text.strip()


def apply_transliteration(
    text: str,
    replacement_map: Optional[Dict[str, str]] = None,
    config: Optional[Dict] = None
) -> str:
    """
    Apply transliteration to words not in the replacement map.
    Only processes words that weren't replaced by CSV and aren't Vietnamese.
    
    Args:
        text: Text to process
        replacement_map: Map of words that were already replaced
        config: Configuration dict
        
    Returns:
        Text with transliterated words
    """
    if not text or not isinstance(text, str):
        return text
    
    if replacement_map is None:
        replacement_map = {}
    
    # Split text into tokens using word boundaries
    # Match word characters including Vietnamese diacritics
    word_pattern = r'[\w\u00C0-\u1EFF]+'
    
    words_seen: Set[str] = set()
    result = text
    
    # Find all words
    for match in re.finditer(word_pattern, text):
        word = match.group(0)
        word_lower = word.lower()
        
        # Skip if already processed
        if word_lower in words_seen:
            continue
        words_seen.add(word_lower)
        
        # Skip if word is in replacement map
        if word_lower in replacement_map:
            continue
        
        # Skip if word is Vietnamese
        if is_vietnamese_word(word):
            continue
        
        # Skip single-character tokens
        if len(word) == 1:
            continue
        
        # Skip special words
        if word_lower in TRANSLITERATION_SKIP_WORDS:
            continue
        
        # Apply transliteration
        transliterated = transliterate_word(word)
        
        # Replace all occurrences (case-insensitive)
        # Preserve capitalization of first letter
        escaped_word = re.escape(word)
        not_word_char = r'[^\w\u00C0-\u1EFF]'
        pattern = rf'(?:^|({not_word_char}))({escaped_word})(?={not_word_char}|$)'
        
        def replace_func(m):
            boundary = m.group(1) or ''
            word_part = m.group(2)
            trans = transliterated
            if word_part and word_part[0].isupper():
                trans = trans[0].upper() + trans[1:] if len(trans) > 1 else trans.upper()
            return boundary + trans
        
        result = re.sub(pattern, replace_func, result, flags=re.IGNORECASE)
    
    return result


def process_text_for_tts(text: str, enable_transliteration: bool = True, config: Optional[Dict] = None) -> str:
    """
    Process text for TTS: clean text, process Vietnamese text, then transliterate if enabled.
    This is the main function that should be called before synthesis.
    
    Args:
        text: Raw text to process
        enable_transliteration: Whether to enable transliteration of non-Vietnamese words
        config: Optional configuration dict
        
    Returns:
        Processed text ready for TTS
    """
    if not text or not isinstance(text, str):
        return ''
    
    # Step 1: Clean the text
    cleaned_text = clean_text_for_tts(text)
    
    # Step 2: Process Vietnamese text (convert numbers, dates, times, etc.)
    vietnamese_processed = process_vietnamese_text(cleaned_text, config)
    
    # Step 3: Normalize to lowercase for consistent processing
    # (This helps with matching non-Vietnamese words)
    mapping_input = vietnamese_processed.lower()
    
    # Step 4: Apply transliteration if enabled
    if enable_transliteration:
        processed_text = apply_transliteration(mapping_input, {}, config)
    else:
        processed_text = mapping_input
    
    return processed_text


def chunk_text(text: str, max_length: int = 500) -> List[str]:
    """
    Chunk text into sentences for TTS processing.
    
    Args:
        text: Text to chunk
        max_length: Maximum length of each chunk
        
    Returns:
        List of text chunks
    """
    if not text or not isinstance(text, str):
        return []
    
    # First, split by newlines
    lines = text.split('\n')
    chunks = []
    
    for line in lines:
        # Skip empty lines
        if not line.strip():
            continue
        
        # Check if the line already ends with punctuation
        ends_with_punct = bool(re.search(r'[.!?]$', line.strip()))
        
        # If it doesn't end with punctuation and it's not empty, add a period
        processed_line = line if ends_with_punct else line.strip() + '.'
        
        # Split the line into sentences based on punctuation
        # Keep punctuation with the sentence
        sentences = re.split(r'(?<=[.!?])(?=\s+|$)', processed_line)
        
        # Each sentence becomes its own chunk
        for sentence in sentences:
            trimmed = sentence.strip()
            if trimmed:
                # If sentence is too long, split by commas
                if len(trimmed) > max_length:
                    sub_chunks = re.split(r'(?<=,)\s+', trimmed)
                    chunks.extend([s.strip() for s in sub_chunks if s.strip()])
                else:
                    chunks.append(trimmed)
    
    return chunks
