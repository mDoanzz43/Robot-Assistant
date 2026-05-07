"""
English to Vietnamese Transliterator
Converts English words to Vietnamese-friendly pronunciation using phonetic rules
"""

import re
from typing import List, Tuple
from vietnamese_detector import is_vietnamese_word


def english_to_vietnamese(word: str) -> str:
    """
    Convert English word to Vietnamese transliteration.
    
    Args:
        word: English word to transliterate
        
    Returns:
        Vietnamese transliteration
    """
    if not word:
        return ""
    
    w = word.lower().strip()
    
    # Handle y at the beginning -> d
    if w.startswith('y'):
        w = 'd' + w[1:]
    
    # Handle d at the beginning (English 'd' -> Vietnamese 'đ')
    if w.startswith('d'):
        w = 'đ' + w[1:]
    
    # STEP 1: High priority rules (endings and special patterns)
    high_priority_rules: List[Tuple[re.Pattern, str]] = [
        # Special endings - MUST BE AT END
        (re.compile(r'tion$'), 'ân'),
        (re.compile(r'sion$'), 'ân'),
        (re.compile(r'age$'), 'ây'),
        (re.compile(r'ing$'), 'ing'),
        (re.compile(r'ture$'), 'chờ'),
        (re.compile(r'cial$'), 'xô'),
        (re.compile(r'tial$'), 'xô'),
        
        # Complex vowel patterns
        (re.compile(r'aught'), 'ót'),
        (re.compile(r'ought'), 'ót'),
        (re.compile(r'ound'), 'ao'),
        (re.compile(r'ight'), 'ai'),
        (re.compile(r'eigh'), 'ây'),
        (re.compile(r'ough'), 'ao'),
        
        # Initial consonant clusters - ONLY AT START
        (re.compile(r'\bst(?!r)'), 't'),
        (re.compile(r'\bstr'), 'tr'),
        (re.compile(r'\bsch'), 'c'),
        (re.compile(r'\bsc(?=h)'), 'c'),
        (re.compile(r'\bsc|sk'), 'c'),
        (re.compile(r'\bsp'), 'p'),
        (re.compile(r'\btr'), 'tr'),
        (re.compile(r'\bbr'), 'r'),
        (re.compile(r'\bcr|pr|gr|dr|fr'), 'r'),
        (re.compile(r'\bbl|cl|sl|pl'), 'l'),
        (re.compile(r'\bfl'), 'ph'),
        
        # Double consonants
        (re.compile(r'ck'), 'c'),
        (re.compile(r'sh'), 's'),
        (re.compile(r'ch'), 'ch'),
        (re.compile(r'th'), 'th'),
        (re.compile(r'ph'), 'ph'),
        (re.compile(r'wh'), 'q'),
        (re.compile(r'qu'), 'q'),
        (re.compile(r'kn'), 'n'),
        (re.compile(r'wr'), 'r'),
    ]
    
    # Apply high priority rules
    for pattern, replacement in high_priority_rules:
        w = pattern.sub(replacement, w)
    
    # ENDING RULES - ONLY apply at end of word (with $)
    ending_rules: List[Tuple[re.Pattern, str]] = [
        # -LE ending -> ồ (table, apple)
        (re.compile(r'le$'), 'ồ'),
        
        # Vowel pairs + final consonant
        (re.compile(r'ook$'), 'úc'),  # book, look, cook
        (re.compile(r'ood$'), 'út'),  # good, food, wood
        (re.compile(r'ool$'), 'un'),  # cool, pool, school
        (re.compile(r'oom$'), 'um'),  # room, boom, zoom
        (re.compile(r'oon$'), 'un'),  # moon, soon, noon
        (re.compile(r'oot$'), 'út'),  # foot, boot, root
        (re.compile(r'iend$'), 'en'),
        (re.compile(r'end$'), 'en'),
        (re.compile(r'eau$'), 'iu'),
        
        (re.compile(r'ail$'), 'ain'),  # mail, tail, sail
        (re.compile(r'ain$'), 'ain'),  # main, rain, train
        (re.compile(r'ait$'), 'ât'),   # wait, bait
        
        (re.compile(r'oat$'), 'ốt'),  # boat, coat, goat
        (re.compile(r'oad$'), 'ốt'),  # road, load, toad
        (re.compile(r'oal$'), 'ôn'),  # goal, coal
        
        (re.compile(r'eep$'), 'íp'),  # keep, deep, sleep
        (re.compile(r'eet$'), 'ít'),  # meet, feet, street
        (re.compile(r'eel$'), 'in'),  # feel, steel, wheel
        
        # -TCH endings
        (re.compile(r'atch$'), 'át'),
        (re.compile(r'etch$'), 'éch'),
        (re.compile(r'itch$'), 'ích'),
        (re.compile(r'otch$'), 'ốt'),
        (re.compile(r'utch$'), 'út'),
        
        # -DGE endings
        (re.compile(r'edge$'), 'ét'),
        (re.compile(r'idge$'), 'ít'),
        (re.compile(r'odge$'), 'ót'),
        (re.compile(r'udge$'), 'út'),
        
        # -CK/-K endings
        (re.compile(r'ack$'), 'ác'),
        (re.compile(r'eck$'), 'éc'),
        (re.compile(r'ick$'), 'ích'),
        (re.compile(r'ock$'), 'óc'),
        (re.compile(r'uck$'), 'úc'),
        
        # Silent-E endings
        (re.compile(r'ake$'), 'ây'),
        (re.compile(r'ame$'), 'am'),
        (re.compile(r'ane$'), 'an'),
        (re.compile(r'ape$'), 'ếp'),
        (re.compile(r'ike$'), 'íc'),
        (re.compile(r'ime$'), 'am'),
        (re.compile(r'ine$'), 'ai'),
        (re.compile(r'oke$'), 'ốc'),
        (re.compile(r'ome$'), 'om'),
        (re.compile(r'one$'), 'oăn'),
        
        # -LL endings
        (re.compile(r'all$'), 'âu'),
        (re.compile(r'ell$'), 'eo'),
        (re.compile(r'ill$'), 'iu'),
        (re.compile(r'oll$'), 'ôn'),
        (re.compile(r'ull$'), 'un'),
        
        # -NG endings
        (re.compile(r'ang$'), 'ang'),
        (re.compile(r'eng$'), 'ing'),
        (re.compile(r'ong$'), 'ong'),
        (re.compile(r'ung$'), 'âng'),
        
        # Complex endings
        (re.compile(r'air$'), 'e'),
        (re.compile(r'ear$'), 'ia'),
        (re.compile(r'ire$'), 'ai'),
        (re.compile(r'ure$'), 'iu'),
        (re.compile(r'our$'), 'ao'),
        (re.compile(r'ore$'), 'o'),
        
        # Double vowels at end
        (re.compile(r'ee$'), 'i'),
        (re.compile(r'ea$'), 'i'),
        (re.compile(r'oo$'), 'u'),
        (re.compile(r'oa$'), 'oa'),
        (re.compile(r'ai$'), 'ai'),
        (re.compile(r'ay$'), 'ay'),
        
        # -R endings
        (re.compile(r'ar$'), 'a'),
        (re.compile(r'er$'), 'ơ'),
        (re.compile(r'ir$'), 'ơ'),
        (re.compile(r'or$'), 'o'),
        (re.compile(r'ur$'), 'ơ'),
        
        # -L endings
        (re.compile(r'al$'), 'an'),
        (re.compile(r'el$'), 'eo'),
        (re.compile(r'il$'), 'iu'),
        (re.compile(r'ol$'), 'ôn'),
        (re.compile(r'ul$'), 'un'),
        
        # Basic closed endings
        (re.compile(r'ab$'), 'áp'),
        (re.compile(r'ad$'), 'át'),
        (re.compile(r'ag$'), 'ác'),
        (re.compile(r'at$'), 'át'),
        (re.compile(r'ap$'), 'áp'),
        (re.compile(r'ed$'), 'ét'),
        (re.compile(r'et$'), 'ét'),
        (re.compile(r'id$'), 'ít'),
        (re.compile(r'it$'), 'ít'),
        (re.compile(r'ip$'), 'íp'),
        (re.compile(r'od$'), 'ót'),
        (re.compile(r'ot$'), 'ót'),
        (re.compile(r'op$'), 'óp'),
        (re.compile(r'ud$'), 'út'),
        (re.compile(r'ut$'), 'út'),
        (re.compile(r'up$'), 'úp'),
        
        # -M/-N endings
        (re.compile(r'am$'), 'am'),
        (re.compile(r'an$'), 'an'),
        (re.compile(r'em$'), 'em'),
        (re.compile(r'en$'), 'en'),
        (re.compile(r'im$'), 'im'),
        (re.compile(r'in$'), 'in'),
        (re.compile(r'om$'), 'om'),
        (re.compile(r'on$'), 'on'),
        (re.compile(r'um$'), 'âm'),
        (re.compile(r'un$'), 'ân'),
    ]
    
    # Apply ending rules
    for pattern, replacement in ending_rules:
        w = pattern.sub(replacement, w)
    
    # GENERAL RULES (for single characters or in the middle)
    general_rules: List[Tuple[re.Pattern, str]] = [
        # Single consonants
        (re.compile(r'j'), 'd'),
        (re.compile(r'z'), 'd'),
        (re.compile(r'w'), 'u'),
        (re.compile(r'x'), 'x'),
        (re.compile(r'v'), 'v'),
        (re.compile(r'f'), 'ph'),
        (re.compile(r's'), 'x'),
        (re.compile(r'c'), 'k'),
        (re.compile(r'q'), 'ku'),
        
        # Single vowels - LAST
        (re.compile(r'a'), 'a'),
        (re.compile(r'e'), 'e'),
        (re.compile(r'i'), 'i'),
        (re.compile(r'o'), 'o'),
        (re.compile(r'u'), 'u'),
    ]
    
    # Apply general rules
    for pattern, replacement in general_rules:
        w = pattern.sub(replacement, w)
    
    # Handle y: y after consonant or at end -> i
    w = re.sub(r'([bcdfghjklmnpqrstvwxz])y', r'\1i', w)
    w = re.sub(r'y$', 'i', w)
    
    # Clean up: remove double consonants
    w = re.sub(r'([brlptdgmnckxsvfzjwqh])\1+', r'\1', w)
    
    # Clean up invalid consonant pairs
    valid_pairs = ['ch', 'th', 'ph', 'sh', 'ng', 'tr', 'nh', 'gh', 'kh']
    vowels = 'aeiouăâêôơưáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ'
    
    result = ''
    i = 0
    while i < len(w):
        if i < len(w) - 1:
            pair = w[i:i+2]
            if pair in valid_pairs:
                result += pair
                i += 2
                continue
            elif w[i] not in vowels and w[i+1] not in vowels:
                # Invalid consonant pair, keep second
                result += w[i+1]
                i += 2
                continue
        result += w[i]
        i += 1
    
    w = result
    
    # Clean up final consonants
    if len(w) > 1 and w[-1] not in vowels:
        valid_endings = ['p', 't', 'c', 'm', 'n', 'g', 's']
        if w[-1] not in valid_endings:
            if w[-1] == 'l':
                w = w[:-1] + 'n'
            else:
                w = w[:-1]
    
    return w


def transliterate_word(word: str) -> str:
    """
    Transliterate a word from English to Vietnamese.
    Checks if word is Vietnamese first - if yes, returns original word unchanged.
    
    Args:
        word: Word to transliterate
        
    Returns:
        Transliterated word or original if Vietnamese
    """
    if not word or not isinstance(word, str):
        return word or ''
    
    # Check if word is Vietnamese - if yes, skip transliteration
    if is_vietnamese_word(word):
        return word
    
    # Otherwise, apply transliteration
    return english_to_vietnamese(word)
