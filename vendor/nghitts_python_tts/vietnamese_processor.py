"""
Vietnamese Text Processor
Converts Vietnamese text to speech-friendly format:
- Numbers to words
- Dates, times to words
- Currency, percentages to words
- Measurement units to Vietnamese names
- Phone numbers read digit-by-digit
"""

import re
from typing import Optional, Dict

# Vietnamese number words
DIGITS = {
    '0': 'không', '1': 'một', '2': 'hai', '3': 'ba', '4': 'bốn',
    '5': 'năm', '6': 'sáu', '7': 'bảy', '8': 'tám', '9': 'chín'
}

TEENS = {
    '10': 'mười', '11': 'mười một', '12': 'mười hai', '13': 'mười ba',
    '14': 'mười bốn', '15': 'mười lăm', '16': 'mười sáu', '17': 'mười bảy',
    '18': 'mười tám', '19': 'mười chín'
}

TENS = {
    '2': 'hai mươi', '3': 'ba mươi', '4': 'bốn mươi', '5': 'năm mươi',
    '6': 'sáu mươi', '7': 'bảy mươi', '8': 'tám mươi', '9': 'chín mươi'
}


def number_to_words(num_str: str) -> str:
    """Convert a number string to Vietnamese words."""
    # Remove leading zeros but keep at least one digit
    num_str = num_str.lstrip('0') or '0'
    
    # Handle negative numbers
    if num_str.startswith('-'):
        return 'âm ' + number_to_words(num_str[1:])
    
    try:
        num = int(num_str)
    except ValueError:
        return num_str
    
    if num == 0:
        return 'không'
    
    if num < 10:
        return DIGITS[str(num)]
    
    if num < 20:
        return TEENS[str(num)]
    
    if num < 100:
        tens = num // 10
        units = num % 10
        if units == 0:
            return TENS[str(tens)]
        elif units == 1:
            return TENS[str(tens)] + ' mốt'
        elif units == 4:
            return TENS[str(tens)] + ' tư'
        elif units == 5:
            return TENS[str(tens)] + ' lăm'
        else:
            return TENS[str(tens)] + ' ' + DIGITS[str(units)]
    
    if num < 1000:
        hundreds = num // 100
        remainder = num % 100
        result = DIGITS[str(hundreds)] + ' trăm'
        if remainder == 0:
            return result
        elif remainder < 10:
            return result + ' lẻ ' + DIGITS[str(remainder)]
        else:
            return result + ' ' + number_to_words(str(remainder))
    
    if num < 1000000:
        thousands = num // 1000
        remainder = num % 1000
        result = number_to_words(str(thousands)) + ' nghìn'
        if remainder == 0:
            return result
        elif remainder < 100:
            if remainder < 10:
                return result + ' không trăm lẻ ' + DIGITS[str(remainder)]
            else:
                return result + ' không trăm ' + number_to_words(str(remainder))
        else:
            return result + ' ' + number_to_words(str(remainder))
    
    if num < 1000000000:
        millions = num // 1000000
        remainder = num % 1000000
        result = number_to_words(str(millions)) + ' triệu'
        if remainder == 0:
            return result
        elif remainder < 100:
            if remainder < 10:
                return result + ' không trăm lẻ ' + DIGITS[str(remainder)]
            else:
                return result + ' không trăm ' + number_to_words(str(remainder))
        else:
            return result + ' ' + number_to_words(str(remainder))
    
    if num < 1000000000000:
        billions = num // 1000000000
        remainder = num % 1000000000
        result = number_to_words(str(billions)) + ' tỷ'
        if remainder == 0:
            return result
        elif remainder < 100:
            if remainder < 10:
                return result + ' không trăm lẻ ' + DIGITS[str(remainder)]
            else:
                return result + ' không trăm ' + number_to_words(str(remainder))
        else:
            return result + ' ' + number_to_words(str(remainder))
    
    # For very large numbers, read digit by digit
    return ' '.join(DIGITS.get(d, d) for d in num_str)


def remove_thousand_separators(text: str) -> str:
    """Remove thousand separators (dots) from numbers."""
    # Match patterns like: 1.000, 140.000, 1.000.000, etc.
    pattern = r'(\d{1,3}(?:\.\d{3})+)(?=\s|$|[^\d.,])'
    return re.sub(pattern, lambda m: m.group(1).replace('.', ''), text)


def convert_decimal(text: str) -> str:
    """Convert decimal numbers: 7,27 -> bảy phẩy hai mươi bảy."""
    pattern = r'(\d+),(\d+)(?=\s|$|[^\d,])'
    def replace_decimal(match):
        integer_part = match.group(1)
        decimal_part = match.group(2)
        integer_words = number_to_words(integer_part)
        decimal_words = number_to_words(decimal_part.lstrip('0') or '0')
        return f'{integer_words} phẩy {decimal_words}'
    return re.sub(pattern, replace_decimal, text)


def convert_percentage(text: str) -> str:
    """Convert percentages: 50% -> năm mươi phần trăm."""
    # Handle percentage ranges first (e.g., "3-5%")
    text = re.sub(
        r'(\d+)\s*[-–—]\s*(\d+)\s*%',
        lambda m: f"{number_to_words(m.group(1))} đến {number_to_words(m.group(2))} phần trăm",
        text
    )
    
    # Handle percentages with decimals (e.g., "3,2%")
    text = re.sub(
        r'(\d+),(\d+)\s*%',
        lambda m: f"{number_to_words(m.group(1))} phẩy {number_to_words(m.group(2).lstrip('0') or '0')} phần trăm",
        text
    )
    
    # Handle whole number percentages (e.g., "50%")
    text = re.sub(
        r'(\d+)\s*%',
        lambda m: number_to_words(m.group(1)) + ' phần trăm',
        text
    )
    
    return text


def convert_currency(text: str) -> str:
    """Convert currency amounts."""
    def replace_vnd(match):
        num = match.group(1).replace(',', '')
        return number_to_words(num) + ' đồng'
    
    # Vietnamese Dong
    text = re.sub(r'(\d+(?:,\d+)?)\s*(?:đồng|VND|vnđ)\b', replace_vnd, text, flags=re.IGNORECASE)
    text = re.sub(r'(\d+(?:,\d+)?)đ(?![a-zà-ỹ])', replace_vnd, text, flags=re.IGNORECASE)
    
    # USD
    def replace_usd(match):
        num = match.group(1).replace(',', '')
        return number_to_words(num) + ' đô la'
    
    text = re.sub(r'\$\s*(\d+(?:,\d+)?)', replace_usd, text)
    text = re.sub(r'(\d+(?:,\d+)?)\s*(?:USD|\$)', replace_usd, text, flags=re.IGNORECASE)
    
    return text


def convert_time(text: str) -> str:
    """Convert time expressions."""
    # HH:MM:SS or HH:MM
    def replace_time(match):
        hour = match.group(1)
        minute = match.group(2)
        second = match.group(3) if match.group(3) else None
        
        result = number_to_words(hour) + ' giờ'
        if minute:
            result += ' ' + number_to_words(minute) + ' phút'
        if second:
            result += ' ' + number_to_words(second) + ' giây'
        return result
    
    text = re.sub(r'(\d{1,2}):(\d{2})(?::(\d{2}))?', replace_time, text)
    
    # xxhxx format: 15h30
    def replace_h_format(match):
        h = int(match.group(1))
        m = int(match.group(2))
        if 0 <= h <= 23 and 0 <= m <= 59:
            return number_to_words(str(h)) + ' giờ ' + number_to_words(str(m))
        return match.group(0)
    
    text = re.sub(r'(\d{1,2})h(\d{2})(?![a-zà-ỹ])', replace_h_format, text, flags=re.IGNORECASE)
    
    # xxh format: 15h
    def replace_h_only(match):
        h = int(match.group(1))
        if 0 <= h <= 23:
            return number_to_words(str(h)) + ' giờ'
        return match.group(0)
    
    text = re.sub(r'(\d{1,2})h(?![a-zà-ỹ\d])', replace_h_only, text, flags=re.IGNORECASE)
    
    return text


def convert_date(text: str) -> str:
    """Convert date expressions."""
    def is_valid_date(day: str, month: str, year: str = None) -> bool:
        try:
            d, m = int(day), int(month)
            if year:
                y = int(year)
                return 1 <= d <= 31 and 1 <= m <= 12 and 1000 <= y <= 9999
            return 1 <= d <= 31 and 1 <= m <= 12
        except ValueError:
            return False
    
    def is_valid_month(month: str) -> bool:
        try:
            m = int(month)
            return 1 <= m <= 12
        except ValueError:
            return False
    
    # Date ranges with "ngày": ngày dd-dd/mm or ngày dd-dd/mm/yyyy
    def replace_date_range(match):
        day1, day2, month, year = match.groups()
        if is_valid_date(day1, month, year) and is_valid_date(day2, month, year):
            result = f'ngày {number_to_words(day1)} đến {number_to_words(day2)} tháng {number_to_words(month)}'
            if year:
                result += f' năm {number_to_words(year)}'
            return result
        return match.group(0)
    
    text = re.sub(
        r'ngày\s+(\d{1,2})\s*[-–—]\s*(\d{1,2})\s*[/-]\s*(\d{1,2})(?:\s*[/-]\s*(\d{4}))?',
        replace_date_range,
        text
    )
    
    # DD/MM/YYYY or DD-MM-YYYY
    def replace_full_date(match):
        day, month, year = match.groups()
        if is_valid_date(day, month, year):
            return f'ngày {number_to_words(day)} tháng {number_to_words(month)} năm {number_to_words(year)}'
        return match.group(0)
    
    text = re.sub(r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})', replace_full_date, text)
    
    # MM/YYYY or MM-YYYY
    def replace_month_year(match):
        month, year = match.groups()
        if is_valid_month(month):
            y = int(year)
            if 1000 <= y <= 9999:
                return f'tháng {number_to_words(month)} năm {number_to_words(year)}'
        return match.group(0)
    
    text = re.sub(r'(?:tháng\s+)?(\d{1,2})\s*[/-]\s*(\d{4})(?![/-]\d)', replace_month_year, text)
    
    # DD/MM or DD-MM
    def replace_day_month(match):
        day, month = match.groups()
        if is_valid_date(day, month):
            return f'{number_to_words(day)} tháng {number_to_words(month)}'
        return match.group(0)
    
    text = re.sub(r'(\d{1,2})\s*[/-]\s*(\d{1,2})(?![/-]\d)(?!\d+\s*%)', replace_day_month, text)
    
    # tháng X
    text = re.sub(
        r'tháng\s*(\d+)',
        lambda m: f'tháng {number_to_words(m.group(1))}' if is_valid_month(m.group(1)) else m.group(0),
        text
    )
    
    # ngày X
    def replace_day(match):
        day = match.group(1)
        d = int(day)
        if 1 <= d <= 31:
            return f'ngày {number_to_words(day)}'
        return match.group(0)
    
    text = re.sub(r'ngày\s*(\d+)', replace_day, text)
    
    return text


def convert_year_range(text: str) -> str:
    """Convert year ranges: 1873-1907."""
    return re.sub(
        r'(\d{4})\s*[-–—]\s*(\d{4})',
        lambda m: f"{number_to_words(m.group(1))} đến {number_to_words(m.group(2))}",
        text
    )


def convert_ordinal(text: str) -> str:
    """Convert ordinals: thứ 2 -> thứ hai."""
    ordinal_map = {
        '1': 'nhất', '2': 'hai', '3': 'ba', '4': 'tư', '5': 'năm',
        '6': 'sáu', '7': 'bảy', '8': 'tám', '9': 'chín', '10': 'mười'
    }
    
    def replace_ordinal(match):
        prefix = match.group(1)
        num = match.group(2)
        if num in ordinal_map:
            return f'{prefix} {ordinal_map[num]}'
        return f'{prefix} {number_to_words(num)}'
    
    return re.sub(
        r'(thứ|lần|bước|phần|chương|tập|số)\s*(\d+)',
        replace_ordinal,
        text,
        flags=re.IGNORECASE
    )


def convert_phone_number(text: str) -> str:
    """Read phone numbers digit by digit."""
    def replace_phone(match):
        digits = re.findall(r'\d', match.group(0))
        return ' '.join(DIGITS.get(d, d) for d in digits)
    
    # Vietnamese phone patterns
    text = re.sub(r'0\d{9,10}', replace_phone, text)
    text = re.sub(r'\+84\d{9,10}', replace_phone, text)
    
    return text


def convert_measurement_units(text: str) -> str:
    """Convert measurement units to Vietnamese names."""
    unit_map = {
        # Length units
        'm': 'mét', 'cm': 'xăng-ti-mét', 'mm': 'mi-li-mét', 'km': 'ki-lô-mét',
        'dm': 'đề-xi-mét', 'hm': 'héc-tô-mét', 'dam': 'đề-ca-mét', 'inch': 'in',
        # Weight units
        'kg': 'ki-lô-gam', 'g': 'gam', 'mg': 'mi-li-gam', 't': 'tấn',
        'tấn': 'tấn', 'yến': 'yến', 'lạng': 'lạng',
        # Volume units
        'ml': 'mi-li-lít', 'l': 'lít', 'lít': 'lít',
        # Area units
        'm²': 'mét vuông', 'm2': 'mét vuông', 'km²': 'ki-lô-mét vuông',
        'km2': 'ki-lô-mét vuông', 'ha': 'héc-ta', 'cm²': 'xăng-ti-mét vuông',
        'cm2': 'xăng-ti-mét vuông',
        # Volume units (cubic)
        'm³': 'mét khối', 'm3': 'mét khối', 'cm³': 'xăng-ti-mét khối',
        'cm3': 'xăng-ti-mét khối', 'km³': 'ki-lô-mét khối', 'km3': 'ki-lô-mét khối',
        # Time units
        's': 'giây', 'sec': 'giây', 'min': 'phút', 'h': 'giờ', 'hr': 'giờ', 'hrs': 'giờ',
        # Speed units
        'km/h': 'ki-lô-mét trên giờ', 'kmh': 'ki-lô-mét trên giờ',
        'm/s': 'mét trên giây', 'ms': 'mét trên giây',
        'mm/h': 'mi-li-mét trên giờ', 'cm/s': 'xăng-ti-mét trên giây',
        # Temperature units
        '°C': 'độ C', '°F': 'độ F', '°K': 'độ K',
    }
    
    # Sort units by length (longest first)
    sorted_units = sorted(unit_map.keys(), key=len, reverse=True)
    
    for unit in sorted_units:
        escaped_unit = re.escape(unit)
        
        # Pattern: digits + optional space + unit
        if len(unit) == 1:
            pattern = rf'(\d+)\s*{escaped_unit}(?!\s*[a-zA-Zà-ỹ])(?=\s*[^a-zA-Zà-ỹ]|$)'
        else:
            pattern = rf'(\d+)\s*{escaped_unit}(?=\s|[^\w]|$)'
        
        text = re.sub(
            pattern,
            lambda m: f"{m.group(1)} {unit_map[unit]}",
            text,
            flags=re.IGNORECASE
        )
    
    return text


def convert_roman_numerals(text: str, unlimited: bool = False) -> str:
    """Convert Roman numerals to Arabic digits."""
    def roman_to_arabic(roman: str) -> Optional[int]:
        roman_map = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        upper_roman = roman.upper()
        
        # Check all characters are valid
        if not all(c in roman_map for c in upper_roman):
            return None
        
        result = 0
        i = 0
        
        while i < len(upper_roman):
            current = roman_map[upper_roman[i]]
            next_val = roman_map[upper_roman[i + 1]] if i + 1 < len(upper_roman) else 0
            
            if current < next_val:
                # Subtractive notation
                valid_pairs = {'I': ['V', 'X'], 'X': ['L', 'C'], 'C': ['D', 'M']}
                if upper_roman[i] not in valid_pairs or upper_roman[i + 1] not in valid_pairs[upper_roman[i]]:
                    return None
                result += next_val - current
                i += 2
            else:
                result += current
                i += 1
        
        return result
    
    def replace_roman(match):
        before = match.group(1)
        roman = match.group(2)
        
        # Only uppercase
        if roman != roman.upper():
            return match.group(0)
        
        # Convert to arabic
        arabic = roman_to_arabic(roman)
        if arabic is None:
            return match.group(0)
        
        # Limit to 1-30 if not unlimited
        if not unlimited and (arabic < 1 or arabic > 30):
            return match.group(0)
        
        return before + str(arabic)
    
    pattern = r'(^|[\s\W])([IVXLCDMivxlcdm]+)(?=[\s\W]|$)'
    return re.sub(pattern, replace_roman, text)


def convert_ranges_with_units_and_currency(text: str) -> str:
    """Convert numeric ranges and fractions followed by measurement units or currency."""
    measurement_units = [
        'm', 'cm', 'mm', 'km', 'dm', 'hm', 'dam', 'inch',
        'kg', 'g', 'mg', 't', 'tấn', 'yến', 'lạng',
        'ml', 'l', 'lít',
        'm²', 'm2', 'km²', 'km2', 'ha', 'cm²', 'cm2',
        'm³', 'm3', 'cm³', 'cm3', 'km³', 'km3',
        's', 'sec', 'min', 'h', 'hr', 'hrs',
        'km/h', 'kmh', 'm/s', 'ms', 'mm/h', 'cm/s',
        '°C', '°F', '°K',
    ]
    
    currency_units = ['đồng', 'VND', 'vnđ', 'đ', 'USD', '$']
    
    all_units = list(set(measurement_units + currency_units))
    all_units.sort(key=len, reverse=True)
    
    escaped_units = [re.escape(u) for u in all_units]
    unit_pattern = '|'.join(escaped_units)
    
    # Ranges: "1-10m", "1 - 10 kg"
    text = re.sub(
        rf'(\d+)\s*[-–—]\s*(\d+)\s*({unit_pattern})\b',
        lambda m: f"{m.group(1)} đến {m.group(2)}{'' if m.group(3).lower() == 'đ' else ' '}{m.group(3)}",
        text,
        flags=re.IGNORECASE
    )
    
    # Fractions: "1/10m", "1/10 kg"
    text = re.sub(
        rf'(\d+)\s*[/:]\s*(\d+)\s*({unit_pattern})\b',
        lambda m: f"{m.group(1)} phần {m.group(2)}{'' if m.group(3).lower() == 'đ' else ' '}{m.group(3)}",
        text,
        flags=re.IGNORECASE
    )
    
    return text


def convert_standalone_numbers(text: str) -> str:
    """Convert remaining standalone numbers to words."""
    return re.sub(r'\b\d+\b', lambda m: number_to_words(m.group(0)), text)


def normalize_unicode(text: str) -> str:
    """Normalize Unicode to NFC form."""
    import unicodedata
    return unicodedata.normalize('NFC', text)


def remove_special_chars(text: str) -> str:
    """Remove or replace special characters that can't be spoken."""
    text = text.replace('&', ' và ')
    text = text.replace('@', ' a còng ')
    text = text.replace('#', ' thăng ')
    text = text.replace('*', '')
    text = text.replace('_', ' ')
    text = text.replace('~', '')
    text = text.replace('`', '')
    text = text.replace('^', '')
    
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+\.\S+', '', text)
    
    return text


def normalize_punctuation(text: str) -> str:
    """Normalize punctuation marks."""
    # Normalize quotes
    text = re.sub(r'[""„‟]', '"', text)
    text = re.sub(r"[''‚‛]", "'", text)
    
    # Normalize dashes
    text = re.sub(r'[–—−]', '-', text)
    
    # Normalize ellipsis
    text = re.sub(r'\.{3,}', '...', text)
    text = text.replace('…', '...')
    
    # Remove multiple punctuation
    text = re.sub(r'([!?.]){2,}', r'\1', text)
    
    return text


def clean_whitespace(text: str) -> str:
    """Clean up extra whitespace."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def process_vietnamese_text(text: str, config: Optional[Dict] = None) -> str:
    """
    Main function to process Vietnamese text for TTS.
    Applies all normalization steps in the correct order.
    
    Args:
        text: Raw Vietnamese text
        config: Optional configuration dict
    
    Returns:
        Normalized text suitable for TTS
    """
    if not text or not isinstance(text, str):
        return ''
    
    # Step 1: Normalize Unicode
    text = normalize_unicode(text)
    
    # Step 2: Remove special characters
    text = remove_special_chars(text)
    
    # Step 3: Normalize punctuation
    text = normalize_punctuation(text)
    
    # Step 4: Remove thousand separators (dots)
    text = remove_thousand_separators(text)
    
    # Step 5: Convert numeric ranges/fractions with units or currency
    text = convert_ranges_with_units_and_currency(text)
    
    # Step 6: Convert year ranges
    text = convert_year_range(text)
    
    # Step 7: Convert dates
    text = convert_date(text)
    
    # Step 8: Convert times
    text = convert_time(text)
    
    # Step 8.5: Convert Roman numerals to Arabic digits
    unlimited_roman = config.get('UnlimitedRomanNumerals', False) if config else False
    text = convert_roman_numerals(text, unlimited=unlimited_roman)
    
    # Step 9: Convert ordinals
    text = convert_ordinal(text)
    
    # Step 10: Convert currency
    text = convert_currency(text)
    
    # Step 11: Convert percentages
    text = convert_percentage(text)
    
    # Step 12: Convert phone numbers
    text = convert_phone_number(text)
    
    # Step 13: Convert decimals
    text = convert_decimal(text)
    
    # Step 14: Convert measurement units
    text = convert_measurement_units(text)
    
    # Step 15: Convert remaining standalone numbers
    text = convert_standalone_numbers(text)
    
    # Step 16: Clean whitespace
    text = clean_whitespace(text)
    
    return text
