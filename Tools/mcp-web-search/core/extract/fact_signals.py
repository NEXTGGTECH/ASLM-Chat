# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import re
from functools import lru_cache

# Commerce symbols (not a full ISO list — digits/context disambiguate).
_CURRENCY_SYMBOLS = r"[$€£¥₽₴₺₩₪₹¢₦฿]"

# ISO 4217 codes common in prices/specs (Latin letters only).
_ISO4217 = (
    r"USD|EUR|GBP|JPY|CNY|RUB|UAH|TRY|INR|CHF|CAD|AUD|NZD|"
    r"SEK|NOK|DKK|PLN|CZK|HUF|RON|BGN|BRL|MXN|ZAR|KRW|THB|"
    r"SGD|HKD|AED|SAR|ILS|PHP|IDR|VND|MYR"
)

# Short currency words (multi-script); kept minimal — symbols + ISO cover most pages.
_CURRENCY_WORDS = (
    r"руб\.?|грн\.?|тенге|тг\.?|"
    r"yen|yuan|won|baht|"
    r"fr\.?|kr\.?|zł|zl"
)

# SI / imperial / technical units (Latin).
_UNITS_LATIN = (
    r"mm|cm|dm|m|km|"
    r"mg|g|kg|t|oz|lb|"
    r"ml|l|cl|dl|"
    r"W|kW|MW|V|mV|A|mA|"
    r"Hz|kHz|MHz|GHz|"
    r"Pa|kPa|MPa|bar|psi|"
    r"°C|°F|℃|"
    r"rpm|dpi|ppi|px|pt|"
    r"ft|in|mph|kn"
)

# Cyrillic abbreviations for the same physical quantities (RU/UA/BY content).
_UNITS_CYR = r"г|гр|кг|мг|мм|см|дм|м|км|мл|л|кл|Вт|кВт|МВт|Гц|кГц|МПа|бар"

# Alternation of every known unit (Latin or Cyrillic), and an integer or decimal number with
# either a dot or a comma as the separator.
_UNITS = rf"(?:{_UNITS_LATIN}|{_UNITS_CYR})"
_NUM = r"\d+(?:[.,]\d+)?"


# Compile currency, measurement, and decimal regexes once per process (case-insensitive).
@lru_cache(maxsize=1)
def _compiled():
    flags = re.IGNORECASE
    sym = _CURRENCY_SYMBOLS
    return {
        # Currency: a number next to a symbol or ISO code (either order), or a currency word.
        "currency": re.compile(
            rf"(?:"
            rf"{sym}\s*{_NUM}"
            rf"|{_NUM}\s*{sym}"
            rf"|{sym}{_NUM}"
            rf"|{_NUM}{sym}"
            rf"|\b(?:{_ISO4217})\s*{_NUM}"
            rf"|{_NUM}\s*(?:{_ISO4217})\b"
            rf"|\b(?:{_CURRENCY_WORDS})\b"
            rf")",
            flags,
        ),
        # Measurement: a number followed by a known unit, a percent sign or a per-mille sign.
        "measurement": re.compile(
            rf"(?:"
            rf"{_NUM}\s*(?:{_UNITS})\b"
            rf"|{_NUM}\s*%"
            rf"|{_NUM}%"
            rf"|{_NUM}\s*‰"
            rf")",
            flags,
        ),
        # Decimal: digits, a dot or comma, digits.
        "decimal": re.compile(r"\d+[.,]\d+"),
    }


# Return True when the text contains a currency amount (symbol or ISO code with a number) or a currency word.
def has_currency(text: str) -> bool:
    return bool(_compiled()["currency"].search(text))


# Return True when the text contains a numeric value with a known unit or a percent sign.
def has_measurement(text: str) -> bool:
    return bool(_compiled()["measurement"].search(text))


# Return True when the text contains a decimal number (dot or comma as the separator).
def has_decimal_number(text: str) -> bool:
    return bool(_compiled()["decimal"].search(text))


# Return the factual-density score: 0.8 for a currency or measurement, 0.4 for a decimal number only, otherwise 0.
def fact_signal_score(text: str) -> float:
    if has_currency(text) or has_measurement(text):
        return 0.8
    if has_decimal_number(text):
        return 0.4
    return 0.0


# Return True when any factual signal is present in the text.
def is_fact_like_text(text: str) -> bool:
    return fact_signal_score(text) > 0.0
