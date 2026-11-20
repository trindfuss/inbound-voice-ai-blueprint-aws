"""Server-side formatting for spoken values (handover "Templates and formatting").

* Money is spelled out as words before it reaches TTS, so the voice can't misread
  it: ``1234.56`` -> "one thousand two hundred thirty-four dollars and fifty-six
  cents" / "mil doscientos treinta y cuatro dólares con cincuenta y seis centavos".
* Account digits are read one at a time: "four, eight, two, one".

Amounts are decimal strings (never floats). Negative amounts are rejected here:
callers pick the overdrawn template and pass the absolute value.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

MAX_AMOUNT = Decimal("999999999.99")

_EN_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
            "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
            "seventeen", "eighteen", "nineteen"]
_EN_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

_ES_ONES = ["cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve",
            "diez", "once", "doce", "trece", "catorce", "quince", "dieciséis", "diecisiete",
            "dieciocho", "diecinueve", "veinte", "veintiuno", "veintidós", "veintitrés",
            "veinticuatro", "veinticinco", "veintiséis", "veintisiete", "veintiocho",
            "veintinueve"]
_ES_TENS = ["", "", "", "treinta", "cuarenta", "cincuenta", "sesenta", "setenta", "ochenta",
            "noventa"]
_ES_HUNDREDS = ["", "ciento", "doscientos", "trescientos", "cuatrocientos", "quinientos",
                "seiscientos", "setecientos", "ochocientos", "novecientos"]


def _family(lang: str) -> str:
    return "es" if str(lang).lower().startswith("es") else "en"


# --------------------------------------------------------------------------- English


def _en_below_1000(n: int) -> str:
    parts: list[str] = []
    if n >= 100:
        parts.append(f"{_EN_ONES[n // 100]} hundred")
        n %= 100
    if n >= 20:
        parts.append(_EN_TENS[n // 10] + (f"-{_EN_ONES[n % 10]}" if n % 10 else ""))
    elif n > 0:
        parts.append(_EN_ONES[n])
    return " ".join(parts)


def en_number(n: int) -> str:
    if n == 0:
        return "zero"
    parts: list[str] = []
    for scale, word in ((1_000_000, "million"), (1000, "thousand")):
        if n >= scale:
            parts.append(f"{_en_below_1000(n // scale)} {word}")
            n %= scale
    if n:
        parts.append(_en_below_1000(n))
    return " ".join(parts)


# --------------------------------------------------------------------------- Spanish


def _es_below_100(n: int, apocope: bool) -> str:
    if n < 30:
        word = _ES_ONES[n]
        if apocope and n == 1:
            return "un"
        if apocope and n == 21:
            return "veintiún"
        return word
    tens, unit = divmod(n, 10)
    if not unit:
        return _ES_TENS[tens]
    return f"{_ES_TENS[tens]} y {'un' if apocope and unit == 1 else _ES_ONES[unit]}"


def _es_below_1000(n: int, apocope: bool) -> str:
    if n == 100:
        return "cien"
    hundreds, rest = divmod(n, 100)
    parts: list[str] = []
    if hundreds:
        parts.append(_ES_HUNDREDS[hundreds])
    if rest:
        parts.append(_es_below_100(rest, apocope))
    return " ".join(parts)


def es_number(n: int, *, apocope: bool = False) -> str:
    """Spanish cardinal. ``apocope`` gives the form before a masculine noun ("un", "veintiún")."""
    if n == 0:
        return "cero"
    parts: list[str] = []
    millions, n = divmod(n, 1_000_000)
    if millions:
        parts.append("un millón" if millions == 1
                     else f"{_es_below_1000(millions, True)} millones")
    thousands, n = divmod(n, 1000)
    if thousands:
        parts.append("mil" if thousands == 1 else f"{_es_below_1000(thousands, True)} mil")
    if n:
        parts.append(_es_below_1000(n, apocope))
    return " ".join(parts)


# --------------------------------------------------------------------------- money


def _split_amount(amount: str | Decimal) -> tuple[int, int]:
    try:
        value = Decimal(str(amount))
    except (InvalidOperation, ValueError):
        raise ValueError("amount must be a decimal string") from None
    if not value.is_finite() or value < 0 or value > MAX_AMOUNT:
        raise ValueError("amount out of range")
    cents_total = int((value * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    return divmod(cents_total, 100)


def money_words(amount: str | Decimal, lang: str) -> str:
    """Spell a non-negative USD amount out in words, in ``lang``."""
    dollars, cents = _split_amount(amount)
    if _family(lang) == "es":
        def unit(n: int, one: str, many: str) -> str:
            if n == 1:
                return f"un {one}"
            # "un millón de dólares": exact millions take "de"
            sep = " de " if n >= 1_000_000 and n % 1_000_000 == 0 else " "
            return f"{es_number(n, apocope=True)}{sep}{many}"
        if dollars and cents:
            return f"{unit(dollars, 'dólar', 'dólares')} con {unit(cents, 'centavo', 'centavos')}"
        if cents:
            return unit(cents, "centavo", "centavos")
        return unit(dollars, "dólar", "dólares")

    def en_unit(n: int, one: str, many: str) -> str:
        return f"{en_number(n)} {one if n == 1 else many}"
    if dollars and cents:
        return f"{en_unit(dollars, 'dollar', 'dollars')} and {en_unit(cents, 'cent', 'cents')}"
    if cents:
        return en_unit(cents, "cent", "cents")
    return en_unit(dollars, "dollar", "dollars")


def digits_words(digits: str, lang: str) -> str:
    """Read digits one at a time: ``"4821"`` -> "four, eight, two, one"."""
    if not digits.isascii() or not digits.isdigit():
        raise ValueError("digits only")
    ones = _ES_ONES if _family(lang) == "es" else _EN_ONES
    return ", ".join(ones[int(d)] for d in digits)
