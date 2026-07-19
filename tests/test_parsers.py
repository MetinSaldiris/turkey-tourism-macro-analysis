import pandas as pd
from src.data_fetch import _norm_tr, _to_number, _as_year


def test_norm_tr_dotted_capital_i():
    assert _norm_tr("İSTANBUL") == "istanbul"


def test_norm_tr_dotless_i():
    assert _norm_tr("Kasım") == "kasim"


def test_norm_tr_full_month_set():
    from src.data_fetch import TUIK_MONTHS_TR
    assert TUIK_MONTHS_TR[_norm_tr("Ağustos")] == 8
    assert TUIK_MONTHS_TR[_norm_tr("EYLÜL")] == 9


def test_to_number_turkish_thousands():
    assert _to_number("1.234.567") == 1234567.0


def test_to_number_decimal_comma():
    assert _to_number("12,5") == 12.5


def test_to_number_dash_and_empty_are_na():
    assert pd.isna(_to_number("-"))
    assert pd.isna(_to_number(""))


def test_as_year_float_and_bounds():
    assert _as_year(2019.0) == 2019
    assert _as_year(2019.5) is None
    assert _as_year(1999) is None
