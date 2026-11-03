# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import pytest

from core.extract.page_find import find_in_page
from core.extract.stemming import stems
from core.search.nltk_languages import (
    COMMERCE_SEEDS,
    PROJECT_LANGUAGES,
    catalog_query,
    commerce_intent_score,
    sentence_spans,
    shopping_intent,
)
from core.search.query_locales import LOCALE_PROFILES

pytestmark = pytest.mark.unit


# Verify that every project locale has commerce seeds.
def test_every_project_locale_has_commerce_seeds():
    assert {profile.language for profile in LOCALE_PROFILES} == set(PROJECT_LANGUAGES)
    assert set(PROJECT_LANGUAGES) <= set(COMMERCE_SEEDS)


# One product query per project language, with an inflected commerce word where the
# language inflects.
@pytest.mark.parametrize("language,query", [
    ("ar", "سعر ايفون 15"),
    ("de", "fahrrad günstig kaufen"),
    ("en", "buying a usb hub"),
    ("es", "comprar zapatillas"),
    ("fr", "acheter un vélo"),
    ("hi", "आईफोन की कीमतें"),
    ("id", "harga laptop"),
    ("it", "comprare una bici"),
    ("ja", "ノートパソコン購入"),
    ("ko", "노트북 가격은"),
    ("nl", "fiets kopen"),
    ("pl", "ceny rowerów"),
    ("pt", "preços de bicicleta"),
    ("ru", "куплю ноутбук"),
    ("tr", "laptop fiyatı"),
    ("uk", "ціни на ноутбуки"),
    ("vi", "giá xe đạp"),
    ("zh", "苹果手机价格"),
    ("zh", "蘋果手機價格"),
])
def test_commerce_words_are_found_in_every_project_language(language, query):
    assert commerce_intent_score(query, language=language) == 1.0
    assert shopping_intent(query, language=language)


# Verify that non commerce queries.
@pytest.mark.parametrize("query", ["высота эльбруса", "история рима", "wie spät ist es", "東京の天気"])
def test_non_commerce_queries(query):
    assert commerce_intent_score(query) == 0.0


# Verify that catalog query strips commerce words.
@pytest.mark.parametrize("language,query,expected", [
    ("zh", "苹果手机价格", "苹果手机"),
    ("tr", "laptop fiyatı", "laptop"),
    ("ko", "노트북 가격", "노트북"),
    ("uk", "купити ноутбук", "ноутбук"),
    ("de", "fahrrad kaufen", "fahrrad"),
])
def test_catalog_query_strips_commerce_words(language, query, expected):
    assert catalog_query(query, language=language) == expected


# Inflections share a key in every project language that inflects: NLTK Snowball where it
# exists, truncation keys elsewhere.
@pytest.mark.parametrize("language,forms", [
    ("ar", ["سعر", "السعر"]),
    ("de", ["preis", "preise", "preisen"]),
    ("en", ["price", "prices", "priced"]),
    ("es", ["precio", "precios"]),
    ("fr", ["chaussure", "chaussures"]),
    ("hi", ["कीमत", "कीमतें"]),
    ("id", ["harga", "harganya"]),
    ("it", ["prezzo", "prezzi"]),
    ("ko", ["가격", "가격은", "가격이"]),
    ("nl", ["boek", "boeken"]),
    ("pl", ["cena", "ceny", "cenę"]),
    ("pt", ["preço", "preços"]),
    ("ru", ["токен", "токены", "токенов", "токенами"]),
    ("tr", ["fiyat", "fiyatı", "fiyatlar"]),
    ("uk", ["ціна", "ціни", "ціну"]),
])
def test_inflections_share_a_stem(language, forms):
    keys = [stems(form, language) for form in forms]
    assert all(a & b for a in keys for b in keys), keys


# Verify that find works on cjk without spaces.
def test_find_works_on_cjk_without_spaces():
    page = "# 手机\n\n---\n\n## 价格\n\n苹果手机的价格是多少。我们比较了三家商店的价格。\n\n## 其他\n\n天气很好。"
    out = find_in_page(page, "价格")
    assert "1 passage" in out and "苹果手机的价格是多少" in out and "天气很好" not in out
    assert "§ 价格" in out


# Verify that find matches inflections in languages without snowball.
def test_find_matches_inflections_in_languages_without_snowball():
    page = "# T\n\n---\n\n## Fiyatlar\n\nBu modelin fiyatı 1000 TL. Fiyatlar her ay değişir.\n\n## Diğer\n\nHava güzel."
    out = find_in_page(page, "fiyat")
    assert "fiyatı 1000 TL" in out and "Hava güzel" not in out


# Verify that sentences split in every script.
@pytest.mark.parametrize("language,text,count", [
    ("hi", "यह कीमत है। क्या यह सस्ता है? हाँ।", 3),
    ("ar", "هذا السعر مرتفع. هل هو رخيص؟ نعم.", 3),
    ("zh", "价格很高。我们买了手机！真的吗？是的。", 4),
    ("ja", "値段は高い。本当ですか？はい。", 3),
    ("ru", "Первое предложение. Второе предложение! Третье?", 3),
])
def test_sentences_split_in_every_script(language, text, count):
    assert len(sentence_spans(text, language)) == count
