import unicodedata

import pytest

from p01kit.retrieval import hit_at_k, mrr_at_k
from p01kit.vntext import has_nfd, nfc, normalize, normalize_tone, strip_diacritics, syllables, tone_style


def test_nfc_and_nfd_look_identical_but_differ():
    composed = "Bảo hiểm xã hội"
    decomposed = unicodedata.normalize("NFD", composed)
    assert composed != decomposed and nfc(decomposed) == composed
    assert has_nfd(decomposed) and not has_nfd(composed)
    assert syllables(decomposed) != syllables(composed)                 # máy tách âm tiết cũng khác
    assert syllables(nfc(decomposed)) == ["Bảo", "hiểm", "xã", "hội"]


@pytest.mark.parametrize("old,new", [("hòa", "hoà"), ("khỏe", "khoẻ"), ("thủy", "thuỷ"), ("Hòa Bình", "Hoà Bình"),
                                     ("tùy", "tuỳ"), ("HÒA", "HOÀ"), ("lòe", "loè")])
def test_tone_placement_both_ways(old, new):
    assert normalize_tone(old, "new") == new
    assert normalize_tone(new, "old") == old
    assert normalize_tone(normalize_tone(old, "new"), "new") == new          # idempotent


@pytest.mark.parametrize("word", ["hoàng", "hoài", "quý", "quỳnh", "khuyên", "hoa", "thuyền", "của", "quá", "hội"])
def test_tone_rule_leaves_other_syllables(word):
    assert normalize_tone(word, "new") == word == normalize_tone(word, "old")


def test_tone_style_counts_mixed_text():
    assert tone_style("hòa giải, hoà nhập, thủy lợi, khỏe mạnh, hoàng") == {"new": 1, "old": 3}
    with pytest.raises(ValueError):
        normalize_tone("x", "giua")


def test_normalize_pipeline():
    q = unicodedata.normalize("NFD", "Trợ cấp thôi việc khi HÒA giải")
    assert normalize(q) == "trợ cấp thôi việc khi hoà giải"
    assert normalize(q, diacritics=False) == "tro cap thoi viec khi hoa giai"
    assert strip_diacritics("Đường") == "Duong"


def test_segment_with_pyvi():
    pytest.importorskip("pyvi")
    from p01kit.vntext import segment

    assert "bảo_hiểm" in segment("Luật bảo hiểm xã hội")


def test_hit_and_mrr_with_multiple_gold():
    ranked = [["a", "b", "c"], ["x", "y", "z"], ["q", "r", "s"]]
    gold = [{"c", "zz"}, {"x"}, {"none"}]
    assert hit_at_k(ranked, gold, 1) == pytest.approx(1 / 3)
    assert hit_at_k(ranked, gold, 3) == pytest.approx(2 / 3)
    assert mrr_at_k(ranked, gold, 10) == pytest.approx((1 / 3 + 1) / 3)
    with pytest.raises(ValueError):
        hit_at_k(ranked, gold[:2], 1)
