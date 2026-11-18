"""Xử lý văn bản tiếng Việt thật — ba "cái bẫy" mà dữ liệu sinh bằng luật không có.

1. **Hai cách mã hóa Unicode cho cùng một chữ.** "ộ" có thể là 1 ký tự dựng sẵn (NFC) hoặc "o" + dấu mũ +
   dấu nặng (NFD, hay gặp khi copy từ PDF/Word cũ hoặc gõ trên macOS). Mắt người thấy giống hệt, máy so chuỗi
   thấy khác → tìm kiếm trượt.
2. **Hai kiểu đặt dấu thanh.** "hoà / hòa", "thuỷ / thủy", "khoẻ / khỏe" đều được dùng trong văn bản thật.
3. **Từ ghép.** "bảo hiểm xã hội" là 2 từ, 4 âm tiết. Nhiều mô hình tiếng Việt (PhoBERT, `vietnamese-bi-encoder`)
   đòi đầu vào đã tách từ: `bảo_hiểm xã_hội`.
"""

from __future__ import annotations

import re
import unicodedata

TONES = {"̀", "́", "̉", "̃", "̣"}       # huyền, sắc, hỏi, ngã, nặng
_WORD_RE = re.compile(r"\w+")
_PAIRS = ("oa", "oe", "uy")                                       # cặp nguyên âm có 2 kiểu đặt dấu


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def has_nfd(text: str) -> bool:
    """True nếu chuỗi có ký tự ở dạng tổ hợp (sẽ đổi khi chuẩn hóa NFC)."""
    return nfc(text) != text


def _retone_word(word: str, style: str) -> str:
    d = unicodedata.normalize("NFD", word)
    tones = [c for c in d if c in TONES]
    if len(tones) != 1:
        return word
    tone = tones[0]
    # Tách thành các "khối" chữ cái gốc + dấu phụ (mũ/móc/trăng) đi kèm, bỏ dấu thanh ra.
    blocks: list[str] = []
    for c in d:
        if c in TONES:
            continue
        if unicodedata.combining(c) and blocks:
            blocks[-1] += c
        else:
            blocks.append(c)
    bases = "".join(b[0] for b in blocks).lower()
    for pair in _PAIRS:
        if not bases.endswith(pair) or len(bases) < 2:
            continue
        i = len(bases) - 2                                       # vị trí nguyên âm thứ nhất của cặp
        if pair == "uy" and i > 0 and bases[i - 1] == "q":        # "quý": "qu" là phụ âm → không đổi
            return word
        if any(len(blocks[j]) > 1 for j in (i, i + 1)):          # có mũ/móc (ví dụ "uyê") → không thuộc luật này
            return word
        target = i + 1 if style == "new" else i
        blocks[target] += tone
        return unicodedata.normalize("NFC", "".join(blocks))
    return word


def normalize_tone(text: str, style: str = "new") -> str:
    """Thống nhất vị trí dấu thanh cho âm tiết kết thúc bằng oa/oe/uy.

    `style="new"`: hoà, khoẻ, thuỷ (dấu trên nguyên âm thứ hai) · `style="old"`: hòa, khỏe, thủy.
    Âm tiết có phụ âm cuối (hoàn, khoẻo...) hay "quý" giữ nguyên vì cả hai kiểu đều đặt dấu như nhau.
    """
    if style not in ("new", "old"):
        raise ValueError("style phải là 'new' hoặc 'old'")
    return _WORD_RE.sub(lambda m: _retone_word(m.group(0), style), nfc(text))


def tone_style(text: str) -> dict:
    """Đếm số từ oa/oe/uy viết theo kiểu mới và kiểu cũ — để biết dữ liệu có trộn 2 kiểu không."""
    new = old = 0
    for w in _WORD_RE.findall(nfc(text)):
        n, o = _retone_word(w, "new"), _retone_word(w, "old")
        if n != o:
            new += w == n
            old += w == o
    return {"new": new, "old": old}


def strip_diacritics(text: str) -> str:
    """Bỏ toàn bộ dấu (kể cả đ → d): mô phỏng người dùng gõ không dấu."""
    s = unicodedata.normalize("NFD", text).replace("đ", "d").replace("Đ", "D")
    return unicodedata.normalize("NFC", "".join(c for c in s if not unicodedata.combining(c)))


def normalize(text: str, tone: str | None = "new", lower: bool = True, diacritics: bool = True) -> str:
    """Chuẩn hóa dùng chung cho truy xuất: NFC → (thống nhất dấu thanh) → (chữ thường) → (bỏ dấu)."""
    t = nfc(text)
    if tone:
        t = normalize_tone(t, tone)
    if lower:
        t = t.lower()
    if not diacritics:
        t = strip_diacritics(t)
    return t


def syllables(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def segment(text: str, backend: str = "pyvi") -> list[str]:
    """Tách từ: trả về danh sách từ, âm tiết trong từ ghép nối bằng '_' (bảo_hiểm, xã_hội).

    Cần cài `pyvi` (nhẹ, CRF) hoặc `underthesea`. Mô hình PhoBERT/`vietnamese-bi-encoder` yêu cầu đúng định dạng này.
    """
    if backend == "pyvi":
        from pyvi import ViTokenizer

        return [w for w in ViTokenizer.tokenize(text).split() if _WORD_RE.search(w)]
    if backend == "underthesea":
        from underthesea import word_tokenize

        return [w.replace(" ", "_") for w in word_tokenize(text) if _WORD_RE.search(w)]
    raise ValueError("backend phải là 'pyvi' hoặc 'underthesea'")
