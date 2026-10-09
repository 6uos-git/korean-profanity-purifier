# coding: utf-8
"""정규화 + 자모 분해·결합 [교과 외]

'시1발'과 '씨발'이 ㅅ/ㅆ 하나 차이인 시퀀스가 되도록 음절을 초성·중성·종성으로 쪼갠다.
각 자모 토큰이 원문의 몇 번째 글자에서 왔는지(char_idx)를 같이 돌려줘서
탐지기의 자모별 확률을 다시 글자 위치로 되돌릴 수 있게 한다.
"""
import re
import unicodedata

CHO = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
JUNG = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
JONG = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ',
        'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

# 종성은 초성과 같은 글자라도 다른 토큰으로 둔다 ('ㄴ' 초성 vs '_ㄴ' 종성) → 음절 경계 보존
JONG_TOKENS = [''] + ['_' + j for j in JONG[1:]]

SBASE, SEND = 0xAC00, 0xD7A3


def normalize(text):
    """NFC 정규화, 소문자화, 반복 문자 축약(ㅋㅋㅋㅋ→ㅋㅋ), 허용 문자 외 제거, 공백 정리"""
    text = unicodedata.normalize('NFC', text).lower()
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'[^0-9a-z가-힣ㄱ-ㅎㅏ-ㅣ\s.,!?~*]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def decompose(text):
    """문자열 → (자모 토큰 리스트, 각 토큰의 원문 글자 인덱스 리스트)"""
    tokens, char_idx = [], []
    for i, ch in enumerate(text):
        code = ord(ch)
        if SBASE <= code <= SEND:
            s = code - SBASE
            cho, jung, jong = s // 588, (s % 588) // 28, s % 28
            parts = [CHO[cho], JUNG[jung]]
            if jong:
                parts.append(JONG_TOKENS[jong])
        elif ch == ' ':
            parts = ['▁']
        else:
            parts = [ch]  # 낱자모(ㅅㅂ), 숫자, 기호는 그대로 한 토큰
        tokens += parts
        char_idx += [i] * len(parts)
    return tokens, char_idx


def compose(tokens):
    """자모 토큰 리스트 → 문자열 (decompose의 역연산)"""
    out, i, n = [], 0, len(tokens)
    while i < n:
        t = tokens[i]
        if t in CHO and i + 1 < n and tokens[i + 1] in JUNG:
            cho, jung = CHO.index(t), JUNG.index(tokens[i + 1])
            jong = 0
            if i + 2 < n and tokens[i + 2] in JONG_TOKENS[1:]:
                jong = JONG_TOKENS.index(tokens[i + 2])
            out.append(chr(SBASE + cho * 588 + jung * 28 + jong))
            i += 3 if jong else 2
        else:
            out.append(' ' if t == '▁' else t.lstrip('_') if t.startswith('_') and len(t) > 1 else t)
            i += 1
    return ''.join(out)


def to_choseong(word):
    """'존나' → 'ㅈㄴ' (초성화 변형 규칙에 사용)"""
    out = []
    for ch in word:
        code = ord(ch)
        out.append(CHO[(code - SBASE) // 588] if SBASE <= code <= SEND else ch)
    return ''.join(out)


if __name__ == '__main__':
    for s in ['시1발 오늘 존나 덥네', '씨발', 'ㅈㄴ 맛있다ㅋㅋㅋㅋㅋ', '개가 너무 귀엽다!!!']:
        n = normalize(s)
        toks, idx = decompose(n)
        print(f'{s!r:28} → {n!r}')
        print('   자모:', ' '.join(toks))
        print('   결합:', compose(toks), '| 왕복 일치:', compose(toks) == n)
