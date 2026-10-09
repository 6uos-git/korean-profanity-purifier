# coding: utf-8
"""역방향 생성: 깨끗한 문장을 일부러 오염시켜 (오염 문장, 원문, 비속어 위치) 학습 쌍을 만든다 [교과 외]

1. 깨끗한 댓글     오늘 {정말} 덥네         ({ } = 순화어 자리)
2. 비속어 주입     오늘 존나 덥네          순화어 사전을 '거꾸로' 적용
3. 변형 규칙 적용   오늘 ㅈㄴ 덥네          초성화 · 숫자/기호 삽입 · 유사 자모 치환

라벨링 비용 없이 (오염 문장, 원문, 글자 단위 비속어 위치) 세트를 원하는 만큼 만든다.
실제 데이터셋(KOLD·UnSmile·HateSpeech 등)은 라이선스상 재배포하지 않고 loader.py의 읽기 함수로만 연결한다.
"""
import random
from jamo import CHO, JUNG, JONG, SBASE, SEND, to_choseong

# ── 순화어 → 비속어 (역방향 사전). 정방향(비속어→순화어)은 purifier.py가 후보 목록으로 쓴다 ──
REVERSE = {
    '정말': ['존나', '졸라', '좆나', '개'],
    '너무': ['존나', '개', '졸라', '씨발'],
    '진짜': ['존나', '씨발', '좆나'],
    '아': ['씨발', '시발', '아 씨발', '젠장'],
    '사람': ['새끼', '놈', '년'],
    '애들': ['새끼들', '놈들'],
    '바보': ['병신', '등신', '븅신'],
    '바보같이': ['병신같이', '등신같이'],
    '화나게': ['빡치게', '좆같게'],
    '화나네': ['빡치네', '좆같네'],
    '화난다': ['빡친다', '좆같다'],
    '짜증나': ['빡쳐', '좆같아'],
    '별로네': ['좆같네', '구리네'],
    '시끄러워': ['닥쳐', '아가리 닥쳐'],
    '먹어': ['처먹어', '쳐먹어'],
    # 실제 댓글에서 흔한 형태 (조사·어미가 붙은 꼴 그대로) — real_data.py 역방향 생성용
    '사람들': ['새끼들', '놈들', '년들', '병신들'],
    '사람들아': ['새끼들아', '놈들아', '병신들아'],
    '사람이': ['새끼가', '놈이', '년이'],
    '사람은': ['새끼는', '놈은', '년은'],
    '난리': ['지랄', '염병'],
    '난리네': ['지랄하네', '지랄이네', '염병하네'],
    '이상한': ['미친', '돌은'],
    '이상하다': ['좆같다', '개좆같다', '미쳤다'],
    '저리 가': ['꺼져', '꺼지라고'],
}
PROFANITY = sorted({w for vs in REVERSE.values() for w in vs if w not in ('구리네', '젠장')}, key=len, reverse=True)

# ── 깨끗한 문장 템플릿. { } 안은 비속어로 바뀔 수 있는 순화어 ──
TEMPLATES = [
    '{아} 오늘 {정말} 덥네', '오늘 {정말} 덥네', '오늘 {너무} 춥다', '날씨가 {너무} 길다', '날씨가 {정말} 좋다',
    '시험이 {정말} 어렵네', '시험이 {너무} 쉬웠다', '과제가 {진짜} 많다', '{아} 과제 {너무} 많아',
    '이 영화 {진짜} 재밌다', '이 노래 {정말} 좋다', '라면이 {정말} 맛있다', '{정말} 맛있다', '{진짜} 웃기네',
    '그 {사람} {진짜} 이상하다', '그 {사람} {진짜} {화나게} 하네', '저 {사람} 왜 저래', '그 {사람} 말 {너무} 많다',
    '그 {사람} 때문에 {화난다}', '저 {사람} {바보같이} 굴어', '{바보}처럼 왜 그래', '너 {바보}야?',
    '그 {애들} {너무} 시끄럽다', '옆집 {애들} {진짜} {화나게} 하네', '{시끄러워} 좀', '조용히 해 {시끄러워}',
    '버스가 또 늦네 {화나네}', '{아} 지하철 {너무} 붐빈다', '{아} 또 지각이네', '{아} 비 온다',
    '서비스가 {별로네}', '음식이 {별로네}', '이 게임 {진짜} {별로네}', '배송이 {너무} 느리다', '가격이 {너무} 비싸다',
    '{아} {짜증나}', '오늘 일 {너무} 많아서 {짜증나}', '팀플 {진짜} {화난다}', '상사가 {정말} {화나게} 하네',
    '빨리 밥 {먹어}', '그냥 조용히 밥 {먹어}', '배고프면 이거 {먹어}', '노래 {정말} 잘 부른다', '경기 {진짜} 잘했다',
    '{아} 졌다', '이번 판 {정말} 아쉽다', '오늘 {너무} 피곤하다', '{정말} 고맙다', '{너무} 보고 싶다',
    '어제 {너무} 늦게 잤다', '주말에 {정말} 쉬고 싶다', '그 {사람} {정말} 착하다', '수업이 {너무} 길다',
    '{아} 수업 듣기 싫다', '컴퓨터가 {너무} 느려', '와이파이 {진짜} 안 된다', '{아} 폰 배터리 다 됐네',
    '이번 업데이트 {진짜} {별로네}', '운전 {바보같이} 하네', '저 차 {바보같이} 끼어드네', '{아} 길 {너무} 막힌다',
    '{정말} 대단하다', '{진짜} 멋있다', '그 {사람} 노래 {정말} 잘한다', '이 집 {진짜} 맛집이다', '{너무} 덥다',
    '줄이 {너무} 길다', '{아} 또 버그네', '코드가 {정말} 안 돌아간다', '디버깅 {너무} 힘들다', '{아} 컴파일 에러',
]
# 비속어 없이 그대로 나와야 하는 문장 (과교정 금지 · 혼동 단어 포함)
CLEAN_TRAPS = [
    '개가 너무 귀엽다', '우리 집 개가 짖는다', '개나리가 피었다', '수업이 끝났다', '시발점과 종착점', '새끼 고양이가 귀엽다',
    '강아지가 새끼를 낳았다', '개구리가 운다', '개미가 줄지어 간다', '시베리아는 춥다', '병원에 갔다', '신발을 샀다',
    '존경하는 선생님', '조나단이 왔다', '년도별 자료', '놈놈놈 영화 봤다', '개강했다', '개인 과제 제출', '시발역에서 내렸다',
    '지금 몇 시야', '오늘 날씨 좋다', '내일 보자', '점심 뭐 먹지', '수고했어', '고마워 친구야', '열심히 하자',
]

TENSE = {'ㄱ': 'ㄲ', 'ㄷ': 'ㄸ', 'ㅂ': 'ㅃ', 'ㅅ': 'ㅆ', 'ㅈ': 'ㅉ'}
TENSE.update({v: k for k, v in TENSE.items()})
VOWEL_SWAP = {'ㅐ': 'ㅔ', 'ㅔ': 'ㅐ', 'ㅗ': 'ㅡ', 'ㅕ': 'ㅛ', 'ㅣ': 'ㅢ'}


def _syllables(word):
    return [(ord(c) - SBASE) if SBASE <= ord(c) <= SEND else None for c in word]


def similar_jamo(word, rng):
    """유사 자모 치환: 씨발→시발, 존나→쫀나, 새끼→세끼"""
    chars = list(word)
    idxs = [i for i, s in enumerate(_syllables(word)) if s is not None]
    if not idxs:
        return word
    i = rng.choice(idxs)
    s = ord(chars[i]) - SBASE
    cho, jung, jong = s // 588, (s % 588) // 28, s % 28
    if CHO[cho] in TENSE and rng.random() < 0.6:
        cho = CHO.index(TENSE[CHO[cho]])
    elif JUNG[jung] in VOWEL_SWAP:
        jung = JUNG.index(VOWEL_SWAP[JUNG[jung]])
    chars[i] = chr(SBASE + cho * 588 + jung * 28 + jong)
    return ''.join(chars)


def insert_noise(word, rng):
    """숫자·기호 삽입: 시발→시1발, 존나→존.나"""
    if len(word) < 2:
        return word
    pos = rng.randint(1, len(word) - 1)
    return word[:pos] + rng.choice(['1', '8', '.', '*', '0']) + word[pos:]


def mutate(word, rng, p=0.5):
    """변형 규칙 하나를 확률 p로 적용"""
    if rng.random() >= p or ' ' in word:
        return word
    rule = rng.choice(['cho', 'noise', 'similar'])
    if rule == 'cho' and len(word) >= 2:
        return to_choseong(word)
    if rule == 'noise':
        return insert_noise(word, rng)
    return similar_jamo(word, rng)


def parse_template(tpl):
    """'오늘 {정말} 덥네' → [('오늘 ', None), ('정말', '정말'), (' 덥네', None)]"""
    segs, i = [], 0
    while i < len(tpl):
        if tpl[i] == '{':
            j = tpl.index('}', i)
            w = tpl[i + 1:j]
            segs.append((w, w))
            i = j + 1
        else:
            j = tpl.find('{', i)
            j = len(tpl) if j < 0 else j
            segs.append((tpl[i:j], None))
            i = j
    return segs


def clean_text(tpl):
    return ''.join(t for t, _ in parse_template(tpl))


def corrupt(tpl, rng, p_inject=0.7, p_mutate=0.5):
    """템플릿 → (오염 문장, 원문, 글자 단위 0/1 라벨). 최소 한 자리는 오염시킨다."""
    segs = parse_template(tpl)
    slots = [k for k, (_, c) in enumerate(segs) if c in REVERSE]
    must = rng.choice(slots) if slots else None
    dirty, labels = '', []
    for k, (text, c) in enumerate(segs):
        if c in REVERSE and (k == must or rng.random() < p_inject):
            bad = mutate(rng.choice(REVERSE[c]), rng, p_mutate)
            dirty += bad
            labels += [0 if ch == ' ' else 1 for ch in bad]
        else:
            dirty += text
            labels += [0] * len(text)
    return dirty, clean_text(tpl), labels


def make_pairs(n, templates=TEMPLATES, traps=CLEAN_TRAPS, p_clean=0.2, seed=0):
    """n개의 (dirty, clean, labels) 생성. p_clean 비율은 비속어 없는 문장(원문=출력)."""
    rng = random.Random(seed)
    pairs = []
    for _ in range(n):
        if rng.random() < p_clean:
            s = rng.choice(traps) if rng.random() < 0.5 else clean_text(rng.choice(templates))
            pairs.append((s, s, [0] * len(s)))
        else:
            pairs.append(corrupt(rng.choice(templates), rng))
    return pairs


def split_templates(ratio=0.8, seed=0):
    """템플릿 단위로 학습/평가 분할 → 평가 문장은 학습 때 본 적 없는 문장"""
    rng = random.Random(seed)
    tpls, traps = TEMPLATES[:], CLEAN_TRAPS[:]
    rng.shuffle(tpls)
    rng.shuffle(traps)
    a, b = int(len(tpls) * ratio), int(len(traps) * ratio)
    return (tpls[:a], traps[:b]), (tpls[a:], traps[b:])


def clean_corpus(templates=TEMPLATES, traps=CLEAN_TRAPS):
    """언어 모델 학습용 깨끗한 문장 전체"""
    return [clean_text(t) for t in templates] + list(traps)


if __name__ == '__main__':
    for dirty, clean, labels in make_pairs(12, seed=1):
        mark = ''.join('^' if l else ' ' for l in labels)
        print(f'{clean:22} → {dirty}\n{"":25}{mark}')
