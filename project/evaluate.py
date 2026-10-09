# coding: utf-8
"""평가 지표: 구간 F1, STA(잔존율), SIM(문자 BLEU), FL(퍼플렉서티), J [교과 외 + Ch05]

'욕설만 지우면 끝'이 아니다. 비속어를 전부 지우면 STA는 만점이지만 의미가 사라지므로
국제 순화 벤치마크(TextDetox·ParaDetox)처럼 STA × SIM × FL 로 균형을 본다.
"""
import math
from collections import Counter
from jamo import normalize

# 손으로 쓴 평가 문장 (템플릿 생성과 별개). (입력, 정답 순화문). 비속어 없는 문장은 정답 = 입력.
# 실제 욕설 문장 200개 평가 세트는 KOLD/Curse-detection에서 추출해 같은 형식으로 추가한다.
HANDWRITTEN = [
    ('시1발 오늘 존나 덥네', '아 오늘 정말 덥네'),
    ('그 새끼 진짜 개빡치게 하네', '그 사람 진짜 화나게 하네'),
    ('ㅈㄴ 맛있다', '정말 맛있다'),
    ('개가 너무 귀엽다', '개가 너무 귀엽다'),
    ('시험이 쫀나 덥네', '시험이 정말 덥네'),
    ('날씨가 개 길다', '날씨가 너무 길다'),
    ('오늘 졸라 덥네', '오늘 정말 덥네'),
    ('수업이 끝났다', '수업이 끝났다'),
    ('ㅅㅂ 과제 존나 많아', '아 과제 너무 많아'),
    ('저 ㅅㄲ 왜 저래', '저 사람 왜 저래'),
    ('배송이 존1나 느리다', '배송이 너무 느리다'),
    ('씨발 버스 또 늦네', '아 버스 또 늦네'),
    ('이 영화 존잼 아니고 좆노잼', '이 영화 존잼 아니고 정말 노잼'),
    ('ㅂㅅ같이 운전하네', '바보같이 운전하네'),
    ('그 놈 말 개많다', '그 사람 말 너무 많다'),
    ('옆집 새끼들 존나 시끄럽다', '옆집 애들 너무 시끄럽다'),
    ('닥쳐 좀', '시끄러워 좀'),
    ('밥이나 처먹어', '밥이나 먹어'),
    ('개나리가 피었다', '개나리가 피었다'),
    ('새끼 고양이가 귀엽다', '새끼 고양이가 귀엽다'),
    ('시발역에서 내렸다', '시발역에서 내렸다'),
    ('팀플 진짜 빡친다', '팀플 진짜 화난다'),
    ('와이파이 ㅈㄴ 안 된다', '와이파이 정말 안 된다'),
    ('아 씨8 또 버그네', '아 또 버그네'),
    ('오늘 일 존나 많아서 빡쳐', '오늘 일 너무 많아서 짜증나'),
]


def span_prf(pred_labels, gold_labels):
    """글자 단위 비속어 구간 정밀도/재현율/F1 (마이크로 평균)"""
    tp = fp = fn = 0
    for p, g in zip(pred_labels, gold_labels):
        for a, b in zip(p, g):
            tp += a and b
            fp += a and not b
            fn += b and not a
    prec = tp / (tp + fp) if tp + fp else 1.0
    rec = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def char_bleu(hyp, ref, n_max=4):
    """문자 단위 문장 BLEU (+1 평활화). 한국어는 띄어쓰기가 흔들려 공백을 제외한다."""
    h, r = hyp.replace(' ', ''), ref.replace(' ', '')
    if not h:
        return 0.0
    logs = []
    for n in range(1, n_max + 1):
        hc = Counter(h[i:i + n] for i in range(len(h) - n + 1))
        rc = Counter(r[i:i + n] for i in range(len(r) - n + 1))
        match = sum(min(c, rc[g]) for g, c in hc.items())
        total = max(len(h) - n + 1, 0)
        logs.append(math.log((match + 1) / (total + 1)))
    bp = 1.0 if len(h) > len(r) else math.exp(1 - len(r) / len(h))
    return bp * math.exp(sum(logs) / n_max)


def evaluate(purify_fn, pairs, detector_flag_fn=None, lm=None, vocab=None):
    """pairs: [(입력, 정답)]. purify_fn(text) → 출력 문자열.
    STA_i: 출력에 비속어가 남지 않음 (detector_flag_fn이 있으면 탐지기 판정, 없으면 정답과 다른 원문 어절 잔존으로 판정)
    SIM_i: 정답 대비 문자 BLEU
    FL_i : min(1, ppl(정답) / ppl(출력))  — 정답만큼 자연스러우면 1
    """
    outs = [purify_fn(x) for x, _ in pairs]
    golds = [normalize(g) for _, g in pairs]
    srcs = [normalize(x) for x, _ in pairs]

    sta = []
    for src, out, gold in zip(srcs, outs, golds):
        if detector_flag_fn is not None:
            sta.append(0.0 if detector_flag_fn(out) else 1.0)
        else:
            bad = set(src.split()) - set(gold.split())  # 입력에만 있는 어절 = 비속어 어절
            sta.append(0.0 if any(w in out.split() for w in bad) else 1.0)
    sim = [char_bleu(o, g) for o, g in zip(outs, golds)]
    if lm is not None:
        po = lm.sentence_perplexity([vocab.encode(o)[0] for o in outs])
        pg = lm.sentence_perplexity([vocab.encode(g)[0] for g in golds])
        fl = [min(1.0, float(g / o)) for o, g in zip(po, pg)]
    else:
        fl = [1.0] * len(outs)
    j = [a * b * c for a, b, c in zip(sta, sim, fl)]
    exact = sum(o == g for o, g in zip(outs, golds)) / len(outs)
    clean_idx = [i for i, (s, g) in enumerate(zip(srcs, golds)) if s == g]
    over = sum(outs[i] != golds[i] for i in clean_idx) / max(len(clean_idx), 1)  # 과교정률
    mean = lambda v: sum(v) / len(v)
    return {'STA': mean(sta), 'SIM': mean(sim), 'FL': mean(fl), 'J': mean(j),
            'EM': exact, '과교정': over, 'outputs': outs}
