# coding: utf-8
"""탐지 → 분기 → 후보 치환 → 언어 모델 선택 → 재검증 (교과 외 + Ch05)

욕설 재생성·과교정을 막는 3중 장치 [교과 외]
 ① 깨끗한 후보만: 순화 후보는 정해진 순화어 목록에서만 고르고, 언어 모델도 깨끗한 문장으로만 학습
 ② 탐지기 재검증: 바꿔 넣은 표현이 다시 비속어로 판정되면 다음 후보로
 ③ 원래 표현도 후보: 탐지기가 잘못 잡은 정상 표현은 언어 모델이 더 자연스럽다고 보면 그대로 남김
"""
import re
from jamo import normalize
from make_pairs import REVERSE, PROFANITY

CANDIDATES = list(REVERSE.keys())          # 순화어 목록 (정말, 너무, 사람, 화나게 ...)
INVERSE = {}                                # 고정 치환용: 비속어 → 첫 번째 순화어
for clean, bads in REVERSE.items():
    for b in bads:
        INVERSE.setdefault(b, clean)


def find_spans(char_probs, text, thr, low_ratio=0.3):
    """글자 확률 → 연속 구간 [(start, end)]. 공백은 구간을 끊는다.
    히스테리시스: thr 이상인 글자에서 시작해, 같은 어절 안의 이웃 글자가 thr*low_ratio 이상이면 구간을 넓힌다
    (실제 댓글에서 '병신새끼들'의 일부만 잡혀 '진짜신사람' 같은 조각 출력이 나오던 문제 대응)."""
    low = thr * low_ratio
    on = [ch != ' ' and p >= thr for ch, p in zip(text, char_probs)]
    changed = True
    while changed:
        changed = False
        for i, ch in enumerate(text):
            if not on[i] and ch != ' ' and char_probs[i] >= low and \
                    ((i > 0 and on[i - 1]) or (i + 1 < len(text) and on[i + 1])):
                on[i] = changed = True
    spans, start = [], None
    for i in range(len(text)):
        if on[i] and start is None:
            start = i
        if not on[i] and start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(text)))
    return spans


def _join(left, mid, right):
    return re.sub(r'\s+', ' ', left + mid + right).strip()


class Purifier:
    def __init__(self, detector, lm, vocab, thr=0.5, mode='lm', recheck=True, keep_original=True):
        """mode='lm' → E4(언어 모델 후보 선택), mode='fixed' → E2/E3(고정 치환)"""
        self.det, self.lm, self.vocab = detector, lm, vocab
        self.thr, self.mode = thr, mode
        self.recheck, self.keep_original = recheck, keep_original

    def char_probs(self, text):
        ids, char_idx = self.vocab.encode(text)
        if len(ids) == 0:
            return [0.0] * len(text)
        p = self.det.predict([ids])[0]
        out = [0.0] * len(text)
        for prob, c in zip(p, char_idx):
            out[c] = max(out[c], float(prob))
        return out

    def detect(self, text):
        text = normalize(text)
        probs = self.char_probs(text)
        return text, find_spans(probs, text, self.thr), probs

    def purify(self, text, verbose=False):
        text, spans, probs = self.detect(text)
        log = []
        if not spans:                                    # 분기: 비속어 없으면 원문 그대로
            return {'input': text, 'output': text, 'spans': [], 'log': log}

        found = [text[s:e] for s, e in spans]
        out = text
        # 뒤 구간부터 바꿔야 앞 구간 인덱스가 밀리지 않는다
        for s, e in reversed(spans):
            span = out[s:e]
            left, right = out[:s], out[e:]
            if self.mode == 'fixed':
                out = _join(left, INVERSE.get(span, '**'), right)
                continue

            cands = CANDIDATES + ['']
            if self.keep_original:
                cands = cands + [span]
            sents = [_join(left, c, right) for c in cands]
            ppl = self.lm.sentence_perplexity([self.vocab.encode(x)[0] for x in sents])
            order = sorted(range(len(cands)), key=lambda k: ppl[k])
            chosen = order[0]
            for k in order:
                c = cands[k]
                if c == span or c == '' or not self.recheck:
                    chosen = k
                    break
                cs = len(left)                                  # 바꿔 넣은 구간만 재검증
                p = self.char_probs(sents[k])[cs:cs + len(c)]
                if max(p) < self.thr:
                    chosen = k
                    break
                log.append(f'재검증 탈락: {c!r} (p={max(p):.2f})')
            log.append(f'{span!r} → {cands[chosen]!r}  (ppl {ppl[chosen]:.2f}, 1순위 {cands[order[0]]!r} {ppl[order[0]]:.2f})')
            out = sents[chosen]
        if verbose:
            print('\n'.join(log))
        return {'input': text, 'output': out, 'spans': found, 'log': log}


def dictionary_baseline(text):
    """E1: 금칙어 사전 정확히 일치하는 것만 ** 로 가림 (변형 욕설은 통과)"""
    text = normalize(text)
    for w in PROFANITY:
        text = text.replace(w, '**')
    return text
