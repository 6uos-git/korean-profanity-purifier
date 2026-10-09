# coding: utf-8
"""자모 어휘, 패딩, 언어 모델 말뭉치 (Ch05 형식 + 교과 외)

- 탐지기 입력: 문장마다 길이가 달라 오른쪽을 <pad>(0)로 채우고 mask로 실제 위치를 표시 [교과 외]
- 언어 모델 입력: ch05 PTB와 같은 형식의 1차원 ID 배열 (문장 사이에 <s>) → RnnlmTrainer에 그대로 넣음
- 실제 데이터셋 읽기 함수: 라이선스상 원본은 재배포하지 않으므로 직접 내려받은 파일 경로를 넘겨 사용
"""
import json
import os
import numpy as np
from jamo import normalize, decompose, CHO, JUNG, JONG_TOKENS

PAD, UNK, BOS = '<pad>', '<unk>', '<s>'


class JamoVocab:
    def __init__(self, sentences=()):
        self.id_to_tok = [PAD, UNK, BOS]
        self.tok_to_id = {t: i for i, t in enumerate(self.id_to_tok)}
        # 자모 전체 + 숫자·기호를 미리 등록 → 처음 보는 변형(쫀나, 씨8)도 <unk>가 되지 않음
        base = CHO + JUNG + JONG_TOKENS[1:] + list('0123456789.,!?~*') + ['▁']
        for t in base:
            self.tok_to_id.setdefault(t, len(self.id_to_tok))
            if len(self.id_to_tok) < len(self.tok_to_id):
                self.id_to_tok.append(t)
        for s in sentences:
            self.add_sentence(s)

    def add_sentence(self, s):
        for t in decompose(normalize(s))[0]:
            if t not in self.tok_to_id:
                self.tok_to_id[t] = len(self.id_to_tok)
                self.id_to_tok.append(t)

    def __len__(self):
        return len(self.id_to_tok)

    def encode(self, s):
        """문장 → (자모 ID 배열, 각 자모의 원문 글자 인덱스). s는 정규화된 문장이어야 한다."""
        toks, char_idx = decompose(s)
        unk = self.tok_to_id[UNK]
        return np.array([self.tok_to_id.get(t, unk) for t in toks], dtype='i'), char_idx


class SyllableVocab:
    """E5 비교용: 음절(글자) 하나 = 토큰. 학습 때 못 본 변형 글자(쫀, 씨8의 8 등)는 <unk>가 된다."""

    def __init__(self, sentences=()):
        self.id_to_tok = [PAD, UNK, BOS]
        self.tok_to_id = {t: i for i, t in enumerate(self.id_to_tok)}
        for s in sentences:
            for ch in normalize(s):
                if ch not in self.tok_to_id:
                    self.tok_to_id[ch] = len(self.id_to_tok)
                    self.id_to_tok.append(ch)

    def __len__(self):
        return len(self.id_to_tok)

    def encode(self, s):
        unk = self.tok_to_id[UNK]
        return np.array([self.tok_to_id.get(ch, unk) for ch in s], dtype='i'), list(range(len(s)))


def pad_batch(id_seqs, pad_value=0):
    """가변 길이 ID 배열 리스트 → (N, T) 배열, (N, T) 마스크"""
    T = max(len(x) for x in id_seqs)
    xs = np.full((len(id_seqs), T), pad_value, dtype='i')
    mask = np.zeros((len(id_seqs), T), dtype='f')
    for i, x in enumerate(id_seqs):
        xs[i, :len(x)] = x
        mask[i, :len(x)] = 1
    return xs, mask


def detector_dataset(pairs, vocab):
    """(오염 문장, 원문, 글자 라벨) → 자모 ID 시퀀스와 자모 단위 라벨 리스트"""
    X, Y = [], []
    for dirty, _, char_labels in pairs:
        ids, char_idx = vocab.encode(dirty)
        X.append(ids)
        Y.append(np.array([char_labels[c] for c in char_idx], dtype='i'))
    return X, Y


def batches(X, Y, batch_size, rng=None):
    """셔플 후 패딩된 미니배치 생성 → (xs, ts, mask)"""
    idx = np.arange(len(X))
    if rng is not None:
        rng.shuffle(idx)
    for s in range(0, len(idx), batch_size):
        b = idx[s:s + batch_size]
        xs, mask = pad_batch([X[i] for i in b])
        ts, _ = pad_batch([Y[i] for i in b])
        yield xs, ts, mask


def lm_corpus(sentences, vocab):
    """깨끗한 문장들 → ch05 형식 1차원 말뭉치 (<s> 문장 <s> 문장 ...)"""
    bos = vocab.tok_to_id[BOS]
    corpus = []
    for s in sentences:
        corpus.append(bos)
        corpus += list(vocab.encode(normalize(s))[0])
    corpus.append(bos)
    return np.array(corpus, dtype='i')


# ── 실제 데이터셋 읽기 (파일은 각 저장소에서 직접 내려받기) ─────────────────────────
def load_curse_detection(path):
    """2runo/Curse-detection-data: 한 줄에 '문장|0 또는 1' → [(문장, 라벨)]"""
    out = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if '|' in line:
                text, label = line.rstrip('\n').rsplit('|', 1)
                if label.strip() in ('0', '1'):
                    out.append((normalize(text), int(label)))
    return out


def load_kold(path):
    """KOLD (boychaboy/KOLD) json → [(문장, 글자 단위 0/1 라벨)]. 구간 키 이름은 버전마다 달라 여러 후보를 확인한다."""
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    rows = data if isinstance(data, list) else list(data.values())
    out = []
    for r in rows:
        text = r.get('comment') or r.get('text') or ''
        labels = [0] * len(text)
        for key in ('off_span', 'OFF_span', 'offensive_span'):
            spans = r.get(key)
            if not spans:
                continue
            for sp in spans if isinstance(spans, list) else [spans]:
                if isinstance(sp, str) and sp in text:  # 구간이 문자열로 주어진 경우
                    st = text.index(sp)
                    sp = (st, st + len(sp))
                if isinstance(sp, (list, tuple)) and len(sp) == 2:
                    for i in range(max(0, sp[0]), min(len(text), sp[1])):
                        labels[i] = 1
        out.append((text, labels))
    return out


def load_tsv_texts(path, text_col=0, label_col=None, keep=None):
    """UnSmile / HateSpeech / K-MHaS 등 tsv → 문장 리스트. keep으로 '깨끗한' 라벨만 골라 언어 모델 말뭉치로 쓴다."""
    out = []
    if not os.path.exists(path):
        return out
    with open(path, encoding='utf-8') as f:
        next(f)  # 헤더
        for line in f:
            cols = line.rstrip('\n').split('\t')
            if label_col is not None and keep is not None and cols[label_col] not in keep:
                continue
            out.append(normalize(cols[text_col]))
    return out
