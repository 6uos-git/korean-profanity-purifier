# coding: utf-8
"""비교 실험 E5~E7 (E1~E4는 train_demo.py, E8은 kcbert_fill.py)

E5  입력 단위: 음절 vs 자모 [교과 외]          → 변형 욕설 대응
E6  임베딩: 랜덤 vs word2vec(ch04 CBOW)        → 사전학습 효과
E7  구간 찾기: 태거 vs Leave-one-out [교과 외]  → 구간 라벨 없는 데이터 활용

실행:  cd project && python experiments.py
"""
import sys
import os
import time
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np
from common.optimizer import SGD, Adam
from common.trainer import Trainer
from common.util import clip_grads, create_contexts_target
from ch04.cbow import CBOW

from make_pairs import make_pairs, split_templates, clean_corpus
from loader import JamoVocab, SyllableVocab, detector_dataset, batches, pad_batch
from models import RnnDetector, SentenceClassifier
from purifier import Purifier
from evaluate import span_prf, HANDWRITTEN
from train_demo import train_detector, predict_char_labels
from jamo import normalize


def char_labels_from_gold(dirty, gold):
    """손으로 쓴 (입력, 정답) 쌍 → 입력에만 있는 어절을 비속어 구간으로 본 글자 라벨"""
    d, g = normalize(dirty), normalize(gold)
    bad = set(d.split()) - set(g.split())
    labels, pos = [0] * len(d), 0
    for w in d.split(' '):
        if w in bad:
            for i in range(pos, pos + len(w)):
                labels[i] = 1
        pos += len(w) + 1
    return labels


def word_f1(pred_labels, gold_labels, texts):
    """어절 단위 F1: 어절 안에 1이 하나라도 있으면 그 어절을 비속어로 본다 (E7은 어절 단위로만 찾을 수 있어 공정 비교용)"""
    def words(labels, text):
        out, pos = [], 0
        for w in text.split(' '):
            out.append(int(any(labels[pos:pos + len(w)])))
            pos += len(w) + 1
        return out
    P = [words(p, t) for p, t in zip(pred_labels, texts)]
    G = [words(g, t) for g, t in zip(gold_labels, texts)]
    return span_prf(P, G)[2]


def setup():
    (tr_tpl, tr_trap), (te_tpl, te_trap) = split_templates(0.8)
    train = make_pairs(4000, tr_tpl, tr_trap, seed=1)
    test = make_pairs(400, te_tpl, te_trap, seed=2)
    hand = [(normalize(x), normalize(g), char_labels_from_gold(x, g)) for x, g in HANDWRITTEN]
    return train, test, hand


def f1_of(det, vocab, pairs):
    p = Purifier(det, None, vocab)
    return span_prf(predict_char_labels(p, pairs), [l for _, _, l in pairs])[2]


def e5(train, test, hand):
    print('\n[E5] 입력 단위: 음절 vs 자모 (양방향 탐지기, 같은 조건)')
    sents = [p[0] for p in train] + clean_corpus()
    for name, vocab in [('음절', SyllableVocab(sents)), ('자모', JamoVocab(sents))]:
        X, Y = detector_dataset(train, vocab)
        det = RnnDetector(len(vocab), 32, 64, bidirectional=True)
        train_detector(det, X, Y, epochs=15)
        # 처음 보는 변형만 따로: 평가 쌍 중 학습 문장에 없던 비속어 글자열을 포함한 것
        seen = {p[0][i:j] for p in train for i, j in _spans(p[2])}
        novel = [p for p in test if any(p[0][i:j] not in seen for i, j in _spans(p[2]))]
        print(f'  {name}: 어휘 {len(vocab):4d} | 평가 F1 {f1_of(det, vocab, test):.3f} | '
              f'처음 보는 변형 {len(novel)}개 F1 {f1_of(det, vocab, novel):.3f} | 손글씨 F1 {f1_of(det, vocab, hand):.3f}')


def _spans(labels):
    out, s = [], None
    for i, l in enumerate(labels + [0]):
        if l and s is None:
            s = i
        if not l and s is not None:
            out.append((s, i))
            s = None
    return out


def e6(train, test, hand):
    print('\n[E6] 임베딩: 랜덤 vs word2vec (ch04 CBOW로 자모 임베딩 사전학습)')
    vocab = JamoVocab([p[0] for p in train] + clean_corpus())
    corpus = np.concatenate([np.concatenate((vocab.encode(p[0])[0], [vocab.tok_to_id['<s>']])) for p in train]).astype('i')
    # ch04 UnigramSampler는 어휘 크기를 '말뭉치에 나온 ID 개수'로 잡으므로 모든 ID를 한 번씩 넣어 둔다
    corpus = np.concatenate((np.arange(len(vocab), dtype='i'), corpus))
    contexts, target = create_contexts_target(corpus, window_size=2)
    cbow = CBOW(len(vocab), 32, 2, corpus)
    Trainer(cbow, Adam()).fit(contexts, target, max_epoch=5, batch_size=100, eval_interval=None)
    w2v = cbow.word_vecs.copy()
    # 사전학습 임베딩이 Embedding 초기값으로 너무 크면 RNN이 포화 → 표준편차를 랜덤 초기화(0.01)와 비슷하게 맞춤
    w2v = w2v / (w2v.std() + 1e-8) * 0.1
    X, Y = detector_dataset(train, vocab)
    for epochs in (2, 5, 15):
        row = []
        for name, W in [('랜덤', None), ('word2vec', w2v)]:
            det = RnnDetector(len(vocab), 32, 64, bidirectional=True, embed_W=W)
            _quiet(lambda: train_detector(det, X, Y, epochs=epochs))
            row.append(f'{name} F1 {f1_of(det, vocab, test):.3f} (손글씨 {f1_of(det, vocab, hand):.3f})')
        print(f'  {epochs:2d}에폭 | ' + ' | '.join(row))


def _quiet(fn):
    stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        return fn()
    finally:
        sys.stdout.close()
        sys.stdout = stdout


def leave_one_out_labels(clf, vocab, text, thr=0.5, min_drop=0.3):
    """문장 분류기만으로 구간 찾기: 어절을 하나씩 빼 보고 욕설 확률이 크게 떨어지는 어절을 비속어로 본다"""
    words = text.split(' ')
    variants = [text] + [' '.join(words[:k] + words[k + 1:]) for k in range(len(words))]
    probs = clf.predict([vocab.encode(v)[0] if v else np.array([vocab.tok_to_id['<s>']], 'i') for v in variants])
    labels, pos = [0] * len(text), 0
    for k, w in enumerate(words):
        if probs[0] >= thr and probs[0] - probs[k + 1] >= min_drop:
            for i in range(pos, pos + len(w)):
                labels[i] = 1
        pos += len(w) + 1
    return labels


def e7(train, test, hand):
    print('\n[E7] 구간 찾기: 태거(구간 라벨로 학습) vs Leave-one-out(문장 라벨만으로 학습)')
    vocab = JamoVocab([p[0] for p in train] + clean_corpus())
    X, Y = detector_dataset(train, vocab)
    tagger = RnnDetector(len(vocab), 32, 64, bidirectional=True)
    _quiet(lambda: train_detector(tagger, X, Y, epochs=15))

    clf = SentenceClassifier(len(vocab), 32, 64)
    S = np.array([int(any(p[2])) for p in train], 'f')
    opt, rng = Adam(0.005), np.random.RandomState(0)
    for ep in range(8):
        idx = rng.permutation(len(X))
        for s in range(0, len(idx), 32):
            b = idx[s:s + 32]
            xs, mask = pad_batch([X[i] for i in b])
            clf.forward(xs, S[b], mask)
            clf.backward()
            clip_grads(clf.grads, 5.0)
            opt.update(clf.params, clf.grads)
    acc = np.mean((clf.predict([vocab.encode(p[0])[0] for p in test]) >= 0.5) == np.array([any(p[2]) for p in test]))
    print(f'  문장 분류 정확도 {acc:.3f}')
    for title, data in [('평가', test), ('손글씨', hand)]:
        texts, gold = [p[0] for p in data], [p[2] for p in data]
        tag = predict_char_labels(Purifier(tagger, None, vocab), data)
        loo = [leave_one_out_labels(clf, vocab, t) for t in texts]
        print(f'  {title}: 어절 F1  태거 {word_f1(tag, gold, texts):.3f} | Leave-one-out {word_f1(loo, gold, texts):.3f}')


if __name__ == '__main__':
    t0 = time.time()
    np.random.seed(0)
    train, test, hand = setup()
    e5(train, test, hand)
    e6(train, test, hand)
    e7(train, test, hand)
    print(f'\n총 소요 {time.time() - t0:.1f}초')
