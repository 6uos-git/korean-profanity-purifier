# coding: utf-8
"""SGD + RnnlmTrainer로 탐지기·언어 모델 학습 후 E1~E4 비교 (Ch05)

실행:  cd project && python train_demo.py
       (CPU, 예시 문장으로 만든 데이터 → 실제 성능이 아닌 동작 확인용)
"""
import sys
import os
import time
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np
from common.optimizer import SGD
from common.trainer import RnnlmTrainer
from common.util import clip_grads

from jamo import normalize
from make_pairs import make_pairs, split_templates, clean_corpus, TEMPLATES, CLEAN_TRAPS
from loader import JamoVocab, detector_dataset, batches, lm_corpus
from models import RnnDetector, SentenceLM
from purifier import Purifier, dictionary_baseline
from evaluate import span_prf, evaluate, HANDWRITTEN

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved')
os.makedirs(SAVE_DIR, exist_ok=True)


def train_detector(model, X, Y, epochs=15, batch_size=32, lr=1.0, max_grad=5.0, seed=0):
    opt = SGD(lr)
    rng = np.random.RandomState(seed)
    for ep in range(epochs):
        tot, cnt = 0.0, 0
        for xs, ts, mask in batches(X, Y, batch_size, rng):
            loss = model.forward(xs, ts, mask)
            model.backward()
            clip_grads(model.grads, max_grad)
            opt.update(model.params, model.grads)
            tot, cnt = tot + loss, cnt + 1
        if ep % 5 == 4 or ep == epochs - 1:
            print(f'  에폭 {ep + 1:2d} | 손실 {tot / cnt:.4f}')


def predict_char_labels(purifier, pairs):
    preds = []
    for dirty, _, _ in pairs:
        probs = purifier.char_probs(dirty)
        preds.append([int(p >= purifier.thr and ch != ' ') for p, ch in zip(probs, dirty)])
    return preds


def main():
    t0 = time.time()
    np.random.seed(0)
    (tr_tpl, tr_trap), (te_tpl, te_trap) = split_templates(0.8)
    train_pairs = make_pairs(4000, tr_tpl, tr_trap, seed=1)
    test_pairs = make_pairs(400, te_tpl, te_trap, seed=2)   # 학습 때 본 적 없는 템플릿
    print(f'학습 쌍 {len(train_pairs)} / 평가 쌍 {len(test_pairs)} (템플릿 {len(tr_tpl)}:{len(te_tpl)} 분할)')

    lm_sents = clean_corpus(TEMPLATES, CLEAN_TRAPS)
    vocab = JamoVocab([p[0] for p in train_pairs] + lm_sents)
    print(f'자모 어휘 {len(vocab)}개')

    X, Y = detector_dataset(train_pairs, vocab)

    # ── ① 탐지기: 단방향(E2, ch05 그대로) vs 양방향(E3, 교과 외) ──
    detectors = {}
    for name, bi in [('단방향(E2)', False), ('양방향(E3)', True)]:
        print(f'\n[탐지기 {name}] 학습')
        det = RnnDetector(len(vocab), 32, 64, bidirectional=bi)
        train_detector(det, X, Y)
        det.save(os.path.join(SAVE_DIR, f'detector_{"bi" if bi else "uni"}.pkl'))
        detectors[name] = det

    # ── ② 문맥 선택기: ch05 SimpleRnnlm을 깨끗한 문장으로만 학습 ──
    print('\n[언어 모델] 깨끗한 문장으로 학습 (RnnlmTrainer, SGD)')
    rng = np.random.RandomState(0)
    shuffled = []
    for _ in range(20):                               # 문장 순서를 섞어 말뭉치를 늘림
        shuffled += [lm_sents[i] for i in rng.permutation(len(lm_sents))]
    corpus = lm_corpus(shuffled, vocab)
    lm = SentenceLM(len(vocab), 32, 96, bos_id=vocab.tok_to_id['<s>'])
    ppl_before = float(np.mean(lm.sentence_perplexity([vocab.encode(normalize(s))[0] for s in lm_sents])))
    trainer = RnnlmTrainer(lm, SGD(0.5))
    trainer.fit(corpus[:-1], corpus[1:], max_epoch=30, batch_size=10, time_size=35,
                max_grad=0.25, eval_interval=None)
    ppl_after = float(np.mean(lm.sentence_perplexity([vocab.encode(normalize(s))[0] for s in lm_sents])))
    lm.save(os.path.join(SAVE_DIR, 'lm.pkl'))
    print(f'  문장 퍼플렉서티 {ppl_before:.2f} → {ppl_after:.2f}')

    # ── 탐지 구간 F1 ──
    print('\n[탐지 구간 F1] 평가 쌍(처음 보는 템플릿)')
    gold = [lab for _, _, lab in test_pairs]
    for name, det in detectors.items():
        p, r, f = span_prf(predict_char_labels(Purifier(det, lm, vocab), test_pairs), gold)
        print(f'  {name}: P {p:.3f}  R {r:.3f}  F1 {f:.3f}')

    # ── E1~E4 비교 ──
    det_uni, det_bi = detectors['단방향(E2)'], detectors['양방향(E3)']
    systems = {
        'E1 금칙어 사전': dictionary_baseline,
        'E2 단방향+고정치환': lambda s: Purifier(det_uni, lm, vocab, mode='fixed').purify(s)['output'],
        'E3 양방향+고정치환': lambda s: Purifier(det_bi, lm, vocab, mode='fixed').purify(s)['output'],
        'E4 양방향+RNNLM+재검증': lambda s: Purifier(det_bi, lm, vocab).purify(s)['output'],
    }
    gen_eval = [(d, c) for d, c, _ in test_pairs]
    for title, data in [('생성 평가 쌍', gen_eval), ('손으로 쓴 문장', HANDWRITTEN)]:
        print(f'\n[{title} {len(data)}개]  STA·SIM·FL·J  (EM=정답 일치, 과교정=깨끗한 문장을 바꾼 비율)')
        for name, fn in systems.items():
            m = evaluate(fn, data, lm=lm, vocab=vocab)
            print(f'  {name:22} STA {m["STA"]:.3f}  SIM {m["SIM"]:.3f}  FL {m["FL"]:.3f}  '
                  f'J {m["J"]:.3f}  EM {m["EM"]:.3f}  과교정 {m["과교정"]:.3f}')

    print('\n[예시] E4 최종')
    final = Purifier(det_bi, lm, vocab)
    for x, g in HANDWRITTEN[:12]:
        r = final.purify(x)
        ok = '성공' if r['output'] == normalize(g) else '다름'
        print(f'  {x:22} → {r["output"]:22} [{ok}] 탐지 {r["spans"]}')
    print(f'\n총 소요 {time.time() - t0:.1f}초, 모델 저장: {SAVE_DIR}')


if __name__ == '__main__':
    main()
