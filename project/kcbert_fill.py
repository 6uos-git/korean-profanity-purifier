# coding: utf-8
"""E8: 순화 단계 비교 — RNNLM 후보 선택(E4, 교재 모델) vs KcBERT Mask-and-Fill(팀원 설계, 사전학습 모델) [교과 외]

탐지는 둘 다 같은 양방향 RNN 탐지기. 순화 단계만 바꾼다.
KcBERT는 [MASK]를 자유롭게 채우므로 욕설이 다시 나올 수 있어 3중 장치가 필수:
  ① 사전 차단: 비속어 패턴·서브워드(##)·기호 토큰은 후보에서 제외
  ② 재검증: 채운 표현을 탐지기로 다시 확인, 비속어면 다음 후보
  ③ 반복 제한: 상위 TOP_K개까지만 시도, 모두 실패하면 삭제

준비: pip install torch transformers  (beomi/kcbert-base 약 440MB를 처음 실행 때 내려받음)
실행: cd project && python train_demo.py && python kcbert_fill.py
"""
import sys
import os
import re
import time
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np

from jamo import normalize
from make_pairs import make_pairs, split_templates, clean_corpus, TEMPLATES, CLEAN_TRAPS
from loader import JamoVocab
from models import RnnDetector, SentenceLM
from purifier import Purifier, _join
from evaluate import evaluate, HANDWRITTEN
from real_data import PROFANE_RE

SAVE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved')
TOP_K = 20


class KcBertFiller:
    def __init__(self, detector, vocab, name='beomi/kcbert-base', thr=0.5):
        import torch
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name)
        self.mlm = AutoModelForMaskedLM.from_pretrained(name).eval()
        self.p = Purifier(detector, None, vocab, thr=thr)
        self.thr = thr
        self.stats = {'차단': 0, '재검증 탈락': 0, '전부 실패': 0}

    def _fill(self, left, right):
        text = f'{left}{self.tok.mask_token}{right}'
        enc = self.tok(text, return_tensors='pt')
        pos = (enc['input_ids'][0] == self.tok.mask_token_id).nonzero()[0, 0]
        with self.torch.no_grad():
            logits = self.mlm(**enc).logits[0, pos]
        return [self.tok.decode([i]).strip() for i in logits.topk(TOP_K * 3).indices.tolist()]

    def purify(self, text):
        text, spans, _ = self.p.detect(text)
        out = text
        for s, e in reversed(spans):
            left, right = out[:s], out[e:]
            chosen, tried = '', 0
            for c in self._fill(left, right):
                if tried >= TOP_K:
                    break
                if not c or c.startswith('##') or not re.search('[가-힣]', c) or PROFANE_RE.search(c):
                    self.stats['차단'] += 1                                   # ① 사전 차단
                    continue
                tried += 1
                cand = _join(left, c, right)
                cs = len(left)
                if max(self.p.char_probs(cand)[cs:cs + len(c)] or [0]) >= self.thr:
                    self.stats['재검증 탈락'] += 1                             # ② 재검증
                    continue
                chosen = c
                break
            else:
                self.stats['전부 실패'] += 1                                   # ③ 반복 제한 → 삭제
            out = _join(left, chosen, right)
        return out


def load_demo_models():
    """train_demo.py가 저장한 탐지기·언어 모델을 같은 어휘로 다시 불러온다"""
    (tr_tpl, tr_trap), (te_tpl, te_trap) = split_templates(0.8)
    train_pairs = make_pairs(4000, tr_tpl, tr_trap, seed=1)
    test_pairs = make_pairs(400, te_tpl, te_trap, seed=2)
    vocab = JamoVocab([p[0] for p in train_pairs] + clean_corpus(TEMPLATES, CLEAN_TRAPS))
    det = RnnDetector(len(vocab), 32, 64, bidirectional=True)
    det.load(os.path.join(SAVE, 'detector_bi.pkl'))
    lm = SentenceLM(len(vocab), 32, 96, bos_id=vocab.tok_to_id['<s>'])
    lm.load(os.path.join(SAVE, 'lm.pkl'))
    return vocab, det, lm, test_pairs


def main():
    t0 = time.time()
    vocab, det, lm, test_pairs = load_demo_models()
    print('[E8] KcBERT 불러오는 중 (처음엔 내려받기)...')
    filler = KcBertFiller(det, vocab)
    rnn = Purifier(det, lm, vocab)
    systems = {'E4 RNNLM 후보 선택': lambda s: rnn.purify(s)['output'], 'E8 KcBERT Mask-and-Fill': filler.purify}
    gen = [(d, c) for d, c, _ in test_pairs][:200]
    for title, data in [('생성 평가 쌍', gen), ('손으로 쓴 문장', HANDWRITTEN)]:
        print(f'\n[{title} {len(data)}개]  (FL은 교재 RNNLM 기준이라 E4에 유리할 수 있음)')
        for name, fn in systems.items():
            ts = time.time()
            m = evaluate(fn, data, lm=lm, vocab=vocab)
            print(f'  {name:24} STA {m["STA"]:.3f}  SIM {m["SIM"]:.3f}  FL {m["FL"]:.3f}  J {m["J"]:.3f}  '
                  f'EM {m["EM"]:.3f}  과교정 {m["과교정"]:.3f}  | {(time.time() - ts) / len(data) * 1000:.0f}ms/문장')
    print(f'\n  KcBERT 안전장치 작동 횟수: {filler.stats}')
    print('\n[예시]')
    for x, g in HANDWRITTEN[:12]:
        print(f'  {x:22} | E4 {rnn.purify(x)["output"]:20} | E8 {filler.purify(x)}')
    print(f'\n총 소요 {time.time() - t0:.0f}초')


if __name__ == '__main__':
    main()
