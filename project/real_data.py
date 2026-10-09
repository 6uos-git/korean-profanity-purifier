# coding: utf-8
"""실제 데이터셋 5종 연결 + 실제 욕설 문장 평가

데이터 (project/data/ 에 각 저장소를 clone, 재배포하지 않음)
  UnSmile · HateSpeech · K-MHaS  → 깨끗한 댓글 → 언어 모델 학습 + 역방향 생성(비속어 주입)으로 탐지기 학습
  UnSmile '악플/욕설' 라벨        → 독립 심판 분류기 (순화 결과에 욕설이 남았는지 판정, 우리 탐지기와 별개)
  Curse-detection (문장 0/1)      → 문장 단위 탐지 평가 + 실제 욕설 문장 200개 평가 세트 (학습에 쓰지 않음)
  KOLD (공격 구간)                → 우리가 찾은 비속어 글자가 KOLD 공격 구간 안에 드는지(정밀도)

실행:  cd project && python real_data.py        (CPU 수 분)
"""
import sys
import os
import re
import random
import time
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np
from common.optimizer import SGD, Adam
from common.trainer import RnnlmTrainer
from common.util import clip_grads

from jamo import normalize
from make_pairs import REVERSE, mutate, make_pairs, TEMPLATES, CLEAN_TRAPS
from loader import JamoVocab, detector_dataset, pad_batch, lm_corpus, load_curse_detection, load_kold
from models import RnnDetector, SentenceLM, SentenceClassifier
from purifier import Purifier, dictionary_baseline
from evaluate import char_bleu, span_prf
from train_demo import train_detector

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
SAVE = os.path.join(HERE, 'saved')
os.makedirs(SAVE, exist_ok=True)
MAX_LEN = 40   # RNN 기울기 소실 → 짧은 댓글로 범위 제한 (발표 12쪽)

# 깨끗한 문장 거르기용 · E1 사전 확장용 비속어 패턴 (명시적 욕설과 대표 변형)
PROFANE_RE = re.compile(r'씨발|시발|씨바|ㅅㅂ|ㅆㅂ|존나|졸라|좆|ㅈㄴ|병신|븅신|ㅂㅅ|새끼|ㅅㄲ|개새|닥쳐|꺼져|지랄|ㅈㄹ|'
                        r'미친|썅|씹|18|느금|애미|애비|등신|빡치|빡쳐|처먹|쳐먹')


def _read_tsv(path):
    with open(path, encoding='utf-8') as f:
        header = f.readline().rstrip('\n').split('\t')
        for line in f:
            yield dict(zip(header, line.rstrip('\n').split('\t')))


def load_clean_comments():
    """세 데이터셋에서 '깨끗한' 라벨 댓글만 → 정규화 → 길이·비속어 필터"""
    out = []
    for r in _read_tsv(os.path.join(DATA, 'korean_unsmile_dataset', 'unsmile_train_v1.0.tsv')):
        if r.get('clean') == '1':
            out.append(r['문장'])
    for r in _read_tsv(os.path.join(DATA, 'korean-hate-speech', 'labeled', 'train.tsv')):
        if r.get('hate') == 'none' and r.get('bias') == 'none':
            out.append(r['comments'])
    for r in _read_tsv(os.path.join(DATA, 'K-MHaS', 'data', 'kmhas_train.txt')):
        if r.get('label', '').strip() == '8':          # 8 = 혐오 아님
            out.append(r['document'].strip('"'))
    sents = set()
    for s in out:
        s = normalize(s)
        if 5 <= len(s) <= MAX_LEN and not PROFANE_RE.search(s) and re.search('[가-힣]', s):
            sents.add(s)
    return sorted(sents)


INTENSIFIERS = ['정말', '너무', '진짜']


END_PAIRS = ['사람들', '사람들아', '난리네', '저리 가', '이상하다']      # 문장 끝에 붙이는 순화어 ↔ 비속어
ADDRESS = ['새끼들아', '병신아', '병신들', '새끼야', '등신아']                 # 끝에 붙는 호칭 욕설 → 순화 시 삭제


def inject_real(sent, rng, p_mutate=0.5):
    """실제 깨끗한 댓글에 비속어 주입 → (오염 문장, 순화 정답, 글자 라벨)
    위치·형태가 한쪽으로 쏠리면 탐지기가 '자리'를 외우므로(1차 실험에서 확인) 1~2개 연산을 무작위 위치에 적용:
      replace  기존 어절이 순화어(정말·너무·사람들 ...)면 그 자리를 비속어로
      insert   강조어 쌍(정말↔존나)을 아무 위치에 끼움
      prefix   어절 앞에 접두사 '개'를 붙임 (정답은 '정말 '+어절)
      start    문장 앞 감탄사 (아↔씨발)
      end      문장 끝 순화어↔비속어 (난리네↔지랄하네 등) 또는 호칭 욕설(새끼들아 → 정답에서는 삭제)
    """
    words = sent.split(' ')
    items = [(w, [(w, 0)]) for w in words]          # (정답 어절, [(오염 조각, 비속어 여부)])

    def bad(clean_word):
        return mutate(rng.choice(REVERSE[clean_word]), rng, p_mutate)

    for _ in range(rng.choice([1, 1, 2])):
        op = rng.choice(['replace', 'insert', 'insert', 'prefix', 'start', 'end', 'end'])
        if op == 'replace':
            slots = [k for k, (c, d) in enumerate(items) if c in REVERSE and len(d) == 1 and not d[0][1]]
            if slots:
                k = rng.choice(slots)
                items[k] = (items[k][0], [(bad(items[k][0]), 1)])
                continue
            op = 'insert'
        if op == 'insert':
            c = rng.choice(INTENSIFIERS)
            items.insert(rng.randint(0, len(items)), (c, [(bad(c), 1)]))
        elif op == 'prefix':
            slots = [k for k, (c, d) in enumerate(items) if len(d) == 1 and not d[0][1] and re.match('[가-힣]{2}', c)]
            if slots:
                k = rng.choice(slots)
                items[k] = ('정말 ' + items[k][0], [(mutate('개', rng, 0.0), 1), (items[k][0], 0)])
        elif op == 'start' and not items[0][1][0][1]:
            items.insert(0, ('아', [(bad('아'), 1)]))
        elif op == 'end':
            if rng.random() < 0.5:
                c = rng.choice(END_PAIRS)
                items.append((c, [(bad(c), 1)]))
            else:
                items.append(('', [(mutate(rng.choice(ADDRESS), rng, p_mutate), 1)]))

    clean = re.sub(r'\s+', ' ', ' '.join(c for c, _ in items)).strip()
    dirty, labels = '', []
    for _, pieces in items:
        if dirty:
            dirty += ' '
            labels.append(0)
        for text, b in pieces:
            dirty += text
            labels += [b if ch != ' ' else 0 for ch in text]
    if not any(labels):
        return None
    return dirty, clean, labels


def train_lm(sents, vocab, epochs, hidden=128):
    lm = SentenceLM(len(vocab), 48, hidden, bos_id=vocab.tok_to_id['<s>'])
    corpus = lm_corpus(sents, vocab)
    trainer = RnnlmTrainer(lm, SGD(0.5))
    trainer.fit(corpus[:-1], corpus[1:], max_epoch=epochs, batch_size=20, time_size=35,
                max_grad=0.25, eval_interval=500)
    return lm


def train_judge(vocab, epochs=10):
    """독립 심판: UnSmile 학습 세트의 '악플/욕설'=1 vs 'clean'=1 문장 분류기"""
    def rows(path):
        X, S = [], []
        for r in _read_tsv(path):
            s = normalize(r['문장'])[:80]
            if not s:
                continue
            if r.get('악플/욕설') == '1':
                X.append(vocab.encode(s)[0]); S.append(1)
            elif r.get('clean') == '1':
                X.append(vocab.encode(s)[0]); S.append(0)
        return X, np.array(S, 'f')
    X, S = rows(os.path.join(DATA, 'korean_unsmile_dataset', 'unsmile_train_v1.0.tsv'))
    Xv, Sv = rows(os.path.join(DATA, 'korean_unsmile_dataset', 'unsmile_valid_v1.0.tsv'))
    judge = SentenceClassifier(len(vocab), 32, 64, seed=7)
    opt, rng = Adam(0.003), np.random.RandomState(0)
    for ep in range(epochs):
        idx = rng.permutation(len(X))
        for s in range(0, len(idx), 32):
            b = idx[s:s + 32]
            xs, mask = pad_batch([X[i] for i in b])
            judge.forward(xs, S[b], mask)
            judge.backward()
            clip_grads(judge.grads, 5.0)
            opt.update(judge.params, judge.grads)
    pred = np.concatenate([judge.predict(Xv[i:i + 256]) for i in range(0, len(Xv), 256)]) >= 0.5
    _, _, f1 = span_prf([pred.astype(int)], [Sv.astype(int)])
    print(f'  심판 분류기 (UnSmile valid {len(Xv)}문장): 정확도 {np.mean(pred == Sv):.3f}, 욕설 F1 {f1:.3f}')
    return judge


def main():
    t0 = time.time()
    rng = random.Random(0)
    np.random.seed(0)

    # ── 1. 깨끗한 댓글 ──
    clean = load_clean_comments()
    rng.shuffle(clean)
    print(f'[데이터] 깨끗한 댓글 {len(clean)}개 (길이 5~{MAX_LEN}, 비속어 패턴 제외)')
    lm_sents = clean[:20000]
    inj_src = clean[:12000]

    curse = [(t, l) for t, l in load_curse_detection(os.path.join(DATA, 'Curse-detection-data', 'dataset.txt')) if t]
    kold = [(normalize(t), lab) for t, lab in load_kold(os.path.join(DATA, 'KOLD', 'data', 'kold_v1.json'))]

    # ── 2. 역방향 생성: 실제 댓글 + 템플릿 ──
    real_pairs = [p for p in (inject_real(s, rng) for s in inj_src) if p]
    tpl_pairs = make_pairs(4000, seed=1)
    neg = [(s, s, [0] * len(s)) for s in clean[12000:16000]]          # 깨끗한 실제 댓글 = 음성 예시
    train_pairs = real_pairs + tpl_pairs + neg
    print(f'[학습 쌍] 실제 댓글 주입 {len(real_pairs)} + 템플릿 {len(tpl_pairs)} + 깨끗한 댓글 {len(neg)}')
    for d, c, _ in real_pairs[:3]:
        print(f'  {c}  →  {d}')

    vocab = JamoVocab([p[0] for p in train_pairs] + lm_sents)
    print(f'  자모 어휘 {len(vocab)}개')
    X, Y = detector_dataset(train_pairs, vocab)

    # ── 3. 탐지기 (단방향 E2 / 양방향 E3) ──
    dets = {}
    for name, bi in [('단방향', False), ('양방향', True)]:
        print(f'\n[탐지기 {name}] 학습 ({time.time() - t0:.0f}s)')
        det = RnnDetector(len(vocab), 32, 64, bidirectional=bi)
        train_detector(det, X, Y, epochs=12, lr=1.0)
        det.save(os.path.join(SAVE, f'real_detector_{"bi" if bi else "uni"}.pkl'))
        dets[name] = det

    # ── 4. 언어 모델 (깨끗한 실제 댓글만) ──
    print(f'\n[언어 모델] 깨끗한 댓글 {len(lm_sents)}개로 학습 ({time.time() - t0:.0f}s)')
    held = clean[20000:20500]
    lm0 = SentenceLM(len(vocab), 48, 128, bos_id=vocab.tok_to_id['<s>'])
    before = float(np.median(lm0.sentence_perplexity([vocab.encode(s)[0] for s in held])))
    lm = train_lm(lm_sents, vocab, epochs=4)
    held_ppl = lm.sentence_perplexity([vocab.encode(s)[0] for s in held])
    ref_ppl = float(np.median(held_ppl))
    print(f'  처음 보는 깨끗한 댓글 500개 퍼플렉서티(중앙값) {before:.2f} → {ref_ppl:.2f}')
    lm.save(os.path.join(SAVE, 'real_lm.pkl'))

    # ── 5. 독립 심판 ──
    print(f'\n[심판] 학습 ({time.time() - t0:.0f}s)')
    judge = train_judge(vocab)

    # ── 6-a. Curse-detection 문장 단위 탐지 (학습에 쓰지 않은 데이터) ──
    print('\n[실제 평가 ①] Curse-detection 문장 단위 탐지 (구간이 하나라도 잡히면 욕설)')
    cur = [(t, l) for t, l in curse if len(t) <= 80]
    gold = np.array([l for _, l in cur])
    explicit = np.array([bool(PROFANE_RE.search(t)) for t, _ in cur])
    systems = {'E1 금칙어 사전': lambda t: dictionary_baseline(t) != t}
    for name, det in dets.items():
        p = Purifier(det, None, vocab)
        systems[f'{name} RNN'] = lambda t, p=p: bool(p.detect(t)[1])
    for name, fn in systems.items():
        pred = np.array([fn(t) for t, _ in cur])
        prec = (pred & (gold == 1)).sum() / max(pred.sum(), 1)
        rec = (pred & (gold == 1)).sum() / (gold == 1).sum()
        rec_exp = (pred & (gold == 1) & explicit).sum() / ((gold == 1) & explicit).sum()
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        print(f'  {name:14} 정밀도 {prec:.3f}  재현율 {rec:.3f}  F1 {f1:.3f}  | 명시적 욕설 포함 문장 재현율 {rec_exp:.3f}')
    print(f'  (Curse-detection 욕설 {int((gold == 1).sum())}문장 중 명시적 욕설 패턴 포함 {int(((gold == 1) & explicit).sum())}문장 —'
          f' 나머지는 비꼼·은어라 이 프로젝트 범위 밖)')

    # ── 6-b. KOLD: 우리가 찾은 글자가 공격 구간 안에 있는가 ──
    print('\n[실제 평가 ②] KOLD 공격 구간 대비 (공격 문장 1000개 · 비공격 문장 1000개)')
    off = [(t, l) for t, l in kold if any(l) and len(t) <= 80][:1000]
    non = [(t, l) for t, l in kold if not any(l) and len(t) <= 80][:1000]
    p_bi = Purifier(dets['양방향'], None, vocab)
    inside = total = 0
    for t, lab in off:
        probs = p_bi.char_probs(t)
        for i, pr in enumerate(probs):
            if pr >= 0.5 and t[i] != ' ':
                total += 1
                inside += lab[i] if i < len(lab) else 0
    false_flag = np.mean([bool(p_bi.detect(t)[1]) for t, _ in non])
    print(f'  예측 비속어 글자 중 KOLD 공격 구간 안 비율 {inside / max(total, 1):.3f} ({total}글자) | 비공격 문장에서 탐지된 비율 {false_flag:.3f}')

    # ── 6-c. 실제 욕설 문장 200개 순화 (E1~E4) ──
    print('\n[실제 평가 ③] Curse-detection 욕설 문장 200개 순화 — 정답 순화문이 없으므로 원문 대비 SIM (ParaDetox 방식)')
    rs = random.Random(42)
    pool = [t for t, l in curse if l == 1 and PROFANE_RE.search(t) and 8 <= len(t) <= MAX_LEN]
    test200 = rs.sample(pool, 200)
    jp = lambda texts: np.concatenate([judge.predict([vocab.encode(t)[0] if t else np.array([2], 'i') for t in texts[i:i + 100]])
                                       for i in range(0, len(texts), 100)])
    systems = {
        'E1 금칙어 사전': dictionary_baseline,
        'E2 단방향+고정치환': lambda s: Purifier(dets['단방향'], lm, vocab, mode='fixed').purify(s)['output'],
        'E3 양방향+고정치환': lambda s: Purifier(dets['양방향'], lm, vocab, mode='fixed').purify(s)['output'],
        'E4 양방향+RNNLM+재검증': lambda s: Purifier(dets['양방향'], lm, vocab).purify(s)['output'],
    }
    src_tox = jp(test200)
    print(f'  입력 200문장 심판 욕설 판정률 {np.mean(src_tox >= 0.5):.3f}')
    results = {}
    for name, fn in systems.items():
        outs = [fn(t) for t in test200]
        tox = jp(outs)
        sta = (tox < 0.5).astype(float)
        sim = np.array([char_bleu(o, s) for o, s in zip(outs, test200)])
        ppl = lm.sentence_perplexity([vocab.encode(o)[0] if o else np.array([2], 'i') for o in outs])
        fl = np.minimum(1.0, ref_ppl / ppl)
        j = sta * sim * fl
        lex = np.mean([bool(PROFANE_RE.search(o)) for o in outs])
        results[name] = outs
        print(f'  {name:22} STA {sta.mean():.3f}  SIM {sim.mean():.3f}  FL {fl.mean():.3f}  J {j.mean():.3f}  | 비속어 패턴 잔존 {lex:.3f}')

    print('\n[예시] 실제 욕설 문장 → E4')
    for i in range(12):
        print(f'  {test200[i]}\n    → {results["E4 양방향+RNNLM+재검증"][i]}')
    print(f'\n총 소요 {time.time() - t0:.0f}초')


if __name__ == '__main__':
    main()
