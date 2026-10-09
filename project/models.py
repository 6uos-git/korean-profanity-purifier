# coding: utf-8
"""RNN 탐지기 + 문장 퍼플렉서티 (Ch04~05)

① 탐지기: ch05 SimpleRnnlm과 같은 구성(Embedding → RNN → Affine)에서 출력만
   '다음 글자 확률(Softmax, V개)' → '이 위치가 비속어인가(Sigmoid 1개)'로 바꾼다.
② 문맥 선택기: ch05 SimpleRnnlm을 그대로 import 해 깨끗한 한국어 자모로 학습하고,
   순화 후보 문장의 퍼플렉서티 = exp(평균 교차 엔트로피)로 자연스러움을 채점한다.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import pickle
import numpy as np
from common.time_layers import TimeEmbedding, TimeRNN, TimeAffine
from common.functions import sigmoid, softmax
from ch05.simple_rnnlm import SimpleRnnlm
from layers_ext import TimeBiRNN, MaskedTimeSigmoidWithLoss
from loader import pad_batch


class RnnDetector:
    def __init__(self, vocab_size, wordvec_size=32, hidden_size=64, bidirectional=True,
                 embed_W=None, pos_weight=2.0, seed=0):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rs = np.random.RandomState(seed)
        rn = rs.randn
        self.bidirectional = bidirectional

        embed_W = (rn(V, D) / 100).astype('f') if embed_W is None else embed_W.astype('f')
        D = embed_W.shape[1]

        def rnn_params():
            return [(rn(D, H) / np.sqrt(D)).astype('f'), (rn(H, H) / np.sqrt(H)).astype('f'), np.zeros(H, 'f')]

        if bidirectional:
            self.rnn = TimeBiRNN(*rnn_params(), *rnn_params())
            out_dim = 2 * H
        else:
            self.rnn = TimeRNN(*rnn_params(), stateful=False)
            out_dim = H
        affine_W = (rn(out_dim, 1) / np.sqrt(out_dim)).astype('f')
        affine_b = np.zeros(1, 'f')

        self.layers = [TimeEmbedding(embed_W), self.rnn, TimeAffine(affine_W, affine_b)]
        self.loss_layer = MaskedTimeSigmoidWithLoss(pos_weight)
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def _score(self, xs, mask):
        if self.bidirectional:
            self.rnn.set_mask(mask)
        h = xs
        for layer in self.layers:
            h = layer.forward(h)
        return h[:, :, 0]  # (N, T)

    def forward(self, xs, ts, mask):
        return self.loss_layer.forward(self._score(xs, mask), ts, mask)

    def backward(self, dout=1):
        d = self.loss_layer.backward(dout)[:, :, None]
        for layer in reversed(self.layers):
            d = layer.backward(d)
        return d

    def predict(self, id_seqs):
        """자모 ID 배열 리스트 → 자모별 비속어 확률 리스트"""
        xs, mask = pad_batch(id_seqs)
        p = sigmoid(self._score(xs, mask))
        return [p[i, :len(s)] for i, s in enumerate(id_seqs)]

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump([p.copy() for p in self.params], f)

    def load(self, path):
        with open(path, 'rb') as f:
            for p, q in zip(self.params, pickle.load(f)):
                p[...] = q


class SentenceClassifier:
    """E7용 문장 단위 욕설 분류기 (구간 라벨 없이 문장 0/1 라벨만으로 학습) [교과 외]
    Embedding → 양방향 RNN → 시간축 최대 풀링(패딩 제외) → Affine → Sigmoid
    """

    def __init__(self, vocab_size, wordvec_size=32, hidden_size=64, seed=0):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.RandomState(seed).randn

        def rnn_params():
            return [(rn(D, H) / np.sqrt(D)).astype('f'), (rn(H, H) / np.sqrt(H)).astype('f'), np.zeros(H, 'f')]

        self.embed = TimeEmbedding((rn(V, D) / 100).astype('f'))
        self.rnn = TimeBiRNN(*rnn_params(), *rnn_params())
        self.W = (rn(2 * H, 1) / np.sqrt(2 * H)).astype('f')
        self.b = np.zeros(1, 'f')
        self.params = self.embed.params + self.rnn.params + [self.W, self.b]
        self.grads = self.embed.grads + self.rnn.grads + [np.zeros_like(self.W), np.zeros_like(self.b)]
        self.cache = None

    def _score(self, xs, mask):
        self.rnn.set_mask(mask)
        hs = self.rnn.forward(self.embed.forward(xs))               # (N, T, 2H)
        masked = np.where(mask[:, :, None] > 0, hs, -np.inf)
        arg = masked.argmax(axis=1)                                  # (N, 2H) 최댓값 위치
        pooled = np.take_along_axis(hs, arg[:, None, :], axis=1)[:, 0, :]
        self.cache = (hs.shape, arg, pooled)
        return (pooled @ self.W + self.b)[:, 0]

    def forward(self, xs, ts, mask):
        s = self._score(xs, mask)
        y = sigmoid(s)
        self.y_t = (y, ts)
        return float(-np.mean(ts * np.log(y + 1e-7) + (1 - ts) * np.log(1 - y + 1e-7)))

    def backward(self, dout=1):
        (N, T, H2), arg, pooled = self.cache
        y, ts = self.y_t
        ds = ((y - ts) / len(y) * dout).astype('f')[:, None]         # (N, 1)
        self.grads[-2][...] = pooled.T @ ds
        self.grads[-1][...] = ds.sum(axis=0)
        dpooled = ds @ self.W.T                                      # (N, 2H)
        dhs = np.zeros((N, T, H2), 'f')
        np.put_along_axis(dhs, arg[:, None, :], dpooled[:, None, :], axis=1)
        self.embed.backward(self.rnn.backward(dhs))

    def predict(self, id_seqs):
        xs, mask = pad_batch(id_seqs)
        return sigmoid(self._score(xs, mask))


class SentenceLM(SimpleRnnlm):
    """ch05 SimpleRnnlm 그대로 + 문장 단위 퍼플렉서티 채점 메서드만 추가"""

    def __init__(self, vocab_size, wordvec_size=32, hidden_size=96, bos_id=2):
        super().__init__(vocab_size, wordvec_size, hidden_size)
        self.bos_id = bos_id

    def sentence_perplexity(self, id_seqs):
        """<s> 문장 <s> 에서 각 자모(+문장 끝)를 맞히는 평균 교차 엔트로피의 exp. 낮을수록 자연스럽다."""
        ins = [np.concatenate(([self.bos_id], s)).astype('i') for s in id_seqs]
        outs = [np.concatenate((s, [self.bos_id])).astype('i') for s in id_seqs]
        xs, mask = pad_batch(ins)
        ts, _ = pad_batch(outs)
        self.reset_state()
        h = xs
        for layer in self.layers:
            h = layer.forward(h)
        self.reset_state()
        N, T, V = h.shape
        p = softmax(h.reshape(N * T, V)).reshape(N, T, V)
        ce = -np.log(np.take_along_axis(p, ts[:, :, None], axis=2)[:, :, 0] + 1e-7)
        return np.exp((ce * mask).sum(axis=1) / mask.sum(axis=1))

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump([p.copy() for p in self.params], f)

    def load(self, path):
        with open(path, 'rb') as f:
            for p, q in zip(self.params, pickle.load(f)):
                p[...] = q
