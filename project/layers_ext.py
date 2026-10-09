# coding: utf-8
"""교재 계층의 확장: 양방향 RNN, 마스크 시그모이드 손실 [교과 외]

교재 TimeGRU(ch06용)는 params/grads가 없고 오타(selfdWx)가 있어 학습이 안 되므로,
ch05 TimeRNN 두 개(정방향 + 입력을 뒤집은 역방향)를 이어 붙여 양방향을 만든다.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
from common.time_layers import TimeRNN
from common.functions import sigmoid


def _reverse_index(mask):
    """패딩을 제외하고 각 문장의 실제 길이 안에서만 뒤집는 인덱스. (N, T)"""
    N, T = mask.shape
    lengths = mask.sum(axis=1).astype(int)
    idx = np.tile(np.arange(T), (N, 1))
    for n, L in enumerate(lengths):
        idx[n, :L] = np.arange(L)[::-1]
    return idx


class TimeBiRNN:
    """ch05 TimeRNN 두 개로 만든 양방향 RNN. 출력은 (N, T, 2H) = [정방향 h ; 역방향 h]"""

    def __init__(self, Wx1, Wh1, b1, Wx2, Wh2, b2):
        self.forward_rnn = TimeRNN(Wx1, Wh1, b1, stateful=False)
        self.backward_rnn = TimeRNN(Wx2, Wh2, b2, stateful=False)
        self.params = self.forward_rnn.params + self.backward_rnn.params
        self.grads = self.forward_rnn.grads + self.backward_rnn.grads
        self.ridx = None

    def set_mask(self, mask):
        self.ridx = _reverse_index(mask)

    def _flip(self, a):
        return np.take_along_axis(a, self.ridx[:, :, None], axis=1)  # 자기 자신의 역연산

    def forward(self, xs):
        if self.ridx is None or self.ridx.shape != xs.shape[:2]:
            self.ridx = np.tile(np.arange(xs.shape[1]), (xs.shape[0], 1))[:, ::-1].copy()
        o1 = self.forward_rnn.forward(xs)
        o2 = self._flip(self.backward_rnn.forward(self._flip(xs)))
        self.H = o1.shape[2]
        return np.concatenate((o1, o2), axis=2)

    def backward(self, dhs):
        H = self.H
        dx1 = self.forward_rnn.backward(dhs[:, :, :H])
        dx2 = self._flip(self.backward_rnn.backward(self._flip(dhs[:, :, H:])))
        return dx1 + dx2


class MaskedTimeSigmoidWithLoss:
    """위치마다 '비속어인가?' 이진 분류 (Ch04 SigmoidWithLoss 원리). 패딩 위치는 손실에서 제외."""

    def __init__(self, pos_weight=1.0):
        self.params, self.grads = [], []
        self.pos_weight = pos_weight  # 비속어 자모가 소수라 양성에 가중치
        self.cache = None

    def forward(self, xs, ts, mask):
        y = sigmoid(xs)
        w = mask * np.where(ts == 1, self.pos_weight, 1.0)
        eps = 1e-7
        loss = -(ts * np.log(y + eps) + (1 - ts) * np.log(1 - y + eps))
        denom = mask.sum()
        self.cache = (y, ts, w, denom)
        return float((loss * w).sum() / denom)

    def backward(self, dout=1):
        y, ts, w, denom = self.cache
        return ((y - ts) * w / denom * dout).astype('f')
