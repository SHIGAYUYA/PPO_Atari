import os

from PIL import Image

import numpy as np
import pandas as pd
from tqdm import tqdm

# 入力サイズ
INPUT_SHAPE = (84, 84)

# フレーム間隔
WINDOWS_SIZE = 4

def preprocess(status, frame_size, tof=True, tol=True):
    """状態の前処理"""

    def _preprocess(observation):
        """画像への前処理"""
        # 画像化
        img = Image.fromarray(observation)
        # サイズを入力サイズへ
        img = img.resize(INPUT_SHAPE)
        # グレースケールに
        img = img.convert('L') 
        # 配列に追加
        return np.array(img)

    # 状態は4つで1状態
    assert len(status) == frame_size

    state = np.empty((*INPUT_SHAPE, frame_size), 'int8')

    for i, s in enumerate(status):
        # 配列に追加
        state[:, :, i] = _preprocess(s)

    if tof:    
        # 画素値を0～1に正規化
        state = state.astype('float32') / 255.0

    if tol:
        state = state.tolist()

    return state
