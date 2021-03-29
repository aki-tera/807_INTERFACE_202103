# -*- coding: utf-8 -*-
"""
    File name:      ecg_qrs.py
    Description:    心電図波形からのハムノイズ除去
                    （QRS検出に py-ecg-detectors ライブラリ使用）
    Author:         Tetsuro Tatsuoka
    Date created:   2020/12/28
    Last modified:  2020/12/28
    Version:        1.0.0
    Python version: 3.7.7
    
    Revision history:
        1.0.0   2020/12/28  - Initial coding.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ecgdetectors import Detectors
from scipy import signal
from scipy.fft import rfft, rfftfreq

Fs = 1000  # サンプリング周波数

Fh = 1.0  # ハイパス・フィルタ遮断周波数
Fl = 30.0  # ローパス・フィルタ遮断周波数
Nf = 1  # ハイパス・フィルタ/ローパス・フィルタの次数

Fn = 50.0  # ノッチ・フィルタの中心周波数
Q = 4.0  # Q ノッチ・フィルタのQ値

M = 20  # 移動平均のポイント数 (1kSPS / 50Hz)

AMP_COEF = 5.0 / 2**16 / 505 * 1000  # 5V / 16bitADC / Gain=505 [mV]

plt.rcParams["font.size"] = 16  # フォントサイズ

YLIM = 0.75  # 波形のY軸レンジ(+/-)

DAT_LEN = 10  # データ長[s]

WINDOW = 'hann'  # 窓関数

ECG_FILE_NAME = 'ecg_with_noise.txt'  # 心電図データファイル
MATCHED_TEMPLATE = 'ecg_qrs_template.csv'  # Matched filter用テンプレートファイル

# フィルタの設計
bh, ah = signal.butter(Nf, Fh, 'high', fs=Fs)
bl, al = signal.butter(Nf, Fl, 'low', fs=Fs)
bn, an = signal.iirnotch(Fn, Q, fs=Fs)


def read_dat(filename):
    """
    波形データファイルの読み込み. (ASCII format)

    Parameters
    ----------
    filename : str
        データファイル名.

    Returns
    -------
    dat : ndarray
        波形データ.

    """
    dat = np.loadtxt(filename, delimiter='\t')
    dat = dat[0:int(Fs * DAT_LEN)] * AMP_COEF
    return dat


def plot_wave(dat, is_wide=True, peak=None, title=''):
    """
    波形データの描画.

    Parameters
    ----------
    dat : ndarray
        入力データ.
    is_wide : bool, optional
        True: 波形エリア横長, False: 波形エリア通常. 初期値：True.
    peak : list, optional
        R波ピークのインデックス. 初期値：None.
    title : str, optional
        グラフのタイトル. 初期値：''.

    Returns
    -------
    なし.

    """
    t = np.arange(len(dat)) / Fs
    if is_wide:
        plt.figure(figsize=[11, 3])
    else:
        plt.figure(figsize=[7, 3])

    plt.plot(t, dat, zorder=1)

    if peak is not None:
        plt.scatter(t[peak], dat[peak], marker='o', color='r', zorder=2)

    plt.ylim(-YLIM, YLIM)
    plt.xlabel('Time [s]')
    plt.ylabel('ECG [mV]')
    plt.title(title)
    plt.show()


def plot_spectrum(dat, window=WINDOW):
    """
    パワー・スペクトル密度の描画(PSD).

    Parameters
    ----------
    dat : ndarray
        入力データ.
    window : str, optional
        窓関数の名前. scipy.signal.get_window()参照.
        初期値はWINDOW.

    Returns
    -------
    なし.

    """
    YLIM_MIN = -140
    LEN = len(dat)
    win = signal.get_window(window, LEN)

    rfft_dat = rfft(dat * win)
    rfft_freq = rfftfreq(LEN, d=1.0 / Fs)
    sp_rdat = np.abs(rfft_dat)**2 / (LEN * LEN)
    sp_rdat[1:-1] *= 2

    plt.figure(figsize=[7, 3])
    plt.plot(rfft_freq, 10 * np.log10(sp_rdat))
    plt.ylim(YLIM_MIN, 0)
    plt.yticks(np.arange(YLIM_MIN, 0, 20))
    plt.grid()
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Power [dBmVrms]')

    plt.show()


def apply_detectors(detectors, dat):
    """
    QRSの検出.（7種のアルゴリズムを適用）

    Parameters
    ----------
    detectors : Detectors object
        QRS検出オブジェクト.
    dat : ndarray
        入力データ.

    Returns
    -------
    なし.

    """
    plot_wave(dat, title='Original waveform')
    r_peaks1 = detectors.hamilton_detector(dat)
    plot_wave(dat, peak=r_peaks1, title='1. Hamilton')
    r_peaks2 = detectors.christov_detector(dat)
    plot_wave(dat, peak=r_peaks2, title='2. Christov')
    r_peaks3 = detectors.engzee_detector(dat)
    plot_wave(dat, peak=r_peaks3, title='3. Engelse and Zeelenberg')
    r_peaks4 = detectors.pan_tompkins_detector(dat)
    plot_wave(dat, peak=r_peaks4, title='4. Pan and Tompkins')
    r_peaks5 = detectors.swt_detector(dat)
    plot_wave(dat, peak=r_peaks5, title='5. Stationary Wavelet Transform')
    r_peaks6 = detectors.two_average_detector(dat)
    plot_wave(dat, peak=r_peaks6, title='6. Two Moving Average')
    r_peaks7 = detectors.matched_filter_detector(dat, MATCHED_TEMPLATE)
    plot_wave(dat, peak=r_peaks7, title='7. Matched Filter')


if __name__ == "__main__":
    # QRS検出オブジェクトの生成
    detectors = Detectors(Fs)

    # 心電図データの読み込み，オフセット除去
    ecg = read_dat(ECG_FILE_NAME)
    ecg = ecg - np.mean(ecg)

    # 処理前の波形とスペクトルの描画
    plot_wave(ecg)
    plot_spectrum(ecg)

    # ノッチ・フィルタを適用した波形とスペクトルの描画，QRS検出
    ecg_notch = signal.lfilter(bn, an, ecg)
    plot_wave(ecg_notch)
    plot_spectrum(ecg_notch)
    apply_detectors(detectors, ecg_notch)

    # 移動平均を適用した波形とスペクトルの描画，QRS検出
    ecg_mva = pd.Series(ecg).rolling(window=M).mean().dropna()
    ecg_mva = np.array(ecg_mva)
    plot_wave(ecg_mva)
    plot_spectrum(ecg_mva)
    apply_detectors(detectors, ecg_mva)

    # 移動平均＋バンドパス・フィルタを適用した波形とスペクトルの描画，QRS検出
    ecg_filt = signal.lfilter(bh, ah, ecg_mva)
    ecg_filt = signal.lfilter(bl, al, ecg_filt)
    apply_detectors(detectors, ecg_filt)

    if False:  # Matched Filterアルゴリズム用テンプレート波形を生成するためのコード
        BEAT_LEN = 1000
        BEAT_BEGIN = -400
        BEAT_END = 600
        r_peaks = detectors.engzee_detector(ecg_filt)
        ave = np.zeros(BEAT_LEN)
        for peak in r_peaks:
            ave += ecg_filt[peak + BEAT_BEGIN:peak + BEAT_END]
        ave = ave / len(r_peaks)

        t = np.arange(BEAT_BEGIN, BEAT_END)
        t = t / Fs
        plt.figure()
        plt.plot(t, ave)
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude [mV]')
        plt.show()

        QRS_BEGIN = -50
        QRS_END = 70
        QRS_BEGIN_IDX = QRS_BEGIN - BEAT_BEGIN
        QRS_END_IDX = QRS_END - BEAT_BEGIN
        t = np.arange(QRS_BEGIN, QRS_END)
        plt.figure()
        plt.plot(t, ave[QRS_BEGIN_IDX:QRS_END_IDX])
        plt.xlabel('Time [ms]')
        plt.ylabel('Amplitude [mV]')
        plt.show()

        qrs_ave = ave[QRS_BEGIN_IDX:QRS_END_IDX]
        np.savetxt(MATCHED_TEMPLATE, qrs_ave, fmt="%0.6g", delimiter=',')
