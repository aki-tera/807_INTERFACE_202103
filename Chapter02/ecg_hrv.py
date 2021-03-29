# -*- coding: utf-8 -*-
"""
    File name:      ecg_hrv.py
    Description:    心拍変動解析（Heart rate variability: HRV）
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
from gatspy.periodic import LombScargleFast
from hrv import HRV
from scipy import signal
from scipy.fft import rfft, rfftfreq
from scipy.interpolate import interp1d
from spectrum import arma2psd, aryule

Fs = 200  # サンプリング周波数

Fh = 1.0  # ハイパス・フィルタ遮断周波数
Fl = 30.0  # ローパス・フィルタ遮断周波数
Nf = 1  # ハイパス・フィルタ/ローパス・フィルタの次数

Fn = 50.0  # ノッチ・フィルタの中心周波数
Q = 4.0  # Q ノッチ・フィルタのQ値

M = 4  # 移動平均のポイント数 (200SPS / 50Hz)

AMP_COEF = 5.0 / 2**12 / 1000 * 1000  # 5V / 12bitADC / Gain=1000 [mV]

plt.rcParams['font.size'] = 18  # フォントサイズ

LFHF_FONT = 22  # グラフ中の LF/HF 比のフォントサイズ

YLIM = 1.0  # 波形のY軸レンジ(+/-)

CH = 0  # 対象チャネル (0: Ch1, 1: Ch2)
DAT_LEN = 10  # データ長[s]

WINDOW = 'hamming'  # 窓関数

LF_MIN = 0.04  # LF周波数範囲（下限）
LF_MAX = 0.15  # LF周波数範囲（上限）
HF_MIN = 0.15  # HF周波数範囲（下限）
HF_MAX = 0.40  # HF周波数範囲（上限）

ECG_FILE_NAME1 = 'ecg_relax.txt'  # 心電図データファイル（リラックス時）
ECG_FILE_NAME2 = 'ecg_terror.txt'  # 心電図データファイル（ストレス時）

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
    dat = dat[:, CH] * AMP_COEF
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


def plot_lf_hf(freq, psd, lf_hf, ylim, title=''):
    """
    パワー・スペクトル密度(PSD)およびLF/HF比の描画.

    Parameters
    ----------
    freq : ndarray
        周波数データ.
    psd : ndarray
        パワー・スペクトル密度(PSD)データ.
    lf_hf : float
        LF/HF比.
    title : str, optional
        グラフのタイトル. 初期値：''.

    Returns
    -------
    なし.

    """
    lf_axis = np.where((freq >= LF_MIN) & (freq <= LF_MAX))[0]
    hf_axis = np.where((freq >= HF_MIN) & (freq <= HF_MAX))[0]
    lf_fill = np.append(lf_axis, np.max(lf_axis) + 1)
    hf_fill = np.append(hf_axis, np.max(hf_axis) + 1)
    plt.figure()
    ax = plt.subplot(111)
    plt.plot(freq, psd, lw=0.2, color='k')
    plt.fill_between(freq[lf_fill], psd[lf_fill], facecolor='red')
    plt.fill_between(freq[hf_fill], psd[hf_fill], facecolor='lime')
    plt.ylim(-ylim / 20, ylim)
    plt.text(0.95,
             0.85,
             'LF/HF: {:.3f}'.format(lf_hf),
             horizontalalignment='right',
             transform=ax.transAxes,
             fontsize=LFHF_FONT)
    plt.title(title)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD')
    plt.show()


def fAnalysis(r_peaks):
    """
    Lomb-Scargle法によるスペクトルの算出とLF/HF比の描画.
    （py-ecg-detectorsライブラリの fAnalysis() 関数にグラフ描画機能を追加）

    Parameters
    ----------
    r_peaks : list
        R波ピークのインデックス配列.

    Returns
    -------
    lf_hf : float
        LF/HF比.

    """
    # fAnalysis()でのrr_intの算出（丸め誤差で微妙に値が異なる）
    # itvl = 1.0 / Fs
    # rr_int = np.diff(np.array(r_peaks) * itvl * 1000) / 1000

    rr_int = np.diff(np.array(r_peaks)) / Fs
    t_rr_int = [i / Fs for i in r_peaks[1:]]
    model = LombScargleFast().fit(t_rr_int, rr_int, 1E-2)
    fmax = 1
    fmin = 0.01
    nsamp = 1000
    df = (fmax - fmin) / nsamp
    power = model.score_frequency_grid(fmin, df, nsamp)
    freq = fmin + df * np.arange(nsamp)

    lf = 0
    hf = 0
    for i in range(int(nsamp / 2)):
        if (freq[i] >= LF_MIN) and (freq[i] <= LF_MAX):
            lf = lf + power[i]
        if (freq[i] >= HF_MIN) and (freq[i] <= HF_MAX):
            hf = hf + power[i]
    lf_hf = lf / hf

    plot_lf_hf(freq, power, lf_hf, 0.12, 'hrv.fAnalysis()')

    return lf_hf


def lfhf_fft(r_peaks):
    """
    5種類の方法によるLF/HF比の算出.
    ・FFT
    ・ピリオドグラム
    ・ユールウォーカ―AR法（次数1/8）
    ・ユールウォーカ―AR法（次数1/16）
    ・ウェルチ法

    Parameters
    ----------
    r_peaks : list
        R波ピークのインデックス配列.

    Returns
    -------
    lf_hf: list
        LF/HF比.(FFT, Periodgram, AR 1/8, AR 1/16, Welch)

    """
    # 2Hzでリサンプリング
    Fsr = 2.0
    r_peaks_sec = np.array(r_peaks) / Fs
    rr_int = np.diff(r_peaks_sec)
    rs_timing = np.arange(np.ceil(r_peaks_sec[1]), np.floor(r_peaks_sec[-1]),
                          1 / Fsr)
    f = interp1d(r_peaks_sec[1:], rr_int, kind='cubic')
    rr_int_rs = f(rs_timing)
    N = len(rr_int_rs)

    # 窓関数の生成
    win = signal.get_window(WINDOW, len(rr_int_rs))

    # FFT
    fft_res = rfft(rr_int_rs * win)
    freq = rfftfreq(N, d=1.0 / Fsr)
    power = np.abs(fft_res)**2 / N / Fsr
    power[1:-1] *= 2
    idx_1Hz = int(N // Fsr)
    lf_axis = np.where((freq >= LF_MIN) & (freq <= LF_MAX))[0]
    hf_axis = np.where((freq >= HF_MIN) & (freq <= HF_MAX))[0]

    lf = np.sum(power[lf_axis])
    hf = np.sum(power[hf_axis])
    lf_hf_1 = lf / hf

    plot_lf_hf(freq[2:idx_1Hz], power[2:idx_1Hz], lf_hf_1, 0.01, 'FFT')

    # Periodgram
    f1, Pxx1 = signal.periodogram(rr_int_rs, Fsr, window=WINDOW)
    lf = np.sum(Pxx1[lf_axis])
    hf = np.sum(Pxx1[hf_axis])
    lf_hf_2 = lf / hf

    plot_lf_hf(f1[2:idx_1Hz], Pxx1[2:idx_1Hz], lf_hf_2, 0.02, 'Periodgram')

    # AR model N: 1/8
    p = N // 8
    AR, P, k = aryule(rr_int_rs * win, p)
    psd = arma2psd(AR, NFFT=N)
    freq = np.linspace(0, Fsr, N, endpoint=False)
    psd = psd / N / Fsr
    lf = np.sum(psd[lf_axis])
    hf = np.sum(psd[hf_axis])
    lf_hf_3 = lf / hf

    plot_lf_hf(freq[2:idx_1Hz], psd[2:idx_1Hz], lf_hf_3, 0.12,
               'Yule Walker AR (Order:N/8)')

    # AR model N: 1/16
    p = N // 16
    AR, P, k = aryule(rr_int_rs * win, p)
    psd = arma2psd(AR, NFFT=N)
    freq = np.linspace(0, Fsr, N, endpoint=False)
    psd = psd / N / Fsr
    lf = np.sum(psd[lf_axis])
    hf = np.sum(psd[hf_axis])
    lf_hf_4 = lf / hf

    plot_lf_hf(freq[2:idx_1Hz], psd[2:idx_1Hz], lf_hf_4, 0.12,
               'Yule Walker AR (Order:N/16)')

    # Welch
    f2, Pxx2 = signal.welch(rr_int_rs,
                            Fsr,
                            window=WINDOW,
                            nperseg=N // 2,
                            noverlap=N // 4)
    idx_1Hz = idx_1Hz // 2
    lf_axis = np.where((f2 >= LF_MIN) & (f2 <= LF_MAX))[0]
    hf_axis = np.where((f2 >= HF_MIN) & (f2 <= HF_MAX))[0]
    lf = np.sum(Pxx2[lf_axis])
    hf = np.sum(Pxx2[hf_axis])
    lf_hf_5 = lf / hf

    plot_lf_hf(f2[2:idx_1Hz], Pxx2[2:idx_1Hz], lf_hf_5, 0.02, 'Welch')

    return lf_hf_1, lf_hf_2, lf_hf_3, lf_hf_4, lf_hf_5


def detect_qrs(filename):
    """
    QRS波の検出.

    Parameters
    ----------
    filename : str
        心電図データファイル名.

    Returns
    -------
    なし.

    """
    detectors = Detectors(Fs)

    ecg = read_dat(filename)
    ecg = ecg - np.mean(ecg)
    plot_wave(ecg, title='Raw waveform')

    ecg_mva = pd.Series(ecg).rolling(window=M).mean().dropna()
    ecg_mva = np.squeeze(np.array(ecg_mva))

    ecg_filt = signal.lfilter(bh, ah, ecg_mva)
    if False:  # True if applying low pass filter
        ecg_filt = signal.lfilter(bl, al, ecg_filt)

    qrs = detectors.engzee_detector(ecg_filt)
    plot_wave(ecg_filt, peak=qrs, title='Detect QRS in filterd wave')
    plt.figure()
    plt.hist(np.diff(qrs) / Fs, bins=10, range=(0.7, 1.2))
    plt.xlabel('R-R interval [s]')
    plt.ylabel('Frequency')
    plt.show()
    return ecg, qrs


def exec_hrv(r_peaks, num_period):
    """
    心拍変動解析(HRV)の実行.

    Parameters
    ----------
    r_peaks : list
        R波ピークのインデックス配列.
    num_period : int
        解析する区間の数.

    Returns
    -------
    なし.

    """
    HRV_SPAN = 200  # HRV解析データ長[s]
    HRV_STEP = 200  # HRV解析データシフト量[s] (オーバーラップなし)
    hrv = HRV(Fs)

    for i in np.arange(num_period):
        b_pts = HRV_STEP * Fs * i
        e_pts = b_pts + HRV_SPAN * Fs
        idx = np.where((b_pts <= r_peaks) & (r_peaks < e_pts))[0]
        r_peaks_sub = r_peaks[min(idx):max(idx)]

        plt.figure()
        plt.plot(np.array(r_peaks_sub[1:]) / Fs, hrv.HR(r_peaks_sub))
        plt.ylim(50, 80)
        plt.xlabel('Time [s]')
        plt.ylabel('Heart rate [bpm]')
        plt.show()
        print(hrv.fAnalysis(r_peaks_sub))  # ライブラリのfAnalysis()関数（確認用）
        print(fAnalysis(r_peaks_sub))  # グラフ描画機能を追加したfAnalysis()関数
        print(lfhf_fft(r_peaks_sub))  # 5種の等間隔サンプリングによる解析


if __name__ == "__main__":

    HRV_NUM_PERIOD_1 = 1  # リラックス時のデータは1区間分解析
    HRV_NUM_PERIOD_2 = 3  # ストレス時のデータは3区間分解析

    # リラックス
    ecg_1, qrs_1 = detect_qrs(ECG_FILE_NAME1)
    exec_hrv(qrs_1, HRV_NUM_PERIOD_1)

    # ストレス
    ecg_2, qrs_2 = detect_qrs(ECG_FILE_NAME2)
    exec_hrv(qrs_2, HRV_NUM_PERIOD_2)
