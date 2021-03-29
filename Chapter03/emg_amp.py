# -*- coding: utf-8 -*-
"""
    File name:      emg_amp.py
    Description:    強度波形の描画
                    - 全波整流平滑化（積分筋電図: iEMG）
                    - 実効値波形（RMS波形）
                    - 包絡線
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
from scipy import signal

Fs = 1000  # サンプリング周波数

Fh = 20.0  # ハイパス・フィルタ遮断周波数
Nf = 1  # ハイパス・フィルタの次数

Nt = 167  # FIRローパス・フィルタのタップ数（包絡線用）
Fc = 6.0  # FIRローパス・フィルタの遮断周波数（包絡線用）

AMP_COEF = 5.0 / 2**16 / 500 * 1000000  # 5V / 16bitADC / Gain=500 [uV]

plt.rcParams["font.size"] = 16  # フォントサイズ

YLIM = 740  # 波形のY軸レンジ(+/-)

# 総指伸筋: Extensor digitorum(ED)， 尺骨手根屈筋: Flexor carpi ulnaris(FCU)
LEGEND = ['ED', 'FCU']

EMG_FILE_NAME = 'emg_arm.txt'  # 筋電図データファイル

# フィルタの設計
bh, ah = signal.butter(Nf, Fh, 'high', fs=Fs)


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
    dat = dat * AMP_COEF
    return dat


def plot_wave_each(dat, ylim=YLIM, ylab1='Ch1', ylab2='Ch2', title=''):
    """
    波形を2つのペインに別々に描画.

    Parameters
    ----------
    dat : ndarray (row: data, column: channels)
        入力データ.
    ylim : float, optional
        波形のY軸レンジ(+/-). 初期値：YLIM.
    ylab1 : str, optional
        Y軸ラベル（上側のペイン）. 初期値：'Ch1'.
    ylab2 : str, optional
        Y軸ラベル（上側のペイン）. 初期値：'Ch2'.
    title : str, optional
        グラフのタイトル. 初期値：''.

    Returns
    -------
    なし.

    """
    t = np.arange(len(dat)) / Fs
    plt.figure(figsize=[11, 6])

    plt.subplot(211)
    plt.plot(t, dat[:, 0])
    plt.ylim(-ylim, ylim)
    plt.ylabel(ylab1 + ' [uV]')
    plt.title(title)

    plt.subplot(212)
    plt.plot(t, dat[:, 1])
    plt.ylim(-ylim, ylim)
    plt.ylabel(ylab2 + ' [uV]')
    plt.xlabel('Time [s]')

    plt.show()


def calc_iemg(dat, fs=Fs):
    """
    全波整流平滑化（積分筋電図: iEMG (Integral EMG)）波形の算出.
    ローパス・フィルタ (2.6Hz 1次バターワース)

    Parameters
    ----------
    dat : ndarray (row: data, column: channels)
        入力データ.（オフセットは除去されたもの）
    fs : float, optional
        サンプリング周波数. 初期値：Fs.

    Returns
    -------
    iemg : ndarray
        iEMG波形データ.

    """
    N = 1  # フィルタ次数
    Fc = 2.6  # 遮断周波数
    b, a = signal.butter(N, Fc, 'low', fs=fs)
    iemg = np.empty_like(dat)
    for i in np.arange(dat.shape[1]):
        iemg[:, i] = signal.lfilter(b, a, np.abs(dat[:, i]))

    return iemg


def calc_rms_emg(dat, fs=Fs):
    """
    実効値波形（RMS波形）の算出. (Window size: 100ms)

    Parameters
    ----------
    dat : ndarray (row: data, column: channels)
        入力データ.（オフセットは除去されたもの）
    fs : float, optional
        サンプリング周波数. 初期値：Fs.

    Returns
    -------
    rms_emg : ndarray
        RMS波形データ.

    """
    win_len = int(0.1 * fs / 2.0) * 2 + 1  # 0.1s = 100ms
    rms_emg = np.empty_like(dat)
    for i in np.arange(dat.shape[1]):
        tmp = dat[:, i]**2
        tmp = pd.Series(tmp).rolling(window=win_len,
                                     min_periods=1,
                                     center=True).mean()
        rms_emg[:, i] = np.sqrt(np.squeeze(np.array(tmp)))

    return rms_emg


def calc_env_emg(dat, fs=Fs, nt=Nt, fc=Fc):
    """
    包絡線の算出.

    Parameters
    ----------
    dat : ndarray (row: data, column: channels)
        入力データ.（オフセットは除去されたもの）
    fs : float, optional
        サンプリング周波数. 初期値：Fs.
    nt : integer, optional
        FIRフィルタのタップ数. 初期値：Nt.
    fc : float, optional
        FIRフィルタの遮断周波数. 初期値：Fc.

    Returns
    -------
    emv_emg : ndarray
        包絡線データ.

    """
    shift = int(nt / 2)
    b = signal.firwin(nt, fc, fs=fs)
    emv_emg = np.empty_like(dat)
    for i in np.arange(dat.shape[1]):
        tmp = (dat[:, i]**2) * 2
        tmp = np.convolve(tmp, b, 'full')
        tmp = np.sqrt(np.abs(tmp))
        emv_emg[:, i] = tmp[shift + 1:-shift + 1]

    return emv_emg


if __name__ == "__main__":
    dat = read_dat(EMG_FILE_NAME)

    # オフセット除去
    CH = dat.shape[1]
    for i in np.arange(CH):
        dat[:, i] = signal.lfilter(bh, ah, dat[:, i])

    # 筋電図の10秒間の拡大波形
    S_IDX = 30100
    E_IDX = 40100
    YLIM_EX = 550
    plot_wave_each(dat[S_IDX:E_IDX, :], YLIM_EX, LEGEND[0], LEGEND[1])

    # 強度波形の描画
    # EMG（元波形）
    amp_mean = np.mean(np.abs(dat))
    plot_wave_each(dat, YLIM, LEGEND[0], LEGEND[1], 'EMG waveforms')

    # iEMG（全波整流平滑化）
    iemg = calc_iemg(dat)
    plot_wave_each(iemg, YLIM, LEGEND[0], LEGEND[1], 'Integral EMG')
    iemg_mean = np.mean(iemg)
    iemg_norm = iemg * amp_mean / iemg_mean
    plot_wave_each(iemg_norm, YLIM, LEGEND[0], LEGEND[1],
                   'Integral EMG (normalized)')

    # RMS（実効値波形）
    rms_emg = calc_rms_emg(dat)
    plot_wave_each(rms_emg, YLIM, LEGEND[0], LEGEND[1], 'RMS EMG')
    rms_mean = np.mean(rms_emg)
    rms_norm = rms_emg * amp_mean / rms_mean
    plot_wave_each(rms_norm, YLIM, LEGEND[0], LEGEND[1],
                   'RMS EMG (normalized)')

    # Envelope（包絡線）
    env_emg = calc_env_emg(dat)
    plot_wave_each(env_emg, YLIM, LEGEND[0], LEGEND[1], 'Envelope EMG')
    env_mean = np.mean(env_emg)
    env_norm = env_emg * amp_mean / env_mean
    plot_wave_each(env_norm, YLIM, LEGEND[0], LEGEND[1],
                   'Envelope EMG (normalized)')

    # 一部区間を拡大し全波形重ね書き
    plt.rcParams["font.size"] = 12  # フォントサイズ

    t = np.arange(len(dat)) / Fs
    plt.figure(figsize=[11, 6])
    for i in np.arange(2):
        plt.subplot(2, 1, i + 1)
        abs_emg = np.abs(dat[:, i])
        plt.plot(t, abs_emg, label='Abs EMG', color='gray', alpha=0.5)
        plt.plot(t, iemg[:, i], label='iEMG', color='C0')
        plt.plot(t, rms_emg[:, i], label='RMS', color='k', lw=1)
        plt.plot(t, env_emg[:, i], label='Envelope', color='C3', lw=3)
        plt.xlim(4, 12.5)
        plt.ylabel(LEGEND[i] + ' [uV]')
        plt.legend(loc='upper left')
        if i == 0:
            plt.ylim(-30, 350)
            plt.title('Comparison')
        else:
            plt.ylim(-5, 80)
            plt.xlabel('Time [s]')

    plt.show()
