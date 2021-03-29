# -*- coding: utf-8 -*-
"""
    File name:      emg_stat.py
    Description:    スペクトル統計値の算出
                    1. Mean frequency: MNF（平均周波数）
                    2. Median frequency: MDF（周波数中央値）
                    3. Total power: TTP（トータルパワー）
                    4. Mean power: MNP（平均パワー）
                    5. Peak frequency: PKF（ピーク周波数）
                    6. Variance of central frequency: VCF（分散）
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
from scipy import signal

Fs = 1000  # サンプリング周波数

Fh = 20.0  # ハイパス・フィルタ遮断周波数
Nf = 1  # ハイパス・フィルタの次数

Nt = 167  # FIRローパス・フィルタのタップ数（包絡線用）
Fc = 6.0  # FIRローパス・フィルタの遮断周波数（包絡線用）

AMP_COEF = 5.0 / 2**16 / 500 * 1000000  # 5V / 16bitADC / Gain=500 [uV]

plt.rcParams["font.size"] = 16  # フォントサイズ

# 統計値配列のインデックス
P_IDX = {
    'MNF': 0,
    'MDF': 1,
    'MDFi': 2,  # 周波数中央値のインデックス
    'TTP': 3,
    'MNP': 4,
    'PKF': 5,
    'PKFi': 6,  # ピーク周波数のインデックス
    'VCF': 7
}

FREQ_LIM = 250  # 周波数グラフのY軸レンジ

EMG_FILE_NAME = 'emg_arm2.txt'  # 筋電図データファイル

YLIM = [1400, 680]  # Ch1, Ch2の波形のY軸レンジ(+/-)
VMAX = [300, 150]  # Ch1, Ch2のスペクトログラムの強度スケール
YLIMF = [30, 85]  # スペクトログラムのY軸レンジ(最小周波数/最大周波数)

# 腕橈骨筋: Brachioradialis(BR)， 尺骨手根屈筋: Flexor carpi ulnaris(FCU)
LEGEND = ['BR', 'FCU']

PSD_IDX_1 = 25  # 力を入れ始めた区間の終了時のPSDデータのインデックス
PSD_IDX_2 = 85  # 測定終了前の区間の終了時のPSDデータのインデックス
MEAN_PERIOD = 5  # 上記区間のデータ数（それぞれの平均値を算出するのに使用）

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


def plot_wave(dat, ylab='EMG', title='', ylim=YLIM):
    """
    波形の描画.

    Parameters
    ----------
    dat : ndarray
        入力データ.
    ylab : str, optional
        Y軸ラベル. 初期値：'EMG'.
    title : str, optional
        グラフのタイトル. 初期値：''.
    ylim : float or list, optional
        波形のY軸レンジ(+/-). 初期値：YLIM.
    Returns
    -------
    なし.

    """
    t = np.arange(len(dat)) / Fs
    plt.figure(figsize=[11, 3])
    plt.plot(t, dat)
    plt.ylim(-ylim, ylim)
    plt.xlabel('Time [s]')
    plt.ylabel(ylab + ' [uV]')
    plt.title(title)
    plt.show()


def calc_spectrogram(dat, window='boxcar', is_plot=True, vmax=VMAX):
    """
    スペクトログラムの算出.

    Parameters
    ----------
    dat : ndarray (row: data, column: channels)
        入力データ.
    window : str, optional
        窓関数の名前. scipy.signal.get_window()参照.
        初期値は'boxcar'.
    is_plot : bool, optional
        グラフを描画するか. 初期値：True.
    vmax : float or list, optional
        カラーマップスケール. 初期値：VMAX.

    Returns
    -------
    f_list : list of ndarray
        周波数軸の値. (縦軸)
    t_list : list of ndarray
        時間軸の値. (横軸)
    Sxx_list : list of ndarray
        パワー・スペクトル（PSD）データ.

    """
    CH = dat.shape[1]
    f_list = []
    t_list = []
    Sxx_list = []
    for i in np.arange(CH):
        f, t, Sxx = signal.spectrogram(dat[:, i],
                                       Fs,
                                       window=window,
                                       nperseg=Fs * 2,
                                       noverlap=Fs,
                                       detrend=False,
                                       scaling='density',
                                       mode='psd')
        f_list.append(f)
        t_list.append(t)
        Sxx_list.append(Sxx)

        if is_plot:
            plt.figure(figsize=[13, 3])
            if type(vmax) == list:
                plt.pcolormesh(t, f, Sxx, cmap='jet', vmax=vmax[i])
            else:
                plt.pcolormesh(t, f, Sxx, cmap='jet', vmax=vmax)

            plt.ylim([0, FREQ_LIM])
            plt.xlabel('Time [s]')
            plt.ylabel('Freq [Hz]')
            plt.colorbar().set_label('PSD [uV^2/Hz]')
            plt.show()

    return f_list, t_list, Sxx_list


def calc_stat_params(f, Pxx):
    """
    スペクトル統計値の算出.

    Parameters
    ----------
    f : ndarray
        周波数データ.
    Pxx : ndarray
        スペクトルデータ.

    Returns
    -------
    mnf : float
        平均周波数.
    mdf : float
        周波数中央値.
    mdf_idx : integer
        周波数中央値のインデックス.
    ttp : float
        トータルパワー.
    mnp : float
        平均パワー.
    pkf : float
        ピーク周波数.
    pkf_idx : integer
        ピーク周波数のインデックス.
    vcf : float
        平均周波数周りの分散.

    """
    # 3. Total power: TTP（トータルパワー）
    ttp = np.sum(Pxx)

    # 1. bMean frequency: MNF（平均周波数）
    mnf = (Pxx @ f) / ttp

    # 2. Median frequency: MDF（周波数中央値）
    area = np.cumsum(Pxx)
    area_half = ttp * 0.5
    mdf_idx = np.squeeze(np.where(area >= area_half))[0]
    mdf = f[mdf_idx]

    # 4. Mean power: MNP（平均パワー）
    mnp = ttp / len(f)

    # 5. Peak frequency: PKF（ピーク周波数）
    pkf_idx = Pxx.argmax()
    pkf = f[pkf_idx]

    # 6. Variance of central frequency: VCF（平均周波数周りの分散）
    vcf = (Pxx @ (f**2)) / ttp - (mnf)**2

    return mnf, mdf, mdf_idx, ttp, mnp, pkf, pkf_idx, vcf


def plot_psd_mnf_mdf(f, Pxx, mnf, mdf, title=''):
    """
    平均周波数と周波数中央値の描画

    Parameters
    ----------
    f : ndarray
        周波数データ.
    Pxx : ndarray
        スペクトルデータ.
    mnf : float
        平均周波数.
    mdf : float
        周波数中央値.
    title : str, optional
        グラフのタイトル. 初期値：''.

    Returns
    -------
    なし.

    """
    plt.figure(figsize=[5, 3])
    ax = plt.subplot(111)
    plt.plot(f, Pxx)
    plt.axvline(x=mnf, color='k', ls='dashed', lw=0.5)
    plt.text(0.95,
             0.88,
             'MNF: {:.1f} Hz'.format(mnf),
             horizontalalignment='right',
             transform=ax.transAxes)
    plt.axvline(x=mdf, color='k', lw=0.5)
    plt.text(0.95,
             0.78,
             'MDF: {:.1f} Hz'.format(mdf),
             horizontalalignment='right',
             transform=ax.transAxes)
    plt.xlim(-FREQ_LIM * 0.05, FREQ_LIM * 1.05)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD [uV^2/Hz]')
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    dat = read_dat(EMG_FILE_NAME)

    # オフセット除去
    Nch = dat.shape[1]
    for i in np.arange(Nch):
        dat[:, i] = signal.lfilter(bh, ah, dat[:, i])
        plot_wave(dat[:, i], title=LEGEND[i], ylim=YLIM[i])

    # スペクトログラムの算出とグラフ描画
    f_list, t_list, Sxx_list = calc_spectrogram(dat, 'hann', True, VMAX)

    Ntime = len(t_list[0])
    Nparam = 8
    params = np.zeros((Nparam, Ntime, Nch))
    for ch in np.arange(Nch):
        # スペクトル統計値の算出
        for t in np.arange(Ntime):
            p = calc_stat_params(f_list[ch], Sxx_list[ch][:, t])
            for i in np.arange(Nparam):
                params[i, t, ch] = p[i]

        # スペクトル統計値のグラフ描画
        plt.figure(figsize=[11, 7])

        plt.subplot(311)
        plt.plot(params[P_IDX['PKF'], :, ch],
                 label='PKF',
                 color='gray',
                 lw=0.5)
        plt.plot(params[P_IDX['MNF'], :, ch], label='MNF', color='C0')
        plt.plot(params[P_IDX['MDF'], :, ch], label='MDF', color='k')
        plt.ylabel('Freq params [Hz]')
        plt.legend(loc='lower right', ncol=3)

        plt.subplot(312)
        ttp_mv = params[P_IDX['TTP'], :, ch] / 1000000  # uV^2 -> mV^2
        plt.plot(ttp_mv, label='TTP')
        plt.ylabel('TTP [mV^2]')

        plt.subplot(313)
        plt.plot(params[P_IDX['VCF'], :, ch], label='VCF')
        plt.ylabel('VCF [Hz^2]')
        plt.xlabel('Time [s]')

        plt.suptitle(LEGEND[ch])
        plt.show()

        # 力を入れ始めた区間のスペクトル
        plot_psd_mnf_mdf(f_list[ch], Sxx_list[ch][:, PSD_IDX_1],
                         params[P_IDX['MNF'], PSD_IDX_1,
                                ch], params[P_IDX['MDF'], PSD_IDX_1, ch],
                         str(LEGEND[ch]) + ' @ ' + str(PSD_IDX_1) + ' s')

        # 測定終了前の区間のスペクトル
        plot_psd_mnf_mdf(f_list[ch], Sxx_list[ch][:, PSD_IDX_2],
                         params[P_IDX['MNF'], PSD_IDX_2,
                                ch], params[P_IDX['MDF'], PSD_IDX_2, ch],
                         str(LEGEND[ch]) + ' @ ' + str(PSD_IDX_2) + ' s')

        # 各区間での平均周波数および周波数中央値の平均を算出
        bgn_idx = PSD_IDX_1 - MEAN_PERIOD
        end_idx = PSD_IDX_1
        mnf_mean = np.mean(params[P_IDX['MNF'], bgn_idx:end_idx, ch])
        mdf_mean = np.mean(params[P_IDX['MDF'], bgn_idx:end_idx, ch])
        print('Ch' + str(ch) + ' MNF: {:.2f} Hz'.format(mnf_mean) +
              ' MDF: {:.2f} Hz'.format(mdf_mean) + ' (' + str(bgn_idx) +
              ' to ' + str(end_idx) + ')')

        bgn_idx = PSD_IDX_2 - MEAN_PERIOD
        end_idx = PSD_IDX_2
        mnf_mean = np.mean(params[P_IDX['MNF'], bgn_idx:end_idx, ch])
        mdf_mean = np.mean(params[P_IDX['MDF'], bgn_idx:end_idx, ch])
        print('Ch' + str(ch) + ' MNF: {:.2f} Hz'.format(mnf_mean) +
              ' MDF: {:.2f} Hz'.format(mdf_mean) + ' (' + str(bgn_idx) +
              ' to ' + str(end_idx) + ')')
