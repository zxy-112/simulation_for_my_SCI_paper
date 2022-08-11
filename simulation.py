from itertools import chain
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import aray
import signl
from utils import *

SAVE_PATH = os.path.join(os.path.expanduser("~"), "coherent_simulation")
if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)
SAVE_FLAG = True
mcmvDoaError = 0.5

colors = ['#037ef3', '#f85a40', '#00c16e', '#7552cc', '#0cb9c1', '#52565e', '#ffc845', '#a51890']
linestyles = [
    "solid",
    (0, (1, 1)),
    (0, (5, 5)),
    (0, (5, 1)),
    (0, (3, 1, 1, 1)),
    (0, (3, 1, 1, 1, 1, 1))
]

def savefig(fig, name):
    fig.savefig(os.path.join(SAVE_PATH, name), dpi=1200, transparent=True)
    plt.close(fig)

def array_maker(elementNumber):
    ary = aray.UniformLineArray(interval=0.5)
    for _ in range(elementNumber):
        ary.add_element(aray.Element())
    return ary

def decibel2value(x):
    return np.sqrt(np.power(10, x / 10) * aray.UniformLineArray.noise_power)

def oneCoherentInterference():

    # initialization
    array16 = array_maker(16)
    expectTheta = 0
    coherentTheta = 20
    snr = 0
    cnr = 10
    samplePoints = 1024
    expectSignal = signl.LfmWave2D(expectTheta, amplitude=decibel2value(snr))
    coherentSignal = signl.LfmWave2D(coherentTheta, amplitude=decibel2value(cnr))  # change this line to use other interference
    # coherentSignal = signl.CosWave2D(coherentTheta, fre_shift=1e6, amplitude=decibel2value(cnr))
    coherentSignal = signl.CossWave2D(coherentTheta, fres=(-4e6, -2e6, 1e6, 2e6), amplitude=decibel2value(cnr))
    array16.receive_signal(expectSignal)
    array16.receive_signal(coherentSignal)
    array16.sample(samplePoints)
    output = array16.output
    receivedSignal = output[0, :]

    # acquire weight and response and beamformer output
    weight, beamformerOutput = mvdr(output, expectTheta, steer_func=array16.steer_vector, returnOutput=True)  # change this line to use different algorithms
    # weight, beamformerOutput = smooth2(output, expectTheta, 2, steer_func=array16.steer_vector, returnOutput=True)
    arrayForPlot = array_maker(weight.size)
    response, thetas = arrayForPlot.response_with(-90, 90, 1801, weight, True)

    # plot
    fig, axs = plt.subplots(2, 5, constrained_layout=True, figsize=(18, 7))
    expectSignal.plot(fig_ax_pair=(fig, axs[0, 0]), color=colors[0], linewidth=2)
    expectSignal.fft_plot(fig_ax_pair=(fig, axs[1, 0]), color=colors[0], linewidth=2)
    coherentSignal.plot(fig_ax_pair=(fig, axs[0, 1]), color=colors[0], linewidth=2)
    coherentSignal.fft_plot(fig_ax_pair=(fig, axs[1, 1]), color=colors[0], linewidth=2)
    axs[0, 2].plot(np.real(receivedSignal), color=colors[0], linewidth=2)
    axs[1, 2].plot(fftfreq(len(receivedSignal)), normalize(np.abs(fft(receivedSignal))), color=colors[0], linewidth=2)
    axs[0, 3].plot(np.real(beamformerOutput), color=colors[0], linewidth=2)
    axs[1, 3].plot(fftfreq(len(beamformerOutput)), normalize(np.abs(fft(beamformerOutput))), color=colors[0], linewidth=2)
    for ax in axs[1, :4]:
        ax.set_xlim([-0.1, 0.1])
    arrayForPlot.response_plot(weight, (fig, axs[0, 4]), color=colors[0], linewidth=2)
    axs[1, 4].plot(thetas, np.angle(response), color=colors[0], linewidth=2)
    for ax in (axs[:, 4]):
        ax.axvline(expectTheta, color=colors[2], linewidth=1, linestyle="dashed")
        ax.axvline(coherentTheta, color=colors[1], linewidth=1, linestyle="dashed")

    # title
    axs[0, 0].set_title("expect signal", fontweight="bold")
    axs[1, 0].set_title("spectrum of expect signal", fontweight="bold")
    axs[0, 1].set_title("interference signal", fontweight="bold")
    axs[1, 1].set_title("spectrum of interference", fontweight="bold")
    axs[0, 2].set_title("received signal", fontweight="bold")
    axs[1, 2].set_title("spectrum of received signal", fontweight="bold")
    axs[0, 3].set_title("beamformer output", fontweight="bold")
    axs[1, 3].set_title("spectrum of beamformer output", fontweight="bold")
    axs[0, 4].set_title("amplitude response", fontweight="bold")
    axs[1, 4].set_title("phase response", fontweight="bold")

    # label
    for ax in axs[0, :4]:
        ax.set_xlabel("sample")
    for ax in axs[1, :4]:
        ax.set_xlabel("frequency(normalized)")
    for ax in axs[:, -1]:
        ax.set_xlabel("degree(\u00b0)")

    if SAVE_FLAG:
        savefig(fig, "oneCoherentInterference.svg")

    fig, axs = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    axs[0].plot(fftfreq(len(beamformerOutput)), normalize(np.abs(fft(beamformerOutput))), color=colors[0], linewidth=2)
    axs[0].set_xlabel("frequency(normalized)")
    axs[0].set_title("spectrum of beamformer output")
    arrayForPlot.response_plot(weight, (fig, axs[1]), color=colors[0], linewidth=2)
    axs[1].set_xlabel("degree(\u00b0)")
    axs[1].set_title("amplitude response")

    if SAVE_FLAG:
        savefig(fig, "oneCoherentInterferenceTwoAxes.svg")
    
def oneCoherentSeveralIncoherent():

    # initialization
    array16 = array_maker(16)
    expectTheta = 10
    coherentTheta = 20
    incoherentThetas = [-25, -10, 0]
    snr = 0
    cnr = 10
    inr = 10
    samplePoints = 1024
    expectSignal = signl.LfmWave2D(expectTheta, amplitude=decibel2value(snr))
    coherentSignal = signl.LfmWave2D(coherentTheta, amplitude=decibel2value(cnr))  # change this line to use other interference
    # coherentSignal = signl.CosWave2D(coherentTheta, fre_shift=1e6, amplitude=decibel2value(cnr))
    # coherentSignal = signl.CossWave2D(coherentTheta, fres=(-4e6, -2e6, 1e6, 2e6), amplitude=decibel2value(cnr))
    incoherentSignals = [signl.NoiseWave2D(theta, amplitude=decibel2value(inr)) for theta in incoherentThetas]
    for signal in chain((expectSignal, coherentSignal), incoherentSignals):
        array16.receive_signal(signal)
    array16.sample(samplePoints)
    output = array16.output
    receivedSignal = output[0, :]
    
    # acquire weight and response and beamformer output
    # weight, beamformerOutput = mvdr(output, expectTheta, steer_func=array16.steer_vector, returnOutput=True)  # change this line to use different algorithms
    # weight, beamformerOutput = smooth2(output, expectTheta, 2, steer_func=array16.steer_vector, returnOutput=True)
    weight, beamformerOutput = proposed(output, expectTheta, array16.steer_vector, True)
    arrayForPlot = array_maker(weight.size)
    response, thetas = arrayForPlot.response_with(-90, 90, 1801, weight, True)

    # plot
    fig, axs = plt.subplots(2, 4, constrained_layout=True, figsize=(14, 7))
    expectSignal.plot(fig_ax_pair=(fig, axs[0, 0]), color=colors[0])
    expectSignal.fft_plot(fig_ax_pair=(fig, axs[1, 0]), color=colors[0])
    axs[0, 1].plot(np.real(receivedSignal), color=colors[0], linewidth=2)
    axs[1, 1].plot(fftfreq(len(receivedSignal)), normalize(np.abs(fft(receivedSignal))), color=colors[0], linewidth=2)
    axs[0, 2].plot(np.real(beamformerOutput), color=colors[0], linewidth=2)
    axs[1, 2].plot(fftfreq(len(beamformerOutput)), normalize(np.abs(fft(beamformerOutput))), color=colors[0], linewidth=2)
    for ax in axs[1, 0:3]:
        ax.set_xlim([-0.1, 0.1])
    arrayForPlot.response_plot(weight, (fig, axs[0, 3]), color=colors[0], linewidth=2)
    axs[1, 3].plot(thetas, np.angle(response), color=colors[0], linewidth=2)
    for ax in (axs[:, 3]):
        ax.axvline(expectTheta, color=colors[2], linewidth=1, linestyle="dashed")
        ax.axvline(coherentTheta, color=colors[1], linewidth=1, linestyle="dashed")
        for theta in incoherentThetas:
            ax.axvline(theta, color=colors[3], linewidth=1, linestyle="dashed")

    # title
    axs[0, 0].set_title("expect signal", fontweight="bold")
    axs[1, 0].set_title("spectrum of expect signal", fontweight="bold")
    axs[0, 1].set_title("received signal", fontweight="bold")
    axs[1, 1].set_title("spectrum of received signal", fontweight="bold")
    axs[0, 2].set_title("beamformer output", fontweight="bold")
    axs[1, 2].set_title("spectrum of beamformer output", fontweight="bold")
    axs[0, 3].set_title("amplitude response", fontweight="bold")
    axs[1, 3].set_title("phase response", fontweight="bold")

    # label
    for ax in axs[0, :3]:
        ax.set_xlabel("sample")
    for ax in axs[1, :3]:
        ax.set_xlabel("frequency(normalized)")
    for ax in axs[:, -1]:
        ax.set_xlabel("degree(\u00b0)")
    if SAVE_FLAG:
        savefig(fig, "oneCoherentSeveralIncoherent.svg")

    fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(8, 3))
    axs[0].plot(fftfreq(len(beamformerOutput)), normalize(np.abs(fft(beamformerOutput))), color=colors[0], linewidth=2)
    axs[0].set_xlabel("frequency(normalized)")
    axs[0].set_title("spectrum of beamformer output")
    axs[0].set_xlim([-0.1, 0.1])
    arrayForPlot.response_plot(weight, (fig, axs[1]), color=colors[0], linewidth=2)
    axs[1].set_xlabel("degree(\u00b0)")
    axs[1].set_title("amplitude response")
    axs[1].axvline(expectTheta, color=colors[2], linewidth=1, linestyle="dashed")
    axs[1].axvline(coherentTheta, color=colors[1], linewidth=1, linestyle="dashed")
    for theta in incoherentThetas:
        axs[1].axvline(theta, color=colors[3], linewidth=1, linestyle="dashed")

    if SAVE_FLAG:
        savefig(fig, "oneCoherentSeveralIncoherentTwoAxes.svg")
    
def patternComparsion():

    # initialization
    array16 = array_maker(16)
    expectTheta = 10
    coherentTheta = 20
    incoherentThetas = [-25, -10, 0]
    snr = 0
    cnr = 10
    inr = 10
    samplePoints = 1024
    expectSignal = signl.LfmWave2D(expectTheta, amplitude=decibel2value(snr))
    coherentSignal = signl.LfmWave2D(coherentTheta, amplitude=decibel2value(cnr)) # change this line to use partially coherent interference
    # coherentSignal = signl.CosWave2D(coherentTheta, amplitude=decibel2value(cnr), fre_shift=1e6)
    incoherentSignals = [signl.NoiseWave2D(theta, amplitude=decibel2value(inr)) for theta in incoherentThetas]
    for signal in chain((expectSignal, coherentSignal), incoherentSignals):
        array16.receive_signal(signal)
    array16.sample(samplePoints)
    output = array16.output

    # acquire weight
    mvdrWeight = mvdr(output, expectTheta, array16.steer_vector, False)
    mcmvWeight = mcmv(output, expectTheta, coherentTheta+mcmvDoaError, array16.steer_vector, False)
    smooth2Weight = smooth2(output, expectTheta, 2, array16.steer_vector, False)
    smooth8Weight = smooth2(output, expectTheta, 8, array16.steer_vector, False)
    proposedWeight = proposed(output, expectTheta, array16.steer_vector, False)
    yangWeight = yang_ho_chi(output, 1, array16.steer_vector, expectTheta, False)

    array15 = array_maker(15)
    array9 = array_maker(9)

    # plot
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    array16.response_plot(mvdrWeight, (fig, ax), color=colors[0], linestyle=linestyles[0], label="MVDR")
    array16.response_plot(mcmvWeight, (fig, ax), color=colors[1], linestyle=linestyles[1], label="MCMV")
    array15.response_plot(smooth2Weight, (fig, ax), color=colors[2], linestyle=linestyles[2], label="spatial smooth with 2 subarrays")
    array9.response_plot(smooth8Weight, (fig, ax), color=colors[3], linestyle=linestyles[3], label="spatial smooth with 8 subarrays")
    array15.response_plot(yangWeight, (fig, ax), color=colors[4], linestyle=linestyles[4], label="method in [14]")
    array15.response_plot(proposedWeight, (fig, ax), color=colors[5], linestyle=linestyles[5], label="proposed")
    ax.legend(bbox_to_anchor=(1., 1), loc="upper left")
    ax.axvline(expectTheta, color=colors[2], linewidth=0.5, linestyle="dashed")
    ax.axvline(coherentTheta, color=colors[1], linewidth=0.5, linestyle="dashed")
    for theta in incoherentThetas:
        ax.axvline(theta, color=colors[3], linewidth=0.5, linestyle="dashed")
    ax.set_xlabel("degree(\u00b0)")

    if SAVE_FLAG:
        savefig(fig, "patternComparsion.svg")

def helper(elementNumber, expect, interferences, samplePoints, calcWeightFunc, monteNum=10, **otherParameters):

    res=[]
    for _ in range(monteNum):
        array = array_maker(elementNumber)
        array.noise_reproducible = True
        array.remove_all_signal()

        array.noise_power = 1
        array.sample(samplePoints)
        onlyNoise = array.output

        array.receive_signal(expect)
        for signal in interferences:
            array.receive_signal(signal)
        array.sample(samplePoints)
        output = array.output

        array.remove_all_signal()
        array.noise_power = 0
        array.receive_signal(expect)
        array.sample(samplePoints)
        onlyExpect = array.output

        array.remove_all_signal()
        for interference in interferences: 
            array.receive_signal(interference)
        array.sample(samplePoints)
        onlyInterference = array.output

        weight = calcWeightFunc(output=output, expect_theta=expect.theta, steer_func=array.steer_vector, **otherParameters)
        res.append(calcSINR(weight, onlyExpect, onlyInterference, onlyNoise))
    return np.sum(res) / len(res)

def SINRversusSNR():
    
    # initialization
    expectTheta = 10
    coherentTheta = 20
    incoherentThetas = [-25, -10, 0]
    snrs = np.linspace(-20, 30, 51, endpoint=True)
    cnr = 10
    inr = 10
    samplePoints = 1024
    coherentSignal = signl.LfmWave2D(coherentTheta, amplitude=decibel2value(cnr)) # change this line to use partially coherent interference
    coherentSignal = signl.CosWave2D(coherentTheta, amplitude=decibel2value(cnr), fre_shift=1e6)
    incoherentSignals = [signl.NoiseWave2D(theta, amplitude=decibel2value(inr)) for theta in incoherentThetas]

    mvdrSINRres = []
    mcmvSINRres = []
    yangSINRres = []
    smooth2SINRres = []
    smooth8SINRres = []
    proposedSINRres = []
    for snr in snrs:
        expectSignal = signl.LfmWave2D(expectTheta, amplitude=decibel2value(snr))
        mvdrSINRres.append(helper(16, expectSignal, chain((coherentSignal,), incoherentSignals), samplePoints, mvdr))
        mcmvSINRres.append(helper(16, expectSignal, chain((coherentSignal,), incoherentSignals), samplePoints, mcmv, coherent_theta= coherentTheta+mcmvDoaError))
        yangSINRres.append(helper(16, expectSignal, chain((coherentSignal,), incoherentSignals), samplePoints, yang_ho_chi, coherent_number= 1))
        smooth2SINRres.append(helper(16, expectSignal, chain((coherentSignal,), incoherentSignals), samplePoints, smooth2, subarray_num= 2))
        smooth8SINRres.append(helper(16, expectSignal, chain((coherentSignal,), incoherentSignals), samplePoints, smooth2, subarray_num= 8))
        proposedSINRres.append(helper(16, expectSignal, chain((coherentSignal,), incoherentSignals), samplePoints, proposed))
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(snrs, mvdrSINRres, label="MVDR", color=colors[0], linestyle=linestyles[0])
    ax.plot(snrs, mcmvSINRres, label="MCMV", color=colors[1], linestyle=linestyles[1])
    ax.plot(snrs, yangSINRres, label="method in [14]", color=colors[2], linestyle=linestyles[2])
    ax.plot(snrs, smooth2SINRres, label="spatial smooth with 2 subarrays", color=colors[3], linestyle=linestyles[3])
    ax.plot(snrs, smooth8SINRres, label="spatial smooth with 8 subarrays", color=colors[4], linestyle=linestyles[4])
    ax.plot(snrs, proposedSINRres, label="proposed", color=colors[5], linestyle=linestyles[5])

    ax.legend()
    ax.set_xlabel("SNR(dB)")
    ax.set_ylabel("SINR(dB)")

    if SAVE_FLAG:
        savefig(fig, "SINRversusSNR.svg")

def SINRversusSnapshots():
    # initialization
    expectTheta = 10
    coherentTheta = 20
    incoherentThetas = [-25, -10, 0]
    snr = 0
    cnr = 10
    inr = 10
    samplePointss = range(1, 100)
    expectSignal = signl.LfmWave2D(expectTheta, amplitude=decibel2value(snr))
    coherentSignal = signl.LfmWave2D(coherentTheta, amplitude=decibel2value(cnr)) # change this line to use partially coherent interference
    coherentSignal = signl.CosWave2D(coherentTheta, amplitude=decibel2value(cnr), fre_shift=1e6)
    incoherentSignals = [signl.NoiseWave2D(theta, amplitude=decibel2value(inr)) for theta in incoherentThetas]

    mvdrSINRres = []
    mcmvSINRres = []
    yangSINRres = []
    smooth2SINRres = []
    smooth8SINRres = []
    proposedSINRres = []
    for samplePoints in samplePointss:
        mvdrSINRres.append(helper(16, expectSignal, chain((coherentSignal,), incoherentSignals), samplePoints, mvdr))
        mcmvSINRres.append(helper(16, expectSignal, chain((coherentSignal,), incoherentSignals), samplePoints, mcmv, coherent_theta= coherentTheta+mcmvDoaError))
        yangSINRres.append(helper(16, expectSignal, chain((coherentSignal,), incoherentSignals), samplePoints, yang_ho_chi, coherent_number= 1))
        smooth2SINRres.append(helper(16, expectSignal, chain((coherentSignal,), incoherentSignals), samplePoints, smooth2, subarray_num= 2))
        smooth8SINRres.append(helper(16, expectSignal, chain((coherentSignal,), incoherentSignals), samplePoints, smooth2, subarray_num= 8))
        proposedSINRres.append(helper(16, expectSignal, chain((coherentSignal,), incoherentSignals), samplePoints, proposed))
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(samplePointss, mvdrSINRres, label="MVDR", color=colors[0], linestyle=linestyles[0])
    ax.plot(samplePointss, mcmvSINRres, label="MCMV", color=colors[1], linestyle=linestyles[1])
    ax.plot(samplePointss, yangSINRres, label="method in [14]", color=colors[2], linestyle=linestyles[2])
    ax.plot(samplePointss, smooth2SINRres, label="spatial smooth with 2 subarrays", color=colors[3], linestyle=linestyles[3])
    ax.plot(samplePointss, smooth8SINRres, label="spatial smooth with 8 subarrays", color=colors[4], linestyle=linestyles[4])
    ax.plot(samplePointss, proposedSINRres, label="proposed", color=colors[5], linestyle=linestyles[5])

    ax.legend()
    ax.set_xlabel("snapshots")
    ax.set_ylabel("SINR(dB)")

    if SAVE_FLAG:
        savefig(fig, "SINRversusSnapshots.svg")

def data_generator():
    """
    data_generator for data generate function. used for DNN training.
    """
    ele_num = 16
    cnr_num = 1
    inr_num = 1
    sample_points = ele_num ** 2

    ary = aray.UniformLineArray()
    for _ in range(ele_num):
        ary.add_element(aray.Element())

    def thetas():
        current = -60
        interval = .1
        while current < 60:
            if -10 <= current <= 10:
                current = 11
            yield current
            current += interval
    thetas = list(thetas())

    def check_thetas(seq):
        for m in range(len(seq)-1):
            for n in range(m+1, len(seq)):
                if abs(seq[m] - seq[n]) < 5:
                    return False
        return True

    def theta_lis(num):
        counts = 0
        while True:
            to_yield = [random.choice(thetas) for _ in range(num)]
            counts += 1
            if check_thetas(to_yield):
                yield to_yield
                counts = 0
            elif counts > 1000:
                break

    def nr_lis(num):
        # yield tuple((np.random.uniform(5, 10) for _ in range(num)))
        while True:
            yield [10 for _ in range(num)]

    expect_theta = 0
    snr = 0  
    noise_power = aray.UniformLineArray.noise_power
    decibel2val = lambda x: np.sqrt(np.power(10, x / 10) * noise_power)
    for _, coherent_theta, incoherent_theta in zip(range(10000), theta_lis(cnr_num), theta_lis(inr_num)):
        for _, coherent_nr, incoherent_nr in zip(range(1), nr_lis(cnr_num), nr_lis(inr_num)):
            signals = []
            signals.append(signl.LfmWave2D(theta=expect_theta, amplitude=decibel2val(snr)))
            real_cov = ary.noise_power * np.eye(ary.element_number, dtype=np.complex128)
            for theta, nr in zip(coherent_theta, coherent_nr):
                amplitude = decibel2val(nr)
                signals.append(signl.LfmWave2D(theta, amplitude=amplitude))
                power = amplitude ** 2
                steer_vector = ary.steer_vector(theta)
                real_cov += power * np.matmul(steer_vector, hermitian(steer_vector))
            for theta, nr in zip(incoherent_theta, incoherent_nr):
                amplitude = decibel2val(nr)
                signals.append(signl.CosWave2D(theta, amplitude=amplitude, fre_shift=6e6))
                power = amplitude ** 2
                steer_vector = ary.steer_vector(theta)
                real_cov += power * np.matmul(steer_vector, hermitian(steer_vector))
            for signal in signals:
                ary.receive_signal(signal)
            ary.sample(sample_points)
            output = ary.output
            info_dic = {
                    'coherent_theta': coherent_theta,
                    'incoherent_theta': incoherent_theta,
                    'expect_theta': expect_theta,
                    'snr': snr,
                    'cnr': nr,
                    'inr': nr
                    }
            yield output, np.linalg.pinv(real_cov), info_dic
            ary.remove_all_signal()


if __name__ == "__main__":
    oneCoherentInterference()
    oneCoherentSeveralIncoherent()
    patternComparsion()
    SINRversusSNR()
    SINRversusSnapshots()
    if not SAVE_FLAG:
        plt.show()
