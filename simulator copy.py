import os.path
import os
from itertools import chain
from cycler import cycler
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib import font_manager
from matplotlib.lines import Line2D
import matplotlib.animation as animation
import matplotlib.patches as patches

import zxy.signal
import zxy.array
import zxy.methods

class SimulatorBase:
    """
    the base simulation class. there is one expect signal, one coherent interference,
    one incoherent interference. The simulation work is done in method sinrSimulate() and
    responseSimulate().
    the snr is 0dB, and the expect theta is 0
    there is one coherent interference, and the cnr is (-10,) dB, the theta is (20,)
    there is one incoherent interference, and the inr is (30,) dB, the theta is (-30,)
    """

    def __init__(self):

        self.saveFig = True
        self.dpi = 600
        self.figSuffix = ".svg"
        self.savePath = os.path.join(os.path.expanduser("~"), "coherentSimulate")
        if (not os.path.exists(self.savePath)):
            os.mkdir(self.savePath)
        self.saveName = 0

        self.noisePower = 1
        self.randomSeed = 1
        self.generator = np.random.default_rng(self.randomSeed)

        self.dt = 1e-9

        self.snr = 0
        self.expectTheta = 0
        self.lfmWidth = 10e-6
        self.lfmBandwidth = 10e6

        self.cnr = (-10,)
        self.coherentTheta = (20,)

        self.inr = (30,)
        self.interTheta = (-30,)

        self.arrayNumber = 16
        self.arraySpacing = 0.5

        self.snapShots = 1024
        self.diagLoad = 0
        
        self.thetas = np.arange(-90, 90, 0.1)
        self.monteCarlo = 20

        if os.path.exists('/System/Library/Fonts/Supplemental/Songti.ttc'):
            self.font = font_manager.FontProperties(fname='/System/Library/Fonts/Supplemental/Songti.ttc')
        else:
            self.font = font_manager.FontProperties()

        self.lineStyles = [
            "solid",
            (0, (1, 1)),
            (0, (5, 5)),
            (0, (5, 1)),
            (0, (3, 1, 1, 1)),
            (0, (3, 1, 1, 1, 1, 1)),
            (0, (5, 2)),
            (0, (5, 2, 2, 2))
        ]
        self.colors = ['#037ef3', '#f85a40', '#00c16e', '#7552cc', '#0cb9c1', '#52565e', '#ffc845', '#a51890']
        self.lineWidth = 1.5
        self.vlineStyle = {"linestyle": (0, (5, 1)), "linewidth": 0.7}
        self.markevery = 0.1

        self.sqrtOf2 = np.sqrt(2)

    def responseSimulate(self):
        """
        the func that do the response simulation work.
        """
        fig, ax = self.subplots()
        aTheta = self.getATheta(self.arrayNumber)
        output = self.getOutput()
        output = output[:, :self.snapShots]

        mvdrWeight = zxy.methods.mvdr(output, self.expectTheta, self.getATheta(self.arrayNumber), self.diagLoad)
        self.responsePlot(mvdrWeight, ax, label="MVDR")

        smoothRange = range(4, 15, 2)
        for k in smoothRange:
            smoothWeight = zxy.methods.spatialSmooth(output, self.expectTheta, aTheta, k, self.diagLoad)
            self.responsePlot(smoothWeight, ax, label="SS"+str(k))
        
        self.setVline(ax)
        ax.legend(framealpha=0.5)
        plt.show()
        self.savefig(fig)
    
    def outputSimulate(self):
        pass

    def sinrSimulate(self):
        """
        the output sinr of different SS method and MVDR.
        """
        oldSNR = self.snr

        aTheta = self.getATheta(self.arrayNumber)
        snrs = np.arange(-40, 30, 1)
        mvdrSINRs = []
        smoothRange = range(4, 15, 2)

        # the sinr result of different subarray size(dimension 0) at different sir(dimension 1).
        #    the sinr result of subarray size 4         the sinr result of subarray size 6
        #                    |                                         |
        # [          [-30, -29, ..., 10],                      [-29, -28, ...,9],           ...]
        ssSINRS = []
        for _ in smoothRange:
            ssSINRS.append([])

        for snr in snrs:
            self.snr = snr
            mvdrSamples = []

            # the sinr result of different subarray size(dimension 0) 
            # in different experiment(dimension 1). 
            ssSamples = []
            for _ in smoothRange:
                ssSamples.append([])
            for _ in range(self.monteCarlo):
                output = self.getOutput()
                output = output[:, :self.snapShots]
                mvdrWeight = zxy.methods.mvdr(output, self.expectTheta, aTheta, self.diagLoad)
                ssWeights = []
                for subarraySize in smoothRange:
                    ssWeights.append(zxy.methods.spatialSmooth(output, \
                        self.expectTheta, aTheta, subarraySize, self.diagLoad))
                mvdrSamples.append(self.getSINR(mvdrWeight))
                for ssSample, ssWeight in zip(ssSamples, ssWeights):
                    ssSample.append(self.getSINR(ssWeight))
            mvdrSINRs.append(sum(mvdrSamples) / len(mvdrSamples))

            for ssSINR, ssSample in zip(ssSINRS, ssSamples):
                ssSINR.append(sum(ssSample) / len(ssSample))

        self.snr = oldSNR

        fig, ax = self.subplots()
        ax.plot(snrs, mvdrSINRs, label="MVDR")
        for ssSINR, subarraySize in zip(ssSINRS, smoothRange):
            ax.plot(snrs, ssSINR, label = "SS"+str(subarraySize))

        ax.legend()
        plt.show()
        self.savefig(fig)
    
    def savefig(self, fig):
        """
        save the matplotlib figure if the saveFlag is True.
        """
        if self.saveFig:
            fig.savefig(os.path.join(self.savePath, type(self).__name__+\
                str(self.saveName)+self.figSuffix), dpi = self.dpi)
        self.saveName += 1
    
    def responsePlot(self, weight: np.ndarray, ax: Axes, **plt_kwargs):
        """
        given the weight, plot the response.
        weight -- the given weight.
        """
        aTheta = self.getATheta(len(weight))
        response = zxy.methods.response(weight, self.thetas, aTheta)
        line, = ax.plot(self.thetas, response, **plt_kwargs)
        return line
    
    def getOutput(self):
        """
        return the output of the array, the noise is added.
        """
        array = self.getArray()
        output = array.getOutput()
        output = self.addNoise(output)
        return output
    
    def setCycler(self, ax: Axes):
        """
        set the cycler of the ax.
        """
        temp = cycler(color=self.colors) + cycler(linestyle=self.lineStyles)
        temp = temp * cycler(linewidth=[self.lineWidth]) * cycler(markevery=[self.markevery,])
        ax.set(prop_cycle=temp)
    
    def subplots(self, rowNum=1, colNum=1, **subplots_kwargs):
        """
        return fig, ax pair that has the prop cycle corresponding to
        the setting.
        """
        fig, axs = plt.subplots(rowNum, colNum, **subplots_kwargs)
        if (rowNum == 1 and colNum == 1):
            self.setCycler(axs)
        else:
            for ax in axs.flatten():
                self.setCycler(ax)
        return fig, axs
    
    def setVline(self, ax: Axes, legend=False):
        """
        helpful method to add lines at expect theta and inter theta when
        plotting response.
        """
        lines = ax.get_lines()
        labels = []
        for line in lines:
            labels.append(line.get_label())

        ax.axvline(self.expectTheta, color=self.colors[2], **self.vlineStyle)
        for theta in self.coherentTheta:
            ax.axvline(theta, color=self.colors[1], **self.vlineStyle)
        for theta in self.interTheta:
            ax.axvline(theta, color=self.colors[3], **self.vlineStyle)

        expectLine = Line2D([], [], color=self.colors[2], **self.vlineStyle)
        coherentLine = Line2D([], [], color=self.colors[1], **self.vlineStyle)
        interLine = Line2D([], [], color=self.colors[3], **self.vlineStyle)
        if len(self.coherentTheta) == 0:
            vlines = [expectLine,  interLine]
            vlineLabels = ["期望信号", "非相干干扰"]
        else:
            vlines = [expectLine, coherentLine, interLine]
            vlineLabels = ["期望信号", "相干干扰", "非相干干扰"]

        if legend == True:
            ax.legend(chain(vlines, lines), chain(vlineLabels, labels), prop=self.font)
    
    def fftPlot(self, signal: np.ndarray, ax: Axes, **plt_kwargs):
        fftRes = np.fft.fft(signal)
        fftRes = np.fft.fftshift(fftRes)
        fre = np.fft.fftfreq(len(fftRes), self.dt)
        fre = np.fft.fftshift(fre)
        lines = ax.plot(fre, np.abs(fftRes), **plt_kwargs)
        ax.set_xlim((-self.lfmBandwidth, self.lfmBandwidth))
        return lines

    def getArray(self):
        """
        return the LineArray which has received the expect 
        signal and interferences. the output of the array haven't
        been added noise.
        """
        expectSignal, expectTheta = self.getExpectSignal()
        coherentSignals = self.getCoherentSignal()
        interSignals = self.getInterSignal()

        array = self.newArray()
        array.receive(expectSignal, expectTheta)
        for signal, theta in coherentSignals:
            array.receive(signal, theta)
        for signal, theta in interSignals:
            array.receive(signal, theta)

        return array


    def getExpectSignal(self):
        """
        return the expect signal and its theta
        """
        expectSignal = zxy.signal.lfmPulse(self.lfmWidth, self.lfmBandwidth, self.dt)
        expectSignal = expectSignal * zxy.methods.snr2value(self.snr, self.noisePower)
        return expectSignal, self.expectTheta
    
    def getCoherentSignal(self):
        """
        return the coherent signal and its theta  which is a list of 
        (signal: numpy.ndarray, theta: float).
        """
        res = []
        for cnr, theta in zip(self.cnr, self.coherentTheta):
            coherentInter = zxy.signal.lfmPulse(self.lfmWidth, self.lfmBandwidth, self.dt)
            coherentInter = coherentInter * zxy.methods.snr2value(cnr, self.noisePower)
            res.append((coherentInter, theta))
        return res
    
    def getInterSignal(self):
        """
        return the interference which is incoherent
         and its theta  which is a list of 
        (signal: numpy.ndarray, theta: float).
        """
        res = []
        for inr, theta in zip(self.inr, self.interTheta):
            inter = zxy.signal.noise(self.lfmWidth, self.dt, self.generator)
            inter = inter * zxy.methods.snr2value(inr, self.noisePower)
            res.append((inter, theta))
        return res

    def addNoise(self, array: np.ndarray):
        """
        add noise to array and return.
        """
        noise = self.generator.standard_normal(array.shape) * self.sqrtOf2 / 2 \
            + 1j * self.generator.standard_normal(array.shape) * self.sqrtOf2 / 2
        noise = noise * self.noisePower
        return array + noise
    
    def getSINR(self, weight: np.ndarray):
        """
        given a specified weight, calculate the output SINR with the 
        specified weight.
        """
        aTheta = self.getATheta(len(weight))

        expectGain = (zxy.methods.hermitian(weight) @ aTheta(self.expectTheta)).item()
        expectGain = abs(expectGain)
        S = zxy.methods.snr2value(self.snr, self.noisePower) * (expectGain * expectGain)

        I = 0
        for inr, theta in chain(zip(self.cnr, self.coherentTheta), zip(self.inr, self.interTheta)):
            interGain = (zxy.methods.hermitian(weight) @ aTheta(theta)).item()
            interGain = abs(interGain)
            I += zxy.methods.snr2value(inr, self.noisePower) * (interGain * interGain)

        squeezedWeight = np.squeeze(weight)
        N = np.sum(np.power(np.abs(squeezedWeight), 2)) * self.noisePower

        return 10 * np.log10(S / (I + N))

    def newArray(self):
        """
        return an array with the array number of self.arrayNumber and 
        spacing of self.arraySpacing.
        """
        return zxy.array.LineArray.uniformArrray(self.arrayNumber, self.arraySpacing)
    
    def getATheta(self, length):
        """
        return the a(theta) function, but with the specified length.
        that is, when a method needs a part of elements to do the 
        beamfrom, this method is useful.
        """
        array = self.newArray()
        res = lambda theta: array.aTheta(theta)[:length, :]
        return res
    
class SimulatorBaseCoherentDegree30(SimulatorBase):
    """
    change the coherent interference theta to (30,)
    """
    def __init__(self):
        super().__init__()
        self.coherentTheta = (30,)

class SimulatorBaseCoherentINR30(SimulatorBase):
    """
    change the cnr to (30,)
    """
    def __init__(self):
        super().__init__()
        self.cnr = (30,)

class SimulatorBaseNoCoherent(SimulatorBase):
    def __init__(self):
        super().__init__()
        self.cnr = ()
        self.coherentTheta = ()

class SimulatorPPT(SimulatorBase):
    def __init__(self):
        super().__init__()
        self.cnr = ()
        self.coherentTheta = ()
        self.inr = (30, 30)
        self.interTheta = (-30, 20)

    def responseSimulate(self):
        output = self.getOutput()
        aTheta = self.getATheta(self.arrayNumber)
        mvdrWeight = zxy.methods.mvdr(output, self.expectTheta, aTheta)
        fig, ax = self.subplots()
        self.responsePlot(mvdrWeight, ax, label="MVDR")
        self.setVline(ax)
        ax.grid()
        self.savefig(fig)
    
    def responseGif(self):

        fig, ax = self.subplots(subplot_kw=dict(projection='polar'))
        plot_thetas = np.arange(-90, 90, 0.1)
        ax.set_thetalim((-np.pi/2, np.pi/2))
        ax.set_theta_zero_location("N")
        ax.set_rlim((-40, 0))
        ax.grid(True)
        ax.set_rticks([])
        ax.set_rgrids((-40, -30, -20, -10, 0))
        aTheta = self.getATheta(self.arrayNumber)
        thetas = np.arange(-60, 60, 0.5)
        line1, = ax.plot([], [], color = '#037ef3')
        # line2, = ax.plot([], [], color = '#f85a40', linestyle='dashed')
        # line3, = ax.plot([], [], color = '#f85a40', linestyle='dashed')
        # line4, = ax.plot([], [], color = 'black', linestyle='-', linewidth=1)
        line1.set_xdata(np.deg2rad(plot_thetas))
        # arrow = patches.FancyArrowPatch((0, 0), (0, 0), arrowstyle='<->', mutation_scale=10)
        # arrow.set_animated(True)
        # ax.add_artist(arrow)
        # text = ax.text(0, 0, "波束宽度", fontproperties=self.font, horizontalalignment='center', verticalalignment='bottom')
        def frames(nums):
            direction = True
            size = thetas.size
            count = 0
            index = size // 2
            while count < nums:
                yield thetas[index] 
                if index == 0 or index == size - 1:
                    direction = not direction
                if direction:
                    index += 1
                else:
                    index -= 1
                count += 1

        def run(theta):
            weight = aTheta(theta)
            response = zxy.methods.response(weight, plot_thetas, aTheta)
            index = np.searchsorted(plot_thetas, theta)
            mainLobe = findZeros(response, index)
            mainLobeThetas = plot_thetas[mainLobe[0]], plot_thetas[mainLobe[1]]
            wordTheta = findWordTheta(mainLobeThetas)

            line1.set_ydata(response)
            # line2.set_ydata([-40, 0])
            # line3.set_ydata([-40, 0])
            # line2.set_xdata(np.deg2rad([mainLobeThetas[0], mainLobeThetas[0]]))
            # line3.set_xdata(np.deg2rad([mainLobeThetas[1], mainLobeThetas[1]]))
            # line4.set_xdata(np.deg2rad([theta, theta]))
            # line4.set_ydata([-40, 0])
            # text.set_position((np.deg2rad(wordTheta), 0))
            # arrow.set_positions((np.deg2rad(mainLobeThetas[0]), -5), (np.deg2rad(mainLobeThetas[1]), -5))

        def findWordTheta(thetas, ):
            newThetas = [theta + 90 for theta in thetas]
            minTheta = min(newThetas)
            distance = newThetas[1] - newThetas[0]
            distance = abs(distance)
            if distance > 90:
                wordTheta = sum(thetas) / 2 + 90
            else:
                wordTheta = minTheta - 90 + distance / 2
            return wordTheta

        def findZeros(array, beginIndex, minValue=-40):
            """
            given a begin index, find the nearest two nulls in the array.
            """
            res = []
            res.append(findZerosHelper(array, beginIndex, minValue, True))
            res.append(findZerosHelper(array, beginIndex, minValue, False))
            return res
        
        def findZerosHelper(array, beginIndex, minValue, forward=True):
            size = array.size
            current = beginIndex
            count = 0
            while array[current] > minValue:
                if count > size:
                    raise Exception()
                if forward:
                    current += 1
                else:
                    current -= 1
                current = (current + size) % size
            
            return current

        ani = animation.FuncAnimation(fig, run, frames(1000), save_count=200, interval=20, cache_frame_data=True)
        ani.save("new.gif", dpi=100)

def noiseInterSimulate():
    def noise(length):
        sqrtOf2 = np.sqrt(2)
        generator = np.random.default_rng()
        return generator.standard_normal(length) * sqrtOf2 / 2 \
            + 1j * generator.standard_normal(length) * sqrtOf2 / 2
    
    snr = 0
    expect = zxy.signal.lfmPulse(10e-6, 10e6, 10e-9)
    reference = expect
    expect = zxy.signal.replicate(expect, 1, 0.5)
    expect = zxy.signal.rightShift(expect, 1000)
    expect = expect * zxy.methods.snr2value(snr, 1)

    fig, ax = plt.subplots()
    line2, = ax.plot([], [], color='#f85a40', label="噪声压制干扰", linewidth=1)
    line1, = ax.plot([], [], color='#037ef3', label="期望信号")
    ax.legend(prop=font_manager.FontProperties(fname='/System/Library/Fonts/Supplemental/Songti.ttc'), loc='upper right')
    line1.set_xdata(np.arange(0, expect.size, 1))
    line2.set_xdata(np.arange(0, expect.size, 0.5))
    ax.set_xlim([0, 2500])
    ax.set_ylim([0, 1200])
    inter = noise(expect.size) 

    def run(inr):
        nonlocal inter
        sPlusN = noise(expect.size) + expect
        # newinter = noise(expect.size) * zxy.methods.snr2value(inr, 1)
        res1 = pulseCompress(sPlusN, reference)
        # res2 = pulseCompress(newinter, reference)
        res2 = noise(expect.size*2) * inr
        line1.set_ydata(np.abs(res1))
        line2.set_ydata(np.abs(res2))

        pass

    def frame():
        maxInr = 50
        inrs = np.arange(0, maxInr, 5)
        inrs = np.concatenate((inrs, np.ones(50)*maxInr))
        inrs = np.ones(200) * maxInr
        for inr in inrs:
            yield inr
    
    def pulseCompress(signal, referenceSignal):
        signal0 = np.concatenate((signal, np.zeros(referenceSignal.size-1)))
        referenceSignal0 = np.concatenate((referenceSignal, np.zeros(signal.size-1)))
        fft = np.fft.fft
        res = np.fft.ifft(fft(signal0) * np.conjugate(fft(referenceSignal0)))
        return res[:signal.size]
    
    ani = animation.FuncAnimation(fig, run, frame, interval=20, save_count=200)
    ani.save("new6.gif", dpi=300)

class SimulatorPPT2(SimulatorBase):

    def responseGif(self):

        fig, ax = self.subplots(subplot_kw=dict(projection='polar'))
        ax.set_thetalim((-np.pi/2, np.pi/2))
        ax.set_rlim((-60, 0))
        ax.set_theta_zero_location("N")
        thetas = np.arange(-90, 90, 0.1)
        line1, = ax.plot([], [])
        aTheta = self.getATheta(self.arrayNumber)
        def frames():
            thetas = [23.6, -49.8, -20.2, 53.5, 12.2]
            for theta in thetas:
                yield theta

        def run(theta):

            weight = aTheta(theta)
            response = zxy.methods.response(weight, thetas, aTheta)
            line1.set_data(np.deg2rad(thetas), response)
            pass

        ani = animation.FuncAnimation(fig, run, frames(), interval=1000, save_count=1000)
        ani.save("new2.gif", dpi=600)
            
if __name__ == "__main__":
    sb = SimulatorBase()
    # sb.responseSimulate()
    # sb.sinrSimulate()

    # sb2 = SimulatorPPT()
    # sb2.responseGif()
    noiseInterSimulate()
