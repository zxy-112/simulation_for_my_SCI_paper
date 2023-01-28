import os.path
import os
from itertools import chain
from cycler import cycler
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib import font_manager
from matplotlib.lines import Line2D

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
        self.plotShow = True

        self.noisePower = 1
        self.randomSeed = 1
        self.generator = np.random.default_rng(self.randomSeed)

        self.dt = 1e-9

        self.snr = 10
        self.expectTheta = 0
        self.lfmWidth = 10e-6
        self.lfmBandwidth = 10e6

        self.cnr = (0,)
        self.coherentTheta = (20,)

        self.inr = (30,)
        self.interTheta = (-30,)

        self.arrayNumber = 16
        self.arraySpacing = 0.5

        self.snapShots = 1024
        self.diagLoad = 0
        
        self.thetas = np.arange(-90, 90, 0.1)
        self.monteCarlo = 20

        self.smoothRange = range(4, 16)

        if os.path.exists('/System/Library/Fonts/Supplemental/Songti.ttc'):
            self.font = font_manager.FontProperties(fname='/System/Library/Fonts/Supplemental/Songti.ttc', size=10)
        elif os.path.exists('simsun.ttc'):
            self.font = font_manager.FontProperties(fname='simsun.ttc', size=10)
        else:
            self.font = font_manager.FontProperties()

        self.lineStyles = [
            "solid",
            (0, (1, 1)),  # dotted
            (0, (5, 5)),  # dashed
            (0, (10, 3)),  # long dash
            (0, (5, 1)),  # densely dashed
            (0, (3, 1, 1, 1)),  # densely dashdotted
            (0, (3, 1, 1, 1, 1, 1)),  # densely dashdotdotted
            'dashdot',
            (0, (10, 1)),
        ]
        self.colors = ['#037ef3', '#f85a40', '#00c16e', '#7552cc', '#0cb9c1', '#ffdd00', '#89ba16', '#7d3f98' ,'#52565e',]
        self.marker = [',', '.']
        self.lineWidth = 1.5
        self.markevery = 0.1
        self.fillstyle = 'none'
        self.markersize = 10
        self.useMarker = True
        self.vlineStyle = {"linestyle": (0, (5, 1)), "linewidth": 1}

        self.plotMVDR = True

        self.sqrtOf2 = np.sqrt(2)

    def responseSimulate(self):
        """
        the func that do the response simulation work.
        """
        self.resetRng()

        fig, ax = self.subplots()
        aTheta = self.getATheta(self.arrayNumber)
        output = self.getOutput()
        output = output[:, :self.snapShots]

        if self.plotMVDR:
            mvdrWeight = zxy.methods.mvdr(output, self.expectTheta, self.getATheta(self.arrayNumber), self.diagLoad)
            self.responsePlot(mvdrWeight, ax, label="MVDR")

        smoothRange = self.smoothRange
        for k in smoothRange:
            smoothWeight = zxy.methods.spatialSmooth(output, self.expectTheta, aTheta, k, self.diagLoad)
            self.responsePlot(smoothWeight, ax, label="SS"+str(k))
        
        self.setVline(ax)
        ax.legend(framealpha=0.5)
        if self.plotShow:
            plt.show(block=False)
        self.savefig(fig)
    
    def resetRng(self):
        self.generator = np.random.default_rng(self.randomSeed)

    def outputSimulate(self, weight=None, beforeName=None, afterName=None):
        self.resetRng()
        array = self.newArray()
        coherentInter = self.getCoherentSignal()
        incoherentInter = self.getInterSignal()
        expect = self.getExpectSignal()

        array.receive(*expect)
        array.receiveAll(coherentInter)
        array.receiveAll(incoherentInter)
        output = array.getOutput()
        noise = self.addNoise(np.zeros(output.shape))
        output = output + noise

        if weight is None:
            weight = zxy.methods.mvdr(output, self.expectTheta, array.aTheta, self.diagLoad)
            #weight = zxy.methods.spatialSmooth(output, self.expectTheta, array.aTheta, 12)
        
        synOutput = zxy.methods.syn(output, weight)

        array.removeAll()
        array.receive(*expect)
        onlyExpects = array.getOutput()
        onlyExpectOutput = zxy.methods.syn(array.getOutput(), weight)
        array.removeAll()
        array.receiveAll(coherentInter)
        onlyCoherentInters = array.getOutput()
        onlyCoherentInterOutput = zxy.methods.syn(array.getOutput(), weight)
        array.removeAll()
        array.receiveAll(incoherentInter)
        onlyIncoherentInters = array.getOutput()
        onlyIncoherentInterOutput = zxy.methods.syn(array.getOutput(), weight)
        onlyNoiseOutput = zxy.methods.syn(noise, weight)

        fig, ax = self.subplots()
        ax.plot(np.real(onlyExpects[0]), alpha=0.8, label='期望信号')
        if len(self.coherentTheta) != 0:
            ax.plot(np.real(onlyCoherentInters[0]), alpha=0.8, label='相干干扰信号')
        ax.plot(np.real(onlyIncoherentInters[0]), alpha=0.6, label='非相干干扰信号')
        ax.plot(np.real(noise[0]), alpha=0.6, label='噪声')
        ax.legend(prop=self.font)
        if self.plotShow:
            plt.show(block=False)
        self.savefig(fig, beforeName)

        fig, ax = self.subplots()
        ax.plot(np.real(onlyExpectOutput), alpha=0.8, label='期望信号')
        ax.plot(np.real(onlyCoherentInterOutput), alpha=0.8, label='相干干扰信号')
        ax.plot(np.real(onlyIncoherentInterOutput), alpha=0.6, label='非相干干扰信号')
        ax.plot(np.real(onlyNoiseOutput), alpha=0.6, label='噪声')
        ax.legend(prop=self.font)
        if self.plotShow:
            plt.show(block=False)

        self.savefig(fig, afterName)

    def sinrVersSnr(self):
        """
        the output sinr of different SS method and MVDR.
        """
        self.resetRng()
        oldSNR = self.snr

        aTheta = self.getATheta(self.arrayNumber)
        snrs = np.arange(-40, 50, 1)
        mvdrSINRs = []
        smoothRange = self.smoothRange

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
        if self.plotMVDR:
            ax.plot(snrs, mvdrSINRs, label="MVDR")
        for ssSINR, subarraySize in zip(ssSINRS, smoothRange):
            ax.plot(snrs, ssSINR, label = "SS"+str(subarraySize))

        ax.legend()
        if self.plotShow:
            plt.show(block=False)
        self.savefig(fig)

    def sinrVersSnapshots(self):
        self.resetRng()
        oldSnapshots = self.snapShots

        aTheta = self.getATheta(self.arrayNumber)
        snapshots = range(1, 64)
        mvdrSINRs = []
        smoothRange = self.smoothRange

        # the sinr result of different subarray size(dimension 0) at different sir(dimension 1).
        #    the sinr result of subarray size 4         the sinr result of subarray size 6
        #                    |                                         |
        # [          [-30, -29, ..., 10],                      [-29, -28, ...,9],           ...]
        ssSINRS = []
        for _ in smoothRange:
            ssSINRS.append([])

        for snapshot in snapshots:
            self.snapShots = snapshot
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

        self.snr = oldSnapshots

        fig, ax = self.subplots()
        if self.plotMVDR:
            ax.plot(snapshots, mvdrSINRs, label="MVDR")
        for ssSINR, subarraySize in zip(ssSINRS, smoothRange):
            ax.plot(snapshots, ssSINR, label = "SS"+str(subarraySize))

        ax.legend()
        if self.plotShow:
            plt.show(block=False)
        self.savefig(fig)

    def sinrVersTheta(self):
        self.resetRng()
        oldTheta = self.coherentTheta

        aTheta = self.getATheta(self.arrayNumber)
        thetas = np.arange(10, 40, 1)
        mvdrSINRs = []
        smoothRange = self.smoothRange

        # the sinr result of different subarray size(dimension 0) at different sir(dimension 1).
        #    the sinr result of subarray size 4         the sinr result of subarray size 6
        #                    |                                         |
        # [          [-30, -29, ..., 10],                      [-29, -28, ...,9],           ...]
        ssSINRS = []
        for _ in smoothRange:
            ssSINRS.append([])

        for theta in thetas:
            self.coherentTheta = (theta,)
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

        self.coherentTheta = oldTheta

        fig, ax = self.subplots()
        if self.plotMVDR:
            ax.plot(thetas, mvdrSINRs, label="MVDR")
        for ssSINR, subarraySize in zip(ssSINRS, smoothRange):
            ax.plot(thetas, ssSINR, label = "SS"+str(subarraySize))

        ax.legend()
        if self.plotShow:
            plt.show(block=False)
        self.savefig(fig)
    
    def sinrVersCnr(self):
        self.resetRng()
        oldCnr = self.cnr

        aTheta = self.getATheta(self.arrayNumber)
        cnrs = np.arange(-30, 30, 1)
        mvdrSINRs = []
        smoothRange = self.smoothRange

        # the sinr result of different subarray size(dimension 0) at different sir(dimension 1).
        #    the sinr result of subarray size 4         the sinr result of subarray size 6
        #                    |                                         |
        # [          [-30, -29, ..., 10],                      [-29, -28, ...,9],           ...]
        ssSINRS = []
        for _ in smoothRange:
            ssSINRS.append([])

        for cnr in cnrs:
            self.cnr = (cnr,)
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

        self.cnr = oldCnr

        fig, ax = self.subplots()
        if self.plotMVDR:
            ax.plot(cnrs, mvdrSINRs, label="MVDR")
        for ssSINR, subarraySize in zip(ssSINRS, smoothRange):
            ax.plot(cnrs, ssSINR, label = "SS"+str(subarraySize))

        ax.legend()
        if self.plotShow:
            plt.show(block=False)
        self.savefig(fig)


    def sinrVersDiagLoad(self):
        self.resetRng()
        oldDiagLoad = self.diagLoad

        aTheta = self.getATheta(self.arrayNumber)
        diagLoads = np.arange(0, 100, 1)
        mvdrSINRs = []
        smoothRange = self.smoothRange

        # the sinr result of different subarray size(dimension 0) at different sir(dimension 1)
        #    the sinr result of subarray size 4         the sinr result of subarray size 6
        #                    |                                         |
        # [          [-30, -29, ..., 10],                      [-29, -28, ...,9],           ...]
        ssSINRS = []
        for _ in smoothRange:
            ssSINRS.append([])

        for diagLoad in diagLoads:
            self.diagLoad = zxy.methods.snr2value(diagLoad, self.noisePower) ** 2
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

        self.diagLoad = oldDiagLoad

        fig, ax = self.subplots()
        if self.plotMVDR:
            ax.plot(diagLoads, mvdrSINRs, label="MVDR")
        for ssSINR, subarraySize in zip(ssSINRS, smoothRange):
            ax.plot(diagLoads, ssSINR, label = "SS"+str(subarraySize))

        ax.legend()
        if self.plotShow:
            plt.show(block=False)
        self.savefig(fig)
    
    def savefig(self, fig, name=None):
        """
        save the matplotlib figure if the saveFlag is True.
        """
        if self.saveFig:
            if name is not None:
                fig.savefig(os.path.join(self.savePath, name + self.figSuffix), dpi = self.dpi)
            else:
                fig.savefig(os.path.join(self.savePath, type(self).__name__+\
                    str(self.saveName)+self.figSuffix), dpi = self.dpi)
                self.saveName += 1
    
    def responsePlot(self, weight: np.ndarray, ax: Axes, minValue: float = -60, **plt_kwargs):
        """
        given the weight, plot the response.
        weight -- the given weight.
        """
        aTheta = self.getATheta(len(weight))
        response = zxy.methods.response(weight, self.thetas, aTheta, minValue=minValue)
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
        if self.useMarker:
            temp = cycler(marker=self.marker) * temp
            temp = temp * cycler(fillstyle=[self.fillstyle])
            temp = temp * cycler(markersize=[self.markersize])
        ax.set(prop_cycle=temp)
    
    def subplots(self, rowNum=1, colNum=1, **subplots_kwargs):
        """
        return fig, ax pair that has the prop cycle corresponding to
        the setting.
        """
        fig, axs = plt.subplots(rowNum, colNum, layout='constrained', **subplots_kwargs)
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
            ax.legend(list(chain(vlines, lines)), list(chain(vlineLabels, labels)), prop=self.font)
    
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
        for index, (inr, theta) in enumerate(zip(self.inr, self.interTheta)):
            inter = zxy.signal.noise(self.lfmWidth, self.dt, np.random.default_rng(seed=self.randomSeed + index))
            inter = inter * zxy.methods.snr2value(inr, self.noisePower)
            res.append((inter, theta))
        return res
    
    def getAllInters(self):
        """
        return all interferences which is a list of
        (signal: numpy.ndarray, theta: float).
        """
        coherent = self.getCoherentSignal()
        inter = self.getInterSignal()
        res = list(chain(coherent, inter))
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
        expectGain = np.abs(expectGain)
        temp = zxy.methods.snr2value(self.snr, self.noisePower)
        S = temp * temp * (expectGain * expectGain)

        I = 0
        for inr, theta in chain(zip(self.cnr, self.coherentTheta), zip(self.inr, self.interTheta)):
            interGain = (zxy.methods.hermitian(weight) @ aTheta(theta)).item()
            interGain = np.abs(interGain)
            temp = zxy.methods.snr2value(inr, self.noisePower) ** 2
            temp = temp * (interGain * interGain)
            I += temp

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

class TestDuvallBasedFast(SimulatorBase):

    def __init__(self):
        super().__init__()
        self.snr = 10
        self.cnr = (30,)
        self.coherentTheta = (25,)
        self.interTheta = (-40,)
        self.inr = (30,)
        self.snapShots = 100

    def responseSimulate(self):
        self.resetRng()

        output = self.getOutput()
        usedOutput = output[:, :self.snapShots]

        weight = zxy.methods.duvallBasedFast(usedOutput, self.expectTheta, self.getATheta(self.arrayNumber), len(self.inr)+len(self.cnr))
        weight = zxy.methods.improvedAdaptiveNulling(usedOutput, self.expectTheta, self.getATheta(self.arrayNumber))
        fig, ax = self.subplots()
        self.responsePlot(weight, ax)
        self.setVline(ax)
        if self.plotShow:
            plt.show()
        self.savefig(fig)
    
    def sinrVersSnapshots(self):
        self.resetRng()

        oldSnapshots = self.snapShots
        snapshotss = np.arange(1, 10)
        sinr = []
        aTheta = self.getATheta(self.arrayNumber)
        for _ in range(self.monteCarlo):
            sinr.append([])
            output = self.getOutput()
            for snapshots in snapshotss:
                usedOutput = output[:, :snapshots]
                weight = zxy.methods.duvallBasedFast(usedOutput, self.expectTheta, aTheta, len(self.cnr)+len(self.inr))
                sinr[-1].append(self.getSINR(weight))
        
        sinr = np.array(sinr)
        sinr = np.sum(sinr, axis=0, keepdims=False) / self.monteCarlo

        fig, ax = self.subplots()
        ax.plot(snapshotss, sinr)
        if self.plotShow:
            plt.show()
        self.savefig(fig)
        self.snapShots = oldSnapshots

if __name__ == "__main__":
    sb = SimulatorBase()
    sb.responseSimulate()
    sb.sinrVersSnr()
    sb.sinrVersCnr()
    plt.show()
    #sb.outputSimulate()
    #sb.sinrVersSnapshots()
    #sb.sinrVersTheta()
    #sb.sinrVersDiagLoad()
    #sb.sinrVersCnr()
