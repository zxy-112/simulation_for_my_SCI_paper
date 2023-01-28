import matplotlib.pyplot as plt
import numpy as np
from simulator import SimulatorBase
import zxy.methods

class SimulatorA(SimulatorBase):
    def __init__(self):
        super().__init__()
        self.snr = 10
        self.cnr = ()
        self.coherentTheta = ()
        self.inr = (30, 40)
        self.interTheta = (-30, 20)
    
    def simulate(self):
        output = self.getOutput()
        print(type(self).__name__, ' snapshots: ', len(output[0]))
        mvdrWeight = zxy.methods.mvdr(output, self.expectTheta, self.getATheta(len(output)), self.diagLoad)

        self.outputSimulate(weight=mvdrWeight)

        output = output[:, :self.snapShots]
        mvdrWeight = zxy.methods.mvdr(output, self.expectTheta, self.getATheta(len(output)), self.diagLoad)
        fig, ax = self.subplots()
        ax.set_xlabel('角度(°)', font=self.font)
        ax.grid(True)
        ax.margins(0)
        self.responsePlot(mvdrWeight, ax)
        self.setVline(ax)
        if self.saveFig:
            self.savefig(fig)

class SimulatorB(SimulatorA):
    def __init__(self):
        super().__init__()
        self.cnr = (10,)
        self.coherentTheta = (20,)
        self.inr = (30,)
        self.interTheta = (-30,)

class SimulatorC(SimulatorBase):

    def __init__(self):
        super().__init__()
        self.snr = 10
        self.cnr = (10,)
        self.coherentTheta = (20,)
        self.interTheta = (-30,)
        self.inr = (30,)
        self.virtualPow = 10e10
        self.widenTheta = 4
        self.doaError = 0.5
    
    def responseSimulate(self):
        self.resetRng()

        output = self.getOutput()
        output = output[:self.snapShots]
        aTheta = self.getATheta(self.arrayNumber)

        coherentDoa = tuple((item + self.doaError for item in self.coherentTheta))
        mvdrWeight = zxy.methods.mvdr(output, self.expectTheta, aTheta, self.diagLoad)
        mvdrVirtualWeight = zxy.methods.mvdrVirtualInter(output, self.expectTheta, \
            coherentDoa, tuple([self.virtualPow] * len(self.coherentTheta)), aTheta)
        mvdrVirturalWidenWeight = zxy.methods.mvdrVirtualInter(output, self.expectTheta,\
            coherentDoa, tuple([self.virtualPow] * len(self.coherentTheta)), aTheta, widenTheta=self.widenTheta, diagLoad=self.diagLoad)
        mcmvWeight = zxy.methods.mcmv(output, self.expectTheta, coherentDoa, aTheta, self.diagLoad)
        ssWeight = zxy.methods.spatialSmooth(output, self.expectTheta, aTheta, 8, self.diagLoad)
        ctmvWeight = zxy.methods.ctmv(output, self.expectTheta, coherentDoa, aTheta, self.noisePower, self.diagLoad)
        
        fig, ax = self.subplots()
        self.responsePlot(mvdrWeight, ax, label='MVDR')
        #self.responsePlot(ssWeight, ax, label='Spatial Smooth')
        self.responsePlot(mvdrVirtualWeight, ax, label='虚拟干扰加载')
        #self.responsePlot(mcmvWeight, ax, label='MCMV')
        #self.responsePlot(ctmvWeight, ax, label='CTMV')
        self.setVline(ax, legend=False)
        ax.legend(prop=self.font)
        ax.grid()
        ax.margins(0)
        ax.set_xlabel('角度(°)', font=self.font)
        if self.plotShow:
            plt.show(block=False)
        if self.saveFig:
            self.savefig(fig)

        self.outputSimulate(mvdrWeight, afterName='mvdrOutput')
        self.outputSimulate(ssWeight, afterName='ssOutput')
        self.outputSimulate(mvdrVirturalWidenWeight, afterName='proposedOutput')
        self.outputSimulate(mcmvWeight, afterName='mcmvOutput')
        self.outputSimulate(ctmvWeight, afterName='ctmvOutput')
    
    def sinrVersSnr(self):
        self.resetRng()
        oldSNR = self.snr

        aTheta = self.getATheta(self.arrayNumber)
        snrs = np.arange(-20, 20, 1)
        mvdrSINRs = []
        ssSINRS = []
        mvdrVirtualWidenSINRs = []
        mcmvSINRs = []
        ctmvSINRs = []
        duvallSINRs = []
        duvallFastSINRs = []

        for snr in snrs:
            self.snr = snr
            mvdrSamples = []
            ssSamples = []
            mvdrVirtualWidenSamples = []
            mcmvSamples = []
            ctmvSamples = []
            duvallFastSamples = []
            duvallSamples = []

            # the sinr result of different subarray size(dimension 0) 
            # in different experiment(dimension 1). 
            for _ in range(self.monteCarlo):
                output = self.getOutput()
                output = output[:, :self.snapShots]
                mvdrWeight = zxy.methods.mvdr(output, self.expectTheta, aTheta, self.diagLoad)
                ssWeight = zxy.methods.spatialSmooth(output, self.expectTheta, aTheta, 8, self.diagLoad)

                coherentDoa = tuple((item + self.doaError for item in self.coherentTheta))
                mvdrVirtualWidenWeight = zxy.methods.mvdrVirtualInter(output, self.expectTheta, coherentDoa, \
                    tuple([self.virtualPow]*len(self.coherentTheta)), aTheta, self.widenTheta, self.diagLoad)
                mcmvWeight = zxy.methods.mcmv(output, self.expectTheta, coherentDoa, aTheta, self.diagLoad)
                ctmvWeight = zxy.methods.ctmv(output, self.expectTheta, coherentDoa, aTheta, self.noisePower, self.diagLoad)
                duvallFastWeight = zxy.methods.duvallBasedFast(output, self.expectTheta, aTheta, len(self.cnr))
                duvallWeight = zxy.methods.duvall(output, self.expectTheta, aTheta, self.arraySpacing)
                
                mvdrSamples.append(self.getSINR(mvdrWeight))
                ssSamples.append(self.getSINR(ssWeight))
                mvdrVirtualWidenSamples.append(self.getSINR(mvdrVirtualWidenWeight))
                mcmvSamples.append(self.getSINR(mcmvWeight))
                ctmvSamples.append(self.getSINR(ctmvWeight))
                duvallFastSamples.append(self.getSINR(duvallFastWeight))
                duvallSamples.append(self.getSINR(duvallWeight))

            mvdrSINRs.append(sum(mvdrSamples) / len(mvdrSamples))
            ssSINRS.append(sum(ssSamples) / len(ssSamples))
            mvdrVirtualWidenSINRs.append(sum(mvdrVirtualWidenSamples) / len(mvdrVirtualWidenSamples))
            mcmvSINRs.append(sum(mcmvSamples) / len(mcmvSamples))
            ctmvSINRs.append(sum(ctmvSamples) / len(ctmvSamples))
            duvallFastSINRs.append(sum(duvallFastSamples) / len(duvallFastSamples))
            duvallSINRs.append(sum(duvallSamples) / len(duvallSamples))

        self.snr = oldSNR

        fig, ax = self.subplots()
        ax.plot(snrs, mvdrSINRs, label="MVDR")
        ax.plot(snrs, ssSINRS, label='空间平滑')
        ax.plot(snrs, mvdrVirtualWidenSINRs, label='所提方法')
        ax.plot(snrs, mcmvSINRs, label='MCMV')
        ax.plot(snrs, ctmvSINRs, label='CTMV')
        ax.plot(snrs, duvallFastSINRs, label = "Duvall Based")
        ax.plot(snrs, duvallSINRs, label = "Duvall")

        ax.legend(prop=self.font)
        ax.set_xlabel('SNR(dB)')
        ax.set_ylabel('SINR(dB)')
        ax.grid(True)
        self.plotShow = True
        if self.plotShow:
            plt.show()
        self.savefig(fig)
    
    def sinrVersDoaError(self):
        self.resetRng()
        oldError = self.doaError

        aTheta = self.getATheta(self.arrayNumber)
        errors = np.arange(-1, 1, 0.01)
        mvdrSINRs = []
        ssSINRS = []
        mvdrVirtualWidenSINRs = []
        mcmvSINRs = []
        ctmvSINRs = []

        for doaError in errors:
            self.doaError = doaError
            mvdrSamples = []
            ssSamples = []
            mvdrVirtualWidenSamples = []
            mcmvSamples = []
            ctmvSamples = []

            # the sinr result of different subarray size(dimension 0) 
            # in different experiment(dimension 1). 
            for _ in range(self.monteCarlo):
                output = self.getOutput()
                output = output[:, :self.snapShots]
                mvdrWeight = zxy.methods.mvdr(output, self.expectTheta, aTheta, self.diagLoad)
                ssWeight = zxy.methods.spatialSmooth(output, self.expectTheta, aTheta, 8, self.diagLoad)

                coherentDoa = tuple((item + self.doaError for item in self.coherentTheta))
                mvdrVirtualWidenWeight = zxy.methods.mvdrVirtualInter(output, self.expectTheta, coherentDoa, \
                    tuple([self.virtualPow]*len(self.coherentTheta)), aTheta, self.widenTheta, self.diagLoad)
                mcmvWeight = zxy.methods.mcmv(output, self.expectTheta, coherentDoa, aTheta, self.diagLoad)
                ctmvWeight = zxy.methods.ctmv(output, self.expectTheta, coherentDoa, aTheta, self.noisePower, self.diagLoad)
                
                mvdrSamples.append(self.getSINR(mvdrWeight))
                ssSamples.append(self.getSINR(ssWeight))
                mvdrVirtualWidenSamples.append(self.getSINR(mvdrVirtualWidenWeight))
                mcmvSamples.append(self.getSINR(mcmvWeight))
                ctmvSamples.append(self.getSINR(ctmvWeight))

            mvdrSINRs.append(sum(mvdrSamples) / len(mvdrSamples))
            ssSINRS.append(sum(ssSamples) / len(ssSamples))
            mvdrVirtualWidenSINRs.append(sum(mvdrVirtualWidenSamples) / len(mvdrVirtualWidenSamples))
            mcmvSINRs.append(sum(mcmvSamples) / len(mcmvSamples))
            ctmvSINRs.append(sum(ctmvSamples) / len(ctmvSamples))

        self.doaError = oldError

        fig, ax = self.subplots()
        ax.plot(errors, mvdrSINRs, label="MVDR")
        ax.plot(errors, ssSINRS, label='Spatial Smooth')
        ax.plot(errors, mvdrVirtualWidenSINRs, label='所提方法')
        ax.plot(errors, mcmvSINRs, label='MCMV')
        ax.plot(errors, ctmvSINRs, label='CTMV')

        ax.legend(prop=self.font)
        ax.set_xlabel('DOA误差(°)', font=self.font)
        ax.set_ylabel('SINR(dB)')
        ax.grid(True)
        if self.plotShow:
            plt.show(block=False)
        self.savefig(fig)

if __name__ == '__main__':
    sb = SimulatorC()
    sb.plotShow = True
    sb.sinrVersDoaError()
    pass
