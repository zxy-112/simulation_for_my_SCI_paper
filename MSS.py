import numpy as np
import matplotlib.pyplot as plt
from simulator import SimulatorBase
import zxy.methods

class SimulatorTest(SimulatorBase):

    def __init__(self):
        super().__init__()
        self.cnr = (10, 10)
        self.coherentTheta = (20, -40)
        self.inr = ()
        self.interTheta = ()
        self.snr = 10

    def responseSimulate(self):
        output = self.getOutput()
        output = output[:, :self.snapShots]
        aTheta = self.getATheta(self.arrayNumber)

        ssWeight = zxy.methods.fss(output, self.expectTheta, aTheta, 7)
        mssWeight = zxy.methods.mss(output, self.expectTheta, aTheta, (7, 5, 6))
        mvdrWeight = zxy.methods.mvdr(output, self.expectTheta, aTheta)

        fig, ax = self.subplots(1, 1)
        self.responsePlot(ssWeight, ax, minValue=-80, label="SS")
        self.responsePlot(mssWeight, ax, minValue=-80, label="MSS")
        self.responsePlot(mvdrWeight, ax, minValue=-80, label="MVDR")
        self.setVline(ax)
        ax.legend()
        ax.set_xlabel("angle(\u00B0)")
        ax.set_ylabel("amplitude(dB)")
        ax.grid(True)
        self.savefig(fig, "多级空间平滑与空间平滑")
    
    def sinrVersSnr(self):
        
        self.resetRng()
        oldSNR = self.snr

        aTheta = self.getATheta(self.arrayNumber)
        snrs = np.arange(-20, 30, 1)
        ssSINRS = []
        mssSINRS = []
        mvdrSINRS = []

        for snr in snrs:
            self.snr = snr
            ssSamples = []
            mssSamples = []
            mvdrSamples = []

            for _ in range(self.monteCarlo):
                output = self.getOutput()
                output = output[:, :self.snapShots]
                ssWeight = zxy.methods.spatialSmooth(output, self.expectTheta, aTheta, 7, self.diagLoad)
                mssWeight = zxy.methods.mss(output, self.expectTheta, aTheta, (7, 5, 6))
                mvdrWeight = zxy.methods.mvdr(output, self.expectTheta, aTheta)
                
                ssSamples.append(self.getSINR(ssWeight))
                mssSamples.append(self.getSINR(mssWeight))               
                mvdrSamples.append(self.getSINR(mvdrWeight))

            ssSINRS.append(sum(ssSamples) / len(ssSamples))
            mssSINRS.append(sum(mssSamples) / len(mssSamples))
            mvdrSINRS.append(sum(mvdrSamples) / len(mvdrSamples))

        self.snr = oldSNR

        fig, ax = self.subplots()
        ax.plot(snrs, ssSINRS, label='SS')
        ax.plot(snrs, mssSINRS, label = "MSS")
        ax.plot(snrs, mvdrSINRS, label="MVDR")

        ax.legend()
        ax.set_xlabel('SNR(dB)')
        ax.set_ylabel('SINR(dB)')
        ax.grid(True)
        self.plotShow = True
        if self.plotShow:
            plt.show()
        self.savefig(fig)

class SimulatorTest2(SimulatorTest):
    
    def responseSimulate(self):
        self.resetRng()
        output = self.getOutput()
        output = output[:, :self.snapShots]
        aTheta = self.getATheta(self.arrayNumber)

        duvalWeight = zxy.methods.duvall(output, self.expectTheta, aTheta)
        proposed = zxy.methods.duvallBasedFast(output, self.expectTheta, aTheta, 2)

        fig, ax = self.subplots(1, 1)
        self.responsePlot(duvalWeight, ax, label="主辅阵对消MVDR")
        self.responsePlot(proposed, ax, label="主辅阵对消Toeplitz重构")
        ax.grid(True)
        ax.legend(prop=self.font)
        self.setVline(ax)

        self.savefig(fig)
    
    def sinrVersSnr(self):
        
        self.resetRng()
        oldSNR = self.snr

        aTheta = self.getATheta(self.arrayNumber)
        snrs = np.arange(-20, 20, 1)
        ssSINRS = []
        fullSSSINRS = []

        for snr in snrs:
            self.snr = snr
            ssSamples = []
            fullSamples = []

            # the sinr result of different subarray size(dimension 0) 
            # in different experiment(dimension 1). 
            for _ in range(self.monteCarlo):
                output = self.getOutput()
                output = output[:, :self.snapShots]
                ssWeight = zxy.methods.duvall(output, self.expectTheta, aTheta)

                fullWeight = zxy.methods.duvallBasedFast(output, self.expectTheta, aTheta, len(self.cnr))
                
                ssSamples.append(self.getSINR(ssWeight))
                fullSamples.append(self.getSINR(fullWeight))               

            ssSINRS.append(sum(ssSamples) / len(ssSamples))
            fullSSSINRS.append(sum(fullSamples) / len(fullSamples))

        self.snr = oldSNR

        fig, ax = self.subplots()
        ax.plot(snrs, ssSINRS, label='主辅阵对消MVDR')
        ax.plot(snrs, fullSSSINRS, label = "主辅阵对消Toeplitz重构")

        ax.legend(prop=self.font)
        ax.set_xlabel('SNR(dB)')
        ax.set_ylabel('SINR(dB)')
        ax.grid(True)
        self.plotShow = True
        if self.plotShow:
            plt.show()
        self.savefig(fig)

st = SimulatorTest()
st.responseSimulate()
st.sinrVersSnr()
plt.show()
