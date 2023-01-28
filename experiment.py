
import matplotlib.pyplot as plt
import numpy as np
from zxy.array import LineArray
import zxy.methods
from simulator import SimulatorBase

class ExperimentSimulator(SimulatorBase):

    def __init__(self):
        super().__init__()
        self.arrayNumber = 8
        self.arraySpacing = 0.872093023255814
        self.snapShots = 1024
        self.filename = '两个都有3000hz.txt'
        self.filename = '只有20°手机3000hz.txt'
        self.expectTheta = 0
        self.coherentTheta = (20,)
        self.virtualPow = 100000
        self.widenTheta = 1
    
    def readData(self, filename):

        data = None
        with open(filename) as f:
            data = f.readlines()
        
        usedData = data[7: 7 + self.snapShots]
        snapMatrix = [item.split() for item in usedData]
        snapMatrix = np.array(snapMatrix, dtype=np.complex128)
        snapMatrix = zxy.methods.hermitian(snapMatrix)
        snapMatrix = np.fft.fft(snapMatrix, axis=1)
        snapMatrix[:, snapMatrix.shape[1]//2+1:] = 0
        snapMatrix = np.fft.ifft(snapMatrix, axis=1)

        for k in range(8):
            plt.plot(np.real(snapMatrix[k, :256]), label='ch'+str(k+1))
        plt.legend()
        plt.show()

        thetas, res = zxy.methods.spatialMusic(snapMatrix, self.getATheta(self.arrayNumber), 4, 2)
        plt.plot(thetas, res)
        plt.show()

        return snapMatrix
    
    def simulate(self):

        aTheta = self.getATheta(self.arrayNumber)
        output = self.readData(self.filename)

        mvdrWeight = zxy.methods.mvdr(output, self.expectTheta, aTheta, self.diagLoad)
        coherentDoa = (20,)
        mvdrVirturalWidenWeight = zxy.methods.mvdrVirtualInter(output, self.expectTheta,\
            coherentDoa, tuple([self.virtualPow] * len(self.coherentTheta)), aTheta, widenTheta=self.widenTheta, diagLoad=self.diagLoad)
        spatialWeight = zxy.methods.spatialSmooth(output, self.expectTheta, aTheta, 4, self.diagLoad)
        mcmvWeight = zxy.methods.mcmv(output, self.expectTheta, self.coherentTheta, aTheta, self.diagLoad)
        ctmvWeight = zxy.methods.ctmv(output, self.expectTheta, self.coherentTheta, aTheta, 0, self.diagLoad)
        fig, ax = self.subplots()
        self.responsePlot(mvdrWeight, ax, label='MVDR')
        self.responsePlot(spatialWeight, ax, label='spatial smooth')
        self.responsePlot(mvdrVirturalWidenWeight, ax, label='proposed')
        self.responsePlot(mcmvWeight, ax, label='MCMV')
        self.responsePlot(ctmvWeight, ax, label='CTMV')
        ax.legend()
        self.savefig(fig, 'experimentResponse.svg')

if __name__ == "__main__":
    es = ExperimentSimulator()
    es.simulate()
