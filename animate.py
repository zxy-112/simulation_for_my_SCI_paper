import numpy as np
import matplotlib.animation as ani
import matplotlib.pyplot as plt
import simulator
import zxy.methods

class SimulatorCancellation(simulator.SimulatorBase):
    def __init__(self):
        super().__init__()
        self.cnr = ()
        self.coherentTheta = ()
        self.inr = (30, 30)
        self.interTheta = (20, -30)
        self.snr = 0
        self.expectTheta = 0
    
    def noCoherentCancellation(self):
        oldSnr = self.snr
        snrs = np.linspace(-40, 40, 801)
        aTheta = self.getATheta(self.arrayNumber)
        thetas = np.linspace(-90, 90, 1801)

        fig, (ax1, ax2) = self.subplots(1, 2)
        lineResponse, = ax1.plot([], [])

        def aniFunc(i):
            self.snr = snrs[i]
            self.resetRng()
            output = self.getOutput()
            mvdrWeight = zxy.methods.mvdr(output, self.expectTheta, aTheta)
            response = zxy.methods.response(mvdrWeight, thetas, aTheta)

            lineResponse.set(xdata=thetas, ydata=response)
            ax1.set_xlim([thetas[0], thetas[-1]])
            ax1.set_ylim([-60, 5])
        
        theAni = ani.FuncAnimation(fig, aniFunc, range(len(snrs)), save_count=len(snrs), interval=200)
        theAni.save(self.savePath + "/noCoherentCancellation.mp4", dpi=600)

        self.snr = oldSnr

if __name__ == "__main__":
    sc = SimulatorCancellation()
    sc.noCoherentCancellation()
