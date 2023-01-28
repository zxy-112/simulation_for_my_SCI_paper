import os.path as path
import numpy as np
import simulator
import zxy.signal
import zxy.methods

class Generator(simulator.SimulatorBase):

    def __init__(self):
        super().__init__()
        self.arrayNumber = 8
        self.sinFre = 3000
        self.dataPath = path.expanduser('~/dnnData')
        self.snapShots = 128
        self.inr = ()
        self.interTheta = ()

        self.expectTheta = 0
        
        self.generateSnrs = np.linspace(0, 10, 100)
        self.generateCnrs = np.linspace(0, 10, 100)
        self.generateCoherentThetas = np.concatenate((np.linspace(-60, -10, 100), np.linspace(10, 60, 100)))
    
    def getCoherentSignal(self):
        res = []
        for cnr, theta in zip(self.cnr, self.coherentTheta):
            coherentInter = zxy.signal.sinPulse(self.lfmWidth, self.sinFre, self.dt)
            coherentInter = coherentInter * zxy.methods.snr2value(cnr, self.noisePower)
            res.append((coherentInter, theta))
        return res
    
    def getExpectSignal(self):
        expectSignal = zxy.signal.sinPulse(self.lfmWidth, self.sinFre, self.dt)
        expectSignal = expectSignal * zxy.methods.snr2value(self.snr, self.noisePower)
        return expectSignal, self.expectTheta

    def generate(self):
        aTheta = self.getATheta(self.arrayNumber)
        for snr in self.generateSnrs:
            for cnr in self.generateCnrs:
                for theta in self.generateCoherentThetas:
                    self.cnr = (cnr,)
                    self.snr = snr
                    self.coherentTheta = (theta,)
                    optWeight = zxy.methods.optWeight(
                        self.expectTheta, self.snr, self.coherentTheta, self.cnr, self.noisePower, aTheta)
                    output = self.getOutput()

if __name__ == "__main__":
    ge = Generator()
    ge.generate()
