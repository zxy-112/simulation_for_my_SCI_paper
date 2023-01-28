from typing import Iterable
import numpy as np

class Array():

    def __init__(self):
        """
        """
        self.signals = []

    def cuttedSignal(self):
        """
        return the signals, but cut all signal to the same length.
        """

        minLength = float('inf')
        for signal in self.signals:
            if signal.size < minLength:
                minLength = signal.size

        res = []
        for signal in self.signals:
            res.append(signal[:minLength])

        return res

class LineArray(Array):

    INIT_POSITION = 0

    def __init__(self):
        super().__init__()
        self.positions = []  # the position of elements.
        self.thetas = []  # the theta of received signals.
    
    def addElement(self, position: float):
        self.positions.append(position)
    
    def receive(self, signal: np.ndarray, theta: float):
        self.signals.append(signal)
        self.thetas.append(np.deg2rad(theta))
    
    def receiveAll(self, signalThetaPairs: Iterable):
        """
        receive all (signal: np.ndarray, theta: float) in signalThetaPairs.
        """
        for signal, theta in signalThetaPairs:
            self.receive(signal, theta)
    
    def removeAll(self):
        """
        remove all signals.
        """
        self.signals = []
        self.thetas = []
    
    def getOutput(self):
        """
        the output of the array, which is a mxn numpy array, where m is
        the element number, n is the minmum length of all the received
        signals.
        """
        
        signals = self.cuttedSignal()
        res = []
        for position in self.positions:
            toSum = []
            for signal, theta in zip(signals, self.thetas):
                toSum.append(signal * LineArray.diff(position, theta))
            res.append(sum(toSum))
        return np.array(res)
    
    def getSynOutput(self, weight: np.ndarray):
        """
        get the beamformer output, which is acquired by apply the weights to the
        return of getOutput().
        weight -- the nx1 numpy ndarray.
        """
        output = self.getOutput()
        return np.squeeze(weight @ output)
    
    def aTheta(self, theta: float):
        """
        the guide vector of specified theta \alpha(theta) which is 
        a nx1 numpy array, n is the number of the array .
        theta -- the specified theta.
        """
        positions = np.array(self.positions) - LineArray.INIT_POSITION
        theta = np.deg2rad(theta)
        res = np.exp(1j*np.pi*2*positions*np.sin(theta))
        res = res.reshape((-1, 1))
        return res

    @staticmethod
    def diff(position: float, theta: float):
        """
        the phase response at position according to INIT POSITION, 
        a signal has different phase at different position, and it 
        is determined by the theta of the signal.
        """
        diffPosition = position - LineArray.INIT_POSITION
        return np.exp(1j * 2 * np.pi * diffPosition * np.sin(theta))

    @staticmethod
    def uniformArrray(number: int, space: float=0.5):
        """
        return an uniform array.
        number -- the number of the elements.
        space -- the space of the adjacent elements.
        """
        res = LineArray()
        current = LineArray.INIT_POSITION
        for _ in range(number):
            res.addElement(current)
            current += space

        return res
