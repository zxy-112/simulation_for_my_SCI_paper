import numpy as np

sqrtOf2 = np.sqrt(2)

def lfmPulse(width: float, bandWidth: float, dt: float):
    """
    return a numpy array which is a LFM pulse.
    width -- the pulse width.
    bandWidth -- the band width.
    dt -- the sampling space, which is 1/Fs.
    """
    beginFre = - bandWidth / 2
    slope = bandWidth / width
    t = np.arange(0, width, dt)
    return np.exp(2j*np.pi*beginFre*t + 1j*np.pi*slope*np.power(t, 2))

def replicate(signal: np.ndarray, times: int, dutyCycle: float):
    """
    return the signal which is relplicated severval times
    and has the specified dutyCycle.
    signal -- the numpy ndarray.
    times -- times of replication.
    dutyCycle -- the ratio that the signal in duty in one PRT.
    """
    signals = []
    if dutyCycle == 1:
        zeroLength = 0
    else:
        zeroLength = int(len(signal) * (1 - dutyCycle) / dutyCycle)
    
    if zeroLength == 0:
        zeroSignal = []
    else:
        zeroSignal = np.zeros(zeroLength)
    
    for _ in range(times):
        signals.append(signal)
        signals.append(zeroSignal)
    
    return np.concatenate(signals)

def sinPulse(width: float, fre: float, dt: float):
    """
    return sinsoid signal.
    width -- width of signal.
    fre -- fre of sinsoid.
    dt -- the sampling space.
    """
    t = np.arange(0, width, dt)
    return np.exp(1j*2*np.pi*fre*t)

def noise(width: float, dt: float, generator=None):
    """
    return noise signal.
    width -- the width of noise.
    dt -- the sampling space.
    """
    sample = int(width / dt)
    if generator is None:
        generator = np.random.default_rng()
    return generator.standard_normal(sample) * sqrtOf2 / 2 \
        + 1j * generator.standard_normal(sample) * sqrtOf2 / 2

def rightShift(signal: np.ndarray, points: int):
    """
    right shift the signal and add zero at the beginning.
    points -- points to shift.
    """
    return np.concatenate((np.zeros(points), signal))
