from math import floor
import numpy as np
import matplotlib.pyplot as plt

def is_num(x):
    return isinstance(x, (int, float))

def is_positive(x):
    return is_num(x) and x > 0

def normalize(x):
    return x / np.max(x)

class Signal:

    def __init__(self, signal_length=10e-6, amplitude=1, fre_shift=0, phase_shift=0, delay=0, signal_type='expect'):
        assert is_positive(signal_length), 'invalid signal length'
        assert is_positive(amplitude)
        for item in (fre_shift, phase_shift, delay):
            assert is_num(item)
        assert isinstance(signal_type, str)

        self.signal_length = signal_length
        self._signal = None
        self.amplitude = amplitude
        self.phase_shift = phase_shift
        self.fre_shift = fre_shift
        self.delay = delay
        self.signal_type = signal_type

    def sample(self, points):
        """
        sample the signal with specified points
        """
        assert isinstance(points, int) and points >=0, 'invalid sample points'
        if points == 0:
            self._signal = None
        else:
            t, self.interval = np.linspace(0, self.signal_length, points, endpoint=False, retstep=True)
            self._signal = self.expression(t)
            if self.delay:
                delay_points = floor(self.delay * points / self.signal_length)
                if delay_points > 0:
                    self._signal[delay_points:], self._signal[:delay_points] = self._signal[:-delay_points], 0
                else:
                    self._signal[:delay_points], self.signal[delay_points:] = self._signal[-delay_points:], 0
        return self.signal

    def expression(self, t):
        """
        the mathematical expression of the signal
        """
        return self.amplitude * np.exp(1j * 2 * np.pi * self.fre_shift * t) * np.exp(1j * self.phase_shift)

    def plot(self, from_to=(None, None), fig_ax_pair=(None, None), **plt_kwargs):
        """
        plot the waveform of the signal
        """
        assert self._signal is not None, 'no signal, maybe first sample using sample method'
        fig, ax = fig_ax_pair
        if fig is None:
            fig, ax = plt.subplots()

        begin, end = self.format(from_to)
        ax.plot(self._signal.real[begin: end], **plt_kwargs)
        ax.set_ylim((-1.2 * self.amplitude, 1.2 * self.amplitude))
        fig.show()
        return fig, ax

    def fft_plot(self, from_to=(None, None), fig_ax_pair=(None, None), **plt_kwargs):
        assert self._signal is not None, 'no signal, maybe first sample using sample method'
        fig, ax = fig_ax_pair
        if fig is None:
            fig, ax = plt.subplots()

        begin, end = self.format(from_to)
        fft = lambda x: np.fft.fftshift(np.fft.fft(x))
        fftfreq = lambda n: np.fft.fftshift(np.fft.fftfreq(n))

        interest_signal = self._signal[begin: end]
        res = fft(interest_signal)
        freq = fftfreq(res.size)
        ax.plot(freq, normalize(np.abs(res)), **plt_kwargs)
        fig.show()
        return fig, ax

    def check(self, begin, end):
        assert isinstance(begin, int) and isinstance(end, int), 'invalid begin or end'
        assert 0 <= begin < end <= len(self._signal), 'invalid begin or end'

    def format(self, from_to):
        begin, end = from_to
        begin_default, end_defult = (0, len(self._signal))
        begin = begin if begin else begin_default
        end = end if end else end_defult
        self.check(begin, end)
        return begin, end

    @property
    def band_width(self):
        """
        bandwidth of the signal
        """
        return 1 / self.signal_length

    @property
    def signal(self):
        """
        return the signal
        """
        assert self._signal is not None, 'not sampling yet, first sample using sample method'
        if self._signal is not None:
            return self._signal
        else:
            raise NoSampleError('not sampling yet, first sample using sample method')

    def __str__(self):
        return 'Signal with length {} at {}'.format(self.signal_length, id(self))

class Wave:

    def __init__(self, theta):
        self.theta = theta

class Wave2D(Wave):

    def __init__(self, theta=0):
        assert isinstance(theta, (int, float)) and -90 <= theta <= 90, 'invalid theta'
        Wave.__init__(self, theta)

class Wave3D(Wave):

    def __init__(self, theta=(0, 0)):
        assert isinstance(theta, (tuple, list)) and len(theta) == 2, 'invalid theta'
        Wave.__init__(self, theta)

class Lfm(Signal):

    def __init__(self, signal_length=10e-6, band_width=10e6, amplitude=1, fre_shift=0, phase_shift=0, delay=0, signal_type='expect'):
        assert isinstance(band_width, (int, float)) and band_width > 0, 'invalid bandwidth'
        Signal.__init__(
                self,
                signal_length=signal_length,
                amplitude=amplitude,
                fre_shift=fre_shift,
                phase_shift=phase_shift,
                delay=delay,
                signal_type=signal_type
                )
        self._band_width = band_width

    def expression(self, t):
        mu = self._band_width / self.signal_length
        power_of_t = np.power(t - self.signal_length/2, 2)
        return Signal.expression(self, t) * np.exp(1j * np.pi * mu * power_of_t)

    @property
    def band_width(self):
        return self._band_width

    @band_width.setter
    def band_width(self, value):
        assert isinstance(value, (int, float)) and value > 0, 'invalid bandwidth'
        self._band_width = value

class LfmWave2D(Lfm, Wave2D):
    """
    can be received by lineArray
    """

    def __init__(self, theta=0, signal_length=10e-6, band_width=10e6, amplitude=1, fre_shift=0, phase_shift=0, delay=0, signal_type='expect'):
        Lfm.__init__(
                self,
                signal_length=signal_length,
                band_width=band_width,
                amplitude=amplitude,
                fre_shift=fre_shift,
                phase_shift=phase_shift,
                delay=delay,
                signal_type=signal_type
                )
        Wave2D.__init__(self, theta)

class LfmWave3D(Lfm, Wave3D):

    def __init__(self, theta=(0, 0), signal_length=10e-6, band_width=10e6, amplitude=1, fre_shift=0, phase_shift=0, delay=0, signal_type='expect'):
        Lfm.__init__(
                self,
                signal_length=signal_length,
                band_width=band_width,
                amplitude=amplitude,
                fre_shift=fre_shift,
                phase_shift=phase_shift,
                delay=delay,
                signal_type=signal_type
                )
        Wave3D.__init__(self, theta)

class Cos(Signal):

    pass

class CosWave2D(Cos, Wave2D):

    def __init__(self, theta=0, signal_length=10e-6, amplitude=1, fre_shift=0, phase_shift=0, delay=0, signal_type='expect'):
        Cos.__init__(
                self,
                signal_length=signal_length,
                amplitude=amplitude,
                fre_shift=fre_shift,
                phase_shift=phase_shift,
                delay=delay,
                signal_type=signal_type
                )
        Wave2D.__init__(self, theta)

class CosWave3D(Cos, Wave3D):

    def __init__(self, theta=(0, 0), signal_length=10e-6, amplitude=1, fre_shift=0, phase_shift=0, delay=0, signal_type='expect'):
        Cos.__init__(
                self,
                signal_length=signal_length,
                amplitude=amplitude,
                fre_shift=fre_shift,
                phase_shift=phase_shift,
                delay=delay,
                signal_type=signal_type
                )
        Wave3D.__init__(self, theta)

class GaussionNoise(Signal):

    def __init__(self, signal_length=10e-6, amplitude=1, fre_shift=0, phase_shift=0, delay=0, signal_type='expect'):
        Signal.__init__(
                self,
                signal_length=signal_length,
                amplitude=amplitude,
                fre_shift=fre_shift,
                phase_shift=phase_shift,
                delay=delay,
                signal_type=signal_type
                )
        self.points = None

    def sample(self, points):
        Signal.sample(self, points)
        self.points = points

    def expression(self, t):
        power = self.amplitude ** 2
        amplitude = np.sqrt(power / 2)
        return amplitude * (np.random.randn(*t.shape) + 1j * np.random.randn(*t.shape))

    def plot(self, from_to=(None, None), fig_ax_pair=(None, None), **plt_kwargs):
        fig, ax = Signal.plot(self, from_to, fig_ax_pair, **plt_kwargs)
        if ax is not None:
            max_value = np.max(np.abs(self.signal)) * 1.2
            ax.set_ylim([-max_value, max_value])
        return fig, ax

    @property
    def band_width(self):
        if self.points is None:
            return None
        else:
            return self.points / self.signal_length

class ZeroSignal(Signal):

    def __init__(self, signal_length):
        Signal.__init__(self, signal_length=signal_length)

    def expression(self, t):
        return np.zeros_like(t)

class NoiseWave2D(GaussionNoise, Wave2D):

    def __init__(self, theta=0, signal_length=10e-6, amplitude=1, fre_shift=0, phase_shift=0, delay=0, signal_type='expect'):
        GaussionNoise.__init__(
                self,
                signal_length=signal_length,
                amplitude=amplitude,
                fre_shift=fre_shift,
                phase_shift=phase_shift,
                delay=delay,
                signal_type=signal_type
                )
        Wave2D.__init__(self, theta)

class NoiseWave3D(GaussionNoise, Wave3D):

    def __init__(self, theta=(0, 0), signal_length=10e-6, amplitude=1, fre_shift=0, phase_shift=0, delay=0, signal_type='expect'):
        GaussionNoise.__init__(
                self,
                signal_length=signal_length,
                amplitude=amplitude,
                fre_shift=fre_shift,
                phase_shift=phase_shift,
                delay=delay,
                signal_type=signal_type
                )
        Wave3D.__init__(self, theta)

class SignalWave2D(Signal, Wave2D):

    def __init__(self, theta=0, signal_length=10e-6, amplitude=1, fre_shift=0, phase_shift=0, delay=0, signal_type='expect', new_expression=None):
        Wave2D.__init__(self, theta=theta)
        Signal.__init__(self,
                signal_length=signal_length,
                amplitude=amplitude,
                fre_shift=fre_shift,
                phase_shift=phase_shift,
                delay=delay,
                signal_type=signal_type)
        self.new_expression = new_expression

    def expression(self, t):
        if self.new_expression is not None:
            return self.new_expression(t)
        else:
            return Signal.expression(self, t)

    @staticmethod
    def concatenate(theta, *signal_seq, signal_type='expect'):

        res_length = 0
        length_seq = []
        for signal in signal_seq:
            length_seq.append(signal.signal_length)
        signal_length = sum(length_seq)
        ratio_seq = [item/signal_length for item in length_seq]
        amplitude = max([item.amplitude for item in signal_seq])

        def new_expression(t):
            length_of_t = len(t)
            points_for_every = []
            for index, item in enumerate(ratio_seq):
                if index == len(ratio_seq) - 1:
                    points = length_of_t - sum(points_for_every)
                else:
                    points = int(length_of_t * item)
                signal_seq[index].sample(points)
                points_for_every.append(points)

            all_sampled = [item.signal for item in signal_seq]
            return np.concatenate(all_sampled)

        res = SignalWave2D(theta=theta,
            signal_length=sum(length_seq),
            amplitude=amplitude,
            new_expression=new_expression,
            signal_type=signal_type)
        return res

class CossWave2D(Signal, Wave2D):
    def __init__(self, theta=0, signal_length=10e-6, amplitude=1, fre_shift=0, phase_shift=0, delay=0, signal_type='expect', fres=(0,)):
        Signal.__init__(
                self,
                signal_length=signal_length,
                amplitude=amplitude,
                fre_shift=fre_shift,
                phase_shift=phase_shift,
                delay=delay,
                signal_type=signal_type
                )
        Wave2D.__init__(self, theta=theta)
        self.fres = fres

    def expression(self, t):
        single_amplitude = 1 / np.sqrt(len(self.fres))
        res = np.zeros(t.shape, dtype=np.complex128)
        for fre in self.fres:
            res += np.exp(1j * 2 * np.pi * fre * t)
        return Signal.expression(self, t) * res * single_amplitude

    def plot(self, from_to=(None, None), fig_ax_pair=(None, None), **plt_kwargs):
        fig, ax = Signal.plot(self, from_to, fig_ax_pair, **plt_kwargs)
        ax.autoscale(True)
        return fig, ax

class NoSampleError(BaseException):
    pass

if __name__ == '__main__':
    def do_wrong(string):
        print(dir())
        try:
            exec(string)
        except AssertionError:
            pass
        else:
            assert False
    # sample_points = 1000
    # do_wrong('signal = Signal(-10)')
    # signal = Signal(10e-6)
    # print(signal)
    # do_wrong('signal.signal')
    # signal.sample(sample_points)
    # assert signal.signal.shape == (sample_points,)
    # print('signal_length: {}'.format(signal.signal_length))
    # print('signal bandwidth: {}'.format(signal.band_width))
    # signal.plot()
    # signal.fre_shift = 1e6
    # signal.sample(sample_points)
    # signal.plot()
    # signal.phase_shift = 0.5 * np.pi
    # signal.sample(sample_points)
    # signal.plot()
    # lfm = LfmWave2D()
    # do_wrong('lfm.signal')
    # lfm.plot()  # 无事发生
    # lfm.sample(sample_points)
    # lfm.plot()
    # print('lfm bandwidth: {}'.format(lfm.band_width))
    # lfm.band_width = 5e6
    # lfm.sample(sample_points)
    # lfm.plot()
    # do_wrong('lfm = LfmWave2D(signal_length=-1)')
    # noise = GaussionNoise(10e-6)
    # print('noise bandwidth: {}'.format(noise.band_width))
    # noise.sample(sample_points)
    # print('noise bandwidth: {}'.format(noise.band_width))

    # print('all test passed')
    lfm = Lfm()
    lfm.sample(1000)
    lfm.fft_plot(from_to=(0, 700))
    lfm.fft_plot(from_to=(0, 200))
