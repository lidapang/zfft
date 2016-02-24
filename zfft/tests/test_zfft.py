from zfft import chirpz, zfft, zfftfreq
import numpy as np
import unittest
from scipy.fftpack import fft, fftfreq, fftshift



class TestCode(unittest.TestCase):
    def test_chirpz(self):
        """Test 1D case."""
        sig = np.array([0., 0., 0., 1., 1., 0., 0., 0.])
        M = sig.shape[-1]
        A = 1.
        W = np.exp(-1j * 2 * np.pi / M)
        out1 = chirpz(x=sig)
        out2 = fft(sig, M)
        self.assertTrue(np.allclose(out1, out2))

        sig = np.array([0., 0., 0., 1., 1., 0., 0., 0.])
        M = sig.shape[-1]
        A = 1.
        W = np.exp(-1j * 2 * np.pi / M)
        out1 = chirpz(x=sig, M=M+1)
        out2 = fft(sig, M+1)
        self.assertTrue(np.allclose(out1, out2))

        sig = np.array([0., 0., 0., 1., 1., 0., 0., 0.])
        M = sig.shape[-1]
        A = 1.
        W = np.exp(-1j * 2 * np.pi / M)
        out1 = chirpz(x=sig, A=A, W=W, M=M)
        out2 = fft(sig, M)
        self.assertTrue(np.allclose(out1, out2))

        """Test 2D case."""
        sig = np.array([[0., 0., 1., 0.],[1., 2., 3., 4.]])
        M = sig.shape[-1]
        A = 1.
        W = np.exp(-1j * 2 * np.pi / M)
        out1 = chirpz(x=sig)
        out2 = fft(sig, M)
        self.assertTrue(np.allclose(out1, out2))

        """Test 3D case."""
        sig = np.array([[[0., 0., 1., 0.],[1., 2., 3., 4.]],
                        [[0., 0., 1., 0.],[1., 2., 3., 4.]]])
        M = sig.shape[-1]
        A = 1.
        W = np.exp(-1j * 2 * np.pi / M)
        out1 = chirpz(x=sig)
        out2 = fft(sig, M)
        self.assertTrue(np.allclose(out1, out2))

    def test_zfft(self):
        """Test 1D case."""
        sig = np.array([0., 0., 0., 1.])
        M = sig.shape[-1]
        f0 = 0.
        f1 = 1.
        fs = 1.
        freq = fftfreq(M)
        out1 = zfft(sig)
        out2 = fft(sig, M)
        self.assertTrue(np.allclose(out1, out2))

        sig = np.array([0., 0., 0., 1.])
        M = len(sig)
        f0 = 0.
        f1 = 1.
        fs = 1.
        freq = fftfreq(M)
        out1 = zfft(sig, f0, f1, fs)
        out2 = fft(sig, M)
        self.assertTrue(np.allclose(out1, out2))

        sig = np.array([0., 0., 0., 1.])
        M = len(sig)
        f0 = 0.
        f1 = 1.
        fs = 1.
        freq = fftfreq(M)
        out1 = zfft(sig, f0, f1, fs, M)
        out2 = fft(sig, M)
        self.assertTrue(np.allclose(out1, out2))

        sig = np.array([0., 0., 0., 1.])
        M = len(sig)
        f0 = 0.
        f1 = 1.
        fs = 1.
        freq = fftfreq(M)
        out1 = zfft(zfft(sig, f0, f1, fs, M), f0, f1, fs, M)
        out2 = fft(fft(sig, M), M)
        self.assertTrue(np.allclose(out1, out2))

        sig = np.array([1j, 0., 0., 1.])
        M = len(sig)
        f0 = 0.
        f1 = 1.
        fs = 1.
        freq = fftfreq(M)
        out1 = zfft(sig, f0, f1, fs, M)
        out2 = fft(sig, M)
        self.assertTrue(np.allclose(out1, out2))

        """Test 2D case."""
        sig = np.array([[0., 0., 1., 0.],[1., 2., 3., 4.]])
        axis = 0
        M = sig.shape[axis]
        f1 = 1.
        fs = 1.
        freq = fftfreq(M)
        out1 = zfft(sig, f0, f1, fs, M, axis)
        out2 = fft(sig, M, axis)
        self.assertTrue(np.allclose(out1, out2))

        sig = np.array([[0., 0., 1., 0.],[1., 2., 3., 4.]])
        axis = 1
        M = sig.shape[axis]
        f1 = 1.
        fs = 1.
        freq = fftfreq(M)
        out1 = zfft(sig, f0, f1, fs, M, axis)
        out2 = fft(sig, M, axis)
        self.assertTrue(np.allclose(out1, out2))

        sig = np.array([[0., 0., 1., 0.],
                        [1., 2., 3., 4.],
                        [4., 3., 2., 1.],
                        [0., 1., 0., 1.]])
        axis = 0
        M = sig.shape[axis]
        f1 = 1.
        fs = 1.
        freq = fftfreq(M)
        out1a = zfft(sig, f0, f1, fs, M, axis)
        out2a = fft(sig, M, axis)
        axis = 1
        out1b = zfft(out1a, f0, f1, fs, M, axis)
        out2b = fft(out2a, M, axis)
        self.assertTrue(np.allclose(out1b, out2b))

        """Test 3D case."""
        sig = np.array([[[0., 0., 1., 0.],[1., 2., 3., 4.]],
                        [[0., 0., 1., 0.],[1., 2., 3., 4.]]])
        axis = 0
        M = sig.shape[axis]
        f1 = 1.
        fs = 1.
        freq = fftfreq(M)
        out1 = zfft(sig, f0, f1, fs, M, axis)
        out2 = fft(sig, M, axis)
        self.assertTrue(np.allclose(out1, out2))

        sig = np.array([[[0., 0., 1., 0.],[1., 2., 3., 4.]],
                        [[0., 0., 1., 0.],[1., 2., 3., 4.]]])
        axis = 1
        M = sig.shape[axis]
        f1 = 1.
        fs = 1.
        freq = fftfreq(M)
        out1 = zfft(sig, f0, f1, fs, M, axis)
        out2 = fft(sig, M, axis)
        self.assertTrue(np.allclose(out1, out2))

        sig = np.array([[[0., 0., 1., 0.],[1., 2., 3., 4.]],
                        [[0., 0., 1., 0.],[1., 2., 3., 4.]]])
        axis = 2
        M = sig.shape[axis]
        f1 = 1.
        fs = 1.
        freq = fftfreq(M)
        out1 = zfft(sig, f0, f1, fs, M, axis)
        out2 = fft(sig, M, axis)
        self.assertTrue(np.allclose(out1, out2))

    def test_zfftfreq1(self):
        f0 = -0.5
        f1 = 0.5
        M = 8
        freq1 = zfftfreq(f0=f0, f1=f1, M=M)
        freq2 = fftshift(fftfreq(M))
        self.assertTrue(np.allclose(freq1, freq2))


if __name__ == '__main__':
    print 'Running unit tests for test_zfft.py'
    unittest.main()
