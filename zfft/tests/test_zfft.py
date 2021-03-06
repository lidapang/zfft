from zfft import chirpz
from zfft import fft as zfft
from zfft import ifft as zifft
from zfft import fftfreq as zfftfreq
import numpy as np
import unittest
from scipy.fftpack import fft, ifft, fftfreq, fftshift



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

    def test_fft(self):
        """Test 1D case."""
        M = 5
        sig = np.random.randn(M) + 1j * np.random.randn(M)
        dt = 0.1
        fs = 1 / dt
        freq1 = fftshift( fftfreq(M, dt) )
        f0 = fftshift( fftfreq(M, dt) )[0]
        f1 = fftshift( fftfreq(M, dt) )[-1]
        out1 = zfft(sig, f0, f1, fs)
        out2 = fftshift( fft(sig, M) )
        self.assertTrue(np.allclose(out1, out2))



        M = 6
        sig = np.random.randn(M) + 1j * np.random.randn(M)
        dt = 0.1
        fs = 1 / dt
        freq1 = fftshift( fftfreq(M, dt) )
        f0 = fftshift( fftfreq(M, dt) )[0]
        f1 = fftshift( fftfreq(M, dt) )[-1]
        out1 = zfft(sig, f0, f1, fs)
        out2 = fftshift( fft(sig, M) )
        self.assertTrue(np.allclose(out1, out2))


        # Find zfft for subset of fft frequency values
        M = 5
        sig = np.random.randn(M) + 1j * np.random.randn(M)
        dt = 0.1
        fs = 1 / dt
        freq1 = fftshift( fftfreq(M, dt) )
        f0 = fftshift( fftfreq(M, dt) )[0]
        f1 = fftshift( fftfreq(M, dt) )[2]
        out1 = zfft(sig, f0, f1, fs, M-2)
        out2 = fftshift( fft(sig, M) )[:3]
        self.assertTrue(np.allclose(out1, out2))




        """Test 2D case."""
        M = 5
        sig = np.random.randn(M, M) + 1j * np.random.randn(M, M)
        dt = 0.1
        fs = 1 / dt
        freq1 = fftshift( fftfreq(M, dt) )
        f0 = fftshift( fftfreq(M, dt) )[0]
        f1 = fftshift( fftfreq(M, dt) )[-1]
        out1 = zfft(sig, f0, f1, fs, axis=0)
        out2 = fftshift( fft(sig, M, axis=0), axes=0)
        self.assertTrue(np.allclose(out1, out2))

        M = 6
        sig = np.random.randn(M, M) + 1j * np.random.randn(M, M)
        dt = 0.1
        fs = 1 / dt
        freq1 = fftshift( fftfreq(M, dt) )
        f0 = fftshift( fftfreq(M, dt) )[0]
        f1 = fftshift( fftfreq(M, dt) )[-1]
        out1 = zfft(sig, f0, f1, fs, axis=0)
        out2 = fftshift( fft(sig, M, axis=0), axes=0)
        self.assertTrue(np.allclose(out1, out2))



        """Test 3D case."""
        M = 5
        sig = np.random.randn(M, M, M) + 1j * np.random.randn(M, M, M)
        dt = 0.1
        fs = 1 / dt
        freq1 = fftshift( fftfreq(M, dt) )
        f0 = fftshift( fftfreq(M, dt) )[0]
        f1 = fftshift( fftfreq(M, dt) )[-1]
        out1 = zfft(sig, f0, f1, fs, axis=0)
        out2 = fftshift( fft(sig, M, axis=0), axes=0)
        self.assertTrue(np.allclose(out1, out2))



        M = 6
        sig = np.random.randn(M, M) + 1j * np.random.randn(M, M)
        dt = 0.1
        fs = 1 / dt
        freq1 = fftshift( fftfreq(M, dt) )
        f0 = fftshift( fftfreq(M, dt) )[0]
        f1 = fftshift( fftfreq(M, dt) )[-1]
        out1 = zfft(sig, f0, f1, fs, axis=0)
        out2 = fftshift( fft(sig, M, axis=0), axes=0)
        self.assertTrue(np.allclose(out1, out2))



    def test_ifft(self):
        """Test 1D case."""
        M = 5
        sig = np.random.randn(M) + 1j * np.random.randn(M)
        dt = 0.1
        t = np.arange(M) * dt
        out1 = zifft(sig, t[0], t[-1], dt, M)
        out2 = ifft(sig, M)
        self.assertTrue(np.allclose(out1, out2))


        M = 5
        sig = np.random.randn(M) + 1j * np.random.randn(M)
        dt = 0.1
        t = np.arange(M) * dt
        out1 = zifft(sig, t[1], t[-1], dt, M-1)
        out2 = ifft(sig, M)
        self.assertTrue(np.allclose(out1, out2[1:]))

        M = 5
        sig = np.random.randn(M) + 1j * np.random.randn(M)
        dt = 0.1
        t = np.arange(M) * dt
        out1 = zifft(sig, t[0], t[-2], dt, M-1)
        out2 = ifft(sig, M)
        self.assertTrue(np.allclose(out1, out2[:-1]))



        M = 6
        sig = np.random.randn(M) + 1j * np.random.randn(M)
        dt = 0.1
        t = np.arange(M) * dt
        out1 = zifft(sig, t[0], t[-1], dt, M)
        out2 = ifft(sig, M)
        self.assertTrue(np.allclose(out1, out2))


        M = 6
        sig = np.random.randn(M) + 1j * np.random.randn(M)
        dt = 0.1
        t = np.arange(M) * dt
        out1 = zifft(sig, t[1], t[-1], dt, M-1)
        out2 = ifft(sig, M)
        self.assertTrue(np.allclose(out1, out2[1:]))

        M = 6
        sig = np.random.randn(M) + 1j * np.random.randn(M)
        dt = 0.1
        t = np.arange(M) * dt
        out1 = zifft(sig, t[0], t[-2], dt, M-1)
        out2 = ifft(sig, M)
        self.assertTrue(np.allclose(out1, out2[:-1]))





    def test_fftfreq(self):
        M = 8
        freq1 = fftshift(fftfreq(M, d=1.0))
        freq2 = zfftfreq(f0=freq1[0], f1=freq1[-1], M=M)
        self.assertTrue(np.allclose(freq1, freq2))


if __name__ == '__main__':
    print 'Running unit tests for test_zfft.py'
    unittest.main()
