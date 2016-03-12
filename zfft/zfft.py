""" Zoom FFT function"""

import numpy as np
from time import time
import scipy.fftpack
from numpy import swapaxes

def chirpz(x, A=None, W=None, M=None):
    """chirpz(x, A, W, M) - Chirp z-transform of variable x

    Keyword Arguments:
    x -- array to evaluate chirp-z transform (along last dimension of array)
    A -- starting point of chirp-z contour
    W -- controls frequency sample spacing and shape of the contour
    M -- number of frequency sample points

    Return values:
    g -- chirp-z tranform coefficients

    From http://www.mail-archive.com/numpy-discussion@scipy.org/msg01812.html
    Last accessed December-06-2012

    Written by Stefan van der Walt

    Modified by Adam Luchies, 09/27/12
    Added support for 2- and 3- deminsional arrays. For 2-dimensional array,
    returns chirpz along axis = 1. For 3-dimensional array, returns chirpz
    along axis = 2.

    Modified by Adam Luchies 12/06/12
    Added support to allow M > N - compute chirp-z transform containing
    more points than the original sequence.

    Reference:
    Rabiner, L.R., R.W. Schafer and C.M. Rader. The Chirp z-Transform
    Algorithm. IEEE Transactions on Audio and Electroacoustics,
    AU-17(2):86--92, 1969

    The discrete z-transform,
    X(z) = \sum_{n=0}^{N-1} x_n z^{-n}
    is calculated at M points,
    z_k = AW^-k, k = 0,1,...,M-1
    for A and W complex, which gives
    X(z_k) = \sum_{n=0}^{N-1} x_n z_k^{-n}

    """


    # Handle default arguments
    if (A == None) & (W == None) & (M == None):
        M = x.shape[-1]
        A = 1.
        W = np.exp(-2. * np.pi * 1j / M)
    elif (A == None) & (W == None):
        A = 1.
        W = np.exp(-2. * np.pi * 1j / M)

    A = np.complex(A)
    W = np.complex(W)
    if np.issubdtype(np.complex, x.dtype) or np.issubdtype(np.float, x.dtype):
        dtype = x.dtype
    else:
        dtype = float

    x = np.asarray(x, dtype=np.complex)
    P = x.shape

    if len(P) == 1:
        N = P[-1]
        L = int(2 ** np.ceil(np.log2(M + N - 1)))

        n = np.arange(N, dtype=float)
        y = np.power(A, -n) * np.power(W, n ** 2 / 2.) *  x
        Y = scipy.fftpack.fft(y, L)

        n = np.arange(L, dtype=float)
        v = np.zeros(L, dtype=np.complex)
        v[:M] = np.power(W, -n[:M] ** 2 / 2.)
        v[L-N+1:] = np.power(W, -(L - n[L-N+1:]) ** 2 / 2.)
        V = scipy.fftpack.fft(v, L)

        g = scipy.fftpack.ifft(V * Y)[:M]
        k = np.arange(M)
        g = g * np.power(W, k ** 2 / 2.)

    elif len(P) == 2:
        N = P[-1]
        L = int(2 ** np.ceil(np.log2(M + N - 1)))

        n = np.arange(N, dtype=float)
        y = np.power(A, -n) * np.power(W, n ** 2 / 2.)
        y = np.tile(y, (P[0], 1)) * x
        Y = scipy.fftpack.fft(y, L)

        n = np.arange(L, dtype=float)
        v = np.zeros(L, dtype=np.complex)
        v[:M] = np.power(W, -n[:M] ** 2 / 2.)
        v[L-N+1:] = np.power(W, -(L - n[L-N+1:]) ** 2 / 2.)
        V = scipy.fftpack.fft(v)

        g = scipy.fftpack.ifft(np.tile(V, (P[0], 1)) * Y)[:,:M]
        k = np.arange(M)
        g = g * np.tile(np.power(W, k ** 2 / 2.), (P[0],1))

    elif len(P) == 3:
        N = P[-1]
        L = int(2 ** np.ceil(np.log2(M + N - 1)))

        n = np.arange(N,dtype=float)
        y = np.power(A,-n) * np.power(W,n ** 2 / 2.)
        y = np.tile(y, (P[0],P[1],1)) * x
        Y = scipy.fftpack.fft(y, L)

        n = np.arange(L, dtype=float)
        v = np.zeros(L, dtype=np.complex)
        v[:M] = np.power(W, -n[:M] ** 2 / 2.)
        v[L-N+1:] = np.power(W, -(L - n[L-N+1:]) ** 2 / 2.)
        V = scipy.fftpack.fft(v)

        g = scipy.fftpack.ifft(np.tile(V, (P[0], P[1],1)) * Y)[:,:,:M]
        k = np.arange(M)
        g = g * np.tile(np.power(W, k ** 2 / 2.), (P[0],P[1],1))
    # Return result
    return g


def fft(x, f0=0., f1=1., fs=1., M=None, axis=-1):
    """zfft(x, f0, f1, fs, M) - Zoom FFT function to evaluate the 1DFT
    coefficients for the rows of an array in the frequency range [f0, f1]
    using N points.

    Keyword arguments:
    x -- array to evaluate DFT (along last dimension of array)
    f0 -- lower bound of frequency bandwidth
    f1 -- upper bound of frequency bandwidth
    fs -- sampling frequency
    M -- number of points used when evaluating the 1DFT (N <= signal length)
    axis -- axis along which the fft's are computed (defaults to last axis)

    Return values:
    y -- DFT coefficients

    """

    # Handle default arguments
    if M == None:
        M = x.shape[-1]

    # Swap axes
    x = swapaxes(a=x, axis1=axis, axis2=-1)

    # Normalize frequency range
    f0_norm = f0 / (fs / 2.)
    f1_norm = f1 / (fs / 2.)

    # Determine shape of signal
    A = np.exp(1j * np.pi * f0_norm)
    W = np.exp(-1j * np.pi * (f1_norm - f0_norm) / (M - 1))
    y = chirpz(x=x, A=A, W=W, M=M)
    # Return result
    return swapaxes(a=y, axis1=axis, axis2=-1)

def ifft(x, f0=0., f1=1., fs=1., M=None, axis=-1):
    """zfft(x, f0, f1, fs, M) - Zoom FFT function to evaluate the 1DFT
    coefficients for the rows of an array in the frequency range [f0, f1]
    using N points.

    Keyword arguments:
    x -- array to evaluate DFT (along last dimension of array)
    f0 -- lower bound of frequency bandwidth
    f1 -- upper bound of frequency bandwidth
    fs -- sampling frequency
    M -- number of points used when evaluating the 1DFT (N <= signal length)
    axis -- axis along which the fft's are computed (defaults to last axis)

    Return values:
    y -- DFT coefficients

    """

    # Handle default arguments
    if M == None:
        M = x.shape[-1]

    # Swap axes
    x = swapaxes(a=x, axis1=axis, axis2=-1)

    # Normalize frequency range
    f0_norm = f0 / (fs / 2.)
    f1_norm = f1 / (fs / 2.)

    # Determine shape of signal
    A = np.exp(1j * np.pi * f0_norm)
    W = np.exp(-1j * np.pi * (f1_norm - f0_norm) / (M - 1))
    y = chirpz(x=x, A=A, W=W, M=M)
    # Return result
    return swapaxes(a=y, axis1=axis, axis2=-1)


def fftfreq(f0, f1, M):
    """zfftfreq(f0, f1, M) - Return frequency values of the zoom FFT
    coefficients returned by zfft().

    Keyword arguments:
    f0 - lower bound of frequency bandwidth
    f1 - upper bound of frequency bandwidth
    fs = sampling rate

    Return values:
    freq - vector of frequency values

    """

    df = (f1 - f0) / (M - 1)
    return np.arange(M) * df + f0


def chirpz_original(x,A,W,M):
    """Unmodified Chirp z-Transform from web address listed below.

    From http://www.mail-archive.com/numpy-discussion@scipy.org/msg01812.html
    Last accessed December-06-2012

    As described in
    Rabiner, L.R., R.W. Schafer and C.M. Rader.
    The Chirp z-Transform Algorithm.
    IEEE Transactions on Audio and Electroacoustics, AU-17(2):86--92, 1969

    Compute the chirp z-transform.
    The discrete z-transform,
    X(z) = \sum_{n=0}^{N-1} x_n z^{-n}
    is calculated at M points,
    z_k = AW^-k, k = 0,1,...,M-1
    for A and W complex, which gives
    X(z_k) = \sum_{n=0}^{N-1} x_n z_k^{-n}

    """
    A = np.complex(A)
    W = np.complex(W)
    if np.issubdtype(np.complex,x.dtype) or np.issubdtype(np.float,x.dtype):
        dtype = x.dtype
    else:
        dtype = float

    x = np.asarray(x,dtype=np.complex)

    N = x.size
    L = int(2**np.ceil(np.log2(M+N-1)))

    n = np.arange(N,dtype=float)
    y = np.power(A,-n) * np.power(W,n**2 / 2.) * x 
    Y = scipy.fftpack.fft(y,L)

    v = np.zeros(L,dtype=np.complex)
    v[:M] = np.power(W,-n[:M]**2/2.)
    v[L-N+1:] = np.power(W,-n[N-1:0:-1]**2/2.)
    V = scipy.fftpack.fft(v)

    g = scipy.fftpack.ifft(V*Y)[:M]
    k = np.arange(M)
    g *= np.power(W,k**2 / 2.)

    return g
