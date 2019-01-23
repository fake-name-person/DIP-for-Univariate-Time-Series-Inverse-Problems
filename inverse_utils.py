import random
import torch
import numpy as np
import math
import scipy.fftpack as spfft
import wavio
from sklearn.linear_model import Lasso
import scipy.io
from scipy.signal import chirp
import collections
import pandas as pd


def read_wav(filename):
    """
    Reads a wav file and returns the associated data

    Parameters
    ----------
    filename : string
        The name of the wav file you want to read

    Returns
    -------
    rate
        The sampling rate of the wav
    length
        The number of samples in each channel
    resolution
        The number of bytes per sample
    nc
        The number of sound channels
    x
        [length, nc]-dimension array containing the wav data
    """

    wave = wavio.read(filename)
    rate = wave.rate
    length = wave.data.shape[0]
    resolution = wave.sampwidth
    nc = wave.data.shape[1]
    x = wave.data

    return [rate, length, resolution, nc, x]

#Use to normalize Audio signals - bits is the current bit resolution of the signal
def audio_normalise(x, bits):
    """
    Normalizes an input audio array to have range [-1, 1]

    Parameters
    ----------
    x : int or double array
        The array of data to normalise
    bits: int
        The bit resolution of x. We use 2^(bits - 1) to normalise range

    Returns
    -------
    x_normed
        The data, range-normalised to [-1, 1]
    """

    return x/(2**(bits-1))

    #mu = np.mean(np.squeeze(x))
    #sigma = np.std(np.squeeze(x))

    #return (x - mu)/sigma

#Use to normalize air quality data
def normalise(x):
    x0 = np.squeeze(x)
    maxi = np.amax(x0)
    mini = np.amin(x0)
    ranges = maxi-mini
    y = 2*(x - mini)/ranges - 1
    return y

#Renormalises array to have +/- 2^(bits-1) range
def renormalise(x, bits):

    return x*(2**(bits-1))

#Generates a size [num_samples, nc] array of Gaussian noise from N(0, std^2)
def get_noise(num_samples = 16384, nc = 1, std = 0.1):
    return (std * np.random.randn(num_samples, nc))

#Generates the sampling matrix phi (for DIP) and measurement matrix A = phi*psi (where psi is the IDCT matrix for sparse reconstruction, e.g. Lasso)
def get_A(case, num_measurements = 1000, original_length = 16384):

    if case == 'Imputation':
        kept_samples = random.sample(range(0, original_length), num_measurements)

        A = spfft.idct(np.identity(original_length), norm='ortho', axis=0)[kept_samples, :]
        phi = np.eye(original_length)[kept_samples, :]

        return [phi, A, kept_samples]

    if case =='CS':
        phi = (1 / math.sqrt(1.0 * num_measurements)) * np.random.randn(num_measurements, original_length)
        A = np.matmul(phi, spfft.idct(np.identity(original_length), norm='ortho', axis=0))

        return [phi, A]

    if case == 'Denoising':
        phi = np.eye(original_length)
        A = spfft.idct(np.identity(original_length), norm='ortho', axis=0)

        return [phi, A]

    if case == 'DCT':
        kept_samples = random.sample(range(0, original_length), num_measurements)

        phi = (spfft.dct(np.eye(original_length), norm='ortho').transpose())[kept_samples, :] #transpose the output because scipy produces transposed DCT matrix
        A = np.matmul(phi, spfft.idct(np.identity(original_length), norm='ortho', axis=0))

        return [phi, A]

    else:
        print("WRONG INPUT TO get_A: INVALID CASE. TRY AGAIN")

        exit(0)

#Run Lasso reconstruction given measurement matrix A and observed measurements y
def run_Lasso(A, y, output_size = 16834, alpha = 1e-5):
    lasso = Lasso(alpha=alpha)
    lasso.fit(A, y)

    x_hat = np.array(lasso.coef_).reshape(output_size)
    x_hat = spfft.idct(x_hat, norm='ortho', axis=0)
    x_hat = x_hat.reshape(-1, 1)

    return x_hat

def save_matrix(mat, filename):
    scipy.io.savemat(filename, mdict={'A': mat})

def save_data(data, filename):
    scipy.io.savemat(filename, mdict={'x': data})

#Use to save test results
def save_log(data, test, method, results, filename):
    if "Denoising" in test: #if denoising, then data is a single point
        text = data + "\n" + test + "\n" + method + "\n" + str(results)

        file = open(filename, "w")
        file.write(text)
        file.close()

    else:
        len = results.shape[0]

        text = data + "\n" + test + "\n" + method + "\n"
        for i in range(len):
            text = text + str(results[i,0]) + " " + str(results[i,1])

            if i < len-1:
                text = text + "\n"

        file = open(filename, "w")
        file.write(text)
        file.close()

#Use to read test results
def read_log(filename):
    f = open(filename, 'r')
    lines = f.read().splitlines()
    f.close()

    test = lines[1]
    if "Denoising" in test:
        return float(lines[3].strip("\r\n\t '"))

    else:
        length = len(lines) - 3
        parsed = np.zeros((length, 2))

        for i in range(length):
            numbers_str = lines[i+3].strip("\r\n\t").split()
            numbers_float = [float(x) for x in numbers_str]

            parsed[i,0] = numbers_float[0]
            parsed[i,1] = numbers_float[1]

        return parsed

def get_chirp(length, fstart, fend):
    t = np.linspace(0, 2, length)
    x0 = chirp(t, f0 = fstart, f1=fend, t1=2, method='linear')
    x = np.zeros((length, 1))
    x[:,0] = x0
    return x

#Use to read air quality data
def get_air_data(loc = "/home/sravula/AirQualityUCI/AirQuality.csv", data = "O3-1", length = 1024):
    x = pd.read_csv(loc)
    one = x[data].dropna().values[0:length]
    return one

