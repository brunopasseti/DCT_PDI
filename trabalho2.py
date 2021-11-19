from numba.misc.special import prange
import numpy as np
from PIL import Image
import sys
import wave
import math
from numba import njit, jit
from numpy.lib.type_check import imag
import matplotlib.pyplot as plt
import typing

def get_image_as_arr(path) -> np.ndarray:
    image_input = Image.open(path)
    return np.array(image_input)

def get_image_as_pil_image(path) -> Image:
    return Image.open(path)

def get_wav_as_arr(path, normalised=True) -> np.array:
    ifile = wave.open(path)
    samples = ifile.getnframes()
    audio = ifile.readframes(samples)

    # Convert buffer to float32 using NumPy                                                                                 
    audio_as_np_int16 = np.frombuffer(audio, dtype=np.int16)
    audio_as_np_float32 = audio_as_np_int16.astype(np.float32)

    # Normalise float32 array so that values are between -1.0 and +1.0                                                      
    max_int16 = 2**15
    if(normalised):
        audio_normalised = audio_as_np_float32 / max_int16
        return audio_normalised
    return audio_as_np_float32

@njit
def bass_filter(K):
    degree = 2
    fc = 100
    g = 0.5
    Y = g * 1 / (math.sqrt(1 + pow(K/fc, 2 * degree))) + 1

    newK = K * Y

    # print('old K: ', K, ' | new K: ', newK, ' | Y: ', Y)
    return newK

@njit(parallel = True)
def dct1d_bass_bosting(x: np.array) -> np.array:
    N = len(x)
    ks = list(range(N))
    c_k = [(0.5)**0.5 if k == 0 else 1 for k in range(N)]
    f_k = [bass_filter(k)/(2*N) for k in range(N)]
    theta_k = [(bass_filter(k)*math.pi)/(2*N) for k in range(N)]
    X = np.zeros(N)
    constant_part = (2/N)**0.5

    for i in prange(N):
        sum = 0
        for n in range(N):
            sum += x[n] * math.cos(2 * math.pi * n * f_k[ks[i]] + theta_k[ks[i]])
            
        X[ks[i]] = constant_part * c_k[ks[i]] * sum
    return X

@njit(parallel = True)
def dct1d(x: np.array, k_function: typing.Callable[[int],float] ) -> np.array:
    N: int = len(x)
    ks = list(range(N))
    c = [(0.5)**0.5 if k == 0 else 1 for k in range(N)]
    f = [k_function(k)/(2*N) for k in range(N)]
    theta_k = [(k_function(k)*math.pi)/(2*N) for k in range(N)]
    X = np.zeros(N)
    constant_part = (2/N)**0.5
    for i in prange(N):
        sum = 0
        for n in range(N):
            sum += x[n]*math.cos(2*math.pi*n*f[ks[i]] + theta_k[ks[i]])
            
        X[ks[i]] = constant_part * c[ks[i]] * sum
    return X

@njit(parallel = True)
def idct1d(X: np.array) -> np.array:
    N = len(X)
    ns = list(range(N))
    c_k = [(0.5)**0.5 if k == 0 else 1 for k in range(N)]
    f_k = [k/(2*N) for k in range(N)]
    theta_k = [(k*math.pi)/(2*N) for k in range(N)]
    x = np.zeros(N)
    constant_part = (2/N)**0.5
    for i in prange(N):
        sum = 0
        for k in range(N):
            sum += c_k[k]*X[k]*math.cos(2*math.pi*ns[i]*f_k[k] + theta_k[k])
            
        x[ns[i]] = constant_part * sum
    return x

def generate_cosine_2d_arr_vertical(w: int, h:int) -> np.ndarray:
    arr = np.zeros((w,h))
    for j in range(w):
        arr[:,j] = np.cos(np.linspace(0, 32*math.pi, num=h).astype(np.float64))
    return arr

def generate_cosine_2d_arr_horizontal(w: int, h:int) -> np.ndarray:
    arr = np.zeros((w,h))
    for i in range(h):
        arr[i,:] = np.cos(np.linspace(0, 32*math.pi, num=w).astype(np.float64))
    return arr

def dct2d(x: np.ndarray) -> np.ndarray:
    w,h = x.shape
    X_arr_in_freq_domain  = np.zeros((x.shape))
    for i in range(w):
        X_arr_in_freq_domain[i,:] = dct1d(x[i,:])
    for j in range(h):
        X_arr_in_freq_domain[:,j] = dct1d(X_arr_in_freq_domain[:,j])
    return X_arr_in_freq_domain

def idct2d(X: np.ndarray) -> np.ndarray:
    w,h = X.shape
    x_arr_in_freq_domain  = np.zeros((X.shape))
    for i in range(w):
        x_arr_in_freq_domain[i,:] = idct1d(X[i,:])
    for j in range(h):
        x_arr_in_freq_domain[:,j] = idct1d(x_arr_in_freq_domain[:,j])
    return x_arr_in_freq_domain

def new_dct(x):
    N = len(x)
    ks = list(range(N))
    X = np.zeros(N)
    constant_part = 2
    for k in ks:
        sum = 0
        for n in range(N):
            sum += x[n]*math.cos((math.pi*k*((2*n) + 1))/(2*N))
        X[k] = constant_part*sum
    c_k = [(1/(4*N))**0.5 if k == 0 else (1/(2*N))**0.5 for k in range(N)]
    return X

def normalize255(arr: np.ndarray):
    return arr*(255.0/arr.max())

def debug_dct():
    w:int = 256
    h:int = 256
    cos_arr_h =  generate_cosine_2d_arr_horizontal(w,h)
    cos_arr_v = generate_cosine_2d_arr_vertical(w,h)
    test_arr = cos_arr_v*cos_arr_h*255
    test_arr *= (255.0/test_arr.max())
    print(test_arr)
    test_image_before = Image.fromarray(test_arr)
    test_image_before.show()
    test_array_frequency_domain = np.zeros((256,256))
    test_array_frequency_domain = dct2d(test_arr)
    test_array_frequency_domain_normalized = normalize255(test_array_frequency_domain)
    test_image_after = Image.fromarray(test_array_frequency_domain_normalized)
    test_image_after.show()
    print(test_array_frequency_domain_normalized)

def run_dct_path_image(path):
    input_image = get_image_as_pil_image(path)
    input_image.show()

    image_input_arr = get_image_as_arr(path)
    image_frequency_domain = np.zeros((256,256))
    image_frequency_domain = dct2d(image_input_arr)
    # image_frequency_domain = np.log(np.abs(image_frequency_domain)+ 1) 
    image_frequency_domain_normalized = normalize255(image_frequency_domain)
    dct_image = Image.fromarray(image_frequency_domain_normalized)
    dct_image.show()
    saveble_image = dct_image.convert("P")
    saveble_image.save(open("dct.png", "wb"), "PNG")

    image_time_domain = np.zeros((256,256))
    output_arr = np.array(dct_image)
    image_time_domain = idct2d(output_arr)
    image_time_domain_normalized = normalize255(image_time_domain)
    dct_image = Image.fromarray(image_time_domain_normalized)
    dct_image.show()
    saveble_image = dct_image.convert("P")
    saveble_image.save(open("idct.png", "wb"), "PNG")


def main(argc, argv) -> int:
    assert argc >= 2, "Arguments should be greater than 2"
    # run_dct_path_image(argv[argc-1])
    wav_as_arr = get_wav_as_arr(argv[argc-1])
    wav_after_dct_filter = dct1d(wav_as_arr, bass_filter)
    wav_after_dct = dct1d(wav_as_arr,  lambda x: x)
    plt.plot(wav_after_dct_filter)
    plt.plot(wav_after_dct)
    plt.show()
    # debug_dct()
    return 0

if __name__ == "__main__":
    main(len(sys.argv), sys.argv)