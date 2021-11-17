from numba.misc.special import prange
from numba.np.ufunc import parallel
import numpy as np
from PIL import Image
import sys
import wave
import math
from numba import njit, jit

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

@njit(parallel = True)
def dct1d(x: np.array) -> np.array:
    N = len(x)
    ks = list(range(N))
    c_k = [(0.5)**0.5 if k == 0 else 1 for k in range(N)]
    f_k = [k/(2*N) for k in range(N)]
    theta_k = [(k*math.pi)/(2*N) for k in range(N)]
    X = np.zeros(N)
    constant_part = (2/N)**0.5
    # for k in ks:
    for i in prange(N):
        sum = 0
        for n in range(N):
            sum += x[n]*math.cos(2*math.pi*n*f_k[ks[i]] + theta_k[ks[i]])
            # sum += x[n]*math.cos(2*math.pi*n*f_k[k] + theta_k[k])
            
        # X[k] = constant_part * c_k[k]*sum
        X[ks[i]] = constant_part * c_k[ks[i]]*sum
    return X

def idct(x: np.array) -> np.array:
    assert False, "TODO: implement iDCT"

def generate_cosine_2d_arr_vertical(w: int, h:int) -> np.ndarray:
    arr = np.zeros((w,h))
    # step = 2*math.pi/w
    # print(step)
    for j in range(w):
        arr[:,j] = np.cos(np.linspace(0, 32*math.pi, num=h).astype(np.float64))
    return arr

def generate_cosine_2d_arr_horizontal(w: int, h:int) -> np.ndarray:
    arr = np.zeros((w,h))
    # step = 2*math.pi/w
    # print(step)
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
    return 

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


def debug_dct():
    w:int = 256
    h:int = 256
    cos_arr_h =  generate_cosine_2d_arr_horizontal(w,h)
    cos_arr_v = generate_cosine_2d_arr_vertical(w,h)
    test_arr = cos_arr_v*cos_arr_h*255
    test_arr *= (255.0/test_arr.max())
    test_image_before = Image.fromarray(test_arr)
    test_image_before.show()
    test_arr_in_freq_domain  = np.zeros((w,h))
    for i in range(w):
        test_arr_in_freq_domain[i,:] = dct1d(test_arr[i,:])
    for j in range(h):
        test_arr_in_freq_domain[:, j] = dct1d(test_arr_in_freq_domain[:,j])
    test_arr_in_freq_domain *= (255.0/test_arr_in_freq_domain.max())
    test_image_after = Image.fromarray(test_arr_in_freq_domain)
    test_image_after.show()
    print(test_arr_in_freq_domain)
    
def run_dct_path_image(path):
    image_input_arr = get_image_as_arr(path)
    image_frequency_domain = np.zeros((256,256))
    
    input_image = get_image_as_pil_image(path)
    input_image.show()
    for i in range(256):
        image_frequency_domain[i, :] = dct1d(image_input_arr[i, :])
    for j in range(256):
        image_frequency_domain[:, j] = dct1d(image_frequency_domain[:, j])
    # image_frequency_domain = 255*np.abs(image_frequency_domain)
    image_frequency_domain = np.log(np.abs(image_frequency_domain)) + 1
    image_frequency_domain *= (255.0/image_frequency_domain.max())
    print(image_frequency_domain)
    # dct_image = Image.fromarray(image_frequency_domain + 1)
    dct_image = Image.fromarray(image_frequency_domain)
    dct_image.show()

def main(argc, argv) -> int:
    assert argc >= 2, "Arguments should be greater than 2"
    run_dct_path_image(argv[argc-1])
    debug_dct()
    return 0




if __name__ == "__main__":
    main(len(sys.argv), sys.argv)