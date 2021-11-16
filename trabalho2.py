import numpy as np
from PIL import Image
import sys
import wave
import math

from numpy.core import numeric

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

def dct1d(x: np.array) -> np.array:
    N = len(x)
    ks = list(range(N))
    c_k = [1 if k == 0 else (0.5)**0.5 for k in range(N)]
    f_k = [k/(2*N) for k in range(N)]
    theta_k = [(k*math.pi)/(2*N) for k in range(N)]
    X = np.zeros(N)
    constant_part = (2/N)**0.5
    for k in ks:
        sum = 0
        for n in range(N):
            sum += x[n]*math.cos(2*math.pi*n*f_k[k] + theta_k[k])
            
        X[k] = constant_part * c_k[k]*sum
    return X

def idct(X: np.array) -> np.array:
    assert False, "TODO: implement iDCT"

def generate_cosine_2d_arr_vertical(w: int, h:int) -> np.ndarray:
    arr = np.zeros((w,h))
    # step = 2*math.pi/w
    # print(step)
    for j in range(h):
        arr[:,j] = np.cos(np.linspace(0, 32*math.pi, num=w))
    return arr

def generate_cosine_2d_arr_horizontal(w: int, h:int) -> np.ndarray:
    arr = np.zeros((w,h))
    # step = 2*math.pi/w
    # print(step)
    for i in range(h):
        arr[i,:] = np.cos(np.linspace(0, 32*math.pi, num=w))
    return arr

def dct2d(X: np.ndarray) -> np.ndarray:
    w,h = X.shape
    X_arr_in_freq_domain  = np.zeros((X.shape))
    for i in range(w):
        X_arr_in_freq_domain[i,:] = dct1d(X[i,:])
    for j in range(h):
        X_arr_in_freq_domain[:, j] = dct1d(X_arr_in_freq_domain[:,j])
    return 


def main(argc, argv) -> int:
    assert argc >= 2, "Arguments should be greater than 2" 
    # image_input_arr = get_image_as_arr(argv[argc-1])
    # image_frequency_domain = np.zeros((256,256))
    
    # for i in range(256):
    #     image_frequency_domain[i, :] = dct1d(image_input_arr[i, :])
    # for j in range(256):
    #     image_frequency_domain[:, j] = dct1d(image_frequency_domain[:, j])
    # image_frequency_domain *= (255.0/image_frequency_domain.max())
    # dct_image = Image.fromarray(np.log(image_frequency_domain)+1)
    # dct_image.show()

    w,h = 256,256
    test_arr = generate_cosine_2d_arr_horizontal(w,h)*generate_cosine_2d_arr_vertical(w,h)*255
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
    return 0




if __name__ == "__main__":
    main(len(sys.argv), sys.argv)