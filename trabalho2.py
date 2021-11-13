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

def dct(x: np.array) -> np.array:
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

def main(argc, argv):
    image_input_arr = get_image_as_arr(argv[argc-1])
    image_frequency_domain = np.ndarray((256,256))
    for i in range(256):
        # TODO: test if dct works
        image_frequency_domain[i] = dct(image_input_arr[i])
    for j in range(256):
        image_frequency_domain[:,j] = dct(image_input_arr[:,j])
    # NOTE: working? how i normalize the image???
    dct_image = Image.fromarray(np.abs(image_frequency_domain)*255)
    # FIXME: Currently not working
    dct_image.show()


if __name__ == "__main__":
    main(len(sys.argv), sys.argv)