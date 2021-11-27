from numba.misc.special import prange
import numpy as np
from PIL import Image
import sys
import wave
import math
from numba import njit, jit
import matplotlib.pyplot as plt
import typing
from progress.bar import Bar

def get_image_as_arr(path) -> np.ndarray:
    image_input = Image.open(path)
    return np.array(image_input)

def get_image_as_pil_image(path) -> Image:
    return Image.open(path)

def get_wav_as_arr(path, normalised=True) -> np.array:
    ifile = wave.open(path)
    samples = ifile.getnframes()
    audio = ifile.readframes(samples)

    # Convert buffer to float64 using NumPy                                                                                 
    audio_as_np_int16 = np.frombuffer(audio, dtype=np.int16)
    audio_as_np_float64 = audio_as_np_int16.astype(np.float64)

    # Normalise float64 array so that values are between -1.0 and +1.0                                                      
    max_int16 = 2**15
    if(normalised):
        audio_normalised = audio_as_np_float64 / max_int16
        return audio_normalised
    return audio_as_np_float64

def write_wav_from_arr(new_data: np.array, new_path: str, original_path: str):
    unormalised_new_data = new_data * 2**15
    unormalised_new_data = unormalised_new_data.astype(np.int16)
    new_file = wave.open(new_path, 'wb')
    original_file = wave.open(original_path)
    # original_data = get_wav_as_arr(original_path)
    # print("""Writing File: {} 
    # \t- Data Size: {}
    # \t- Nframes: {}
    # \t- Nchannels: {}
    # \t- Width: {}
    # \t- Framerate: {}""".format(
    #     original_path,
    #     len(original_data),
    #     original_file.getnframes(),
    #     original_file.getnchannels(),
    #     original_file.getsampwidth(),
    #     original_file.getframerate()
    # ))
    new_file.setnchannels(original_file.getnchannels())
    new_file.setsampwidth(original_file.getsampwidth())
    new_file.setframerate(original_file.getframerate())
    new_file.setnframes(len(unormalised_new_data))
    new_file.writeframesraw(unormalised_new_data)
    # print("""Writing File: {} 
    # \t- Data Size: {}
    # \t- Nframes: {}
    # \t- Nchannels: {}
    # \t- Width: {}
    # \t- Framerate: {}""".format(
    #     new_path,
    #     len(unormalised_new_data),
    #     new_file.getnframes(),
    #     new_file.getnchannels(),
    #     new_file.getsampwidth(),
    #     new_file.getframerate()
    # ))
    return

@jit
def bass_boost(K: int ) -> float:
    degree = 6
    fc = 25000
    g = 0.2
    Y:float = g / (math.sqrt(1 + math.pow(K/fc, 2 * degree))) + 1 
    newK:float = K * Y
    return newK

@jit
def low_pass_filter(K: int) -> float:
    if K < 12_520:
        return K
    return 0

@jit
def func_identity(x: int) -> float:
    return x

@njit(parallel = True)
def dct1d(x: np.array, k_function: typing.Callable = func_identity) -> np.array:
    N = len(x)
    ks = list(range(N))
    c = [(0.5)**0.5 if k == 0 else 1 for k in range(N)]
    f = [k_function(k)/(2.0*N) for k in ks]
    theta_k = [(k_function(k)*math.pi)/(2.0*N) for k in ks]
    X = np.zeros(N)
    constant_part = (2/N)**0.5
    for i in prange(N):
        sum = 0
        for n in range(N):
            sum += x[n]*math.cos(2*math.pi*n*f[ks[i]] + theta_k[ks[i]])
            
        X[ks[i]] = constant_part * c[ks[i]] * sum
    return X

@njit(parallel = True)
def idct1d(X: np.array, k_function: typing.Callable = func_identity) -> np.array:
    N = len(X)
    ns = list(range(N))
    c = [(0.5)**0.5 if k == 0 else 1 for k in range(N)]
    f = [k_function(k)/(2.0*N) for k in range(N)]
    theta_k = [(k_function(k)*math.pi)/(2.0*N) for k in range(N)]
    x = np.zeros(N)
    constant_part = (2/N)**0.5
    for i in prange(N):
        sum = 0
        for k in range(N):
            sum += c[k]*X[k]*math.cos(2*math.pi*ns[i]*f[k] + theta_k[k])
            
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
    progress_bar = Bar("\tDCT2D", max=w+h,suffix = '%(percent).1f%% - %(elapsed)ds')
    for i in range(w):
        X_arr_in_freq_domain[i,:] = dct1d(x[i,:])
        progress_bar.next()
    for j in range(h):
        X_arr_in_freq_domain[:,j] = dct1d(X_arr_in_freq_domain[:,j])
        progress_bar.next()
    print("")
    return X_arr_in_freq_domain

def idct2d(X: np.ndarray) -> np.ndarray:
    w,h = X.shape
    x_arr_in_freq_domain  = np.zeros((X.shape))
    progress_bar = Bar("\tiDCT2D", max=w+h,suffix = '%(percent).1f%% - %(elapsed)ds')
    for i in range(w):
        x_arr_in_freq_domain[i,:] = idct1d(X[i,:])
        progress_bar.next()
    for j in range(h):
        x_arr_in_freq_domain[:,j] = idct1d(x_arr_in_freq_domain[:,j])
        progress_bar.next()
    print("")
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
    # dct_image.show()
    saveble_image = dct_image.convert("P")
    saveble_image.save(open("dct.png", "wb"), "PNG")

    image_time_domain = np.zeros((256,256))
    output_arr = np.array(dct_image)
    image_time_domain = idct2d(output_arr)
    image_time_domain_normalized = normalize255(image_time_domain)
    dct_image = Image.fromarray(image_time_domain_normalized)
    # dct_image.show()
    saveble_image = dct_image.convert("P")
    saveble_image.save(open("idct.png", "wb"), "PNG")


def save_image(image: Image, title):
    saveble_image = image.convert("P")
    saveble_image.save(open(title, "wb"), "PNG", compress_level=0)
    pass

def topic_1(path, n=100):
    print("[1] Running Topic 1:")
    print("\t[1.1] Doing normalized DCT without DC component:")

    image_input_arr = get_image_as_arr(path)
    image_frequency_domain = np.zeros(image_input_arr.shape)
    image_frequency_domain = dct2d(image_input_arr)
    save_image(Image.fromarray(image_frequency_domain), "output/dct_lena.png")

    image_without_dc = image_frequency_domain.copy()
    image_without_dc[0][0] = 0
    image_normalized_without_dc = normalize255(image_without_dc)
    dct_image = Image.fromarray(image_normalized_without_dc)
    print("\t DC Component value:", image_frequency_domain[0][0])
    save_image(dct_image, "output/dct_of_lena_normalized_without_dc.png")
    
    lena_without_dc = np.zeros(image_input_arr.shape)
    output_arr = np.array(dct_image)
    lena_without_dc = idct2d(output_arr)
    lena_without_dc_normalized = normalize255(lena_without_dc)
    image_lena_normalized_without_dc = Image.fromarray(lena_without_dc_normalized)
    save_image(image_lena_normalized_without_dc, "output/lena_normalized_without_dc.png")
    
    print("\t[1.2] Making an approximation using only the n largest coefficients. n =", n)
    image_approximation = image_frequency_domain.flatten()
    image_approximation_without_dc = image_approximation[1:]
    indexes_to_be_zeroed = np.abs(image_approximation_without_dc).argsort()[:len(image_approximation)-n]
    image_approximation_without_dc[indexes_to_be_zeroed] = 0
    image_approximation[1:] = image_approximation_without_dc
    image_approximation = image_approximation.reshape(image_input_arr.shape)
    image_approximation_normalized = normalize255(image_approximation)
    dct_image_aprox = Image.fromarray(image_approximation_normalized)
    save_image(dct_image_aprox, "output/dct_lena_aprox.png")

    output_arr = np.array(dct_image_aprox)
    image_aprox_time_domain = idct2d(output_arr)
    image_aprox_time_domain_normalized = normalize255(image_aprox_time_domain)
    dct_image_aprox_output = Image.fromarray(image_aprox_time_domain_normalized)
    save_image(dct_image_aprox_output, "output/lena_aprox_output.png")

def topic_2(path):
    print("[2] Running Topic 2:")
    progress_bar = Bar("\tApplying filter to audio", max=4,suffix = '%(percent).1f%%')
    print("")
    wav_as_arr = get_wav_as_arr(path)
    progress_bar.next()
    filtered_wav_frequency_domain = dct1d(wav_as_arr, bass_boost)
    progress_bar.next()
    filtered_wav_time_domain = idct1d(filtered_wav_frequency_domain)
    progress_bar.next()
    wav_frequency_domain = dct1d(wav_as_arr)
    progress_bar.next()

    progress_bar = Bar("\tSaving Image", max=1,suffix = '%(percent).1f%%')
    print("")
    fig, axs = plt.subplots(2,1)
    axs[0].plot(wav_frequency_domain)
    axs[1].plot(filtered_wav_frequency_domain)
    plt.savefig("output/wav_compare.png", format="png")
    progress_bar.next()

    progress_bar = Bar("\tSaving result", max=1,suffix = '%(percent).1f%%')
    print("")
    write_wav_from_arr(filtered_wav_time_domain, "output/result.wav", path)
    progress_bar.next()
    print("")

def main(argc, argv) -> int:
    assert argc >= 2, "Arguments should be greater than 2"
    topic_1(argv[argc-3], int(argv[argc-2]))
    topic_2(argv[argc-1])
    # debug_dct()
    return 0

if __name__ == "__main__":
    main(len(sys.argv), sys.argv)