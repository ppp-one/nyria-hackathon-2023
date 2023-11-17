from scipy.io import wavfile
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
matplotlib.use('Agg')

from Zernike import Zernike
import sys
sys.path.insert(0, "..")

def process_audio(file_selection: str, instant_length: int = None, instant_step: int = 1000, display: bool = False):

    input_wave_file = f"./{file_selection}.wav"
    
    # Read the audio file
    sample_rate, data = wavfile.read(input_wave_file)
    print(len(data), type(data), data.shape)

    if not instant_length:
        # Default to 1 second?
        instant_length = sample_rate

    for i in range(0, len(data), instant_step):
        # print(i)
        # t0 = time.time()
        # print(time.time() - t0)
        idata = data[i:i+instant_length]
        # print(idata[0:10])
        # print(time.time() - t0)
        fft_abs = np.abs(np.fft.fft(idata))
        # print(time.time() - t0)
        # fft_freq = np.fft.fftfreq(n, d=1/sample_rate)

        # Create vector of weights
        weights = fft_abs[:instant_length//2][:32]
        # print(time.time() - t0)

        z = Zernike(modes_num=len(weights), N_pupil=64)
        # print(time.time() - t0)
        wavefront = z.wavefrontFromModes(coefs_inp=weights)
        # print(time.time() - t0)

        # print(type(wavefront))
        # save numpy array to file
        np.save(f"./frames/{i}.npy", wavefront)
        

        # Generate image
        fig = plt.figure()

        ax = fig.gca()
        ax.imshow(wavefront)
        
        ax.set_xticks([])
        ax.set_yticks([])

        plt.savefig(f"./frames/{i}.png", dpi=50, format='png')
        plt.close(fig)

def process_audio_stream(data: np.array, instant_length: int = None, instant_step: int = 1000, display: bool = False):

    sample_rate = 44100

    if not instant_length:
        # Default to 1 second?
        instant_length = sample_rate

    fft_abs = np.abs(np.fft.fft(data))

    # Create vector of weights
    weights = fft_abs[:instant_length//2][:32]
    # print(time.time() - t0)

    z = Zernike(modes_num=len(weights), N_pupil=64)
    # print(time.time() - t0)
    wavefront = z.wavefrontFromModes(coefs_inp=weights)

    return wavefront

if __name__ == "__main__":

    process_audio(
        file_selection = "potato",
        instant_length = 5_000,
        instant_step = 1_000
        )
