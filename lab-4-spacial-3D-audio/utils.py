import numpy as np
import matplotlib.pyplot as plt

# Load a remote WAVE file given its URL, and return the sample rate and waveform
def wavreadurl( url):
    import urllib.request, io, scipy.io.wavfile
    f = urllib.request.urlopen( urllib.request.Request( url))
    sr,s = scipy.io.wavfile.read( io.BytesIO( f.read()))
    return sr, s.astype( 'float32')/32768

# Load a local WAVE file given its filepath, and return the sample rate and waveform
def wavreadfile(filename):
    import scipy.io.wavfile
    sr, s =  scipy.io.wavfile.read(filename)
    return sr, s.astype( 'float32')/32768

# Make a sound player function that plays array "x" with a sample rate "rate", and labels it with "label"
def sound( x, rate=8000, label=''):
    from IPython.display import display, Audio, HTML
    display( HTML( 
    '<style> table, th, td {border: 0px; }</style> <table><tr><td>' + label + 
    '</td><td>' + Audio( x, rate=rate)._repr_html_()[3:] + '</td></tr></table>'
    ))

# Convert input sound to time/frequency domain for spectrogram plotting
def stft(input_sound, dft_size, hop_size, window):
    # get windowed segments of input sound and compute DFT of each
    spectrogram = []
    n = len(input_sound)

    for i in range(0, n - dft_size - 1, hop_size):
        #  get frame, padding with zeros if necessary
        frame = []
        if i + dft_size < n:
            frame = input_sound[i:i+dft_size]
        else:
            frame = np.append(input_sound[i:n], np.zeros((i + dft_size) - n))

        # compute DFT
        frame_dft = np.fft.rfft(frame * window)

        # append to spectrogram array
        spectrogram.append(frame_dft) # append as rows

    # Return a complex-valued spectrogram (frequencies x time)
    return np.array(spectrogram).T

# Convert time/frequency domain sound back into time-domain waveform
def istft(stft_output, dft_size, hop_size, window):
    num_frames = np.shape(stft_output)[1]
    n = dft_size + ((num_frames - 1) * hop_size) 
    output_sound = np.zeros(n)

    for i in range(num_frames):
        # compute inverse FFT for current frame
        frame_idft = np.fft.irfft(stft_output[:,i], n=dft_size)
        frame = frame_idft * np.sqrt(window)

        # add result to output_sound at correct indices
        start_idx = i * hop_size
        output_sound[start_idx : start_idx + dft_size] += frame
    
    # Return reconstructed waveform
    return output_sound
    
# Plot the given spectrogram using numpy.pcolormesh
def plot_spectrogram(spectrogram, dft_size, hop_size, sample_rate, label, cmap="viridis"):
    # calculate the time axis labels
    num_frames = np.shape(spectrogram)[1]
    samples = (num_frames * hop_size) + dft_size
    times = np.round(np.linspace(0, samples / sample_rate, num=5), 2)

    # calculate the frequency axis labels
    max_freq = sample_rate / 2
    freqs = np.linspace(0, sample_rate / 2, num=6)

    # plot the spectrogram
    plt.pcolormesh(np.abs(spectrogram)**0.2, cmap=cmap) # np.abs gives magnitude of complex number
    plt.xticks(ticks=np.linspace(0, spectrogram.shape[1], len(times)), labels=times)
    plt.yticks(np.linspace(0, spectrogram.shape[0], len(freqs)), labels=freqs)
    plt.title("Spectrogram for " + label + " Sound")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency (Hz)")