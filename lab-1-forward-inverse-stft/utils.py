import numpy as np
import matplotlib.pyplot as plt

# Load a remote WAVE file given its URL, and return the sample rate and waveform
def wavreadurl( url):
    import urllib.request, io, scipy.io.wavfile
    f = urllib.request.urlopen( urllib.request.Request( url))
    sr,s = scipy.io.wavfile.read( io.BytesIO( f.read()))
    return sr, s.astype( 'float32')/32768

# Make a sound player function that plays array "x" with a sample rate "rate", and labels it with "label"
def sound( x, rate=8000, label=''):
    from IPython.display import display, Audio, HTML
    display( HTML( 
    '<style> table, th, td {border: 0px; }</style> <table><tr><td>' + label + 
    '</td><td>' + Audio( x, rate=rate)._repr_html_()[3:] + '</td></tr></table>'
    ))

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