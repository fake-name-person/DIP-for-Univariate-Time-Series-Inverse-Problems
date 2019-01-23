import numpy as np
import wavio
from scipy import signal
import matplotlib.pyplot as plt

OUTPUT_RATE = 8192
OUTPUT_LENGTH = 2
OUTPUT_CHANNELS = 1
OUTPUT_RES = 2

in_filename = "audio_data/clinton.wav" #The location of the file you wish to convert
out_filename = "audio_data/clinton_8192hz_2s.wav" #the save destination for your output

wav = wavio.read(in_filename)
rate = wav.rate
channels = wav.data.shape[1]
resolution = wav.sampwidth
duration = wav.data.shape[0]/rate

print("Sampling Rate: ", rate)
print("Num Channels: ", channels)
print("Resolution: ", resolution)
print("Length: ", duration)

output_samples = round(OUTPUT_RATE*duration)
resampled_wave = np.zeros((output_samples, OUTPUT_CHANNELS))

for i in range(OUTPUT_CHANNELS):
    resampled_wave[:, i] = signal.resample(x = wav.data[:, i], num = output_samples)

output_wave = np.zeros((OUTPUT_RATE*OUTPUT_LENGTH, OUTPUT_CHANNELS))

if (OUTPUT_RATE*OUTPUT_LENGTH <= resampled_wave.shape[0]):
    output_wave = resampled_wave[0:OUTPUT_RATE*OUTPUT_LENGTH, :]
else:
    output_wave = resampled_wave

x = output_wave/(2**(8*resolution -1))
spectrum =np.fft.fft(x[:,0], norm='ortho')
spectrum = abs(spectrum[0:round(len(spectrum)/2)]) # Just first half of the spectrum, as the second is the negative copy

# plt.figure()
# plt.plot(spectrum, 'r')
# plt.xlabel('Frequency (hz)')
# plt.title('Original Waveform')
# plt.xlim(0, OUTPUT_RATE/2)
# plt.show()
#
# plt.figure()
# plt.plot(range(OUTPUT_RATE*OUTPUT_LENGTH), output_wave[:,0])
# plt.show()

wavio.write(out_filename, output_wave, OUTPUT_RATE, sampwidth=OUTPUT_RES)


