# don't try to understand it without the base of musical theory

import pyaudio
import numpy as np
import time
import matplotlib.pyplot as plt

CHUNK = 44100               # each buffer will contain 'CHUNK' samples = bytes
                            # how much audio will be processed at a time
                            # how many samples/frame we're gonna display

FORMAT =  pyaudio.paInt16   # bit depth (usually 16/24 bits) bytes/sample
CHANNELS = 1                # amount of input channels
RATE = 44100                # samples/second


# Determine notes range (scale), max octave - 9
fst = 'A0' # first note
lst = 'F5' # last note

notes = ('C C# D D# E F F# G G# A A# B').split()
all_notes = []  # array which will contain elements from fst to lst including octaves numbers
freq = []       # array which will contain frequencies of our notes

def allnotes(low,high):
    l = (notes.index(low[0]), int(low[1]))
    h = (notes.index(high[0]), int(high[1]))
    delta = 12 * (h[1] - l[1]) + h[0] - l[0] + 1
    
    global all_notes
    for i in range(delta):
        all_notes.append(notes[(i + l[0]) % 12] + str((i + l[0] + l[1] * 12) // 12)) # array of notes with octaves
    
    global freq
    for i in range(12*(l[1]-4)+(l[0]-9), 12*(l[1]-4)+(l[0]-9)+delta): # 4 & 9 - notes[9] - A, 4 - octave 
        f = 440 * 2 ** (i/12) # 440 Htz = [A4] is the reference frequency.
        freq = np.append(freq,f)
        
allnotes(fst,lst)
    
print(all_notes)
print(freq)

for i, element in enumerate(all_notes):
    if i % 2 == 0:
        all_notes[i] = '' # to make the plot more readable by missing a half elements

# instantiate PyAudio
p = pyaudio.PyAudio()

# stream object to get data from the outside
stream = p.open(
    format = FORMAT,
    channels = CHANNELS,
    rate = RATE,
    input = True,
    output = True,
    frames_per_buffer = CHUNK,
    input_device_index = 0
)

window = np.hanning(CHUNK) # it applies for accurate frequency distribution of spectrum
x_fft = np.around(np.linspace(0, RATE, CHUNK),1)[0:int(freq[-1])+2] # +2 instead of +1 - is stock
data = np.zeros(CHUNK, dtype=np.float32)

start_time = time.time() # for an average time of iterations checking

count = 0 # for limited plotting

while stream.is_active():
   
    data = stream.read(CHUNK, exception_on_overflow=False) # bytes
    data_int = (np.fromstring(data, dtype = np.int16)) * window
    y_fft = np.append(np.zeros(27), np.abs(np.fft.fft(data_int)[27:int(freq[-1])+2] / (128 * CHUNK)))
    
    Array_of_freqs = x_fft[np.argwhere(y_fft > 1.5)] # all freqs in Hz under condition
    note = [] # final filtered notes
    
    for i in range(0,len(Array_of_freqs)):
        a = np.abs(freq-Array_of_freqs[i]).argmin()
        note.append(notes[a % 12] + str((a+9) // 12))
    print(note) 
    
    count += 1   
    if count == 2:
        break

rate = count / (time.time() - start_time)
print(rate)

# static plot option

fig, ax = plt.subplots(figsize = (15.4,10))

ax.semilogx(x_fft, y_fft)
ax.plot(np.argwhere(y_fft > 2.5), y_fft[np.argwhere(y_fft > 2.5)],'.')

for b in freq:
    ax.axvline(x=b, linewidth = 0.5)
    
ax.set_xticks(freq)
ax.set_xticklabels(all_notes)
ax.set_xlim(freq[0],freq[-1])

if max(y_fft) >= 2:
    ax.set_ylim(0, max(y_fft) * 1.1)
else:
    ax.set_ylim(0, 4)
    
fig.tight_layout()
plt.show()
