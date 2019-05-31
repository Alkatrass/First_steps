import pyaudio
import numpy as np
import time
import matplotlib.pyplot as plt

CHUNK = 44100               # each buffer will contain 1024*2 samples = bytes
                            # how much audio will be processed at a time
                            # how many samples/frame we're gonna display

FORMAT =  pyaudio.paInt16   # bit depth (usually 16/24 bits) bytes/sample
CHANNELS = 1                
RATE = 44100                # samples/second

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

window = np.hanning(CHUNK)

x_fft = np.around(np.linspace(0, RATE, CHUNK),1)[0:CHUNK//14]

count = 0
freq = []

data = np.zeros(CHUNK, dtype=np.float32)

for i in range(-48, 36): #A0...G#7
        f = 440 * 2 ** (i/12)
        freq = np.append(freq,f)

notes = ('A A# B C C# D D# E F F# G G#').split()
'''
def allnotes(l,h):
    pass

print(allnotes(A0,E5))
'''

k = ['A0','','B0','','C#1','','D#1','','F1','','G1','','A1','','B1','','C#2','','D#2','','F2','','G2','','A3','','B3','','C#3','','D#3','','F3','','G3','','A3','','B3','','C#4','','D#4','','F4','','G4','','A4','','B4','','C#5','','D#5','']

start_time = time.time()

while stream.is_active():
   
    data = stream.read(CHUNK, exception_on_overflow=False)     # bytes
    data_int = (np.fromstring(data, dtype = np.int16)) * window
    y_fft = np.append(np.zeros(27), np.abs(np.fft.fft(data_int)[27:CHUNK//14] / (64 * 2 * CHUNK)))
    
    Array_of_freqs = x_fft[np.argwhere(y_fft > 1.5)]  # 2.5, max[i]/x  
    
    fin_arr = []
    note = []
    
    for i in range(0,len(Array_of_freqs)):
        a = np.abs(freq-Array_of_freqs[i]).argmin()
        fin_arr.append(round(freq[a],1))
        note.append(notes[a % 12] + str((a+9) // 12))
    
    print(note) # not only peaks
    
    count += 1   
    if count == 2:
        break

rate = count / (time.time() - start_time)
print(rate)

fig, ax = plt.subplots(figsize = (15.4,10))

ax.semilogx(x_fft, y_fft)
ax.plot(np.argwhere(y_fft > 2.5), y_fft[np.argwhere(y_fft > 2.5)],'.')

for b in freq:
    ax.axvline(x=b, linewidth = 0.5)
    
ax.set_xticks(freq[:55+1])
ax.set_xticklabels(k)
ax.set_xlim(freq[0],freq[55])

if max(y_fft) >= 2:
    ax.set_ylim(0, max(y_fft) * 1.1)
else:
    ax.set_ylim(0, 4)
    
fig.tight_layout()
plt.show()
