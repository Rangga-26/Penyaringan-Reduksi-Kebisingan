from obspy import read
import matplotlib.pyplot as plt

st = read("sumut_m56.mseed")  

print(st)
for tr in st:
    print(f"Channel: {tr.stats.channel}, Start: {tr.stats.starttime}, Sampling Rate: {tr.stats.sampling_rate}")
st.plot()
plt.figure()

for i, tr in enumerate(st):
    plt.subplot(len(st), 1, i+1)
    plt.plot(tr.times(), tr.data)
    plt.title(tr.stats.channel)

plt.tight_layout()
plt.show()