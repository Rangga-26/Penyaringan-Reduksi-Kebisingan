import os
from obspy import read

input_file = r"C:\Users\rangg\Downloads\jatim_m58.mseed"

if not os.path.isfile(input_file):
    print("File hilang lagi. Cek ulang.")
    exit()

print(f"Processing file: {input_file}")

base_dir = "raw"
st = read(input_file)

print(f"Total trace: {len(st)}")

count_saved = 0
count_skip = 0

for tr in st:
    try:
        if len(tr.data) == 0:
            count_skip += 1
            continue

        sta = tr.stats.station
        net = tr.stats.network
        ch_full = tr.stats.channel 
        comp = ch_full[-1]

        if comp not in ["Z", "N", "E"]:
            print(f"[SKIP] Channel tidak standar: {ch_full}")
            count_skip += 1
            continue
        folder_path = os.path.join(base_dir, sta, comp)
        os.makedirs(folder_path, exist_ok=True)
        start = tr.stats.starttime.strftime("%Y%m%dT%H%M%S")
        end = tr.stats.endtime.strftime("%Y%m%dT%H%M%S")

        filename = f"{net}_{sta}_{ch_full}_{start}_{end}.mseed"
        filepath = os.path.join(folder_path, filename)
        tr.write(filepath, format="MSEED")

        print(f"[OK] {filepath}")
        count_saved += 1

    except Exception as e:
        print(f"[ERROR TRACE] {e}")
        count_skip += 1
        continue

print("\n===== HASIL =====")
print(f"Berhasil disimpan : {count_saved}")
print(f"Di-skip           : {count_skip}")
print(f"Total             : {len(st)}")