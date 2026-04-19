import os
import pandas as pd
import numpy as np
from obspy import read, UTCDateTime
from obspy.taup import TauPyModel
from obspy.geodetics import locations2degrees
import glob

# ================= KONFIGURASI =================
# Pilih event yang akan diproses (ubah sesuai kebutuhan)
# EVENT_NAME = "aceh_m55"    # Untuk satu event
# EVENT_NAME = "jatim_m58"   # Untuk satu event
# Atau proses semua event:
PROCESS_ALL_EVENTS = True   # Set True untuk proses semua event

# ================= PARAMETER GEMPA =================
# GANTI DENGAN DATA SEBENARNYA!
EVENT_PARAMS = {
    'aceh_m55': {
        'origin_time': UTCDateTime("2021-12-21T12:29:50"),
        'latitude': 5.0,      # GANTI dengan koordinat gempa Aceh yang benar
        'longitude': 95.0,    # GANTI dengan koordinat gempa Aceh yang benar
        'depth_km': 30,
        'magnitude': 5.5
    },
    'jatim_m58': {
        'origin_time': UTCDateTime("2021-12-21T12:29:50"),  # GANTI dengan waktu gempa Jatim
        'latitude': -7.5,     # GANTI dengan koordinat gempa Jawa Timur
        'longitude': 112.5,   # GANTI dengan koordinat gempa Jawa Timur
        'depth_km': 30,
        'magnitude': 5.8
    },
    'papua_m59': {
        'origin_time': UTCDateTime("2021-12-21T12:29:50"),  # GANTI dengan waktu gempa Papua
        'latitude': -3.0,     # GANTI dengan koordinat gempa Papua
        'longitude': 138.0,   # GANTI dengan koordinat gempa Papua
        'depth_km': 30,
        'magnitude': 5.9
    },
    'sumbawa_m52': {
        'origin_time': UTCDateTime("2021-12-21T12:29:50"),  # GANTI dengan waktu gempa Sumbawa
        'latitude': -8.5,     # GANTI dengan koordinat gempa Sumbawa
        'longitude': 117.5,   # GANTI dengan koordinat gempa Sumbawa
        'depth_km': 30,
        'magnitude': 5.2
    },
    'sumut_m58': {
        'origin_time': UTCDateTime("2021-12-21T12:29:50"),  # GANTI dengan waktu gempa Sumut
        'latitude': 2.0,      # GANTI dengan koordinat gempa Sumatera Utara
        'longitude': 98.0,    # GANTI dengan koordinat gempa Sumatera Utara
        'depth_km': 30,
        'magnitude': 5.8
    }
}

# ================= KOORDINAT STASIUN =================
# GANTI DENGAN KOORDINAT SEBENARNYA!
STATION_COORDS = {
    'BKB': {'lat': 4.5, 'lon': 95.5, 'network': 'GE'},
    'BKNI': {'lat': 4.2, 'lon': 95.8, 'network': 'GE'},
    'PMBI': {'lat': 4.7, 'lon': 94.9, 'network': 'GE'},
    'BNDI': {'lat': -7.5, 'lon': 112.5, 'network': 'GE'},  # GANTI koordinat BNDI
    'SMRI': {'lat': -7.0, 'lon': 110.0, 'network': 'GE'},   # GANTI koordinat SMRI
}

MODEL = TauPyModel(model="iasp91")
NOISE_WINDOW = 10.0       # 10 detik sebelum P
SIGNAL_WINDOW = 5.0       # 5 detik setelah P untuk SNR

# Paths
BASE_DIR = os.getcwd()
RAW_DIR = os.path.join(BASE_DIR, "raw")
META_DIR = os.path.join(BASE_DIR, "metadata")
os.makedirs(META_DIR, exist_ok=True)
# ===============================================

def calculate_p_arrival(station_lat, station_lon, event_params):
    """Hitung waktu tiba P menggunakan Taup"""
    try:
        dist_deg = locations2degrees(event_params['latitude'], event_params['longitude'], 
                                     station_lat, station_lon)
        arrivals = MODEL.get_travel_times(source_depth_in_km=event_params['depth_km'],
                                          distance_in_degree=dist_deg,
                                          phase_list=["P"])
        if arrivals:
            return event_params['origin_time'] + arrivals[0].time
    except Exception as e:
        print(f"    ⚠️ Error calculating P arrival: {e}")
    return None

def find_channel_file(event_dir, station, channel):
    """
    Mencari file untuk satu stasiun dan komponen.
    Mendukung struktur: raw/event/station/channel/*.mseed
    """
    channel_dir = os.path.join(event_dir, station, channel)
    if not os.path.exists(channel_dir):
        return None
    
    # Cari file .mseed di folder channel
    pattern = os.path.join(channel_dir, "*.mseed")
    files = glob.glob(pattern)
    
    # Cari juga file .sac jika ada
    if not files:
        pattern = os.path.join(channel_dir, "*.sac")
        files = glob.glob(pattern)
    
    if files:
        return files[0]  # Ambil file pertama
    return None

def compute_noise_and_snr(trace, p_time, noise_win, signal_win):
    """RMS noise sebelum P dan SNR kasar"""
    sr = trace.stats.sampling_rate
    data = trace.data
    starttime = trace.stats.starttime
    
    # Hitung sample index untuk P arrival
    p_sample = int((p_time - starttime) * sr)
    
    # Validasi
    if p_sample < 0:
        # Gunakan 30 detik setelah start sebagai estimasi
        p_sample = int(30 * sr)
    if p_sample >= len(data):
        p_sample = len(data) // 2
    
    # Noise window: 10 detik sebelum P (atau dari awal jika tidak cukup)
    noise_start = max(0, p_sample - int(noise_win * sr))
    noise_data = data[noise_start:p_sample]
    
    if len(noise_data) == 0:
        return np.nan, np.nan
    
    rms_noise = np.sqrt(np.mean(noise_data**2))
    
    # Signal window: 5 detik setelah P
    signal_end = min(len(data), p_sample + int(signal_win * sr))
    signal_data = data[p_sample:signal_end]
    
    if len(signal_data) == 0:
        return rms_noise, np.nan
    
    rms_signal = np.sqrt(np.mean(signal_data**2))
    snr = rms_signal / rms_noise if rms_noise > 0 else np.nan
    
    return rms_noise, snr

def rename_file(file_path, station, channel, event_name, event_time_str):
    """Rename file dengan pola yang informatif"""
    dir_name = os.path.dirname(file_path)
    base_name = os.path.basename(file_path)
    name, ext = os.path.splitext(base_name)
    
    # Format baru: STA_CH_EVENT_YYYYMMDD_HHMMSS.ext
    new_name = f"{station}_{channel}_{event_name}_{event_time_str}{ext}"
    new_path = os.path.join(dir_name, new_name)
    
    if not os.path.exists(new_path) and new_path != file_path:
        os.rename(file_path, new_path)
        return new_name
    return "skipped"

def process_event(event_name, event_params):
    """Proses satu event gempa"""
    print(f"\n{'='*70}")
    print(f"📊 MEMPROSES EVENT: {event_name.upper()}")
    print(f"   Magnitude: {event_params.get('magnitude', 'N/A')}")
    print(f"   Origin Time: {event_params['origin_time']}")
    print(f"   Location: {event_params['latitude']}°, {event_params['longitude']}°")
    print(f"   Depth: {event_params['depth_km']} km")
    print(f"{'='*70}")
    
    event_dir = os.path.join(RAW_DIR, event_name)
    if not os.path.exists(event_dir):
        print(f"❌ Folder event tidak ditemukan: {event_dir}")
        return []
    
    # Dapatkan daftar stasiun yang tersedia
    stations = [d for d in os.listdir(event_dir) 
                if os.path.isdir(os.path.join(event_dir, d))]
    print(f"\n📍 Stasiun yang tersedia: {stations}")
    
    if len(stations) < 1:
        print(f"⚠️ Tidak ada stasiun ditemukan")
        return []
    
    results = []
    event_time_str = event_params['origin_time'].strftime('%Y%m%d_%H%M%S')
    
    for station in stations:
        print(f"\n{'─'*50}")
        print(f"📍 Stasiun: {station}")
        
        # Dapatkan koordinat stasiun
        if station in STATION_COORDS:
            coords = STATION_COORDS[station]
            print(f"   Koordinat: {coords['lat']}°, {coords['lon']}°")
        else:
            print(f"   ⚠️ Koordinat untuk {station} tidak ditemukan, menggunakan default")
            coords = {'lat': 0.0, 'lon': 0.0}
        
        # Hitung P arrival
        p_time = calculate_p_arrival(coords['lat'], coords['lon'], event_params)
        if p_time:
            print(f"   ⏱️ P arrival (teori): {p_time}")
        else:
            print(f"   ⚠️ Tidak bisa hitung P arrival, menggunakan estimasi +30s")
            p_time = event_params['origin_time'] + 30
        
        # Proses 3 komponen
        for channel in ['Z', 'N', 'E']:
            file_path = find_channel_file(event_dir, station, channel)
            
            if not file_path:
                print(f"   ❌ Komponen {channel}: file tidak ditemukan")
                continue
            
            print(f"   📄 Komponen {channel}: {os.path.basename(file_path)}")
            
            try:
                # Baca file
                st = read(file_path)
                trace = st[0]
                
                # Hitung noise dan SNR
                rms_noise, snr = compute_noise_and_snr(trace, p_time, NOISE_WINDOW, SIGNAL_WINDOW)
                
                # Rename file
                new_name = rename_file(file_path, station, channel, event_name, event_time_str)
                
                # Simpan hasil
                result = {
                    'event': event_name,
                    'magnitude': event_params.get('magnitude', 'N/A'),
                    'station': station,
                    'channel': channel,
                    'original_file': os.path.basename(file_path),
                    'renamed_to': new_name,
                    'p_arrival_theory': p_time.isoformat() if p_time else 'N/A',
                    'rms_noise': rms_noise,
                    'snr': snr,
                    'noise_window_sec': NOISE_WINDOW,
                    'signal_window_sec': SIGNAL_WINDOW,
                    'sampling_rate': trace.stats.sampling_rate,
                    'n_samples': trace.stats.npts
                }
                results.append(result)
                
                print(f"      ✅ RMS noise = {rms_noise:.6e}")
                print(f"      ✅ SNR = {snr:.2f}")
                
            except Exception as e:
                print(f"      ❌ Error: {e}")
    
    return results

def main():
    print(f"{'='*70}")
    print(f"📊 PROGRAM HITUNG NOISE DAN SNR UNTUK DATA GEMPA TEKTONIK")
    print(f"{'='*70}")
    
    if PROCESS_ALL_EVENTS:
        # Proses semua event
        events_to_process = list(EVENT_PARAMS.keys())
        print(f"\n📋 Akan memproses {len(events_to_process)} event:")
        for i, event in enumerate(events_to_process, 1):
            print(f"   {i}. {event} (M{EVENT_PARAMS[event].get('magnitude', '?')})")
    else:
        # Proses satu event saja
        if 'EVENT_NAME' not in globals():
            print("❌ Set PROCESS_ALL_EVENTS = False dan tentukan EVENT_NAME")
            return
        events_to_process = [EVENT_NAME]
    
    all_results = []
    
    for event_name in events_to_process:
        if event_name not in EVENT_PARAMS:
            print(f"⚠️ Parameter untuk event {event_name} tidak ditemukan, skip")
            continue
        
        results = process_event(event_name, EVENT_PARAMS[event_name])
        all_results.extend(results)
    
    # Simpan semua hasil ke CSV
    if all_results:
        df = pd.DataFrame(all_results)
        
        # Simpan file master untuk semua event
        master_csv = os.path.join(META_DIR, "snr_initial_all_events.csv")
        df.to_csv(master_csv, index=False)
        
        # Simpan juga per event
        for event_name in df['event'].unique():
            event_df = df[df['event'] == event_name]
            event_csv = os.path.join(META_DIR, f"snr_initial_{event_name}.csv")
            event_df.to_csv(event_csv, index=False)
        
        print(f"\n{'='*70}")
        print(f"✅ HASIL SEMUA EVENT")
        print(f"{'='*70}")
        print(f"📁 Master CSV: {master_csv}")
        
        # Ringkasan per event
        print(f"\n📊 RINGKASAN PER EVENT:")
        for event_name in df['event'].unique():
            event_df = df[df['event'] == event_name]
            print(f"\n  {event_name.upper()}:")
            print(f"    Total traces: {len(event_df)}")
            print(f"    Rata-rata SNR: {event_df['snr'].mean():.2f}")
            print(f"    Std SNR: {event_df['snr'].std():.2f}")
            print(f"    Min SNR: {event_df['snr'].min():.2f}")
            print(f"    Max SNR: {event_df['snr'].max():.2f}")
            
            # Tampilkan per stasiun
            for station in event_df['station'].unique():
                station_df = event_df[event_df['station'] == station]
                snr_values = station_df['snr'].values
                print(f"      {station}: SNR = {snr_values[0]:.2f} (Z), {snr_values[1]:.2f} (N), {snr_values[2]:.2f} (E)" if len(snr_values)>=3 else f"      {station}: {station_df['channel'].values} = {snr_values}")
        
        # Tampilkan tabel lengkap
        print(f"\n📋 TABEL LENGKAP SEMUA DATA:")
        print(df[['event', 'station', 'channel', 'snr', 'rms_noise']].to_string(index=False))
        
    else:
        print(f"\n{'='*70}")
        print("❌ TIDAK ADA DATA YANG DIPROSES")
        print(f"{'='*70}")
        print("\n💡 TROUBLESHOOTING:")
        print("   1. Periksa struktur folder: raw/event_name/station/[Z,N,E]/*.mseed")
        print("   2. Pastikan event name sesuai dengan nama folder di raw/")
        print("   3. Cek koordinat gempa dan stasiun (isi dengan data sebenarnya)")
        print("   4. Untuk event jatim_m58, pastikan folder 'BNDI' memiliki subfolder Z,N,E")

if __name__ == "__main__":
    main()