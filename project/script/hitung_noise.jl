# compute_snr.jl - Hitung Noise dan SNR untuk Data Gempa Tektonik
# Jalankan di VS Code dengan: Ctrl+Enter (per baris) atau Run (tombol segitiga)

using Seis
using DSP
using CSV
using DataFrames
using StatsBase
using Dates

# ================= KONFIGURASI =================
const NOISE_WINDOW = 10.0      # 10 detik sebelum P-wave
const SIGNAL_WINDOW = 5.0      # 5 detik setelah P-wave

const BASE_DIR = pwd()
const RAW_DIR = joinpath(BASE_DIR, "raw")
const RESULTS_DIR = joinpath(BASE_DIR, "results")

# Buat folder results jika belum ada
if !isdir(RESULTS_DIR)
    mkdir(RESULTS_DIR)
end

# ================= PARAMETER GEMPA =================
# !!! GANTI DENGAN DATA SEBENARNYA !!!
const EVENT_PARAMS = Dict(
    "jatim_m58" => Dict(
        "origin_time" => DateTime("2021-12-21T12:29:50"),
        "latitude" => -7.5,     # GANTI dengan koordinat gempa Jatim
        "longitude" => 112.5,   # GANTI dengan koordinat gempa Jatim
        "depth_km" => 30.0,
        "magnitude" => 5.8
    )
)

# ================= KOORDINAT STASIUN =================
# !!! GANTI DENGAN KOORDINAT SEBENARNYA !!!
const STATION_COORDS = Dict(
    "BNDI" => (lat=-7.5, lon=112.5)  # GANTI dengan koordinat BNDI
)

# ================= FUNGSI =================

function haversine(lat1, lon1, lat2, lon2)
    R = 6371.0
    φ1 = deg2rad(lat1)
    φ2 = deg2rad(lat2)
    Δφ = deg2rad(lat2 - lat1)
    Δλ = deg2rad(lon2 - lon1)
    
    a = sin(Δφ/2)^2 + cos(φ1) * cos(φ2) * sin(Δλ/2)^2
    c = 2 * asin(sqrt(a))
    return R * c
end

function estimate_p_arrival(event_params, station_lat, station_lon)
    distance_km = haversine(
        event_params["latitude"], event_params["longitude"],
        station_lat, station_lon
    )
    vp = 6.0  # km/s
    travel_time_seconds = distance_km / vp
    p_arrival = event_params["origin_time"] + Second(round(travel_time_seconds))
    return p_arrival, travel_time_seconds, distance_km
end

function read_seis_channel(filepath)
    try
        data = Seis.read_data(filepath)
        return data
    catch e
        println("      Error: $e")
        return nothing
    end
end

function compute_rms_noise_and_snr(seis_data, p_arrival_time, noise_win, signal_win)
    data = seis_data.x
    fs = seis_data.fs
    start_time = seis_data.t
    
    p_offset = (p_arrival_time - start_time).value / 1000.0
    
    if p_offset < 0
        p_offset = 30.0  # default jika P sebelum rekaman
    end
    
    p_sample = round(Int, p_offset * fs)
    
    if p_sample < 1 || p_sample > length(data)
        return (NaN, NaN)
    end
    
    noise_start = max(1, p_sample - round(Int, noise_win * fs))
    noise_data = data[noise_start:p_sample]
    signal_end = min(length(data), p_sample + round(Int, signal_win * fs))
    signal_data = data[p_sample:signal_end]
    
    rms_noise = length(noise_data) > 0 ? sqrt(mean(noise_data .^ 2)) : NaN
    rms_signal = length(signal_data) > 0 ? sqrt(mean(signal_data .^ 2)) : NaN
    snr = (rms_noise > 0 && !isnan(rms_noise)) ? rms_signal / rms_noise : NaN
    
    return (rms_noise, snr)
end

# ================= MAIN PROGRAM =================

println("="^60)
println("PROGRAM HITUNG NOISE DAN SNR - JULIA")
println("="^60)
println("Working directory: $(pwd())")
println("Raw directory: $RAW_DIR")
println("Exists? $(isdir(RAW_DIR))")
println()

# Cek folder raw
if !isdir(RAW_DIR)
    println("❌ ERROR: Folder 'raw' tidak ditemukan!")
    println("Pastikan Anda membuka folder project yang benar.")
    println("Current directory: $(pwd())")
    exit(1)
end

# List semua event yang tersedia
available_events = filter(d -> isdir(joinpath(RAW_DIR, d)), readdir(RAW_DIR))
println("📂 Event yang tersedia: $available_events")
println()

# Pilih event yang akan diproses (ganti sesuai kebutuhan)
EVENT_NAME = "jatim_m58"  # Ganti dengan nama event yang ingin diproses

if !(EVENT_NAME in available_events)
    println("❌ Event '$EVENT_NAME' tidak ditemukan!")
    println("Event yang tersedia: $available_events")
    exit(1)
end

println("📊 Memproses event: $EVENT_NAME")
println()

event_params = EVENT_PARAMS[EVENT_NAME]
event_dir = joinpath(RAW_DIR, EVENT_NAME)

# Dapatkan daftar stasiun
stations = filter(d -> isdir(joinpath(event_dir, d)), readdir(event_dir))
println("📍 Stasiun yang tersedia: $stations")
println()

results = []

for station in stations
    println("─"^50)
    println("📍 Stasiun: $station")
    
    # Koordinat stasiun
    if haskey(STATION_COORDS, station)
        coords = STATION_COORDS[station]
        println("   Koordinat: $(coords.lat)°, $(coords.lon)°")
    else
        println("   ⚠️ Koordinat tidak ditemukan, menggunakan default")
        coords = (lat=0.0, lon=0.0)
    end
    
    # Estimasi P arrival
    p_arrival, travel_time, distance_km = estimate_p_arrival(event_params, coords.lat, coords.lon)
    println("   Jarak: $(round(distance_km, digits=1)) km")
    println("   Travel time: $(round(travel_time, digits=1)) s")
    println("   P arrival: $p_arrival")
    
    # Proses 3 komponen
    for comp in ["Z", "N", "E"]
        comp_dir = joinpath(event_dir, station, comp)
        
        if !isdir(comp_dir)
            println("   ❌ $comp: folder tidak ditemukan")
            continue
        end
        
        # Cari file .mseed
        files = filter(f -> endswith(f, ".mseed"), readdir(comp_dir))
        if isempty(files)
            println("   ❌ $comp: file .mseed tidak ditemukan")
            continue
        end
        
        filepath = joinpath(comp_dir, files[1])
        println("   📄 $comp: $(files[1])")
        
        # Baca data
        seis_data = read_seis_channel(filepath)
        if seis_data === nothing
            continue
        end
        
        # Hitung noise dan SNR
        rms_noise, snr = compute_rms_noise_and_snr(seis_data, p_arrival, NOISE_WINDOW, SIGNAL_WINDOW)
        
        push!(results, Dict(
            "event" => EVENT_NAME,
            "station" => station,
            "channel" => comp,
            "file" => files[1],
            "distance_km" => round(distance_km, digits=1),
            "travel_time_s" => round(travel_time, digits=1),
            "rms_noise" => rms_noise,
            "snr" => snr,
            "sampling_rate" => seis_data.fs
        ))
        
        println("      ✅ RMS noise = $(@sprintf("%.6e", rms_noise))")
        println("      ✅ SNR = $(@sprintf("%.2f", snr))")
    end
    println()
end

# Simpan hasil
if !isempty(results)
    df = DataFrame(results)
    output_file = joinpath(RESULTS_DIR, "snr_initial.csv")
    CSV.write(output_file, df)
    
    println("="^60)
    println("✅ HASIL")
    println("="^60)
    println("📁 File saved: $output_file")
    println()
    println("📊 RINGKASAN:")
    println(df[:, [:station, :channel, :snr, :rms_noise]])
else
    println("❌ Tidak ada data yang diproses")
end

println()
println("✨ Selesai!")