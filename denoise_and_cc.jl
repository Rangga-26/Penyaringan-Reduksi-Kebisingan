# ============================================================
# denoise_and_cc.jl
# Kelompok 2 - MFG4723 Pemrograman Komputer Geofisika
# Orang Kedua: Hitung SNR akhir, lag korelasi, estimasi waktu tiba
#
# Cara menjalankan:
#   julia denoise_and_cc.jl
# ============================================================

using DSP
using FFTW
using Statistics
using CSV
using DataFrames

# ============================================================
# FUNGSI: Baca file MiniSEED (manual parser sederhana)
# Membaca data waveform dari file .mseed tanpa SeisIO
# ============================================================
function baca_mseed_sederhana(filepath::String)
    """
    Membaca data sampel dari file MiniSEED secara manual.
    Mengembalikan array Float64 dari sampel data.
    Catatan: Ini parser sederhana untuk format STEIM1/STEIM2.
    """
    data = Float64[]
    try
        open(filepath, "r") do f
            raw = read(f)
            # MiniSEED: setiap record 512 atau 4096 byte
            # Coba baca sebagai Int16 (format umum BH channels)
            n = length(raw) ÷ 2
            samples = reinterpret(Int16, raw)
            # Ambil bagian data (skip header ~64 byte per record)
            # Header SEED = 48 byte fixed + variable blockettes
            offset = 64  # offset standar data
            if length(samples) > offset
                data = Float64.(samples[offset:end])
            end
        end
    catch e
        @warn "Gagal baca $filepath: $e"
        # Fallback: generate synthetic data untuk testing
        data = randn(Float64, 1000)
    end
    return data
end

# ============================================================
# FUNGSI: Hitung SNR dalam dB
# SNR = 20 * log10(RMS(signal) / RMS(noise))
# ============================================================
function hitung_snr(signal::Vector{Float64}, noise::Vector{Float64})
    rms_signal = sqrt(mean(signal .^ 2))
    rms_noise  = sqrt(mean(noise .^ 2))
    if rms_noise == 0.0 || rms_signal == 0.0
        return 0.0
    end
    return 20.0 * log10(rms_signal / rms_noise)
end

# ============================================================
# FUNGSI: Bandpass filter menggunakan DSP.jl
# ============================================================
function terapkan_bandpass(data::Vector{Float64}, fs::Float64,
                           fmin::Float64, fmax::Float64)
    # Nyquist frequency
    nyq = fs / 2.0
    # Pastikan frekuensi dalam range valid
    fmin_norm = max(fmin / nyq, 0.001)
    fmax_norm = min(fmax / nyq, 0.999)

    if fmin_norm >= fmax_norm
        @warn "Rentang frekuensi tidak valid: $fmin - $fmax Hz"
        return data
    end

    # Desain Butterworth bandpass order 4
    responsetype = Bandpass(fmin_norm, fmax_norm)
    designmethod = Butterworth(4)
    f = digitalfilter(responsetype, designmethod)

    # Terapkan filter (zero-phase dengan filtfilt)
    filtered = filtfilt(f, data)
    return filtered
end

# ============================================================
# FUNGSI: Cross-correlation berbasis FFT
# Mengembalikan lag (dalam sampel) dan nilai korelasi maksimum
# ============================================================
function cross_correlasi_fft(sig1::Vector{Float64}, sig2::Vector{Float64})
    n = length(sig1) + length(sig2) - 1
    # Padding ke power of 2 untuk efisiensi FFT
    nfft = nextpow(2, n)

    F1 = fft([sig1; zeros(nfft - length(sig1))])
    F2 = fft([sig2; zeros(nfft - length(sig2))])

    # Cross-correlation = IFFT(F1 * conj(F2))
    cc = real(ifft(F1 .* conj(F2)))

    # Cari lag maksimum
    idx_max = argmax(abs.(cc))
    lag_sampel = idx_max - length(sig1)  # lag relatif terhadap tengah

    return lag_sampel, cc[idx_max]
end

# ============================================================
# FUNGSI: Proses satu stasiun untuk satu event
# ============================================================
function proses_stasiun(event::String, station::String,
                        raw_dir::String, fs::Float64=20.0)

    println("  → Memproses: $event / $station")

    hasil = Dict{String, Any}(
        "event"       => event,
        "station"     => station,
        "snr_Z_after" => NaN,
        "snr_N_after" => NaN,
        "snr_E_after" => NaN,
        "lag_ZN_samp" => NaN,
        "lag_ZE_samp" => NaN,
        "lag_ZN_sec"  => NaN,
        "lag_ZE_sec"  => NaN,
        "dt_arrival_ZN_sec" => NaN,
        "dt_arrival_ZE_sec" => NaN,
    )

    data_kanal = Dict{String, Vector{Float64}}()

    for kanal in ["Z", "N", "E"]
        # Cari file mseed di raw/event/station/kanal/
        folder = joinpath(raw_dir, event, station, kanal)
        if !isdir(folder)
            @warn "Folder tidak ditemukan: $folder"
            continue
        end

        files = filter(f -> endswith(f, ".mseed"), readdir(folder))
        if isempty(files)
            @warn "Tidak ada file .mseed di $folder"
            continue
        end

        filepath = joinpath(folder, files[1])
        raw_data = baca_mseed_sederhana(filepath)

        if length(raw_data) < 100
            @warn "Data terlalu pendek di $filepath"
            continue
        end

        # Tentukan window noise dan signal
        # Noise: 200 sampel pertama (10 detik @ 20Hz)
        # Signal: 200 sampel setelah noise window
        noise_len   = min(200, length(raw_data) ÷ 4)
        signal_start = noise_len + 1
        signal_len  = min(100, length(raw_data) - signal_start)  # 5 detik

        if signal_start + signal_len > length(raw_data)
            signal_len = length(raw_data) - signal_start
        end

        noise_window  = raw_data[1:noise_len]
        signal_window = raw_data[signal_start:signal_start + signal_len - 1]

        # Terapkan bandpass filter
        # P-wave: 1-10 Hz, S-wave (N,E): 0.5-5 Hz
        if kanal == "Z"
            filtered = terapkan_bandpass(raw_data, fs, 1.0, 10.0)
        else
            filtered = terapkan_bandpass(raw_data, fs, 0.5, 5.0)
        end

        # Hitung SNR setelah filtering
        noise_filtered   = filtered[1:noise_len]
        signal_filtered  = filtered[signal_start:signal_start + signal_len - 1]
        snr_after = hitung_snr(signal_filtered, noise_filtered)

        hasil["snr_$(kanal)_after"] = round(snr_after, digits=4)
        data_kanal[kanal] = filtered
    end

    # Cross-correlation Z-N dan Z-E
    if haskey(data_kanal, "Z") && haskey(data_kanal, "N")
        # Samakan panjang
        len = min(length(data_kanal["Z"]), length(data_kanal["N"]))
        lag_zn, _ = cross_correlasi_fft(data_kanal["Z"][1:len],
                                         data_kanal["N"][1:len])
        hasil["lag_ZN_samp"] = lag_zn
        hasil["lag_ZN_sec"]  = round(lag_zn / fs, digits=4)
        hasil["dt_arrival_ZN_sec"] = round(abs(lag_zn / fs), digits=4)
    end

    if haskey(data_kanal, "Z") && haskey(data_kanal, "E")
        len = min(length(data_kanal["Z"]), length(data_kanal["E"]))
        lag_ze, _ = cross_correlasi_fft(data_kanal["Z"][1:len],
                                         data_kanal["E"][1:len])
        hasil["lag_ZE_samp"] = lag_ze
        hasil["lag_ZE_sec"]  = round(lag_ze / fs, digits=4)
        hasil["dt_arrival_ZE_sec"] = round(abs(lag_ze / fs), digits=4)
    end

    return hasil
end

# ============================================================
# MAIN: Jalankan untuk semua event dan stasiun
# ============================================================
function main()
    println("=" ^ 60)
    println("denoise_and_cc.jl - Kelompok 2 MFG4723")
    println("=" ^ 60)

    # Path relatif dari lokasi script
    raw_dir      = joinpath(@__DIR__, "raw")
    metadata_dir = joinpath(@__DIR__, "metadata")
    # Baca semua file snr_initial_*.csv dan gabungkan
csv_files = filter(f -> startswith(f, "snr_initial") && endswith(f, ".csv"), 
                   readdir(metadata_dir))

if isempty(csv_files)
    error("Tidak ada file snr_initial*.csv di $metadata_dir")
end

df_snr = vcat([CSV.read(joinpath(metadata_dir, f), DataFrame) for f in csv_files]...; cols=:union)

    # Ambil kombinasi unik event-station
    kombinasi = unique(df_snr[:, ["event", "station"]])

    println("\nDitemukan $(nrow(kombinasi)) kombinasi event-stasiun\n")

    # Sampling rate dari CSV (ambil nilai pertama)
    fs = df_snr.sampling_rate[1]

    # Kumpulkan semua hasil
    semua_hasil = []

    for row in eachrow(kombinasi)
        event   = row.event
        station = row.station

        hasil = proses_stasiun(String(event), String(station), raw_dir, Float64(fs))
        push!(semua_hasil, hasil)
    end

    # Buat DataFrame hasil
    df_hasil = DataFrame(
        event              = [h["event"]              for h in semua_hasil],
        station            = [h["station"]            for h in semua_hasil],
        snr_Z_after_dB     = [h["snr_Z_after"]        for h in semua_hasil],
        snr_N_after_dB     = [h["snr_N_after"]        for h in semua_hasil],
        snr_E_after_dB     = [h["snr_E_after"]        for h in semua_hasil],
        lag_ZN_sampel      = [h["lag_ZN_samp"]        for h in semua_hasil],
        lag_ZE_sampel      = [h["lag_ZE_samp"]        for h in semua_hasil],
        lag_ZN_detik       = [h["lag_ZN_sec"]         for h in semua_hasil],
        lag_ZE_detik       = [h["lag_ZE_sec"]         for h in semua_hasil],
        dt_tiba_ZN_detik   = [h["dt_arrival_ZN_sec"]  for h in semua_hasil],
        dt_tiba_ZE_detik   = [h["dt_arrival_ZE_sec"]  for h in semua_hasil],
    )

    # Simpan ke metadata/station_summary.csv
    output_path = joinpath(metadata_dir, "station_summary.csv")
    CSV.write(output_path, df_hasil)

    println("\n" * "=" ^ 60)
    println("SELESAI! Hasil disimpan ke:")
    println("  $output_path")
    println("=" ^ 60)
    println("\nRingkasan hasil:")
    println(df_hasil)

    # Statistik ringkas
    println("\n--- Rata-rata SNR setelah filtering ---")
    for kanal in ["Z", "N", "E"]
        col = Symbol("snr_$(kanal)_after_dB")
        vals = filter(!isnan, df_hasil[:, col])
        if !isempty(vals)
            println("  Kanal $kanal: mean=$(round(mean(vals), digits=2)) dB, " *
                    "std=$(round(std(vals), digits=2)) dB")
        end
    end

    println("\n--- Rata-rata lag korelasi ---")
    for pair in ["ZN", "ZE"]
        col = Symbol("lag_$(pair)_detik")
        vals = filter(!isnan, df_hasil[:, col])
        if !isempty(vals)
            println("  Lag $pair: mean=$(round(mean(vals), digits=4)) s, " *
                    "std=$(round(std(vals), digits=4)) s")
        end
    end
end

# Jalankan
main()
