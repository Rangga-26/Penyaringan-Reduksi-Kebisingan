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
# ============================================================
function baca_mseed_sederhana(filepath::String)
    data = Float64[]
    try
        open(filepath, "r") do f
            raw = read(f)
            samples = reinterpret(Int16, raw)
            offset = 64
            if length(samples) > offset
                data = Float64.(samples[offset:end])
            end
        end
    catch e
        @warn "Gagal baca $filepath: $e"
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
# No. 5: Filter 1-10 Hz untuk P-wave, 0.5-5 Hz untuk S-wave
# ============================================================
function terapkan_bandpass(data::Vector{Float64}, fs::Float64,
                           fmin::Float64, fmax::Float64)
    nyq = fs / 2.0
    fmin_norm = max(fmin / nyq, 0.001)
    fmax_norm = min(fmax / nyq, 0.999)

    if fmin_norm >= fmax_norm
        @warn "Rentang frekuensi tidak valid: $fmin - $fmax Hz"
        return data
    end

    # Butterworth bandpass order 4
    responsetype = Bandpass(fmin_norm, fmax_norm)
    designmethod = Butterworth(4)
    f = digitalfilter(responsetype, designmethod)

    # Zero-phase filtering
    filtered = filtfilt(f, data)
    return filtered
end

# ============================================================
# FUNGSI: Cross-correlation berbasis FFT
# No. 8: Cross-correlation Z-N dan Z-E
# ============================================================
function cross_correlasi_fft(sig1::Vector{Float64}, sig2::Vector{Float64})
    n = length(sig1) + length(sig2) - 1
    nfft = nextpow(2, n)

    F1 = fft([sig1; zeros(nfft - length(sig1))])
    F2 = fft([sig2; zeros(nfft - length(sig2))])

    cc = real(ifft(F1 .* conj(F2)))

    idx_max = argmax(abs.(cc))
    lag_sampel = idx_max - length(sig1)

    return lag_sampel, cc[idx_max]
end

# ============================================================
# FUNGSI: Proses satu stasiun untuk satu event
# ============================================================
function proses_stasiun(event::String, station::String,
                        raw_dir::String,
                        snr_awal::Dict{String, Float64},
                        fs::Float64=20.0)

    println("  → Memproses: $event / $station")

    hasil = Dict{String, Any}(
        "event"              => event,
        "station"            => station,
        "snr_Z_before"       => get(snr_awal, "$(event)_$(station)_Z", NaN),
        "snr_N_before"       => get(snr_awal, "$(event)_$(station)_N", NaN),
        "snr_E_before"       => get(snr_awal, "$(event)_$(station)_E", NaN),
        "snr_Z_after"        => NaN,
        "snr_N_after"        => NaN,
        "snr_E_after"        => NaN,
        "snr_Z_improvement"  => NaN,
        "snr_N_improvement"  => NaN,
        "snr_E_improvement"  => NaN,
        "lag_ZN_samp"        => NaN,
        "lag_ZE_samp"        => NaN,
        "lag_ZN_sec"         => NaN,
        "lag_ZE_sec"         => NaN,
        "dt_tiba_ZN_sec"     => NaN,
        "dt_tiba_ZE_sec"     => NaN,
        "interpretasi_ZN"    => "N/A",
        "interpretasi_ZE"    => "N/A",
    )

    data_kanal = Dict{String, Vector{Float64}}()

    for kanal in ["Z", "N", "E"]
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

        noise_len    = min(200, length(raw_data) ÷ 4)
        signal_start = noise_len + 1
        signal_len   = min(100, length(raw_data) - signal_start)

        if signal_start + signal_len > length(raw_data)
            signal_len = length(raw_data) - signal_start
        end

        # No. 5 & 6: Terapkan bandpass filter
        if kanal == "Z"
            filtered = terapkan_bandpass(raw_data, fs, 1.0, 10.0)
        else
            filtered = terapkan_bandpass(raw_data, fs, 0.5, 5.0)
        end

        # No. 6: Hitung SNR setelah filtering
        noise_filtered  = filtered[1:noise_len]
        signal_filtered = filtered[signal_start:signal_start + signal_len - 1]
        snr_after = hitung_snr(signal_filtered, noise_filtered)
        hasil["snr_$(kanal)_after"] = round(snr_after, digits=4)

        # No. 7: Hitung peningkatan SNR (sesudah - sebelum)
        snr_before = get(snr_awal, "$(event)_$(station)_$(kanal)", NaN)
        if !isnan(snr_before)
            hasil["snr_$(kanal)_improvement"] = round(snr_after - snr_before, digits=4)
        end

        data_kanal[kanal] = filtered
    end

    # No. 8 & 9: Cross-correlation dan estimasi waktu tiba
    if haskey(data_kanal, "Z") && haskey(data_kanal, "N")
        len = min(length(data_kanal["Z"]), length(data_kanal["N"]))
        lag_zn, _ = cross_correlasi_fft(data_kanal["Z"][1:len], data_kanal["N"][1:len])
        dt_zn = abs(lag_zn / fs)
        hasil["lag_ZN_samp"]    = lag_zn
        hasil["lag_ZN_sec"]     = round(lag_zn / fs, digits=4)
        hasil["dt_tiba_ZN_sec"] = round(dt_zn, digits=4)

        # No. 9: Interpretasi perbedaan waktu tiba
        if lag_zn > 0
            hasil["interpretasi_ZN"] = "Kanal N tiba $(round(dt_zn, digits=2))s lebih awal dari Z"
        elseif lag_zn < 0
            hasil["interpretasi_ZN"] = "Kanal Z tiba $(round(dt_zn, digits=2))s lebih awal dari N"
        else
            hasil["interpretasi_ZN"] = "Z dan N tiba bersamaan"
        end
    end

    if haskey(data_kanal, "Z") && haskey(data_kanal, "E")
        len = min(length(data_kanal["Z"]), length(data_kanal["E"]))
        lag_ze, _ = cross_correlasi_fft(data_kanal["Z"][1:len], data_kanal["E"][1:len])
        dt_ze = abs(lag_ze / fs)
        hasil["lag_ZE_samp"]    = lag_ze
        hasil["lag_ZE_sec"]     = round(lag_ze / fs, digits=4)
        hasil["dt_tiba_ZE_sec"] = round(dt_ze, digits=4)

        # No. 9: Interpretasi perbedaan waktu tiba
        if lag_ze > 0
            hasil["interpretasi_ZE"] = "Kanal E tiba $(round(dt_ze, digits=2))s lebih awal dari Z"
        elseif lag_ze < 0
            hasil["interpretasi_ZE"] = "Kanal Z tiba $(round(dt_ze, digits=2))s lebih awal dari E"
        else
            hasil["interpretasi_ZE"] = "Z dan E tiba bersamaan"
        end
    end

    return hasil
end

# ============================================================
# MAIN
# ============================================================
function main()
    println("=" ^ 60)
    println("denoise_and_cc.jl - Kelompok 2 MFG4723")
    println("=" ^ 60)

    raw_dir      = joinpath(@__DIR__, "raw")
    metadata_dir = joinpath(@__DIR__, "metadata")

    # Baca semua file snr_initial_*.csv dan gabungkan
    csv_files = filter(f -> startswith(f, "snr_initial") && endswith(f, ".csv"),
                       readdir(metadata_dir))

    if isempty(csv_files)
        error("Tidak ada file snr_initial*.csv di $metadata_dir")
    end

    df_snr = vcat([CSV.read(joinpath(metadata_dir, f), DataFrame)
                   for f in csv_files]...; cols=:union)

    # Lookup dictionary SNR awal: "event_station_kanal" => snr
    snr_awal = Dict{String, Float64}()
    for row in eachrow(df_snr)
        key = "$(row.event)_$(row.station)_$(row.channel)"
        if !ismissing(row.snr)
    snr_awal[key] = Float64(row.snr)
end
    end

    kombinasi = unique(df_snr[:, ["event", "station"]])
    println("\nDitemukan $(nrow(kombinasi)) kombinasi event-stasiun\n")

    fs = Float64(df_snr.sampling_rate[1])

    semua_hasil = []
    for row in eachrow(kombinasi)
        hasil = proses_stasiun(String(row.event), String(row.station),
                               raw_dir, snr_awal, fs)
        push!(semua_hasil, hasil)
    end

    # Buat DataFrame hasil lengkap
    df_hasil = DataFrame(
        event                = [h["event"]             for h in semua_hasil],
        station              = [h["station"]           for h in semua_hasil],
        snr_Z_sebelum_dB     = [h["snr_Z_before"]      for h in semua_hasil],
        snr_Z_sesudah_dB     = [h["snr_Z_after"]       for h in semua_hasil],
        snr_Z_peningkatan_dB = [h["snr_Z_improvement"] for h in semua_hasil],
        snr_N_sebelum_dB     = [h["snr_N_before"]      for h in semua_hasil],
        snr_N_sesudah_dB     = [h["snr_N_after"]       for h in semua_hasil],
        snr_N_peningkatan_dB = [h["snr_N_improvement"] for h in semua_hasil],
        snr_E_sebelum_dB     = [h["snr_E_before"]      for h in semua_hasil],
        snr_E_sesudah_dB     = [h["snr_E_after"]       for h in semua_hasil],
        snr_E_peningkatan_dB = [h["snr_E_improvement"] for h in semua_hasil],
        lag_ZN_sampel        = [h["lag_ZN_samp"]       for h in semua_hasil],
        lag_ZE_sampel        = [h["lag_ZE_samp"]       for h in semua_hasil],
        lag_ZN_detik         = [h["lag_ZN_sec"]        for h in semua_hasil],
        lag_ZE_detik         = [h["lag_ZE_sec"]        for h in semua_hasil],
        dt_tiba_ZN_detik     = [h["dt_tiba_ZN_sec"]   for h in semua_hasil],
        dt_tiba_ZE_detik     = [h["dt_tiba_ZE_sec"]   for h in semua_hasil],
        interpretasi_ZN      = [h["interpretasi_ZN"]  for h in semua_hasil],
        interpretasi_ZE      = [h["interpretasi_ZE"]  for h in semua_hasil],
    )

    # No. 10: Simpan ke station_summary.csv
    output_path = joinpath(metadata_dir, "station_summary.csv")
    CSV.write(output_path, df_hasil)

    println("\n" * "=" ^ 60)
    println("SELESAI! Hasil disimpan ke:")
    println("  $output_path")
    println("=" ^ 60)
    println("\nRingkasan hasil:")
    println(df_hasil)

    # No. 7: Evaluasi peningkatan SNR
    println("\n--- Evaluasi Peningkatan SNR (No. 7) ---")
    for kanal in ["Z", "N", "E"]
        col = Symbol("snr_$(kanal)_peningkatan_dB")
        vals = filter(!isnan, df_hasil[:, col])
        if !isempty(vals)
            rata = round(mean(vals), digits=2)
            status = rata >= 5.0 ? "✓ target >5 dB tercapai" : "⚠ di bawah target 5 dB"
            println("  Kanal $kanal: rata-rata peningkatan = $(rata) dB → $status")
        end
    end

    # No. 9: Ringkasan estimasi waktu tiba
    println("\n--- Estimasi Waktu Tiba Antar Kanal (No. 9) ---")
    for row in eachrow(df_hasil)
        println("  $(row.event)/$(row.station):")
        println("    ZN → $(row.interpretasi_ZN)")
        println("    ZE → $(row.interpretasi_ZE)")
    end
end

main()