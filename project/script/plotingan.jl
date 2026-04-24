# plotting.jl - Versi Mandiri
# Simpan file ini di direktori utama repositori

using Plots
using DSP
using FFTW
using Statistics

# ============================================
# Fungsi Baca MiniSEED (copy dari denoise_and_cc.jl)
# ============================================
function baca_mseed_sederhana(filepath::String)
    io = open(filepath, "r")
    data_raw = read(io)
    close(io)
    
    # Parse sample rate dari byte 16-17 (little-endian)
    fs = reinterpret(Float32, data_raw[17:20])[1]
    
    # Baca data integer 16-bit mulai offset 64
    offset = 65
    n_samples = div(length(data_raw) - 64, 2)
    data_int16 = reinterpret(Int16, data_raw[offset:end])
    data_float = Float64.(data_int16[1:n_samples])
    
    return data_float, fs
end

# ============================================
# Fungsi Korelasi Silang
# ============================================
function hitung_korelasi(x::Vector{Float64}, y::Vector{Float64}, fs::Float64)
    n = min(length(x), length(y))
    x_cut = x[1:n]
    y_cut = y[1:n]
    
    # Zero-padding ke pangkat 2 terdekat
    n_fft = nextpow(2, 2n - 1)
    
    # Korelasi via FFT
    X = fft(x_cut, n_fft)
    Y = fft(y_cut, n_fft)
    korelasi = real.(ifft(X .* conj(Y)))
    
    # Susun ulang: bagian kedua ke awal (negative lags)
    korelasi = [korelasi[end-n+2:end]; korelasi[1:n]]
    
    # Cari lag maksimum
    idx_max = argmax(korelasi)
    lag_samples = idx_max - n
    lag_detik = lag_samples / fs
    
    # Buat sumbu lag dalam detik
    lags_detik = collect(range(-(n-1)/fs, step=1/fs, length=2n-1))
    
    return lags_detik, korelasi, lag_detik
end

# ============================================
# Fungsi Plot Perbandingan Sinyal
# ============================================
function plot_sinyal(raw_Z, raw_N, raw_E, filt_Z, filt_N, filt_E, fs, event, station)
    n_samples = length(raw_Z)
    t = range(0, step=1/fs, length=n_samples)

    p1 = plot(t, raw_Z, label="Raw Z", linewidth=0.8, color=:gray,
              title="Komponen Z", xlabel="Waktu (detik)", ylabel="Amplitudo")
    plot!(p1, t, filt_Z, label="Filtered Z", linewidth=1.2, color=:blue)

    p2 = plot(t, raw_N, label="Raw N", linewidth=0.8, color=:gray,
              title="Komponen N", xlabel="Waktu (detik)", ylabel="Amplitudo")
    plot!(p2, t, filt_N, label="Filtered N", linewidth=1.2, color=:red)

    p3 = plot(t, raw_E, label="Raw E", linewidth=0.8, color=:gray,
              title="Komponen E", xlabel="Waktu (detik)", ylabel="Amplitudo")
    plot!(p3, t, filt_E, label="Filtered E", linewidth=1.2, color=:green)

    p_total = plot(p1, p2, p3, layout=(3,1), size=(1000, 900),
                   plot_title="Perbandingan Sinyal Raw vs Filtered\n$event - $station")
    return p_total
end

# ============================================
# Fungsi Plot Korelasi Silang
# ============================================
function plot_korelasi(lags, corr_ZN, corr_ZE, lag_ZN, lag_ZE, event, station)
    p1 = plot(lags, corr_ZN, linewidth=1.0, color=:purple,
              title="Korelasi Silang Z-N", xlabel="Lag (detik)",
              ylabel="Koefisien Korelasi", label="CC Z-N")
    vline!(p1, [lag_ZN], linestyle=:dash, color=:red, linewidth=1.5,
           label="Lag = $(round(lag_ZN, digits=2)) s")

    p2 = plot(lags, corr_ZE, linewidth=1.0, color=:darkorange,
              title="Korelasi Silang Z-E", xlabel="Lag (detik)",
              ylabel="Koefisien Korelasi", label="CC Z-E")
    vline!(p2, [lag_ZE], linestyle=:dash, color=:red, linewidth=1.5,
           label="Lag = $(round(lag_ZE, digits=2)) s")

    p_total = plot(p1, p2, layout=(2,1), size=(1000, 700),
                   plot_title="Fungsi Korelasi Silang\n$event - $station")
    return p_total
end

# ============================================
# Fungsi Utama
# ============================================
function buat_semua_plot()
    events = filter(d -> isdir("raw/$d"), readdir("raw/"))

    for event in events
        stations = filter(d -> isdir("raw/$event/$d"), readdir("raw/$event/"))

        for station in stations
            println("Membuat plot: $event - $station")

            try
                # Baca data
                raw_Z, fs = baca_mseed_sederhana("raw/$event/$station/Z/" * 
                                                  first(filter(f -> endswith(f, ".mseed"),
                                                         readdir("raw/$event/$station/Z/"))))
                raw_N, _  = baca_mseed_sederhana("raw/$event/$station/N/" * 
                                                  first(filter(f -> endswith(f, ".mseed"),
                                                         readdir("raw/$event/$station/N/"))))
                raw_E, _  = baca_mseed_sederhana("raw/$event/$station/E/" * 
                                                  first(filter(f -> endswith(f, ".mseed"),
                                                         readdir("raw/$event/$station/E/"))))

                # Tapis
                resp_Z  = digitalfilter(Bandpass(1.0, 10.0, fs=fs), Butterworth(4))
                resp_NE = digitalfilter(Bandpass(0.5, 5.0, fs=fs), Butterworth(4))

                filt_Z = filtfilt(resp_Z, raw_Z)
                filt_N = filtfilt(resp_NE, raw_N)
                filt_E = filtfilt(resp_NE, raw_E)

                # Korelasi
                lags, corr_ZN, lag_ZN = hitung_korelasi(filt_Z, filt_N, fs)
                _,    corr_ZE, lag_ZE = hitung_korelasi(filt_Z, filt_E, fs)

                # Simpan plot
                out_dir = "Interpretation/$event/$station"
                mkpath(out_dir)

                p1 = plot_sinyal(raw_Z, raw_N, raw_E, filt_Z, filt_N, filt_E, fs, event, station)
                savefig(p1, "$out_dir/$(event)_$(station)_sinyal.png")

                p2 = plot_korelasi(lags, corr_ZN, corr_ZE, lag_ZN, lag_ZE, event, station)
                savefig(p2, "$out_dir/$(event)_$(station)_korelasi.png")

                println("  -> Berhasil disimpan di $out_dir/")

            catch e
                println("  -> GAGAL: $e")
            end
        end
    end
    println("\nSemua plot selesai dibuat.")
end

# Jalankan
buat_semua_plot()