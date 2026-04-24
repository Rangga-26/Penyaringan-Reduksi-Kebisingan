using DSP
using FFTW
using Plots

# pakai fungsi dari script utama
include("denoise_and_cc.jl")

raw_dir = joinpath(@__DIR__, "raw")
fs = 20.0   # samakan dengan datamu

events = readdir(raw_dir)

for event in events
    stations = readdir(joinpath(raw_dir, event))

    for station in stations

        println("\nEvent: $event | Station: $station")

        data_raw  = Dict{String,Vector{Float64}}()
        data_filt = Dict{String,Vector{Float64}}()

        # ==========================
        # LOAD DATA + FILTER
        # ==========================
        for kanal in ["Z","N","E"]

            folder = joinpath(raw_dir, event, station, kanal)
            if !isdir(folder)
                continue
            end

            files = filter(f -> endswith(f, ".mseed"), readdir(folder))
            isempty(files) && continue

            filepath = joinpath(folder, files[1])

            raw = baca_mseed_sederhana(filepath)

            if kanal == "Z"
                filt = terapkan_bandpass(raw, fs, 1.0, 10.0)
            else
                filt = terapkan_bandpass(raw, fs, 0.5, 5.0)
            end

            data_raw[kanal]  = raw
            data_filt[kanal] = filt
        end

        # ==========================
        # CROSS CORRELATION ZN & ZE
        # ==========================
        cc_zn = nothing
        cc_ze = nothing
        lag_zn = 0
        lag_ze = 0

        if haskey(data_filt,"Z") && haskey(data_filt,"N")
            len = min(length(data_filt["Z"]), length(data_filt["N"]))

            lag_zn, _ = cross_correlasi_fft(
                data_filt["Z"][1:len],
                data_filt["N"][1:len]
            )

            n = len*2 - 1
            nfft = nextpow(2, n)

            F1 = fft([data_filt["Z"][1:len]; zeros(nfft - len)])
            F2 = fft([data_filt["N"][1:len]; zeros(nfft - len)])

            cc_zn = real(ifft(F1 .* conj(F2)))
        end

        if haskey(data_filt,"Z") && haskey(data_filt,"E")
            len = min(length(data_filt["Z"]), length(data_filt["E"]))

            lag_ze, _ = cross_correlasi_fft(
                data_filt["Z"][1:len],
                data_filt["E"][1:len]
            )

            n = len*2 - 1
            nfft = nextpow(2, n)

            F1 = fft([data_filt["Z"][1:len]; zeros(nfft - len)])
            F2 = fft([data_filt["E"][1:len]; zeros(nfft - len)])

            cc_ze = real(ifft(F1 .* conj(F2)))
        end

        # ==========================
        # PLOT PER CHANNEL
        # ==========================
        for kanal in ["Z","N","E"]

            if !haskey(data_filt, kanal)
                continue
            end

            raw  = data_raw[kanal]
            filt = data_filt[kanal]

            base_plot = joinpath(@__DIR__, "Interpretation", event, station)
            mkpath(base_plot)

            p = plot(layout=(2,1), size=(800,600))

            # -------- RAW vs FILTERED --------
            plot!(p[1], raw, label="Raw", alpha=0.4)
            plot!(p[1], filt, label="Filtered", linewidth=2)
            title!(p[1], "Channel $kanal")

            # -------- CROSS CORRELATION --------
            if cc_zn !== nothing
                plot!(p[2], cc_zn, label="ZN")
                vline!(p[2], [lag_zn], label="Lag ZN")
            end

            if cc_ze !== nothing
                plot!(p[2], cc_ze, label="ZE")
                vline!(p[2], [lag_ze], label="Lag ZE")
            end

            title!(p[2], "Cross-correlation ZN & ZE")

            savepath = joinpath(base_plot, "$(kanal).png")
            println("✔ Save: $savepath")
            savefig(p, savepath)
        end
    end
end

println("\nSELESAI PLOTTING")








