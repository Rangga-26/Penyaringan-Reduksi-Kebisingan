using SeisIO
using CSV
using DataFrames

df = CSV.read("metadata/snr_initial_all_events.csv", DataFrame)

println(first(df, 5))

groups = groupby(df, [:event, :station])

println("Jumlah group: ", length(groups))

for g in groups
    println("====")
    println("Event: ", g.event[1])
    println("Station: ", g.station[1])

    for i in 1:nrow(g)
        file = g.original_file[i]
        event = g.event[i]
        station = g.station[i]
        channel = g.channel[i]

        filepath = joinpath("raw", event, station, channel, file)

        println("File path: ", filepath)
    end
end

# =========================
# test 1 file outside loop)
# =========================

testfile = "raw/aceh_m55/BKB/Z/BKB_Z_aceh_m55_20211221_122950.mseed"

data = read_data(testfile)

println("==== HASIL BACA DATA ====")
println(data)