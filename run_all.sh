#!/bin/bash

echo "======================================"
echo "PIPELINE PROCESS DIMULAI"
echo "======================================"

# 1. Organisasi data
echo "[1/6] Menjalankan Organisasi.py ..."
python3 Organisasi.py
if [ $? -ne 0 ]; then
    echo "❌ Organisasi.py gagal"
    exit 1
fi

# 2. Baca MSEED
#echo "[2/5] Menjalankan bacamseed.py ..."
#python3 bacamseed.py
#if [ $? -ne 0 ]; then
    #echo "❌ bacamseed.py gagal"
    #exit 1
#fi

# 3. Cari folder/event
echo "[3/6] Menjalankan cari_folder.py ..."
python3 cari_folder.py
if [ $? -ne 0 ]; then
    echo "❌ cari_folder.py gagal"
    exit 1
fi

# 4. Hitung noise / SNR
echo "[4/6] Menjalankan hitung_noise.py ..."
python3 hitung_noise.py
if [ $? -ne 0 ]; then
    echo "❌ hitung_noise.py gagal"
    exit 1
fi

# 5. Denoising + cross correlation (Julia)
echo "[5/6] Menjalankan denoise_and_cc.jl ..."
julia --project=. denoise_and_cc.jl
if [ $? -ne 0 ]; then
    echo "❌ denoise_and_cc.jl gagal"
    exit 1
fi

# 5. Plotingan
echo "[6/6] Menjalankan plotingan.jl ..."
julia --project=. plotingan.jl
if [ $? -ne 0 ]; then
    echo "❌ dplotingan.jl gagal"
    exit 1
fi

echo "======================================"
echo "SEMUA PROSES SELESAI DENGAN SUKSES"
echo "======================================"

