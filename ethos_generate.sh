#!/bin/bash

set -e

convert_model() {
    local MODEL_PATH=$1
    local MODEL_PATH_NO_EXT=${MODEL_PATH%.*}
    vela "$MODEL_PATH" --config "$PWD/config/vela.ini" --accelerator-config ethos-u55-256 --system-config Ethos_U55_High_End_Embedded --memory-mode Sram_Only --optimise Performance --verbose-weights --verbose-cycle-estimate  --verbose-performance --show-cpu-operations > "${MODEL_PATH_NO_EXT}_vela.txt"
}

convert_model tmp/denoiser_dns_p_int8.tflite
convert_model tmp/denoiser_dns_p_int8_per_tensor.tflite
convert_model tmp/denoiser_dns_p_int8_per_tensor_dense.tflite
convert_model tmp/denoiser_dns_p_int16_8_per_tensor_dense.tflite