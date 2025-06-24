#!/bin/bash

set -e

# Put STM32N6570-DK board in dev mode beforehand
# Adjust path in $STEDGEAI_CORE_DIR/scripts/N6_scripts/config.json

# See https://stedgeai-dc.st.com/assets/embedded-docs/stneuralart_getting_started.html

INPUT_MODEL=${1:-tmp/denoiser_dns_q.onnx}
INPUT_MODEL_FILE=$(basename $INPUT_MODEL)
BASE_NAME=${INPUT_MODEL_FILE%.*}

STEDGEAI_CORE_DIR=$(find $HOME/STM32Cube/Repository/Packs/STMicroelectronics/X-CUBE-AI -mindepth 1 -maxdepth 1 | sort | tail -n1)
"$STEDGEAI_CORE_DIR/Utilities/linux/stedgeai" analyze --target stm32n6 --name "${BASE_NAME}" -m "${INPUT_MODEL}" --st-neural-art "n6-noextmem@config/user_neuralart.json" --verbosity 3
dot -Tsvg "st_ai_output/${BASE_NAME}_OE_3_2_0.dot" > "st_ai_output/${BASE_NAME}_OE_3_2_0.svg"
