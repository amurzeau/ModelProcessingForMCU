#!/bin/bash

set -e

INPUT_MODEL=${1:-denoiser_dns_rewritten_q.onnx}

# Put STM32N6570-DK board in dev mode beforehand
# Adjust path in $STEDGEAI_CORE_DIR/scripts/N6_scripts/config.json

# See https://stedgeai-dc.st.com/assets/embedded-docs/stneuralart_getting_started.html

STEDGEAI_CORE_DIR=$(find $HOME/STM32Cube/Repository/Packs/STMicroelectronics/X-CUBE-AI -mindepth 1 -maxdepth 1 | sort | tail -n1)
$STEDGEAI_CORE_DIR/Utilities/linux/stedgeai analyze --target stm32n6 --name network -m ${INPUT_MODEL} --st-neural-art "n6-noextmem-noec@user_neuralart.json" --verbosity 3
$STEDGEAI_CORE_DIR/Utilities/linux/python $STEDGEAI_CORE_DIR/scripts/N6_scripts/n6_loader.py --build-config N6-DK  --skip-flash
$STEDGEAI_CORE_DIR/Utilities/linux/stedgeai validate --target stm32n6 --name network -m ${INPUT_MODEL} --st-neural-art "n6-noextmem-noec@user_neuralart.json" --verbosity 3 --mode target -d serial:921600
