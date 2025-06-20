#!/bin/bash

# Put STM32N6570-DK board in dev mode beforehand
# Adjust path in $STEDGEAI_CORE_DIR/scripts/N6_scripts/config.json

# See https://stedgeai-dc.st.com/assets/embedded-docs/stneuralart_getting_started.html

STEDGEAI_CORE_DIR=$(find $HOME/STM32Cube/Repository/Packs/STMicroelectronics/X-CUBE-AI -mindepth 1 -maxdepth 1 | sort | tail -n1)
$STEDGEAI_CORE_DIR/Utilities/linux/stedgeai analyze --target stm32n6 --name denoiser_gemm -m denoiser_dns_rewritten_q.onnx --st-neural-art "n6-noextmem@user_neuralart.json" --verbosity 3
dot -Tsvg st_ai_output/denoiser_dns_rewritten_q_OE_3_2_0.dot > st_ai_output/denoiser_dns_rewritten_q_OE_3_2_0.svg