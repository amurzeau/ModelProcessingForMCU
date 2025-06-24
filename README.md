# ModelProcessingForMCU

This repository process tiny_denoiser model from [GreenWaves](https://github.com/GreenWaves-Technologies/tiny_denoiser) to make it compatible with STM32N6 and ARM Ethos.

Prerequisites:
- ST Edge AI
- Python 3.12
- Make
- denoiser_dns.onnx in the root folder

Install:
```
python3.12 -m venv venv
. ./venv/bin/activate
pip install -r requirements.txt
make
```

Generated files are in `tmp` folder and `st_ai_output` folder.

The operations done are:
- Replace LSTM and GRU with lowered equivalents
- For ST, split Conv and Gemm with prime number in a dimension
- Reduce amount of Transpose/Unsqueeze/Squeeze
- Quantize onnx model using QDQ
- Generate tflite with int8 and int16 quantization
