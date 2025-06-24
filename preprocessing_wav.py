import librosa
import os
import onnx
import onnxruntime
import numpy
import sys

WIN_LENGTH = 400
HOP_LENGTH = 100
N_FFT = 512
SAMPLERATE = 16000
WIN_FUNC = "hann"

def open_wav(file, expected_sr=SAMPLERATE, verbose=False):
    data, sr = librosa.load(file, sr=expected_sr)
    if sr != expected_sr:
        if verbose:
            print(f"expected sr: {expected_sr} real: {sr} -> resampling")
        data = librosa.resample(data, orig_sr=sr, target_sr=expected_sr)
    return data

def preprocessing(input_file):
    if isinstance(input_file, str):
        data = open_wav(input_file)
    else:
        data = input_file
    stft = numpy.abs(librosa.stft(data, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, window=WIN_FUNC, center=False)).T
    stft = (stft - numpy.mean(stft, axis=0)) / numpy.std(stft, axis=0)
    return stft

def generate_calibration_data(model_file, with_outputs=False, fast=False, end_with_none=True):
    quant_files = [os.path.join("data/quant", f) for f in os.listdir("data/quant") if os.path.isfile(os.path.join("data/quant", f))]
    onnx_model = onnx.load(model_file)
    model_runner =  onnxruntime.InferenceSession(model_file)
    print([ output.name for output in model_runner.get_outputs() ])

    for file in quant_files:
        stfts = preprocessing(file)
        print(f"stft shape: {stfts.shape}")
        # Generate hidden layer data for next batch
        hidden_layers_data = {}
        for i in range(3):
            for stft in stfts:
                inputs = {}
                for i, input in enumerate(onnx_model.graph.input):
                    if i == 0:
                        dim = [ d.dim_value for d in input.type.tensor_type.shape.dim ]
                        inputs[input.name] = stft.astype(numpy.float32)[0:256].reshape(*dim)
                    else:
                        previous_output_name = input.name.replace("input", "output")
                        if previous_output_name in hidden_layers_data:
                            inputs[input.name] = hidden_layers_data[previous_output_name]
                        else:
                            dim = [ d.dim_value for d in input.type.tensor_type.shape.dim ]
                            inputs[input.name] = numpy.zeros(dim).astype(numpy.float32)

                outputs: list = model_runner.run(None, inputs)
                for i, output in enumerate(onnx_model.graph.output):
                    if i != 0:
                        hidden_layers_data[output.name] = outputs[i]

                if with_outputs:
                    yield inputs, outputs
                else:
                    yield inputs

                if fast:
                    if end_with_none:
                        yield None
                    return
    
    if end_with_none:
        yield None

if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "tmp/denoiser_dns_q.onnx"

    basename = os.path.basename(os.path.splitext(model_path)[0])
    validation_data_path = f"tmp/{basename}_validation.npz"
    for i, (inputs, outputs) in enumerate(generate_calibration_data(model_path, True)):
        if i < 100:
            continue
        args = {}

        input_index = 0
        for input in inputs.values():
            args[f"m_inputs_{input_index}"] = input
            input_index = input_index + 1

        output_index = 0
        for output in outputs:
            assert output is not None
            args[f"m_outputs_{output_index}"] = output
            output_index = output_index + 1
        numpy.savez(validation_data_path, **args)
        print(f"Done, {input_index} inputs, {output_index} outputs")
        break
