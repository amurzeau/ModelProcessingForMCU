from typing import Callable

import sys
import numpy
import onnx
from onnx.reference import ReferenceEvaluator

def to_shape(x, target_shape):
    padding_list = []
    for x_dim, target_dim in zip(x.shape, target_shape):
        pad_value = int(target_dim - x_dim)
        pad_tuple = ((pad_value//2, pad_value//2 + pad_value%2))
        padding_list.append(pad_tuple)
    
    return numpy.pad(x, tuple(padding_list), mode='constant')

def generate_data(model: onnx.ModelProto, generator_input: Callable[[list[int]], numpy.ndarray], generator_hidden_state = None):
    if generator_hidden_state is None:
        generator_hidden_state = generator_input

    example_input = {}
    for i, input in enumerate(model.graph.input):
        # If dim_value is 0, this is a symbolic shape like batch_size and we assume it should be 1.
        dim = [ max(d.dim_value, 1) for d in input.type.tensor_type.shape.dim ]
        if i == 0:
            example_input[input.name] = generator_input(dim)
        else:
            example_input[input.name] = generator_hidden_state(dim)
    return example_input

# Validate model execution
def compare_models(original_model: onnx.ModelProto, modified_model: onnx.ModelProto) -> dict[str, numpy.float32]:
    """Compare models execution and return diff RMS between them"""

    def data_to_original_shape(x: dict):
        modified_data = {}
        for key, value in x.items():
            shape =  [input.type.tensor_type.shape.dim for input in original_model.graph.input if input.name == key]
            if len(shape) == 0:
                continue
            dims = [ max(d.dim_value, 1) for d in shape[0] ]
            modified_data[key] = to_shape(value, dims)
        return modified_data

    example_input = generate_data(modified_model,
                                lambda shape: numpy.random.randn(*shape).astype(numpy.float32),
                                lambda shape: numpy.zeros(shape).astype(numpy.float32))

    output_original = ReferenceEvaluator(original_model).run(None, data_to_original_shape(example_input), intermediate=True)
    output_modified = ReferenceEvaluator(modified_model).run(None, example_input, intermediate=True)

    results = {}
    for k in output_original.keys() & output_modified.keys():
        if output_original[k] is None or output_modified[k] is None:
            continue

        modified_output_tensor = output_modified[k]
        s = [slice(0, dim) for dim in output_modified[k].shape]
        original_output_tensor = output_original[k][*s]

        diff_rms = numpy.sqrt(numpy.mean(numpy.square(original_output_tensor - modified_output_tensor)))
        results[k] = diff_rms

    return results

if __name__ == "__main__":
    model_1 = onnx.load(sys.argv[1])
    model_2 = onnx.load(sys.argv[2])
    results = compare_models(model_1, model_2)
    for k, diff_rms in results.items():
        if diff_rms != 0.0:
            print(f"{k}: {diff_rms}")