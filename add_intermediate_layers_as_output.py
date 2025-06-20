import onnx
import os
import sys

dirname = os.path.dirname(__file__)


def modify_model_test(onnx_model: onnx.ModelProto):
    value_info_protos = []
    shape_info = onnx.shape_inference.infer_shapes(onnx_model)

    existing_input_names = [ input.name for input in shape_info.graph.input ]
    existing_output_names = [ output.name for output in shape_info.graph.output ]

    print(existing_input_names)

    for node in shape_info.graph.node:
        for output in node.output:
            if output not in ["val_12", "gru_0_split_zrt_0_DequantizeLinear_Output"]:
                continue
            if output not in existing_output_names and output not in existing_input_names:
                #if "Linear" in output:
                #    continue
                print(output)
                try:
                    value_info = next(x for x in shape_info.graph.value_info if x.name == output)
                    assert value_info is not None
                    value_info_protos.append(value_info)
                except StopIteration:
                    print(f"Output {output} not in value_info array")

    # in inference stage, these tensor will be added to output dict.
    shape_info.graph.output.extend(value_info_protos)

    onnx.checker.check_model(shape_info)
    return shape_info

if __name__ == "__main__":
    input = sys.argv[1]
    output = os.path.splitext(sys.argv[1])[0] + "_intermediate.onnx"
    print(f"Generating {output} from {input} with intermediate layers output")
    onnx_model = onnx.load(input)
    onnx_model = modify_model_test(onnx_model)
    onnx.save(onnx_model, output)
