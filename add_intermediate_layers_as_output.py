import onnx
import os
import sys


def modify_model_test(onnx_model: onnx.ModelProto, node_type_to_include=None, tensor_name_to_include=None):
    value_info_protos = []
    shape_info = onnx.shape_inference.infer_shapes(onnx_model)

    existing_input_names = [ input.name for input in shape_info.graph.input ]
    existing_output_names = [ output.name for output in shape_info.graph.output ]

    print(existing_input_names)

    for node in shape_info.graph.node:
        if node_type_to_include is not None and node.op_type not in node_type_to_include:
            continue
        for output in node.output:
            # Skip tensor which are already in outputs, either directly or via DequantizeLinear/QuantizeLinear
            stripped_output_name = output.replace("_QuantizeLinear_Output", "").replace("_DequantizeLinear_Output", "").replace("_QuantizeLinear_Input", "").replace("_DequantizeLinear_Input", "")
            output_names_to_check = [
                stripped_output_name,
                f"{stripped_output_name}_QuantizeLinear_Output",
                f"{stripped_output_name}_DequantizeLinear_Output",
                f"{stripped_output_name}_QuantizeLinear_Input",
                f"{stripped_output_name}_DequantizeLinear_Input",
            ]
            is_already_output = False
            for output_to_check in output_names_to_check:
                if output_to_check in existing_output_names or output in existing_input_names:
                    is_already_output = True
                    break
            if is_already_output:
                continue

            if tensor_name_to_include is not None and output not in tensor_name_to_include:
                continue

            print(output)
            try:
                value_info = next(x for x in shape_info.graph.value_info if x.name == output)
                assert value_info is not None
                value_info_protos.append(value_info)
            except StopIteration:
                print(f"Output {output} not in value_info array")

    # in inference stage, these tensor will be added to output dict.
    value_info_protos.sort(key=lambda x: x.name)
    shape_info.graph.output.extend(value_info_protos)

    print(f"Model now has {len(shape_info.graph.output)} outputs")

    onnx.checker.check_model(shape_info)
    return shape_info

if __name__ == "__main__":
    input = sys.argv[1]
    output = os.path.splitext(sys.argv[1])[0] + "_intermediate.onnx"
    print(f"Generating {output} from {input} with intermediate layers output")
    onnx_model = onnx.load(input)
    onnx_model = modify_model_test(onnx_model, node_type_to_include=["Gemm", "Mul", "Add"])
    onnx.save(onnx_model, output)
