import argparse
import sys
import os
from onnxscript.rewriter import pattern, rewrite
from onnxscript import ir
import onnx
import numpy
import preprocessing_wav
import model_compare


from onnxruntime.quantization import QuantFormat, QuantType, StaticQuantConfig, quantize, preprocess, CalibrationMethod
from onnxruntime.quantization import CalibrationDataReader


def Conv257(op, X, W, B, dilations, kernel_shape, pads, strides):
    return op.Conv(X, W, B, dilations=dilations, kernel_shape=kernel_shape, pads=pads, strides=strides)

def Conv257Cond(op, X: ir.Value, W, B, dilations, kernel_shape, pads, strides):
    if X.shape is not None and len(X.shape) >= 2 and X.shape[1] == 257:
        return True
    return False

def Conv257Replace(op, X: ir.Value, W: ir.Value, B: ir.Value, dilations: ir.Attr, kernel_shape: ir.Attr, pads: ir.Attr, strides: ir.Attr):
    # Split Gemm
    split_number = int(numpy.ceil(X.shape[1] / 86))
    split_outputs = [f"{X.name}_split_{i}" for i in range(split_number)]

    single_dim = 86
    split_dim = [single_dim] * split_number
    split_dim[-1] = X.shape[1] - (split_number-1)*single_dim
    print(f"orig={X.shape}, split_dim={split_dim}")
    split_dim_tensor = op.initializer(ir.tensor(split_dim, name=f"{X.name}_split_dim"))

    X_splitted: ir.Value = op.Split(X, split_dim_tensor, axis=1, _outputs=split_outputs)

    for i, split_item in enumerate(split_dim):
        shape = []
        for j, s in enumerate(X.shape):
            if j == 1:
                shape.append(split_item)
            else:
                shape.append(s)
        X_splitted[i].type = X.type
        X_splitted[i].shape = ir.Shape(shape)

    split_outputs = [f"{W.name}_split_{i}" for i in range(split_number)]
    W_splitted = op.Split(W, split_dim_tensor, axis=1, _outputs=split_outputs)

    if B is not None:
        output = op.Conv(X_splitted[0], W_splitted[0], B, dilations=dilations, kernel_shape=kernel_shape, pads=pads, strides=strides)
    else:
        output = op.Conv(X_splitted[0], W_splitted[0], dilations=dilations, kernel_shape=kernel_shape, pads=pads, strides=strides)

    for i in range(split_number-1):
        output = op.Add(output, op.Conv(X_splitted[i+1], W_splitted[i+1], dilations=dilations, kernel_shape=kernel_shape, pads=pads, strides=strides))

    return output


def Conv257Out(op, X, W, B, dilations, kernel_shape, pads, strides):
    return op.Conv(X, W, B, dilations=dilations, kernel_shape=kernel_shape, pads=pads, strides=strides)

def Conv257OutCond(op, X: ir.Value, W: ir.Value, B, dilations, kernel_shape, pads, strides):
    if W.shape is not None and len(W.shape) >= 1 and W.shape[0] == 257:
        return True
    return False

def Conv257OutReplace(op, X: ir.Value, W: ir.Value, B: ir.Value, dilations: ir.Attr, kernel_shape: ir.Attr, pads: ir.Attr, strides: ir.Attr):
    # Split Conv
    split_number = int(numpy.ceil(W.shape[0] / 136))
    split_outputs = [f"{W.name}_w_split_{i}" for i in range(split_number)]

    single_dim = 136
    split_dim = [single_dim] * split_number
    split_dim[-1] = W.shape[0] - (split_number-1)*single_dim
    print(f"orig={W.shape}, split_dim={split_dim}")
    split_dim_tensor = op.initializer(ir.tensor(split_dim, name=f"{W.name}_split_dim"))

    W_splitted: ir.Value = op.Split(W, split_dim_tensor, axis=0, _outputs=split_outputs)

    for i, split_item in enumerate(split_dim):
        shape = []
        for j, s in enumerate(W.shape):
            if j == 0:
                shape.append(split_item)
            else:
                shape.append(s)
        W_splitted[i].type = W.type
        W_splitted[i].shape = ir.Shape(shape)

    split_outputs = [f"{W.name}_b_split_{i}" for i in range(split_number)]
    B_splitted: ir.Value = op.Split(B, split_dim_tensor, axis=0, _outputs=split_outputs)

    for i, split_item in enumerate(split_dim):
        shape = []
        for j, s in enumerate(B.shape):
            if j == 0:
                shape.append(split_item)
            else:
                shape.append(s)
        B_splitted[i].type = B.type
        B_splitted[i].shape = ir.Shape(shape)

    conv_outputs = []

    for i in range(split_number):
        conv_outputs.append(op.Conv(X, W_splitted[i], B_splitted[i], dilations=dilations, kernel_shape=kernel_shape, pads=pads, strides=strides))

    return op.Concat(*conv_outputs, axis=1)

def GemmWithC(op, A, B, C, alpha, beta, transA, transB):
    return op.Gemm(A, B, C, alpha=alpha, beta=beta, transA=transA, transB=transB)

def GemmWithCCond(op, A: ir.Value, B, C, alpha, beta, transA, transB):
    if A.shape is not None and len(A.shape) >= 2 and A.shape[1] == 257:
        return True
    return False

def GemmWithCReplace(op, A: ir.Value, B: ir.Value, C: ir.Value, alpha: ir.Attr, beta: ir.Attr, transA: ir.Attr, transB: ir.Attr):
    # Split Gemm
    split_number = int(numpy.ceil(A.shape[1] / 130))
    split_outputs = [f"{A.name}_split_{i}" for i in range(split_number)]

    single_dim = 130
    split_dim = [single_dim] * split_number
    split_dim[-1] = A.shape[1] - (split_number-1)*single_dim
    print(f"orig={A.shape}, split_dim={split_dim}")
    split_dim_tensor = op.initializer(ir.tensor(split_dim, name=f"{A.name}_split_dim"))

    axis_to_split = 1 if transA.value == 0 else 0
    print(f"axis={axis_to_split}")
    A_splitted: ir.Value = op.Split(A, split_dim_tensor, axis=axis_to_split, _outputs=split_outputs)

    for i, split_item in enumerate(split_dim):
        shape = []
        for j, s in enumerate(A.shape):
            if j == 1:
                shape.append(split_item)
            else:
                shape.append(s)
        A_splitted[i].type = A.type
        A_splitted[i].shape = ir.Shape(shape)

    split_outputs = [f"{B.name}_split_{i}" for i in range(split_number)]
    axis_to_split = 0 if transB.value == 0 else 1
    print(f"axis={axis_to_split}")
    B_splitted = op.Split(B, split_dim_tensor, axis=axis_to_split, _outputs=split_outputs)

    if C is not None:
        output = op.Gemm(A_splitted[0], B_splitted[0], C, alpha=alpha, beta=beta, transA=transA, transB=transB)
    else:
        output = op.Gemm(A_splitted[0], B_splitted[0], alpha=alpha, beta=beta, transA=transA, transB=transB)

    for i in range(split_number-1):
        output = op.Add(output, op.Gemm(A_splitted[i+1], B_splitted[i+1], alpha=alpha, beta=beta, transA=transA, transB=transB))

    return output


def GemmWithoutC(op, A, B, alpha, beta, transA, transB):
    return op.Gemm(A, B, alpha=alpha, beta=beta, transA=transA, transB=transB)

def GemmWithoutCCond(op, A: ir.Value, B, alpha, beta, transA, transB):
    return GemmWithCCond(op, A, B, None, alpha, beta, transA, transB)

def GemmWithoutCReplace(op, A: ir.Value, B: ir.Value, alpha: ir.Attr, beta: ir.Attr, transA: ir.Attr, transB: ir.Attr):
    return GemmWithCReplace(op, A, B, None, alpha, beta, transA, transB)


# Split Conv and Gemm with prime numbers in a dimension to allow ST Edge AI to use parallel computation
# See https://community.st.com/t5/edge-ai/stm32n6-npu-acceleration-sometimes-not-used-for-1x1-conv-or-gemm/m-p/811050
def apply_rewrite(model, fast: bool):
    rule = [
        pattern.RewriteRule(
            GemmWithC,  # Target Pattern
            GemmWithCReplace,  # Replacement Pattern
            GemmWithCCond, # condition_function,
            #verbose=10
        ),
        pattern.RewriteRule(
            GemmWithoutC,  # Target Pattern
            GemmWithoutCReplace,  # Replacement Pattern
            GemmWithoutCCond, # condition_function,
            #verbose=10
        ),
    ]

    if fast:
        rule.extend([
            pattern.RewriteRule(
                Conv257Out,  # Target Pattern
                Conv257OutReplace,  # Replacement Pattern
                Conv257OutCond, # condition_function,
                #verbose=10
            ),
            pattern.RewriteRule(
                Conv257,  # Target Pattern
                Conv257Replace,  # Replacement Pattern
                Conv257Cond, # condition_function,
                #verbose=10
            ),
        ])

    model_with_rewrite_applied = rewrite(
        model,
        pattern_rewrite_rules=rule,
    )
    return model_with_rewrite_applied

parser = argparse.ArgumentParser(
                    description='Quantize ONNX model')
parser.add_argument('input_model')
parser.add_argument('output_model')
parser.add_argument('-a', '--accurate', action='store_true')
parser.add_argument('-f', '--fast', action='store_true')
args = parser.parse_args()

basename = os.path.basename(os.path.splitext(args.output_model)[0])
processed_model_path = f"tmp/{basename}_p.onnx"
quantized_model_path = args.output_model
os.makedirs("tmp", exist_ok=True)


onnx_model = onnx.load(args.input_model)
modified_model =  onnx.shape_inference.infer_shapes(onnx_model, True, True, True)

onnx_model_ir = ir.serde.deserialize_model(modified_model)
onnx_model_ir = apply_rewrite(onnx_model_ir, args.fast)
modified_model = ir.serde.serialize_model(onnx_model_ir)
# Intermediate save
onnx.save(modified_model, processed_model_path)
modified_model =  onnx.shape_inference.infer_shapes(modified_model, True, True, True)


onnx.checker.check_model(modified_model)
# Final save of preprocessed model
preprocess.quant_pre_process(modified_model, processed_model_path)


# Validate model execution
print(f"Model diff {args.input_model}, {processed_model_path}")
results = model_compare.compare_models(onnx_model, modified_model)
for k, diff_rms in results.items():
    if diff_rms != 0.0:
        print(f"{k}: {diff_rms}")


# Quantization

# Model input is in range [0.0, 256[
calibration_data = [
    model_compare.generate_data(modified_model, lambda shape: numpy.ones(shape).astype(
        numpy.float32) * 255, lambda shape: numpy.ones(shape).astype(numpy.float32)*0.9921875),
    model_compare.generate_data(modified_model, lambda shape: numpy.zeros(shape).astype(numpy.float32))]

for i in range(20):
    calibration_data.append(
        model_compare.generate_data(modified_model, lambda shape: numpy.random.uniform(0, 255, size=shape).astype(numpy.float32),
                                    lambda shape: numpy.random.uniform(-1, 0.9921875, size=shape).astype(numpy.float32)))

class XXXDataReader(CalibrationDataReader):
    def __init__(self):
        self.enum_data = None
        self.data = preprocessing_wav.generate_calibration_data(processed_model_path)
        pass

    def get_next(self):
        #if self.enum_data is None:
        #    self.enum_data = iter(calibration_data)
        #return next(self.enum_data, None)
        return next(self.data)

    def rewind(self):
        pass

dr = XXXDataReader()

if args.fast:
    nodes_to_exclude=None
else:
    # Note: L2R obtained using python3 rewrite_lstm.py denoiser_dns.onnx && python3 preprocessing_wav.py denoiser_dns_rewritten_p.onnx && ./validate_stm32n6_true.sh denoiser_dns_rewritten_q.onnx
    # Quantize only GRU layers => L2R = 0.043470766
    nodes_to_exclude=["Conv_24", "Conv_2", "Conv_22", "node_Conv_14", "node_Conv_13", "node_Conv_8", "node_Conv_7", "node_Conv_12", "node_Conv_11", "Relu_3", "node_Concat_9", "node_Concat_15", "node_Concat_13", "node_Conv_17", "node_Conv_16", "node_Conv_2", "node_Conv_3", "node_Conv_5", "Sigmoid_25", "node_Add_6", "node_Add_4", "node_Split_0", "node_Concat_18"]
    # Quantize only GRU + Conv at end => L2R = 0.150335535
    #nodes_to_exclude=["node_Conv_12", "node_Conv_11", "Relu_3", "node_Concat_13"],
    # Quantize only GRU + Conv at begin => L2R = 0.053942584
    #nodes_to_exclude=["node_Conv_17", "node_Conv_16", "node_Conv_2", "node_Conv_3", "node_Conv_5", "Sigmoid_25", "node_Add_6", "node_Add_4", "node_Split_0", "node_Concat_18"],
    # Quantize only GRU + Conv at begin + one Conv at end => L2R = 0.053498846
    #nodes_to_exclude=["node_Conv_2", "node_Conv_3", "node_Conv_5", "Sigmoid_25", "node_Add_6", "node_Add_4", "node_Split_0", "node_Concat_18"],
    # Quantize only GRU + Conv at begin + one Conv and split/concat at end => L2R = 0.051184967
    #nodes_to_exclude=["node_Conv_2", "node_Conv_3", "node_Conv_5", "Sigmoid_25", "node_Add_6", "node_Add_4"],
    #nodes_to_exclude=["Conv_24", "Sigmoid_25"],
    # Quantize all except Sigmoid and Add at end => L2R = 0.104095928
    # Force rmin of last Conv weights to -2 instead of -5.5 => L2R = 0.106459364
    #nodes_to_exclude=["Sigmoid_25", "node_Add_6", "node_Add_4"],

conf = StaticQuantConfig(
    calibration_data_reader=dr,
    quant_format=QuantFormat.QDQ,
    calibrate_method=CalibrationMethod.MinMax,
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
    nodes_to_exclude=nodes_to_exclude,
    extra_options={
        "ForceQuantizeNoInputCheck":True,
        #"TensorQuantOverrides": {
        #    "onnx::Sigmoid_173": [
        #        {"rmin": numpy.float32(-10), "rmax": numpy.float32(10) }
        #    ]
        #}
    },
    per_channel=True)
      
quantize(processed_model_path, quantized_model_path, conf)
