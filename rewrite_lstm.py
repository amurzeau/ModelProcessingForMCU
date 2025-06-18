import sys
import os
from typing import Optional, Any
from onnxscript import script
# We use ONNX opset 15 to define the function below.
from onnxscript import opset15 as op
from onnxscript import onnx_types
from onnxscript.rewriter import pattern, rewrite
from onnxscript import ir
from onnxscript.ir._tape import Builder
import onnx
import numpy
from onnx.reference import ReferenceEvaluator
from onnx import numpy_helper, version_converter

from google.protobuf.internal.containers import (  # type: ignore
    RepeatedCompositeFieldContainer,
)

from onnxruntime.quantization import QuantFormat, QuantType, StaticQuantConfig, quantize, preprocess, CalibrationMethod
from onnxruntime.quantization import CalibrationDataReader
from onnxruntime.tools.onnx_model_utils import make_dim_param_fixed

def split(op, input: ir.Value, num_outputs, name_base=None):
    if name_base is None:
        name_base = input.name
    split_outputs = [f"{name_base}_split_{i}" for i in range(num_outputs)]
    t = op.Split(input, axis=1, _outputs=split_outputs)
    return t

# Rewrite LSTM with Gemm
index = 0
def LSTM(op, X, W, R, B, sequence_lens, initial_h, initial_c):
    return op.Squeeze(op.LSTM(X, W, R, B, sequence_lens, initial_h, initial_c, direction="forward", hidden_size=256, input_forget=0), axes=[1])

def LoweredLSTM(op: Builder, X: ir.Value,
        W: ir.Value,
        R: ir.Value,
        B: ir.Value,
        sequence_lens: ir.Value, initial_h: ir.Value, initial_c: ir.Value):
    global index
    global onnx_model_ir

    x = op.Squeeze(X, axes=[0])
    w = op.Squeeze(W, axes=[0])
    r = op.Squeeze(R, axes=[0])
    Wb , Rb = split(op, B, 2)
    wb_plus_rb = op.Squeeze(op.Add(Wb, Rb), axes=[0])
    # x = 1, 257
    # w = 257, 1024
    # wb_plus_rb = 1, 1024
    # Output = 1, 1024 = x * w + wb_plus_rb
    # gemm1 = Xt*(Wi^T) for i, o, f, c
    gemm1: ir.Value = op.Gemm(x, w, wb_plus_rb, alpha=1.0, beta=1.0, transA=0, transB=1)
    gemm1.name = f"lstm_{index}_gemm_0"


    Y_h_input = ir.Value(name=f"lstm_hidden_input_h_{index}", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 256]))
    Y_c_input = ir.Value(name=f"lstm_hidden_input_c_{index}", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 256]))

    Y_h_output = f"lstm_hidden_output_h_{index}"
    Y_c_output = f"lstm_hidden_output_c_{index}"

    onnx_model_ir.graph.inputs.append(Y_h_input)
    onnx_model_ir.graph.inputs.append(Y_c_input)

    # gates2 = Ht-1*(Ri^T) for i, o, f, c

    gemm2 = op.Gemm(Y_h_input, r, alpha=1.0, beta=1.0, transA=0, transB=1)
    gemm2.name = f"lstm_{index}_gemm_1"
    gates2 = op.Add(gemm2, gemm1)


    it, ot, ft, ct = split(op, gates2, 4, f"lstm_{index}_gates")
    it = op.Sigmoid(it)
    ot = op.Sigmoid(ot)
    ft = op.Sigmoid(ft)
    ct = op.Tanh(ct)

    Y_c = op.Tanh(op.Add(op.Mul(Y_c_input, ft), op.Mul(it, ct)), _outputs=[Y_c_output])
    Y_c.type = ir.TensorType(ir.DataType.FLOAT)
    Y_c.shape = ir.Shape([1, 256])

    Y_h = op.Mul(Y_c, ot, _outputs=[Y_h_output])
    Y_h.type = ir.TensorType(ir.DataType.FLOAT)
    Y_h.shape = ir.Shape([1, 256])
    
    onnx_model_ir.graph.outputs.append(Y_c)
    onnx_model_ir.graph.outputs.append(Y_h)


    output = op.Unsqueeze(Y_h, axes=[0])
    
    index = index + 1

    return output


# Rewrite GRU with Gemm
def GRU(op, X, W, R, B, sequence_lens, initial_h, hidden_size, linear_before_reset):
    return op.Squeeze(op.GRU(X, W, R, B, sequence_lens, initial_h, direction="forward", hidden_size=hidden_size, linear_before_reset=linear_before_reset), axes=[1])

const_1 = None
def LoweredGRU(op: Builder, X: ir.Value,
        W: ir.Value,
        R: ir.Value,
        B: ir.Value,
        sequence_lens: ir.Value,
        initial_h: ir.Value,
        hidden_size: ir.Attr,
        linear_before_reset: ir.Attr):
    global index
    global onnx_model_ir
    global const_1

    hidden_size_val = hidden_size.as_int()

    x = op.Squeeze(X, axes=[0])
    w = op.Squeeze(W, axes=[0])
    r = op.Squeeze(R, axes=[0])
    b = op.Squeeze(B, axes=[0])

    # Bias must be rank 1 to avoid this error:
    # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()

    split_outputs = [f"gru_{index}_split_b_{i}" for i in range(2)]
    b_splitted = op.Split(b, split=[hidden_size_val*3, hidden_size_val*3], axis=0, _outputs=split_outputs)
    Wb, Rb = b_splitted

    Y_h_input = ir.Value(name=f"gru_hidden_input_h_{index}", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, hidden_size_val]))

    Y_h_output = f"gru_hidden_output_h_{index}"

    onnx_model_ir.graph.inputs.append(Y_h_input)

    if linear_before_reset.as_int() == 0:
        gemmX = op.Gemm(x, w, op.Add(Wb, Rb), alpha=1.0, beta=1.0, transA=0, transB=1)
        gemmY = op.Gemm(Y_h_input, r, alpha=1.0, beta=1.0, transA=0, transB=1)
    else:
        gemmX = op.Gemm(x, w, Wb, alpha=1.0, beta=1.0, transA=0, transB=1)
        gemmY = op.Gemm(Y_h_input, r, Rb, alpha=1.0, beta=1.0, transA=0, transB=1)

    split_outputs = [f"gru_{index}_split_gemmX_{i}" for i in range(2)]
    gemmX_splitted = op.Split(gemmX, split=[hidden_size_val*2, hidden_size_val], axis=1, _outputs=split_outputs)
    zrtX, htX = gemmX_splitted

    split_outputs = [f"gru_{index}_split_gemmY_{i}" for i in range(2)]
    gemmY_splitted = op.Split(gemmY, split=[hidden_size_val*2, hidden_size_val], axis=1, _outputs=split_outputs)
    zrtY, htY = gemmY_splitted

    zrt = op.Sigmoid(op.Add(zrtX, zrtY))

    split_outputs = [f"gru_{index}_split_zrt_{i}" for i in range(2)]
    zrt_splitted = op.Split(zrt, split=[hidden_size_val, hidden_size_val], axis=1, _outputs=split_outputs)
    zt, rt = zrt_splitted

    ht = op.Tanh(op.Add(htX, op.Mul(htY, rt)))

    # split_outputs = [f"gru_{index}_split_b_{i}" for i in range(6)]
    # b = op.Split(b, axis=0, _outputs=split_outputs)
    # Wbz, Wbr, Wbh, Rbz, Rbr, Rbh = b
    # 
    # split_outputs = [f"gru_{index}_split_w_{i}" for i in range(3)]
    # w = op.Split(w, axis=0, _outputs=split_outputs)
    # Wz, Wr, Wh = w
    # 
    # split_outputs = [f"gru_{index}_split_r_{i}" for i in range(3)]
    # r = op.Split(r, axis=0, _outputs=split_outputs)
    # Rz, Rr, Rh = r

    # zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
    #zt = op.Sigmoid(
    #    op.Add(
    #        op.Gemm(x, Wz, alpha=1.0, beta=1.0, transA=0, transB=1),
    #        op.Gemm(Y_h_input, Rz, op.Add(Wbz, Rbz), alpha=1.0, beta=1.0, transA=0, transB=1)
    #    )
    #)
    #
    ## rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)
    #rt = op.Sigmoid(
    #    op.Add(
    #        op.Gemm(x, Wr, alpha=1.0, beta=1.0, transA=0, transB=1),
    #        op.Gemm(Y_h_input, Rr, op.Add(Wbr, Rbr), alpha=1.0, beta=1.0, transA=0, transB=1)
    #    )
    #)

    # rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)
    #if linear_before_reset.as_int() == 0:
    #    # ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh) # default, when linear_before_reset = 0
    #    ht = op.Tanh(
    #        op.Add(
    #            op.Gemm(x, Wh, op.Add(Wbh, Rbh), alpha=1.0, beta=1.0, transA=0, transB=1),
    #            op.Mul(op.Gemm(Y_h_input, Rh), rt)
    #        )
    #    )
    #else:
    #    # ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh) # when linear_before_reset != 0
    #    ht = op.Tanh(
    #        op.Add(
    #            op.Gemm(x, Wh, Wbh, alpha=1.0, beta=1.0, transA=0, transB=1),
    #            op.Mul(op.Gemm(Y_h_input, Rh, Rbh, alpha=1.0, beta=1.0, transA=0, transB=1), rt)
    #        )
    #    )

    # Ht = (1 - zt) (.) ht + zt (.) Ht-1
    if const_1 is None:
        const_1 = op.initializer(ir.tensor([numpy.float32(1)], name=f"const_1"))
    Ht = op.Add(op.Mul(op.Add(const_1, op.Neg(zt)), ht), op.Mul(zt, Y_h_input), _outputs=[Y_h_output])
    Ht.type = ir.TensorType(ir.DataType.FLOAT)
    Ht.shape = ir.Shape([1, hidden_size_val])

    onnx_model_ir.graph.outputs.append(Ht)
    
    output = op.Unsqueeze(Ht, axes=[0])
    
    index = index + 1

    return output

def TransposeSqueeze(op, X, perm, axes):
    return op.Squeeze(op.Transpose(X, perm=perm), axes=axes)

def TransposeSqueezeCond(op, X, perm, axes):
    perm = perm.as_ints()
    axes = axes.as_ints()
    if perm[0] == 2 and perm[1] == 0 and perm[2] == 1 and axes[0] == 0:
        return True
    return False

def TransposeSqueezeReplace(op, X, perm: ir.Attr, axes: ir.Attr):
    idx = axes.as_ints()[0]
    idx = perm.as_ints()[idx]
    return op.Squeeze(X, axes=[idx])

def UnsqueezeTranspose(op, X, perm, axes):
    return op.Transpose(op.Unsqueeze(X, axes=axes), perm=perm)

def UnsqueezeTransposeCond(op, X, perm, axes):
    perm = perm.as_ints()
    axes = axes.as_ints()
    print(perm, axes)
    if perm[0] == 1 and perm[1] == 2 and perm[2] == 0 and axes[0] == 0:
        return True
    return False

def UnsqueezeTransposeReplace(op, X, perm: ir.Attr, axes: ir.Attr):
    print(perm, axes)
    return op.Unsqueeze(X, axes=[2])



def UnSqueezeSqueeze(op, X, axes0):
    return op.Squeeze(op.Unsqueeze(X, axes=axes0), axes=axes0)

def UnSqueezeSqueezeReplace(op, X, axes0: ir.Attr):
    return X



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

    X_splitted: ir.Value = op.Split(X, axis=1, split=split_dim, _outputs=split_outputs)

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
    W_splitted = op.Split(W, axis=1, split=split_dim, _outputs=split_outputs)

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

    W_splitted: ir.Value = op.Split(W, axis=0, split=split_dim, _outputs=split_outputs)

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
    B_splitted: ir.Value = op.Split(B, axis=0, split=split_dim, _outputs=split_outputs)

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

    axis_to_split = 1 if transA.value == 0 else 0
    print(f"axis={axis_to_split}")
    A_splitted: ir.Value = op.Split(A, axis=axis_to_split, split=split_dim, _outputs=split_outputs)

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
    B_splitted = op.Split(B, axis=axis_to_split, split=split_dim, _outputs=split_outputs)

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

def apply_rewrite(model):
    rule = [
        pattern.RewriteRule(
            LSTM,  # Target Pattern
            LoweredLSTM,  # Replacement Pattern
            None, # condition_function,
            name="LSTM",
            verbose=10
        ),
        pattern.RewriteRule(
            GRU,  # Target Pattern
            LoweredGRU,  # Replacement Pattern
            None, # condition_function,
            name="GRU",
            verbose=10
        ),
        pattern.RewriteRule(
            TransposeSqueeze,  # Target Pattern
            TransposeSqueezeReplace,  # Replacement Pattern
            TransposeSqueezeCond, # condition_function,
            name="TransposeSqueeze",
            verbose=10
        ),
        pattern.RewriteRule(
            UnSqueezeSqueeze,  # Target Pattern
            UnSqueezeSqueezeReplace,  # Replacement Pattern
            None, # condition_function,
            name="UnSqueezeSqueeze",
            verbose=10
        ),
        pattern.RewriteRule(
            UnsqueezeTranspose,  # Target Pattern
            UnsqueezeTransposeReplace,  # Replacement Pattern
            UnsqueezeTransposeCond, # condition_function,
            name="UnsqueezeTranspose",
            verbose=10
        )
    ]
    model_with_rewrite_applied = rewrite(
        model,
        pattern_rewrite_rules=rule,
    )
    return model_with_rewrite_applied


def apply_rewrite2(model):
    rule = [
        pattern.RewriteRule(
            Conv257,  # Target Pattern
            Conv257Replace,  # Replacement Pattern
            Conv257Cond, # condition_function,
            verbose=10
        ),
        pattern.RewriteRule(
            Conv257Out,  # Target Pattern
            Conv257OutReplace,  # Replacement Pattern
            Conv257OutCond, # condition_function,
            verbose=10
        ),
        pattern.RewriteRule(
            GemmWithC,  # Target Pattern
            GemmWithCReplace,  # Replacement Pattern
            GemmWithCCond, # condition_function,
            verbose=10
        ),
        pattern.RewriteRule(
            GemmWithoutC,  # Target Pattern
            GemmWithoutCReplace,  # Replacement Pattern
            GemmWithoutCCond, # condition_function,
            verbose=10
        )
    ]
    model_with_rewrite_applied = rewrite(
        model,
        pattern_rewrite_rules=rule,
    )
    return model_with_rewrite_applied

def crop_tensor_from_257_to_256(graph: onnx.GraphProto):
    def update_dim_values(value_infos: RepeatedCompositeFieldContainer[onnx.ValueInfoProto]):
        for vi in value_infos:
            if vi.type.HasField("tensor_type"):
                shape = vi.type.tensor_type.shape
                if shape:
                    for dim in shape.dim:
                        if dim.HasField("dim_value") and dim.dim_value == 257:
                            dim.Clear()
                            dim.dim_value = 256

    def resize_initializer(name: str, new_dim):
        for initializer in graph.initializer:
            if initializer.name == name:
                data = numpy_helper.to_array(initializer)
                new_slice = [ slice(0, dim) for dim in new_dim ]
                data = data[*new_slice]
                name = initializer.name
                initializer.Clear()
                initializer.MergeFrom(numpy_helper.from_array(data, name))
                break

    update_dim_values(graph.input)
    update_dim_values(graph.output)

    # Remove all value_info to be rediscovered after changing initializers and inputs
    while len(graph.value_info) > 0:
        graph.value_info.pop()


    # Adjust first Conv
    for node in graph.node:
        if node.op_type == "Conv":
            node.input[1]
            resize_initializer(node.input[1], [257, 256, 1])
            break
    
    # Adjust last Conv
    for node in reversed(graph.node):
        if node.op_type == "Conv":
            node.input[1]
            resize_initializer(node.input[1], [256, 257, 1])
            resize_initializer(node.input[2], [256])
            break


basename = os.path.basename(os.path.splitext(sys.argv[1])[0])
onnx_model = onnx.load(sys.argv[1])
onnx_model = version_converter.convert_version(onnx_model, 13)
make_dim_param_fixed(onnx_model.graph, "batch_size", 1)
modified_model = onnx.load(sys.argv[1])
make_dim_param_fixed(modified_model.graph, "batch_size", 1)

crop_tensor_from_257_to_256(modified_model.graph)

preprocess.quant_pre_process(modified_model, f"{basename}_rewritten_b.onnx")

modified_model = onnx.load(f"{basename}_rewritten_b.onnx")

onnx_model_ir: ir.Model = ir.serde.deserialize_model(modified_model)
onnx_model_ir = apply_rewrite(onnx_model_ir)
modified_model = ir.serde.serialize_model(onnx_model_ir)
onnx.save(modified_model, f"{basename}_rewritten_p.onnx")
modified_model =  onnx.shape_inference.infer_shapes(modified_model, True, True, True)

onnx_model_ir = ir.serde.deserialize_model(modified_model)
onnx_model_ir = apply_rewrite2(onnx_model_ir)
modified_model = ir.serde.serialize_model(onnx_model_ir)
onnx.save(modified_model, f"{basename}_rewritten_p.onnx")
modified_model =  onnx.shape_inference.infer_shapes(modified_model, True, True, True)


onnx.checker.check_model(modified_model)


# Validate model execution

def generate_data(generator_input, generator_hidden_state = None):
    if generator_hidden_state is None:
        generator_hidden_state = generator_input

    example_input = {}
    for i, input in enumerate(modified_model.graph.input):
        dim = [ d.dim_value for d in input.type.tensor_type.shape.dim ]
        if i == 0:
            example_input[input.name] = generator_input(dim)
        else:
            example_input[input.name] = generator_hidden_state(dim)
    return example_input


def to_shape(x, target_shape):
    padding_list = []
    for x_dim, target_dim in zip(x.shape, target_shape):
        pad_value = int(target_dim - x_dim)
        pad_tuple = ((pad_value//2, pad_value//2 + pad_value%2))
        padding_list.append(pad_tuple)
    
    return numpy.pad(x, tuple(padding_list), mode='constant')

def data_to_original_shape(x: dict):
    modified_data = {}
    for key, value in x.items():
        shape =  [input.type.tensor_type.shape.dim for input in onnx_model.graph.input if input.name == key]
        if len(shape) == 0:
            continue
        dims = [ d.dim_value for d in shape[0] ]
        modified_data[key] = to_shape(value, dims)
    return modified_data

example_input = generate_data(lambda shape: numpy.random.randn(*shape).astype(numpy.float32),
                              lambda shape: numpy.zeros(shape).astype(numpy.float32))

output_original = ReferenceEvaluator(onnx_model).run(None, data_to_original_shape(example_input), intermediate=True)
output_modified = ReferenceEvaluator(modified_model).run(None, example_input, intermediate=True)

for k in output_original.keys() & output_modified.keys():
    if output_original[k] is None or output_modified[k] is None:
        continue

    modified_output_tensor = output_modified[k]
    s = [slice(0, dim) for dim in output_modified[k].shape]
    original_output_tensor = output_original[k][*s]

    diff_rms = numpy.sqrt(numpy.mean(numpy.square(original_output_tensor - modified_output_tensor)))
    if diff_rms != 0.0:
        print(f"{k}: {diff_rms}")
print(numpy.max(numpy.abs(output_modified["val_15"])))

preprocess.quant_pre_process(modified_model, f"{basename}_rewritten_p.onnx")


# Model input is in range [0.0, 256[
calibration_data = [
    generate_data(lambda shape: numpy.ones(shape).astype(numpy.float32) * 255, lambda shape: numpy.ones(shape).astype(numpy.float32)*0.9921875),
    generate_data(lambda shape: numpy.zeros(shape).astype(numpy.float32))]

for i in range(20):
    calibration_data.append(
        generate_data(lambda shape: numpy.random.uniform(0, 255, size=shape).astype(numpy.float32),
                      lambda shape: numpy.random.uniform(-1, 0.9921875, size=shape).astype(numpy.float32)))


class XXXDataReader(CalibrationDataReader):
    def __init__(self):
        self.enum_data = None
        pass

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(calibration_data)
        return next(self.enum_data, None)

    def rewind(self):
        pass

dr = XXXDataReader()

conf = StaticQuantConfig(
    calibration_data_reader=dr,
    quant_format=QuantFormat.QDQ,
    calibrate_method=CalibrationMethod.MinMax,
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
    #op_types_to_quantize=["Conv","Slice"],
    extra_options={
        "ForceQuantizeNoInputCheck":True,
        #"TensorQuantOverrides": {
        #    "gru_hidden_output_h_1": [
        #        {"symmetric": True}
        #    ],
        #    "lstm_hidden_input_h_1": [
        #        {"symmetric": True}
        #    ]
        #}
        #"ActivationSymmetric": True,
        #"TensorQuantOverrides": {
        #    "lstm_hidden_input_h_0": [
        #        {"symmetric": True}
        #    ],
        #    "lstm_hidden_input_h_1": [
        #        {"symmetric": True}
        #    ],
        #    "lstm_0_gemm_0": [
        #        {"symmetric": True}
        #    ],
        #    "lstm_0_gemm_1": [
        #        {"symmetric": True}
        #    ],
        #    "lstm_1_gemm_0": [
        #        {"symmetric": True}
        #    ],
        #    "lstm_1_gemm_1": [
        #        {"symmetric": True}
        #    ]
        #}
    },
    per_channel=True)
      
quantize(f"{basename}_rewritten_p.onnx", f"{basename}_rewritten_q.onnx", conf)
