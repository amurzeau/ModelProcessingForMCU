import sys
import os
from onnxscript.rewriter import pattern, rewrite
from onnxscript import ir
from onnxscript.ir._tape import Builder
import onnx
import numpy
from onnx import numpy_helper, version_converter
import model_compare

from google.protobuf.internal.containers import (  # type: ignore
    RepeatedCompositeFieldContainer,
)

from onnxruntime.quantization import preprocess
from onnxruntime.tools.onnx_model_utils import make_dim_param_fixed


def split(op, input: ir.Value, num_outputs, name_base=None):
    if name_base is None:
        name_base = input.name
    split_outputs = [f"{name_base}_split_{i}" for i in range(num_outputs)]
    t = op.Split(input, axis=1, _outputs=split_outputs)
    return t


# Rewrite LSTM with Gemm
index = 0


def LSTM(op, X, W, R, B, sequence_lens, initial_h, initial_c, axes):
    return op.Squeeze(op.LSTM(X, W, R, B, sequence_lens, initial_h, initial_c, direction="forward", hidden_size=256, input_forget=0), axes)


def LSTMCond(op, X, W, R, B, sequence_lens, initial_h, initial_c, axes: ir.Value):
    return axes.const_value.numpy() == numpy.array([1])


def LoweredLSTM(op: Builder, X: ir.Value,
                W: ir.Value,
                R: ir.Value,
                B: ir.Value,
                sequence_lens: ir.Value, initial_h: ir.Value, initial_c: ir.Value, axes):
    global index
    global onnx_model_ir

    zero_value_tensor = op.initializer(ir.tensor([0], name=f"zero_const"))

    x = op.Squeeze(X, zero_value_tensor)
    w = op.Squeeze(W, zero_value_tensor)
    r = op.Squeeze(R, zero_value_tensor)
    Wb, Rb = split(op, B, 2)
    wb_plus_rb = op.Squeeze(op.Add(Wb, Rb), zero_value_tensor)
    # x = 1, 257
    # w = 257, 1024
    # wb_plus_rb = 1, 1024
    # Output = 1, 1024 = x * w + wb_plus_rb
    # gemm1 = Xt*(Wi^T) for i, o, f, c
    gemm1: ir.Value = op.Gemm(
        x, w, wb_plus_rb, alpha=1.0, beta=1.0, transA=0, transB=1)
    gemm1.name = f"lstm_{index}_gemm_0"

    Y_h_input = ir.Value(name=f"lstm_hidden_input_h_{index}", type=ir.TensorType(
        ir.DataType.FLOAT), shape=ir.Shape([1, 256]))
    Y_c_input = ir.Value(name=f"lstm_hidden_input_c_{index}", type=ir.TensorType(
        ir.DataType.FLOAT), shape=ir.Shape([1, 256]))

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

    Y_c = op.Tanh(op.Add(op.Mul(Y_c_input, ft), op.Mul(it, ct)),
                  _outputs=[Y_c_output])
    Y_c.type = ir.TensorType(ir.DataType.FLOAT)
    Y_c.shape = ir.Shape([1, 256])

    Y_h = op.Mul(Y_c, ot, _outputs=[Y_h_output])
    Y_h.type = ir.TensorType(ir.DataType.FLOAT)
    Y_h.shape = ir.Shape([1, 256])

    onnx_model_ir.graph.outputs.append(Y_c)
    onnx_model_ir.graph.outputs.append(Y_h)

    output = op.Unsqueeze(Y_h, zero_value_tensor)

    index = index + 1

    return output


# Rewrite GRU with Gemm
def GRU(op, X, W, R, B, sequence_lens, initial_h, hidden_size, linear_before_reset, axes):
    return op.Squeeze(op.GRU(X, W, R, B, sequence_lens, initial_h, direction="forward", hidden_size=hidden_size, linear_before_reset=linear_before_reset), axes)


def GRUCond(op, X, W, R, B, sequence_lens, initial_h, hidden_size, linear_before_reset, axes: ir.Value):
    return axes.const_value.numpy() == numpy.array([1])


const_1 = None
const_minus_1 = None


def LoweredGRU(op: Builder, X: ir.Value,
               W: ir.Value,
               R: ir.Value,
               B: ir.Value,
               sequence_lens: ir.Value,
               initial_h: ir.Value,
               hidden_size: ir.Attr,
               linear_before_reset: ir.Attr,
               axes):
    global index
    global onnx_model_ir
    global const_1
    global const_minus_1

    hidden_size_val = hidden_size.as_int()

    zero_value_tensor = op.initializer(ir.tensor([0], name=f"zero_const"))

    x = op.Squeeze(X, zero_value_tensor)
    w = op.Squeeze(W, zero_value_tensor)
    r = op.Squeeze(R, zero_value_tensor)
    b = op.Squeeze(B, zero_value_tensor)

    # Bias must be rank 1 to avoid this error:
    # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()

    split_outputs = [f"gru_{index}_split_b_{i}" for i in range(2)]

    b_splitted = op.Split(b, axis=0, _outputs=split_outputs)
    Wb, Rb = b_splitted

    Y_h_input = ir.Value(name=f"gru_hidden_input_h_{index}", type=ir.TensorType(
        ir.DataType.FLOAT), shape=ir.Shape([1, hidden_size_val]))

    Y_h_output = f"gru_hidden_output_h_{index}"

    onnx_model_ir.graph.inputs.append(Y_h_input)

    if linear_before_reset.as_int() == 0:
        gemmX = op.Gemm(x, w, op.Add(Wb, Rb), alpha=1.0,
                        beta=1.0, transA=0, transB=1)
        gemmY = op.Gemm(Y_h_input, r, alpha=1.0, beta=1.0, transA=0, transB=1)
    else:
        gemmX = op.Gemm(x, w, Wb, alpha=1.0, beta=1.0, transA=0, transB=1)
        gemmY = op.Gemm(Y_h_input, r, Rb, alpha=1.0,
                        beta=1.0, transA=0, transB=1)

    split_outputs = [f"gru_{index}_split_gemmX_{i}" for i in range(2)]
    gru_x_split = op.initializer(
        ir.tensor([hidden_size_val*2, hidden_size_val], name=f"gru_x_split"))
    gemmX_splitted = op.Split(
        gemmX, gru_x_split, axis=1, _outputs=split_outputs)
    zrtX, htX = gemmX_splitted

    split_outputs = [f"gru_{index}_split_gemmY_{i}" for i in range(2)]
    gemmY_splitted = op.Split(
        gemmY, gru_x_split, axis=1, _outputs=split_outputs)
    zrtY, htY = gemmY_splitted

    zrt = op.Sigmoid(op.Add(zrtX, zrtY))

    split_outputs = [f"gru_{index}_split_zrt_{i}" for i in range(2)]
    zrt_splitted = op.Split(zrt, axis=1, _outputs=split_outputs)
    zt, rt = zrt_splitted

    ht = op.Tanh(op.Add(htX, op.Mul(htY, rt)))

    # Ht = (1 - zt) (.) ht + zt (.) Ht-1
    if const_1 is None:
        const_1 = op.initializer(
            ir.tensor([numpy.float32(1)], name=f"const_1"))
    if const_minus_1 is None:
        const_minus_1 = op.initializer(
            ir.tensor([numpy.float32(-1)], name=f"const_minus_1"))
    Ht = op.Add(op.Mul(op.Add(const_1, op.Mul(zt, const_minus_1)),
                ht), op.Mul(zt, Y_h_input), _outputs=[Y_h_output])
    Ht.type = ir.TensorType(ir.DataType.FLOAT)
    Ht.shape = ir.Shape([1, hidden_size_val])

    onnx_model_ir.graph.outputs.append(Ht)

    output = op.Unsqueeze(Ht, zero_value_tensor)

    index = index + 1

    return output


def TransposeSqueeze(op, X, perm, axes):
    return op.Squeeze(op.Transpose(X, perm=perm), axes)


def TransposeSqueezeCond(op, X, perm, axes: ir.Value):
    perm = perm.as_ints()
    axes = axes.const_value.numpy()
    if perm[0] == 2 and perm[1] == 0 and perm[2] == 1 and axes[0] == 0:
        return True
    return False


def TransposeSqueezeReplace(op: Builder, X, perm: ir.Attr, axes: ir.Value):
    idx = axes.const_value.numpy()[0]
    idx = perm.as_ints()[idx]
    axes_tensor = op.initializer(ir.tensor([idx], name=f"{X.name}_squeeze"))
    return op.Squeeze(X, axes_tensor)


def UnsqueezeTranspose(op, X, perm, axes):
    return op.Transpose(op.Unsqueeze(X, axes), perm=perm)


def UnsqueezeTransposeCond(op, X, perm, axes: ir.Value):
    perm = perm.as_ints()
    axes = axes.const_value.numpy()
    print(perm, axes)
    if perm[0] == 1 and perm[1] == 2 and perm[2] == 0 and axes[0] == 0:
        return True
    return False


def UnsqueezeTransposeReplace(op: Builder, X, perm: ir.Attr, axes: ir.Value):
    print(perm, axes)
    axes_tensor = op.initializer(ir.tensor([2], name=f"{X.name}_unsqueeze"))
    return op.Unsqueeze(X, axes_tensor)


def UnSqueezeSqueeze(op, X, axes0, axes1):
    return op.Squeeze(op.Unsqueeze(X, axes=axes0), axes=axes1)


def UnSqueezeSqueezeCond(op, X, axes0: ir.Value, axes1: ir.Value):
    return axes0.const_value.numpy() == axes1.const_value.numpy()


def UnSqueezeSqueezeReplace(op, X, axes0: ir.Value, axes1: ir.Value):
    return X


def apply_rewrite(model):
    rule = [
        pattern.RewriteRule(
            LSTM,  # Target Pattern
            LoweredLSTM,  # Replacement Pattern
            LSTMCond,  # condition_function,
            name="LSTM",
            # verbose=10
        ),
        pattern.RewriteRule(
            GRU,  # Target Pattern
            LoweredGRU,  # Replacement Pattern
            GRUCond,  # condition_function,
            name="GRU",
            verbose=10
        ),
        pattern.RewriteRule(
            TransposeSqueeze,  # Target Pattern
            TransposeSqueezeReplace,  # Replacement Pattern
            TransposeSqueezeCond,  # condition_function,
            name="TransposeSqueeze",
            # verbose=10
        ),
        pattern.RewriteRule(
            UnSqueezeSqueeze,  # Target Pattern
            UnSqueezeSqueezeReplace,  # Replacement Pattern
            UnSqueezeSqueezeCond,  # condition_function,
            name="UnSqueezeSqueeze",
            # verbose=10
        ),
        pattern.RewriteRule(
            UnsqueezeTranspose,  # Target Pattern
            UnsqueezeTransposeReplace,  # Replacement Pattern
            UnsqueezeTransposeCond,  # condition_function,
            name="UnsqueezeTranspose",
            # verbose=10
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
                new_slice = [slice(0, dim) for dim in new_dim]
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
base_model_path = f"tmp/{basename}_b.onnx"
processed_model_path = f"tmp/{basename}_p.onnx"
os.makedirs("tmp", exist_ok=True)

onnx_model = onnx.load(sys.argv[1])
onnx_model = version_converter.convert_version(onnx_model, 13)
make_dim_param_fixed(onnx_model.graph, "batch_size", 1)
modified_model = onnx.load(sys.argv[1])
modified_model = version_converter.convert_version(modified_model, 13)
make_dim_param_fixed(modified_model.graph, "batch_size", 1)

crop_tensor_from_257_to_256(modified_model.graph)

preprocess.quant_pre_process(modified_model, base_model_path)

modified_model = onnx.load(base_model_path)

onnx_model_ir: ir.Model = ir.serde.deserialize_model(modified_model)
onnx_model_ir = apply_rewrite(onnx_model_ir)
modified_model = ir.serde.serialize_model(onnx_model_ir)
onnx.save(modified_model, processed_model_path)
modified_model = onnx.shape_inference.infer_shapes(
    modified_model, True, True, True)

onnx.checker.check_model(modified_model)
preprocess.quant_pre_process(modified_model, processed_model_path)

# Validate model execution
print(f"Model diff {sys.argv[1]}, {processed_model_path}")
results = model_compare.compare_models(onnx_model, modified_model)
for k, diff_rms in results.items():
    if diff_rms != 0.0:
        print(f"{k}: {diff_rms}")

