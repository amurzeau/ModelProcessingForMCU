from onnxruntime.quantization.qdq_loss_debug import create_weight_matching, create_activation_matching, compute_activation_error, collect_activations, modify_model_output_intermediate_tensors
from onnxruntime.quantization import CalibrationDataReader
import onnxruntime

import numpy as np
import matplotlib.pyplot as plt
import preprocessing_wav
import add_intermediate_layers_as_output
import onnx

def run_model(model_path, input_reader):
	session_options = onnxruntime.SessionOptions()
	session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL

	inference_session = onnxruntime.InferenceSession(
		model_path,
		sess_options=session_options,
		providers=["CPUExecutionProvider"],
	)

	intermediate_outputs = []
	for input_d in input_reader:
		intermediate_outputs.append(inference_session.run(None, input_d))
	if not intermediate_outputs:
		raise RuntimeError("No data is collected while running augmented model!")

	output_dict = {}
	output_info = inference_session.get_outputs()
	for batch in intermediate_outputs:
		for output, output_data in zip(output_info, batch, strict=False):
			output_dict.setdefault(output.name, []).append(output_data)

	return output_dict


class XXXDataReader(CalibrationDataReader):
	def __init__(self):
		self.data = preprocessing_wav.generate_calibration_data("denoiser_dns_rewritten_p_intermediate.onnx")
		for i in range(100):
			next(self.data)
		self.done = False
		pass

	def get_next(self):
		if self.done:
			return None
		else:
			self.done = True
			return next(self.data)

	def rewind(self):
		pass

float_model = onnx.load("denoiser_dns_rewritten_p.onnx")
float_model = add_intermediate_layers_as_output.modify_model_test(float_model, tensor_name_to_include=["onnx::Sigmoid_173", "nn_out.3.weight", "input.28"])
onnx.save(float_model, "denoiser_dns_rewritten_p_intermediate.onnx")

float_model = onnx.load("denoiser_dns_rewritten_q.onnx")
float_model = add_intermediate_layers_as_output.modify_model_test(float_model, tensor_name_to_include=["onnx::Sigmoid_173", "onnx::Sigmoid_173_DequantizeLinear_Output", "nn_out.3.weight", "nn_out.3.weight_DequantizeLinear_Output", "input.28", "input.28_DequantizeLinear_Output"])
onnx.save(float_model, "denoiser_dns_rewritten_q_intermediate.onnx")

float_activations = run_model("denoiser_dns_rewritten_p_intermediate.onnx", XXXDataReader())
qdq_activations = run_model("denoiser_dns_rewritten_q_intermediate.onnx", XXXDataReader())


w = create_weight_matching("denoiser_dns_rewritten_p.onnx", "denoiser_dns_rewritten_q.onnx")
a = create_activation_matching(qdq_activations, float_activations)

print(float_activations.keys())
print(qdq_activations.keys())
print(a.keys())

act_error = compute_activation_error(a)
for act_name, err in act_error.items():
	print(f"Cross model error of '{act_name}': {err['xmodel_err']}")
	print(f"QDQ error of '{act_name}': {err['qdq_err']}")

#w_to_compare = w["nn_out.3.weight"]
a_to_compare = a["onnx::Sigmoid_173"]

#w_diff = np.abs(w_to_compare["float"] - w_to_compare["dequantized"])

#print(f"Diff min: {np.min(w_diff)}, max: {np.max(w_diff)}")

#plt.figure()
#plt.imshow(w_diff)
#plt.show()

a_diff = np.abs(np.clip(a_to_compare["float"][0], -10, 10) - np.clip(a_to_compare["pre_qdq"][0], -10, 10))

print(f"Diff min: {np.min(a_diff)}, max: {np.max(a_diff)}")

plt.figure()
plt.plot(a_to_compare["float"][0].flatten())
plt.plot(a_to_compare["pre_qdq"][0].flatten())
plt.plot(a_to_compare["post_qdq"][0].flatten())
plt.show()
