import tensorflow as tf
import sys
import preprocessing_wav
import onnx2tf
import os

def representative_dataset():
    return preprocessing_wav.generate_calibration_data(sys.argv[1], end_with_none=False)

basename = os.path.basename(os.path.splitext(sys.argv[1])[0])
tflite_model_path = f"tmp/{basename}_tflite"

onnx2tf.convert(input_onnx_file_path=sys.argv[1], output_folder_path=tflite_model_path)

converter = tf.lite.TFLiteConverter.from_saved_model(tflite_model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_quant_model = converter.convert()

with open(f"tmp/{basename}_int8.tflite", 'wb') as w:
  w.write(tflite_quant_model)

converter = tf.lite.TFLiteConverter.from_saved_model(tflite_model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter._experimental_disable_per_channel_quantization_for_dense_layers = True
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_quant_model = converter.convert()

with open(f"tmp/{basename}_int8_per_tensor_dense.tflite", 'wb') as w:
  w.write(tflite_quant_model)

converter = tf.lite.TFLiteConverter.from_saved_model(tflite_model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter._experimental_disable_per_channel = True
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_quant_model = converter.convert()

with open(f"tmp/{basename}_int8_per_tensor.tflite", 'wb') as w:
  w.write(tflite_quant_model)

converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
converter._experimental_disable_per_channel_quantization_for_dense_layers = True
converter.inference_input_type = tf.int16
converter.inference_output_type = tf.int16
tflite_quant_model = converter.convert()

with open(f"tmp/{basename}_int16_8_per_tensor_dense.tflite", 'wb') as w:
  w.write(tflite_quant_model)
