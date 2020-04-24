#include "tflite_wrapper.h"

#include <iostream>
#include <memory>
#include <vector>

#include "label_utils.h"
#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/kernels/register.h"

namespace edge {

TfLiteWrapper::TfLiteWrapper(
    const std::string& model_path, const std::string& label_path, const float threshold,
    std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_context, const bool edgetpu) {
  m_model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  // Initialzes interpreter.
  if (edgetpu && edgetpu_context) {
    InitTfLiteWrapperEdgetpu(edgetpu_context);
  } else {
    InitTfLiteWrapper();
  }
  m_interpreter->SetNumThreads(1);
  m_interpreter->AllocateTensors();
  // Set input tensor shape.
  const auto* dims = m_interpreter->tensor(m_interpreter->inputs()[0])->dims;
  m_input_shape = {dims->data[0], dims->data[1], dims->data[2], dims->data[3]};
  // set output tensor shape.
  const auto& out_tensor_indices = m_interpreter->outputs();
  m_output_shape.resize(out_tensor_indices.size());
  for (size_t i = 0; i < out_tensor_indices.size(); i++) {
    const auto* tensor = m_interpreter->tensor(out_tensor_indices[i]);
    // We are assuming that outputs tensor are only of type float.
    m_output_shape[i] = tensor->bytes / sizeof(float);
  }
  m_labels = ParseLabel(label_path);
  m_threshold = threshold;
}

void TfLiteWrapper::InitTfLiteWrapperEdgetpu(
    std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_context) {
  tflite::ops::builtin::BuiltinOpResolver resolver;
  resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());
  std::cout << "edgetpu::RegisterCustomOp()\n";
  if (tflite::InterpreterBuilder(*m_model, resolver)(&m_interpreter) != kTfLiteOk) {
    std::cout << "Failed to build Interpreter\n";
    std::abort();
  }
  m_interpreter->SetExternalContext(kTfLiteEdgeTpuContext, edgetpu_context.get());
}

void TfLiteWrapper::InitTfLiteWrapper() {
  tflite::ops::builtin::BuiltinOpResolver resolver;
  if (tflite::InterpreterBuilder(*m_model, resolver)(&m_interpreter) != kTfLiteOk) {
    std::cout << "Failed to build Interpreter\n";
    std::abort();
  }
}

const std::vector<int> TfLiteWrapper::GetInputShape() {
  return m_input_shape;
}

const std::vector<InferenceResult> TfLiteWrapper::GetResults(
    const std::vector<std::vector<float>>& output) {
  std::vector<InferenceResult> results;
  int n = lround(output[3][0]);
  for (int i = 0; i < n; i++) {
    int id = lround(output[1][i]);
    float score = output[2][i];
    if (score > m_threshold) {
      InferenceResult result;
      result.candidate = m_labels.at(id);
      result.score = score;
      result.y1 = std::max(static_cast<float>(0.0), output[0][4 * i]);
      result.x1 = std::max(static_cast<float>(0.0), output[0][4 * i + 1]);
      result.y2 = std::min(static_cast<float>(1.0), output[0][4 * i + 2]);
      result.x2 = std::min(static_cast<float>(1.0), output[0][4 * i + 3]);
      results.push_back(result);
    }
  }
  return results;
}

const std::vector<InferenceResult> TfLiteWrapper::RunInference(
    const std::vector<uint8_t>& input_data) {
  std::vector<float> output_data;
  uint8_t* input = m_interpreter->typed_input_tensor<uint8_t>(0);
  std::memcpy(input, input_data.data(), input_data.size());
  m_interpreter->Invoke();

  const auto& output_indices = m_interpreter->outputs();
  const int num_outputs = output_indices.size();
  int out_idx = 0;
  for (int i = 0; i < num_outputs; ++i) {
    const auto* out_tensor = m_interpreter->tensor(output_indices[i]);
    assert(out_tensor != nullptr);
    if (out_tensor->type == kTfLiteFloat32) {
      const int num_values = out_tensor->bytes / sizeof(float);
      output_data.resize(out_idx + num_values);
      const float* output = m_interpreter->typed_output_tensor<float>(i);
      for (int j = 0; j < num_values; ++j) {
        output_data[out_idx++] = output[j];
      }
    } else {
      std::cerr << "Unsupported output type: " << out_tensor->type
                << "\n Tensor Name: " << out_tensor->name;
    }
  }

  const auto* result_raw = output_data.data();
  std::vector<std::vector<float>> results(m_output_shape.size());
  int offset = 0;
  for (size_t i = 0; i < m_output_shape.size(); ++i) {
    const size_t size_of_output_tensor_i = m_output_shape[i];
    results[i].resize(size_of_output_tensor_i);
    std::memcpy(results[i].data(), result_raw + offset, sizeof(float) * size_of_output_tensor_i);
    offset += size_of_output_tensor_i;
  }
  return GetResults(results);
}

}  // namespace edge
