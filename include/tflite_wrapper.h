#ifndef EDGETPU_TFLITE_CV_TFLITE_WRAPPER_H_
#define EDGETPU_TFLITE_CV_TFLITE_WRAPPER_H_

#include <array>
#include <map>
#include <string>

#include "edgetpu.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"

namespace edge {

// Represent inference results.
struct InferenceResult {
  std::string candidate;
  float score;
  float x1;
  float y1;
  float x2;
  float y2;
};

class TfLiteWrapper {
public:
  // Constructors.
  TfLiteWrapper(
      const std::string& model, const std::string& label_path, const float threshold,
      std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_context, const bool edgetpu);
  TfLiteWrapper() = delete;
  // Initializes a tflite::Interpreter for CPU usage.
  void InitTfLiteWrapper();
  // Initializes a tflite::Interpreter with edgetpu custom ops.
  void InitTfLiteWrapperEdgetpu(std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_context);
  // Runinference;
  const std::vector<InferenceResult> RunInference(const std::vector<uint8_t>& input_data);
  // Transform output tensors to a vector of InferneceResults.
  const std::vector<InferenceResult> GetResults(const std::vector<std::vector<float>>& output);
  // Exposes the input tensor shape.
  const std::vector<int> GetInputShape();
  // Destructor.
  ~TfLiteWrapper() = default;

private:
  std::unique_ptr<tflite::FlatBufferModel> m_model;
  std::unique_ptr<tflite::Interpreter> m_interpreter;
  std::vector<int> m_input_shape;
  std::vector<size_t> m_output_shape;
  std::map<int, std::string> m_labels;
  float m_threshold;
};

}  // namespace edge
#endif  // EDGETPU_TFLITE_CV_TFLITE_WRAPPER_H
