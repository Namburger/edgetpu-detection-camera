#include "detection_camera.h"

#include "edgetpu.h"
#include "opencv2/opencv.hpp"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"

namespace edge {

DetectionCamera::DetectionCamera(
    const std::string& model_path, const std::string& label_path, const float threshold,
    std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_context, const bool edgetpu, const int source,
    const int height, const int width, const bool verbose)
    : m_interpreter(model_path, label_path, threshold, edgetpu_context, edgetpu),
      m_camera(source),
      m_height(height),
      m_width(width),
      m_verbose(verbose) {}

void DetectionCamera::Run() {
  const auto& input_tensor_shape = m_interpreter.GetInputShape();
  const auto width = input_tensor_shape[1];
  const auto height = input_tensor_shape[2];

  // Initializing cameras.
  if (!m_camera.isOpened()) {
    std::cerr << "Unable to open camera!\n";
    exit(0);
  } else {
    m_camera.set(cv::CAP_PROP_FPS, 30.0f);
    m_camera.set(cv::CAP_PROP_FRAME_HEIGHT, m_height);
    m_camera.set(cv::CAP_PROP_FRAME_WIDTH, m_width);
    // In case user gives incorrect parameters, cv will re-adjust, we reset our values to fit cv.
    m_height = m_camera.get(cv::CAP_PROP_FRAME_HEIGHT);
    m_width = m_camera.get(cv::CAP_PROP_FRAME_WIDTH);
  }

  cv::Mat frame;
  for (;;) {
    m_camera.read(frame);
    if (!m_camera.read(frame)) break;  // Blank frame!
    ++m_frame_counter;
    cv::Mat resized_frame;
    // Converts image colors.
    cvtColor(frame, resized_frame, cv::COLOR_BGR2RGB);
    // Resize image to fit input tensors shape.
    cv::resize(resized_frame, resized_frame, cv::Size(width, height));
    std::vector<uint8_t> input(
        resized_frame.data,
        resized_frame.data + (resized_frame.cols * resized_frame.rows * resized_frame.elemSize()));

    const auto& candidates = m_interpreter.RunInference(input);
    for (const auto& candidate : candidates) {
      int top = static_cast<int>(candidate.y1 * m_height + 0.5f);
      int lft = static_cast<int>(candidate.x1 * m_width + 0.5f);
      int btm = static_cast<int>(candidate.y2 * m_height + 0.5f);
      int rgt = static_cast<int>(candidate.x2 * m_width + 0.5f);
      const auto& cvred = cv::Scalar(0, 0, 255);
      const auto& cvblue = cv::Scalar(255, 0, 0);
      const auto& c = "candidate: " + candidate.candidate;
      const auto& s = "score: " + std::to_string(candidate.score);

      cv::rectangle(
          frame, cv::Point(lft, top), cv::Point(rgt, btm), cvblue, 2, 1, 0);
      cv::putText(
          frame, c, cv::Point(lft, top - 25), cv::FONT_HERSHEY_COMPLEX, .8, cvred, 1.5, 8, 0);
      cv::putText(
          frame, s, cv::Point(lft, top - 5), cv::FONT_HERSHEY_COMPLEX, .8, cvred, 1.5, 8, 0);

      if (m_verbose) {
        std::cout << "\n-----\nFrame number: " << m_frame_counter << " " << c << " " << s
                  << "\ntop: " << top << " lft: " << lft << " btm: " << btm << " rgt: " << rgt;
      }
    }

    cv::imshow("Live Inference", frame);
    cv::waitKey(1);
  }
}

DetectionCamera::~DetectionCamera() {}

}  // namespace edge
