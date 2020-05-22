#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <ostream>
#include <regex>
#include <string>

#include "cxxopts.hpp"
#include "detection_camera.h"
#include "edgetpu.h"
#include "opencv2/opencv.hpp"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "tflite_wrapper.h"

cxxopts::ParseResult parse_args(int argc, char** argv) {
  cxxopts::Options options(
      "edge_tflite_cv", "An example of using opencv with tflite/edgetpu in c++.");
  // clang-format off
  options.add_options()
    ("model_path", "Path to .tflite model_file", cxxopts::value<std::string>())
    ("label_path", "Path to label file.", cxxopts::value<std::string>())
    ("video_source", "Video source.", cxxopts::value<int>()->default_value("0"))
    ("threshold", "Minimum confidence threshold.", cxxopts::value<float>()->default_value(".5"))
    ("verbose", "To run in verbose mode.", cxxopts::value<bool>()->default_value("false"))
    ("edgetpu", "To run with EdgeTPU.", cxxopts::value<bool>()->default_value("false"))
    ("height", "Camera image height.", cxxopts::value<int>()->default_value("480"))
    ("width", "Camera image width.", cxxopts::value<int>()->default_value("640"))
    ("help", "Print Usage");
  // clang-format on

  const auto& args = options.parse(argc, argv);
  if (args.count("help") || !args.count("model_path") || !args.count("label_path")) {
    std::cerr << options.help() << "\n";
    exit(0);
  }
  return args;
}

int main(int argc, char** argv) {
  const auto& args = parse_args(argc, argv);
  // Building Interpreter.
  const auto& model_path = args["model_path"].as<std::string>();
  const auto& label_path = args["label_path"].as<std::string>();
  const auto threshold = args["threshold"].as<float>();
  const auto with_edgetpu = args["edgetpu"].as<bool>();
  auto image_height = args["height"].as<int>();
  auto image_width = args["width"].as<int>();
  const auto source = args["video_source"].as<int>();
  const bool verbose = args["verbose"].as<bool>();

  // Get edgetpu context.
  std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_context =
      edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();
  // Creates the detection camera instance.
  edge::DetectionCamera dc(
      model_path, label_path, threshold, edgetpu_context, with_edgetpu, source, image_height,
      image_width, verbose);
  dc.Run();

  return 0;
}
