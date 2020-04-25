import os, argparse, cv2, sys, time, numpy
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate
def load_label(path, encoding='utf-8'):
    with open(path, 'r', encoding=encoding) as f:
        lines = f.readlines()
        if not lines:
            return {}
        if lines[0].split(' ', maxsplit=1)[0].isdigit():
            pairs = [line.split(' ', maxsplit=1) for line in lines]
            return {int(index): label.strip() for index, label in pairs}
        else:
            return {index: line.strip() for index, line in enumerate(lines)}
def get_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Path to tflite model.', required=True)
    parser.add_argument('--labels', help='Path to label file.', required=True)
    parser.add_argument(
        '--threshold', help='Minimum confidence threshold.', default=0.5)
    parser.add_argument('--source', help='Video source.', default=0)
    parser.add_argument('--edgetpu', help='With EdgeTpu', default=False)
    return parser.parse_args()
def main():
    args = get_cmd()
    if args.edgetpu:
        interpreter = Interpreter(args.model, experimental_delegates=[
                                  load_delegate('libedgetpu.so.1.0')])
    else:
        interpreter = Interpreter(args.model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    width = input_details[0]['shape'][2]
    height = input_details[0]['shape'][1]
    labels = load_label(args.labels)
    cap = cv2.VideoCapture(args.source)
    image_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    image_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    while(True):
        ret, frame = cap.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = numpy.expand_dims(frame_resized, axis=0)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        classes = interpreter.get_tensor(output_details[1]['index'])[0]
        scores = interpreter.get_tensor(output_details[2]['index'])[0]
        for i in range(len(scores)):
            if ((scores[i] > args.threshold) and (scores[i] <= 1.0)):
                ymin = int(max(1, (boxes[i][0] * image_height)))
                xmin = int(max(1, (boxes[i][1] * image_width)))
                ymax = int(min(image_height, (boxes[i][2] * image_height)))
                xmax = int(min(image_width, (boxes[i][3] * image_width)))
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 4)
                object_name = labels[int(classes[i])]
                label = '%s: %d%%' % (object_name, int(scores[i]*100))
                labelSize, baseLine = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                label_ymin = max(ymin, labelSize[1] + 10)
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (
                    xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED)
                cv2.putText(frame, label, (xmin, label_ymin-7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.imshow('Object detector', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
