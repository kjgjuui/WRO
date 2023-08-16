# python3 -m pip install tflite-runtime
# python3 -m pip install teachable-machine-lite==0.0.1
import time
from teachable_machine_lite import TeachableMachineLite
from tflite_runtime.interpreter import Interpreter
import YB_Pcb_Car  # Import Yahboom car library
import cv2


car = YB_Pcb_Car.YB_Pcb_Car()
cap = cv2.VideoCapture(0)
model_path = 'model.tflite'
my_model = TeachableMachineLite(model_type='tflite', model_path=model_path)


def look_to_object():
    print('Start Look to object')
   
def say_cup():
    print('green')
   
def say_card():
    print('red')
    
def main():
    model_path = '/home/pi/Desktop/WRO/model.tflite'
    my_model = TeachableMachineLite(model_type='tflite', model_path=model_path)
    interpreter = Interpreter(model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    #####
    #interpreter = Interpreter(model_path)
    #interpreter.allocate_tensors()
    dim = my_model.get_image_dimensions(interpreter)
    height, width = dim['height'], dim['width']
    look_to_object()
    while True:
        ret, frame = cap.read()
        cv2.imshow('PiCam', frame)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imwrite("frame.jpg", image)
        img = cv2.resize(image, (width, height))
        my_model.transform_image(interpreter, img)
        interpreter.invoke()
        results = my_model.classify_image(interpreter)
        label_id = results['highest_class_id']
        if label_id == 0:
            say_cup()
        elif label_id == 1:
            say_card()
        look_to_object()
        time.sleep(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
    cap.release()
