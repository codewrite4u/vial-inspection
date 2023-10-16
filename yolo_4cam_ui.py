import sys
import cv2
import neoapi
from PIL import Image
import numpy as np
import torch
import time

# Load the model outside the yolomodel function
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('C:/Users/pc/Downloads/yolov5-master', 'custom', path = 'D:/Mallesh/ML_Labels/moulded_ls/yolov5/runs/train/exp4/weights/best.pt', source = 'local')

def yolomodel(img):
    # Run inference
    start_time = time.time()
    results = model(img)
    end_time = time.time()

    processing_time = (end_time - start_time)*1000
    print("Processing Time: {:.2f} milliseconds".format(processing_time))

    return results

try:
    # Connect to the cameras
    cameras = []
    camera_names = ["Top", "Shoulder", "Neck", "Body", "Bottom"]#acA2040-120um
    #camera_names = ["VCXU.2-50M", "VCXU.2-50M"]
    for i in camera_names:
        camera = neoapi.Cam()
        camera.Connect(i)
        camera.f.ExposureTime.Set(600)#8000
        cameras.append(camera) 

    # Set window properties for input images
    numberofcam = len(camera_names)
    input_window_names = ['Input Image {}'.format(i+1) for i in range(numberofcam)]
    for window_name in input_window_names:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 450, 450)

    # Set window properties for output images
    output_window_names = ['Output Image {}'.format(i+1) for i in range(numberofcam)]
    for window_name in output_window_names:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 450, 450)

    # Main loop for capturing and processing images
    counter = 0  # Counter variable for tracking the number of processed images
    while True:
        for i, camera in enumerate(cameras):
            try:
                # Capture image from the camera
                img = camera.GetImage()

                if not img.IsEmpty():
                    imgarray = img.GetNPArray()
                    imgarray = np.squeeze(imgarray, axis=2)  # Remove the single channel dimension
                    imgarray = np.stack((imgarray,) * 3, axis=-1)  # Duplicate the channel axis

                    input_img = Image.fromarray(imgarray.astype(np.uint8))

                    # Perform prediction with your deep learning model
                    results = yolomodel(input_img)

                    # Display the input image in the input window
                    input_img_cv2 = np.array(input_img)
                    cv2.imshow(input_window_names[i], input_img_cv2)

                    # Extract the detected image from the results
                    detected_img = results.render()[0]
                    
                    # Convert the detected image to BGR format for OpenCV
                    detected_img_cv2 = cv2.cvtColor(np.array(detected_img), cv2.COLOR_RGB2BGR)

                    # Display the detected image in the output window
                    cv2.imshow(output_window_names[i], detected_img_cv2)

                    # Print the detected class names
                    if results.xyxy[0].shape[0] > 0:      
                        for obj in results.xyxy[0]:
                            class_name = model.names[int(obj[5])]
                            print(f"Detected Class: {class_name}")
                    else:
                        print("No defect detected")

                    counter += 1  # Increment the counter after processing each image
                    if counter >= 1000:  # Break the loop after processing 50 images
                        break
                    #time.sleep(0.10)
            except neoapi.NeoException as exc:
                print('Camera Error:', exc)
                # Disconnect the camera and remove it from the list
                #camera.Disconnect()
                #cameras.remove(camera)

        key = cv2.waitKey(1)
        if key == ord('0'):  # Press '0' to close all input image windows
            for window_name in input_window_names:
                cv2.destroyWindow(window_name)
        elif key == ord('1'):  # Press '1' to close all output image windows
            for window_name in output_window_names:
                cv2.destroyWindow(window_name)
        elif key == 27:  # Press 'Esc' to exit the program
            break
        

    cv2.destroyAllWindows()
except Exception as exc:
    print('Error:', exc)
    sys.exit(1)


