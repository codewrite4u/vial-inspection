import sys
import cv2
import neoapi
from PIL import Image
import numpy as np
import torch
import time
import matplotlib.pyplot as plt

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('C:/Users/pc/Downloads/yolov5-master', 'custom', path = 'D:/Mallesh/stations/ssv/runs/train/exp4/weights/best.pt', source = 'local')

"""
def yolomodel(img):
    # Run inference
    start_time = time.time()
    results = model(img)
    end_time = time.time()

    processing_time = (end_time - start_time)*1000
    print("Processing Time: {:.2f} milliseconds".format(processing_time))

    return results

"""
try:
    # Connect to the first camera
    camera1 = neoapi.Cam()
    camera1.Connect("Body")
    camera1.f.ExposureTime.Set(600)

    # Set window properties for input images
    input_window_name = 'Input Images'
    cv2.namedWindow(input_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(input_window_name, 800, 600)

    # Set window properties for output images
    output_window_name = 'Output Images'
    cv2.namedWindow(output_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(output_window_name, 800, 600)

    # Main loop for capturing and processing images
    counter = 0  # Counter variable for tracking the number of processed images
    while True:
        # Capture image from the first camera
        img1 = camera1.GetImage()

        if not img1.IsEmpty():
            imgarray1 = img1.GetNPArray()
            imgarray1 = np.squeeze(imgarray1, axis=2)  # Remove the single channel dimension
            imgarray1 = np.stack((imgarray1,) * 3, axis=-1)  # Duplicate the channel axis

            input_img = Image.fromarray(imgarray1.astype(np.uint8))

            # Perform prediction with your deep learning model
            start_time = time.time()
            results = model(input_img)
            end_time = time.time()

            processing_time = (end_time - start_time)*1000
            print("Processing Time: {:.2f} milliseconds".format(processing_time))
    
            # Display the input image in the input window
            input_img_cv2 = np.array(input_img)
            cv2.imshow(input_window_name, input_img_cv2)

            # Extract the detected image from the results
            detected_img = results.render()[0]

            # Convert the detected image to BGR format for OpenCV
            detected_img_cv2 = cv2.cvtColor(np.array(detected_img), cv2.COLOR_RGB2BGR)

            # Display the detected image in the output window
            cv2.imshow(output_window_name, detected_img_cv2)
            #save_path = f"detected_image_{counter}.png"
            #cv2.imwrite(save_path, detected_img_cv2)

            if results.xyxy[0].shape[0]>0:
                    #pixel_to_mm = 0.1      
                    for obj in results.xyxy[0]:
                        class_name = model.names[int(obj[5])]
                        print(f"Detected Class: {class_name}")
                        x_min, y_min, x_max, y_max = obj[0], obj[1], obj[2], obj[3]
                        width = x_max - x_min
                        height = y_max - y_min

                        print(width)
                        print(height)

                        width1 = max(width, height)

                        # Check if the detected defect is labeled as "bubble" and its width is less than 30 mm
                        if class_name == "Bubble" and width1 < 40:
                            print(f"Accepted: {width1}")
                            # Handle the accepted defect here
                        else:
                            print(f"Rejected - : {width1}")
                        # Calculate area in square millimeters
                        #area_mm2 = (width * pixel_to_mm) * (height * pixel_to_mm)
                        #print(f"Bounding Box Area (mm^2): {area_mm2}")

                        area = width * height
                        print(f"Bounding Box Area: {area}")
            else:
                    print("No defect detected")

            counter += 1  # Increment the counter after processing each image
            if counter >= 1000:  # Break the loop after processing 500 images
                break

        key = cv2.waitKey(1)
        if key == ord('0'):  # Press '0' to close the input image window
            cv2.destroyWindow(input_window_name)
        elif key == ord('1'):  # Press '1' to close the output image window
            cv2.destroyWindow(output_window_name)
        elif key == 27:  # Press 'Esc' to exit the program
            break
        time.sleep(0.1)

    cv2.destroyAllWindows()

except (neoapi.NeoException, Exception) as exc:
    print('Error:', exc)
    sys.exit(1)
