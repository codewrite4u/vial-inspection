

import sys
import cv2
import neoapi
from PIL import Image
import numpy as np
import os

try:
    # Connect to the cameras
    cameras = []
    camera_names = ["Body", "shoulder", "Neck", "Bottom", "inside"]
    
    for name in camera_names:
        camera = neoapi.Cam()
        camera.Connect(name)
        camera.f.ExposureTime.Set(600)
        cameras.append((name, camera))

    # Create a window for each camera
    window_names = [f'Camera: {name}' for name, _ in cameras]
    for window_name in window_names:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 600, 550)

    # Continuous loop for capturing and displaying images
    while True:
        for (name, camera), window_name in zip(cameras, window_names):
            try:
                # Capture image from the camera
                img = camera.GetImage()

                if not img.IsEmpty():
                    imgarray = img.GetNPArray()
                    imgarray = np.squeeze(imgarray, axis=2)
                    imgarray = np.stack((imgarray,) * 3, axis=-1)

                    input_img = Image.fromarray(imgarray.astype(np.uint8))

                    # Display the input image in the corresponding window
                    input_img_cv2 = np.array(input_img)
                    cv2.imshow(window_name, input_img_cv2)

            except neoapi.NeoException as exc:
                print(f'Camera Error ({name}):', exc)

        key = cv2.waitKey(1)
        if key == 27:  # Press 'Esc' key (ASCII value 27) to exit and close all windows
            break

    # Close all OpenCV windows before exiting
    cv2.destroyAllWindows()

except Exception as exc:
    print('Error:', exc)
    sys.exit(1)
