import sys
import cv2
import neoapi
from PIL import Image
import numpy as np
import torch
import time
import os

# Load the YOLO model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('C:/Users/pc/Downloads/yolov5-master', 'custom', path='D:/Mallesh/ML_Labels/moulded_ls/yolov5/runs/train/exp4/weights/best.pt', source='local')

def yolomodel(img):
    # Run inference
    results = model(img)
    return results

try:
    # Connect to the camera
    camera = neoapi.Cam()
    camera.Connect("Neck")
    camera.f.ExposureTime.Set(600)
    # Set window properties for input images
    input_window_name = 'Input Images'
    cv2.namedWindow(input_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(input_window_name, 800, 600)

    # Set window properties for output images
    output_window_name = 'Output Images'
    cv2.namedWindow(output_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(output_window_name, 800, 600)

    # Initialize counters for accepted and rejected vials
    accepted_vial_count = 0
    rejected_vial_count = 0

    # Create directories to save accepted and rejected images
    #accepted_dir = "accepted_images"
    #rejected_dir = "rejected_images"
    #os.makedirs(accepted_dir, exist_ok=True)
    #os.makedirs(rejected_dir, exist_ok=True)

    # Main loop for capturing and processing images
    image_counter = 0  # Counter variable for tracking the number of processed images in the current vial
    vial_counter = 0   # Counter variable for tracking the number of vials processed

    while True:
        # Capture an image from the camera
        img = camera.GetImage()

        if not img.IsEmpty():
            img_array = img.GetNPArray()
            img_array = np.squeeze(img_array, axis=2)
            img_array = np.stack((img_array,) * 3, axis=-1)

            input_img = Image.fromarray(img_array.astype(np.uint8))

            # Perform prediction with your YOLO model
            results = yolomodel(input_img)

            # Extract the detected image from the results
            detected_img = results.render()[0]

            if results.xyxy[0].shape[0] > 0:
                # If defects are detected, save the image to the rejected folder
                #save_path = os.path.join(rejected_dir, f"rejected_{vial_counter}.png")
                #cv2.imwrite(save_path, cv2.cvtColor(np.array(detected_img), cv2.COLOR_RGB2BGR))
                image_counter += 1
            else:
                # If no defects are detected, save the image to the accepted folder
                #save_path = os.path.join(accepted_dir, f"accepted_{vial_counter}.png")
                #cv2.imwrite(save_path, cv2.cvtColor(np.array(detected_img), cv2.COLOR_RGB2BGR))
                image_counter += 1

            # Display the input and detected images
            cv2.imshow(input_window_name, np.array(input_img))
            cv2.imshow(output_window_name, cv2.cvtColor(np.array(detected_img), cv2.COLOR_RGB2BGR))

            # Check after processing 6 images (one vial)
            if image_counter >= 6:
                if results.xyxy[0].shape[0] > 0:
                    # If defects are detected in any of the 6 images, increment the rejected vial count
                    rejected_vial_count += 1
                    print(f"Vial {vial_counter} completed with REJECTED images.")
                else:
                    # If no defects are detected in all 6 images, increment the accepted vial count
                    accepted_vial_count += 1
                    print(f"Vial {vial_counter} completed with ACCEPTED images.")
                
                # Reset the image counter for the next vial
                image_counter = 0
                vial_counter += 1

        key = cv2.waitKey(1)
        if key == 27:  # Press 'Esc' to exit the program
            break
        time.sleep(0.1)

    cv2.destroyAllWindows()

    # Print the final vial counts
    print(f"Accepted Vials: {accepted_vial_count}")
    print(f"Rejected Vials: {rejected_vial_count}")

except (neoapi.NeoException, Exception) as exc:
    print('Error:', exc)
    sys.exit(1)
