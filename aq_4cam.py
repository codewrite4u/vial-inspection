import sys
import cv2
import neoapi
from PIL import Image
import numpy as np
import torch
import time

# Load the model
model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5-master/runs/train/exp12/weights/best.pt') #this for all defects
def yolomodel(img):
    # Run inference
    start_time = time.time()
    results = model(img)
    results.show()
    end_time = time.time()

    processing_time = (end_time - start_time)*1000
    print("Processing Time: {:.2f} milliseconds".format(processing_time))
    
    #cv2.imshow(window_name, np.array(results))

    for obj in results.xyxy[0]:
        class_name = model.names[int(obj[5])]
        return print(class_name)
        # return results.show()
    else:
        return print("No defect detected")

try:
    # Connect to the four cameras
    camera1 = neoapi.Cam()
    camera1.Connect("VCXU.2-50M")
    camera1.f.ExposureTime.Set(10000)

    camera2 = neoapi.Cam()
    camera2.Connect("VCXU-50M")
    camera2.f.ExposureTime.Set(10000)

##    camera3 = neoapi.Cam()
##    camera3.Connect()
##    camera3.f.ExposureTime.Set(10000)
##
##    camera4 = neoapi.Cam()
##    camera4.Connect()
##    camera4.f.ExposureTime.Set(10000)

    # Set window properties
    #window_name = 'Camera Feed'
    #cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    #cv2.resizeWindow(window_name, 800, 600)

    # Main loop for capturing and processing images
    counter = 0  # Counter variable for tracking the number of processed images
    while True:
        # Capture image from camera 1
        img1 = camera1.GetImage()
        if not img1.IsEmpty():
            imgarray1 = img1.GetNPArray()
            imgarray1 = np.squeeze(imgarray1, axis=2)  # Remove the single channel dimension
            imgarray1 = np.stack((imgarray1,) * 3, axis=-1)  # Duplicate the channel axis
            img = Image.fromarray(imgarray1.astype(np.uint8))
            yolomodel(img)
            counter += 1  # Increment the counter after processing each image
            #img.save(f"camera1_image_{counter}.png")

        #time.sleep(0.3)  # Delay of 0.3 seconds after capturing each image

        # Capture image from camera 2
        img2 = camera2.GetImage()
        if not img2.IsEmpty():
            imgarray2 = img2.GetNPArray()
            imgarray2 = np.squeeze(imgarray2, axis=2)
            imgarray2 = np.stack((imgarray2,) * 3, axis=-1)
            img = Image.fromarray(imgarray2.astype(np.uint8))
            yolomodel(img)
            counter += 1
            #img.save(f"camera2_image_{counter}.png")

        #time.sleep(0.3)

        # Capture image from camera 3
##        img3 = camera3.GetImage()
##        if not img3.IsEmpty():
##            imgarray3 = img3.GetNPArray()
##            imgarray3 = np.squeeze(imgarray3, axis=2)
##            imgarray3 = np.stack((imgarray3,) * 3, axis=-1)
##            img = Image.fromarray(imgarray3.astype(np.uint8))
##            yolomodel(img)
##            counter += 1
##            #img.save(f"camera3_image_{counter}.png")
##
##        #time.sleep(0.3)
##
##        # Capture image from camera 4
##        img4 = camera4.GetImage()
##        if not img4.IsEmpty():
##            imgarray4 = img4.GetNPArray()
##            imgarray4 = np.squeeze(imgarray4, axis=2)
##            imgarray4 = np.stack((imgarray4,) * 3, axis=-1)
##            img = Image.fromarray(imgarray4.astype(np.uint8))
##            yolomodel(img)
##            counter += 1
            #img.save(f"camera4_image_{counter}.png")

        #time.sleep(0.3)

        if counter >= 10:  # Break the loop after processing 100 images
            break

        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()

except (neoapi.NeoException, Exception) as exc:
    print('error: ', exc)
    sys.exit(1)
