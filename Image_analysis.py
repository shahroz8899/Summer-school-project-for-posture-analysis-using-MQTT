import time
import logging
import os
import cv2
import numpy as np
import math
import threading
from receiver import start_mqtt_receiver
from config_reader import config_reader
from scipy.ndimage import gaussian_filter
from model import get_testing_model
import util


# Configuration
received_folder = 'received_images'
processed_folder = 'processed_images'
model_weights = './model/keras/model.h5'

# Set up logging
logging.basicConfig(filename='posture_analysis.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')

# Initialize variables
model = None
params = None
model_params = None
colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
          [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
          [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

def initialize_model():
    global model, params, model_params
    model = get_testing_model()
    model.load_weights(model_weights)
    params, model_params = config_reader()


def process(input_image):
    ''' Start of finding the Key points of full body using Open Pose.'''
    oriImg = cv2.imread(input_image)  # B,G,R order
    multiplier = [x * model_params['boxsize'] / oriImg.shape[0] for x in params['scale_search']]
    heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
    paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))

    for m in range(1):
        scale = multiplier[m]
        imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model_params['stride'],
                                                          model_params['padValue'])
        input_img = np.transpose(np.float32(imageToTest_padded[:, :, :, np.newaxis]), (3, 0, 1, 2))  # required shape (1, width, height, channels)
        output_blobs = model.predict(input_img)
        heatmap = np.squeeze(output_blobs[1])  # output 1 is heatmaps
        heatmap = cv2.resize(heatmap, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                             interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
        paf = np.squeeze(output_blobs[0])  # output 0 is PAFs
        paf = cv2.resize(paf, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                         interpolation=cv2.INTER_CUBIC)
        paf = paf[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        paf = cv2.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
        heatmap_avg = heatmap_avg + heatmap / len(multiplier)
        paf_avg = paf_avg + paf / len(multiplier)

    all_peaks = []  # To store all the key points which are detected.
    peak_counter = 0

    for part in range(18):
        map_ori = heatmap_avg[:, :, part]
        map = gaussian_filter(map_ori, sigma=3)

        map_left = np.zeros(map.shape)
        map_left[1:, :] = map[:-1, :]
        map_right = np.zeros(map.shape)
        map_right[:-1, :] = map[1:, :]
        map_up = np.zeros(map.shape)
        map_up[:, 1:] = map[:, :-1]
        map_down = np.zeros(map.shape)
        map_down[:, :-1] = map[:, 1:]

        peaks_binary = np.logical_and.reduce(
            (map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > params['thre1']))
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
        peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
        id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)

    connection_all = []
    special_k = []
    mid_num = 10

    # Check if all_peaks contains the expected number of elements
    if len(all_peaks) < 18:
        print("Error: Missing key points in all_peaks")
        return oriImg, 0  # Return the original image and a default position
    
    position = checkPosition(all_peaks)  # check position of spine.
    checkKneeling(all_peaks)  # check whether kneeling or not
    checkHandFold(all_peaks)  # check whether hands are folding or not.
    canvas1 = draw(input_image, all_peaks)  # show the image.
    return canvas1, position





def checkPosition(all_peaks):
    try:
        f = 0
        if all_peaks[16]:
            a = all_peaks[16][0][0:2]  # Right Ear
            f = 1
        else:
            a = all_peaks[17][0][0:2]  # Left Ear
        b = all_peaks[11][0][0:2]  # Hip
        angle = calcAngle(a, b)
        degrees = round(math.degrees(angle))
        if f:
            degrees = 180 - degrees
        if degrees < 70:
            return 1
        elif degrees > 110:
            return -1
        else:
            return 0
    except Exception as e:
        print("Person not in lateral view and unable to detect ears or hip")

def calcAngle(a, b):
    try:
        ax, ay = a
        bx, by = b
        if ax == bx:
                if all_peaks[4][0][0:2]:
                    distance = calcDistance(all_peaks[3][0][0:2], all_peaks[4][0][0:2])  # distance between right arm-joint and right palm.
                    armdist = calcDistance(all_peaks[2][0][0:2], all_peaks[3][0][0:2])  # distance between left arm-joint and left palm.
                    if (distance < (armdist + 100) and distance > (armdist - 100)):  # this value 100 is arbitrary.
                        print("Not Folding Hands")
                    else:
                        print("Folding Hands")
    except Exception as e:
        print("Folding Hands")
    except Exception as e:
        try:
            if all_peaks[7][0][0:2]:
                distance = calcDistance(all_peaks[6][0][0:2], all_peaks[7][0][0:2])
                armdist = calcDistance(all_peaks[6][0][0:2], all_peaks[5][0][0:2])
                if (distance < (armdist + 100) and distance > (armdist - 100)):
                    print("Not Folding Hands")
                else:
                    print("Folding Hands")
        except Exception as e:
            print("Unable to detect arm joints")

def calcDistance(a, b):  # calculate distance between two points.
    try:
        x1, y1 = a
        x2, y2 = b
        return math.hypot(x2 - x1, y2 - y1)
    except Exception as e:
        print("Unable to calculate distance")

def checkKneeling(all_peaks):
    try:
        if all_peaks[7][0][0:2] and all_peaks[8][0][0:2]:
            a = all_peaks[7][0][0:2]  # Left Knee
            b = all_peaks[8][0][0:2]  # Right Knee
            angle = calcAngle(a, b)
            degrees = round(math.degrees(angle))
            if degrees < 90:
                print("Kneeling")
            else:
                print("Not Kneeling")
        else:
            print("legs not detected")
    except Exception as e:
        print("legs not detected")

def showimage(img):
    screen_res = 1280, 720  # My screen resolution.
    scale_width = screen_res[0] / img.shape[1]
    scale_height = screen_res[1] / img.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(img.shape[1] * scale)
    window_height = int(img.shape[0] * scale)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', window_width, window_height)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_image(image_path):
    canvas, position = process(image_path)
    
    # Save processed image to a new directory
    processed_image_path = os.path.join(processed_folder, os.path.basename(image_path))
    cv2.imwrite(processed_image_path, canvas)
    
    # Print the position
    if position == 1:
        print("Hunchback")
    elif position == -1:
        print("Reclined")
    else:
        print("Straight")
    
    # Remove the processed image from the received directory
   # os.remove(image_path)

def start_mqtt_thread(received_folder, process_image_callback):
    print("Starting MQTT receiver in a separate thread...")
    start_mqtt_receiver(received_folder, process_image_callback)


def main():
    print("Initializing model...")
    initialize_model()
    #try:
                     # Start the MQTT receiver with a callback to process images
print("Starting MQTT receiver...")
mqtt_thread = threading.Thread(target=start_mqtt_thread, args=(received_folder, process_image))
mqtt_thread.start()
                        
   # except Exception as e:
                  #  print(f"Error receiving image via MQTT {filename}: {e}")


    

print('Start monitoring...')
try:
        while True:
            # Check if an image is present in the folder
            for filename in os.listdir(received_folder):
                image_path = os.path.join(received_folder, filename)
                
                if os.path.isfile(image_path):
                    print(f"Processing image: {filename}")
                    
                    # Process the image
                    try:
                        process_image(image_path)
                        showimage(image_path)
                    except Exception as e:
                        print(f"Error processing image {filename}: {e}")
                    
                    # Optionally delete the image after processing
                     #os.remove(image_path)
                
            # Sleep before checking for new images again
            time.sleep(20)  # Sleep to keep the script running and avoid busy-waiting

except KeyboardInterrupt:
        print("Interrupted by user")
except Exception as e:
        print(f"Unexpected error occurred: {e}")

if __name__ == '__main__':
    main()
