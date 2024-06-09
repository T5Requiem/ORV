import cv2
import os
import numpy as np
from pyfcm import FCMNotification


def capture_video_and_extract_frames(user_id, duration=3, save_path='datasetraw'):
    # Ustvari mapo, če ne obstaja
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Inicializacija kamere
    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)  # število slik na sekundo
    total_frames = int(duration * fps)  # Skupno število slik

    frame_count = 0
    while frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            continue

        # Shrani vsako frame kot sliko
        img_name = f"{save_path}/user_{user_id}_{frame_count}.jpg"
        cv2.imwrite(img_name, frame)
        print(f"{img_name} saved.")

        frame_count += 1

        cv2.imshow("Capture", frame)

        # Prekini zajemanje s 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# capture_video_and_extract_frames(user_id=1)


def preprocess_image(image_path):
    image = cv2.imread(image_path)

    # Odstranjevanje šuma
    image = cv2.GaussianBlur(image, (5, 5), 0)

    # Pretvorba v sivinsko lestvico
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return grayImage

def preprocess_dataset(dataset_path='datasetraw', processed_path='datasetprocessed'):
    if not os.path.exists(processed_path):
        os.makedirs(processed_path)

    for filename in os.listdir(dataset_path):
        img_path = os.path.join(dataset_path, filename)
        processed_img = preprocess_image(img_path)

        # Shrani obdelano sliko
        processed_img_path = os.path.join(processed_path, filename)
        cv2.imwrite(processed_img_path, processed_img)
        print(f"{processed_img_path} saved.")

preprocess_dataset()