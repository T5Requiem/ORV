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

capture_video_and_extract_frames(user_id=1)


def preprocess_image(image_path):
    image = cv2.imread(image_path)

    # Odstranjevanje šuma
    image = cv2.GaussianBlur(image, (5, 5), 0)

    # Pretvorba v sivinsko lestvico
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Spremeni velikost slike v 128x128 pikslov
    gray_rescaled_image = cv2.resize(gray_image, (128, 128))

    return gray_rescaled_image

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

def augment_image(image):
    # Horizontalno zrcaljenje
    chance = np.random.uniform(0, 1)
    if chance == 1:
        image = np.fliplr(image)

    # Sprememba svetlosti
    image = np.clip(image * 1.2 + 30, 50, 205).astype(np.uint8)

    # Sprememba kontrasta
    image = np.clip(image * 1.5, 50, 205).astype(np.uint8)

    # Rotacija slike za naključni kot med -10 in 10 stopinj
    rows, cols = image.shape[:2]
    angle = np.random.uniform(-10, 10)
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    image = cv2.warpAffine(image, M, (cols, rows))

    # Dodajanje šuma soli in popra
    salt_prob = 0.01
    pepper_prob = 0.01
    num_salt = np.ceil(salt_prob * image.size)
    num_pepper = np.ceil(pepper_prob * image.size)

    # Dodaj sol
    salt_coords = [np.random.randint(0, image.shape[dim], int(num_salt)) for dim in range(image.ndim)]
    image[salt_coords] = 0

    # Dodaj poper
    pepper_coords = [np.random.randint(0, image.shape[dim], int(num_pepper)) for dim in range(image.ndim)]
    image[pepper_coords] = 10

    return image


def augment_dataset(dataset_path='datasetprocessed', augmented_path='datasetaugmented'):
    if not os.path.exists(augmented_path):
        os.makedirs(augmented_path)

    for filename in os.listdir(dataset_path):
        img_path = os.path.join(dataset_path, filename)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        augmented_images = augment_image(image)
        for i, aug_img in enumerate(augmented_images):
            aug_img_path = os.path.join(augmented_path, f"{filename.split('.')[0]}_aug_{i}.jpg")
            cv2.imwrite(aug_img_path, aug_img)
            print(f"{aug_img_path} saved.")

augment_dataset()