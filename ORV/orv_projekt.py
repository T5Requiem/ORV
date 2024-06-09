import cv2
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder


def capture_video_and_extract_frames(user_id, duration=5, save_path='datasetraw'):
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
    # Horizontalno zrcaljenje, sprememba svetlosti in kontrasta
    chance = np.round(np.random.uniform(0, 1))
    if chance == 1:
        image = np.fliplr(image)
        image = np.clip(image * 1.2 + 30, 0, 255).astype(np.uint8)
        image = np.clip(image * 1.5, 0, 255).astype(np.uint8)
    else:
        image = np.clip(image * 0.8 - 30, 0, 255).astype(np.uint8)
        image = np.clip(image * 0.5, 0, 255).astype(np.uint8)

    # Rotacija slike za naključni kot med -10 in 10 stopinj
    rows, cols = image.shape[:2]
    angle = np.random.uniform(-10, 10)
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    image = cv2.warpAffine(image, M, (cols, rows))

    # Dodajanje šuma soli in popra
    salt_prob = 0.001
    pepper_prob = 0.001
    num_salt = np.ceil(salt_prob * image.size).astype(int)
    num_pepper = np.ceil(pepper_prob * image.size).astype(int)

    # Dodaj sol
    salt_indices = np.random.choice(image.size, num_salt, replace=False)
    salt_coords = np.unravel_index(salt_indices, image.shape)
    image[salt_coords] = 0

    # Dodaj poper
    pepper_indices = np.random.choice(image.size, num_pepper, replace=False)
    pepper_coords = np.unravel_index(pepper_indices, image.shape)
    image[pepper_coords] = 255

    return image


def augment_dataset(dataset_path='datasetprocessed', augmented_path='datasetaugmented'):
    if not os.path.exists("images/" + augmented_path):
        os.makedirs("images/" + augmented_path)

    image_index = 0
    for filename in os.listdir(dataset_path):
        img_path = os.path.join(dataset_path, filename)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        augmented_image = augment_image(image)
        aug_img_path = os.path.join("images/" + augmented_path, filename)
        cv2.imwrite(aug_img_path, augmented_image)
        print(f"{aug_img_path} saved.")
        image_index += 1

augment_dataset()

# Nalaganje dataseta
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = ImageFolder(root='./images', transform=transform)
test_dataset = ImageFolder(root='./images', transform=transform)

# Razdelitev učne množice na učni in validacijski del
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# Določitev nalagalnikov
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)


# Definicija modela
class TrafficSignNet(nn.Module):
    def __init__(self, num_classes=1, num_repeats=3, base_channels=32):
        super(TrafficSignNet, self).__init__()
        layers = []
        channels = base_channels
        layers.append(nn.Conv2d(3, channels, kernel_size=3, padding=1))
        layers.append(nn.SELU())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        input_channels = 32
        output_channels = 64
        for _ in range(num_repeats - 1):
            layers.append(nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1))
            layers.append(nn.SELU())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            input_channels = input_channels * 2
            output_channels = output_channels * 2

        self.conv_layers = nn.Sequential(*layers)
        self.fc1 = nn.Linear(128 * 8 * 8 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = torch.tanh(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=1)
        return x


# Primer osnovnega modela
model = TrafficSignNet()

# Priprava za učenje
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Trening modela
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                corrects += torch.sum(preds == labels.data)

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = corrects.double() / len(val_loader.dataset) * 100

        print(
            f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}%')


train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)
torch.save(model.state_dict(), './model.pth')


# Ocena natančnosti na testni množici
def evaluate_model(model, test_loader):
    model.eval()
    corrects = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)

    accuracy = corrects.double() / len(test_loader.dataset) * 100
    print(f'Test Accuracy: {accuracy:.4f}%')


evaluate_model(model, test_loader)

model.load_state_dict(torch.load('./model.pth'))