import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import numpy as np
import copy

###CREATE PYTORCH DATASET CLASS###

class FER2013Dataset(Dataset):
    def __init__(self, csv_file, usage = "Training", transform = None):
        self.data = pd.read_csv(csv_file)
        self.data = self.data[self.data['Usage'] == usage]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        emotion = int(self.data.iloc[idx]['emotion'])
        pixels = self.data.iloc[idx]['pixels']
        img = np.array(pixels.split(), dtype="uint8").reshape(48,48)

        #Convert to torch tensor and apply transform if needed
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0

        return img, emotion

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((48,48)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(5),
    transforms.RandomCrop((48,48), padding=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((48,48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = FER2013Dataset(csv_file="fer2013.csv", usage="Training", transform=train_transform)
test_data  = FER2013Dataset(csv_file="fer2013.csv", usage="PublicTest", transform=test_transform)


###LOAD DATA###
train_loader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size = 64, shuffle = False)

###DEFINE A SIMPLE CNN###
class EmotionCNN(nn.Module):
    def __init__(self, num_classes = 7):
        super(EmotionCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128*6*6, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EmotionCNN().to(device)

###DEFINE LOSS AND OPTIMISER###
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

###CREATE TRAINING LOOP###
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

###EVALUATE MODEL PERFORMANCE ACROSS SIMULATED QUANTIZATION SETTINGS###

# Evaluate any version of the trained model
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in loader:
            outputs = model(images.to("cpu"))
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total


# Simulate symmetric weight quantization
def quantize_weights(model, bits, per_channel=False):
    quant_model = copy.deepcopy(model).cpu().eval()

    qmax = 2 ** (bits - 1) - 1

    with torch.no_grad():
        for layer in quant_model.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                weights = layer.weight

                if per_channel:
                    # Separate scale for each output channel
                    dims = tuple(range(1, weights.ndim))
                    max_abs = weights.abs().amax(
                        dim=dims, keepdim=True
                    )
                else:
                    # One scale for the entire weight tensor
                    max_abs = weights.abs().max()

                scale = (max_abs / qmax).clamp(min=1e-12)

                quantized = torch.round(weights / scale)
                quantized = torch.clamp(
                    quantized, -qmax, qmax
                )

                # Dequantize back to floating point
                weights.copy_(quantized * scale)

    return quant_model


# Run experiments
results = []

baseline = evaluate(model.cpu(), test_loader)

results.append({
    "Bits": 32,
    "Granularity": "FP32 baseline",
    "Accuracy": baseline
})

for bits in [8, 6, 4, 2]:
    for granularity in ["per_tensor", "per_channel"]:

        quant_model = quantize_weights(
            model,
            bits=bits,
            per_channel=(granularity == "per_channel")
        )

        accuracy = evaluate(quant_model, test_loader)

        results.append({
            "Bits": bits,
            "Granularity": granularity,
            "Accuracy": accuracy
        })

df = pd.DataFrame(results)
df["Accuracy drop (pp)"] = baseline - df["Accuracy"]

print(df.to_string(index=False))
