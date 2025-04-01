from flask import Flask, request, jsonify
from PIL import Image
import torch
import torchvision.transforms as T
import io
import torch.nn as nn
import torch.nn.functional as F
import os

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model_path = "D:\\Temp\\SixthModel.pth"  # Your model path
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define label categories
colors = ['white', 'blue', 'black', 'red', 'green', 'brown']
types = ["t-shirt", "shirt", "polo", "formal shirt", "casual shirt", "dress", "pants", "shoes", "shorts"]
patterns = ["plain", "checkered", "striped", "floral", "dotted", "printed"]
classes = colors + types + patterns

# Define image transformations
imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
transform = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor(),
    T.Normalize(*imagenet_stats)
])

# Load the model
class MultilabelImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, targets = batch
        out = self(images)
        loss = F.binary_cross_entropy(out, targets)
        return loss

    def validation_step(self, batch):
        images, targets = batch
        out = self(images)
        loss = F.binary_cross_entropy(out, targets)
        return {'val_loss': loss.detach()}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        return {'val_loss': epoch_loss.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}".format(
            epoch, result['val_loss']))

class ResNet15(MultilabelImageClassificationBase):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
            
        )
        self.res1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
            
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(4),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
            
        )
        self.res2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 512, kernel_size=3, padding=1),
            nn.MaxPool2d(4),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.res3 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.MaxPool2d(4),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.res4 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(1024 * 1 * 1, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.res1(out) + out
        out = self.conv2(out)
        out = self.res2(out) + out
        out = self.conv3(out)
        out = self.res3(out) + out
        out = self.conv4(out)
        out = self.res4(out) + out
        out = self.classifier(out)
        out = torch.sigmoid(out)  # Use sigmoid for multi-label classification
        return out

# Load the model
model = ResNet15(3, len(classes)).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

@app.route('/predict_with_actual', methods=['POST'])
def predict_with_actual():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']

    custom_attributes = request.form.get('custom_attributes', '').lower().split(',')
    custom_attributes = [label.strip() for label in custom_attributes if label.strip()]

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)[0].cpu()

        label_scores = {}
        custom_attributes_threshold_scores = {label: 0.0 for label in custom_attributes}
        is_fashion = False

        for i, score in enumerate(output):
            threshold_score = score.item()

            if classes[i] in custom_attributes:
                custom_attributes_threshold_scores[classes[i]] = threshold_score

            if threshold_score >= 0.1:
                is_fashion = True
                label_scores[classes[i]] = threshold_score

        detected_labels = list(label_scores.keys())
        fashion_detected = any(label in classes for label in detected_labels)

        is_non_fashion = not fashion_detected

        prediction = {"labels": label_scores}

        CUSTOM_ATTRIBUTES = {"prediction": prediction}

        if custom_attributes:
            CUSTOM_ATTRIBUTES["custom_attributes_threshold_scores"] = custom_attributes_threshold_scores

        if is_non_fashion:
            CUSTOM_ATTRIBUTES["result"] = "non_fashion"

        return jsonify([CUSTOM_ATTRIBUTES])

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)














































"""from flask import Flask, request, jsonify
from PIL import Image
import torch
import torchvision.transforms as T
import io
import torch.nn as nn
import torch.nn.functional as F

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model_path = "D:\\Temp\\SixthModel.pth"  # Path to your saved model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define classes
colors = ['white', 'blue', 'black', 'red', 'green', 'brown']
types = ["t-shirt", "shirt", "polo", "formal shirt", "casual shirt", "dress", "pants", "shoes", "shorts"]
patterns = ["plain", "checkered", "striped", "floral", "dotted", "printed"]
classes = colors + types + patterns


# Define the model architecture (same as during training)
class MultilabelImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, targets = batch
        out = self(images)
        loss = F.binary_cross_entropy(out, targets)
        return loss

    def validation_step(self, batch):
        images, targets = batch
        out = self(images)
        loss = F.binary_cross_entropy(out, targets)
        return {'val_loss': loss.detach()}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        return {'val_loss': epoch_loss.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}".format(
            epoch, result['val_loss']))


class ResNet15(MultilabelImageClassificationBase):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # Input 3 x 128 x 128
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.res1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(4),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.res2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 512, kernel_size=3, padding=1),
            nn.MaxPool2d(4),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.res3 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.MaxPool2d(4),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.res4 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(1024 * 1 * 1, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.res1(out) + out
        out = self.conv2(out)
        out = self.res2(out) + out
        out = self.conv3(out)
        out = self.res3(out) + out
        out = self.conv4(out)
        out = self.res4(out) + out
        out = self.classifier(out)
        out = torch.sigmoid(out)  # Use sigmoid for multi-label classification
        return out


# Load the model
model = ResNet15(3, len(classes)).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Define image transformations
imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
transform = T.Compose([
    T.RandomResizedCrop((128, 128), scale=(0.8, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(30),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.RandomErasing(p=0.5, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# API endpoint for label-specific threshold with non-fashion detection
@app.route('/predict_label', methods=['POST'])
def predict_label():
    if 'file' not in request.files or 'label' not in request.form:
        return jsonify({"error": "File and label required"}), 400

    file = request.files['file']
    label = request.form['label']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if label not in classes:
        return jsonify({"error": f"Label '{label}' not found in classes"}), 400

    try:
        # Read and preprocess the image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            output = model(image)[0].cpu()

        # Get threshold score for the specified label
        label_index = classes.index(label)
        threshold_score = float(output[label_index])

        # Check for non-fashion image
        if threshold_score < 0.1:
            return jsonify({
                "label": label,
                "threshold_score": threshold_score,
                "message": "Non-fashion image detected"
            })

        # Return threshold score
        return jsonify({
            "label": label,
            "threshold_score": threshold_score,
            "message": "Fashion image detected"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)"""