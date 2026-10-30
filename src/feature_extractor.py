"""
Deep Feature Extractor Module using EfficientNetB0 Transfer Learning.

Fine-tunes the last 20 layers of EfficientNetB0 pretrained on ImageNet
on the 4-class MRI dataset, extracts 1,280-D Global Average Pooling (GAP)
feature vectors per image, and exports NumPy feature arrays (X_train, y_train, etc.)
to models/.
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    MODELS_DIR,
    NUM_CLASSES,
    FEATURE_DIM,
    RANDOM_SEED,
)
from src.data_loader import get_dataloaders

# Fast CPU / GPU Execution Parameters
BATCH_SIZE = 64
NUM_EPOCHS = 3
LEARNING_RATE = 1e-4


def get_efficientnet_model(num_classes: int = NUM_CLASSES):
    """
    Instantiates EfficientNetB0 with ImageNet weights, freezes early layers,
    and unfreezes the final 20 layers for fine-tuning on MRI scans.
    """
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)

    # Freeze all layers initially
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last 20 layers for task-specific fine-tuning
    all_params = list(model.parameters())
    for param in all_params[-20:]:
        param.requires_grad = True

    # Replace classifier head for 4-class MRI task
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


class PyTorchFeatureExtractor:
    """
    Inference feature extractor class for single images.
    Loads fine-tuned EfficientNetB0 backbone and extracts 1280-D GAP features.
    """
    def __init__(self, weights_path=None, device="cpu"):
        self.device = torch.device(device)
        self.model = get_efficientnet_model()
        if weights_path is not None and Path(weights_path).exists():
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1], nn.AdaptiveAvgPool2d((1, 1))).to(self.device)

        from torchvision import transforms
        import cv2
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_features_single(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Extracts 1280-D feature vector from a single preprocessed BGR image (224x224x3).
        """
        import cv2
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if len(img_bgr.shape) == 3 else img_bgr
        tensor_img = self.transform(img_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feats = self.feature_extractor(tensor_img)
            feats = torch.flatten(feats, 1)
        return feats.cpu().numpy().squeeze(0)


def fine_tune_backbone(model, train_loader, val_loader, device, epochs: int = NUM_EPOCHS):
    """
    Fine-tunes the EfficientNetB0 backbone on the 4-class MRI training split.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    print(f"\nFine-tuning EfficientNetB0 backbone for {epochs} epochs on device: {device}...")

    best_val_acc = 0.0
    best_weights_path = MODELS_DIR / "efficientnet_b0_finetuned.pth"

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        train_loss = running_loss / total

        # Validation loop
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0

        with torch.no_grad():
            for images, labels, _ in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        val_loss = val_loss / val_total

        print(f"Epoch {epoch+1}/{epochs} -> Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_weights_path)
            print(f" Saved new best backbone model to {best_weights_path}")

    # Load best weights
    if best_weights_path.exists():
        model.load_state_dict(torch.load(best_weights_path))
    return model


def extract_features(model, loader, device):
    """
    Extracts 1,280-D Global Average Pooling feature vectors from the fine-tuned EfficientNetB0.
    """
    model.eval()
    # Remove final classification layer to get GAP features (1280-D)
    feature_extractor = nn.Sequential(*list(model.children())[:-1], nn.AdaptiveAvgPool2d((1, 1)))
    feature_extractor = feature_extractor.to(device)

    features_list = []
    labels_list = []

    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc="Extracting GAP Feature Vectors"):
            images = images.to(device)
            # EfficientNet features output shape: (B, 1280, 1, 1) -> flatten to (B, 1280)
            feats = feature_extractor(images)
            feats = torch.flatten(feats, 1)

            features_list.append(feats.cpu().numpy())
            labels_list.append(labels.numpy())

    X = np.vstack(features_list)
    y = np.concatenate(labels_list)
    return X, y


def main():
    print("Starting Stage 5: Deep Feature Extraction (EfficientNetB0 Transfer Learning)...")

    # Set CPU multi-threading
    num_cpus = os.cpu_count() or 4
    torch.set_num_threads(num_cpus)

    # Set reproducibility seeds
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device} (threads: {num_cpus})")

    # Get data loaders
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)

    # Instantiate model
    model = get_efficientnet_model()

    # Fine-tune backbone on MRI scans
    model = fine_tune_backbone(model, train_loader, val_loader, device=device, epochs=NUM_EPOCHS)

    # Extract 1280-D feature vectors for Train, Val, Test splits
    print("\nExtracting Global Average Pooling (GAP) 1280-D Features...")
    X_train, y_train = extract_features(model, train_loader, device)
    X_val, y_val = extract_features(model, val_loader, device)
    X_test, y_test = extract_features(model, test_loader, device)

    # Save feature arrays to models/
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(MODELS_DIR / "X_train.npy", X_train)
    np.save(MODELS_DIR / "y_train.npy", y_train)
    np.save(MODELS_DIR / "X_val.npy", X_val)
    np.save(MODELS_DIR / "y_val.npy", y_val)
    np.save(MODELS_DIR / "X_test.npy", X_test)
    np.save(MODELS_DIR / "y_test.npy", y_test)

    print("\n" + "=" * 60)
    print("           DEEP FEATURE EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"X_train Shape: {X_train.shape}, y_train Shape: {y_train.shape}")
    print(f"X_val Shape:   {X_val.shape}, y_val Shape:   {y_val.shape}")
    print(f"X_test Shape:  {X_test.shape}, y_test Shape:  {y_test.shape}")
    print(f"Saved feature arrays to: {MODELS_DIR}")
    print("=" * 60 + "\n")
    print("Stage 5 completed successfully!")


if __name__ == "__main__":
    main()
