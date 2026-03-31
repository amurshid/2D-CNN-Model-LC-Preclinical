import torch
from data_loader import create_data_loaders
from example_training import SimpleLungCancerCNN

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create loaders (adjust batch_size if you want)
    train_loader, val_loader, test_loader, class_names = create_data_loaders(
        data_dir="processed_data", batch_size=8, num_workers=0, augment=False
    )

    # Get one batch
    images, labels = next(iter(train_loader))
    print("batch images shape:", images.shape)  # (B, C, H, W)

    # Use one sample (batch size = 1)
    x = images[0:1].to(device)

    # Instantiate model (ensure in_channels matches images.shape[1])
    model = SimpleLungCancerCNN(num_classes=len(class_names))
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        feats = model.features(x)                       # output of conv blocks
        print("features shape (after convs):", feats.shape)
        pooled = model.classifier[0](feats)             # AdaptiveAvgPool2d((7,7))
        flat = model.classifier[1](pooled)              # Flatten()
        print("flattened shape (after pool+flatten):", flat.shape)
        print("flattened vector length:", flat.shape[1])

if __name__ == "__main__":
    main()