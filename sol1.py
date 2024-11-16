import cv2
import numpy as np
from pathlib import Path
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torch.nn.functional import normalize

# Load a pre-trained ResNet model for feature extraction
def load_model():
    model = resnet50(pretrained=True)
    model = model.eval()  # Set the model to evaluation mode
    return model

# Extract features using a deep learning model
def extract_features(image, model, device):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(image)
    return normalize(features, p=2, dim=1).cpu().numpy()

# Function to find the best match using feature matching
def find_location_with_features(query_image, ortho_image, model, device):
    # Convert images to color if they are grayscale
    if len(query_image.shape) < 3:
        query_image = cv2.cvtColor(query_image, cv2.COLOR_GRAY2BGR)
    if len(ortho_image.shape) < 3:
        ortho_image = cv2.cvtColor(ortho_image, cv2.COLOR_GRAY2BGR)

    query_features = extract_features(query_image, model, device)
    ortho_features = extract_features(ortho_image, model, device)

    # Use a simple method like Euclidean distance to find the best matching location
    # This is just a placeholder, consider more sophisticated methods for real applications
    center = (0, 0)  # Placeholder for the actual center calculation
    best_score = -1  # Placeholder for the actual score
    
    # Implement feature matching logic here (e.g., using FLANN or BFMatcher)
    # Update center and best_score with the actual calculations

    return center, best_score

# Main function to process the dataset and find matches
def main():
    DATA_PATH = Path("/kaggle/input/gnss-denied-localization/data")
    TEST_IMAGE_PATH = DATA_PATH / "test_data" / "test_images"
    
    ortho_image = cv2.imread(str(DATA_PATH / "map.png"))
    image_paths = sorted(TEST_IMAGE_PATH.iterdir())

    # Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model().to(device)

    print(f"{'id':>5s},{'x_pixel':>9s},{'y_pixel':>9s}")
    for img_path in image_paths:
        img_id = int(img_path.stem)
        query_image = cv2.imread(str(img_path))

        center, _ = find_location_with_features(query_image, ortho_image, model, device)

        # Print to terminal or write to .csv directly
        print(f"{img_id:5d},{center[0]:9d},{center[1]:9d}")

# Run the main function
if __name__ == "__main__":
    main()