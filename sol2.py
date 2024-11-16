import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # Use pretrained ResNet as backbone
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        
    def forward(self, x):
        return self.features(x)

class LocationPredictor:
    def __init__(self, map_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_extractor = FeatureExtractor().to(self.device)
        self.feature_extractor.eval()
        
        # Load and preprocess map
        self.map_image = cv2.imread(str(map_path), cv2.IMREAD_COLOR)
        self.map_features = self._extract_map_features()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

    def _extract_map_features(self):
        """Extract features from the map using sliding window"""
        window_size = 224
        stride = 112
        features_map = {}
        
        for y in range(0, self.map_image.shape[0] - window_size, stride):
            for x in range(0, self.map_image.shape[1] - window_size, stride):
                window = self.map_image[y:y+window_size, x:x+window_size]
                window_pil = Image.fromarray(cv2.cvtColor(window, cv2.COLOR_BGR2RGB))
                window_tensor = self.transform(window_pil).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    features = self.feature_extractor(window_tensor)
                features_map[(x, y)] = features
                
        return features_map

    def _compute_similarity(self, query_features, map_features):
        """Compute similarity between query and map features"""
        similarities = {}
        query_features = query_features.squeeze()
        
        for (x, y), map_feat in self.map_features.items():
            map_feat = map_feat.squeeze()
            similarity = torch.nn.functional.cosine_similarity(
                query_features.flatten(),
                map_feat.flatten(),
                dim=0
            )
            similarities[(x, y)] = similarity.item()
            
        return similarities

    def find_location(self, query_image):
        """Find location of query image in map"""
        # Preprocess query image
        query_pil = Image.fromarray(cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB))
        query_tensor = self.transform(query_pil).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            query_features = self.feature_extractor(query_tensor)
        
        # Compute similarities
        similarities = self._compute_similarity(query_features, self.map_features)
        
        # Find best match
        best_loc = max(similarities.items(), key=lambda x: x[1])
        x, y = best_loc[0]
        
        # Refine position using local search
        refined_x, refined_y = self._refine_position(query_image, x, y)
        
        return refined_x, refined_y

    def _refine_position(self, query_image, x, y, search_radius=20):
        """Refine position using local template matching"""
        window_size = 224
        search_area = self.map_image[
            max(0, y-search_radius):min(self.map_image.shape[0], y+window_size+search_radius),
            max(0, x-search_radius):min(self.map_image.shape[1], x+window_size+search_radius)
        ]
        
        query_resized = cv2.resize(query_image, (window_size, window_size))
        result = cv2.matchTemplate(search_area, query_resized, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(result)
        
        refined_x = x - search_radius + max_loc[0]
        refined_y = y - search_radius + max_loc[1]
        
        return refined_x + window_size//2, refined_y + window_size//2

def main():
    DATA_PATH = Path("/kaggle/input/gnss-denied-localization/data")
    TEST_IMAGE_PATH = DATA_PATH / "test_data" / "test_images"
    
    # Initialize predictor
    predictor = LocationPredictor(DATA_PATH / "map.png")
    
    # Process test images
    print(f"{'id':>5s},{'x_pixel':>9s},{'y_pixel':>9s}")
    for img_path in sorted(TEST_IMAGE_PATH.iterdir()):
        img_id = int(img_path.stem)
        query_image = cv2.imread(str(img_path))
        
        # Predict location
        x, y = predictor.find_location(query_image)
        
        print(f"{img_id:5d},{int(x):9d},{int(y):9d}")

if __name__ == "__main__":
    main()
