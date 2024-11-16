You are expert in Kaggle solving challenges, read this description and naive solution and provide high level plan of implementing this challenge:
```
Typically, autonomous drones use Global Navigation Satellite Systems (GNSS) for localization, e.g., GPS. However, its accuracy can be limited by factors like signal loss in urban environments and can be compromised by signal jamming.

Visual localization, on the other hand, offers a promising alternative to GNSS and leverages visual cues such as landmarks, terrain features, and textures. It is less susceptible to GPS signal disruptions and can operate effectively in environments where GPS may fail.

In this challenge, you are going to implement a visual localization system.

Goal: Given a drone image and a aerial image of an area, the aim is to predict/estimate the drone's position in the aerial image.

Example Notebook: We upload a very naive solution here. It should mostly serve to explain how to work with the data -- we do not think that classic template matching will be a good solution.
```

and the naive solution:
```
# This is a very naive implementation and should just serve to understand the data
# We believe that a good solution will likely use deep-learning-based techniques

import cv2
from pathlib import Path

def find_location(query_image, ortho_image, method):
    best_score = None
    center = None

    # template matching for different scales
    scales = [0.05, 0.1, 0.2]

    for scale in scales:
        scaled_query_image = cv2.resize(
            query_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
        # template matching for different rotations
        for _ in range(4):
            res = cv2.matchTemplate(scaled_query_image, ortho_image, method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                top_left = min_loc
                current_score = -min_val
            else:
                top_left = max_loc
                current_score = max_val

            if not best_score or current_score > best_score:
                best_score = current_score
                center = (top_left[0] + scaled_query_image.shape[1]//2, top_left[1] + scaled_query_image.shape[0]//2)
                
            scaled_query_image = cv2.rotate(scaled_query_image, cv2.ROTATE_90_CLOCKWISE)

    return center, best_score

DATA_PATH = Path("/kaggle/input/gnss-denied-localization/data")
TEST_IMAGE_PATH = DATA_PATH / "test_data" / "test_images"

ortho_image = cv2.imread(DATA_PATH / "map.png", cv2.IMREAD_GRAYSCALE)
image_paths = sorted(TEST_IMAGE_PATH.iterdir())

# This loop is very slow due to the naive template matching
print(f"{'id':>5s},{'x_pixel':>9s},{'y_pixel':>9s}")
for img_path in image_paths:
    img_id = int(img_path.stem)
    query_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    center, _ = find_location(query_image, ortho_image, cv2.TM_CCOEFF_NORMED)

    # print to terminal or write to .csv directly
    print(f"{img_id:5d},{center[0]:9d},{center[1]:9d}")
```