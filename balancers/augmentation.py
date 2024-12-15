import os
import cv2
import numpy as np
import shutil

class Augmentation:
    def __init__(self, random_bound=0.5):
        self.random_bound = random_bound
    
    def augment_image(self, image, rotate=True):
        if np.random.rand() > self.random_bound:
            image = cv2.flip(image, 1)
        if rotate:
            angle = np.random.choice([90, 180, 270])
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
            image = cv2.warpAffine(image, rotation_matrix, (width, height))
        factor = np.random.uniform(1.0, 1.05)
        image = cv2.convertScaleAbs(image, alpha=factor, beta=0)
        return image
    
    def augment_image_file(self, path_to_input_image, path_to_output_image, rotate=True):
        image = cv2.imread(path_to_input_image)
        image = self.augment_image(image)
        cv2.imwrite(path_to_output_image, image)
        
class AugmentationBalancer:
    def __init__(self, random_bound=0.5):
        self.augmentation = Augmentation(random_bound=random_bound)
    
    def fit(self, path_to_input_image_folder, delta=5, debug=False):
        self.path_to_folder=path_to_input_image_folder        
        directories = os.listdir(path_to_input_image_folder)
        real_directories = {}
        self.to_generate = {}
        for directory in directories:
            if os.path.isdir(f"{path_to_input_image_folder}/{directory}"):
                count = len(os.listdir(f"{path_to_input_image_folder}/{directory}"))
                real_directories[directory]=count
        max_id = max(real_directories, key=real_directories.get)
        print(real_directories)
        self.total_classes = [key for key in real_directories]
        for directory in real_directories:
            diff = real_directories[max_id] - real_directories[directory]
            if(diff>delta):
                self.to_generate[directory]=diff
        if debug:
            print(self.to_generate)
        
    def balance(self, path_to_output_image_folder, debug=False):
        if os.path.exists(path_to_output_image_folder):
            shutil.rmtree(path_to_output_image_folder)
        os.makedirs(path_to_output_image_folder)

        for category in self.total_classes:
            source_folder = f"{self.path_to_folder}/{category}"
            destination_folder = f"{path_to_output_image_folder}/{category}"
            os.makedirs(destination_folder)
            for file in os.listdir(source_folder):
                source_file = os.path.join(source_folder, file)
                destination_file = os.path.join(destination_folder, file)
                if os.path.isfile(source_file):
                    shutil.copy2(source_file, destination_file)
        if debug:
            print("Classes to balance:", self.to_generate)
        for category in self.to_generate:
            category_folder = f"{path_to_output_image_folder}/{category}"
            last_id = -1
            for file in os.listdir(category_folder):
                if file.endswith(".jpg"):
                    try:
                        id = int(file.split("_")[1].split(".")[0])
                        if id > last_id:
                            last_id = id
                    except (IndexError, ValueError):
                        continue
            if last_id == -1:
                last_id = 0
            for i in range(self.to_generate[category]):
                selected_id = np.random.randint(1, last_id + 1) 
                selected_id_str = f"{selected_id:02d}"
                last_id += 1
                new_id_str = f"{last_id:02d}"
                path_to_selected_image = f"{category_folder}/{category}_{selected_id_str}.jpg"
                path_to_generate_image = f"{category_folder}/{category}_{new_id_str}.jpg"
                if not os.path.exists(path_to_selected_image):
                    if debug:
                        print(f"File does not exist: {path_to_selected_image}, skipping.")
                    continue
                if debug:
                    print(f"Augmenting: {path_to_selected_image} -> {path_to_generate_image}")
                self.augmentation.augment_image_file(
                    path_to_selected_image,
                    path_to_generate_image
                )