import os
import shutil
import numpy as np
from PIL import Image
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

class ImageSMOTE:
    def __init__(self) -> None:
        self.label_encoder = LabelEncoder()
    
    """
    Fitting/loading the images and preprocessing to flatten vectors for SMOTE.
    """
    def fit(self, path_to_input_image_folder: str, width_of_image: int, height_of_image: int, debug: bool = False):
        self.width_of_image = width_of_image
        self.height_of_image = height_of_image
        self.max_map = ()
        self.unbalanced_maps = {}
        max_number_of_images = -1
        for directory_name in os.listdir(path_to_input_image_folder):
            path = f"{path_to_input_image_folder}/{directory_name}"
            if os.path.isdir(path):
                images = []
                labels = []
                for file_name in os.listdir(path):
                    try:
                        image = Image.open(f"{path}/{file_name}")
                        image = image.resize((self.width_of_image, self.height_of_image))
                        if image.mode != "RGB":
                            image = image.convert("RGB")  # Ensure all images are in RGB mode
                        image_array = np.array(image).flatten()
                        images.append(image_array)
                        labels.append(directory_name)  # Use directory name as label
                    except Exception as e:
                        print(f"Error processing {file_name}: {e}")
                number_of_images = len(images)
                if debug:
                    print(f"Class: {directory_name}, Number of images: {number_of_images}")
                if number_of_images > max_number_of_images:
                    max_number_of_images = number_of_images
                    self.max_map = (directory_name, len(images), images, labels)
                self.unbalanced_maps[directory_name] = (directory_name, len(images), images, labels)
        if debug:
            print(f"Largest class: {self.max_map[0]}, Number of images: {self.max_map[1]}")

    """
    Balancing the dataset.
    """
    def balance(self, path_to_output_image_folder: str, debug: bool = False):
        if os.path.exists(path_to_output_image_folder):
            shutil.rmtree(path_to_output_image_folder)
        os.makedirs(path_to_output_image_folder)
        
        max_images = np.array(self.max_map[2])
        max_labels = self.max_map[3]
        
        for key in self.unbalanced_maps:
            if key != self.max_map[0]:
                current_images = np.array(self.unbalanced_maps[key][2])
                current_labels = self.unbalanced_maps[key][3]
                images = np.vstack([max_images, current_images])
                labels = max_labels + current_labels  
                
                if debug:
                    print(f"Balancing class: {key}")
                    print(f"Images shape: {images.shape}, len of labels: {len(labels)}")
                
                labels_encoded = self.label_encoder.fit_transform(labels)
                smote = SMOTE(sampling_strategy="auto", random_state=42)
                images_resampled, labels_resampled = smote.fit_resample(images, labels_encoded)
                
                if debug:
                    print(f"Resampled images shape: {images_resampled.shape}")
                    print(f"Resampled labels length: {len(labels_resampled)}")
                
                class_counter = {}  
                for i, img_array in enumerate(images_resampled):
                    img = Image.fromarray(img_array.reshape(self.width_of_image, self.height_of_image, 3).astype(np.uint8))
                    label = self.label_encoder.inverse_transform([labels_resampled[i]])[0]
                    class_output_path = f"{path_to_output_image_folder}/{label}"
                    if not os.path.exists(class_output_path):
                        os.makedirs(class_output_path)
                    if label not in class_counter:
                        class_counter[label] = 1 
                    else:
                        class_counter[label] += 1
                    image_index = str(class_counter[label]).zfill(2)
                    img.save(f"{class_output_path}/{label}_{image_index}.jpg")