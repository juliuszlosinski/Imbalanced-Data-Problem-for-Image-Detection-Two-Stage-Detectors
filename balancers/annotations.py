import os
import shutil

class Annotations:
    def __init__(self, x_center="0.5", y_center="0.5", width="1.0", height="1.0"):
        """
        Bounding box format for YOLO: <class_id> <x-center> <y-center> <width> <height>
        """
        self.bounding_box = f"{x_center} {y_center} {width} {height}"

    def annotate(self, path_to_input_images, path_to_output_annotations):
        id = 0
        for category in os.listdir(path_to_input_images):
            category_path = os.path.join(path_to_input_images, category)
            if not os.path.isdir(category_path):  
                continue
            number_of_images = len(os.listdir(category_path))
            path_to_annotation_directory = os.path.join(path_to_output_annotations, category)
            if os.path.exists(path_to_annotation_directory):
                shutil.rmtree(path_to_annotation_directory)
            os.makedirs(path_to_annotation_directory, exist_ok=True)
            for i in range(number_of_images):
                result = f"{id} {self.bounding_box}"
                if i < 9:
                    file_name = f"{category}_0{i+1}.txt"
                else:
                    file_name = f"{category}_{i+1}.txt"
                
                path_to_annotation_file = os.path.join(path_to_annotation_directory, file_name)
                with open(path_to_annotation_file, "w") as annotation_file:
                    annotation_file.write(result)
            id += 1