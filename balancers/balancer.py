from .configuration_reader import ConfigurationReader
from .augmentation import AugmentationBalancer
from .adasyn import ImageADASYN
from .smote import ImageSMOTE
from .autoencoder import AEBalancer
from .dgan import DGANBalancer
from .annotations import Annotations

class Balancer:
    def __init__(self, path_to_configuration):
        self.path_to_configuration = path_to_configuration
        self.configuration_reader = ConfigurationReader()
        self.configuration_reader.read(path_to_configuration)
        self.annotator = Annotations()

    def annotate(self, path_to_input_image_folder, path_to_output_annotations):
        self.annotator.annotate(
                path_to_input_images=path_to_input_image_folder,
                path_to_output_annotations=path_to_output_annotations
        )
    
    def fit(self, mode, path_to_input_image_folder, debug=False):
        print(f"{10*'='}{mode}{10*'='}")
        self.mode = mode
        match mode:
            case "AUGMENTATION":
                self.aug_balancer = AugmentationBalancer()
                self.aug_balancer.fit(
                    path_to_input_image_folder=path_to_input_image_folder, 
                    debug=debug
                )
            case "ADASYN":
                self.adasyn_balancer = ImageADASYN()
                self.adasyn_balancer.fit(
                    path_to_input_image_folder=path_to_input_image_folder,
                    width_of_image=self.configuration_reader.width_of_image,
                    height_of_image=self.configuration_reader.height_of_image
                )
            case "SMOTE":
                self.smote_balancer = ImageSMOTE()
                self.smote_balancer.fit(
                    path_to_input_image_folder=path_to_input_image_folder, 
                    width_of_image=self.configuration_reader.width_of_image, 
                    height_of_image=self.configuration_reader.height_of_image
                )
            case "DGAN":
                self.dgan_balancer = DGANBalancer()
                self.dgan_balancer.fit(
                    path_to_input_image_folder=path_to_input_image_folder,
                    latent_dimension=self.configuration_reader.latent_dimension, 
                    learning_rate=self.configuration_reader.learning_rate, 
                    beta_01=self.configuration_reader.beta,
                    batch_size=self.configuration_reader.batch_size, 
                    number_of_epochs=self.configuration_reader.number_of_epochs, 
                    delta=self.configuration_reader.delta
                )
            case "AE":
                self.ae_balancer = AEBalancer()
                self.ae_balancer.fit(
                    path_to_input_image_folder=path_to_input_image_folder, 
                    batch_size=self.configuration_reader.batch_size, 
                    number_of_epochs=self.configuration_reader.number_of_epochs, 
                    delta=self.configuration_reader.delta
                )
        print(f"{(len(mode)+20)*'='}")
        
    def balance(self, path_to_output_image_folder, debug=False):
        print(f"{10*'='}{self.mode}{10*'='}")
        match self.mode:
            case "AUGMENTATION":
                self.aug_balancer.balance(
                    path_to_output_image_folder=path_to_output_image_folder, 
                    debug=debug
                )
            case "ADASYN":
                self.adasyn_balancer.balance(
                    path_to_output_image_folder=path_to_output_image_folder,
                    number_of_neighbors = self.configuration_reader.number_of_neighbors
                )
            case "SMOTE":
                self.smote_balancer.balance(
                    path_to_output_image_folder=path_to_output_image_folder
                )
            case "DGAN":
                self.dgan_balancer.balance(
                    path_to_output_image_folder=path_to_output_image_folder
                )
            case "AE":
                self.ae_balancer.balance(
                    path_to_output_image_folder=path_to_output_image_folder,
                    debug=debug
                )
        print(f"{(len(self.mode)+20)*'='}")
        
    def print_configuration(self):
        self.configuration_reader.print()