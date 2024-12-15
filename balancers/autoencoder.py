import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
from PIL import Image
import shutil
import os

# Define the Encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv2d_01 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1
        )  # (64, 64, 64)
        self.relu_01 = nn.ReLU()

        self.conv2d_02 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1
        )  # (32, 32, 128)
        self.relu_02 = nn.ReLU()

        self.conv2d_03 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1
        )  # (16, 16, 256)
        self.relu_03 = nn.ReLU()

        self.flatten_04 = nn.Flatten()
        self.linear_04 = nn.Linear(
            in_features=256 * 16 * 16, out_features=1024
        )  # Compressed to latent vector
        self.relu_04 = nn.ReLU()

    def forward(self, x):
        x = self.conv2d_01(x)
        x = self.relu_01(x)
        x = self.conv2d_02(x)
        x = self.relu_02(x)
        x = self.conv2d_03(x)
        x = self.relu_03(x)
        x = self.flatten_04(x)
        x = self.linear_04(x)
        x = self.relu_04(x)
        return x

# Define the Decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.linear_01 = nn.Linear(
            in_features=1024, out_features=256 * 16 * 16
        )
        self.relu_01 = nn.ReLU()
        self.unflatten_01 = nn.Unflatten(
            dim=1, unflattened_size=(256, 16, 16)
        )  # (16, 16, 256)

        self.conv_transpose2d_02 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1
        )  # (32, 32, 128)
        self.relu_02 = nn.ReLU()

        self.conv_transpose2d_03 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1
        )  # (64, 64, 64)
        self.relu_03 = nn.ReLU()

        self.conv_transpose2d_04 = nn.ConvTranspose2d(
            in_channels=64, out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1
        )  # (128, 3, 3)
        self.tanh_04 = nn.Tanh()  # Range: [-1, 1]

    def forward(self, x):
        x = self.linear_01(x)
        x = self.relu_01(x)
        x = self.unflatten_01(x)
        x = self.conv_transpose2d_02(x)
        x = self.relu_02(x)
        x = self.conv_transpose2d_03(x)
        x = self.relu_03(x)
        x = self.conv_transpose2d_04(x)
        x = self.tanh_04(x)
        return x

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def forward(self, x):
        latent = self.encoder(x)  
        reconstructed = self.decoder(latent)
        return reconstructed
    
    def fit(self, dataset_path, number_of_epochs=10, batch_size=32, lr=1e-3):
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        class_name = str(dataset_path).split("/")[-1]
        tmp_path = f"{dataset_path}/{class_name}_tmp"
        if os.path.exists(tmp_path):
            shutil.rmtree(tmp_path)
        os.mkdir(tmp_path)
        for file in os.listdir(dataset_path):
            if os.path.isfile(f"{dataset_path}/{file}"):
                shutil.copy2(f"{dataset_path}/{file}", tmp_path)
        
        dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        self.to(self.device)

        for epoch in range(number_of_epochs):
            self.train()
            running_loss = 0.0

            for batch_idx, (images, _) in enumerate(dataloader):
                images = images.to(self.device)
                # Forward pass
                outputs = self(images)
                loss = criterion(outputs, images)
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / len(dataloader)
            print(f"Epoch [{epoch + 1}/{number_of_epochs}], Loss: {avg_loss:.4f}")
        shutil.rmtree(tmp_path)
    
    def predict(self, path_to_output_image):
        latent_vector = torch.randn(1, 1024).to(self.device)  # Random latent vector
        generated_image = self.decoder(latent_vector) # Generating image
        generated_image = generated_image.squeeze(0).cpu()
        generated_image = (generated_image + 1) / 2.0 # Normalizing [-1, 1] to [0, 1]
        pil_image = transforms.ToPILImage()(generated_image)
        pil_image.save(path_to_output_image)
        
class AEBalancer:
    def __init__(self):
        pass
    
    def fit(self, path_to_input_image_folder, batch_size=128, number_of_epochs= 200, delta=5):
        self.path_to_folder=path_to_input_image_folder        
        directories = os.listdir(path_to_input_image_folder)
        real_directories = {}
        to_generate = {}
        for directory in directories:
            if os.path.isdir(f"{path_to_input_image_folder}/{directory}"):
                count = len(os.listdir(f"{path_to_input_image_folder}/{directory}"))
                real_directories[directory]=count
        max_id = max(real_directories, key=real_directories.get)
        self.total_classes = [key for key in real_directories]
        for directory in real_directories:
            diff = real_directories[max_id] - real_directories[directory]
            if(diff>delta):
                to_generate[directory]=diff
        self.maps = {}
        for key in to_generate:
            count = to_generate[key]
            self.maps[key]=(count, Autoencoder())
        for key in self.maps:
            print(f"{key}: {self.maps[key][0]}, {self.maps[key][1]}")
            autoencoder_model = self.maps[key][1]
            autoencoder_model.fit(
                dataset_path=f"{path_to_input_image_folder}/{key}",
                batch_size=batch_size,
                number_of_epochs=number_of_epochs
            )
            
    def balance(self, path_to_output_image_folder, debug=False):
        for category in self.total_classes:
            source_folder = f"{self.path_to_folder}/{category}"
            destination_folder = f"{path_to_output_image_folder}/{category}"
            
            if os.path.exists(destination_folder) and os.path.isdir(destination_folder):
                shutil.rmtree(destination_folder)
            shutil.copytree(source_folder, destination_folder)
        if debug:
            print(self.maps)
        for category in self.maps:
            last_id = -1
            for file in os.listdir(f"{path_to_output_image_folder}/{category}"):
                id = int(file.split("_")[1].split(".")[0])
                if id > last_id:
                    last_id = id
            for i in range(self.maps[category][0]):
                try:
                    last_id += 1
                    output_file = f"{path_to_output_image_folder}/{category}/{category}_{last_id}.jpg"
                    os.makedirs(os.path.dirname(output_file), exist_ok=True)
                    self.maps[category][1].predict(output_file)
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"Error generating image for category {category}: {e}")