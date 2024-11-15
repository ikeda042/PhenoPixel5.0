import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import pickle
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import Column, Integer, String, FLOAT, BLOB
import math


Base = declarative_base()


class Cell(Base):
    __tablename__ = "cells"
    id = Column(Integer, primary_key=True)
    cell_id = Column(String)
    label_experiment = Column(String)
    manual_label = Column(Integer)
    perimeter = Column(FLOAT)
    area = Column(FLOAT)
    img_ph = Column(BLOB)
    img_fluo1 = Column(BLOB, nullable=True)
    img_fluo2 = Column(BLOB, nullable=True)
    contour = Column(BLOB)
    center_x = Column(FLOAT)
    center_y = Column(FLOAT)


# Database setup

dbpath = "sqlite:///experimental/U-net_Pytorch/test_contour_label_data.db"
engine = create_engine(dbpath)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()


# Image parsing function
def parse_image(cell: Cell) -> tuple:
    img_ph = cv2.imdecode(np.frombuffer(cell.img_ph, np.uint8), cv2.IMREAD_COLOR)
    contour = pickle.loads(cell.contour)
    mask = np.zeros_like(img_ph)
    cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)
    masked = cv2.bitwise_and(img_ph, mask)
    masked[mask > 0] = 255
    return img_ph, masked, contour


# Load data from database
cells_with_label_1 = session.query(Cell).filter(Cell.manual_label == 1).all()


# Define the function to combine images into a single image grid
def combine_images_grid(images, grid_size):
    img_height, img_width, _ = images[0].shape
    combined_image = np.zeros(
        (img_height * grid_size, img_width * grid_size, 3), dtype=np.uint8
    )

    for i, img in enumerate(images):
        row = i // grid_size
        col = i % grid_size
        combined_image[
            row * img_height : (row + 1) * img_height,
            col * img_width : (col + 1) * img_width,
        ] = img

    return combined_image


# Load and parse images
images = []
for cell in cells_with_label_1:
    img_ph, _, contour = parse_image(cell)
    cv2.drawContours(img_ph, [contour], -1, (0, 255, 0), 2)
    images.append(img_ph)

# Determine grid size
num_images = len(images)
grid_size = math.ceil(math.sqrt(num_images))

# Pad the images list to ensure it fills the grid completely
while len(images) < grid_size * grid_size:
    images.append(np.zeros_like(images[0]))  # Add blank images for padding if necessary

# Combine images into a single grid
combined_image = combine_images_grid(images, grid_size)

# Save the final combined image
output_path = "experimental/U-net_Pytorch/images/combined_ph_cells_label1.png"
cv2.imwrite(output_path, combined_image)
print(f"Combined image saved as {output_path}")

# # Custom Dataset
# class CellDataset(Dataset):
#     def __init__(self, cells):
#         self.cells = cells

#     def __len__(self):
#         return len(self.cells)

#     def __getitem__(self, idx):
#         cell = self.cells[idx]
#         img_ph, masked = parse_image(cell)
#         img_ph = cv2.resize(img_ph, (256, 256)) / 255.0
#         masked = cv2.resize(masked, (256, 256)) / 255.0
#         img_ph = torch.tensor(img_ph.transpose(2, 0, 1), dtype=torch.float32)
#         masked = torch.tensor(masked[:, :, 0], dtype=torch.float32).unsqueeze(0)
#         return img_ph, masked


# # Define U-Net Model in PyTorch
# class UNet(nn.Module):
#     def __init__(self):
#         super(UNet, self).__init__()
#         # Encoder
#         self.enc1 = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#         )
#         self.pool1 = nn.MaxPool2d(2)

#         self.enc2 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(128, 128, kernel_size=3, padding=1),
#             nn.ReLU(),
#         )
#         self.pool2 = nn.MaxPool2d(2)

#         self.enc3 = nn.Sequential(
#             nn.Conv2d(128, 256, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(),
#         )
#         self.pool3 = nn.MaxPool2d(2)

#         # Bottleneck
#         self.bottleneck = nn.Sequential(
#             nn.Conv2d(256, 512, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             nn.ReLU(),
#         )

#         # Decoder
#         self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
#         self.dec3 = nn.Sequential(
#             nn.Conv2d(512, 256, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(),
#         )

#         self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
#         self.dec2 = nn.Sequential(
#             nn.Conv2d(256, 128, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(128, 128, kernel_size=3, padding=1),
#             nn.ReLU(),
#         )

#         self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
#         self.dec1 = nn.Sequential(
#             nn.Conv2d(128, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#         )

#         # Output layer
#         self.out_conv = nn.Conv2d(64, 1, kernel_size=1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         # Encoder path
#         enc1 = self.enc1(x)
#         enc1_pooled = self.pool1(enc1)

#         enc2 = self.enc2(enc1_pooled)
#         enc2_pooled = self.pool2(enc2)

#         enc3 = self.enc3(enc2_pooled)
#         enc3_pooled = self.pool3(enc3)

#         # Bottleneck
#         bottleneck = self.bottleneck(enc3_pooled)

#         # Decoder path
#         dec3 = self.upconv3(bottleneck)
#         dec3 = torch.cat([dec3, enc3], dim=1)  # Concatenate encoder output
#         dec3 = self.dec3(dec3)

#         dec2 = self.upconv2(dec3)
#         dec2 = torch.cat([dec2, enc2], dim=1)  # Concatenate encoder output
#         dec2 = self.dec2(dec2)

#         dec1 = self.upconv1(dec2)
#         dec1 = torch.cat([dec1, enc1], dim=1)  # Concatenate encoder output
#         dec1 = self.dec1(dec1)

#         # Output layer
#         out = self.out_conv(dec1)
#         return self.sigmoid(out)


# # Set device to MPS
# device = torch.device("mps")

# print("Loading data from database...")
# # Prepare Dataset and DataLoader
# dataset = CellDataset(cells_with_label_1)
# train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# print("Training the model...")
# # Instantiate the model, loss, and optimizer
# model = UNet().to(device)
# criterion = nn.BCELoss()
# optimizer = optim.Adam(model.parameters(), lr=1e-4)

# # Training loop
# num_epochs = 20
# for epoch in range(num_epochs):
#     model.train()
#     epoch_loss = 0
#     for images, masks in train_loader:
#         images, masks = images.to(device), masks.to(device)
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, masks)
#         loss.backward()
#         optimizer.step()
#         epoch_loss += loss.item()
#         print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

#     print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader)}")

# # Save the trained model
# torch.save(model.state_dict(), "experimental/U-net_Pytorch/unet_model.pth")


# # Prediction on a single image
# def predict_contour(model, img_ph):
#     model.eval()
#     img_resized = cv2.resize(img_ph, (256, 256)) / 255.0
#     img_resized = (
#         torch.tensor(img_resized.transpose(2, 0, 1), dtype=torch.float32)
#         .unsqueeze(0)
#         .to(device)
#     )
#     with torch.no_grad():
#         prediction = model(img_resized)
#     prediction = (prediction > 0.5).cpu().numpy().astype(np.uint8) * 255
#     return prediction[0][0]


# for cell in cells_with_label_1:
#     img_ph = cv2.imdecode(np.frombuffer(cell.img_ph, np.uint8), cv2.IMREAD_COLOR)
#     prediction = predict_contour(model, img_ph)
#     cv2.imwrite(
#         f"experimental/U-net_Pytorch/images/predicted/{cell.cell_id}.png", prediction
#     )
