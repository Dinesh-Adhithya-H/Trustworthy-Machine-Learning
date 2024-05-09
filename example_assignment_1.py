import torch
from torch.utils.data import Dataset
from typing import Tuple
import numpy as np
import requests
import pandas as pd
import matplotlib.pyplot as plt


#### LOADING THE MODEL

from torchvision.models import resnet18

model = resnet18(weights=False)
model.fc = torch.nn.Linear(512, 44)

ckpt = torch.load("./01_MIA_67.pt", map_location="cpu")

model.load_state_dict(ckpt)

#### DATASETS

class TaskDataset(Dataset):
    def __init__(self, transform=None):

        self.ids = []
        self.imgs = []
        self.labels = []

        self.transform = transform

    def __getitem__(self, index) -> Tuple[int, torch.Tensor, int]:
        id_ = self.ids[index]
        img = self.imgs[index]
        if not self.transform is None:
            img = self.transform(img)
        label = self.labels[index]
        return id_, img, label

    def __len__(self):
        return len(self.ids)


class MembershipDataset(TaskDataset):
    def __init__(self, transform=None):
        super().__init__(transform)
        self.membership = []

    def __getitem__(self, index) -> Tuple[int, torch.Tensor, int, int]:
        id_, img, label = super().__getitem__(index)
        return id_, img, label, self.membership[index]


data: MembershipDataset = torch.load("./priv_out.pt")

data: MembershipDataset = torch.load("./pub.pt")


# # Load a few images from the dataset
# num_images = 5
# for i in range(num_images):
#     id_, img, label,member_status = data[i]
#     plt.figure()
#     plt.imshow(img.permute(1, 2, 0))  # permute dimensions to match what matplotlib expects
#     plt.title(f"Label: {label}")
#     plt.show()

# Assuming `model` is your trained model
scores = []
id_, img, label, member_status = data[0:10]
img = torch.stack(img)  # convert list of tensors to a single tensor
score = model(img)  # get model's score
scores.append(score.detach())  # detach the score from the computation graph
    
print(scores[0])
confidence_score = torch.nn.functional.softmax(scores[0], dim=1)
print(confidence_score)

# Now `scores` is a list of model's scores for each data point in the dataset
    
# #### EXAMPLE SUBMISSION
# """ Do a memebership inference attack on the dataset. """

# df = pd.DataFrame(
#     {
#         "ids": data.ids,
#         "score": np.random.randn(len(data.ids)),
#     }
# )
# df.to_csv("test.csv", index=None)
# response = requests.post("http://35.184.239.3:9090/mia", files={"file": open("test.csv", "rb")}, headers={"token": "TOKEN"})
# print(response.json())
