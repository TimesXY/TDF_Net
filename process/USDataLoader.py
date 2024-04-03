import os
import random

from PIL import Image
from torch.utils.data import Dataset


class USIDataset(Dataset):
    def __init__(self, txt_path, transform=None):
        fh_txt = open(txt_path, 'r')
        images_labels = []

        # loop over txt files
        for line in fh_txt:
            line = line.rstrip()  # default deletion is blank characters ('\n', '\r', '\t', ' ')
            words = line.split()  # default split by space, newline (\n), tab (\t)
            images_labels.append((words[0], int(words[1])))

        # the image paths and labels
        self.images_labels = images_labels
        self.transform = transform

    def __getitem__(self, index):

        # the image paths and labels
        images_path, label = self.images_labels[index]

        # define the selected image prefixes
        prefixes = ["BUS_", "DUS_", "EUS_"]

        # one randomly selected image from each type
        selected_images = []

        for prefix in prefixes:
            # get the all matching images
            matching_images = [f for f in os.listdir(images_path) if f.startswith(prefix) and f.endswith(".jpg")]

            # if the matching images exist, choose one at random
            if matching_images:
                selected_image = random.choice(matching_images)
                selected_images.append(selected_image)

        # get the images of different modalities
        all_images = []

        # get the image
        for img_name in selected_images:
            img_path = os.path.join(images_path, img_name)
            img_s = Image.open(img_path).convert('RGB')

            # data augmentation
            if self.transform is not None:
                img_s = self.transform(img_s)

            # get the images of different modalities
            all_images.append(img_s)

        return all_images, label

    def __len__(self):
        return len(self.images_labels)


class USIDatasetFix(Dataset):
    def __init__(self, txt_path, transform=None):
        fh_txt = open(txt_path, 'r')
        images_labels = []

        # loop over txt files
        for line in fh_txt:
            line = line.rstrip()  # default deletion is blank characters ('\n', '\r', '\t', ' ')
            words = line.split()  # default split by space, newline (\n), tab (\t)
            images_labels.append((words[0], int(words[1])))

        # the image paths and labels
        self.images_labels = images_labels
        self.transform = transform

    def __getitem__(self, index):

        # the image paths and labels
        images_path, label = self.images_labels[index]

        # define the selected image prefixes
        prefixes = ["BUS_1", "DUS_1", "EUS_1"]

        # one randomly selected image from each type
        selected_images = []

        for prefix in prefixes:
            # get the all matching images
            matching_images = [f for f in os.listdir(images_path) if f.startswith(prefix) and f.endswith(".jpg")]

            # if the matching images exist, choose one at random
            if matching_images:
                selected_image = random.choice(matching_images)
                selected_images.append(selected_image)

        # get the images of different modalities
        all_images = []

        # get the image
        for img_name in selected_images:
            img_path = os.path.join(images_path, img_name)
            img_s = Image.open(img_path).convert('RGB')

            # data augmentation
            if self.transform is not None:
                img_s = self.transform(img_s)

            # get the images of different modalities
            all_images.append(img_s)

        return all_images, label

    def __len__(self):
        return len(self.images_labels)


class USIDatasetWithPath(Dataset):
    def __init__(self, txt_path, transform=None):
        fh_txt = open(txt_path, 'r')
        images_labels = []

        # loop over txt files
        for line in fh_txt:
            line = line.rstrip()  # default deletion is blank characters ('\n', '\r', '\t', ' ')
            words = line.split()  # default split by space, newline (\n), tab (\t)
            images_labels.append((words[0], int(words[1])))

        # the image paths and labels
        self.images_labels = images_labels
        self.transform = transform

    def __getitem__(self, index):

        # the image paths and labels
        images_path, label = self.images_labels[index]

        # define the selected image prefixes
        prefixes = ["BUS_", "DUS_", "EUS_"]

        # one randomly selected image from each type
        selected_images = []

        for prefix in prefixes:
            # get the all matching images
            matching_images = [f for f in os.listdir(images_path) if f.startswith(prefix) and f.endswith(".jpg")]

            # if the matching images exist, choose one at random
            if matching_images:
                selected_image = random.choice(matching_images)
                selected_images.append(selected_image)

        # get the images of different modalities
        all_images = []

        # get the image
        for img_name in selected_images:
            img_path = os.path.join(images_path, img_name)
            img_s = Image.open(img_path).convert('RGB')

            # data augmentation
            if self.transform is not None:
                img_s = self.transform(img_s)

            # get the images of different modalities
            all_images.append(img_s)

        return all_images, label, images_path

    def __len__(self):
        return len(self.images_labels)
