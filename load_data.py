import os
import en_core_web_sm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
from PIL import Image
from collections import Counter
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

BASE_DIR = "flickr-image-dataset"
IMAGES_DIR = os.path.join("flickr30k_images", "flickr30k_images", "flickr30k_images")
CAPTIONS_DIR = "flickr30k_images"
CAPTIONS_FILE = "results.csv"


class CaptionsTokenizer:
    """
    This class allows to vectorize captions, by turning each text into a sequence of integers
     (each integer being the index of a token in a dictionary)
    """

    def __init__(self, max_unk_freq):
        """
        Args:
            max_unk_freq (int): Maximum frequency for a token to be considered unknown and be replaced with <UNK> token.
        """
        self.max_unk_freq = max_unk_freq
        self.convert_str_idx = {'<PAD>': 0, '<UNK>': 2, '<SOS>': 3, '<EOS>': 4}
        self.convert_idx_str = {v: k for k, v in self.convert_str_idx.items()}
        self.unk = []
        self.nlp = en_core_web_sm.load()

    def _tokenize(self, caption):
        """
        This method separates a caption into a sequence of tokens
        Args:
            caption (str): A text with a caption of an image
        Returns:
            a sequence of the caption's tokens
        """
        return [token.text for token in
                self.nlp.tokenizer(str(caption).lower().strip())]

    def fit_on_text(self, captions):
        """
        This method is building a vocabulary based on the given captions and creates the mapping from indexes to tokens.
        Args:
            captions (list): A  list of texts with captions of images

        """
        tokenized_captions = []
        for caption in captions:
            tokenized_text = self._tokenize(caption)
            tokenized_captions.append(tokenized_text)
        tok_count = Counter(np.hstack(tokenized_captions))
        i = max(self.convert_idx_str.keys()) + 1
        for word, n in tok_count.items():
            if n > self.max_unk_freq:
                self.convert_str_idx[word] = i
                self.convert_idx_str[i] = word
                i += 1
            else:
                self.unk.append(word)

    def text_to_sequence(self, caption):
        """
        This method takes a text and converts it to a numerical sequence based on the
         fitted vocabulary mapping (fit_on_text should be called first), including adding start, end and unknown tokens.
        Args:
            caption (str): A text with a caption of an image
        Returns:
            a sequence of the caption's tokens in their numeric representation, based on the mapping
            created when fit_on_text was called
        """
        tokenized_text = self._tokenize(caption)
        numerical_sequence = [
            self.convert_str_idx[token] if token in self.convert_str_idx
            else self.convert_str_idx["<UNK>"] for token in tokenized_text
        ]
        return [self.convert_str_idx['<SOS>']] + numerical_sequence + [self.convert_str_idx['<EOS>']]


class CaptionsDataset(Dataset):
    """
    Build a custom torch dataset for captions and images
    """

    def __init__(self, captions_file, images_dir, transform=None, max_unk_freq=2):
        """
        Args:
            captions_file (str): Path to the csv file with the captions, the first column should be the image name,
            the second column should be the number of image's caption and the third column should be the caption's text.
            images_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            max_unk_freq (int): Maximum frequency for a token to be considered unknown and be replaced with <UNK> token.
        """
        self.df = pd.read_csv(captions_file, delimiter='|', header=0, names=['image_name', 'caption_number', 'caption'])
        self.images_dir = images_dir
        self.transform = transform
        self.images_names = self.df['image_name']
        self.captions = self.df['caption']

        # Initialize the tokenizer based on the captions
        self.tokenizer = CaptionsTokenizer(max_unk_freq)
        self.tokenizer.fit_on_text(self.captions.to_list())

    def __len__(self):
        """Returns the length of the dataset"""
        return self.df.shape[0]

    def __getitem__(self, idx):
        """
        This method return a single item from the dataset on the specified index.
         An item consists of an image and a caption
        Args:
            idx (int): index of the desired item in the dataset
        Returns:
            (tuple) An image and a numerical sequence tensor representing a caption
        """
        caption = self.captions[idx]
        image_name = self.images_names[idx]
        image = Image.open(os.path.join(self.images_dir, image_name)).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        sequence = self.tokenizer.text_to_sequence(caption)
        return image, torch.tensor(sequence)


class CaptionsLoader:
    def __init__(self, captions_file, images_dir, transform,
                 batch_size=32, num_workers=8, shuffle=True, pin_memory=True, max_unk_freq=2):
        self.captions_file = captions_file
        self.images_dir = images_dir
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = 8
        self.shuffle = shuffle
        self.pin_memory = True
        self.dataset = CaptionsDataset(captions_file, images_dir, transform, max_unk_freq)

    def _generate_batch(self, batch):
        images = []
        captions = []
        for image, caption in batch:
            image = image.unsqueeze(0)
            images.append(image)
            captions.append(caption)
        captions = pad_sequence(captions, batch_first=True,
                                padding_value=self.dataset.tokenizer.convert_str_idx['<PAD>'])
        return torch.cat(images, dim=0), captions

    def get_loader(self):
        data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle,
                                 num_workers=self.num_workers, collate_fn=self._generate_batch,
                                 pin_memory=self.pin_memory)
        return data_loader, self.dataset


if __name__ == "__main__":
    captions = os.path.join(BASE_DIR, CAPTIONS_DIR, CAPTIONS_FILE)
    images = os.path.join(BASE_DIR, IMAGES_DIR)
    trans = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(), ]
    )

    cl = CaptionsLoader(captions, images, trans,
                        batch_size=32, num_workers=8, shuffle=True,
                        pin_memory=True, max_unk_freq=2)
    loader, dataset = cl.get_loader()
    print(os.path.dirname(os.path.realpath(__file__)))
    for idx, (iamges, captions) in enumerate(loader):
        print(iamges.shape)
        print(captions.shape)
