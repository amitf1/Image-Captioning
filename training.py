import os
import torch
from torch.backends import cudnn
from tqdm import tqdm
from torch import nn
from torch import optim
from torchvision import transforms
from load_data import CaptionsLoader
from build_model import EncoderDecoder

BASE_DIR = "flickr-image-dataset"
IMAGES_DIR = os.path.join("flickr30k_images", "flickr30k_images", "flickr30k_images")
CAPTIONS_DIR = "flickr30k_images"
CAPTIONS_FILE = "results.csv"
LOAD_MODEL_PATH = None
SAVE_MODEL_PATH = None


def flatten_sequences(output, truth):
    """
    flatten the sequences in each example of the true captions and predictions so we have tensors
    of words instead of sequences for loss and accuracy calculations
    Args:
        output(torch.Tensor): output of the model with sequences of captions (n, seq_len, vocabulary_size)
        truth(torch.Tensor): correct captions (n, seq_len, vocabulary_size)
    """
    sequence_flattened_output = output.reshape(-1, output.shape[-1])
    sequence_flattened_truth = truth.reshape(-1)
    return sequence_flattened_output, sequence_flattened_truth


def fit_model():
    captions_path = os.path.join(BASE_DIR, CAPTIONS_DIR, CAPTIONS_FILE)
    images_path = os.path.join(BASE_DIR, IMAGES_DIR)
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ]
    )
    cl = CaptionsLoader(captions_path, images_path, transform,
                        batch_size=32, num_workers=2, shuffle=True,
                        pin_memory=True, max_unk_freq=2)
    loader, dataset = cl.get_loader()
    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EncoderDecoder(
        embedding_size=256,
        train_all=False,
        num_lstms=1,
        hidden_size=256,
        vocab_size=len(dataset.tokenizer),
        index_to_string=dataset.tokenizer.convert_idx_str
    ).to(device)

    num_epochs = 1000
    lr = 3e-4
    optimizer = optim.Adam(model.parameters(), lr)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.tokenizer.convert_str_idx['<PAD>'])

    # images, captions = next(iter(loader))

    for epoch in range(num_epochs):
        loop = tqdm(enumerate(loader), total=len(loader))
        for batch_idx, (images, captions) in loop:
            images = images.to(device)
            captions = captions.to(device)
            out = model(images, captions[:, :-1])  # don't take the end token as inupt

            # flatten the sequences in each example so we have a tensor of words instead of sequences
            sequence_flattened_output, sequence_flattened_truth = flatten_sequences(out, captions)
            loss = criterion(sequence_flattened_output, sequence_flattened_truth)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # print(loss.item())
            loop.set_description(f'{epoch}/{num_epochs}')
            loop.set_postfix(loss=loss.item())


fit_model()