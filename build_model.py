import torch
from torch import nn
from torchvision import models


class ImageEncoder(nn.Module):
    def __init__(self, embedding_size, train_all=False):
        super(ImageEncoder, self).__init__()
        self.train_all = train_all
        self.image_model = models.resnext50_32x4d(pretrained=True)
        self.image_model.fc = nn.Linear(self.image_model.fc.in_features, embedding_size)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, images):
        if not self.train_all:
            for name, param in self.image_model.named_parameters():
                if 'fc.weight' not in name and 'fc.bias' not in name:
                    param.requires_grad = False
        return self.dropout(self.relu(self.image_model(images)))


class SequenceDecoder(nn.Module):
    def __init__(self, embedding_size, num_lstms, hidden_size, vocab_size):
        super(SequenceDecoder, self).__init__()
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_lstms, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, sequences):
        hidden_states, _ = self.lstm(sequences)
        return self.fc(hidden_states)


class EncoderDecoder(nn.Module):
    def __init__(self, embedding_size, train_all, num_lstms, hidden_size, vocab_size, index_to_string):
        super(EncoderDecoder, self).__init__()
        self.index_to_string = index_to_string
        self.encoder = ImageEncoder(embedding_size, train_all)
        self.decoder = SequenceDecoder(embedding_size, num_lstms, hidden_size, vocab_size)
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, images, captions):
        image_features = self.encoder(images).unsqueeze(1)
        embeddings = self.dropout(self.embedding(captions))
        embeddings = torch.cat((image_features, embeddings), dim=1)
        return self.decoder(embeddings)

    def predict(self, image, max_pred_length):
        prediction = []
        cell_states = None

        with torch.no_grad():
            x = self.encoder(image)
            for _ in range(max_pred_length):
                hidden_states, cell_states = self.decoder.lstm(x, cell_states)
                out = self.decoder.fc(hidden_states[1]).argmax(1)
                prediction.append(out.item())
                if self.index_to_string[out.item()] == '<EOS>':
                    break
                x = self.embedding(out).unsqueeze(0)
        prediction = [self.index_to_string[idx] for idx in prediction
                      if self.index_to_string[idx] not in ['<SOS>', '<EOS>']]
        return prediction
