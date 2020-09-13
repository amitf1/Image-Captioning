from build_model import EncoderDecoder
from torchvision import transforms
from torch import optim
import torch
from PIL import Image
import streamlit as st


LOAD_MODEL_PATH = 'image_captioning_model.pth.tar'
MAX_LEN = 50
st.set_option('deprecation.showfileUploaderEncoding', False)


def load_model(path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(path)

    model = EncoderDecoder(
        embedding_size=256,
        train_all=False,
        num_lstms=2,
        hidden_size=256,
        vocab_size=9859,
        index_to_string=checkpoint['index_to_string']
    ).to(device)

    lr = 2e-4
    optimizer = optim.Adam(model.parameters(), lr)

    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.eval()
    return model, device


def predict(image, device, model, max_len):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ]
    )
    image = transform(image.convert("RGB")).unsqueeze(0).to(device)
    prediction = model.predict(image, max_len)
    return prediction


def main():
    model, device = load_model(LOAD_MODEL_PATH)
    st.write("""
             # Image Captioning - Describe the Image
             """
             )
    st.write("This app will describe your picture after uploading it")
    file = st.file_uploader("Please upload an image file", type=["jpg", "png"])
    if file is None:
        st.text("Please upload an image file")
    else:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        prediction = predict(image, device, model, MAX_LEN)
        st.write(prediction)


if __name__ == '__main__':
    main()
