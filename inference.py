import torch
import pathlib
from unet import UNet
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from transformations import normalize_01, re_normalize
import matplotlib.pyplot as plt


def predict(img, model, preprocess, postprocess, device):
    model.eval()
    img = preprocess(img)  # preprocess image
    x = torch.from_numpy(img).to(device)  # to torch, send to device
    with torch.no_grad():
        out = model(x)  # send through model/network

    out_softmax = torch.softmax(out, dim=1)  # perform softmax on outputs
    result = postprocess(out_softmax)  # postprocess outputs

    return result


# root directory
root = pathlib.Path.cwd() / 'CHASEDB1_augmented' / 'Test'
print(root)
def get_filenames_of_path(path: pathlib.Path, ext: str = '*'):
    """Returns a list of files in a directory/path. Uses pathlib."""
    filenames = [file for file in path.glob(ext) if file.is_file()]
    return filenames

# input and target files
images_names = get_filenames_of_path(root / 'Input')
targets_names = get_filenames_of_path(root / 'Target')

# read images and store them in memory
images = [imread(img_name) for img_name in images_names]
targets = [imread(tar_name) for tar_name in targets_names]

# Resize images and targets
images_res = [resize(img, (128, 128, 3)) for img in images]
print(np.where(images_res[0] != 0))
print(len(images_res))
resize_kwargs = {'order': 0, 'anti_aliasing': False, 'preserve_range': True}
targets_res = [resize(tar, (128, 128), **resize_kwargs) for tar in targets]

# device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# model
model = UNet(in_channels=3,
             out_channels=2,
             n_blocks=4,
             start_filters=32,
             activation='relu',
             normalization='batch',
             conv_mode='same',
             dim=2).to(device)


model_name = 'chase_augmented_regularized_model_adam_batch10_lr01.pt'
model_weights = torch.load(pathlib.Path.cwd() / model_name)
print(pathlib.Path.cwd() / model_name)

model.load_state_dict(model_weights)
print(model_weights)

# preprocess function
def preprocess(img: np.ndarray):
    img = np.moveaxis(img, -1, 0)  # from [H, W, C] to [C, H, W]
    img = normalize_01(img)  # linear scaling to range [0-1]
    img = np.expand_dims(img, axis=0)  # add batch dimension [B, C, H, W]
    img = img.astype(np.float32)  # typecasting to float32
    return img


# postprocess function
def postprocess(img: torch.tensor):
    img = torch.argmax(img, dim=1)  # perform argmax to generate 1 channel
    img = img.cpu().numpy()  # send to cpu and transform to numpy.ndarray
    img = np.squeeze(img)  # remove batch dim and channel dim -> [H, W]
    img = re_normalize(img)  # scale it to the range [0-255]
    return img

# predict the segmentation maps 
output = [predict(img, model, preprocess, postprocess, device) for img in images_res]

for i in np.arange(9):
    print(np.where(output[i] != 0))

plt.imshow(images_res[7])
plt.scatter(np.where(output[7] == 1)[0], np.where(output[7] == 1)[1], color='black', s=3)
