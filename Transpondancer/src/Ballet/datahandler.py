# do the imports
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt

'''
If you need to load an image dataset, it's more convenient to use the ImageFolder class from the 
torchvision.datasets module.
To do so, you need to structure your data as follows:
root_dir
    |_train
        |_class_1
            |_xxx.png
        .....
        .....    
        |_class_n
            |_xxx.png
    |_validation
        |_class_1
            |_xxx.png
        .....
        .....
        |_class_n
            |_xxx.png
that means that each class has its own directory.
By giving this structure, the name of the class will be taken by the name of the folder!
    '''

def custom_transform(padding=(0,0), inference=False):
    """
    padding[0] is the height
    padding[1] is the width
    """
    config = []
    config += [transforms.Grayscale(num_output_channels=1)]
    config += [transforms.Pad(padding, fill=0)]
    config += [transforms.Resize((90, 160))]
    if not inference:
        config += [transforms.RandomHorizontalFlip()]
    config += [transforms.ToTensor(),]
    config += [transforms.Normalize([0.5], [0.5])]

    # custom = transforms.Compose([
    #                     transforms.Grayscale(num_output_channels=1),
    #                     transforms.Pad(padding, fill=0),
    #                     transforms.Resize((90, 160)),
    #                     transforms.RandomHorizontalFlip(),
    #                     transforms.ToTensor(),
    #                     transforms.Normalize([0.5],
    #                                         [0.5])])
                        
    return transforms.Compose(config)

def collate_function(batch, inference=False):
    """
    Since images are of different size in the current problem. This custom function transforms everything 
    to a defined shape by adding respective padding
    """
    samples = [sample[0] for sample in batch]
    labels = [sample[1] for sample in batch]
    images = []
    for image in samples:
        ratio = image.width/image.height
        # print(ratio)
        if 16/9 -0.03<= ratio <= 16/9 +0.03:
            transform = custom_transform(inference=inference)
            image = transform(image)
        elif ratio > 16/9:
                x = int((9/16*image.width - image.height)/2)
                transform = custom_transform((0,x), inference)
                image = transform(image)
        elif ratio < 16/9:
                x = int((16/9*image.height-image.width)/2)
                transform = custom_transform((x,0), inference)
                # print(transform)
                image = transform(image)
        images.append(image)
        
    return images, torch.tensor(labels)

# define a function which takes in path of root_directory, batchsize anad returns the dataloaders
# for both train and test.
def pre_processor(root_dir, batchsize):
    train_data = datasets.ImageFolder(root_dir + '/Train')
    test_data = datasets.ImageFolder(root_dir + '/Validation')

    # create the dataloaders
    train_loader = DataLoader(train_data, batch_size=batchsize, collate_fn=collate_function,  shuffle=True)

    test_loader = DataLoader(test_data, batch_size=batchsize, collate_fn=collate_function,
                                            shuffle=False)

    return train_loader, test_loader


def preprocess_inference(image_path):
    """
    Opens an image and processes it using the same logic as the collate_function.
    Since collate_function expects a batch of (image, label) pairs, we create a dummy batch.
    """
    img = Image.open(image_path)
    # Create a batch with one sample and a dummy label (e.g., 0)
    batch = [(img, 0)]
    processed_images, _ = collate_function(batch, True)
    
    return processed_images[0]  # return the single processed image


def show_tensor_image(tensor, denormalize=True):
    """
    See what the image looks like after preprocessing.
    """
    # Assume tensor shape is (C, H, W). For grayscale, C=1.
    if denormalize:
        tensor = tensor * 0.5 + 0.5  # Reverse normalization
    # Remove channel dimension if it's 1
    if tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    plt.imshow(tensor, cmap='gray')
    plt.axis('off')
    plt.show()