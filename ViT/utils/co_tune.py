from PIL import Image
from torchvision import transforms
from torchvision import datasets
import os
from torch.utils.data import DataLoader


__all__ = ['get_transforms']


class ResizeImage():
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))


class PlaceCrop(object):
    """Crops the given PIL.Image at the particular index.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (w, h), a square crop (size, size) is
            made.
    """

    def __init__(self, size, start_x, start_y):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.start_x = start_x
        self.start_y = start_y

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        th, tw = self.size
        return img.crop((self.start_x, self.start_y, self.start_x + tw, self.start_y + th))


class ForceFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        return img.transpose(Image.FLIP_LEFT_RIGHT)


def transform_train(normalize, resize_size=256, crop_size=224, color_distort=False):
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])

    if color_distort:
        print("#################### with color_distort ####################")
        color_transform = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)

        tfs = [
            ResizeImage(resize_size),
            color_transform,
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    else:
        tfs = [
            ResizeImage(resize_size),
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]

    return transforms.Compose(tfs)


def transform_val(normalize, resize_size=256, crop_size=224):
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    start_center = (resize_size - crop_size - 1) / 2

    return transforms.Compose([
        ResizeImage(resize_size),
        PlaceCrop(crop_size, start_center, start_center),
        transforms.ToTensor(),
        normalize
    ])


def transform_test(normalize, resize_size=256, crop_size=224):
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    # ten crops for image test
    start_first = 0
    start_center = (resize_size - crop_size - 1) / 2
    start_last = resize_size - crop_size - 1
    data_transforms = {}
    data_transforms['test0'] = transforms.Compose([
        ResizeImage(resize_size), ForceFlip(),
        PlaceCrop(crop_size, start_first, start_first),
        transforms.ToTensor(),
        normalize
    ])
    data_transforms['test1'] = transforms.Compose([
        ResizeImage(resize_size), ForceFlip(),
        PlaceCrop(crop_size, start_last, start_last),
        transforms.ToTensor(),
        normalize
    ])
    data_transforms['test2'] = transforms.Compose([
        ResizeImage(resize_size), ForceFlip(),
        PlaceCrop(crop_size, start_last, start_first),
        transforms.ToTensor(),
        normalize
    ])
    data_transforms['test3'] = transforms.Compose([
        ResizeImage(resize_size), ForceFlip(),
        PlaceCrop(crop_size, start_first, start_last),
        transforms.ToTensor(),
        normalize
    ])
    data_transforms['test4'] = transforms.Compose([
        ResizeImage(resize_size), ForceFlip(),
        PlaceCrop(crop_size, start_center, start_center),
        transforms.ToTensor(),
        normalize
    ])
    data_transforms['test5'] = transforms.Compose([
        ResizeImage(resize_size),
        PlaceCrop(crop_size, start_first, start_first),
        transforms.ToTensor(),
        normalize
    ])
    data_transforms['test6'] = transforms.Compose([
        ResizeImage(resize_size),
        PlaceCrop(crop_size, start_last, start_last),
        transforms.ToTensor(),
        normalize
    ])
    data_transforms['test7'] = transforms.Compose([
        ResizeImage(resize_size),
        PlaceCrop(crop_size, start_last, start_first),
        transforms.ToTensor(),
        normalize
    ])
    data_transforms['test8'] = transforms.Compose([
        ResizeImage(resize_size),
        PlaceCrop(crop_size, start_first, start_last),
        transforms.ToTensor(),
        normalize
    ])
    data_transforms['test9'] = transforms.Compose([
        ResizeImage(resize_size),
        PlaceCrop(crop_size, start_center, start_center),
        transforms.ToTensor(),
        normalize
    ])

    return data_transforms


def get_transforms(normalize, resize_size=256, crop_size=224, color_distort=False):
    transforms = {
        'train': transform_train(normalize, resize_size, crop_size, color_distort),
        'val': transform_val(normalize, resize_size, crop_size),
    }
    transforms.update(transform_test(normalize, resize_size, crop_size))

    return transforms


def get_data_loader(configs):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    data_transforms = get_transforms(normalize, resize_size=256, crop_size=224)

    # build dataset
    train_dataset = datasets.ImageFolder(
        os.path.join(configs.data, 'train'),
        transform=data_transforms['train'])
    determin_train_dataset = datasets.ImageFolder(
        os.path.join(configs.data, 'train'),
        transform=data_transforms['val'])
    val_dataset = datasets.ImageFolder(
        os.path.join(configs.data, 'val'),
        transform=data_transforms['val'])
    test_datasets = datasets.ImageFolder(
        os.path.join(configs.data, 'test'),
        transform=data_transforms['val'])

    # build dataloader
    train_loader = DataLoader(train_dataset, batch_size=configs.train_batch_size, shuffle=True,
                              num_workers=configs.num_workers, pin_memory=True)
    # determin_train_loader = DataLoader(determin_train_dataset, batch_size=configs.batch_size, shuffle=False,
    #                                    num_workers=configs.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=configs.eval_batch_size, shuffle=False,
                            num_workers=configs.num_workers, pin_memory=True)
    test_loader = DataLoader(test_datasets, batch_size=configs.eval_batch_size, shuffle=False,
                             num_workers=configs.num_workers, pin_memory=True)
    # test_loaders = {
    #     'test' + str(i):
    #         DataLoader(
    #             test_datasets["test" + str(i)],
    #             batch_size=4, shuffle=False, num_workers=configs.num_workers
    #     )
    #     for i in range(10)
    # }

    return train_loader, val_loader, test_loader
