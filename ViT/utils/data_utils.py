import logging

import torch

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler

from ViT.dataset.init_datasets import init_datasets

from .co_tune import get_transforms


def get_loader(args):
    if "resnet" in args.model_type:
        print("norm of resnet")
        normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    else:
        normalization = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    if args.cotuning_trans:
        print("employ the transform of co-tuning")
        if args.img_size == 32:
            resize_size = 32
            crop_size = 32
        elif args.img_size == 224:
            resize_size = 256
            crop_size = 224
        else:
            raise ValueError("unknow image size of {}".format(args.img_size))

        data_transforms = get_transforms(normalization, resize_size, crop_size, args.color_distort)
        transform_train, transform_test = data_transforms['train'], data_transforms['val']
    else:
        train_trans = [
            transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
            transforms.ToTensor(),
            normalization,
        ]

        if args.train_resize_first:
            train_trans = [transforms.Resize((args.img_size, args.img_size)), ] + train_trans

        if args.color_distort:
            print("########## color_distort ##########")
            color_transform = transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
            )
            train_trans = [color_transform, ] + train_trans

        transform_train = transforms.Compose(train_trans)
        transform_test = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            normalization,
        ])

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    trainset, valset, testset = init_datasets(args, transform_train, transform_test)

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = DistributedSampler(trainset)
    val_sampler = SequentialSampler(valset)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=args.num_workers,
                              pin_memory=True)
    val_loader = DataLoader(valset,
                            sampler=val_sampler,
                            batch_size=args.eval_batch_size,
                            num_workers=args.num_workers,
                            pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=args.num_workers,
                             pin_memory=True)

    return train_loader, val_loader, test_loader
