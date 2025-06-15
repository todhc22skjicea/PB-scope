from data.dataset_implementations import dataset_readers
from data.transforms import CCTransforms
from data.dataset_classes import PBDataset


def _get_cc_datasets(data, labels, crop_size, blur=False, color_jitter_s=0.5, data_val=None,
                     labels_val=None):
    if data_val is None:
        data_val, labels_val = data, labels
    if crop_size != 224:
        blur = True
    train_dataset = PBDataset(data, annotations=labels,
                                 transformations=CCTransforms(crop_size, s=color_jitter_s,
                                                              blur=blur,k=1))


    val_dataset = PBDataset(data_val, annotations=labels_val,
                               transformations=CCTransforms(crop_size, validation=True))
    return train_dataset, val_dataset


def PB_space_cc(dataset_path=None, crop_size=224, *args, **kwargs):
    data, labels, _, _ = dataset_readers.get_PB(dataset_path, "PB-space")
    train_dataset, val_dataset = _get_cc_datasets(data, labels, crop_size, blur=True, color_jitter_s=1)
    return train_dataset, val_dataset

def cifar10_cc(dataset_path=None, crop_size=224, *args, **kwargs):
    data, labels = dataset_readers.get_cifar10(dataset_path, ["merge"])
    train_dataset, val_dataset = _get_cc_datasets(data, labels, crop_size, blur=False)
    return train_dataset, val_dataset