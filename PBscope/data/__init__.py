

def build_dataset(clustering_framework: str, data: str, dataset_path:str, args):
    if clustering_framework.lower()=="cc":
        from data.dataset_implementations import cc as cc
        if data=="PB_scope":
            return cc.PB_space_cc(dataset_path, args.crop_size)
        if data.lower()=="cifar10":
            return cc.cifar10_cc(dataset_path, args.crop_size)


        