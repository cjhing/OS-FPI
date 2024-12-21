import torch
from .SiamUAV import SiamUAVCenter,SiamUAV_test


def make_dataset(opt,train=True):
    if train:
        image_datasets = SiamUAVCenter(opt.data_dir,opt)
        dataloaders =torch.utils.data.DataLoader(image_datasets,
                                                 batch_size=opt.batchsize,
                                                 shuffle=True,
                                                 num_workers=opt.num_worker,
                                                 pin_memory=True,
                                                 # collate_fn=train_collate_fn
                                                 )
        dataset_sizes = {x: len(image_datasets) for x in ['satellite', 'drone']}
        return dataloaders, dataset_sizes

    else:
        dataset_test = SiamUAV_test(opt.data_dir, opt,mode="merge_test_700-1800_cr0.95_stride100")
        dataloaders = torch.utils.data.DataLoader(dataset_test,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=opt.num_worker,
                                                  pin_memory=True)
        return dataloaders




