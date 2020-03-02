import torch

class Dataloader():

    def __init__(self):
        SEED = 1

        # CUDA?
        cuda = torch.cuda.is_available()
        print("CUDA Available?", cuda)

        if cuda:
            torch.cuda.manual_seed(SEED)
        else:
            # For reproducibility
            torch.manual_seed(SEED)

            # dataloader arguments - something you'll fetch these from cmdprmt
        dataloader_args = dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True) \
            if cuda else dict(shuffle=True, batch_size=64)


        def GetTrainDataLoader(traindataset):
            return torch.utils.data.DataLoader(traindataset, **dataloader_args)

        def GetTestDataLoader(testdataset):
            return torch.utils.data.DataLoader(testdataset, **dataloader_args)
