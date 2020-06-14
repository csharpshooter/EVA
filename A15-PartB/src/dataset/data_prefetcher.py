import torch


class DataPrefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        # self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1, 3, 1, 1)
        # self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1, 3, 1, 1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        # if record_stream() doesn't work, another option is to make sure device inputs are created
        # on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input, device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target, device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use by the main stream
        # at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.next_input[0] = self.next_input[0].cuda(non_blocking=True)
            self.next_input[1] = self.next_input[1].cuda(non_blocking=True)
            self.next_input[2] = self.next_input[2].cuda(non_blocking=True)
            self.next_input[3] = self.next_input[3].cuda(non_blocking=True)
            # self.next_target = self.next_target.cuda(non_blocking=True)
            # more code for the alternative if record_stream() doesn't work:
            # copy_ will record the use of the pinned source tensor in this side stream.
            # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
            # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
            # self.next_input = self.next_input_gpu
            # self.next_target = self.next_target_gpu

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input[0] = self.next_input[0].float()
            self.next_input[1] = self.next_input[1].float()
            self.next_input[2] = self.next_input[2].float()
            self.next_input[3] = self.next_input[3].float()

            # self.next_input[0] = self.next_input[0].sub_(self.mean).div_(self.std)
            # self.next_input[1] = self.next_input[1].sub_(self.mean).div_(self.std)
            # self.next_input[2] = self.next_input[2].sub_(self.mean).div_(self.std)
            # self.next_input[3] = self.next_input[3].sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        input[0] = self.next_input[0]
        input[1] = self.next_input[1]
        input[2] = self.next_input[2]
        input[3] = self.next_input[3]

        target = self.next_target
        if input is not None:
            input[0].record_stream(torch.cuda.current_stream())
            input[1].record_stream(torch.cuda.current_stream())
            input[2].record_stream(torch.cuda.current_stream())
            input[3].record_stream(torch.cuda.current_stream())

        # if target is not None:
        #     target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target
