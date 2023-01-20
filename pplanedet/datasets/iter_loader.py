class IterLoader:
    def __init__(self, dataloader, epoch=0):
        self._dataloader = dataloader
        self.iter_loader = iter(self._dataloader)
        self._epoch = epoch

    @property
    def epoch(self):
        return self._epoch

    def __next__(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            self._epoch += 1
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)

        return data

    def __len__(self):
        return len(self._dataloader)