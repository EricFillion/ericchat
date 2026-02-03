from fsspec.callbacks import Callback


class BytesCallback(Callback):
    def __init__(self, on_update):
        super().__init__()
        self.on_update = on_update
    def set_size(self, size):
        super().set_size(size)
        self.on_update(0, size)
    def relative_update(self, inc):
        super().relative_update(inc)
        self.on_update(self.value, self.size)
    def absolute_update(self, value):
        super().absolute_update(value)
        self.on_update(value, self.size)

