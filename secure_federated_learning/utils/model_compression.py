import torch


class ModelCompression:
    def __init__(self, model):
        self.model = model

    def quantize_model(self):
        # Apply simple quantization to reduce the size of the model
        for param in self.model.parameters():
            param.data = torch.round(param.data)
        return self.model
