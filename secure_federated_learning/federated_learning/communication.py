import torch
import numpy as np


class CommunicationOptimization:
    def __init__(self, model, compression_factor=0.5):
        self.model = model
        self.compression_factor = compression_factor

    def compress_model(self):
        # Simple compression technique: Quantization of model weights
        for param in self.model.parameters():
            param.data = (
                torch.round(param.data / self.compression_factor)
                * self.compression_factor
            )

        return self.model
