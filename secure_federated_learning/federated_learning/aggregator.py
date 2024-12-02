import torch


class Aggregator:
    def __init__(self):
        pass

    def aggregate(self, model_updates):
        # Aggregates model weights by averaging across all clients' models
        aggregated_weights = {}

        for key in model_updates[0].keys():
            aggregated_weights[key] = torch.mean(
                torch.stack([update[key] for update in model_updates]), dim=0
            )

        return aggregated_weights
