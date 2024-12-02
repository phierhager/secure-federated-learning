import torch
import tenseal as ts


class FederatedTrainer:
    def __init__(self, model, train_data, workers, epochs=5, batch_size=32):
        # Initialize the trainer with the model, training data, workers (clients), and training parameters.
        self.model = model
        self.train_data = train_data
        self.workers = workers
        self.epochs = epochs
        self.batch_size = batch_size

    def train(self):
        # This method handles the federated training process for multiple workers.
        # Each worker will receive a portion of the data, train the model locally, and send updates back.

        # Use the first worker's context to create an Encryptor
        encryptor = self.workers[0].encryptor

        for epoch in range(self.epochs):
            # Step 1: Split the data across workers and send it for training.
            for i, worker in enumerate(self.workers):
                # Split data for each worker
                worker_data = self.train_data[
                    i :: len(self.workers)
                ]  # Distribute data to workers

                # Step 2: Encrypt the data for the current worker using the Encryptor
                encrypted_data = [
                    encryptor(tensor.tolist()) for tensor, _ in worker_data
                ]

                # Step 3: Simulate training on encrypted data.
                # Here we would normally train on the data, but this example doesn't perform actual updates.
                for data, target in encrypted_data:
                    output = self.model(
                        torch.tensor(data)
                    )  # Perform a forward pass on encrypted data.
                    loss = torch.nn.functional.nll_loss(
                        output, torch.tensor(target)
                    )  # Calculate the loss.
                    loss.backward()  # Backpropagate the loss.

            print(
                f"Epoch {epoch + 1}/{self.epochs} completed."
            )  # Notify when an epoch finishes
