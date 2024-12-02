import tenseal as ts
import torch


class SMPC:
    def __init__(self, workers):
        self.workers = workers
        # Initialize a TenSEAL context for each worker (using homomorphic encryption)
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 60],
        )
        self.context.generate_galois_keys()

    def share_data(self, tensor):
        # Encrypt tensor and send to workers
        encrypted_tensor = self.context.encrypt(
            torch.tensor(tensor, dtype=torch.float32).tolist()
        )
        return encrypted_tensor

    def reconstruct_data(self, encrypted_tensor):
        # Decrypt the data after computation
        return torch.tensor(self.context.decrypt(encrypted_tensor))
