import tenseal as ts
import torch


class DPTrainer:
    def __init__(self, model, noise_multiplier=0.5, l2_norm_clip=1.0):
        self.model = model
        self.noise_multiplier = noise_multiplier
        self.l2_norm_clip = l2_norm_clip
        # Initialize TenSEAL context for encryption
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 60],
        )
        self.context.generate_galois_keys()

    def apply_gradient_noise(self, gradients):
        # Add noise to gradients for differential privacy (encrypted form)
        noise = torch.randn_like(gradients) * self.noise_multiplier
        noisy_gradients = gradients + noise
        encrypted_gradients = self.context.encrypt(noisy_gradients.tolist())
        return encrypted_gradients
