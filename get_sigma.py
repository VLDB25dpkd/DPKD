from opacus.accountants import PRVAccountant
import math
from opacus.accountants.utils import get_noise_multiplier


# detla 1/D
# sst2 3.04462890625
# qnli 4.947070312500001
# mnli 3.90185546875
# qqp  2.9130371093750003

# delta 1/10D
# sst2 3.458203125
# qnli 5.6087890625
# mnli 4.360546875
# qqp  3.24765625

print(
    get_noise_multiplier(
        target_epsilon=1,
        target_delta=1 / 250000,
        sample_rate=1024 / 250000,
        epochs=3,
        accountant="prv"
    )
)

def binary_search():
    left = 0.3
    right = 10
    tolerance = 0.001
    target_epsilon = 1.0
    data_size = 364000
    batch_size = 8192 
    epochs = 20
    while True:
        accountant = PRVAccountant()
        accountant.history.append(
            (0.869140625, (1024 / 250000) * (1 / 5), 3 * math.ceil(250000 / 1024))
        )
        mid = (left + right) / 2
        accountant.history.append(
            (mid, batch_size / data_size, epochs * math.ceil(data_size / batch_size))
        )
        epsilon = accountant.get_epsilon(1 / (10*data_size))
        
        print("result")
        print(epsilon)
        print(mid)
        if abs(epsilon - target_epsilon) <= tolerance:
            break
        if epsilon < target_epsilon:
            right = mid
        else:
            left = mid


binary_search()
