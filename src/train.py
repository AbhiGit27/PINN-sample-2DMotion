import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from model import PINN, physics_loss, initial_condition_loss, data_loss, g, h0, v0

# 1. Generating completely synthetic data
t_min, t_max = 0.0, 2.0
N_data = 10
t_data = np.linspace(t_min, t_max, N_data)
def true_solution(t): return h0 + v0*t - 0.5*g*(t**2)
h_data_noisy = true_solution(t_data) + 0.7*np.random.randn(N_data)
t_data_tensor = torch.tensor(t_data, dtype=torch.float32).view(-1, 1)
h_data_tensor = torch.tensor(h_data_noisy, dtype=torch.float32).view(-1, 1)

# 2. Initialize model and the optimizer
model = PINN(n_hidden=20)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
lambda_data = lambda_ode = lambda_ic = 2.0

# 3. Efficient Training loop 
model.train()
for epoch in range(4000):
    optimizer.zero_grad()
    l_data = data_loss(model, t_data_tensor, h_data_tensor)
    l_ode = physics_loss(model, t_data_tensor)
    l_ic = initial_condition_loss(model)
    loss = lambda_data*l_data + lambda_ode*l_ode + lambda_ic*l_ic
    loss.backward(); optimizer.step()
    if (epoch+1) % 1000 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")

# 4. Plotting final results
model.eval()
t_plot = np.linspace(t_min, t_max, 100).reshape(-1, 1).astype(np.float32)
h_pred_plot = model(torch.tensor(t_plot)).detach().numpy()
plt.figure(figsize=(8,5))
plt.scatter(t_data, h_data_noisy, color='red', label='Noisy Data')
plt.plot(t_plot, true_solution(t_plot), 'k--', label='True')
plt.plot(t_plot, h_pred_plot, 'b', label='PINN')
plt.legend(); plt.savefig('results/trajectory.png'); plt.show()
print("Plot saved to results/trajectory.png")
