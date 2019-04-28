#%%
%matplotlib inline

import torch
import matplotlib.pyplot as plt

relu = torch.nn.ReLU()
x = torch.range(-5., 5, 0.1)
y = relu(x)

plt.plot(x.numpy(), y.detach().numpy())
plt.show()

prelu = torch.nn.PReLU(num_parameters=1)
y = prelu(x)

plt.plot(x.numpy(), y.detach().numpy())
plt.show()
