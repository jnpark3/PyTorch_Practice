import torch
from torch_geometric.data import Data

"""
1. Data format:
"""

# edge_index denotes connections between nodes, dimensions 2xE (remember both directions)

edge_index = torch.tensor([[0, 1, 1, 2, 3, 2],
                           [1, 0, 2, 1, 2, 3]], dtype=torch.long)

# x denotes feature matrix for each node

x = torch.tensor([[-1, 2], [0, 3], [1, 4], [2, 4]], dtype=torch.float)

# creating an instance of torch_geometric.data.Data

data = Data(x=x, edge_index=edge_index)

# check to make sure created instance is valid

data.validate(raise_on_error=True)

# note that data object only contains the attributes you give it

for key, item in data:
    print(f'{key} found in data')

"""
2. Data tools:
"""

# get number of nodes

print("Number of nodes "f"{data.num_nodes}")

# get number of edges

print("Number of edges "f"{data.num_edges}")

# get number of features in data

print("Number of features in data "f"{data.num_node_features}")

# does data have isolated nodes?

print(data.has_isolated_nodes())

# does data have self loops?

print(data.has_self_loops())

# is the graph directed?

print(data.is_directed())

"""
3. Transfer data object to GPU
"""

# Transfer data object to GPU.

device = torch.device('cpu')
data = data.to(device)
