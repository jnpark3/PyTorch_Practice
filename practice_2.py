from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch

"""
1. Sample dataset enzymes
"""

# Note dataset is an array of Data instances

dataset = TUDataset(root='./datasets/enzymes', name='enzymes')

print("Enzymes: --------------------------------------")
print("length: "f"{len(dataset)}, # classes: "f"{dataset.num_classes} # node features: "f"{dataset.num_node_features}")

data = dataset[0]

print(dataset)
print(data)

"""
2. Sample dataset Cora
"""

dataset = Planetoid(root='./datasets/planetoid', name='Cora')

print("Planetoid: --------------------------------------")
print("length: "f"{len(dataset)}, # classes: "f"{dataset.num_classes} # node features: "f"{dataset.num_node_features}")

data = dataset[0]

print(dataset)
print(data)

# This data has a mask to denote which nodes to test, train, validate

print("Train: " + str(data.train_mask.sum().item()))
print("Validate: " + str(data.val_mask.sum().item()))
print("Test: " + str(data.test_mask.sum().item()))

"""
3. Dataset operations
"""

dataset = TUDataset(root='./datasets/enzymes', name='enzymes')

# Shuffling dataset

dataset = dataset.shuffle()

# 90-10 Train-test split

train_dataset = dataset[:540]
test_dataset = dataset[540:]

# Divides dataset into a collection of batche objects, which inherets the 
# data object with a new property called "batch" which designates which 
# graph each node is in.

loader = DataLoader(dataset, batch_size=32, shuffle=True)

for data in loader:
    print(data)

    print(data.num_graphs)

    x = scatter(data.x, data.batch, dim=0, reduce='mean')
    print(x.size())

"""
4. 
"""

dataset = Planetoid(root='./datasets/planetoid', name='Cora')

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, 16)
        self.conv3 = GCNConv(16, dataset.num_classes)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)   
        
        weights, biases = list(self.conv2.parameters())     
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)

        return F.log_softmax(x, dim=1)


device = torch.device('cpu')
model = GCN().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    # print(out[data.train_mask])
    loss.backward()
    optimizer.step()

model.eval()
pred = model(data).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print(f'Accuracy: {acc:.4f}')