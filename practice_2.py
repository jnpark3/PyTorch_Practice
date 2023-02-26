from torch_geometric.datasets import TUDataset

dataset = TUDataset(root='./datasets/enzymes', name='enzymes')

print(len(dataset))

print(dataset.num_classes)

print(dataset.num_node_features)

data = dataset[0]
