from torch_geometric.datasets import TUDataset
from torch_geometric.utils import scatter
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch

dataset = TUDataset(root='./datasets/Peking_1', name='Peking_1')

class FFGNN(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, 16)
        self.conv3 = GCNConv(16, dataset.num_classes)

        self.threshold = 0
        self.layers = [self.conv1, self.conv2, self.conv3]
        self.training_rate = 0.005


    def forward(self, data, positive):
        """
        Greedy layer-based training given a posive or negative data set
        inputs:
            data (graph): batch of data to use for training
            positive (boolean): positivity of given graph
        output: Aggregate Loss
        """

        x, edge_index = data.x, data.edge_index
        aggregate_loss = 0

        for conv in self.layers:
            z = self.activation_normalizer(x)

            # Message Passing
            z = conv(x, edge_index)

            # RELU
            z = F.relu(z)

            # Goodness/Loss
            goodness = self.layer_goodness(z)
            loss = self.local_loss(goodness, positive)
            aggregate_loss += loss

            # Update weights and biases
            driv_loss = self.loss_derivative(goodness, positive)
            driv_loss = self.activation_gradient(z)
            weighted_deriv = driv_loss * driv_loss

            dL_dW = torch.matmul(x.t(), weighted_deriv)
            dL_db = torch.sum(weighted_deriv, dim=0)

            biases, weights = list(self.conv1.parameters())     

            weights.data -= self.training_rate * dL_dW.t()
            biases.data -= self.training_rate * dL_db

            # Set next layer
            x = z

        return aggregate_loss


    def layer_goodness(self, activation_matrix):
        """
        Layer goodness function given an activation matrix
        inputs:
            activation_matrix (2d tensor): matrix of activation vectors for each node
        output: float denoting the layer goodness value
        """

        return (1/len(activation_matrix[0])) * activation_matrix.pow(2).sum()


    def local_loss(self, goodness, positive = True):
        """
        Layer loss function given a layer's goodness
        inputs:
            goodness (float): the goodness value of a layer
            positive (boolean): denoting whether or not the training data is positive
        output: float denoting the layer loss value
        """

        if positive:
            if goodness > 10 + self.threshold:
                return 0
            if goodness < self.threshold - 10:
                return self.threshold - goodness
            return torch.log(1 + torch.exp(-goodness + self.threshold))

        if goodness > 10 + self.threshold:
            return self.threshold + goodness
        if goodness < self.threshold - 10:
            return 0
        return torch.log(1 + torch.exp(goodness + self.threshold))


    def activation_normalizer(self, activation_matrix):
        """
        Function that normalizes every activation vector in matrix
        inputs:
            activation_matrix (2d tensor): matrix of activation vectors for each node
        output: activation_matrix normalized using Frobenius norm
        """

        row_norms = torch.norm(activation_matrix, p=2, dim=1, keepdim=True)
        return activation_matrix / row_norms


    def loss_derivative(self, goodness, positive = True):
        """
        Layer loss's derivative given a layer's goodness
        inputs:
            goodness (float): the goodness value of a layer
            positive (boolean): denoting whether or not the training data is positive
        output: float denoting the layer loss's derivative
        """

        if positive:
            return -1/(1 + 1/(torch.exp(self.threshold - goodness)))
        return 1/(1 + 1/(torch.exp(goodness + self.threshold)))


    def activation_gradient(self, activation_matrix):
        """
        Layer loss's derivative given a layer's goodness
        inputs:
            goodness (float): the goodness value of a layer
            positive (boolean): denoting whether or not the training data is positive
        output: float denoting the layer loss's derivative
        """

        return torch.abs(activation_matrix).mul(2)


for data in dataset:
    print(data.x.shape)
    twos_tensor = torch.full((data.x.shape[0], 1), data.y[0])
    data.x = torch.cat([data.x, twos_tensor], dim=1)
    
    print(data.x)


device = torch.device('cpu')
model = FFGNN().to(device)

print(dataset[0])
for key, item in dataset[0]:
    print(f'{key} found in data')



data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()

model.eval()
