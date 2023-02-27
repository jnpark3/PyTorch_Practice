import torch
import torch.nn as nn
import torch.optim as optim

# Define the 1-layer graph neural network
class GraphNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GraphNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the loss function
criterion = nn.MSELoss()

# Define the hyperparameters
input_size = 10
hidden_size = 5
output_size = 1
learning_rate = 0.01
num_epochs = 1000

# Create an instance of the graph neural network
model = GraphNeuralNetwork(input_size, hidden_size, output_size)

# Create an optimizer object
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Generate some dummy data
X = torch.randn(100, input_size)
y = torch.randn(100, output_size)

# Train the neural network
for epoch in range(num_epochs):
    # Forward pass
    output = model(X)
    
    # Compute the loss
    loss = criterion(output, y)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Update the weights
    optimizer.step()
    
    # Print the loss every 100 epochs
    if (epoch+1) % 100 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))