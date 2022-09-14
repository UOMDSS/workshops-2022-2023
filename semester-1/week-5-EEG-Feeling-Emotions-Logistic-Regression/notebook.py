import torch, pandas, numpy
from torch.utils.data import DataLoader, Dataset

# Load data
class EmotionDataset(Dataset):
    def __init__(self, df, transform=None, target_transform=None):
        self.data = df
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data.iloc[index, :-1].values.astype(numpy.float64)
        y = self.data.iloc[index, -1]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x, y

# Create model
class EmotionLogisticRegressionModel(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(EmotionLogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
    
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

# Environment variables        
FILENAME = "data/emotions.csv"
device = "cuda" if torch.cuda.is_available() else "cpu"
data = pandas.read_csv(FILENAME)
# Avoid underfitting because of too few data
ranges = [("fft_0_a", "fft_20_a"), ("fft_0_b", "fft_20_b"), ("label", "label")]
data = pandas.concat([data.loc[:, i:j] for i, j in ranges], axis=1)[data["label"] != "NEUTRAL"]
input_size = data.shape[1] - 1

output_size = 1
train_proportion = 0.8


# Hyperparameters
batch_size = 64
learning_rate = 0.01
num_epochs = 50

# Create dataset
dataset = EmotionDataset(data, transform=lambda x: torch.from_numpy(x).float(
), target_transform=lambda x: torch.tensor([0]).float() if x == "POSITIVE" else torch.tensor([1]).float())
train_size = int(train_proportion * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Create model
model = EmotionLogisticRegressionModel(input_size, output_size).to(device)

# Loss and optimizer
loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# print(next(iter(train_loader)))

# Train model
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# Test model
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X,y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)

            test_loss += loss_fn(pred, y).item()
            correct += (pred == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: Accuracy: {100*(correct):>0.1f}%, Avg loss: {test_loss:>8f} \n\n")


for t in range(num_epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_loader, model, loss_fn, optimizer)
    test(test_loader, model, loss_fn)
print("Done!")
