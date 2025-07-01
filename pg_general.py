import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# --- 1. Initialize W&B Run ---
# It's good practice to initialize wandb at the start of your script.
# You can pass a dictionary of hyperparameters to 'config' to log them automatically.
# Use 'project' to organize your runs, and 'name' for a specific run identifier.
wandb.init(
    project="my-ml-project",
    name="experiment-v1-lr0.001-epochs10",
    config={
        "learning_rate": 0.001,
        "epochs": 10,
        "batch_size": 32,
        "model_type": "SimpleNN",
        # ... other hyperparameters
    }
)

# Access the config from wandb (useful if you're using config files)
config = wandb.config

# --- 2. Define your Model, Data, Loss, Optimizer ---


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


# Dummy data
X_train = torch.randn(100, 10)
y_train = torch.randn(100, 1)
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

# --- 3. Training Loop ---
best_val_loss = float('inf')  # To track the best validation loss

for epoch in range(config.epochs):
    model.train()  # Set model to training mode
    train_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # --- 4. Validation (Optional, but highly recommended) ---
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():  # No need to calculate gradients during validation
        # Create dummy validation data
        X_val = torch.randn(20, 10)
        y_val = torch.randn(20, 1)
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

        for data, target in val_loader:
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    # --- 5. Log Metrics to W&B Each Epoch ---
    # Call wandb.log() inside your epoch loop.
    # The 'step' argument ensures that all metrics for a given epoch are grouped
    # under that specific step (which will be your epoch number on the X-axis).
    wandb.log({
        "train/loss": avg_train_loss,   # Use forward slashes for grouping in W&B UI
        "val/loss": avg_val_loss,
        "epoch": epoch                  # Explicitly log epoch as a metric for X-axis
    }, step=epoch)

    print(f"Epoch [{epoch+1}/{config.epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # --- 6. Log Best Model (Optional) ---
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        wandb.log({"best_val_loss": best_val_loss}, step=epoch)  # Log to summary or specific step
        # You can also save the model as a W&B artifact here
        # torch.save(model.state_dict(), "best_model.pth")
        # wandb.save("best_model.pth") # Saves file to W&B run directory and uploads it

# --- 7. End W&B Run (Optional, but good practice) ---
wandb.finish()
