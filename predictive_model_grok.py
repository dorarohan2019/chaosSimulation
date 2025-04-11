import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class LSTMAutoencoder(nn.Module):
    def __init__(self, timesteps=5, features=25):
        """Initialize the LSTM Autoencoder model.

        Args:
            timesteps (int): Number of timesteps in each sequence.
            features (int): Number of features in the input data.
        """
        super(LSTMAutoencoder, self).__init__()
        self.timesteps = timesteps
        self.features = features
        self.mean = 0.0  # Default value
        self.std = 1.0   # Default value
        # Encoder: LSTM to compress input
        self.encoder = nn.LSTM(input_size=features, hidden_size=64, batch_first=True)
        # Decoder: LSTM to reconstruct from encoded representation
        self.decoder = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)
        # Fully connected layer to map to original features
        self.fc = nn.Linear(64, features)

    def forward(self, x):
        """Forward pass through the autoencoder.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, timesteps, features).

        Returns:
            torch.Tensor: Reconstructed output with shape (batch_size, timesteps, features).
        """
        _, (h_n, _) = self.encoder(x)  # h_n: (1, batch_size, 64)
        encoded = h_n[-1]  # (batch_size, 64)
        input_to_decoder = encoded.unsqueeze(1).repeat(1, self.timesteps, 1)  # (batch_size, timesteps, 64)
        output, _ = self.decoder(input_to_decoder)  # (batch_size, timesteps, 64)
        output = self.fc(output)  # (batch_size, timesteps, features)
        return output

    def prepare_data(self, data, step=1):
        """Prepare overlapping sequences for training.

        Args:
            data (np.ndarray): Input data with shape (samples, features).
            step (int): Step size for overlapping sequences.

        Returns:
            torch.Tensor: Prepared sequences with shape (num_samples, timesteps, features).
        """
        num_samples = (len(data) - self.timesteps) // step + 1
        sequences = np.array([data[i * step:i * step + self.timesteps] for i in range(num_samples)])
        return torch.tensor(sequences, dtype=torch.float32)

    def train_model(self, data, epochs=100, step=1, learning_rate=0.005, batch_size=32):
        """Train the model with standardized data.

        Args:
            data (np.ndarray): Input data with shape (samples, features).
            epochs (int): Number of training epochs.
            step (int): Step size for overlapping sequences.
            learning_rate (float): Learning rate for the optimizer.
            batch_size (int): Batch size for training.
        """
        # Compute normalization parameters
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        self.std[self.std == 0] = 1.0  # Prevent division by zero
        data_std = (data - self.mean) / self.std
        sequences = self.prepare_data(data_std, step=step)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        for epoch in range(epochs):
            self.train()
            total_loss = 0
            for i in range(0, len(sequences), batch_size):
                batch = sequences[i:i + batch_size]
                optimizer.zero_grad()
                output = self(batch)
                loss = criterion(output, batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / (len(sequences) / batch_size)}")
        self.save()

    def get_anomaly_score(self, state):
        """Calculate the anomaly score for a given state.

        Args:
            state (np.ndarray): Input state with shape (timesteps, features) or (1, timesteps, features).

        Returns:
            float: Reconstruction error as the anomaly score.
        """
        #print("Input state shape:", state.shape)
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)
        #print("State after tensor conversion:", state.shape)
        if len(state.shape) == 2:
            state = state.unsqueeze(0)
        elif len(state.shape) != 3:
            raise ValueError(f"Expected 2D or 3D input, got {state.shape}")
        #print("State after unsqueeze:", state.shape)
        state_std = (state - torch.tensor(self.mean) / torch.tensor(self.std))
        #print("state_std shape:", state_std.shape)
        with torch.no_grad():
            output = self(state_std)
            reconstruction_error = torch.mean((output - state_std) ** 2).item()
        return reconstruction_error

    def save(self, model_path='lstm_autoencoder.pth', mean_path='mean.npy', std_path='std.npy'):
        """Save the model and normalization parameters."""
        torch.save(self.state_dict(), model_path)
        np.save(mean_path, self.mean)
        np.save(std_path, self.std)

    def load(self, model_path='lstm_autoencoder.pth', mean_path='mean.npy', std_path='std.npy'):
        """Load the model and normalization parameters."""
        self.load_state_dict(torch.load(model_path))
        self.mean = np.load(mean_path)
        self.std = np.load(std_path)

if __name__ == "__main__":
    states = np.load("environment_states_grok.npy")
    model = LSTMAutoencoder(timesteps=5, features=25)
    model.train_model(states, epochs=100, step=1, learning_rate=0.005, batch_size=32)