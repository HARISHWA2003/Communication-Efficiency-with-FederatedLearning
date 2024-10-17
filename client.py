# Client script (client.py)
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import socket
import pickle
import struct
import argparse
from tqdm import tqdm

# Define the improved model


class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Quantization functions


def quantize_tensor(tensor, num_bits=8):
    scale = (tensor.max() - tensor.min()) / (2 ** num_bits - 1)
    quantized = torch.round((tensor - tensor.min()) / scale).to(torch.int8)
    return quantized, tensor.min(), scale


def dequantize_tensor(quantized, min_val, scale):
    return (quantized.float() * scale) + min_val

# Client setup


def start_client(quantization=False):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('localhost', 12345))
    client_socket.settimeout(30)
    print("[Client] Connected to server")

    # Prepare the dataset
    transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
         transforms.RandomRotation(10),
         transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))]
    )
    dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)
    device_data, _ = torch.utils.data.random_split(
        dataset, [len(dataset) // 2, len(dataset) // 2])
    device_loader = torch.utils.data.DataLoader(
        device_data, batch_size=4, shuffle=True)

    # Initialize model, criterion, and optimizer
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    local_model = ImprovedCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(local_model.parameters(), lr=0.01)

    while True:
        # Receive global model from server
        print("[Client] Waiting for global model from server")
        data_length_buffer = b""
        while len(data_length_buffer) < 4:
            try:
                packet = client_socket.recv(4 - len(data_length_buffer))
            except socket.timeout:
                raise ConnectionError(
                    "Timeout while waiting to receive data length from server.")
            if not packet:
                raise ConnectionError(
                    "Failed to receive data length from server. Possible disconnection.")
            data_length_buffer += packet
        data_length = struct.unpack('>I', data_length_buffer)[
            0]  # Receive length of incoming data
        received_data = b""
        with tqdm(total=data_length, unit='B', unit_scale=True, desc="Receiving global model from server") as pbar:
            while len(received_data) < data_length:
                packet = client_socket.recv(4096)
                if not packet:
                    break
                received_data += packet
                pbar.update(len(packet))

        global_state_dict = pickle.loads(received_data)
        print("[Client] Received global model from server")
        local_model.load_state_dict(global_state_dict)

        # Train local model
        print("[Client] Starting local training")
        local_model.train()
        for i, data in enumerate(tqdm(device_loader, desc="[Client] Local training", unit="batch")):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = local_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print("[Client] Finished local training")

        # Quantize local model if quantization is enabled
        if quantization:
            print("[Client] Quantization is enabled")
            quantized_state_dict = {}
            for key, value in local_model.state_dict().items():
                quantized_state_dict[key], min_val, scale = quantize_tensor(
                    value)
                quantized_state_dict[key] = (
                    quantized_state_dict[key], min_val, scale)
            print("[Client] Communication cost before dumping (quantized):", len(
                str(quantized_state_dict)), "bytes")
            serialized_data = pickle.dumps(quantized_state_dict)
        else:
            print("[Client] Communication cost before dumping (non-quantized):",
                  len(str(local_model.state_dict())), "bytes")
            serialized_data = pickle.dumps(local_model.state_dict())

        # Update communication cost to reflect actual sent data size
        communication_cost = len(serialized_data)
        print(
            f"[Client] Communication cost after dumping: {communication_cost} bytes")

        # Send local model updates to server
        print(
            f"[Client] Communication cost before sending: {len(serialized_data)} bytes")
        # Send length of data
        client_socket.sendall(struct.pack('>I', len(serialized_data)))
        client_socket.sendall(serialized_data)  # Send actual model data
        print("[Client] Sent local updates back to server, waiting for the next round")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--quantization', action='store_true',
                        help='Enable quantization')
    parser.add_argument('--no-quantization', action='store_false',
                        dest='quantization', help='Disable quantization')
    args = parser.parse_args()
    start_client(quantization=args.quantization)
