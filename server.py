# Server script (server.py)
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import socket
import pickle
import struct
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

# Dequantization function
def dequantize_tensor(quantized, min_val, scale):
    return quantized * scale + min_val

# Server setup
def start_server():
    num_clients = 2  # Number of clients can be changed here
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('localhost', 12345))
    server_socket.listen(num_clients)
    server_socket.settimeout(30)
    print(f"[Server] Server is listening on port 12345 for {num_clients} clients")

    # Initialize model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    global_model = ImprovedCNN().to(device)

    # Accept connections from clients
    clients = []
    for _ in range(num_clients):
        client_socket, addr = server_socket.accept()
        print(f"[Server] Connected to client at {addr}")
        clients.append(client_socket)

    # Federated training
    num_epochs = 1000  # Run until reaching 85% accuracy
    accuracy_threshold = 99.0
    epoch = 0
    communication_cost = 0  # Only track local update communication cost

    while True:
        epoch += 1
        print(f"\n{'-' * 50}\n[Server] Starting Round {epoch}\n{'-' * 50}")
        print(f"\n[Server] Starting epoch {epoch}")

        # Send global model to clients
        serialized_model = pickle.dumps(global_model.state_dict())
        model_length = len(serialized_model)
        # communication_cost += model_length * len(clients)  # Exclude the cost of sending the global model

        for idx, client_socket in enumerate(clients):
            if client_socket is None:
                continue
            print(f"[Server] Sending global model to client {idx + 1}")
            client_socket.sendall(struct.pack('>I', model_length))  # Send length of data
            try:
                client_socket.sendall(serialized_model)  # Send actual model data
            except (socket.error, ConnectionAbortedError) as e:
                print(f"[Server] Error sending data to client {idx + 1}: {e}")
                client_socket.close()
                clients[idx] = None  # Mark the client as disconnected

            print(f"[Server] Global model sent to client {idx + 1}")

        # Receive local updates from clients
        local_updates = []
        for idx, client_socket in enumerate(clients):
            print(f"[Server] Waiting for local updates from client {idx + 1}")
            data_length_buffer = b""
            while len(data_length_buffer) < 4:
                packet = client_socket.recv(4 - len(data_length_buffer))
                if not packet:
                    raise ConnectionError("Failed to receive data length from client.")
                data_length_buffer += packet
            data_length = struct.unpack('>I', data_length_buffer)[0]  # Receive length of incoming data
            received_data = b""
            with tqdm(total=data_length, unit='B', unit_scale=True, desc=f"Receiving update from client {idx + 1}") as pbar:
                while len(received_data) < data_length:
                    packet = client_socket.recv(4096)
                    if not packet:
                        break
                    received_data += packet
                    pbar.update(len(packet))
            client_update = pickle.loads(received_data)

            # Dequantize if the update is quantized
            if isinstance(client_update[list(client_update.keys())[0]], tuple):
                dequantized_state_dict = {}
                for key, (quantized, min_val, scale) in client_update.items():
                    dequantized_state_dict[key] = dequantize_tensor(quantized, min_val, scale)
                local_updates.append(dequantized_state_dict)
            else:
                local_updates.append(client_update)

            # Update communication cost to reflect actual received data size (only for local updates)
            communication_cost += len(received_data)
            print(f"[Server] Received local updates from client {idx + 1}")

        # Average the local updates to update the global model
        print("[Server] Averaging local updates to update global model")
        global_state_dict = global_model.state_dict()
        for key in global_state_dict:
            global_state_dict[key] = torch.mean(torch.stack([local_updates[i][key].float() for i in range(num_clients)]), dim=0)
        global_model.load_state_dict(global_state_dict)
        print("[Server] Global model updated")

        # Evaluate the global model
        print("[Server] Evaluating global model")
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)

        correct = 0
        total = 0
        with torch.no_grad():
            for data in tqdm(testloader, desc="[Server] Evaluating global model", unit="batch"):
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = global_model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'\n[Server] Epoch {epoch} Summary:\n    Accuracy: {accuracy:.2f}%\n    Communication Cost (local updates only): {communication_cost} bytes')

        if accuracy >= accuracy_threshold:
            print(f"\n{'-' * 50}\n[Server] End of Round {epoch}\n{'-' * 50}")
            round_number += 1
            print(f"\n[Server] Reached target accuracy of {accuracy_threshold}% at epoch {epoch}")
            break

    print('[Server] Finished Training')
    for client_socket in clients:
        client_socket.close()
        print(f"[Server] Closed connection to client")

if __name__ == "__main__":
    start_server()