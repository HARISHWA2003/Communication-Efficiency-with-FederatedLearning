# Federated Learning with PyTorch

This repository contains an implementation of a Federated Learning system using PyTorch. The project is split into two main components: the **server** and the **client** scripts. This setup simulates a Federated Learning scenario where multiple clients train a model locally and share the updates with a central server.

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Setup Instructions](#setup-instructions)
- [Usage](#usage)
  - [Running the Server](#running-the-server)
  - [Running the Client](#running-the-client)
- [Model Details](#model-details)
- [Quantization](#quantization)
- [Notes](#notes)
- [Contributing](#contributing)

## Overview

Federated Learning is a decentralized approach to training machine learning models, allowing multiple clients to train on their local data and then send their updates to a central server for aggregation. This setup is particularly useful for privacy-sensitive applications, where raw data is not shared directly.

This project implements a simple Federated Learning setup using the **MNIST dataset** for training an improved Convolutional Neural Network (CNN). The project also includes an optional quantization feature to reduce communication costs when sending updates between the server and clients.

## Architecture

- **Server (`server.py`)**: Manages communication between clients, aggregates local model updates, and maintains the global model.
- **Client (`client.py`)**: Loads and preprocesses local datasets, trains the model locally, and sends the model updates to the server.

The server and client communicate via sockets. Each client connects to the server, trains the model on local data, and sends its model updates back to the server for aggregation.

## Getting Started

### Prerequisites

- Python 3.7+
- PyTorch
- torchvision
- tqdm

To install the dependencies, run:

```sh
pip install torch torchvision tqdm
```

### Setup Instructions

1. Clone this repository:
   ```sh
   git clone https://github.com/HARISHWA2003/federated-learning-pytorch.git
   cd federated-learning-pytorch
   ```

2. Install the required Python libraries as mentioned in the [Prerequisites](#prerequisites).

## Usage

### Running the Server

Start the server first to allow incoming client connections.

```sh
python server.py
```

- The server listens for connections from clients and aggregates their local updates.

### Running the Client

After the server is running, start a client to connect to the server and begin training.

```sh
python client.py
```

- The client trains a model locally on the MNIST dataset and sends model updates to the server.
- Multiple clients can be started to simulate a federated setup.
- The client can run with or without quantization. To enable quantization, use the `--quantization` flag:

  ```sh
  python client.py --quantization
  ```

  Without the flag, the client will run without quantization by default.

## Model Details

The **ImprovedCNN** model is defined in `client.py` and consists of the following layers:

- **Convolutional Layers**: Two convolutional layers with batch normalization and max-pooling.
- **Fully Connected Layers**: One fully connected layer with dropout for regularization, followed by the output layer.

The model is trained using the MNIST dataset with data augmentation techniques such as random horizontal flips and random rotations.

## Quantization

The client script includes an option for **8-bit quantization** of model updates to reduce communication costs between the server and clients. The quantization functions are used to scale and reduce the precision of model updates before sending them to the server, thus reducing the overall data size.

To run the client with quantization, use the `--quantization` flag as described in the [Usage](#usage) section.

## Notes

- This implementation is for educational purposes to demonstrate the basics of Federated Learning using PyTorch. It uses sockets for communication, which may not be suitable for a production environment.
- Ensure the server is started before connecting any clients to avoid connection issues.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue if you would like to improve the project.

Feel free to reach out for any questions or suggestions.

Happy learning!