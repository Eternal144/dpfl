#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
import random
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from collections import Counter

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    dict_users = {}
    num_items = int(len(dataset) / num_users)
    all_idxs = [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    dict_users = {}
    num_shards, num_imgs = num_users * 2, int(len(dataset) / (num_users * 2))
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    
    # data distribution analysis
    print("data distribution analysis")
    for i in range(num_users):
        print(f"Client {i} has {len(dict_users[i])} samples")
        print(f"Data distribution: {np.bincount(labels[dict_users[i]])}")
    
    return dict_users

def mnist_mixed_noniid(dataset, num_users, frac_iid=0.3, shards_per_client=2):
    """
    Sample the MNIST dataset in a mixed IID and Non-IID manner:
      - IID clients receive samples approximating the global distribution
      - Non-IID clients receive a limited number of shards (biased distribution)
    
    Args:
        dataset: MNIST/FashionMNIST dataset
        num_users: Total number of clients
        frac_iid: Fraction of clients with IID distribution (0~1)
        shards_per_client: Number of shards assigned to each non-IID client
        
    Returns:
        dict_users: Dictionary mapping {client_id: np.array of sample indices}
    """
    print("partitioning dataset into IID and Non-IID clients")
    # --- 1. Initialization & parameter calculation ---
    num_iid = int(num_users * frac_iid)
    num_noniid = num_users - num_iid
    num_items = len(dataset) // num_users                # Total samples per client
    num_shards = num_noniid * shards_per_client          # Total number of shards
    # shard_size = num_items // shards_per_client          # Size of each shard

    # Get all indices and labels
    all_idxs = np.arange(len(dataset))
    
    # Handle compatibility with different dataset versions
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'train_labels'):
        labels = dataset.train_labels.numpy() if not isinstance(dataset.train_labels, np.ndarray) else dataset.train_labels
    else:
        raise AttributeError("Dataset doesn't have 'targets' or 'train_labels' attribute")

    dict_users = {}

    # --- 2. Assign globally random samples to IID clients ---
    for i in range(num_iid):
        chosen = np.random.choice(all_idxs, num_items, replace=False)
        dict_users[i] = chosen
        all_idxs = np.setdiff1d(all_idxs, chosen, assume_unique=True)

    # --- 3. Non-IID: Sort remaining samples and split into shards ---
    # 3.1 Sort remaining indices by label
    if len(all_idxs) > 0:  # Check if there are samples left
        rem_labels = labels[all_idxs]
        sorted_idx = all_idxs[np.argsort(rem_labels)]
        
        # 3.2 Split into num_shards continuous chunks
        shards = np.array_split(sorted_idx, num_shards)

        # --- 4. Assign shards to Non-IID clients ---
        shard_ids = np.arange(num_shards)
        for j in range(num_noniid):
            client_id = num_iid + j
            chosen_shards = np.random.choice(shard_ids, shards_per_client, replace=False)
            # Merge multiple shards
            samples = np.concatenate([shards[s] for s in chosen_shards])
            dict_users[client_id] = samples
            # Remove assigned shards from available list
            shard_ids = np.setdiff1d(shard_ids, chosen_shards, assume_unique=True)
    else:
        # Handle edge case where all samples are allocated to IID clients
        for j in range(num_noniid):
            dict_users[num_iid + j] = np.array([], dtype=np.int64)
            
    return dict_users


def fashion_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    dict_users = {}
    num_items = int(len(dataset) / num_users)
    all_idxs = [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def fashion_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = num_users * 2, int(len(dataset) / (num_users * 2))
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users

def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    dict_users = {}
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def cifar_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = num_users * 2, int(len(dataset) / (num_users * 2))
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    # labels = dataset.train_labels.numpy()
    labels = np.array(dataset.targets)
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users

    return plt

def visualize_client_distributions(dataset, dict_users, selected_client_indices, num_classes=10):
    """
    Visualize the distribution of labels for selected clients
    
    Args:
        dataset: The dataset containing the labels
        dict_users: Dictionary mapping client IDs to their data indices
        selected_client_indices: Indices of clients to visualize
        num_classes: Number of classes in the dataset
    """
    # Get dataset labels
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'train_labels'):
        labels = dataset.train_labels.numpy() if not isinstance(dataset.train_labels, np.ndarray) else dataset.train_labels
    else:
        raise AttributeError("Dataset doesn't have 'targets' or 'train_labels' attribute")
    
    num_clients = len(selected_client_indices)
    plt.figure(figsize=(15, 10))
    
    # Plot global distribution first
    plt.subplot(num_clients + 1, 1, 1)
    global_counter = Counter(labels)
    global_dist = [global_counter.get(i, 0) for i in range(num_classes)]
    plt.bar(range(num_classes), global_dist)
    plt.title("Global Distribution", fontweight='bold')
    plt.ylim(0, max(global_dist) * 1.1)  # Set y-limit for consistent scale
    plt.xticks(range(num_classes))
    plt.ylabel("Count")
    
        # Plot distributions for selected clients
    for i, client_idx in enumerate(selected_client_indices):
        plt.subplot(num_clients + 1, 1, i + 2)
        
        # Get indices for this client
        indices = dict_users[client_idx]
        client_labels = labels[indices]
        
        # Count occurrences of each class
        counter = Counter(client_labels)
        client_dist = [counter.get(i, 0) for i in range(num_classes)]
        
        # Determine client type based on threshold in global partitioning
        num_iid_clients = int(num_users * frac_iid)
        client_type = "IID" if client_idx < num_iid_clients else "Non-IID"
        
        plt.bar(range(num_classes), client_dist)
        plt.title(f"Client {client_idx} ({client_type})", fontweight='bold')
        plt.ylabel("Count")
        
        if i == num_clients - 1:
            plt.xlabel("Digit Class")
        
        plt.xticks(range(num_classes))
        
        # Auto-scale the y-axis to make distribution differences visible
        # Calculate a reasonable maximum for the y-axis
        if max(client_dist) > 0:
            plt.ylim(0, max(client_dist) * 1.2)  # Dynamic scaling based on actual data
        else:
            # Fallback if no data
            samples_per_client = len(dataset) // num_users
            plt.ylim(0, samples_per_client * 0.5)
    
    plt.tight_layout()
    return plt

if __name__ == '__main__':
    # Set up the MNIST dataset
    trans_mnist = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)

    # Set a fixed random seed for reproducibility
    np.random.seed(42)

    # Parameters
    num_users = 100  # Total number of clients
    frac_iid = 0.3  # 30% clients have IID data
    shards_per_client = 2  # Each Non-IID client gets 2 shards

    # Generate client data distribution
    user_groups = mnist_mixed_noniid(dataset_train, num_users, frac_iid, shards_per_client)

    num_clients_to_show = 10  # Show 10 random clients

    selected_clients = np.random.choice(range(num_users), size=num_clients_to_show) # Shuffle to mix IID and Non-IID in display

    plt = visualize_client_distributions(dataset_train, user_groups, selected_client_indices=selected_clients, num_classes=10)

    plt.savefig('client_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Create a second visualization showing distribution statistics
    plt.figure(figsize=(12, 8))

    # trans_fashion_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    # dataset_train = datasets.FashionMNIST('../data/fashion-mnist', train=True, download=True,
    #                                       transform=trans_fashion_mnist)
    # num = 100
    # d = mnist_iid(dataset_train, num)
    # path = '../data/fashion_iid_100clients.dat'
    # file = open(path, 'w')
    # for idx in range(num):
    #     for i in d[idx]:
    #         file.write(str(i))
    #         file.write(',')
    #     file.write('\n')
    # file.close()
    # trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    # dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
    # print(fashion_iid(dataset_train, 1000)[0])


