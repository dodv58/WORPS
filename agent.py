import os
import random
import time
from dataclasses import dataclass
import torch
import torch
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import FlattenObservation
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from env_gym import Network, CatObservation
import torch.nn.functional as F

from common import *
import json

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )
    def set_device(self, device):
        pass

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


class GCN_Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.edge_adjacency = torch.from_numpy(envs.envs[0].edge_adjacency).float()
        self.in_channels = envs.envs[0].n_features
        self.extra_features = envs.envs[0].extra_features
        n_edges = (envs.single_observation_space.shape[0] - self.extra_features)//self.in_channels
        hidden_channels = 64
        # out_channels = 64
        out_channels = 32

        self.critic = nn.ModuleDict({
            "gcn1": GCNLayer1(self.in_channels, out_channels),
            "extra_lin": layer_init(nn.Linear(self.extra_features, 8)),
            "lin1": layer_init(nn.Linear(n_edges * out_channels + 8, 64)),
            "lin2": layer_init(nn.Linear(64, 1), std=1.0),
        })

        self.actor = nn.ModuleDict({
            "gcn1": GCNLayer1(self.in_channels, out_channels),
            "extra_lin": layer_init(nn.Linear(self.extra_features, 8)),
            "lin1": layer_init(nn.Linear(n_edges * out_channels + 8, 64)),
            "lin2": layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01)
        })

        self.tanh = nn.Tanh()
        self.flattern = nn.Flatten()

    def set_device(self, device):
        self.edge_adjacency = self.edge_adjacency.to(device)


    def get_value(self, x):
        batch_size = x.shape[0]
        x, extras = x.split([x.shape[-1] - 1, 1], dim=-1)
        x = x.reshape(batch_size, -1, self.in_channels)
        adj_matrix = self.edge_adjacency.expand(batch_size, x.shape[1], x.shape[1])

        x = self.tanh(self.critic["gcn1"](x, adj_matrix))
        x = self.flattern(x)
        extras = self.tanh(self.critic["extra_lin"](extras))
        x = torch.cat([x, extras], dim=-1)
        x = self.tanh(self.critic["lin1"](x))
        return self.critic["lin2"](x)


    def get_action_and_value(self, x, action=None):
        batch_size = x.shape[0]
        _x, extras = x.split([x.shape[-1] - 1, 1], dim=-1)
        _x = _x.reshape(batch_size, -1, self.in_channels)
        adj_matrix = self.edge_adjacency.expand(batch_size, _x.shape[1], _x.shape[1])

        _x = self.tanh(self.actor["gcn1"](_x, adj_matrix))
        _x = self.flattern(_x)
        extras = self.tanh(self.critic["extra_lin"](extras))
        _x = torch.cat([_x, extras], dim=-1)
        _x = self.tanh(self.actor["lin1"](_x))
        logits = self.actor["lin2"](_x)

        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        # return action, probs.log_prob(action), probs.entropy(), self.critic(x, adj_matrix)
        return action, probs.log_prob(action), probs.entropy(), self.get_value(x)

class MixedAgent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.edge_adjacency = torch.from_numpy(envs.envs[0].edge_adjacency).float()
        self.in_channels = envs.envs[0].n_features
        self.extra_features = envs.envs[0].extra_features
        n_edges = (envs.single_observation_space.shape[0] - self.extra_features)//self.in_channels
        hidden_channels = 64
        # out_channels = 64
        out_channels = 32

        self.critic = nn.ModuleDict({
            "gcn1": GCNLayer1(self.in_channels, out_channels),
            "lin1": layer_init(nn.Linear(out_channels*n_edges + envs.single_observation_space.shape[0], 128)),
            "lin2": layer_init(nn.Linear(128, 64)),
            "lin3": layer_init(nn.Linear(64, 1), std=1.0),
        })

        self.actor = nn.ModuleDict({
            "gcn1": GCNLayer1(self.in_channels, out_channels),
            "lin1": layer_init(nn.Linear(out_channels*n_edges + envs.single_observation_space.shape[0], 128)),
            "lin2": layer_init(nn.Linear(128, 64)),
            "lin3": layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01)
        })

        self.tanh = nn.Tanh()
        self.flattern = nn.Flatten()

    def set_device(self, device):
        self.edge_adjacency = self.edge_adjacency.to(device)

    def get_value(self, x):
        batch_size = x.shape[0]
        _x, _ = x.split([x.shape[-1] - 1, 1], dim=-1)
        _x = _x.reshape(batch_size, -1, self.in_channels)
        adj_matrix = self.edge_adjacency.expand(batch_size, _x.shape[1], _x.shape[1])

        _x = self.tanh(self.critic["gcn1"](_x, adj_matrix))
        _x = self.flattern(_x)
        _x = torch.cat([x, _x], dim=-1)
        _x = self.tanh(self.critic["lin1"](_x))
        _x = self.tanh(self.critic["lin2"](_x))
        return self.critic["lin3"](_x)


    def get_action_and_value(self, x, action=None):
        batch_size = x.shape[0]
        _x, extras = x.split([x.shape[-1] - 1, 1], dim=-1)
        _x = _x.reshape(batch_size, -1, self.in_channels)
        adj_matrix = self.edge_adjacency.expand(batch_size, _x.shape[1], _x.shape[1])

        _x = self.tanh(self.actor["gcn1"](_x, adj_matrix))
        _x = self.flattern(_x)
        _x = torch.cat([x, _x], dim=-1)
        _x = self.tanh(self.actor["lin1"](_x))
        _x = self.tanh(self.actor["lin2"](_x))
        logits = self.actor["lin3"](_x)

        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        # return action, probs.log_prob(action), probs.entropy(), self.critic(x, adj_matrix)
        return action, probs.log_prob(action), probs.entropy(), self.get_value(x)

class GCNLayer(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.projection = nn.Linear(c_in, c_out)

    def forward(self, node_feats, adj_matrix):
        """Forward.

        Args:
            node_feats: Tensor with node features of shape [batch_size, num_nodes, c_in]
            adj_matrix: Batch of adjacency matrices of the graph. If there is an edge from i to j,
                         adj_matrix[b,i,j]=1 else 0. Supports directed edges by non-symmetric matrices.
                         Assumes to already have added the identity connections.
                         Shape: [batch_size, num_nodes, num_nodes]
        """
        # Num neighbours = number of incoming edges
        num_neighbours = adj_matrix.sum(dim=-1, keepdims=True)
        node_feats = self.projection(node_feats)
        node_feats = torch.bmm(adj_matrix, node_feats)
        node_feats = node_feats / num_neighbours
        return node_feats


class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.empty(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm)

        # Step 6: Apply a final bias vector.
        out = out + self.bias

        return out

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j

class GCNLayer1(nn.Module):
    def __init__(self, c_in, c_out, c_hidden=64):
        super().__init__()
        self.encode = nn.Linear(c_in, c_hidden)
        self.message = nn.Linear(c_hidden, c_hidden)
        self.k = 4
        self.update_fn = nn.Sequential(nn.Linear(c_hidden*2, c_hidden), nn.Tanh())
        self.decode = nn.Linear(c_hidden, c_out)
        # self.update_fn = nn.Linear(c_hidden * 2, c_out)

    def forward(self, node_feats, adj_matrix):
        """Forward.

        Args:
            node_feats: Tensor with node features of shape [batch_size, num_nodes, c_in]
            adj_matrix: Batch of adjacency matrices of the graph. If there is an edge from i to j,
                         adj_matrix[b,i,j]=1 else 0. Supports directed edges by non-symmetric matrices.
                         Assumes to already have added the identity connections.
                         Shape: [batch_size, num_nodes, num_nodes]
        """
        encoded_feats = self.encode(node_feats) # (batch, n_nodes, c_hidden)

        for _ in range(self.k):
            messages = self.message(encoded_feats) # (batch, n_nodes, c_hidden)
            n_nodes = adj_matrix.shape[1]
            # add self-loops
            eye = torch.eye(n_nodes, dtype=bool).unsqueeze(dim=0).repeat(adj_matrix.shape[0], 1, 1).to(adj_matrix.device)
            adj_matrix = adj_matrix.masked_fill(eye, 1)

            adj_matrix_T = torch.transpose(adj_matrix, 1, 2)
            aggregated_feats = torch.bmm(adj_matrix_T, messages) # (batch, n_nodes, c_hidden) sum of encoded features of incoming neighbours

            num_incoming_neighbours = adj_matrix_T.sum(dim=-1, keepdims=True)
            aggregated_feats = aggregated_feats / num_incoming_neighbours # mean

            aggregated_feats = torch.concatenate([encoded_feats, aggregated_feats], dim=-1)
            encoded_feats = self.update_fn(aggregated_feats)

        return self.decode(encoded_feats)