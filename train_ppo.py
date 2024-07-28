# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
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

# import numpy as np
from common import *
import json
# from scipy.sparse import csr_matrix
# from scipy.sparse.csgraph import dijkstra

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "WORPS-v1"
    """the id of the environment"""
    total_timesteps: int = 4000_000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 5
    """the number of parallel game environments"""
    # num_steps: int = 128
    num_steps: int = 200
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

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
        out_channels = 64

        self.critic = nn.ModuleDict({
            "gcn1": GCNLayer1(self.in_channels, hidden_channels),
            "gcn2": GCNLayer1(hidden_channels, out_channels),
            "extra_lin": layer_init(nn.Linear(self.extra_features, 8)),
            "lin1": layer_init(nn.Linear(n_edges * out_channels + 8, 64)),
            "lin2": layer_init(nn.Linear(64, 1), std=1.0),
        })
        # self.critic = nn.ModuleDict({
        #     "gcn1": GCNConv(self.in_channels, hidden_channels),
        #     "gcn2": GCNConv(self.in_channels, hidden_channels),
        #     "extra_lin": layer_init(nn.Linear(self.extra_features, 8)),
        #     "lin1": layer_init(nn.Linear(n_edges * out_channels + 8, 64)),
        #     "lin2": layer_init(nn.Linear(64, 1), std=1.0),
        # })

        self.actor = nn.ModuleDict({
            "gcn1": GCNLayer1(self.in_channels, hidden_channels),
            "gcn2": GCNLayer1(hidden_channels, out_channels),
            "extra_lin": layer_init(nn.Linear(self.extra_features, 8)),
            "lin1": layer_init(nn.Linear(n_edges * out_channels + 8, 64)),
            "lin2": layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01)
        })
        # self.actor = nn.ModuleDict({
        #     "gcn1": GCNConv(self.in_channels, hidden_channels),
        #     "gcn2": GCNConv(self.in_channels, hidden_channels),
        #     "extra_lin": layer_init(nn.Linear(self.extra_features, 8)),
        #     "lin1": layer_init(nn.Linear(n_edges * out_channels + 8, 64)),
        #     "lin2": layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01)
        # })

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
        x = self.tanh(self.critic["gcn2"](x, adj_matrix))
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
        _x = self.tanh(self.actor["gcn2"](_x, adj_matrix))
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
    def __init__(self, c_in, c_out):
        super().__init__()
        self.c_hidden = 64
        self.encode = nn.Linear(c_in, self.c_hidden)
        self.message = nn.Linear(self.c_hidden, self.c_hidden)
        self.k = 4
        self.update_fn = nn.Sequential(nn.Linear(self.c_hidden*2, c_out), nn.Tanh())

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

        return encoded_feats

def make_env(dataset):
    def inner():
        net = nx.read_gml(f'datasets/{dataset}/topo.gml')
        net = nx.convert_node_labels_to_integers(net)

        with open(f'datasets/{dataset}/traffic.txt') as f:
            demands = json.load(f)
            demands = [[Session(*s) for s in traffic] for traffic in demands]
            # traffic = demands[0]

        env = Network(net=net, demands = demands)
        # env = CatObservation(env)
        # env = FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return inner


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    # device = torch.device("mps")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env("025") for i in range(args.num_envs)],
    )

    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # print(envs.observation_space)
    # print(envs.single_observation_space)

    obs = envs.reset()
    # agent = Agent(envs).to(device)
    agent = GCN_Agent(envs).to(device)
    agent.set_device(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()