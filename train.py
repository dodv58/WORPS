from env import Network, Session
import networkx as nx
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn

from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm
import multiprocessing
from torchrl.envs.transforms import CatTensors, UnsqueezeTransform

is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)
num_cells = 256  # number of cells in each layer i.e. output dim.
lr = 3e-4
max_grad_norm = 1.0

frames_per_batch = 100
# For a complete training, bring the number of frames up to 1M
total_frames = 1_000

sub_batch_size = 64  # cardinality of the sub-samples gathered from the current data in the inner loop
num_epochs = 10  # optimisation steps per batch of data collected
clip_epsilon = (
    0.2  # clip value for PPO loss: see the equation in the intro for more context.
)
gamma = 0.99
lmbda = 0.95
entropy_eps = 1e-4




dataset = "043"
net = nx.read_gml(f"datasets/{dataset}/topo.gml")
net = nx.convert_node_labels_to_integers(net)
demand_data = json.load(open(f"datasets/{dataset}/traffic.txt"))
demands = [[Session(*s) for s in traffic] for traffic in demand_data]
batch_size = 5

env = Network(net=net, traffic=demands[0])
check_env_specs(env)
# rollout = env.rollout(3)

env = TransformedEnv(
    env,
    # UnsqueezeTransform(
    #     unsqueeze_dim=-1,
    #     in_keys=["weights", "loads"],
    #     in_keys_inv=["weights", "loads"],
    # ),
    Compose(
        CatTensors(
            in_keys=["weights", "loads"], dim=-1, out_key="observation", del_keys=False
        ),
        # ObservationNorm(in_keys="observation"),
        DoubleToFloat(),
    )
    # CatTensors(
    #         in_keys=["weights", "loads"], dim=-1, out_key="observation", del_keys=False
    #     )
)
# env.transform[1].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)
# check_env_specs(env)
# rollout = env.rollout(3)

torch.manual_seed(0)
env.set_seed(0)

net = nn.Sequential(
    nn.LazyLinear(64),
    nn.Tanh(),
    nn.LazyLinear(64),
    nn.Tanh(),
    nn.LazyLinear(64),
    nn.Tanh(),
    nn.LazyLinear(1),
)
policy = TensorDictModule(
    net,
    in_keys=["observation"],
    out_keys=["action"],
)
optim = torch.optim.Adam(policy.parameters(), lr=2e-3)


total_steps = 2_000
pbar = tqdm(range(total_steps // batch_size))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, total_steps)
logs = defaultdict(list)

for _ in pbar:
    init_td = env.reset()
    print(init_td.shape)
    rollout = env.rollout(100, policy, tensordict=init_td, auto_reset=False)
    traj_return = rollout["next", "reward"].mean()
    (-traj_return).backward()
    gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    optim.step()
    optim.zero_grad()
    pbar.set_description(
        f"reward: {traj_return: 4.4f}, "
        f"last reward: {rollout[..., -1]['next', 'reward'].mean(): 4.4f}, gradient norm: {gn: 4.4}"
    )
    logs["return"].append(traj_return.item())
    logs["last_reward"].append(rollout[..., -1]["next", "reward"].mean().item())
    scheduler.step()


def plot():
    import matplotlib
    from matplotlib import pyplot as plt

    is_ipython = "inline" in matplotlib.get_backend()
    if is_ipython:
        from IPython import display

    with plt.ion():
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(logs["return"])
        plt.title("returns")
        plt.xlabel("iteration")
        plt.subplot(1, 2, 2)
        plt.plot(logs["last_reward"])
        plt.title("last reward")
        plt.xlabel("iteration")
        if is_ipython:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        plt.show()


plot()