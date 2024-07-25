import json
from collections import defaultdict, namedtuple
from typing import Optional

import networkx as nx
import numpy as np
import torch
import tqdm
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule
from torch import nn
import networkx

from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec, MultiOneHotDiscreteTensorSpec, MultiDiscreteTensorSpec, OneHotDiscreteTensorSpec
from torchrl.envs import (
    CatTensors,
    EnvBase,
    Transform,
    TransformedEnv,
    UnsqueezeTransform,
)
from torchrl.envs.transforms.transforms import _apply_to_composite
from torchrl.envs.utils import check_env_specs, step_mdp
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
Session = namedtuple('Session', 'id s ds bw delta') # multicast

W_MAX = 100
W_MIN = 1
MAX_STEP = 100


def gen_params(net, traffic, batch_size=None) -> TensorDictBase:
    """Returns a ``tensordict`` containing the physical parameters such as gravitational force and torque or speed limits."""
    if batch_size is None:
        batch_size = []
    edges = list(net.edges)
    e_sources = torch.tensor([m for m, n in edges], dtype=torch.int64)
    e_dests = torch.tensor([n for m, n in edges], dtype=torch.int64)
    e_index_map = torch.zeros((len(net.nodes), len(net.nodes)), dtype=torch.int64)
    for i, (m,n) in enumerate(edges):
        e_index_map[m, n] = i

    traffic_matrix = torch.zeros((len(traffic), len(net.nodes), len(net.nodes)), dtype=torch.int64)
    for i, s in enumerate(traffic):
        traffic_matrix[i, s.s, s.ds] = True

    td = TensorDict(
        {
            "params": TensorDict(
                {
                    "n_nodes": len(net.nodes),
                    "n_edges": len(net.edges),
                    "e_sources": e_sources,
                    "e_dests": e_dests,
                    "e_cap": torch.tensor([net[m][n]["capacity"] for m, n in edges], dtype=torch.float32),
                    "e_index_map": e_index_map,
                    "n_traffic": len(traffic),
                    "traffic_bw": torch.tensor([s.bw for s in traffic], dtype=torch.float32),
                    "traffic_matrix": traffic_matrix,
                    "traffic_sources": torch.tensor([s.s for s in traffic], dtype=torch.int64),
                    "max_step": MAX_STEP
                },
                [],
            )
        },
        [],
    )
    if batch_size:
        td = td.expand(batch_size).contiguous()
    return td

def _make_spec(self, td_params):
    # Under the hood, this will populate self.output_spec["observation"]
    self.observation_spec = CompositeSpec(
        weights=BoundedTensorSpec(
            low=W_MIN,
            high=W_MAX,
            shape=(*td_params.shape, td_params["params", "n_edges"], 1),
            dtype=torch.long,
        ),
        loads=UnboundedContinuousTensorSpec(
            shape=(*td_params.shape, td_params["params", "n_edges"], 1),
            dtype=torch.float32,
        ),
        # we need to add the ``params`` to the observation specs, as we want
        # to pass it at each step during a rollout
        params=make_composite_from_td(td_params["params"]),
        step=BoundedTensorSpec(
            low=0,
            high=1e3,
            shape=(*td_params.shape, 1),
            dtype=torch.long,
        ),
        shape=(*td_params.shape,),
    )
    # since the environment is stateless, we expect the previous output as input.
    # For this, ``EnvBase`` expects some state_spec to be available
    self.state_spec = self.observation_spec.clone()
    # action-spec will be automatically wrapped in input_spec when
    # `self.action_spec = spec` will be called supported
    # self.action_spec = BoundedTensorSpec(
    #     low=0,
    #     high=td_params["params", "n_edges"],
    #     shape=(*td_params.shape, 1),
    #     dtype=torch.int64,
    # )
    self.action_spec = OneHotDiscreteTensorSpec(
        n=td_params["params", "n_edges"].item(),
        shape=(*td_params.shape, td_params["params", "n_edges"].item()),
        dtype=torch.int64
    )
    self.reward_spec = UnboundedContinuousTensorSpec(shape=(*td_params.shape, 1))


def make_composite_from_td(td):
    # custom function to convert a ``tensordict`` in a similar spec structure
    # of unbounded values.
    composite = CompositeSpec(
        {
            key: make_composite_from_td(tensor)
            if isinstance(tensor, TensorDictBase)
            else UnboundedContinuousTensorSpec(
                dtype=tensor.dtype, device=tensor.device, shape=tensor.shape
            )
            for key, tensor in td.items()
        },
        shape=td.shape,
    )
    return composite

def _reset(self, tensordict):
    if tensordict is None or tensordict.is_empty():
        # if no ``tensordict`` is passed, we generate a single set of hyperparameters
        # Otherwise, we assume that the input ``tensordict`` contains all the relevant
        # parameters to get started.
        tensordict = self.gen_params(net=self.net, traffic=self.traffic, batch_size=self.batch_size)

    # for non batch-locked environments, the input ``tensordict`` shape dictates the number
    # of simulators run simultaneously. In other contexts, the initial
    # random state's shape will depend upon the environment batch-size instead.
    params = tensordict["params"]
    # weights = torch.randint(low=W_MIN, high=W_MAX, size=(*tensordict.shape, *params["n_edges"], 1), generator=self.rng, device=self.device)
    weights = torch.randint(low=W_MIN, high=W_MAX, size=self.observation_spec['weights'].shape, generator=self.rng, device=self.device)
    loads = get_network_loads(params, weights)
    out = TensorDict(
        {
            "weights": weights,
            "params": params,
            "step": torch.zeros(*tensordict.shape, 1, dtype=torch.int64, device=tensordict.device),
            "loads": loads
        },
        batch_size=tensordict.shape,
    )
    return out

def _step(tensordict):
    params = tensordict["params"]
    new_weights = tensordict["weights"] + tensordict["action"].unsqueeze(-1)

    loads = get_network_loads(params, new_weights)
    costs = loads.max(dim=1).values
    reward = -costs.view(*tensordict.shape, 1)
    done = torch.zeros_like(reward, dtype=torch.bool) if tensordict["step"] < params["max_step"] else torch.ones_like(reward, dtype=torch.bool)
    out = TensorDict(
        {
            "weights": new_weights,
            "params": tensordict["params"],
            "reward": reward,
            "done": done,
            "step": tensordict["step"] + 1,
            "loads": loads
        },
        tensordict.shape,
    )
    return out

def get_network_loads(params, weights):
    weights = weights.squeeze(-1)
    if not params.shape:
        batch_size = 1
    else:
        batch_size = params.shape[0]
        for i in range(batch_size):
            params = params[i]

            n_edges = params["n_edges"].item()
            n_traffic = params["n_traffic"].item()
            e_sources = params["e_sources"]
            e_dests = params["e_dests"]
            n_nodes = params["n_nodes"].item()
            e_index_map = params["e_index_map"]
            traffic_matrix = params["traffic_matrix"]
            traffic_sources = params["traffic_sources"]
        shortest_path_trees = torch.zeros((*params.shape, n_edges, n_traffic), dtype=torch.float32)
            nodes = list(range(n_nodes))
    for i in range(batch_size):
        w = weights[i]
        shortest_paths = torch.zeros((n_nodes, n_nodes, n_edges), dtype=int)
        # shortest_path_trees = torch.zeros((n_edges, n_traffic), dtype=torch.float32)

        graph = csr_matrix((w, (e_sources, e_dests)), shape=(n_nodes, n_nodes))
        _, predecessors = dijkstra(csgraph=graph, directed=True, return_predecessors=True)

        for m in nodes:
            for n in nodes:
                if m == n:
                    continue
                u, v = n, n
                while v != m:
                    u = predecessors[m][v]
                    shortest_paths[m, n, e_index_map[u, v]] = 1
                    v = u

        for j in range(n_traffic):
            source = traffic_sources[j]
            dests = traffic_matrix[j, source].nonzero().squeeze()
            shortest_path_trees[i, :, j] += shortest_paths[source, dests, :].sum(axis=0)

    shortest_path_trees[shortest_path_trees > 1] = 1

    traffic_bw = torch.unsqueeze(params["traffic_bw"], 2)
    allocated_bw = torch.bmm(shortest_path_trees, traffic_bw).squeeze(dim=2)
    loads = allocated_bw / params["e_cap"]
    return loads.unsqueeze(-1)


class Network(EnvBase):
    batch_locked = False

    def __init__(self, net, traffic, seed=None, device="cpu"):
        self.net = net
        self.traffic = traffic
        td_params = self.gen_params(net, traffic)

        super().__init__(device=device, batch_size=[])
        self._make_spec(td_params)
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    # Helpers: _make_step and gen_params
    gen_params = staticmethod(gen_params)
    _make_spec = _make_spec

    # Mandatory methods: _step, _reset and _set_seed
    _reset = _reset
    _step = staticmethod(_step)

    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng


if __name__ == "__main__":
    dataset = "043"
    net = nx.read_gml(f"datasets/{dataset}/topo.gml")
    net = nx.convert_node_labels_to_integers(net)
    demand_data = json.load(open(f"datasets/{dataset}/traffic.txt"))
    demands = [[Session(*s) for s in traffic] for traffic in demand_data]

    env = Network(net=net, traffic=demands[0], batch_size=[1])
    check_env_specs(env)
    td = env.reset()

    td = env.rand_step(td)
    print(td)

    # action = env.
    # out = env.step(action)
    # print(out)