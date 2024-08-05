import torch
import tyro
from env_gym import make_env
import gymnasium as gym
from dataclasses import dataclass
from agent import GCN_Agent, Agent, MixedAgent

@dataclass
class Args:
    run: str = "WORPS-v1__train_ppo__1__1722182581"
    """seed of the experiment"""
    model: str = "latest"
    """ latest / best_episode_return / best_increasing_steps / best_episode_improvement """
    agent: str = "mlp"
    """mlp / gcn / mixed"""
    cuda: str = "0"

    dataset: str = "025"
    seed: int = 111
    env_max_step: int = 200
    early_stop: bool = True

if __name__ == "__main__":
    args = tyro.cli(Args)
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.dataset, args.env_max_step, args.early_stop)],
    )

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    if args.agent == "mlp":
        agent = Agent(envs).to(device)
    elif args.agent == "gcn":
        agent = GCN_Agent(envs).to(device)
    elif args.agent == "mixed":
        agent = MixedAgent(envs).to(device)
    else:
        raise ValueError(f"Unknown agent {args.agent}")
    checkpoint = torch.load(f"runs/{args.run}/{args.model}.pth", weights_only=True)
    agent.load_state_dict(checkpoint)
    agent.set_device(device)

    next_obs, infos = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)

    for _ in range(len(envs.envs[0].demands)):
        terminated = False
        print(">>>>>>>>>>>>")
        print(f"traffic index: {infos['traffic_index'][0]}")
        print(f"initial network cost: {infos['initial_network_cost']}")
        while not terminated:
            last_step_infos = infos
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                next_obs = torch.Tensor(next_obs).to(device)
                terminated = terminations[0]

            if terminated:
                print(f"final network cost: {last_step_infos['network_cost'][0]}, step_count {last_step_infos['step_count'][0]}")
                # print(f"next traffic index: {infos['traffic_index'][0]}")
            else:
                last_step_infos = infos