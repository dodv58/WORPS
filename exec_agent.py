import torch
import tyro
from env_gym import make_env
import gymnasium as gym
from dataclasses import dataclass
from agent import GCN_Agent

@dataclass
class Args:
    run: str = "WORPS-v1__train_ppo__1__1722182581"
    """seed of the experiment"""
    model: str = "latest"
    """ latest / best_episode_return / best_increasing_steps / best_episode_improvement """

    dataset: str = "025"
    seed: int = 1111

if __name__ == "__main__":
    args = tyro.cli(Args)
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.dataset)],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = GCN_Agent(envs)
    checkpoint = torch.load(f"runs/{args.run}/{args.model}.pth")
    agent.load_state_dict(checkpoint)


    next_obs, info = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    print(f"initial network cost: {info}")

    terminated = False    
    while not terminated:
        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(next_obs)
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_obs = torch.Tensor(next_obs).to(device)
            terminated = terminations[0]
            

    print(f"final network cost: {envs.envs[0].get_network_cost()}")