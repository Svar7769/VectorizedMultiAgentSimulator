#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import torch

from vmas import render_interactively
from vmas.simulator.core import Agent, Landmark, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        n_agents = kwargs.get("n_agents", 4)
        self.share_reward = kwargs.get("share_reward", False)
        self.penalise_by_time = kwargs.get("penalise_by_time", False)
        self.food_radius = kwargs.get("food_radius", 0.05)
        self.pos_range = kwargs.get("pos_range", 1.0)
        n_food = kwargs.get("n_food", n_agents)

        # Make world
        world = World(
            batch_dim,
            device,
            x_semidim=self.pos_range,
            y_semidim=self.pos_range,
        )
        # Add agents
        for i in range(n_agents):
            # Constraint: all agents have same action range and multiplier
            agent = Agent(
                name=f"agent_{i}",
                collide=False,
                shape=Sphere(radius=0.035),
            )
            world.add_agent(agent)
        # Add landmarks
        for i in range(n_food):
            food = Landmark(
                name=f"food_{i}",
                collide=False,
                shape=Sphere(radius=self.food_radius),
                color=Color.GREEN,
            )
            world.add_landmark(food)

        return world

    def reset_world_at(self, env_index: int = None):
        for agent in self.world.agents:
            agent.set_pos(
                torch.zeros(
                    self.world.dim_p,
                    device=self.world.device,
                    dtype=torch.float32,
                ),
                batch_index=env_index,
            )
        for landmark in self.world.landmarks:
            landmark.set_pos(
                torch.zeros(
                    (
                        (1, self.world.dim_p)
                        if env_index is not None
                        else (self.world.batch_dim, self.world.dim_p)
                    ),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -self.pos_range,
                    self.pos_range,
                ),
                batch_index=env_index,
            )
            if env_index is None:
                landmark.eaten = torch.full(
                    (self.world.batch_dim,), False, device=self.world.device
                )
                landmark.just_eaten = torch.full(
                    (self.world.batch_dim,), False, device=self.world.device
                )
                landmark.reset_render()
            else:
                landmark.eaten[env_index] = False
                landmark.just_eaten[env_index] = False
                landmark.is_rendering[env_index] = True

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]
        is_last = agent == self.world.agents[-1]

        rews = torch.zeros(self.world.batch_dim, device=self.world.device)

        for landmark in self.world.landmarks:
            if is_first:
                landmark.how_many_on_food = torch.stack(
                    [
                        torch.linalg.vector_norm(
                            a.state.pos - landmark.state.pos, dim=1
                        )
                        < a.shape.radius + landmark.shape.radius
                        for a in self.world.agents
                    ],
                    dim=1,
                ).sum(-1)
                landmark.anyone_on_food = landmark.how_many_on_food > 0
                landmark.just_eaten[landmark.anyone_on_food] = True

            assert (landmark.how_many_on_food <= len(self.world.agents)).all()

            if self.share_reward:
                rews[landmark.just_eaten * ~landmark.eaten] += 1
            else:
                on_food = (
                    torch.linalg.vector_norm(
                        agent.state.pos - landmark.state.pos, dim=1
                    )
                    < agent.shape.radius + landmark.shape.radius
                )
                eating_rew = landmark.how_many_on_food.reciprocal().nan_to_num(
                    posinf=0, neginf=0
                )
                rews[on_food * ~landmark.eaten] += eating_rew[on_food * ~landmark.eaten]

            if is_last:
                landmark.eaten += landmark.just_eaten
                landmark.just_eaten[:] = False
                landmark.is_rendering[landmark.eaten] = False

        if self.penalise_by_time:
            rews[rews == 0] = -0.01
        return rews

    def observation(self, agent: Agent):
        obs = []
        for landmark in self.world.landmarks:
            obs.append(
                torch.cat(
                    [
                        landmark.state.pos - agent.state.pos,
                        landmark.eaten.to(torch.int).unsqueeze(-1),
                    ],
                    dim=-1,
                )
            )
        return torch.cat(
            obs,
            dim=-1,
        )

    def observation_from_pos(self, pos: torch.Tensor, env_index: int = None):
        """
        Get observation from a given position.
        
        Args:
            pos: Position tensor of shape [batch_size, 2] or [2]
            env_index: Index of the environment (optional)
            
        Returns:
            Observation tensor as if an agent were at the given position
        """
        # Ensure pos has correct shape
        if pos.dim() == 1:
            pos = pos.unsqueeze(0)  # [2] -> [1, 2]
        
        batch_size = pos.shape[0]
        
        obs = []
        for landmark in self.world.landmarks:
            if env_index is None:
                # Use all environments
                landmark_pos = landmark.state.pos  # [batch_dim, 2]
                landmark_eaten = landmark.eaten  # [batch_dim]
                
                # If pos batch doesn't match world batch_dim, expand landmark data
                if batch_size != landmark_pos.shape[0]:
                    # Assume we want first env if sizes don't match
                    landmark_pos = landmark_pos[0:1].expand(batch_size, -1)
                    landmark_eaten = landmark_eaten[0:1].expand(batch_size)
            else:
                # Use specific environment
                landmark_pos = landmark.state.pos[env_index].unsqueeze(0)  # [1, 2]
                landmark_eaten = landmark.eaten[env_index].unsqueeze(0)  # [1]
                
                # Expand to match batch size
                landmark_pos = landmark_pos.expand(batch_size, -1)  # [batch_size, 2]
                landmark_eaten = landmark_eaten.expand(batch_size)  # [batch_size]
            
            # Compute relative position from given pos
            relative_pos = landmark_pos - pos  # [batch_size, 2]
            
            obs.append(
                torch.cat(
                    [
                        relative_pos,
                        landmark_eaten.to(torch.int).unsqueeze(-1),
                    ],
                    dim=-1,
                )
            )
        
        return torch.cat(obs, dim=-1)

    def done(self):
        return torch.all(
            torch.stack(
                [landmark.eaten for landmark in self.world.landmarks],
                dim=1,
            ),
            dim=-1,
        )


if __name__ == "__main__":
    render_interactively(
        __file__,
        control_two_agents=True,
        n_agents=4,
        share_reward=False,
        penalise_by_tim=False,
    )
