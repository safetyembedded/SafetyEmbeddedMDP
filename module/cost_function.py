from typing import List, Optional, Tuple
import theseus as th
import torch


class ObstacleCost(th.CostFunction):
    def __init__(
        self,
        cost_weight: th.CostWeight,
        agent_pos: th.Point2,
        obstacle_pos: th.Point2,
        max_distance: float,
        name: Optional[str] = None,
    ):
        super().__init__(cost_weight, name=name) 

        # add checks to ensure the input arguments are of the same class and dof:
        if not isinstance(agent_pos, obstacle_pos.__class__):
            raise ValueError(
                "agent_posiable for the VectorDifference inconsistent with the given obstacle_pos."
            )
        if not agent_pos.dof() == obstacle_pos.dof():
            raise ValueError(
                "agent_posiable and obstacle_pos in the VectorDifference must have identical dof."
            )

        self.agent_pos = agent_pos
        self.obstacle_pos = obstacle_pos
        self.max_distance = max_distance

        # register agent_posiable and obstacle_pos
        self.register_optim_vars(["agent_pos"])
        self.register_aux_vars(["obstacle_pos"])
    def dim(self) -> int:
        return self.agent_pos.dof()

    def _copy_impl(self, new_name: Optional[str] = None) -> "ObstacleCost":
        return ObstacleCost(  # type: ignore
            self.weight.copy(), self.agent_pos.copy(), self.obstacle_pos.copy(), self.max_distance, name=new_name
        )

    def error(self) -> torch.Tensor:
        error = torch.clamp((self.max_distance - (self.agent_pos - self.obstacle_pos).norm(dim=-1)), 0)
        # error = torch.pow(torch.clamp((self.max_distance - (self.agent_pos - self.obstacle_pos).norm(dim=-1)), 0),2)
        error = error.unsqueeze(1)
        return error
        # return (self.agent_pos - self.obstacle_pos).tensor

    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        mask = (self.agent_pos - self.obstacle_pos).norm(dim=-1) < self.max_distance
        mask = torch.tile(mask.unsqueeze(-1), (1,2)).float()
        jac_1 = (self.agent_pos[:,0] - self.obstacle_pos[:,0])/(self.agent_pos - self.obstacle_pos).norm(dim=-1)
        jac_2 = (self.agent_pos[:,1] - self.obstacle_pos[:,1])/(self.agent_pos - self.obstacle_pos).norm(dim=-1)
        jac = torch.stack((jac_1, jac_2), dim=1).to(self.agent_pos.device)
        # jac = jac * 2 * (self.max_distance - (self.agent_pos - self.obstacle_pos).norm(dim=-1)).unsqueeze(-1)
        jac = jac * mask
        jac = jac.unsqueeze(1)
        jac = -jac
        return [jac], self.error()

class ObstacleCostWithVelocity(th.CostFunction):
    def __init__(
        self,
        cost_weight: th.CostWeight,
        agent_pos: th.Point2,
        agent_v, 
        obstacle_pos: th.Point2,
        max_distance: float,
        name: Optional[str] = None,
    ):
        super().__init__(cost_weight, name=name) 

        # add checks to ensure the input arguments are of the same class and dof:
        if not isinstance(agent_pos, obstacle_pos.__class__):
            raise ValueError(
                "agent_posiable for the VectorDifference inconsistent with the given obstacle_pos."
            )
        if not agent_pos.dof() == obstacle_pos.dof():
            raise ValueError(
                "agent_posiable and obstacle_pos in the VectorDifference must have identical dof."
            )

        self.agent_pos = agent_pos
        self.obstacle_pos = obstacle_pos
        self.agent_v = agent_v
        self.max_distance = max_distance

        # register agent_posiable and obstacle_pos
        self.register_optim_vars(["agent_pos"])
        self.register_optim_vars(["agent_v"])
        self.register_aux_vars(["obstacle_pos"])
    def dim(self) -> int:
        return self.agent_pos.dof() + self.agent_v.dof()

    def _copy_impl(self, new_name: Optional[str] = None) -> "ObstacleCost":
        return ObstacleCostWithVelocity(  # type: ignore
            self.weight.copy(), self.agent_pos.copy(), self.agent_v.copy(), self.obstacle_pos.copy(), self.max_distance, name=new_name
        )

    def error(self) -> torch.Tensor:
        error = torch.clamp((self.max_distance - (self.agent_pos - self.obstacle_pos).norm(dim=-1)), 0) * self.agent_v.norm(dim=-1)
        # error = torch.pow(torch.clamp((self.max_distance - (self.agent_pos - self.obstacle_pos).norm(dim=-1)), 0),2)
        error = error.unsqueeze(1)
        return error
        # return (self.agent_pos - self.obstacle_pos).tensor

    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        self.agent_v.tensor += 1e-10
        mask = (self.agent_pos - self.obstacle_pos).norm(dim=-1) < self.max_distance
        mask = torch.tile(mask.unsqueeze(-1), (1,4)).float()
        jac_1 = (self.agent_pos[:,0] - self.obstacle_pos[:,0])/(self.agent_pos - self.obstacle_pos).norm(dim=-1)
        jac_2 = (self.agent_pos[:,1] - self.obstacle_pos[:,1])/(self.agent_pos - self.obstacle_pos).norm(dim=-1)
        jac_3 = (self.max_distance - (self.agent_pos - self.obstacle_pos).norm(dim=-1)) * self.agent_v[:, 0] / self.agent_v.norm(dim=-1)
        jac_4 = (self.max_distance - (self.agent_pos - self.obstacle_pos).norm(dim=-1)) * self.agent_v[:, 1] / self.agent_v.norm(dim=-1)
        jac = torch.stack((jac_1, jac_2, jac_3, jac_4), dim=1).to(self.agent_pos.device)
        # jac = jac * 2 * (self.max_distance - (self.agent_pos - self.obstacle_pos).norm(dim=-1)).unsqueeze(-1)
        jac = jac * mask
        jac = jac.unsqueeze(1)
        jac = -jac
        return [jac], self.error()
if __name__ == "__main__":
    agent_pos = th.Point2(name="agent", tensor=torch.zeros(2))
    obstacle_pos = th.Point2(name="obstacle", tensor=torch.ones(2)) 
    cost_function = ObstacleCost(th.ScaleCostWeight(1.0), agent_pos, obstacle_pos, max_distance=1, name='obstacle_cost')
    cost_function.jacobians()
    print(cost_function.error())

    cost_weight = th.ScaleCostWeight(1.0)

    # construct cost functions and add to objective
    objective = th.Objective()
    num_test_fns = 10
    for i in range(num_test_fns):
        a = th.Point2(torch.randn(2), name=f"a_{i}")
        b = th.Point2(torch.randn(2), name=f"b_{i}")
        cost_fn = ObstacleCost(cost_weight, a, b, max_distance=1)
        objective.add(cost_fn)
        
    # create data for adding to the objective
    theseus_inputs = {}
    for i in range(num_test_fns):
        # each pair of var/target has a difference of [1, 1]
        theseus_inputs.update({f"a_{i}": torch.ones((1,2)), f"b_{i}": torch.ones((1,2))})

    print(objective.error_squared_norm())
    objective.update(theseus_inputs)
    # sum of squares of errors [1, 1] for 10 cost fns: the result should be 20
    error_sq = objective.error_squared_norm()
    print(f"Sample error squared norm: {error_sq.item()}")