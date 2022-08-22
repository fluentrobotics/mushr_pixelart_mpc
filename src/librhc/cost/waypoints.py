# Copyright (c) 2019, The Personal Robotics Lab, The MuSHR Team, The Contributors of MuSHR
# License: BSD 3-Clause. See LICENSE.md file in root directory.

import torch
import numpy as np
import yaml


class Waypoints:
    NPOS = 3  # x, y, theta

    def __init__(self, params, logger, dtype, map, world_rep):
        self.params = params
        self.logger = logger
        self.dtype = dtype
        self.map = map

        self.world_rep = world_rep

        self.viz_rollouts = self.params.get_bool("debug/flag/viz_rollouts", False)
        self.n_viz = self.params.get_int("debug/viz_rollouts/n", -1)
        self.print_stats = self.params.get_bool("debug/viz_rollouts/print_stats", False)
        self.time_now = None
        self.des_speed = 0.4
        self.agents = None
        self.agent_paths = None
        self.own_index = 0
        self.target_index = 0
        self.wheel_base = self.params.get_float("model/wheel_base", default=0.29)
        self.reset()

    def reset(self):
        self.T = self.params.get_int("T", default=15)
        self.K = self.params.get_int("K", default=62)
        self.dist_w = self.params.get_float("cost_fn/dist_w", default=1.0)
        self.obs_dist_w = self.params.get_float("cost_fn/obs_dist_w", default=5.0)
        self.smoothing_discount_rate = self.params.get_float(
            "cost_fn/smoothing_discount_rate", default=0.04
        )
        self.bounds_cost = self.params.get_float("cost_fn/bounds_cost", default=100.0)

        self.obs_dist_cooloff = torch.arange(1, self.T + 1).mul_(2).type(self.dtype)
        self.horizon_dist = self.params.get_float("horizon/distance", default = 1.2)
        self.discount = self.dtype(self.T - 1)

        self.discount[:] = 1 + self.smoothing_discount_rate
        self.discount.pow_(torch.arange(0, self.T - 1).type(self.dtype) * -1)
        self.des_speed = self.params.get_float("trajgen/desired_speed", default=0.4)
        self.world_rep.reset()

    def set_multi_agent_stuff(self, agents, agent_paths, own_index):
        """
        get current pose and twist of all agents
        """
        # interpolate using cubic bezier curve looool.
        self.agents = agents
        self.agent_paths = agent_paths
        self.own_index = own_index

    def get_bezier(self, state_i, state_f, t):
        P0 = state_i[:2]
        P3 = state_f[:2]
        L = np.linalg.norm(P3 - P0)
        P1 = P0 + 0.33 * L * np.array([np.cos(state_i[2]), np.sin(state_i[2])])
        P2 = P3 - 0.33 * L * np.array([np.cos(state_f[2]), np.sin(state_f[2])])
        T = np.array([(1-t)**3,3*t*(1-t)**2,3*t*t*(1-t),t**3])
        B = np.zeros(2)
        B[0] = T[0]*P0[0] + T[1]*P1[0] + T[2]*P2[0] + T[3]*P3[0]
        B[1] = T[0]*P0[1] + T[1]*P1[1] + T[2]*P2[1] + T[3]*P3[1]
        return B


    def get_expected_location(self, T, agent_path):
        N = len(agent_path)
        index = 0
        time_scale = 1
        for i in range(N):
            if(T < agent_path[i, 3]):
                time_scale = agent_path[i, 3] - agent_path[i - 1, 3]
                index = i - 1
                break
        t = (T - agent_path[index,3])/time_scale  # time within the segment
        if(time_scale == 0):
            t = 0  # set to initial point
        return self.get_bezier(agent_path[index,:3], agent_path[index+1,:3], t)

    def get_winding_number(self, T_,input_loc = None):
        """
        get winding number expected based on the time entered. This uses the agent paths and interpolates them
        to find the expected location at any point in time. Based on these expected locations and the initial locations,
        the winding numbers between all agents can be found in the ref. frame of the ego agent.
        """
        N = len(self.agents)
        if(input_loc == None):
            T = T_
            loc = []
            for agent_path in self.agent_paths:
                loc.append(self.get_expected_location(T,agent_path))
            loc = np.array(loc)
            winding = np.zeros(N)
        else:
            winding = np.zeros((self.T, N))

        for i in range(N):
            if(i != self.own_index):
                if(input_loc == None):
                    delta = loc[i] - loc[self.own_index]
                    winding[i] = np.arctan2(delta[1], delta[0])  # winding number
                else:
                    delta = input_loc[i] - input_loc[self.own_index]  # when winding is based on measurement
                    winding[:, i] = np.arctan2(delta[:, 1], delta[:, 0])  # winding number
        return winding

    def propagate_forward(self):
        N = len(self.agents)
        poses = np.zeros((N, self.T, 2))
        i = 0
        for agent in self.agents:
            t = np.arange(0, 1.5, (1.5/self.T))
            # K = m.tan(agents[4])/self.wheel_base
            # angle = agents[2] + agents[3]*K*t
            vec = agent[3]*np.array([np.cos(agent[2]), np.sin(agent[2])])
            x = agent[0] + vec[0]*t
            y = agent[1] + vec[1]*t
            poses[i][:, 0] = x
            poses[i][:, 1] = y
            i += 1
        return poses

    def winding_cost(self, torch_poses):
        poses = np.array(torch_poses[:,:,:2])
        all_poses = self.propagate_forward()
        winding_cost = np.zeros(self.K)
        t = np.arange(0, 1.5, (1.5/self.T)) + self.time_now  # this needs to be parameterized
        expected_winding = np.zeros((self.T, 2))  # TODO don't hardcode number of agents
        for i in range(self.T):
            expected_winding[i] = self.get_winding_number(t[i], None)

        i = 0
        for pose in poses:
            all_poses[self.own_index] = pose  # set own pose
            winding_cost[i] = np.sum(self.get_winding_number(0, all_poses) - expected_winding)
            i += 1
            # for i in range(self.T):
            #     cost += self.get_winding_number(t[i], all_poses[:,i,:]) - expected_winding
        #     winding_cost.append(cost)
        winding_cost = np.array(winding_cost)
        return torch.from_numpy(np.abs(winding_cost))

    def agent_collisions(self, torch_poses):
        N = len(self.agents)
        own_poses = np.array(torch_poses[:,:,:2])
        all_poses = self.propagate_forward()
        collisions = np.zeros(self.K)
        thres = 0.5 # TODO parameterize?

        # # print(self.agents, self.own_index, self.agent_paths, self.T)
        # agent_path = self.agent_paths[self.own_index]
        # agent = self.agents[self.own_index]
        # min_dist = 2
        # closest_index = 0
        # for i in range(len(agent_path)):
        #     pos = agent_path[i, :2]
        #     if( np.linalg.norm(pos - agent[:2]) < min_dist ):
        #         min_dist = np.linalg.norm(pos - agent[:2])
        #         closest_index = i
        # now_time = agent_path[closest_index,3]
        # t = np.arange(0, 2.0, (2.0/float(self.T))) + now_time
        # expected_winding = np.zeros((self.T, len(self.agents)))  # TODO don't hardcode number of agents
        # for i in range(self.T):
        #     expected_winding[i] = self.get_winding_number(t[i], None) - agent[2]

        for k in range(len(own_poses)):
            for t in range(len(own_poses[k])):
                for j in range(N):
                    dist = np.linalg.norm(own_poses[k][t] - all_poses[j][t])
                    if j != self.own_index and dist < 1.5*thres:
                        # theta = expected_winding[j,0]
                        # dtheta = np.mean(np.diff(expected_winding[j]))
                        # multiplier = 1
                        # if(theta * dtheta < 0):
                        #     multiplier = 10
                        # else:
                        #     multiplier = 0.5
                        collisions[k] += (0.5 + ((1.5*thres - dist)/thres))  # normalize by threshold.

        return torch.from_numpy(collisions)

    def apply(self, poses, goal, path, car_pose):
        """
        Args:
        poses [(K, T, 3) tensor] -- Rollout of T positions
        goal  [(3,) tensor]: Goal position in "world" mode
        path nav_msgs.Path: Current path to the goal with orientation and positions
        car_pose geometry_msgs.PoseStamped: Current car position

        Returns:
        [(K,) tensor] costs for each K paths
        """
        assert poses.size() == (self.K, self.T, self.NPOS)
        assert goal.size() == (self.NPOS,)

        all_poses = poses.view(self.K * self.T, self.NPOS)

        # get all collisions (K, T, tensor)
        collisions = self.world_rep.check_collision_in_map(all_poses).view(
            self.K, self.T
        )
        collision_cost = collisions.sum(dim=1).mul(self.bounds_cost)

        # calculate lookahead
        distance_lookahead = self.horizon_dist

        # calculate closest index to car position
        ulim = min(self.target_index + 15, len(path))
        llim = max(self.target_index, 0)
        diff = np.sqrt(
            ((path[llim:ulim, 0] - car_pose[0]) ** 2) + ((path[llim:ulim, 1] - car_pose[1]) ** 2)
        )

        index = np.argmin(diff)
        self.target_index = index + llim
        # iterate to closest lookahead to distance
        while index < len(diff) - 1 and diff[index] < distance_lookahead:
            index += 1

        if abs(diff[index - 1] - distance_lookahead) < abs(
            diff[index] - distance_lookahead
        ):
            index -= 1

        x_ref, y_ref, theta_ref, time_ref = path[index + llim]

        time_array = torch.from_numpy(np.arange(time_ref + 1.5, time_ref , -(1.5/self.T)))
        d_ref = self.des_speed*(time_array - self.time_now)

        cross_track_error = np.abs(
            -(poses[:, :, 0] - x_ref) * np.sin(theta_ref)
            + (poses[:, :, 1] - y_ref) * np.cos(theta_ref)
        )
        # take the sum of error along the trajs
        cross_track_error = torch.sum(cross_track_error, dim=1)

        along_track_error = np.abs(
            (poses[:, :, 0] - x_ref) * np.cos(theta_ref)
            + (poses[:, :, 1] - y_ref) * np.sin(theta_ref)
        )

        time_error = np.zeros_like(along_track_error)
        for i in range(self.K):
            time_error[i,:] = (d_ref - along_track_error[i,:])**2
        time_error = np.abs(time_error)
        time_error = torch.from_numpy(time_error)

        # take the sum of error along the trajs
        along_track_error = torch.sum(along_track_error, dim=1)
        time_error = torch.sum(time_error, dim=1)
        # take the heading error from last pos in trajs
        heading_error = np.abs((poses[:, -1, 2] - theta_ref))

        # winding_cost = self.winding_cost(poses)

        agent_collision_cost = self.agent_collisions(poses)

        cross_track_error *= self.params.get_int("/car1/rhcontroller/control/cte_weight", default=100)  
        along_track_error *= self.params.get_int("/car1/rhcontroller/control/ate_weight", default=100)
        heading_error *= self.params.get_int("/car1/rhcontroller/control/he_weight", default= 10)
        time_error *= self.params.get_int("/car1/rhcontroller/control/time_weight", default = 10)
        # winding_cost *= self.params.get_int("/car1/rhcontroller/control/winding_weight", default= 10)
        # print(self.params.get_int("/car1/rhcontroller/control/winding_weight", default= 10))
        agent_collision_cost *= self.params.get_int("/car1/rhcontroller/control/collision_weight", 1000)

        result = cross_track_error.add(along_track_error).add(heading_error).add(time_error).add(agent_collision_cost)

        colliding = collision_cost.nonzero()
        result[colliding] = 1000000000

        return result

    def set_goal(self, goal):
        self.goal = goal
        return True
