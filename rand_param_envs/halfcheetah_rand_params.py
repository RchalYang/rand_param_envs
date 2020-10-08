import numpy as np
from rand_param_envs.base import RandomEnv, default_rand_params
from rand_param_envs.gym import utils

class HalfCheetahRandParamsEnv(RandomEnv, utils.EzPickle):
    def __init__(self, rand_params=default_rand_params, log_scale_limit=3.0):
        RandomEnv.__init__(self, log_scale_limit, 'half_cheetah.xml', 5, rand_params=rand_params)
        utils.EzPickle.__init__(self)

    def _step(self, action):
        xposbefore = self.model.data.qpos[0, 0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.model.data.qpos[0, 0]
        ob = self._get_obs()
        reward_ctrl = - 0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore)/self.dt
        reward = reward_ctrl + reward_run
        done = False
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[1:],
            self.model.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
    # def viewer_setup(self):
    #     self.viewer.cam.trackbodyid = 2
    #     self.viewer.cam.distance = self.model.stat.extent * 0.75
    #     self.viewer.cam.lookat[2] += .8
    #     self.viewer.cam.elevation = -20

if __name__ == "__main__":

    env = HalfCheetahRandParamsEnv()
    tasks = env.sample_tasks(40)
    while True:
        env.reset()
        env.set_task(np.random.choice(tasks))
        print(env.model.body_mass)
        task = env.get_task()
        print(env.get_task())
        for item in task:
            print(item, task[item].flatten().shape)
            print(np.prod(task[item].shape))
        for _ in range(100):
            # env.render()
            env.step(env.action_space.sample())  # take a random action
