import logging
import gym
import minerl
from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
from minerl.herobraine.env_specs.map_localize_specs import MapLocalize

import coloredlogs
coloredlogs.install(logging.DEBUG)

def test_turn(resolution):
    #env = HumanSurvival(resolution=resolution).make()
    #env = gym.make("MineRLBasaltBuildVillageHouse-v0")
    env = gym.make("MyMapLocalize-v0")
    #env = gym.make("MineRLBasaltFindCave-v0")
    env.reset()
    obs, _, _, info = env.step(env.action_space.noop())
    print(obs["location_stats"])
    N = 1000
    for i in range(N):
        ac = env.action_space.noop()
        ac['camera'] = [0.0, 2 * 360 / N]
        _, _, _, info = env.step(ac)
        env.render()
    env.close()

if __name__ == '__main__':
    test_turn((640, 360))



