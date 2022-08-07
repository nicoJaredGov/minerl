from typing import List
import minerl.herobraine
import minerl.herobraine.hero.handlers as handlers
from minerl.herobraine.hero.handler import Handler
from minerl.herobraine.env_specs.simple_embodiment import SimpleEmbodimentEnvSpec


class ChestClock(SimpleEmbodimentEnvSpec):

    def determine_success_from_rewards(self, rewards: list) -> bool:
        return 10 in rewards

    def __init__(self, version: int, stop_early=False, noisy=True):

        name = 'ChestClock{}{}{}-v0'.format('EarlyStop'if stop_early else '',
                                            'Noisy' if noisy else '',
                                            version)

        self.stop_early = stop_early
        self.noisy = noisy
        xml = 'chest_clock_{}.xml'.format(version)
        super().__init__(name, xml)

    def is_from_folder(self, folder: str) -> bool:
        return False

    def create_mission_handlers(self) -> List[Handler]:
        mission_handlers = [
            handlers.EpisodeLength(1000),
        ]
        return mission_handlers

    def create_observables(self) -> List[Handler]:
        return [
            handlers.POVObservation(self.resolution),
            handlers.FlatInventoryObservation(['diamond_pickaxe', 'clock', 'redstone', 'gold_block', 'gold_ingot'])
        ]

    def create_actionables(self) -> List[Handler]:
        return super().create_actionables() + [
            handlers.PlaceBlock(['none', 'dirt'],
                                _other='none', _default='none')]

    def create_rewardables(self) -> List[Handler]:
        return [
                   handlers.RewardForTouchingBlockType([
                       {'type': 'diamond_block', 'behaviour': 'onceOnly',
                        'reward': 100.0},
                   ])
               ] + ([handlers.RewardForDistanceTraveledToCompassTarget(
            reward_per_block=1.0
        )] if self.dense else [])

    def create_agent_start(self) -> List[Handler]:
        return [
            handlers.SimpleInventoryAgentStart([
                dict(type='compass', quantity='1')
            ])
        ]
    
    def create_agent_handlers(self) -> List[Handler]:
        return [
            handlers.AgentQuitFromTouchingBlockType(
                ["diamond_block"]
            )
        ]

    def create_server_world_generators(self) -> List[Handler]:
        return [
            handlers.DefaultWorldGenerator(
                force_reset=True
            )
        ]

    def create_server_quit_producers(self) -> List[Handler]:
        return [
            handlers.ServerQuitFromTimeUp(NAVIGATE_STEPS * MS_PER_STEP),
            handlers.ServerQuitWhenAnyAgentFinishes()
        ]

    def create_server_decorators(self) -> List[Handler]:
        return [handlers.NavigationDecorator(
            max_randomized_radius=64,
            min_randomized_radius=64,
            block='diamond_block',
            placement='surface',
            max_radius=8,
            min_radius=0,
            max_randomized_distance=8,
            min_randomized_distance=0,
            randomize_compass_location=True
        )]

    def create_server_initial_conditions(self) -> List[Handler]:
        return [
            handlers.TimeInitialCondition(
                allow_passage_of_time=False,
                start_time=6000
            ),
            handlers.WeatherInitialCondition('clear'),
            handlers.SpawningInitialCondition('false')
        ]

    def get_docstring(self):
        return """        
        In this task, the agent must collect materials in order to craft a clock, and then open the chest while holding
        the clock to complete the mission. 
        """