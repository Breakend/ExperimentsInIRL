import gym

def register_custom_envs():
    custom_envs = {"CustomHopperGravityHalf-v0" :
                    dict(path='envs.transfer.modified_hopper:ModifiedHopperEnv',
                         max_episode_steps=1000,
                         reward_threshold=3800.0,
                         kwargs= dict(xml_name='hopper_gravity_half.xml'))
                         }
    for key, value in custom_envs.items():
        gym.envs.register(
            id=key,
            entry_point=value["path"],
            max_episode_steps=value["max_episode_steps"],
            reward_threshold=value["reward_threshold"],
            kwargs=value["kwargs"]
        )
