from gym.envs.registration import register

ABS_PATH = "custom_envs.envs"

# =========================================================================== #
#                                   Cheetah                                   #
# =========================================================================== #

CHEETAH_LEN = 1000

register(
    id="HCWithPos-v0",
    entry_point=ABS_PATH+".half_cheetah:HalfCheetahWithPos",
    max_episode_steps=CHEETAH_LEN,
    reward_threshold=None,
    nondeterministic=False,
)

ANT_LEN = 500

register(
    id="AntWall-v0",
    entry_point=ABS_PATH+".ant:AntWall",
    max_episode_steps=ANT_LEN,
    reward_threshold=None,
    nondeterministic=False,
)

