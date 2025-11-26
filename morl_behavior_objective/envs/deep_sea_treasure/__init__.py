from gymnasium.envs.registration import register

from mo_gymnasium.envs.deep_sea_treasure.deep_sea_treasure import (
    CONCAVE_MAP,
    MIRRORED_MAP,
)
from morl_behavior_objective.envs.deep_sea_treasure.deep_sea_treasure import LEFT_RIGHT_DST_MAP, SMOOTH_MAP, LOUVRES_DST_MAP


register(
    id="deep-sea-treasure-v0",
    entry_point="mo_gymnasium.envs.deep_sea_treasure.deep_sea_treasure:DeepSeaTreasure",
    max_episode_steps=100,
)

register(
    id="deep-sea-treasure-concave-v0",
    entry_point="mo_gymnasium.envs.deep_sea_treasure.deep_sea_treasure:DeepSeaTreasure",
    max_episode_steps=100,
    kwargs={"dst_map": CONCAVE_MAP},
)

register(
    id="deep-sea-treasure-mirrored-v0",
    entry_point="mo_gymnasium.envs.deep_sea_treasure.deep_sea_treasure:DeepSeaTreasure",
    max_episode_steps=100,
    kwargs={"dst_map": MIRRORED_MAP},
)

register(
    id="left-right-dst-v0",
    entry_point="morl_behavior_objective.envs.deep_sea_treasure.deep_sea_treasure:DeepSeaTreasure",
    max_episode_steps=100,
    kwargs={
        "dst_map": LEFT_RIGHT_DST_MAP,
    },
)

register(
    id="deep-sea-treasure-louvres-dst-v0",
    entry_point="morl_behavior_objective.envs.deep_sea_treasure.deep_sea_treasure:DeepSeaTreasure",
    max_episode_steps=100,
    kwargs={"dst_map": LOUVRES_DST_MAP},
)

register(
    id="deep-sea-treasure-smooth-v0",
    entry_point="morl_behavior_objective.envs.deep_sea_treasure.deep_sea_treasure:DeepSeaTreasure",
    max_episode_steps=100,
    kwargs={"dst_map": SMOOTH_MAP},
)
