import recsim
from recsim.environments import interest_evolution

env_config = {
    "slate_size": 2,
    "seed": 0,
    "num_candidates": 5,
    "resample_documents": True,
}

env = interest_evolution.create_environment(env_config)
obs = env.reset()

print("RecSim env created successfully.")
print("Observation type:", type(obs))
print("Observation:", obs)