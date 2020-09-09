from gym_derk.envs import DerkEnv
import time

print('"n_arenas", "create env", "reset", "run"')

for n_arenas in [1, 16, 128, 256, 512]:
  print(str(n_arenas) + ', ', end="")
  env_start = time.time()
  env = DerkEnv(
    n_arenas=n_arenas,
    turbo_mode=True,
  )
  print(str(time.time() - env_start) + ', ', end="")

  reset_start = time.time()
  observation_n = env.reset()
  print(str(time.time() - reset_start) + ', ', end="")

  # action_space.sample() can take a lot of time so we just run it once outside the loop
  action_n = [env.action_space.sample() for i in range(env.n_agents)]

  run_start = time.time()
  while True:
    observation_n, reward_n, done_n, info_n = env.step(action_n)
    if all(done_n):
      break
  print(str(time.time() - run_start))
  env.close()
