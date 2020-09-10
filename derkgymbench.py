from gym_derk.envs import DerkEnv
import time
from argparse import ArgumentParser

def benchmark(simulation_only=False, format="csv"):
  """Run benchmark

  Args:
    simulation_only: Skip sending actions and observations,
      to get a benchmark of just how the simulation is performing
  """
  delim = ', ' if format == 'csv' else ' | '

  for n_arenas in [1, 16, 128, 256, 512]:
    env_start = time.time()
    env = DerkEnv(
      n_arenas=n_arenas,
      turbo_mode=True,
      debug_no_observations=simulation_only
    )
    if n_arenas == 1:
      print('simulation_only=' + str(simulation_only) + ' ' + env.get_webgl_renderer())
      if format == 'csv':
        print('"n_arenas", "create env", "reset", "run"')
      else:
        print('n_arenas | create env | reset | run')
        print('--- | --- | --- | ---')
    print(str(n_arenas) + delim, end="")
    print(str(time.time() - env_start) + delim, end="")
    # Run reset and step once first to compile shaders. These
    # are not included in the benchmark, as this warmup only happens
    # once normally
    env.reset()
    env.step()

    reset_start = time.time()
    observation_n = env.reset()
    print(str(time.time() - reset_start) + delim, end="")

    # action_space.sample() can take a lot of time so we just run it once outside the loop
    action_n = None if simulation_only else [env.action_space.sample() for i in range(env.n_agents)]

    run_start = time.time()
    while True:
      observation_n, reward_n, done_n, info_n = env.step(action_n)
      if all(done_n):
        break
    print(str(time.time() - run_start))
    env.close()

if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("-s", "--simulation_only", dest="simulation_only", default=False)
  parser.add_argument("-f", "--format", dest="format", default="markdown")
  args = parser.parse_args()
  benchmark(simulation_only=args.simulation_only, format=args.format)