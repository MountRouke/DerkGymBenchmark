from gym_derk.envs import DerkEnv
import time
from argparse import ArgumentParser

def benchmark(no_observations=False, no_actions=False, format="csv", arenas=[1, 16, 128, 256, 512]):
  """Run benchmark

  Args:
    simulation_only: Skip sending actions and observations,
      to get a benchmark of just how the simulation is performing
  """
  delim = ', ' if format == 'csv' else ' | '

  first = True
  for n_arenas in arenas:
    env_start = time.time()
    env = DerkEnv(
      n_arenas=n_arenas,
      turbo_mode=True,
      debug_no_observations=no_observations
    )
    if first:
      first = False
      print('no_observations=' + str(no_observations) + ' ' + 'no_actions=' + str(no_actions) + ' ' + env.get_webgl_renderer())
      if format == 'csv':
        print('"n_arenas", "create env", "reset", "run"')
      else:
        print('n_arenas | create env | reset | run')
        print('--- | --- | --- | ---')
    print(str(n_arenas) + delim, end="")
    print(str(time.time() - env_start) + delim, end="")

    # action_space.sample() can take a lot of time so we just run it once outside the loop
    action_n = None if no_actions else [env.action_space.sample() for i in range(env.n_agents)]

    n_samples = 20
    reset_time = 0
    step_time = 0
    for i in range(n_samples):
      reset_start = time.time()
      observation_n = env.reset()
      reset_time = reset_time + time.time() - reset_start

      run_start = time.time()
      while True:
        observation_n, reward_n, done_n, info_n = env.step(action_n)
        if all(done_n):
          break
      step_time = step_time + time.time() - run_start
    print(str(reset_time / n_samples) + delim, end="")
    print(str(step_time / n_samples))
    env.close()

if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("-s", "--simulation_only", action="store_true", dest="simulation_only", default=False)
  parser.add_argument("-f", "--format", dest="format", default="markdown")
  args = parser.parse_args()
  benchmark(simulation_only=args.simulation_only, format=args.format)
