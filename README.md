# Pacman Capture the Flag agent

Our team for the Autonomous Systems (EMAI, UPF) Pacman Contest. The whole agent is in
[`my_team.py`](my_team.py): two `FlexAgent`s sharing one `TeamBrain`, an attacker and a
defender that adapt their role to the score, the time left and the opponents.

- **Opponent tracking**: a position belief per opponent from sonar, visibility, eaten
  food and side of the board.
- **Time-aware safe search**: cells are used only if we reach them before any dangerous
  ghost; dead-end corridors only if we can leave them in time.
- **Priority rules per role**: go home, capsules, scared ghosts and food for the
  attacker; chase, block, mark, raid and guard for the defender.
- **Stable decisions**: remembered ghosts, sticky targets and border marking, so that
  the agents do not oscillate.

See [STRATEGY.md](STRATEGY.md) for the full design, the parameters and local results.
Team details are in [TEAM.md](TEAM.md).

## Setting up the environment

1. Clone this repository and go into it.
2. Pull the contest framework: `git submodule update --init --remote`, or clone
   [pacman-contest](https://github.com/AI4EDUC/pacman-contest-cluster) into `pacman-contest/`.
3. Create and activate a virtual environment: `python3.8 -m venv venv`, then
   `source venv/bin/activate`.
4. Install the framework:
    - `cd pacman-contest/`
    - `pip install -r requirements.txt`
    - `pip install -e .`

## Running a game

From `pacman-contest/src/contest/`:

```bash
# against the baseline team, with graphics
python capture.py -r ../../../my_team.py -b baseline_team

# on a random layout, without graphics
python capture.py -r ../../../my_team.py -b baseline_team -l RANDOM42 -q

# record a game and replay it
python capture.py -r ../../../my_team.py -b baseline_team --record
python capture.py --replay=www/contest_default/replays/match_0.replay
```

Set `DEBUG = True` at the top of `my_team.py` to see exceptions instead of the random
fallback move.
