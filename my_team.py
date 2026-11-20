# my_team.py
# ---------------
# licensing information: please do not distribute or publish solutions to this
# project. you are free to use and extend these projects for educational
# purposes. the pacman ai projects were developed at uc berkeley, primarily by
# john denero (denero@cs.berkeley.edu) and dan klein (klein@cs.berkeley.edu).
# for more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html
#
# team overview:
# - a shared team brain keeps maze data, dead-end analysis and a position belief
#   for every opponent (sonar readings, eaten food, side constraints).
# - the attacker plans with a time-aware safe search: a cell is only used if no
#   dangerous ghost can reach it before us, and dead-end corridors are only
#   entered when we can leave them before a ghost blocks the exit.
# - the defender guards the crossing the most threatening enemy will use, marks
#   enemies waiting at the border and cuts off invaders on their way home.
# - roles adapt to the score, the remaining time and the number of invaders.
# - decisions are kept stable between turns (remembered ghosts, sticky food
#   targets) so that enemies flickering in and out of sight, or stepping in and
#   out at the border, do not make our agents oscillate.

import random
from collections import deque

from contest.capture_agents import CaptureAgent
from contest.game import Directions, Actions


#################
# team creation #
#################

def create_team(first_index, second_index, is_red,
                first='FlexAgent', second='FlexAgent', num_training=0):
    """
    returns the two agents of the team. the first one starts as attacker,
    the roles are decided by the team brain from the starting positions.
    """
    brain = TeamBrain()
    return [eval(first)(first_index, brain), eval(second)(second_index, brain)]


MOVES = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
INF = 10 ** 6
DEBUG = False          # raise exceptions instead of falling back to a random move
CARRY_BASE = 6         # food carried before banking (grows with the way home)
LATE_ATTACK = 150      # moves left when a trailing defender joins the attack
LOCK_LEAD = 60         # moves left (+ way home) when a leading attacker defends
RISK1 = 10             # turns blocked at the border before ignoring guessed ghosts
RISK2 = 30             # turns blocked before keeping only a radius-2 safety margin
RAID = True            # the defender may grab food close to the border
RAID_DEPTH = 5         # how far from the border a raid may go
RAID_CARRY = 2         # food carried before a raid returns
USE_HIDDEN = True      # plan around the estimated positions of unseen ghosts
GHOST_MEMORY = 20      # timeleft units (4 per round) a ghost stays remembered
TARGET_STICK = 2       # bonus for keeping the previous food target
SHALLOW_DEPTH = 2      # an invader this close to the border is "at the border"
COVER_DIST = 8         # the defender covers invaders within this distance


##############
# team brain #
##############

class TeamBrain:
    """
    knowledge shared by both agents of the team (one instance per team).
    """

    def __init__(self):
        self.registered = set()
        self.ready = False

    # ------------------------------------------------------------------ setup

    def setup(self, agent, game_state):
        """
        called by every agent in register_initial_state. a second call from the
        same agent index means a new game started, so everything is rebuilt.
        """
        if agent.index in self.registered:
            self.registered = set()
            self.ready = False
        self.registered.add(agent.index)
        if self.ready:
            return

        self.red = agent.red
        self.distancer = agent.distancer
        self._table = getattr(self.distancer, '_distances', None) or {}
        walls = game_state.get_walls()
        self.walls = walls
        self.width = walls.width
        self.height = walls.height
        self.cells = [(x, y) for x in range(self.width) for y in range(self.height) if not walls[x][y]]
        self.cell_set = set(self.cells)
        self.neighbors = {}
        for (x, y) in self.cells:
            nbrs = []
            for move in MOVES:
                dx, dy = Actions.direction_to_vector(move)
                nxt = (int(x + dx), int(y + dy))
                if nxt in self.cell_set:
                    nbrs.append(nxt)
            self.neighbors[(x, y)] = nbrs

        mid = self.width // 2
        if self.red:
            self.home_x = mid - 1
            self.enemy_x = mid
        else:
            self.home_x = mid
            self.enemy_x = mid - 1
        # home border cells that touch an enemy cell (the real crossing points)
        self.home_border = [(self.home_x, y) for y in range(self.height)
                            if (self.home_x, y) in self.cell_set and (self.enemy_x, y) in self.cell_set]
        if not self.home_border:
            self.home_border = [(self.home_x, y) for y in range(self.height) if (self.home_x, y) in self.cell_set]
        self.enemy_border = [(self.enemy_x, y) for y in range(self.height)
                             if (self.enemy_x, y) in self.cell_set and (self.home_x, y) in self.cell_set]

        self.home_cells = [c for c in self.cells if self.is_home(c)]
        self.enemy_cells = [c for c in self.cells if not self.is_home(c)]

        # distance from every cell to the closest home border cell
        self.dist_home = {}
        for c in self.cells:
            self.dist_home[c] = min(self.distancer.get_distance(c, b) for b in self.home_border)

        self._compute_dead_ends()

        self.opponents = agent.get_opponents(game_state)
        self.team = agent.get_team(game_state)
        self.start_pos = {}
        self.beliefs = {}
        for opp in self.opponents:
            start = game_state.get_initial_agent_position(opp)
            start = (int(start[0]), int(start[1]))
            self.start_pos[opp] = start
            self.beliefs[opp] = {start: 1.0}
        self.last_seen = {opp: None for opp in self.opponents}
        # last position where we saw each enemy ghost: (cell, timeleft)
        self.ghost_memory = {}
        self.prev_defended_food = set(agent.get_food_you_are_defending(game_state).as_list())
        self.prev_defended_caps = set(agent.get_capsules_you_are_defending(game_state))
        self.total_food = len(agent.get_food(game_state).as_list())
        self.targets = {}
        self.stall = {}
        self.blocked = {}
        self.last_score = 0
        self.last_carry = {}
        self._assign_roles(agent, game_state)
        self.ready = True

    def _assign_roles(self, agent, game_state):
        """
        the agent closer to the enemy food attacks, the other one defends.
        """
        food = agent.get_food(game_state).as_list()
        scores = []
        for i in self.team:
            p = game_state.get_initial_agent_position(i)
            p = (int(p[0]), int(p[1]))
            d = min([self.dist(p, f) for f in food] or [0])
            scores.append((d, i))
        scores.sort()
        self.roles = {scores[0][1]: 'attack'}
        for _d, i in scores[1:]:
            self.roles[i] = 'defend'

    def is_home(self, cell):
        if self.red:
            return cell[0] < self.width // 2
        return cell[0] >= self.width // 2

    def dist(self, a, b):
        # direct lookup in the precomputed maze distance table (much faster
        # than the generic distancer call); falls back for odd positions
        try:
            return self._table[(a, b)]
        except (KeyError, TypeError):
            return self.distancer.get_distance(a, b)

    def _compute_dead_ends(self):
        """
        iteratively prunes cells with a single open neighbour. every pruned cell
        lies inside a dead-end corridor; we store the junction that leads out of
        it and how many steps away that junction is.
        """
        degree = {c: len(self.neighbors[c]) for c in self.cells}
        removed = set()
        queue = deque([c for c in self.cells if degree[c] <= 1])
        while queue:
            c = queue.popleft()
            if c in removed:
                continue
            removed.add(c)
            for n in self.neighbors[c]:
                if n not in removed:
                    degree[n] -= 1
                    if degree[n] <= 1:
                        queue.append(n)
        self.trap_exit = {}
        self.trap_depth = {}
        for c in self.cells:
            if c not in removed:
                self.trap_exit[c] = c
                self.trap_depth[c] = 0
        # bfs from the non-trap cells into the pruned regions
        frontier = deque([c for c in self.cells if c not in removed])
        while frontier:
            c = frontier.popleft()
            for n in self.neighbors[c]:
                if n in removed and n not in self.trap_exit:
                    self.trap_exit[n] = self.trap_exit[c]
                    self.trap_depth[n] = self.trap_depth[c] + 1
                    frontier.append(n)
        # a maze that is a pure tree: fall back to "no traps"
        for c in self.cells:
            if c not in self.trap_exit:
                self.trap_exit[c] = c
                self.trap_depth[c] = 0

    # --------------------------------------------------------------- tracking

    def update_beliefs(self, agent, game_state):
        """
        updates the opponents' position beliefs with the observation received by
        the agent that is about to move.
        """
        my_pos = agent.my_pos
        team_pos = [game_state.get_agent_position(i) for i in self.team]
        team_pos = [(int(p[0]), int(p[1])) for p in team_pos if p is not None]
        noisy = game_state.get_agent_distances()
        mover = (agent.index - 1) % game_state.get_num_agents()

        # food or capsules that vanished from our side since the last update
        defended = set(agent.get_food_you_are_defending(game_state).as_list())
        caps = set(agent.get_capsules_you_are_defending(game_state))
        eaten = (self.prev_defended_food - defended) | (self.prev_defended_caps - caps)
        food_came_back = len(defended - self.prev_defended_food) > 0
        self.prev_defended_food = defended
        self.prev_defended_caps = caps

        for opp in self.opponents:
            enemy = game_state.get_agent_state(opp)
            pos = enemy.get_position()
            if pos is not None:
                pos = (int(pos[0]), int(pos[1]))
                self.beliefs[opp] = {pos: 1.0}
                self.last_seen[opp] = pos
                if enemy.is_pacman:
                    self.ghost_memory.pop(opp, None)
                else:
                    self.ghost_memory[opp] = (pos, game_state.data.timeleft)
                continue
            if enemy.is_pacman:
                self.ghost_memory.pop(opp, None)

            belief = self.beliefs[opp]
            # an enemy we could see next to one of our ghosts just vanished: eaten
            last = self.last_seen[opp]
            if last is not None and len(belief) == 1 and last in belief:
                if any(abs(last[0] - t[0]) + abs(last[1] - t[1]) <= 1 for t in team_pos) or food_came_back:
                    belief = {self.start_pos[opp]: 1.0}
                    self.ghost_memory.pop(opp, None)
            self.last_seen[opp] = None

            # time elapse: only the opponent that played just before us moved
            if opp == mover:
                new_belief = {}
                for c, p in belief.items():
                    options = self.neighbors[c] + [c]
                    share = p / len(options)
                    for n in options:
                        new_belief[n] = new_belief.get(n, 0.0) + share
                belief = new_belief

            eaten_here = [e for e in eaten if opp == mover]
            belief = self._observe(belief, enemy, my_pos, team_pos, noisy[opp], eaten_here)
            if not belief:
                belief = self._reset_belief(opp, enemy, my_pos, team_pos, noisy[opp])
            self.beliefs[opp] = belief

    def _consistent(self, cell, enemy, team_pos):
        if enemy.is_pacman != self.is_home(cell):
            return False
        for t in team_pos:
            if abs(cell[0] - t[0]) + abs(cell[1] - t[1]) <= 5:
                return False
        return True

    def _observe(self, belief, enemy, my_pos, team_pos, noisy_dist, eaten_here):
        if eaten_here:
            spot = eaten_here[0]
            if enemy.is_pacman:
                return {spot: 1.0}
        result = {}
        total = 0.0
        for c, p in belief.items():
            if not self._consistent(c, enemy, team_pos):
                continue
            if noisy_dist is not None:
                true_d = abs(c[0] - my_pos[0]) + abs(c[1] - my_pos[1])
                if abs(noisy_dist - true_d) > 6:
                    continue
            result[c] = p
            total += p
        if total <= 0:
            return {}
        return {c: p / total for c, p in result.items()}

    def _reset_belief(self, opp, enemy, my_pos, team_pos, noisy_dist):
        start = self.start_pos[opp]
        if not enemy.is_pacman and self._consistent(start, enemy, team_pos):
            # most likely it just respawned
            near_start = {start: 1.0}
            obs = self._observe(near_start, enemy, my_pos, team_pos, noisy_dist, [])
            if obs:
                return obs
        candidates = {c: 1.0 for c in self.cells}
        obs = self._observe(candidates, enemy, my_pos, team_pos, noisy_dist, [])
        if obs:
            return obs
        side = [c for c in self.cells if self.is_home(c) == enemy.is_pacman]
        if not side:
            side = self.cells
        return {c: 1.0 / len(side) for c in side}

    def estimate(self, opp):
        belief = self.beliefs.get(opp)
        if not belief:
            return self.start_pos[opp]
        return max(belief.items(), key=lambda kv: kv[1])[0]


##########
# agents #
##########

class BaseAgent(CaptureAgent):
    """
    common machinery for both roles: setup, safe search, helpers.
    """

    def __init__(self, index, brain, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.brain = brain
        self.my_pos = None
        self.recent = deque(maxlen=12)

    def register_initial_state(self, game_state):
        self.red = game_state.is_on_red_team(self.index)
        self.register_team(self.get_team(game_state))
        brain = self.brain
        # the maze distances are computed only once per team
        if brain.ready and brain.registered and self.index not in brain.registered \
                and getattr(brain, 'distancer', None) is not None:
            self.distancer = brain.distancer
        else:
            from contest import distance_calculator
            self.distancer = distance_calculator.Distancer(game_state.data.layout)
            self.distancer.get_maze_distances()
        import __main__
        if '_display' in dir(__main__):
            self.display = __main__._display
        brain.setup(self, game_state)
        self.recent = deque(maxlen=12)

    # -------------------------------------------------------------- main loop

    def choose_action(self, game_state):
        legal = game_state.get_legal_actions(self.index)
        try:
            pos = game_state.get_agent_position(self.index)
            self.my_pos = (int(pos[0]), int(pos[1]))
            self.brain.update_beliefs(self, game_state)
            action = self.decide(game_state)
            if action not in legal:
                action = self.fallback(game_state, legal)
        except Exception:
            if DEBUG:
                raise
            action = self.fallback(game_state, legal)
        self.recent.append(self.my_pos)
        return action

    def fallback(self, game_state, legal):
        options = [a for a in legal if a != Directions.STOP]
        if not options:
            return Directions.STOP
        return random.choice(options)

    # ---------------------------------------------------------------- helpers

    def step(self, cell, action):
        dx, dy = Actions.direction_to_vector(action)
        return (int(cell[0] + dx), int(cell[1] + dy))

    def action_towards(self, game_state, nxt):
        for a in game_state.get_legal_actions(self.index):
            if self.step(self.my_pos, a) == nxt:
                return a
        return None

    def moves_left(self, game_state):
        # timeleft counts the moves of all four agents
        return game_state.data.timeleft // 4

    def threat_ghosts(self, game_state, include_hidden=False):
        """
        returns a list of (position, scared_timer, visible) for enemy ghosts that
        may be able to eat us.
        """
        threats = []
        for opp in self.brain.opponents:
            st = game_state.get_agent_state(opp)
            if st.is_pacman:
                continue
            p = st.get_position()
            if p is not None:
                threats.append(((int(p[0]), int(p[1])), st.scared_timer, True))
            elif include_hidden:
                threats.append((self.brain.estimate(opp), st.scared_timer, False))
        return threats

    def hidden_aware(self, game_state, threats):
        """
        adds unseen enemy ghosts at their estimated position, but never closer
        than they can be (an unseen enemy is more than 5 steps from both of us).
        this keeps plans stable when a ghost flickers in and out of sight.
        """
        brain = self.brain
        result = list(threats)
        for opp in brain.opponents:
            st = game_state.get_agent_state(opp)
            if st.is_pacman or st.get_position() is not None:
                continue
            belief = brain.beliefs.get(opp, {})
            if not belief or max(belief.values()) < 0.2:
                continue
            est = brain.estimate(opp)
            result.append((est, st.scared_timer, False))
        return result

    def remembered_ghosts(self, game_state, window=GHOST_MEMORY):
        """
        enemy ghosts we cannot see now but saw during the last few turns, at the
        cell where we last saw them. without this a ghost that flickers in and
        out of sight makes the plan flip every turn.
        """
        brain = self.brain
        now = game_state.data.timeleft
        result = []
        for opp, (cell, seen_at) in brain.ghost_memory.items():
            st = game_state.get_agent_state(opp)
            if st.is_pacman or st.get_position() is not None:
                continue
            if seen_at - now <= window:
                result.append((cell, st.scared_timer, False))
        return result

    def bfs(self, start, allowed):
        """
        breadth first search from start. allowed(cell, t) tells whether the cell
        can be occupied after t steps. returns (time, first_step) dictionaries.
        """
        nbrs = self.brain.neighbors
        times = {start: 0}
        first = {start: None}
        queue = deque([start])
        while queue:
            c = queue.popleft()
            t = times[c] + 1
            for n in nbrs[c]:
                if n in times:
                    continue
                if not allowed(n, t):
                    continue
                times[n] = t
                first[n] = n if c == start else first[c]
                queue.append(n)
        return times, first

    def danger_fn(self, threats, slack=0, cap=INF):
        """
        builds a predicate telling whether a cell is safe when reached after t
        steps, given the threatening ghosts. home cells are always safe for a
        non-scared agent (we are a ghost there).
        """
        brain = self.brain
        dist = brain.dist

        def allowed(cell, t):
            if brain.is_home(cell):
                return True
            for gpos, scared, _vis in threats:
                gd = dist(gpos, cell)
                # a scared ghost is harmless until its timer expires
                if scared > 0 and scared > t + 1:
                    continue
                if gd <= min(t + slack, cap):
                    return False
            return True
        return allowed

    def escape_ok(self, cell, t, threats, margin=1):
        """
        true if, after reaching cell at time t, we can leave its dead-end
        corridor (if any) before a ghost reaches the corridor exit.
        """
        brain = self.brain
        depth = brain.trap_depth.get(cell, 0)
        if depth == 0:
            return True
        exit_cell = brain.trap_exit[cell]
        out_time = t + depth
        for gpos, scared, _vis in threats:
            if scared > out_time + 1:
                continue
            if brain.dist(gpos, exit_cell) <= out_time + margin:
                return False
        return True

    def chase_step(self, game_state, target, stay_home=False):
        """
        first action of a shortest path to target. when stay_home is set the
        path never leaves our half (so a defender stays a ghost).
        """
        brain = self.brain
        if target == self.my_pos:
            return Directions.STOP
        best, best_d = None, INF
        for a in game_state.get_legal_actions(self.index):
            if a == Directions.STOP:
                continue
            nxt = self.step(self.my_pos, a)
            if stay_home and not brain.is_home(nxt):
                continue
            d = brain.dist(nxt, target)
            if d < best_d:
                best, best_d = a, d
        return best

    def home_path_step(self, game_state, target):
        """
        shortest path to target restricted to our half.
        """
        brain = self.brain
        times, first = self.bfs(self.my_pos, lambda c, t: brain.is_home(c))
        if target in first and first[target] is not None:
            return self.action_towards(game_state, first[target])
        return self.chase_step(game_state, target, stay_home=True)

    # ------------------------------------------------------- role decisions

    def invaders(self, game_state):
        result = []
        for opp in self.brain.opponents:
            st = game_state.get_agent_state(opp)
            if st.is_pacman:
                p = st.get_position()
                if p is not None:
                    pos, vis = (int(p[0]), int(p[1])), True
                else:
                    pos, vis = self.brain.estimate(opp), False
                result.append({'index': opp, 'pos': pos, 'visible': vis,
                               'carrying': st.num_carrying})
        return result

    def teammate_index(self):
        return [i for i in self.brain.team if i != self.index][0]

    def covered_by_mate(self, game_state, inv):
        """
        true for an invader that carries nothing, stands at the border and is
        close to our teammate acting as a (non-scared) ghost. such an invader is
        often just stepping in and out to lure us: leave it to the teammate.
        """
        brain = self.brain
        if inv['carrying'] > 0 or brain.dist_home.get(inv['pos'], INF) > SHALLOW_DEPTH:
            return False
        mate = self.teammate_index()
        st = game_state.get_agent_state(mate)
        p = st.get_position()
        if p is None or st.is_pacman or st.scared_timer > 0:
            return False
        return brain.dist((int(p[0]), int(p[1])), inv['pos']) <= COVER_DIST

    def hovering_enemy(self, game_state):
        """
        the visible enemy ghost closest to our border, if it is within
        SHALLOW_DEPTH steps of crossing. returns its position or None.
        """
        brain = self.brain
        best, best_d = None, INF
        for opp in brain.opponents:
            st = game_state.get_agent_state(opp)
            p = st.get_position()
            if st.is_pacman or p is None:
                continue
            p = (int(p[0]), int(p[1]))
            d = min(brain.dist(p, b) for b in brain.home_border)
            if d <= SHALLOW_DEPTH and d < best_d:
                best, best_d = p, d
        return best

    def mark(self, game_state, enemy_pos):
        """
        stand on the home border cell in front of an enemy waiting at our
        border, so it cannot step in without being eaten.
        """
        brain = self.brain
        cell = min(brain.home_border,
                   key=lambda b: (brain.dist(b, enemy_pos), brain.dist(self.my_pos, b)))
        if cell == self.my_pos:
            return Directions.STOP
        return self.home_path_step(game_state, cell)

    # ------------------------------------------------------------- offense

    def offense(self, game_state, raid=False):
        brain = self.brain
        my_state = game_state.get_agent_state(self.index)
        carrying = my_state.num_carrying
        food = self.get_food(game_state).as_list()
        capsules = self.get_capsules(game_state)
        moves_left = self.moves_left(game_state)
        my_home_dist = brain.dist_home[self.my_pos]

        threats = self.threat_ghosts(game_state)
        hidden_threats = self.threat_ghosts(game_state, include_hidden=True)
        active = [g for g in threats if g[1] <= 2]
        active_close = [g for g in active if brain.dist(g[0], self.my_pos) <= 6]

        # opportunistic kill of an invader while we are still a ghost at home
        if not my_state.is_pacman and my_state.scared_timer == 0:
            for inv in self.invaders(game_state):
                if self.covered_by_mate(game_state, inv):
                    continue
                if inv['visible'] and brain.dist(self.my_pos, inv['pos']) <= 2:
                    a = self.home_path_step(game_state, inv['pos'])
                    if a is not None:
                        return a

        # progress tracking: time spent near the border without eating
        last_carry = brain.last_carry.get(self.index, 0)
        brain.last_carry[self.index] = carrying
        if carrying > last_carry:
            brain.blocked[self.index] = 0
        elif my_home_dist <= 3:
            brain.blocked[self.index] = brain.blocked.get(self.index, 0) + 1

        # stuck behind a wall of ghosts while not ahead: accept more risk
        risk = self.risk_level(game_state)
        # at higher risk we drop the guessed positions of hidden ghosts, but we
        # still keep the ghosts seen a moment ago (hysteresis)
        recent_threats = threats + self.remembered_ghosts(game_state)
        if risk >= 1:
            hidden_threats = recent_threats
        cap = 2 if risk >= 2 else INF
        plan_threats = recent_threats if (risk >= 1 or not USE_HIDDEN) else self.hidden_aware(game_state, threats)
        allowed = self.danger_fn(plan_threats, cap=cap)
        times, first = self.bfs(self.my_pos, allowed)

        returned = sum(game_state.get_agent_state(i).num_returned for i in brain.team)
        needed = brain.total_food - 2 - returned

        # ---------------------------------------------- when to go back home
        go_home = False
        if carrying > 0:
            if len(food) <= 2 or carrying >= needed:
                go_home = True
            elif moves_left <= my_home_dist + 3:
                go_home = True
            elif active_close and carrying >= 2:
                go_home = True
            elif raid and carrying >= RAID_CARRY:
                go_home = True
            elif carrying >= self.carry_limit(game_state, my_home_dist):
                go_home = True
        if not my_state.is_pacman:
            go_home = False

        if go_home:
            a = self.go_home(game_state, times, first, threats, capsules, carrying)
            if a is not None:
                return a

        # ---------------------------------------------- scared ghosts nearby
        for gpos, scared, vis in threats:
            d = brain.dist(self.my_pos, gpos)
            if vis and scared > d + 1 and d <= 4 and not brain.is_home(gpos):
                if gpos in first and first[gpos] is not None:
                    return self.action_towards(game_state, first[gpos])

        # ---------------------------------------------- capsule when hunted
        if capsules and active_close:
            best_cap, best_t = None, INF
            for cap in capsules:
                if cap in times and times[cap] < best_t:
                    best_cap, best_t = cap, times[cap]
            if best_cap is not None and best_t <= 10:
                return self.action_towards(game_state, first[best_cap])

        # ---------------------------------------------- choose a food target
        if raid:
            # only food close to the border: we must be back quickly
            food = [f for f in food if brain.dist_home[f] <= RAID_DEPTH]
            capsules = []
        target = self.pick_food(game_state, food, capsules, times,
                                hidden_threats if risk < 2 else [], carrying)
        if target is not None:
            brain.targets[self.index] = target
            if my_state.is_pacman:
                brain.stall[self.index] = 0
            if first.get(target) is not None:
                return self.action_towards(game_state, first[target])

        # ---------------------------------------------- nothing safe to eat
        if raid and not my_state.is_pacman:
            return self.guard(game_state)
        if my_state.is_pacman:
            a = self.go_home(game_state, times, first, threats, capsules, carrying)
            if a is not None:
                return a
        # wait on our side, at a crossing point far from the blocking ghosts
        brain.stall[self.index] = brain.stall.get(self.index, 0) + 1
        return self.probe_border(game_state, active)

    def risk_level(self, game_state):
        """
        0: fully cautious. 1: ignore guessed positions of hidden ghosts.
        2: only keep a small safety radius around visible ghosts.
        the level grows with the time spent blocked at the border and resets
        whenever the team scores.
        """
        brain = self.brain
        score = self.get_score(game_state)
        if score > brain.last_score:
            brain.blocked = {}
        brain.last_score = score
        if score > 0:
            return 0
        blocked = brain.blocked.get(self.index, 0)
        if blocked > RISK2:
            return 2
        if blocked > RISK1:
            return 1
        return 0

    def carry_limit(self, game_state, home_dist):
        """
        how much food we are willing to carry before banking it. longer trips
        back home and a lead in score make us bank earlier.
        """
        food_left = len(self.get_food(game_state).as_list())
        limit = CARRY_BASE + home_dist // 6
        if self.get_score(game_state) > 5:
            limit -= 1
        if food_left <= 6:
            limit = min(limit, 3)
        return max(2, limit)

    def pick_food(self, game_state, food, capsules, times, threats, carrying):
        brain = self.brain
        mate = self.teammate_index()
        mate_target = brain.targets.get(mate)
        mate_pos = game_state.get_agent_position(mate)
        mate_attacking = getattr(self, '_mate_attacking', False)
        prev_target = brain.targets.get(self.index)
        best, best_score = None, INF
        candidates = list(food)
        for cap in capsules:
            candidates.append(cap)
        for f in candidates:
            t = times.get(f)
            if t is None:
                continue
            if not self.escape_ok(f, t, threats):
                continue
            score = t
            # once we carry something, prefer food that keeps us close to home
            score += 0.25 * carrying * brain.dist_home[f] / 3.0
            if f in capsules:
                active = [g for g in threats if g[1] <= 2 and g[2]]
                score += -4 if active else 6
            # keep the two attackers apart
            if mate_target is not None and mate_pos is not None and mate_attacking:
                if brain.dist(f, mate_target) <= 3:
                    score += 6
            # stick to the current target while it stays safe
            if f == prev_target:
                score -= TARGET_STICK
            if score < best_score:
                best, best_score = f, score
        return best

    def go_home(self, game_state, times, first, threats, capsules, carrying):
        brain = self.brain
        best, best_t = None, INF
        for b in brain.home_border:
            t = times.get(b)
            if t is not None and t < best_t:
                best, best_t = b, t
        cap_choice = None
        for cap in capsules:
            t = times.get(cap)
            if t is not None and (cap_choice is None or t < cap_choice[1]):
                cap_choice = (cap, t)
        if best is not None:
            if cap_choice is not None and cap_choice[1] < best_t * 0.5 and carrying < 8:
                return self.action_towards(game_state, first[cap_choice[0]])
            if first[best] is not None:
                return self.action_towards(game_state, first[best])
        if cap_choice is not None and first[cap_choice[0]] is not None:
            return self.action_towards(game_state, first[cap_choice[0]])
        # no fully safe path: accept tighter margins
        for slack in (-1, -2):
            allowed = self.danger_fn(threats, slack=slack)
            t2, f2 = self.bfs(self.my_pos, allowed)
            goals = [b for b in brain.home_border if b in t2] + [c for c in capsules if c in t2]
            if goals:
                g = min(goals, key=lambda c: t2[c])
                if f2[g] is not None:
                    return self.action_towards(game_state, f2[g])
        return self.evade(game_state, threats)

    def evade(self, game_state, threats):
        """
        last resort: move to the neighbour that maximises the distance to the
        closest dangerous ghost, preferring cells closer to home.
        """
        brain = self.brain
        best, best_score = None, -INF
        for a in game_state.get_legal_actions(self.index):
            nxt = self.step(self.my_pos, a)
            gd = min([brain.dist(g[0], nxt) for g in threats if g[1] <= 2] or [INF])
            score = min(gd, 10) * 10 - brain.dist_home[nxt] - brain.trap_depth[nxt] * 3
            if a == Directions.STOP:
                score -= 5
            if score > best_score:
                best, best_score = a, score
        return best

    def probe_border(self, game_state, ghosts):
        """
        stay on our side and slide along the border to the crossing point that
        is farthest from the ghosts blocking us.
        """
        brain = self.brain
        if not brain.home_border:
            return self.fallback(game_state, game_state.get_legal_actions(self.index))
        best, best_score = None, -INF
        mate_pos = game_state.get_agent_position(self.teammate_index())
        for b in brain.home_border:
            gd = min([brain.dist(g[0], b) for g in ghosts] or [20])
            score = min(gd, 12) * 2 - brain.dist(self.my_pos, b)
            if mate_pos is not None and getattr(self, '_mate_attacking', False):
                score += min(brain.dist(mate_pos, b), 10)
            if score > best_score:
                best, best_score = b, score
        if best == self.my_pos:
            return Directions.STOP
        return self.home_path_step(game_state, best)

    # ------------------------------------------------------------- defense

    def defense(self, game_state, target_inv=None):
        brain = self.brain
        my_state = game_state.get_agent_state(self.index)

        if my_state.is_pacman:
            # we drifted into enemy land: get back safely
            threats = self.threat_ghosts(game_state)
            times, first = self.bfs(self.my_pos, self.danger_fn(threats))
            a = self.go_home(game_state, times, first, threats, [], my_state.num_carrying)
            if a is not None:
                return a

        invaders = self.invaders(game_state)
        if target_inv is None and invaders:
            target_inv = min(invaders, key=lambda inv: brain.dist(self.my_pos, inv['pos'])
                             - (3 if inv['visible'] else 0) - inv['carrying'])

        if target_inv is not None:
            inv_pos = target_inv['pos']
            if my_state.scared_timer > 0:
                return self.shadow(game_state, inv_pos)
            d = brain.dist(self.my_pos, inv_pos)
            brain.targets[('chase', self.index)] = (target_inv['index'], game_state.data.timeleft)
            # two of us on the same invader: the farther one blocks its escape
            partner = self.partner_on(game_state, target_inv['index'])
            if partner is not None and brain.dist(partner, inv_pos) < d and d > 1:
                block = self.block_cell(inv_pos, partner, target_inv['carrying'])
                if block is not None and block != self.my_pos:
                    return self.home_path_step(game_state, block)
                if block == self.my_pos:
                    return Directions.STOP
            return self.home_path_step(game_state, inv_pos)

        return self.guard(game_state)

    def partner_on(self, game_state, inv_index):
        """
        position of the teammate if it is a ghost chasing the same invader.
        """
        brain = self.brain
        mate = self.teammate_index()
        info = brain.targets.get(('chase', mate))
        if info is None or info[0] != inv_index:
            return None
        # the information must be fresh (set during the last couple of turns)
        if info[1] - game_state.data.timeleft > 4:
            return None
        st = game_state.get_agent_state(mate)
        if st.is_pacman or st.scared_timer > 0:
            return None
        p = st.get_position()
        return (int(p[0]), int(p[1])) if p is not None else None

    def block_cell(self, inv_pos, partner, carrying):
        """
        cell that best closes the invader's escape while the partner chases it:
        inside the region the invader reaches before the partner, pick the cell
        we reach first that lies on the invader's way home (when it carries food)
        or that is farthest from the partner.
        """
        brain = self.brain
        dist = brain.dist
        region = [c for c in brain.home_cells
                  if dist(inv_pos, c) <= 8 and dist(inv_pos, c) < dist(partner, c)]
        if carrying > 0:
            exit_cell = min(brain.home_border, key=lambda b: dist(inv_pos, b))
            on_path = [c for c in brain.home_cells
                       if dist(inv_pos, c) + dist(c, exit_cell) == dist(inv_pos, exit_cell)
                       and dist(self.my_pos, c) <= dist(inv_pos, c)]
            if on_path:
                return min(on_path, key=lambda c: (dist(inv_pos, c), dist(self.my_pos, c)))
            return exit_cell
        if not region:
            return inv_pos
        return max(region, key=lambda c: (dist(partner, c) - dist(self.my_pos, c), -dist(inv_pos, c)))

    def shadow(self, game_state, inv_pos):
        """
        while scared, stay two steps away from the invader so we can pounce as
        soon as the scared timer runs out, without being eaten.
        """
        brain = self.brain
        best, best_score = None, -INF
        for a in game_state.get_legal_actions(self.index):
            nxt = self.step(self.my_pos, a)
            if not brain.is_home(nxt):
                continue
            d = brain.dist(nxt, inv_pos)
            if d <= 1:
                score = -100 + d
            else:
                score = -abs(d - 2)
            if score > best_score:
                best, best_score = a, score
        return best if best is not None else Directions.STOP

    def guard(self, game_state):
        """
        no invader: wait on the home border cell closest to the enemy that is
        most likely to attack next, breaking ties towards our food.
        """
        brain = self.brain
        candidates = brain.home_border
        if not candidates:
            return Directions.STOP
        # the enemy closest to our border is the most likely next attacker
        enemies = [(opp, brain.estimate(opp)) for opp in brain.opponents]
        enemies.sort(key=lambda e: brain.dist_home.get(e[1], INF))
        focus = enemies[0][1]
        defended = self.get_food_you_are_defending(game_state).as_list()

        def food_score(b):
            if not defended:
                return 0
            return min(brain.dist(b, f) for f in defended)

        mate = self.teammate_index()
        mate_pos = game_state.get_agent_position(mate)
        mate_defending = getattr(self, '_mate_defending', False)

        best, best_score = None, INF
        for b in candidates:
            score = brain.dist(b, focus) * 2 + food_score(b) * 0.5
            if mate_defending and mate_pos is not None:
                # two defenders: cover different crossings
                second = enemies[1][1] if len(enemies) > 1 else focus
                score = brain.dist(b, second) * 2 + food_score(b) * 0.5
                if brain.dist(b, mate_pos) <= 3:
                    score += 8
            if score < best_score:
                best, best_score = b, score
        if best == self.my_pos:
            return Directions.STOP
        return self.home_path_step(game_state, best)

    # -------------------------------------------------------------- role

    def decide(self, game_state):
        raise NotImplementedError


class FlexAgent(BaseAgent):
    """
    plays the role assigned by the team brain: the agent that starts closer to
    the enemy food attacks, the other one defends.
    """

    def decide(self, game_state):
        if self.brain.roles.get(self.index) == 'attack':
            return self.decide_attacker(game_state)
        return self.decide_defender(game_state)

    # ------------------------------------------------------------ attacker

    def decide_attacker(self, game_state):
        brain = self.brain
        my_state = game_state.get_agent_state(self.index)
        score = self.get_score(game_state)
        moves_left = self.moves_left(game_state)
        # invaders the defender already handles do not pull us back home
        invaders = [inv for inv in self.invaders(game_state)
                    if not self.covered_by_mate(game_state, inv)]
        self._mate_attacking = False
        self._mate_defending = True

        if not my_state.is_pacman or my_state.num_carrying == 0:
            # protect a lead at the end of the game
            if score > 0 and moves_left < LOCK_LEAD + brain.dist_home[self.my_pos] and my_state.num_carrying == 0:
                if not my_state.is_pacman:
                    return self.defense(game_state, self.second_invader(invaders))
            # help the defender against invaders we are closer to
            if invaders and not my_state.is_pacman and my_state.scared_timer == 0:
                inv = self.second_invader(invaders)
                if inv is not None:
                    my_d = brain.dist(self.my_pos, inv['pos'])
                    mate_pos = game_state.get_agent_position(self.teammate_index())
                    mate_d = brain.dist(mate_pos, inv['pos']) if mate_pos is not None else INF
                    if len(invaders) >= 2 and my_d <= 10:
                        brain.stall[self.index] = 0
                        return self.defense(game_state, inv)
                    if my_d < mate_d and my_d <= 12:
                        brain.stall[self.index] = 0
                        return self.defense(game_state, inv)
        return self.offense(game_state)

    def second_invader(self, invaders):
        """
        the invader the defender is not chasing.
        """
        if not invaders:
            return None
        brain = self.brain
        if len(invaders) == 1:
            return invaders[0]
        mate = self.teammate_index()
        mate_target = brain.targets.get(('def', mate))
        others = [inv for inv in invaders if inv['index'] != mate_target]
        if not others:
            others = invaders
        return min(others, key=lambda inv: brain.dist(self.my_pos, inv['pos']))


    # ------------------------------------------------------------ defender

    def decide_defender(self, game_state):
        brain = self.brain
        my_state = game_state.get_agent_state(self.index)
        score = self.get_score(game_state)
        moves_left = self.moves_left(game_state)
        invaders = self.invaders(game_state)
        self._mate_attacking = True
        self._mate_defending = False

        # trailing late in the game with nobody invading: attack as well
        if not invaders and score <= 0 and moves_left < LATE_ATTACK:
            return self.offense(game_state)
        # the attacker is stuck at the border: open a second front
        mate = self.teammate_index()
        if not invaders and score <= 0 and brain.stall.get(mate, 0) > 15:
            return self.offense(game_state)

        if invaders:
            target = min(invaders, key=lambda inv: brain.dist(self.my_pos, inv['pos'])
                         - (3 if inv['visible'] else 0) - inv['carrying'])
            brain.targets[('def', self.index)] = target['index']
            return self.defense(game_state, target)
        brain.targets[('def', self.index)] = None
        # an enemy waiting at our border: block its crossing instead of
        # switching between chasing and guarding every time it steps in and out
        if my_state.scared_timer == 0 and not my_state.is_pacman:
            hover = self.hovering_enemy(game_state)
            if hover is not None:
                return self.mark(game_state, hover)
        if RAID and my_state.scared_timer == 0 and self.raid_ok(game_state):
            return self.offense(game_state, raid=True)
        return self.defense(game_state)

    def raid_ok(self, game_state):
        """
        the defender may grab food near the border while no enemy is on our
        side, as long as the enemy attackers are not about to cross.
        """
        brain = self.brain
        for opp in brain.opponents:
            st = game_state.get_agent_state(opp)
            if st.is_pacman:
                return False
            est = brain.estimate(opp)
            # an enemy hovering at its border is about to invade
            if brain.dist(est, self.my_pos) > 0 and min(brain.dist(est, b) for b in brain.home_border) <= 2 \
                    and brain.beliefs[opp].get(est, 0) > 0.5 and st.get_position() is not None \
                    and brain.dist(st.get_position(), self.my_pos) > 3:
                return False
        return True
