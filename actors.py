import math
import nnet
import util


class Actor:
	# Size of simulation world
	_WORLD_SIZE = (1, 1)

	@staticmethod
	def set_world_size(size):
		""" Set size of the simulation world """
		Actor._WORLD_SIZE = size

	def __init__(self):
		self.x = self.y = self.radians = 0
		self.move_to_random()
		self.radius = 1
		self.health = 100

	def on_tick(self):
		"""
		Called every tick of the simulation; derived classes must implement this
		function in order to have any real behavior
		"""
		pass

	def move(self, dx, dy, dr):
		""" Bump in some direction """
		self.move_to(self.x + dx, self.y + dy, self.radians + dr)

	def move_to(self, x, y, radians):
		""" Move to some location while staying within bounds of the world """
		self.x = x % Actor._WORLD_SIZE[0]
		self.y = y % Actor._WORLD_SIZE[1]
		self.radians = radians % (2 * math.pi)

	def move_to_random(self):
		""" Move to a random location and heading within the world """
		self.move_to(util.rand(0, Actor._WORLD_SIZE[0]),
					 util.rand(0, Actor._WORLD_SIZE[1]),
					 util.rand(0, 2 * math.pi))

	def get_pos(self):
		""" Get position as a tuple (x,y) """
		return (self.x, self.y)

	def is_alive(self):
		""" Is this Actor alive? """
		return self.health > 0


class Agent(Actor):
	_RADIUS = 20
	# Mutation during breeding
	_MUTATE_SYNAPSE_ODDS = 0.2
	_MUTATE_SYNAPSE_SHIFT = 0.5
	# Decremented from health every tick
	_HUNGER_PER_TICK = 0.34
	_HUNGER_MOVEMENT_RATIO = 0.05
	# Speed and limits of forward and turn movement
	_FORWARD_SPEED = 7
	_FORWARD_MAX = 5
	_TURN_SPEED = math.pi / 18.0 # 10 degrees
	_TURN_MAX = math.pi / 18.0 # 10 degrees
	# Position of smell sensor relative to Agent
	_SMELL_ANGLE = math.pi / 6.0 # 30 degrees
	_SMELL_LENGTH = 20
	# How far away an Agent can smell Food
	_SMELL_REACH = 50
	# Position of Agent sight sensor relative to Agent
	_SIGHT_ANGLE = math.pi / 4.5 # 40 degrees
	_SIGHT_LENGTH = 40
	# How far away an Agent can see other Agents
	_SIGHT_REACH = 60
	# Multiplier on Prisoner's Dilemma reward for health effect
	_PD_HEALTH_MULTIPLIER = 20

	@staticmethod
	def create_random_brain():
		""" Create a brain with randomized parameters """
		brain = nnet.NeuralNetwork()

		# Logic neurons
		logic_0 = nnet.Neuron("lgc_0")
		logic_1 = nnet.Neuron("lgc_1")
		logic_2 = nnet.Neuron("lgc_2")
		logic_3 = nnet.Neuron("lgc_3")
		# Input (hunger) neuron turns on as health goes down
		hunger = nnet.Neuron("hunger", is_input = True)
		# Input (smell) neurons for nearby Food
		food_left = nnet.Neuron("fd_lft", is_input = True)
		food_right = nnet.Neuron("fd_rght", is_input = True)
		# Input (sight) neurons for nearby Agents
		agent_left = nnet.Neuron("agnt_lft", is_input = True)
		agent_right = nnet.Neuron("agnt_rght", is_input = True)
		# Output (movement) neurons
		move_left = nnet.Neuron("mv_lft")
		move_right = nnet.Neuron("mv_rght")
		# Output (attack) neuron
		attack = nnet.Neuron("atk")

		# Input layer
		brain.add_neuron(agent_left)
		brain.add_neuron(food_left)
		brain.add_neuron(hunger)
		brain.add_neuron(food_right)
		brain.add_neuron(agent_right)
		# Hidden layer
		brain.add_neuron(logic_0)
		brain.add_neuron(logic_1)
		brain.add_neuron(logic_2)
		brain.add_neuron(logic_3)
		# Output layer
		brain.add_neuron(move_left)
		brain.add_neuron(attack)
		brain.add_neuron(move_right)

		# Input to hidden layer: left side
		brain.add_synapse(nnet.Synapse(agent_left, logic_0, util.rand(-0.25, 1)))
		brain.add_synapse(nnet.Synapse(agent_left, logic_1, util.rand(-1, 1)))
		brain.add_synapse(nnet.Synapse(food_left, logic_0, util.rand(-0.25, 0.25)))
		brain.add_synapse(nnet.Synapse(food_left, logic_1, util.rand(-0.25, 1)))
		# Input to hidden layer: center
		brain.add_synapse(nnet.Synapse(hunger, logic_1, util.rand(-0.25, 0.75)))
		brain.add_synapse(nnet.Synapse(hunger, logic_2, util.rand(-0.25, 0.75)))
		# Input to hidden layer: right side
		brain.add_synapse(nnet.Synapse(food_right, logic_2, util.rand(-0.25, 1)))
		brain.add_synapse(nnet.Synapse(food_right, logic_3, util.rand(-0.25, 0.25)))
		brain.add_synapse(nnet.Synapse(agent_right, logic_2, util.rand(-0.25, 0.25)))
		brain.add_synapse(nnet.Synapse(agent_right, logic_3, util.rand(-0.25, 1)))
		# Hidden to output layer: left side
		brain.add_synapse(nnet.Synapse(logic_0, move_left, util.rand(-0.25, 0.25)))
		brain.add_synapse(nnet.Synapse(logic_0, attack, util.rand(-0.5, 0.5)))
		brain.add_synapse(nnet.Synapse(logic_1, move_left, util.rand(-0.25, 1)))
		brain.add_synapse(nnet.Synapse(logic_1, attack, util.rand(-0.25, 0.25)))
		# Hidden to output layer: right side
		brain.add_synapse(nnet.Synapse(logic_2, attack, util.rand(-0.25, 0.25)))
		brain.add_synapse(nnet.Synapse(logic_2, move_right, util.rand(-0.25, 1)))
		brain.add_synapse(nnet.Synapse(logic_3, attack, util.rand(-0.5, 0.5)))
		brain.add_synapse(nnet.Synapse(logic_3, move_right, util.rand(-0.25, 0.25)))

		return brain

	@staticmethod
	def breed_brain(parent_1_brain, parent_2_brain):
		"""
		Create a brain with mixed traits from two parents by starting with a
		copy of parent_1_brain then swapping or mixing with parent_2_brain
		"""
		brain = parent_1_brain.make_copy()

		# For all synapses: replace with parent 2, average, or leave the same
		for i, s in enumerate(brain.synapses):
			chance = util.rand(0, 1)
			parent_2_weight = parent_2_brain.synapses[i].weight
			if chance > 0.66:
				s.weight = parent_2_weight
			elif chance > 0.33:
				s.weight = (s.weight + parent_2_weight) / 2.0

		return brain

	@staticmethod
	def mutate_brain(brain):
		""" Add random mutations to a brain """
		# For all synapses: shift in some direction with random chance
		for s in brain.synapses:
			if util.rand(0, 1) <= Agent._MUTATE_SYNAPSE_ODDS:
				s.weight += Agent._MUTATE_SYNAPSE_SHIFT * util.rand(-1, 1)
				s.weight = util.clamp(s.weight, -1, 1)

	def __init__(self, generation, brain):
		Actor.__init__(self)
		self.radius = Agent._RADIUS
		self.turn_force = self.forward_force = 0
		self.generation = generation
		self.brain = brain
		# Map of every interaction from id(other Agent) to whether or not that
		# Agent attacked during the previous encounter (True/False)
		self.memory = {}
		# The Agent interacted with this tick, last tick, and the result of the
		# most recent interaction
		self.interact_agent = None
		self.prev_interact_agent = None
		self.interact_attacked = False

	def reset(self):
		""" Reset all Agent properties but retain configuration of the brain """
		self.move_to_random()
		self.health = 100
		self.turn_force = self.forward_force = 0
		self.brain.reset()
		self.memory.clear()
		self.interact_agent = None
		self.prev_interact_agent = None
		self.interact_attacked = False

	def on_tick(self, world_agents, world_food):
		""" Update Agent state each tick of the simulation """
		# Neural network
		self._update_hunger_sensor()
		self._update_agent_sensors(world_agents)
		self._update_food_sensors(world_food)
		self.brain.update()
		# Movement
		self._update_movement_forward()
		self._update_movement_turn()
		self.move(math.cos(self.radians) * self.forward_force,
				  math.sin(self.radians) * self.forward_force,
				  self.turn_force)
		# Attacking and eating
		self._update_attack(world_agents)
		self._update_food(world_food)

	def process_attacks(self, model):
		"""
		Update status depending on what happened attack-wise last tick, then
		reset attack status for the upcoming tick. Don't carry out the effect
		of an attack if we're still interacting with the Agent we interacted
		with during the previous tick
		"""
		other = self.interact_agent
		if other is not None and other is not self.prev_interact_agent:
			# Update health
			did_attack = self.interact_attacked
			got_attacked = other.interact_attacked
			reward = 0
			if did_attack and got_attacked:
				# Defect-defect
				reward = -1
			elif not did_attack and got_attacked:
				# Cooperate-defect
				reward = -2
			elif did_attack and not got_attacked:
				# Defect-cooperate
				reward = 1
			self.health += reward * Agent._PD_HEALTH_MULTIPLIER
			# Store into memory
			self._remember_interaction(other, got_attacked)
			self.prev_interact_agent = other
			# Notify Model of event
			if not did_attack and not got_attacked:
				model.log_event("cc")
			elif did_attack and got_attacked:
				model.log_event("dd")
			else:
				model.log_event("cd")
		self.interact_agent = None

	def _update_hunger_sensor(self):
		""" Update hunger input neuron in Agent brain according to health """
		hunger = (100 - self.health) / 100.0
		hunger = util.clamp(hunger, -1, 1)
		self.brain.find_neuron("hunger").energy = hunger

	def _update_agent_sensors(self, world_agents):
		""" Allow Agents to "see" nearby Agents and map to the brain """
		rdn = Agent._SIGHT_ANGLE
		lngth = Agent._SIGHT_LENGTH
		rch = Agent._SIGHT_REACH
		sight_lft, a_lft = self._get_sensor_at(world_agents, -rdn, lngth, rch)
		sight_rght, a_rght = self._get_sensor_at(world_agents, rdn, lngth, rch)
		self.brain.find_neuron("agnt_lft").energy = sight_lft
		self.brain.find_neuron("agnt_rght").energy = sight_rght
		# Modify neurons based on the history of the nearest Agent
		if a_lft is not None and self._was_attacked_by(a_lft):
			self.brain.find_neuron("agnt_lft").energy += 0.5
		if a_rght is not None and self._was_attacked_by(a_rght):
			self.brain.find_neuron("agnt_rght").energy += 0.5

	def _update_food_sensors(self, world_food):
		""" Allow Agents to "smell" nearby food and map to the brain """
		rdn = Agent._SMELL_ANGLE
		lngth = Agent._SMELL_LENGTH
		rch = Agent._SMELL_REACH
		scent_lft, a_lft = self._get_sensor_at(world_food, -rdn, lngth, rch)
		scent_rght, a_rght = self._get_sensor_at(world_food, rdn, lngth, rch)
		self.brain.find_neuron("fd_lft").energy = scent_lft
		self.brain.find_neuron("fd_rght").energy = scent_rght

	def _update_movement_forward(self):
		""" Set desire to move forward by combining left/right movement neurons """
		# Get energy level from output neurons
		energy_left = self.brain.find_neuron("mv_lft").get_activation() / 2.0
		energy_right = self.brain.find_neuron("mv_rght").get_activation() / 2.0

		# Compute desire to move forward
		self.forward_force = (energy_left + energy_right) * Agent._FORWARD_SPEED
		self.forward_force = util.clamp(self.forward_force, 0,
										Agent._FORWARD_MAX)

	def _update_movement_turn(self):
		""" Set desire to turn by combining left/right movement neurons """
		# Get energy level from output neurons
		energy_left = self.brain.find_neuron("mv_lft").get_activation()
		energy_right = self.brain.find_neuron("mv_rght").get_activation()

		# Compute desire to turn
		self.turn_force = (energy_right - energy_left) * Agent._TURN_SPEED
		self.turn_force = util.clamp(self.turn_force, -Agent._TURN_MAX,
									 Agent._TURN_MAX)

	def _update_attack(self, world_agents):
		"""
		Allow Agents to attack other Agents and store the result of the
		interaction
		"""
		# If we've interacted with an Agent this tick then don't do so again
		if self.interact_agent is not None:
			return
		attack_dist_sqr = (self.radius * 2) ** 2
		for other in world_agents:
			# Can we interact with this Agent?
			if other is self or other.interact_agent is not None:
				continue
			dist_sqr = util.dist_sqr(self.get_pos(), other.get_pos())
			if dist_sqr > attack_dist_sqr:
				continue
			# Execute interaction between self and other
			self.interact_agent = other
			other.interact_agent = self
			if other is not self.prev_interact_agent:
				self.interact_attacked = self._will_attack()
				other.interact_attacked = other._will_attack()
			break
		if self.interact_agent is None:
			self.interact_attacked = False

	def _will_attack(self):
		""" Determine if this Agent will attack another in an interaction """
		prob = self.brain.find_neuron("atk").get_activation()
		prob = util.clamp(prob, 0, 1)
		return util.rand(0, 1) <= prob

	def _update_food(self, world_food):
		""" Allow Agents to eat Food objects """
		# Eat any piece of Food that the Agent has collided with
		for food in world_food:
			this_dist = util.dist(self.get_pos(), food.get_pos())
			max_dist = self.radius + food.radius
			if this_dist <= max_dist and food.is_alive():
				self.health += food.eat()
				return
		# If no food was eaten the Agent is hungry
		self.health -= (self.forward_force * Agent._HUNGER_MOVEMENT_RATIO +
						Agent._HUNGER_PER_TICK)

	def _get_sensor_at(self, actors, radians, length, reach):
		"""
		Compute the strength of an Actor sensor at some location away from the
		Agent at radians with distance out length and max sensing reach. Return
		tuple (sense, closest actor)
		"""
		# Compute sensor location
		length += self.radius
		sensor_x = self.x + math.cos(self.radians + radians) * length
		sensor_y = self.y + math.sin(self.radians + radians) * length
		pos = (sensor_x, sensor_y)
		# Compute sensor value
		sense = 0
		reach_sqr = reach ** 2
		closest = None
		closest_dist = 0
		for actor in actors:
			if actor is self:
				continue
			dist_sqr = util.dist_sqr(pos, actor.get_pos())
			if dist_sqr > reach_sqr:
				continue
			dist = math.sqrt(dist_sqr)
			if closest is None or dist < closest_dist:
				closest = actor
				closest_dist = dist
			ratio = (reach - dist) / float(reach)
			sense += ratio
		return (sense, closest)

	def _remember_interaction(self, other, other_attacked):
		""" Store the memory of an interaction with another Agent """
		key = id(other)
		self.memory[key] = other_attacked

	def _was_attacked_by(self, other):
		""" 
		Was this Agent attacked by other? If no previous encounters with other
		Agent, return false
		"""
		key = id(other)
		if not key in self.memory or self.memory[key] == False:
			return False
		return True


class Food(Actor):
	_ENERGY = 50
	_RADIUS = 5

	def __init__(self):
		Actor.__init__(self)
		self.radius = Food._RADIUS

	def eat(self):
		""" Called when an Agent eats a Food object. Returns energy gained """
		self.health = 0
		return Food._ENERGY
