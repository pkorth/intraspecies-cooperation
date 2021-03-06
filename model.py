import actors
import random


class Model:
	# Count of Agents per generation and how many are used to produce the next
	_AGENT_COUNT = 20
	_SURVIVOR_PERCENT = 0.25
	# Number of Food objects in the simulation per Agent object
	_FOOD_PER_AGENT = 1.0

	def __init__(self, size):
		self.size = size[:]
		actors.Actor.set_world_size(self.size[:])
		self.agents = []
		self.food = []
		self.tick = self.generation = 0
		self._log_lifetime = [0]
		self._log_cc = [0]
		self._log_cd = [0]
		self._log_dd = [0]
		self._log_event = [0]

	def on_tick(self):
		"""
		Called once every step of the simulation. Update Agents, Food, and
		internal state
		"""
		self.tick += 1
		for agent in self.agents:
			if agent.is_alive():
				agent.on_tick(self.agents, self.food)
		for agent in self.agents:
			if agent.is_alive():
				agent.process_attacks(self)
		for food in self.food:
			if food.is_alive():
				food.on_tick()
		self._update_world()

	def on_exit(self):
		""" Called when main application is closing """
		# Get rid of extra data on front and back in model logs
		self._log_lifetime = self._log_lifetime[1:-1]
		self._log_cc = self._log_cc[1:-1]
		self._log_cd = self._log_cd[1:-1]
		self._log_dd = self._log_dd[1:-1]
		self._log_event = self._log_event[1:-1]
		# Output results
		print "---> Model parameters"
		print "World size:         (%d,%d)" % self.size
		print "Agents:             %d" % Model._AGENT_COUNT
		print "Survivor percent:   %.2f" % Model._SURVIVOR_PERCENT
		print "Food per Agent:     %.2f" % Model._FOOD_PER_AGENT
		print "---> Model results"
		print "Final generation:   %d" % (self.generation - 1)
		print "Lifetimes:          [%s]" % ','.join([str(x) for x in
													 self._log_lifetime])
		print "All events:         [%s]" % ','.join([str(x) for x in
													 self._log_event])
		print "C-C events:         [%s]" % ','.join([str(x) for x in
													 self._log_cc])
		print "C-D events:         [%s]" % ','.join([str(x) for x in
													 self._log_cd])
		print "D-D events:         [%s]" % ','.join([str(x) for x in
													 self._log_dd])

		print "--->  Configuration of random living Agent"
		if len(self.agents) > 0:
			print random.choice(self.agents).brain.pretty_print()
		else:
			print "No current Agents"

	def get_gen_tick(self):
		""" Return a tuple with current (generation, tick) """
		return (self.generation, self.tick)

	def log_event(self, kind):
		""" Log an event of some sort to be saved in Model results """
		if kind == "cc":
			self._log_cc[self.generation] += 1
		elif kind == "cd":
			self._log_cd[self.generation] += 1
		elif kind == "dd":
			self._log_dd[self.generation] += 1

	def _create_initial_gen(self):
		""" Create an initial population of Agents """
		next_gen = []
		for i in range(Model._AGENT_COUNT):
			brain = actors.Agent.create_random_brain()
			child = actors.Agent(self.generation, brain)
			next_gen.append(child)
		self.agents[:] = next_gen

	def _create_next_gen(self):
		""" Take remaining Agents and create the next Agent generation """
		next_gen = []
		# Ensure that every remaining Agent is in the next generation and that
		# each spawns a descendent (which will have random genetic mutations)
		for parent in self.agents:
			brain = parent.brain.make_copy()
			actors.Agent.mutate_brain(brain)
			child = actors.Agent(self.generation, brain)
			next_gen.append(child)
			parent.reset()
			next_gen.append(parent)
		# Fill in any remaining spots with Agents bred from random parents
		while len(next_gen) < Model._AGENT_COUNT:
			parent_1_brain = random.choice(self.agents).brain
			parent_2_brain = random.choice(self.agents).brain
			brain = actors.Agent.breed_brain(parent_1_brain, parent_2_brain)
			actors.Agent.mutate_brain(brain)
			child = actors.Agent(self.generation, brain)
			next_gen.append(child)
		self.agents[:] = next_gen

	def _create_initial_food(self):
		""" Create an initial population of Food objects """
		next_gen = []
		for i in range(int(Model._AGENT_COUNT * Model._FOOD_PER_AGENT)):
			food = actors.Food()
			next_gen.append(food)
		self.food[:] = next_gen

	def _start_next_generation(self):
		"""
		Start the next generation of Agents and Food either from nothing or from
		a pre-existing state
		"""
		if self.generation == 0:
			self._create_initial_gen()
		else:
			self._log_lifetime[self.generation] = self.tick
			self._log_event[self.generation] = (self._log_cc[self.generation] +
												self._log_cd[self.generation] +
												self._log_dd[self.generation])
			self._create_next_gen()
		self._create_initial_food()
		self.generation += 1
		self.tick = 0
		self._log_lifetime.append(0)
		self._log_cc.append(0)
		self._log_cd.append(0)
		self._log_dd.append(0)
		self._log_event.append(0)

	def _update_world(self):
		"""
		Remove dead Agents and Food, then check if we should advance to the next
		generation of Agents
		"""
		# Only remove one Agent per tick to avoid them all dying at once
		for agent in self.agents:
			if not agent.is_alive():
				self.agents.remove(agent)
				break
		# Replace Food objects as they are eaten
		for food in self.food[:]:
			if not food.is_alive():
				self.food.remove(food)
				self.food.append(actors.Food())
		# Do we need to start the next generation?
		if len(self.agents) <= Model._AGENT_COUNT * Model._SURVIVOR_PERCENT:
			self._start_next_generation()
