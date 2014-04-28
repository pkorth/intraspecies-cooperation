import math
import model
import pygame
import sys
import util


class GraphicsApp:
	# Frames per second for each speed toggle
	_FPS_SLOW = 10
	_FPS_NORMAL = 40
	_FPS_FAST = 160
	_FPS_REALTIME = 0
	# Keycode for pygame left mouse button (pygame doesn't have one defined)
	_PYGAME_MOUSE_LEFT = 1

	def __init__(self, argc, argv):
		# Command line arguments
		if argc != 2 or int(argv[1]) <= 0:
			self._print_usage(argv)
			exit()
		self.max_generation = int(argv[1])
		# Graphics window
		self.size = (1024, 768)
		self.title = "Evolution of Cooperation"
		self.buffer = None
		# Misc
		self.fps = GraphicsApp._FPS_NORMAL
		self.is_running = False
		self.model = None

 	def _print_usage(self, argv):
 		""" Print command line argument info to standard output """
 		print "usage: python %s max_generation" % argv[0]

	def on_init(self):
		""" Initialize the pygame module """
		pygame.init()
		self.buffer = pygame.display.set_mode(self.size, pygame.HWSURFACE |
											  pygame.DOUBLEBUF)
		pygame.display.set_caption(self.title)

	def print_instructions(self):
		""" Print usage instructions to standard output """
		print "Controls:"
		print "   q   quit"
		print "   r   reset model"
		print "   1   slow speed"
		print "   2   normal speed"
		print "   3   fast speed"
		print "   4   real-time"
		print "Click on an Agent to view its neural network in real time"

	def on_execute(self):
		""" Run the simulation until the user quits or reach max_generation """
		self.is_running = True
		self.model = model.Model(self.size)
		self.focus_agent = None

		while(self.is_running):
			for event in pygame.event.get():
				self._on_event(event)
			self.model.on_tick()
			if self.fps > 0:
				self._on_render()
			if self.model.generation == self.max_generation + 1:
				self.is_running = False

	def on_exit(self):
		""" Quit out of the pygame module """
		pygame.quit()
		self.model.on_exit()

	def _on_event(self, event):
		""" Pygame event handler """
		if event.type == pygame.QUIT:
			self.is_running = False
		elif event.type == pygame.KEYDOWN:
			if event.key == pygame.K_r:
				# Start over with a fresh Model
				self.model = model.Model(self.size)
				self.focus_agent = None
			elif event.key == pygame.K_1:
				self.fps = GraphicsApp._FPS_SLOW
			elif event.key == pygame.K_2:
				self.fps = GraphicsApp._FPS_NORMAL
			elif event.key == pygame.K_3:
				self.fps = GraphicsApp._FPS_FAST
			elif event.key == pygame.K_4:
				self.fps = GraphicsApp._FPS_REALTIME
				# Render the screen one more time so the user knows the FPS
				self._on_render()
			elif event.key == pygame.K_q:
				self.is_running = False
		elif event.type == pygame.MOUSEBUTTONDOWN:
			if event.button == GraphicsApp._PYGAME_MOUSE_LEFT:
				self.focus_agent = None
				for agent in self.model.agents:
					dist = util.dist(event.pos, agent.get_pos())
					if dist <= agent.radius:
						self.focus_agent = agent
						break

	def _on_render(self):
		""" Render world and information to the screen """
		self._draw_background()
		self._draw_food()
		self._draw_agents()
		self._draw_info()
		self._draw_focus()
		# Show screen and slow down program
		pygame.display.flip()
		if self.fps > 0:
			pygame.time.Clock().tick(self.fps)

	def _draw_background(self):
		""" Clear the screen with white """
		white = pygame.Color(255,255,255)
		self.buffer.fill(white)

	def _draw_food(self):
		""" Draw all Food objects to the sceen """
		green = pygame.Color(0,200,0)
		for food in self.model.food:
			pygame.draw.circle(self.buffer, green,
							   util.int_tuple(food.get_pos()), food.radius, 0)

	def _draw_agents(self):
		""" Draw all Agent objects to the screen """
		blue = pygame.Color(100,100,200)
		black = pygame.Color(0,0,0)
		green = pygame.Color(0,255,0)
		red = pygame.Color(255,0,0)
		for agent in self.model.agents:
			health = agent.health / 100.0
			health = util.clamp(health, 0, 1)
			pos = util.int_tuple(agent.get_pos())
			radians = agent.radians
			radius = agent.radius
			# Draw a black line showing current heading
			line_p0 = agent.get_pos()
			line_r = radius * 1.5
			line_p1 = (line_p0[0] + math.cos(radians)*line_r,
					   line_p0[1] + math.sin(radians)*line_r)
			pygame.draw.line(self.buffer, black, line_p0, line_p1, 2)
			# Draw a circle for the body. Blue for normal, red for attacking
			col = blue
			if agent.interact_attacked:
				col = red
			pygame.draw.circle(self.buffer, col, pos, radius, 0)
			pygame.draw.circle(self.buffer, black, pos, radius, 1)
			# Draw a green health bar
			rect = (int(agent.x)-20, int(agent.y)-30, 40, 3)
			pygame.draw.rect(self.buffer, red, rect)
			rect = (int(agent.x)-20, int(agent.y)-30, int(40*health), 3)
			pygame.draw.rect(self.buffer, green, rect)

	def _draw_info(self):
		""" Draw simulation statistics to the screen """
		black = pygame.Color(0,0,0)
		font = pygame.font.Font(None, 24)
		# Generation and tick
		content = "generation: %d   tick: %d" % self.model.get_gen_tick()
		text = font.render(content, False, black)
		self.buffer.blit(text, (5, 5))
		# Simulation speed
		content = ""
		if self.fps == GraphicsApp._FPS_SLOW:
			content = "speed: slow"
		elif self.fps == GraphicsApp._FPS_NORMAL:
			content = "speed: normal"
		elif self.fps == GraphicsApp._FPS_FAST:
			content = "speed: fast"
		elif self.fps == GraphicsApp._FPS_REALTIME:
			content = "speed: real-time"
		text = font.render(content, False, black)
		self.buffer.blit(text, (5, 25))

	def _draw_focus(self):
		"""
		Draw the neural network of the currently-selected Agent (if any). Very
		hackish and would need to be updated if the structure of an Agent neural
		network were to change
		"""
		if self.focus_agent is None or not self.focus_agent.is_alive():
			self.focus_agent = None
			return

		black = pygame.Color(0,0,0)
		white = pygame.Color(255,255,255)
		agent = self.focus_agent

		# Draw white x on the agent being observed
		pos_00 = (int(agent.x) - 12, int(agent.y) - 12)
		pos_01 = (int(agent.x) + 12, int(agent.y) + 12)
		pos_10 = (int(agent.x) + 12, int(agent.y) - 12)
		pos_11 = (int(agent.x) - 12, int(agent.y) + 12)
		pygame.draw.line(self.buffer, white, pos_00, pos_01, 2)
		pygame.draw.line(self.buffer, white, pos_10, pos_11, 2)

		# Get normalized Neuron info from the Agent brain
		circle_col = []
		for i in range(len(agent.brain.neurons)):
			circle_col.append(agent.brain.neurons[i].energy)
		circle_pos = [(33,150), (66,150), (100,150), (133,150), (166,150),
					  (40,100), (80,100), (120,100), (160,100),
					  (50,50), (100,50), (150,50)
					 ]
		# Get normalized Synapse info from the Agent brain
		line_col = []
		for i in range(len(agent.brain.synapses)):
			line_col.append(agent.brain.synapses[i].weight)
		line_pos = [(0,5), (0,6), (1,5), (1,6),
					(2,6), (2,7),
					(3,7), (3,8), (4,7), (4,8),
					(5,9), (5,10), (6,9), (6,10),
					(7,10), (7,11), (8,10), (8,11)
				   ]

		# Sanity check to catch if the Agent neural network structure changes
		assert len(agent.brain.neurons) == len(circle_pos)
		assert len(agent.brain.synapses) == len(line_pos)

		# Create a semi-transparent black surface
		transparent = (255,0,255)
		surf = pygame.Surface((200,200))
		surf.fill(transparent)
		surf.set_colorkey(transparent)
		pygame.draw.rect(surf, black, (0,0,200,200))

		# Draw Synapse lines onto the surface
		for i in range(len(line_pos)):
			j, k = line_pos[i]
			col_r = 0 if line_col[i] > 0 else -line_col[i] * 255
			col_g = 0 if line_col[i] < 0 else line_col[i] * 255
			col_r = util.clamp(col_r, 0, 255)
			col_g = util.clamp(col_g, 0, 255)
			pos_0 = circle_pos[j]
			pos_1 = circle_pos[k]
			pygame.draw.line(surf, (col_r, col_g, 0), pos_0, pos_1, 2)
		# Draw Neuron circles onto the surface
		white = pygame.Color(255,255,255)
		for i in range(len(circle_pos)):
			col_r = 0 if circle_col[i] > 0 else -circle_col[i] * 255
			col_g = 0 if circle_col[i] < 0 else circle_col[i] * 255
			col_r = util.clamp(col_r, 0, 255)
			col_g = util.clamp(col_g, 0, 255)
			pygame.draw.circle(surf, (col_r, col_g, 0), circle_pos[i], 10)
			pygame.draw.circle(surf, white, circle_pos[i], 10, 1)
		# Draw Agent generation
		font = pygame.font.Font(None, 16)
		content = "generation: %d" % agent.generation
		text = font.render(content, False, white)
		surf.blit(text, (5, 5))

		# Blit surface onto screen (opaque == 255)
		surf.set_alpha(180)
		self.buffer.blit(surf, (5 ,self.size[1] - 205))


if __name__ == "__main__" :
	app_instance = GraphicsApp(len(sys.argv), sys.argv)
	app_instance.on_init()
	app_instance.print_instructions()
	app_instance.on_execute()
	app_instance.on_exit()
