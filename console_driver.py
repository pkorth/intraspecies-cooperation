import math
import model
import sys
import util


class ConsoleApp:
	def __init__(self, argc, argv):
		# Command line arguments
		if argc != 2 or int(argv[1]) <= 0:
			self.print_usage(argv)
			exit()
		self.max_generation = int(argv[1])
		self.is_running = False
		self.size = (1024, 768)
		self.model = None
 
 	def print_usage(self, argv):
 		""" Print command line argument info to standard output """
 		print "usage: python %s max_generation" % argv[0]

	def on_init(self):
		""" Nothing to do """
		pass

	def on_execute(self):
		""" Run the simulation until the user quits or reach max_generation """
		self.is_running = True
		self.model = model.Model(self.size)

		while(self.is_running):
			self.model.on_tick()
			if self.model.generation == self.max_generation + 1:
				self.is_running = False

	def on_exit(self):
		""" Model outputs results """
		self.model.on_exit()


if __name__ == "__main__" :
	app_instance = ConsoleApp(len(sys.argv), sys.argv)
	app_instance.on_init()
	app_instance.on_execute()
	app_instance.on_exit()
