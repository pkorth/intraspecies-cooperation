import math
import util


class NeuralNetwork:
	def __init__(self):
		self.neurons = []
		self.synapses = []

	def make_copy(self):
		""" Create and return a deep-copy of this NeuralNetwork """
		copy = NeuralNetwork()
		for neuron in self.neurons:
			copy.add_neuron(neuron.make_copy())
		for synapse in self.synapses:
			src = copy.find_neuron(synapse.src.name)
			dest = copy.find_neuron(synapse.dest.name)
			copy.add_synapse(Synapse(src, dest, synapse.weight))
		return copy

	def add_neuron(self, n):
		self.neurons.append(n)

	def add_synapse(self, s):
		self.synapses.append(s)

	def find_neuron(self, name):
		""" Find a Neuron by name or None if not found """
		for neuron in self.neurons:
			if neuron.name == name:
				return neuron
		return None

	def update(self):
		""" Update all components """
		for synapse in self.synapses:
			synapse.update_energy()
		for neuron in self.neurons:
			neuron.clear_energy()
		for synapse in self.synapses:
			synapse.transfer_energy()

	def reset(self):
		""" Reset all components to initial state """
		for synapse in self.synapses:
			synapse.reset()
		for neuron in self.neurons:
			neuron.reset()

	def serialize(self):
		"""
		Generate and return a string representation of all components. Format:
		[n0.name:n0.is_input,...][s0.weight:s0.src:s0.dest,...]
		"""
		neuron_str = ["%s:%d"%(n.name, n.is_input) for n in self.neurons]
		synapse_str = ["%f:%s:%s"%(s.weight, s.src.name, s.dest.name)
					   for s in self.synapses]
		output = ""
		output += "[" + ",".join(neuron_str) + "]"
		output += "[" + ",".join(synapse_str) + "]"
		return output

	def deserialize(self, serial):
		"""
		Clear all contents and replace with serialized content from input.
		Assumes correct input format but notifies user if this is not the case
		"""
		# Construct new NeuralNetwork using content then swap out to self
		temp_nnet = NeuralNetwork()
		try:
			neuron_str = serial[serial.index('[')+1:serial.index(']')]
			neuron_data = neuron_str.split(',')
			synapse_str = serial[serial.rindex('[')+1:serial.rindex(']')]
			synapse_data = synapse_str.split(',')

			for data in neuron_data:
				name, is_input = data.split(':')
				temp_nnet.add_neuron(Neuron(name, bool(is_input)))
			for data in synapse_data:
				weight, src_name, dest_name = data.split(':')
				src = temp_nnet.find_neuron(src_name)
				dest = temp_nnet.find_neuron(dest_name)
				temp_nnet.add_synapse(Synapse(float(weight), src, dest))
		except:
			print "NeuralNetwork.deserialize() exception: bad input format"
			raise
		# Replace old content with new
		self.neurons[:] = temp_nnet.neurons
		self.synapses[:] = temp_nnet.synapses

	def pretty_print(self):
		"""
		Generate and return a human-readable representation of all components.
		Similar to serialize() but the output is designed to be easier to read
		"""
		synapse_str = ["%s * %3.2f -> %s"%(s.src.name, s.weight, s.dest.name)
					   for s in self.synapses]
		output = ", ".join(synapse_str)
		return output

class Neuron:
	def __init__(self, name, is_input = False):
		self.name = name
		self.is_input = is_input
		self.energy = 0

	def make_copy(self):
		""" Create and return a deep-copy of this Neuron """
		copy = Neuron(str(self.name), self.is_input)
		return copy

	def clear_energy(self):
		""" Reset the energy of this Neuron except if it gets external input """
		if not self.is_input:
			self.energy = 0

	def get_activation(self):
		"""
		Get activiation energy of this Neuron. The proper way to do this is
		return util.sigmoid(self.energy) - 0.5
		but using math.sqrt() is giving good results at this time
		"""
		return math.sqrt(abs(self.energy)) * util.sgn(self.energy)

	def reset(self):
		""" Hard-reset in the event of a NeuralNetwork reset """
		self.energy = 0


class Synapse:
	def __init__(self, src, dest, weight):
		self.src = src
		self.dest = dest
		self.weight = weight
		self.energy = 0

	def update_energy(self):
		""" Compute energy level from source Neuron """
		self.energy = self.src.get_activation() * self.weight
 
	def transfer_energy(self):
		""" Add energy to destination Neuron """
		self.dest.energy += self.energy

	def reset(self):
		""" Hard-reset in the event of a NeuralNetwork reset """
		self.energy = 0
