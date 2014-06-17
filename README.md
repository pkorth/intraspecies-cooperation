#Intraspecies Cooperation
An agent-based model designed to investigate the evolution of intraspecies cooperation, written in Python.  Seeks to optimize a feed-forward neural network selected for by a simplistic genetic algorithm.

Users with [pygame](http://www.pygame.org) installed may run `gfx_driver.py` to watch the simulation in real-time; otherwise, `console_driver.py` will run the same model but without the accompanying graphical display. Both will output statistics at the conclusion of the simulation.

## Overview
The 2D world is made up of agents and particles of food. An agent can move, attack other agents, smell nearby food, see nearby agents, tell when it's hungry, and eat food particles. Agents gradually lose health due to hunger and are hurt when attacked; eating food replenishes this health. Should an agent's health run out, the agent will die.

Each agent in the simulation represents a unique genetic code akin to DNA. An agent's genetic code is expressed in the brain of the agent, specifying the strength of each of its 18 synapses. The actions taken by an agent are controlled by the agent's senses and brain, thus an agent's genetic code influences how it will behave.

The simulation is split up into a series of distinct trials. The first generation starts with a collection of randomly-generated agents; they have randomly-generated brains and so will not survive long. Agents compete for limited food and can attack one another until only a select number of agents remain. These agents breed to create the next generation of agents, and the next trial begins.

The theory: over time, "bad" genetic code that causes poor survival behavior will be selected against and so the population of agents will gradually evolve to live longer. An agents with "bad" genetic code is likely to die out sooner than an agent with fitter genetic code, thus the fitter agent will live on to breed and create the next generation.

## Code
Results of the simulated natural selection depend both on pseudo-random chance and the characteristics of the environment. All model parameters can be found at the top of the `Agent`, `Food`, and `Model` classes; world size is found in the `__init__()` function of your chosen driver (`GraphicsApp` or `ConsoleApp`).

## License
Intraspecies Cooperation is licensed under the [MIT license](https://github.com/pkorth/intraspecies-cooperation/blob/master/LICENSE).
