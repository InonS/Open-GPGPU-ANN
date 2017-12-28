from matplotlib.pyplot import plot, show
from nengo import Connection, Ensemble, Network, Node, Probe
# from nengo.utils.simulator import operator_dependency_graph
from nengo_dl import Simulator
from numpy import sin

# define the model
with Network() as model:
    stim = Node(sin)
    a = Ensemble(100, 1)
    b = Ensemble(100, 1)
    Connection(stim, a)
    Connection(a, b, function=lambda x: x ** 2)

    probe_a = Probe(a, synapse=0.01)
    probe_b = Probe(b, synapse=0.01)

# build and run the model
with Simulator(model) as sim:
    sim.run(10)

# plot the results
plot(sim.trange(), sim.data[probe_a])
plot(sim.trange(), sim.data[probe_b])
show()
