from pyvis.network import Network
import networkx as nx

G = nx.Graph()
G.add_edges_from([(1, 2), (1, 3), (2, 4)])

net = Network(notebook=True)
net.from_nx(G)
net.show("pyvis_networkx.html")
