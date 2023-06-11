import matplotlib.pyplot as plt
import networkx as nx


def plot_nn_graph(model):
    G = nx.DiGraph()
    input_size = model.input_size
    for i, module in enumerate(model.shared_repr):
        if isinstance(module, nn.Linear):
            G.add_node(f"Linear{i}", pos=(i, 0), color='red')
            if i == 0:
                G.add_edge(f"Input{input_size}", f"Linear{i}")
            else:
                G.add_edge(f"Linear{i - 1}", f"Linear{i}")
        elif isinstance(module, nn.ReLU):
            G.add_node(f"ReLU{i}", pos=(i, 1), color='blue')
            G.add_edge(f"Linear{i}", f"ReLU{i}")
    for i, module in enumerate(model.treatment_head):
        if isinstance(module, nn.Linear):
            G.add_node(f"Linear{i}T", pos=(i, 2), color='red')
            if i == 0:
                G.add_edge(f"ReLU{i}", f"Linear{i}T")
            else:
                G.add_edge(f"Linear{i - 1}T", f"Linear{i}T")
    for i, module in enumerate(model.control_head):
        if isinstance(module, nn.Linear):
            G.add_node(f"Linear{i}C", pos=(i, 3), color='red')
            if i == 0:
                G.add_edge(f"ReLU{i}", f"Linear{i}C")
            else:
                G.add_edge(f"Linear{i - 1}C", f"Linear{i}C")

    pos = nx.get_node_attributes(G, 'pos')
    color_map = nx.get_node_attributes(G, 'color')
    color_values = [color_map.get(node) for node in G.nodes()]
    nx.draw(G, pos, node_color=color_values, with_labels=True)
    plt.show()


# Define your model
input_size = 10
model = CFRModel(input_size, output_size)

# Plot the graph
plot_nn_graph(model)
