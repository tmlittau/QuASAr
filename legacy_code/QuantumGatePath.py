"""
Alternative gate-centric DAG representation for quantum circuits.

Each unique gate operation is represented by a single node.  Qubits are
stored as paths through these nodes.  If two qubits execute the same
sequence of gates, their paths share the same nodes which allows the
sequence to be evaluated only once.

This is in contrast to ``QuantumCircuitDAG`` where nodes represent the
state of qubit sets after each gate application and gates are duplicated
for every occurrence.  The new representation is more suitable for
identifying duplicated gate sequences.
"""

import itertools, math
from .utils import is_classical, flips_bit
from .Gates import gate_dispatcher, ControlledGate
from .simulation_methods import *

# global variables
viz_options = {
    "font_size": 72,
    "node_size": 1000,
    "node_color": "#ff6161",
    "alpha": 1,
    "edgecolors": "black",
    "arrowstyle":"->",
    "linewidths": 1,
    "width": 2,
    "with_labels": False,
    "arrows": True
}

simulation_dispatcher = {
    "statevec": sim_statevec,
    "mps": sim_mps,
    "dd": sim_dd,
    "tableau": sim_tableau
}

class GateNode:
    """Node representing a single gate operation."""

    _id_counter = itertools.count()

    def __init__(self, gate_name: str, params: dict | None = None):
        self.id = next(GateNode._id_counter)
        self.gate_name = gate_name
        self.params = params or {}

        # Edges to predecessor and successor nodes.  No acyclicity is assumed.
        self.prev_nodes: set["GateNode"] = set()
        self.next_nodes: set["GateNode"] = set()

    def __repr__(self) -> str:
        return f"GateNode(id={self.id}, gate={self.gate_name})"
    
class GatePathNode:
    """Node representing a qubit path in the quantum circuit graph."""
    
    _id_counter = itertools.count()

    def __init__(self, path: list[GateNode]):
        self.id = next(GatePathNode._id_counter)
        self.path: tuple[GateNode] = tuple(path)
        self.sim_method: str = "statevec"

        self.key: tuple[int, ...] = tuple(g.id for g in self.path)

        self.prev_nodes: set["GatePathNode"] = set()
        self.next_nodes: set["GatePathNode"] = set()

    def __repr__(self) -> str:
        return f"GatePathNode(id={self.id}, path={'->'.join([n.gate_name for n in self.path])})"
    
    def find_optimal_method(self):
        """Checks this path to find the optimal simulation method"""
        return "statevec"
    
class QubitMap:
    """Map from qubit sets to their path nodes in the circuit graph"""

    _id_counter = itertools.count()

    def __init__(self, qubits: list[int], pathnode: GatePathNode, qubit_app: list[tuple[int, ...]] | None = None):
        if len(qubits) == 0:
            raise ValueError("qubits cannot be empty!")

        self.id = next(QubitMap._id_counter)
        self.key: tuple[int] = tuple(sorted(qubits)) # Unique key for the qubit set
        self.qubits: set[int] = set(qubits)
        self.pathnode: tuple[int] = pathnode.key if pathnode else tuple()
        self.path = pathnode
        self.qubit_gate_map: list[tuple[int, ...]] = qubit_app if qubit_app is not None else [tuple(qubits)]

        self.prev_nodes: set["QubitMap"] = set()
        self.next_nodes: set["QubitMap"] = set()

    def __repr__(self) -> str:
        return f"QubitMap(id={self.id}, qubits={self.qubits}, pathnode={self.pathnode})"
    
    def simulate(self, t0: list[tuple[int]]):
        qc = {
            "n_qubits": len(self.qubits),
            "gates": []
        }

        for g, q in zip(self.path.path, self.qubit_gate_map):
            g_dict = {"gate": g.gate_name,"qubits": q}
            if g.params:
                g_dict["params"] = g.params
            qc["gates"].append(g_dict)

        return simulation_dispatcher[self.path.sim_method](qc, t0)

class QuantumGatePath:
    """Graph representation of a quantum circuit using gate-centric nodes."""

    def __init__(self, n_qubits: int | None = None, qc_dict: dict | None = None):
        if n_qubits is None:
            if qc_dict is None:
                raise ValueError("n_qubits or qc_dict must be provided")
            n_qubits = qc_dict.get("n_qubits")
        self.n_qubits = n_qubits

        # map from (gate_name, sorted params) -> Node, with INIT node already created
        self.gate_nodes: dict[tuple, GateNode] = {}
        self.gate_path_nodes: dict[tuple, GatePathNode] = {}
        init_gate_node = self._get_or_create_gate_node("INIT", {})
        init_path_node = self._get_or_create_gate_path_node((init_gate_node,))

        # store the QubitMaps
        self.maps: dict[tuple, QubitMap] = {}

        # for each qubit store its path as a list of nodes
        self.active_nodes: dict[int, QubitMap] = {}

        # Track whether qubits are classical and their classical bit value
        self.classical_flags: dict[int, bool] = {}
        self.classical_bits: dict[int, int] = {}

        for q in range(n_qubits):
            self.active_nodes[q] = QubitMap([q], init_path_node)
            self.maps[self.active_nodes[q].key] = self.active_nodes[q]
            self.classical_flags[q] = True
            self.classical_bits[q] = 0

        if qc_dict is not None:
            for gate_info in qc_dict.get("gates", []):
                self.add_gate(gate_info)

    #------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _gatenode_key(self, gate_name: str, params: dict) -> tuple:
        """Unique hashable key for a gate."""
        return (gate_name, tuple(sorted(params.items())))
    
    def _get_or_create_gate_node(self, gate_name: str, params:dict) -> GateNode:
        """Get existing node or create a new one."""
        key = self._gatenode_key(gate_name, params)
        if key not in self.gate_nodes:
            self.gate_nodes[key] = GateNode(gate_name, params)
        return self.gate_nodes[key]
    
    def _get_or_create_gate_path_node(self, path: tuple[GateNode, ...]) -> GatePathNode:
        """Get existing path node or create a new one."""
        key = tuple(g.id for g in path)
        if key not in self.gate_path_nodes:
            self.gate_path_nodes[key] = GatePathNode(path)

        return self.gate_path_nodes[key]
    
    def _get_or_create_qubit_map(self, qubits: list[int], gate_path_node: GatePathNode, qubit_app: list[int]) -> QubitMap:
        """Get existing QubitMap or create a new one."""
        key = tuple(sorted(qubits))
        if key not in self.maps:
            new_map = QubitMap(qubits, gate_path_node, qubit_app)
            self.maps[key] = new_map

        return self.maps[key]

    def _pathnode_map_exists(self, pathnode: GatePathNode) -> None:
        """Check if a path node exists in the QubitMaps and remove if it does not."""
        for qubit_map in self.maps.values():
            if qubit_map.pathnode == pathnode.key:
                return
        
        # If the path node does not exist, 
        # remove it from respective next and previous node
        for pn in pathnode.prev_nodes:
            pn.next_nodes.remove(pathnode)
        # remove it from the maps
        self.gate_path_nodes.pop(pathnode.key, None)
        
    #------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def add_gate(self, gate: dict) -> None:
        """Add a gate to the graph, updating paths and nodes."""
        gate_name = gate["gate"]
        qubits = gate.get("qubits", [])
        params = gate.get("params", {})

        # Simplify controlled gates when the control is classical
        if gate_name.startswith("C"):
            ctrl = qubits[0]
            if self.classical_flags[ctrl]:
                if self.classical_bits[ctrl] == 0:
                    # control qubit is |0> -> gate has no effect
                    return
                else:
                    # act as non-controlled gate on the target(s)
                    self.add_gate({"gate": gate_name[1:], "qubits": qubits[1:], "params": params})
                    return
                
        gate_node = self._get_or_create_gate_node(gate_name, params)

        # Get all qubits pre-existing in the corresponding QubitMaps
        q_subsystem: set[int] = set()
        prev_paths: set[GatePathNode] = set()

        for q in qubits:
            q_subsystem = q_subsystem | self.active_nodes[q].qubits
            prev_paths.add(self.gate_path_nodes[self.active_nodes[q].pathnode])
            self.gate_path_nodes[self.active_nodes[q].pathnode].path[-1].next_nodes.add(gate_node)
            gate_node.prev_nodes.add(self.gate_path_nodes[self.active_nodes[q].pathnode].path[-1])

        qubit_map = self._get_or_create_qubit_map(list(q_subsystem), None, qubits)

        if qubit_map.key != self.active_nodes[qubits[0]].key:
            gate_path_node = self._get_or_create_gate_path_node((gate_node,))
            qubit_map.pathnode = gate_path_node.key

            # If the qubit map changed, qubits have been entangled and they cannot be treated classical anymore
            for q in q_subsystem:
                self.active_nodes[q].next_nodes.add(qubit_map)
                self.gate_path_nodes[self.active_nodes[q].pathnode].next_nodes.add(gate_path_node)
                gate_path_node.prev_nodes.add(self.gate_path_nodes[self.active_nodes[q].pathnode])
                qubit_map.prev_nodes.add(self.active_nodes[q])
                self.active_nodes[q] = qubit_map
                self.classical_flags[q] = False
                self.classical_bits[q] = None

        else:
            # If the qubit map remained the same, just update the path node and gate node
            previous_path = self.gate_path_nodes.get(qubit_map.pathnode)
            gate_path_node = self._get_or_create_gate_path_node(previous_path.path + (gate_node,))
            qubit_map.pathnode = gate_path_node.key
            qubit_map.qubit_gate_map.append(tuple(qubits))

            # Add all previous path nodes to the new path node as predecessors
            for prev_node in previous_path.prev_nodes:
                gate_path_node.prev_nodes.add(prev_node)
                prev_node.next_nodes.add(gate_path_node)

            # Check if this operation keeps involved qubits (can only be one) classical
            if len(qubits) == 1:
                qubit = qubits[0]
                classical_operation = is_classical(gate_name) and self.classical_flags[qubit]
                if classical_operation and flips_bit(gate_name):
                    # If the gate flips a classical bit, change the classical bit value
                    self.classical_bits[qubit] ^= 1
                elif not classical_operation:
                    # If the gate is not classical, mark the qubit as non-classical and set bit value to None
                    self.classical_flags[qubit] = False
                    self.classical_bits[qubit] = None

            # Finally check if the previous path node exists in the maps
            self._pathnode_map_exists(previous_path)
            
    def topological_layers(self, nodes: list[GatePathNode] | list[QubitMap]):
        """
        Optional: Return layers of state nodes
        A simple BFS topological layering
        """

        # Count in-degrees
        in_degrees = {node: 0 for node in nodes}
        for node in nodes:
            for nxt in node.next_nodes:
                if nxt not in nodes:
                    continue
                in_degrees[nxt] += 1
        
        # Kahn's algorithm for topological order, but produce layers
        layer = [n for n in nodes if in_degrees[n] == 0]
        layers = []
        while layer:
            layers.append(layer)
            new_layer = []
            for node in layer:
                for nxt in node.next_nodes:
                    if nxt not in nodes:
                        print("next node not in all nodes")
                        continue
                    in_degrees[nxt] -= 1
                    if in_degrees[nxt] == 0:
                        new_layer.append(nxt)
            layer = new_layer
        return layers
    
    def visualize_qubit_map_graph(self, title: str | None = None) -> None:
        """Visualize the graph representing the qubit map evolution with networkx and matplotlib"""
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
            plt.rcParams['text.usetex'] = True
        except ImportError:
            raise ImportError("networkx and matplotlib required for visualization")
        
        layers:list[list[QubitMap]] = self.topological_layers(self.maps.values())
        
        G = nx.DiGraph()
        pos = {}

        for i, layer in enumerate(layers):
            # Add nodes with labels
            occupied = []
            for node in layer:
                label = f"{", ".join([f"$q_{q}$" for q in node.qubits])}"

                G.add_node(node.id, label=label)
                # set position based on qubit index and layer
                ypos = self.n_qubits - max(node.qubits)
                while ypos in occupied:
                    ypos = ypos + 1
                occupied.append(ypos)
                pos[node.id] = (i, ypos)

        # Add edges
        for node in self.maps.values():
            for nxt in node.next_nodes:
                G.add_edge(node.id, nxt.id)

        # Draw nodes
        nx.draw(G, pos, **viz_options)
        # Draw node labels separately
        labels = nx.get_node_attributes(G, 'label')
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)

        e_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=e_labels, font_size=8)

        if not title:
            title = "Quantum Circuit DAG"
        plt.title(title)
        plt.axis('off')
        plt.show()

    def visualize_path_graph(self, title: str | None = None) -> None:
        """Visualize the path graph with networkx and matplotlib"""
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
            plt.rcParams['text.usetex'] = True
        except ImportError:
            raise ImportError("networkx and matplotlib required for visualization")
        
        G = nx.DiGraph()
        pos = {}
        for node in self.gate_path_nodes.values():
            label = f"({', '.join(f'{gn.gate_name.replace("INIT", "$|0\\rangle$")}' for gn in node.path)})"
            G.add_node(node.id, label=label)
        # Add edges
        for node in self.gate_path_nodes.values():
            for nxt in node.next_nodes:
                G.add_edge(node.id, nxt.id)

        # Draw nodes
        pos = nx.spring_layout(G)
        nx.draw(G, pos, **viz_options)
        labels = nx.get_node_attributes(G, 'label')
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)

        if not title:
            title = "Quantum Circuit DAG"
        plt.title(title)
        plt.axis('off')
        plt.show()

    def visualize_gate_graph(self, title: str | None = None) -> None:
        """Visualize the gate graph with networkx and matplotlib"""
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
            plt.rcParams['text.usetex'] = True
        except ImportError:
            raise ImportError("networkx and matplotlib required for visualization")
        
        G = nx.DiGraph()
        for node in self.gate_nodes.values():
            if node.params:
                label = f"{node.gate_name}({', '.join(f'{k}={v}' for k, v in node.params.items())})"
                if "theta" in node.params:
                    theta = round(math.degrees(node.params.get("theta")))
                    label = f"${node.gate_name[:-1]}_{node.gate_name[-1]}$\n${theta}^\\circ$"
            else:
                label = node.gate_name if node.gate_name != "INIT" else "$|0\\rangle$"
            G.add_node(node.id, label=label)
        for node in self.gate_nodes.values():
            for nxt in node.next_nodes:
                G.add_edge(node.id, nxt.id)

        # pos = nx.spring_layout(G, seed=42)
        pos = nx.spring_layout(G)
        nx.draw(G, pos, **viz_options)
        labels = nx.get_node_attributes(G, 'label')
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
        if title:
            plt.title(title)
        else:
            plt.title("Quantum Gate Path Graph")
        plt.axis('off')
        plt.show()