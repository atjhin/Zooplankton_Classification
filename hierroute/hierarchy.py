class Node:
    def __init__(self, node_id, name, parent=None):
        self.node_id = node_id
        self.name = name
        self.parent = parent
        self.children = []
        self.depth = None

    def add_child(self, child_id):
        """Add a child node ID to this node's children list."""
        if child_id not in self.children:
            self.children.append(child_id)
    
    def __repr__(self):
        return f"Node(id={self.node_id}, name={self.name}, parent={self.parent}, children={self.children})"


class Hierarchy:
    def __init__(self):
        self.nodes = {}
        self.root = None

    def __len__(self):
        return len(self.nodes.keys())

    def add_node(self, node_id, name, parent=None):
        self.nodes[node_id] = Node(node_id, name, parent)
        self.nodes[node_id].depth = self.depth(node_id)
        if parent is not None:
            self.nodes[parent].add_child(node_id)
        else:
            self.root = node_id

    def children(self, node_id):
        return self.nodes[node_id].children

    def parent(self, node_id):
        return self.nodes[node_id].parent

    def is_leaf(self, node_id):
        return len(self.nodes[node_id].children) == 0

    def depth(self, node_id):
        d = 0
        cur = node_id
        while self.nodes[cur].parent is not None:
            d += 1
            cur = self.nodes[cur].parent
        return d

    def get_path_to_root(self, leaf_id):
        """
        Get the path from root to a leaf node.
        
        Args:
            leaf_id: The ID of the leaf node (or any node in the hierarchy)
        
        Returns:
            List of node IDs from root to leaf, e.g., [root_id, ..., leaf_id]
        """
        if leaf_id not in self.nodes:
            raise ValueError(f"Node {leaf_id} not found in hierarchy")
        
        path = []
        current_id = leaf_id
        
        # Traverse from leaf to root using parent pointers
        while current_id is not None:
            path.append(self.nodes[current_id].node_id)
            current_id = self.nodes[current_id].parent
        
        # Reverse to get path from root to leaf
        return list(reversed(path))

    def descendants(self, node_id):
        result = []

        def dfs(n):
            for c in self.nodes[n].children:
                result.append(c)
                dfs(c)

        dfs(node_id)
        return result

    def subtree_leaves(self, node_id):
        leaves = []

        def dfs(n):
            if self.is_leaf(n):
                leaves.append(n)
            for c in self.nodes[n].children:
                dfs(c)

        dfs(node_id)
        return leaves

    def get_leaf_index(self):
        """
        Returns a list where each position corresponds to a node (ordered by node_id),
        with 1 if the node is a leaf, 0 otherwise.
        
        e.g. if nodes 5 and 6 are leaves → [0, 0, 0, 0, 0, 1, 1]
        """
        sorted_nodes = sorted(self.nodes.values(), key=lambda n: n.node_id)
        return [1 if self.is_leaf(n.node_id) else 0 for n in sorted_nodes]