import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

adjacency = {
    'root': ['Bacilllariophytina', 'Coscinodiscophyceae'],

    'Bacilllariophytina': ['Bacilliorhycaea', 'Mediophycaea'],
    'Coscinodiscophyceae': ['Corethon', 'Rhizosoleniaceae'],

    'Bacilliorhycaea': ['Bacillariaceae', 'Thalassiomema'],
    'Mediophycaea': ['Chaetocerothophycidae', 'Thalassiosirophycidae'],

    'Chaetocerothophycidae': ['Chaetocerotales', 'Hemiaulaceae'],
    'Thalassiosirophycidae': ['Ditylum', 'Thalassiosoreles'],

    'Chaetocerotales': ['Chaetoceros', 'Leptocylindrus'],
    'Hemiaulaceae': ['Cerataulina', 'Eucampia'],
    'Thalassiosoreles': ['Skeletonema', 'Thalassiosira'],

    'Bacillariaceae': ['Cylindrotheca', 'Pseudonitzchia'],
    'Rhizosoleniaceae': ['Dactyliosolem', 'Guinardia', 'Rhizosolenia'],

    'Corethon': [],
    'Thalassiomema': [],
    'Cylindrotheca': [],
    'Pseudonitzchia': [],
    'Chaetoceros': [],
    'Leptocylindrus': [],
    'Cerataulina': [],
    'Eucampia': [],
    'Ditylum': [],
    'Skeletonema': [],
    'Thalassiosira': [],
    'Dactyliosolem': [],
    'Guinardia': [],
    'Rhizosolenia': [],
}

leaves = {n for n, children in adjacency.items() if not children}

# Build directed graph
G = nx.DiGraph()
for parent, children in adjacency.items():
    G.add_node(parent)
    for child in children:
        G.add_edge(parent, child)

# Hierarchical layout: assign depth via BFS, then spread x evenly per level
def hierarchy_pos(G, root, width=1.0, vert_gap=1.0):
    pos = {}
    level_nodes = {}

    queue = [(root, 0)]
    visited = set()
    while queue:
        node, depth = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)
        level_nodes.setdefault(depth, []).append(node)
        for child in G.successors(node):
            if child not in visited:
                queue.append((child, depth + 1))

    for depth, nodes in level_nodes.items():
        for i, node in enumerate(nodes):
            x = (i - (len(nodes) - 1) / 2) * (width / max(len(nodes), 1))
            pos[node] = (x, -depth * vert_gap)

    return pos

pos = hierarchy_pos(G, 'root', width=26.0, vert_gap=1.8)

# Shift leaf nodes down by half a gap so they visually sit at their own level
for node in leaves:
    x, y = pos[node]
    pos[node] = (x, y)

node_colors = ['#a8d5a2' if n in leaves else '#4a7fb5' for n in G.nodes()]
node_shapes_label_color = ['#2c2c2c' if n in leaves else 'white' for n in G.nodes()]

fig, ax = plt.subplots(figsize=(28, 14))

# Draw edges
nx.draw_networkx_edges(
    G, pos, ax=ax,
    arrows=True, arrowsize=12,
    edge_color='#999999', width=1.2,
    connectionstyle='arc3,rad=0.0'
)

# Draw internal nodes
internal_nodes = [n for n in G.nodes() if n not in leaves]
nx.draw_networkx_nodes(
    G, pos, nodelist=internal_nodes, ax=ax,
    node_color='#4a7fb5', node_size=2200, node_shape='s'
)

# Draw leaf nodes
nx.draw_networkx_nodes(
    G, pos, nodelist=list(leaves), ax=ax,
    node_color='#a8d5a2', node_size=2200, node_shape='s'
)

# Labels — wrap long names
def wrap(name, max_len=14):
    if len(name) <= max_len:
        return name
    mid = len(name) // 2
    # Find nearest space or split point near middle
    for i in range(mid, len(name)):
        if name[i] == '_':
            return name[:i] + '\n' + name[i+1:]
    return name[:mid] + '-\n' + name[mid:]

labels = {n: wrap(n) for n in G.nodes()}
for node, (x, y) in pos.items():
    color = 'white' if node not in leaves else '#2c2c2c'
    ax.text(x, y, labels[node], ha='center', va='center',
            fontsize=7.5, fontweight='bold', color=color,
            multialignment='center')

# Level annotations on the left
level_nodes = {}
for node, (x, y) in pos.items():
    level = round(-y / 1.8)
    level_nodes.setdefault(level, [])
    level_nodes[level].append(y)

for level, ys in sorted(level_nodes.items()):
    avg_y = sum(ys) / len(ys)
    ax.text(-14.2, avg_y, f'Level {level}', ha='right', va='center',
            fontsize=9, color='#555555', style='italic')

# Legend
legend_handles = [
    mpatches.Patch(color='#4a7fb5', label='Internal node'),
    mpatches.Patch(color='#a8d5a2', label='Leaf node'),
]
ax.legend(handles=legend_handles, loc='lower right', fontsize=10, framealpha=0.9)

ax.set_title('WHOI Hierarchy — whoi_adjacency_graph_l', fontsize=14, fontweight='bold', pad=16)
ax.axis('off')
plt.tight_layout()

out = 'whoi_hierarchy.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved → {out}')
plt.show()
