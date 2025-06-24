import json

import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go

def convert_to_int(col):
    col.fillna(0, inplace=True)
    col = col.astype(int)
    col = col.replace(0, None)
    return col

def pass_to_notes(df, col, text):
    try:
        df['observacoes'].fillna('', inplace=True)
        df.loc[df[col].notnull(), 'observacoes'] = df['observacoes'] + '\n' + text + df[col]
        df.drop(columns=[col], inplace=True)
    except KeyError:
        df['notes'].fillna('', inplace=True)
        df.loc[df[col].notnull(), 'notes'] = df['notes'] + '\n' + text + df[col]
        df.drop(columns=[col], inplace=True)

def create_map(table, col):
    col_map = table.set_index('id')[col].to_dict()
    return col_map

def create_network_from_ids(df, id_col='id', parent_col='parentID', name_col='nome'):
    """Create a NetworkX graph from a dataframe with id and parentID columns."""
    G = nx.DiGraph()
    id_to_name = dict(zip(df[id_col], df[name_col]))

    # Add all nodes with their names as attributes
    for _, row in df.iterrows():
        G.add_node(row[id_col], label=row[name_col])

    # Add edges from parent to child
    for _, row in df.iterrows():
        child_id = row[id_col]
        parent_id = row[parent_col]

        if pd.notna(parent_id):
            G.add_edge(parent_id, child_id)

    return G, id_to_name

def hierarchy_pos_horizontal(G, root=None, width=1., vert_gap=1.5, horiz_gap=2.0, xcenter=0):
    """
    Create a horizontal (left-to-right) hierarchical layout for tree-like graphs.
    Increased spacing to prevent label overlap.
    """
    if root is None:
        roots = [n for n in G.nodes() if G.in_degree(n) == 0]
        root = roots[0] if roots else list(G.nodes())[0]

    def _hierarchy_pos(G, root, width=1., vert_gap=1.5, horiz_gap=2.0, xcenter=0, pos=None, parent=None, level=0):
        if pos is None:
            pos = {root: (level * horiz_gap, xcenter)}
        else:
            pos[root] = (level * horiz_gap, xcenter)

        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)

        if len(children) != 0:
            dy = width / len(children) if len(children) > 1 else 0
            nexty = xcenter - width/2 - dy/2 if len(children) > 1 else xcenter
            for child in children:
                if len(children) > 1:
                    nexty += dy
                else:
                    nexty = xcenter
                pos = _hierarchy_pos(G, child, width=width*vert_gap, vert_gap=vert_gap,
                                   horiz_gap=horiz_gap, xcenter=nexty,
                                   pos=pos, parent=root, level=level+1)
        return pos

    return _hierarchy_pos(G, root, width, vert_gap, horiz_gap, xcenter)

def create_interactive_network_horizontal(G, id_to_name, title="", height=5500):
    """
    Create a horizontal interactive network visualization using Plotly.
    Labels positioned to avoid overlap.
    """
    # Create horizontal hierarchical layout
    pos = hierarchy_pos_horizontal(G, width=300.0, vert_gap=0.5, horiz_gap=100.0)

    # Extract node and edge information
    node_x = []
    node_y = []
    node_text = []
    node_info = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(id_to_name.get(node, str(node)))

        adjacencies = list(G.neighbors(node))
        predecessors = list(G.predecessors(node))

        # Map adjacent node IDs to their names
        connected_names = [id_to_name.get(adj, str(adj)) for adj in adjacencies]

        node_info.append(
            f'<b>{id_to_name.get(node, str(node))}</b><br>' +
            f'ID: {node}<br>' +
            f'Children: {len(adjacencies)}<br>' +
            f'Parents: {len(predecessors)}<br>' +
            f'Connections: {", ".join(connected_names) if connected_names else "None"}'
        )
    # Create edges
    edge_x = []
    edge_y = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        mode='lines',
        showlegend=False
    )

    # Create node trace (circles only, no text here)
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        hovertext=node_info,
        marker=dict(
            size=20,
            color='lightblue',
            line=dict(width=2, color='DarkSlateGrey')
        ),
        showlegend=False
    )

    # Create separate text trace for labels (positioned to avoid overlap)
    text_x = []
    text_y = []
    text_labels = []

    for node in G.nodes():
        x, y = pos[node]
        # Position text to the right of the node to avoid overlap
        text_x.append(x)  # Offset text to the right
        text_y.append(y)
        text_labels.append(id_to_name.get(node, str(node)))

    text_trace = go.Scatter(
        x=text_x, y=text_y,
        mode='text',
        text=text_labels,
        textposition='middle right',
        textfont=dict(size=12, color='black'),
        hoverinfo='none',
        showlegend=False
    )

    # Create figure
    fig = go.Figure(
        data=[edge_trace, node_trace, text_trace],
        layout=go.Layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=16)
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=40, l=40, r=40, t=60),
            annotations=[
                dict(
                    text="Drag to pan, scroll to zoom, hover nodes for details",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(color="grey", size=12)
                )
            ],
            xaxis=dict(
                showgrid=True,
                gridcolor='lightgray',
                zeroline=False,
                showticklabels=False,
                title="Hierarchy Level"
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='lightgray',
                zeroline=False,
                showticklabels=False
            ),
            plot_bgcolor='white',
            # Set aspect ratio to prevent stretching
            width=5000,
            height=height
        )
    )

    return fig

def create_compact_horizontal_network(G, id_to_name, title="Compact Horizontal Network"):
    """
    Create a more compact horizontal layout with better text positioning
    """
    pos = hierarchy_pos_horizontal(G, width=6.0, vert_gap=2, horiz_gap=2.0)

    # Calculate text positioning to avoid overlaps
    node_positions = [(pos[node][0], pos[node][1], node) for node in G.nodes()]

    # Extract coordinates
    node_x = [pos[0] for pos in node_positions]
    node_y = [pos[1] for pos in node_positions]
    nodes = [pos[2] for pos in node_positions]

    # Create hover text
    hover_text = []
    for node in nodes:
        adjacencies = list(G.neighbors(node))
        predecessors = list(G.predecessors(node))
        hover_text.append(
            f'<b>{id_to_name.get(node, str(node))}</b><br>' +
            f'ID: {node}<br>' +
            f'Level: {int(pos[node][0] / 4)}<br>' +
            f'Children: {len(adjacencies)}<br>' +
            f'Parents: {len(predecessors)}'
        )

    # Create edges
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    # Edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='rgba(50,50,50,0.5)'),
        hoverinfo='none',
        mode='lines',
        showlegend=False
    )

    # Node trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        marker=dict(
            size=25,
            color='lightblue',
            line=dict(width=2, color='navy'),
            opacity=0.8
        ),
        hoverinfo='text',
        hovertext=hover_text,
        showlegend=False
    )

    # Text annotations (positioned carefully)
    annotations = []
    for i, node in enumerate(nodes):
        name = id_to_name.get(node, str(node))
        # Truncate long names for display
        display_name = name if len(name) <= 15 else name[:12] + "..."

        annotations.append(
            dict(
                x=node_x[i] + 0.4,  # Position text to the right of node
                y=node_y[i],
                text=display_name,
                showarrow=False,
                font=dict(size=11, color="black"),
                xanchor="left",
                yanchor="middle"
            )
        )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(text=title, x=0.5, font=dict(size=16)),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=40, l=40, r=40, t=60),
            annotations=annotations,
            xaxis=dict(
                showgrid=True,
                gridcolor='lightgray',
                zeroline=False,
                showticklabels=True,
                title="Hierarchy Level"
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='lightgray',
                zeroline=False,
                showticklabels=False
            ),
            plot_bgcolor='white',
            width=1400,
            height=900
        )
    )

    return fig

def analyze_network(G):
    """Print basic network statistics."""
    print("Network Analysis:")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Is directed: {G.is_directed()}")
    print(f"Is connected: {nx.is_weakly_connected(G)}")

    roots = [n for n in G.nodes() if G.in_degree(n) == 0]
    print(f"Root nodes: {roots}")

    leaves = [n for n in G.nodes() if G.out_degree(n) == 0]
    print(f"Leaf nodes: {leaves}")

    # Calculate maximum depth
    if roots:
        max_depth = 0
        for root in roots:
            for node in G.nodes():
                try:
                    depth = nx.shortest_path_length(G, root, node)
                    max_depth = max(max_depth, depth)
                except nx.NetworkXNoPath:
                    pass
        print(f"Maximum depth: {max_depth}")

with open('data/funcaoThesaurusHORIZ.json', 'r') as f:
    out = json.load(f)
thesaurus_horiz = pd.DataFrame(out['content'])
num_cols = ['id', 'HparentID', 'VparentID', 'ExprID']
for col in num_cols:
    thesaurus_horiz[col] = convert_to_int(thesaurus_horiz[col])

with open('data/funcaoThesaurusVERT.json', 'r') as f:
    out = json.load(f)
thesaurus_vert = pd.DataFrame(out['content'])
num_cols = ['id', 'parentID']
for col in num_cols:
    thesaurus_vert[col] = convert_to_int(thesaurus_vert[col])


# Create the network
print("Creating network from dataframe...")
G, id_to_name = create_network_from_ids(thesaurus_vert)

# Keep only nodes that are connected to node 3870 (either reachable or reaching it)
target_node = 3870

if target_node in G:
    # Get descendants (all nodes that come after 3870 in the hierarchy)
    connected_nodes = set(nx.descendants(G, target_node))
    connected_nodes.add(target_node)  # include the node itself

    # Remove all other nodes
    nodes_to_remove = set(G.nodes()) - connected_nodes
    G.remove_nodes_from(nodes_to_remove)
else:
    print(f"Node {target_node} not found in the graph.")


highlight_name = "Espectáculo"  # Name to highlight
highlight_node = None

for node_id, name in id_to_name.items():
    if name.lower() == highlight_name.lower():
        highlight_node = node_id
        break

if highlight_node is None:
    print(f"Node with name '{highlight_name}' not found.")

# Analyze the network
analyze_network(G)

# Create horizontal layout with separated text
fig1 = create_interactive_network_horizontal(G, id_to_name)
# fig1.show()

# ids 3887, 3881, 2210, 2209, 2207

st.set_page_config(layout="wide")
st.title("Relações Das Funções Da CETBase")

# Add search functionality
search_name = st.text_input("Procurar nome do nó (atenção especial aos acentos!):")

if search_name:
    # Find node ID by name
    searched_node_id = None
    for node_id, name in id_to_name.items():
        if name.lower() == search_name.lower():
            searched_node_id = node_id
            break

    if searched_node_id is None:
        st.warning(f"Nó com nome '{search_name}' não encontrado.")
    else:
        # Find a root (node with no parents)
        roots = [n for n in G.nodes() if G.in_degree(n) == 0]
        path_found = False

        for root in roots:
            try:
                path = nx.shortest_path(G, source=root, target=searched_node_id)
                path_found = True
                break  # use the first valid path found
            except nx.NetworkXNoPath:
                continue

        if not path_found:
            st.warning(f"Não há caminho até o nó '{search_name}' a partir de nenhuma raiz.")
        else:
            # Filter graph to only include nodes and edges in the path
            subG = G.subgraph(path).copy()
            filtered_id_to_name = {node: id_to_name[node] for node in subG.nodes()}
            fig_filtered = create_interactive_network_horizontal(subG, filtered_id_to_name, title=f"Caminho até '{search_name}'", height=200)
            st.plotly_chart(fig_filtered, use_container_width=True)
else:
    st.plotly_chart(fig1, use_container_width=True)
