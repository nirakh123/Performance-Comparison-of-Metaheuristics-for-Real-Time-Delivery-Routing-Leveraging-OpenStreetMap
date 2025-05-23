import numpy as np
import random
import networkx as nx
import osmnx as ox
import folium
import matplotlib.pyplot as plt
import csv
import statistics
import plotly.graph_objs as go
from plotly.offline import plot

# Area setup
area_center = (28.7041, 77.1025)
graph = ox.graph_from_point(area_center, dist=80000, network_type="drive")
graph = nx.to_undirected(graph)

def get_largest_connected_component(graph):
    largest_cc = max(nx.connected_components(graph), key=len)
    return graph.subgraph(largest_cc).copy()

graph = get_largest_connected_component(graph)

# Delivery points
delivery_nodes_coords = [
    (28.711249, 77.274662), (28.729463, 77.319015), (28.743314, 76.949369),
    (28.661281, 76.998206), (28.543951, 76.960416), (28.602703, 77.129179),
    (28.419527, 76.934636), (28.612199, 77.086896), (28.838031, 77.310871),
    (28.502807, 76.978761), (28.473335, 77.144981), (28.767983, 77.238037),
    (28.793922, 77.026761), (28.831478, 76.924839), (28.474453, 77.191706)
]

delivery_nodes_ids = [ox.distance.nearest_nodes(graph, X=lon, Y=lat) for lat, lon in delivery_nodes_coords]
connected_nodes = set(graph.nodes)
valid_indices = [i for i, node in enumerate(delivery_nodes_ids) if node in connected_nodes]
delivery_nodes_ids = [delivery_nodes_ids[i] for i in valid_indices]
delivery_nodes_coords = [delivery_nodes_coords[i] for i in valid_indices]
num_nodes = len(delivery_nodes_ids)

print(f"Using {num_nodes} connected nodes.")

# Distance matrix
def compute_distance_matrix(graph, nodes):
    n = len(nodes)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        lengths = nx.single_source_dijkstra_path_length(graph, nodes[i], weight="length")
        for j in range(n):
            dist_matrix[i][j] = lengths.get(nodes[j], float("inf"))
    return dist_matrix

distance_matrix = compute_distance_matrix(graph, delivery_nodes_ids)

# Save distance matrix to CSV
with open("distance_matrix.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([""] + list(range(num_nodes)))
    for i in range(num_nodes):
        row = [i] + list(np.round(distance_matrix[i], 2))
        writer.writerow(row)

if np.any(np.isinf(distance_matrix)):
    print("Error: Some nodes are unreachable.")
    exit(1)

# ACO parameters
num_ants = 50
num_iterations = 200
alpha = 1.0
beta = 3.0
evaporation_rate = 0.5
initial_pheromone = 1.0

pheromone = np.ones((num_nodes, num_nodes)) * initial_pheromone
heuristic = np.zeros_like(distance_matrix)
with np.errstate(divide='ignore'):
    heuristic = 1.0 / distance_matrix
heuristic[np.isinf(heuristic)] = 0

def construct_solution():
    route = [0]
    unvisited = set(range(1, num_nodes))
    while unvisited:
        current = route[-1]
        probabilities = []
        valid_next = []
        for n in unvisited:
            tau = pheromone[current][n]
            eta = heuristic[current][n]
            if eta > 0:
                valid_next.append(n)
                probabilities.append((tau ** alpha) * (eta ** beta))
        total = sum(probabilities)
        if total == 0 or not probabilities:
            break
        probabilities = [p / total for p in probabilities]
        next_node = random.choices(valid_next, weights=probabilities, k=1)[0]
        route.append(next_node)
        unvisited.remove(next_node)
    route.append(0)
    return route

def calculate_distance(route):
    return sum(distance_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1))

def update_pheromones(routes, distances):
    global pheromone
    pheromone *= (1 - evaporation_rate)
    for route, dist in zip(routes, distances):
        if dist == float("inf"):
            continue
        for i in range(len(route) - 1):
            pheromone[route[i]][route[i + 1]] += 1.0 / dist

def aco_tsp():
    best_route = None
    best_distance = float("inf")
    distances_per_iteration = []
    for _ in range(num_iterations):
        routes = [construct_solution() for _ in range(num_ants)]
        distances = [calculate_distance(route) for route in routes]
        valid_distances = [d for d in distances if d != float("inf")]
        if valid_distances:
            min_dist = min(valid_distances)
            distances_per_iteration.append(min_dist)
            if min_dist < best_distance:
                best_distance = min_dist
                best_route = routes[distances.index(min_dist)]
        update_pheromones(routes, distances)
    return best_route, best_distance, distances_per_iteration

# Run ACO once
best_route, best_distance, convergence = aco_tsp()

print(f"\nMinimum Distance: {round(best_distance, 2)}")
print(f"Route (index of delivery nodes): {best_route}")

# Save convergence plot
plt.figure(figsize=(10, 6))
plt.plot(convergence, marker='o', linestyle='-', color='blue')
plt.title("ACO Convergence (Single Run)")
plt.xlabel("Iteration")
plt.ylabel("Best Distance")
plt.grid(True)
plt.savefig("aco_convergence_plot.png")
plt.close()

# Visualize route using folium
def visualize_route(graph, delivery_ids, route):
    route_nodes = []
    for i in range(len(route) - 1):
        part = nx.shortest_path(graph, source=delivery_ids[route[i]],
                                target=delivery_ids[route[i+1]], weight='length')
        route_nodes.extend(part[:-1])
    route_nodes.append(delivery_ids[route[-1]])

    route_latlon = [(graph.nodes[n]['y'], graph.nodes[n]['x']) for n in route_nodes]

    m = folium.Map(location=route_latlon[0], zoom_start=10)
    folium.PolyLine(locations=route_latlon, color='red', weight=5).add_to(m)

    for idx, (lat, lon) in enumerate(delivery_nodes_coords):
        folium.Marker(location=(lat, lon), popup=f"Node {idx}").add_to(m)

    m.save("tsp_route_map.html")

visualize_route(graph, delivery_nodes_ids, best_route)
