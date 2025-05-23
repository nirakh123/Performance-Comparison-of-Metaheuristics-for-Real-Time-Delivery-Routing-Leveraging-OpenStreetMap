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
    (28.711249, 77.274662),
    (28.729463, 77.319015),
    (28.743314, 76.949369),
    (28.661281, 76.998206),
    (28.543951, 76.960416),
    (28.602703, 77.129179),
    (28.419527, 76.934636),
    (28.612199, 77.086896),
    (28.838031, 77.310871),
    (28.502807, 76.978761),
    (28.473335, 77.144981),
    (28.767983, 77.238037),
    (28.793922, 77.026761),
    (28.831478, 76.924839),
    (28.474453, 77.191706)
]

delivery_nodes_ids = [ox.distance.nearest_nodes(graph, X=lon, Y=lat) for lat, lon in delivery_nodes_coords]
connected_nodes = set(graph.nodes)
valid_indices = [i for i, node in enumerate(delivery_nodes_ids) if node in connected_nodes]
delivery_nodes_ids = [delivery_nodes_ids[i] for i in valid_indices]
delivery_nodes_coords = [delivery_nodes_coords[i] for i in valid_indices]

print(f"Using {len(delivery_nodes_ids)} connected nodes.")

def compute_distance_matrix(graph, nodes):
    n = len(nodes)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        lengths = nx.single_source_dijkstra_path_length(graph, nodes[i], weight="length")
        for j in range(n):
            dist_matrix[i][j] = lengths.get(nodes[j], float("inf"))
    return dist_matrix

distance_matrix = compute_distance_matrix(graph, delivery_nodes_ids)
if np.any(np.isinf(distance_matrix)):
    print("Error: Some nodes are unreachable.")
    exit(1)

# ACO Parameters
num_ants = 50
num_iterations = 200
alpha = 1.0
beta = 3.0
evaporation_rate = 0.5
initial_pheromone = 1.0
num_nodes = len(delivery_nodes_ids)

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
    total = 0
    for i in range(len(route) - 1):
        dist = distance_matrix[route[i]][route[i + 1]]
        if dist == float("inf"):
            return float("inf")
        total += dist
    return total

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
    for _ in range(num_iterations):
        routes = [construct_solution() for _ in range(num_ants)]
        distances = [calculate_distance(route) for route in routes]
        valid_distances = [d for d in distances if d != float("inf")]
        if valid_distances:
            min_dist = min(valid_distances)
            if min_dist < best_distance:
                best_distance = min_dist
                best_route = routes[distances.index(min_dist)]
            update_pheromones(routes, distances)
    return best_route, best_distance

# New: Save convergence plots (matplotlib and Plotly)
def save_convergence_plot(run_numbers, distances):
    # Matplotlib plot
    plt.figure(figsize=(10, 6))
    plt.plot(run_numbers, distances, marker='o', linestyle='-', color='blue')
    plt.title("ACO Convergence Over 30 Runs")
    plt.xlabel("Run Number")
    plt.ylabel("Minimum Distance")
    plt.grid(True)
    plt.savefig("aco_convergence_plot.png")
    plt.close()

    # Plotly HTML plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=run_numbers, y=distances, mode='lines+markers', name='Distance',
                             line=dict(color='blue')))
    fig.update_layout(title="ACO Convergence Over 30 Runs",
                      xaxis_title="Run Number", yaxis_title="Minimum Distance")
    plot(fig, filename="aco_convergence_plot.html", auto_open=False)

# Run ACO multiple times and log results
def run_multiple_aco(runs=30):
    results = []
    run_ids = []
    csv_filename = "aco_30runs_summary.csv"

    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["run_number", "minimum_distance"])

        for run in range(1, runs + 1):
            print(f"\n=== ACO Run {run} ===")
            global pheromone
            pheromone = np.ones((num_nodes, num_nodes)) * initial_pheromone
            _, min_distance = aco_tsp()
            results.append(min_distance)
            run_ids.append(run)
            writer.writerow([run, round(min_distance, 2)])

    mean_val = round(statistics.mean(results), 2)
    std_dev = round(statistics.stdev(results), 2)

    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([])
        writer.writerow(["Mean", mean_val])
        writer.writerow(["Standard Deviation", std_dev])

    # Save convergence graph
    save_convergence_plot(run_ids, results)

    print("\n=== Summary ===")
    print(f"Mean Minimum Distance: {mean_val}")
    print(f"Standard Deviation: {std_dev}")
    print("CSV, PNG, and HTML plot files have been saved.")

# Run it all
run_multiple_aco(runs=30)
