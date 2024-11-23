#!/usr/bin/env python3

"""
MTVRP Solver using Nearest Neighbor Construction
and Intra-Route Optimization with Visualization
"""

#------------------------ Constants ------------------------#

MAX_VEHICLE_TRIPS = 100
DEFAULT_BATCH_SIZE = 1000
DEPOT_ID = 0

import os
import sys
import time
import math
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np
import argparse

#------------------------ Data Structures ------------------------#

class _Point:
    """Internal coordinate representation"""
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        
    def dist_to(self, other) -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
        
    def as_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)

class Stop:
    """Represents a customer stop"""
    def __init__(self, id: int, demand: int, service_time: int, x: float = 0, y: float = 0):
        self.id = id
        self.demand = demand
        self.serv_time = service_time
        self.loc = _Point(x, y) if x or y else None
        
class Route:
    """Sequence of stops"""
    def __init__(self):
        self.stop_list: List[Stop] = []
        self.total_load = 0
        self.duration = 0
        
    def add_stop(self, stop: Stop) -> None:
        self.stop_list.append(stop)
        self.total_load += stop.demand
        
    def can_add(self, stop: Stop, capacity: int) -> bool:
        return self.total_load + stop.demand <= capacity
        
    def get_stop_ids(self) -> List[int]:
        """Return list of stop IDs in route order"""
        return [stop.id for stop in self.stop_list]

#------------------------ Problem Data ------------------------#

class ProblemData:
    def __init__(self):
        self.stops: Dict[int, Stop] = {}
        self.depot: Stop = None
        self.n_stops = 0
        self.n_trips = 0
        self.capacity = 0
        self.dist_matrix: List[List[int]] = []
        self.has_coordinates = False

def read_problem_file(fname: str) -> ProblemData:
    """Read and parse problem instance"""
    if not os.path.exists(fname):
        sys.exit(f"Error: File {fname} not found")
        
    prob = ProblemData()
    
    with open(fname, 'r') as f:
        # Read header
        prob.n_stops = int(next(f).split(':')[1])
        prob.n_trips = int(next(f).split(':')[1])
        prob.capacity = int(next(f).split(':')[1])
        
        # Skip blank line and label
        next(f)
        next(f)
        
        # Read demands
        demand_list = [int(x) for x in next(f).split()]
        
        # Skip blank line and label
        next(f)
        next(f)
        
        # Read service times
        time_list = [int(x) for x in next(f).split()]
        
        # Create depot
        prob.depot = Stop(DEPOT_ID, 0, 0)
        
        # Handle coordinate vs matrix format
        if fname.startswith('V'):
            prob.has_coordinates = True
            # Skip coordinate header
            next(f)
            next(f)
            
            # Read coordinates and create stops
            coords = []
            for i in range(prob.n_stops + 1):
                x, y = map(float, next(f).split())
                if i == 0:
                    prob.depot.loc = _Point(x, y)
                else:
                    prob.stops[i] = Stop(i, demand_list[i-1], time_list[i-1], x, y)
                coords.append(_Point(x, y))
                    
            # Build distance matrix
            prob.dist_matrix = build_dist_matrix(coords)
            
        else:
            prob.has_coordinates = False
            # Create stops without coordinates
            for i in range(prob.n_stops):
                prob.stops[i+1] = Stop(i+1, demand_list[i], time_list[i])
                
            # Skip "TravelTimes:" label
            next(f)
            
            # Read distance matrix
            prob.dist_matrix = []
            for _ in range(prob.n_stops + 1):
                prob.dist_matrix.append([int(x) for x in next(f).split()])
                
    return prob

def build_dist_matrix(points: List[_Point]) -> List[List[int]]:
    """Convert point list to distance matrix"""
    size = len(points)
    matrix = [[0] * size for _ in range(size)]
    
    for i in range(size):
        for j in range(size):
            if i != j:
                matrix[i][j] = round(points[i].dist_to(points[j]))
                
    return matrix

#------------------------ Solution Building ------------------------#

def build_nearest_neighbor(prob: ProblemData) -> List[Route]:
    """Construct initial solution using nearest neighbor"""
    unvisited = set(prob.stops.keys())
    route_list = []
    
    while unvisited and len(route_list) < prob.n_trips:
        # Start new route
        curr_route = Route()
        
        # Pick seed customer with best demand/distance ratio
        if unvisited:
            seed = min(unvisited,
                      key=lambda x: prob.dist_matrix[0][x] / prob.stops[x].demand)
            curr_route.add_stop(prob.stops[seed])
            unvisited.remove(seed)
        
        # Build route
        while unvisited:
            last = curr_route.stop_list[-1]
            next_stop = None
            best_dist = float('inf')
            
            # Find nearest feasible neighbor
            for stop_id in unvisited:
                stop = prob.stops[stop_id]
                if curr_route.can_add(stop, prob.capacity):
                    dist = prob.dist_matrix[last.id][stop.id]
                    if dist < best_dist:
                        best_dist = dist
                        next_stop = stop
                        
            if next_stop:
                curr_route.add_stop(next_stop)
                unvisited.remove(next_stop.id)
            else:
                break
                
        route_list.append(curr_route)
        
    return route_list

#------------------------ Route Optimization ------------------------#

def calc_route_time(route: Route, dist_matrix: List[List[int]]) -> int:
    """Calculate time to complete route"""
    if not route.stop_list:
        return 0
        
    total = 0
    prev_id = DEPOT_ID
    
    for stop in route.stop_list:
        total += dist_matrix[prev_id][stop.id]
        total += stop.serv_time
        prev_id = stop.id
        
    total += dist_matrix[prev_id][DEPOT_ID]  # Return to depot
    return total

def calc_total_latency(route_list: List[Route], dist_matrix: List[List[int]]) -> int:
    """Calculate total latency of solution"""
    total = 0
    elapsed = 0
    
    for route in route_list:
        # Base route latency
        curr_time = 0
        prev_id = DEPOT_ID
        route_latency = 0
        
        for stop in route.stop_list:
            curr_time += dist_matrix[prev_id][stop.id]
            route_latency += curr_time + elapsed
            curr_time += stop.serv_time
            prev_id = stop.id
            
        total += route_latency
        
        # Update elapsed time
        elapsed += calc_route_time(route, dist_matrix)
        
    return total

def apply_2opt_move(route: Route, pos1: int, pos2: int) -> None:
    """Reverse segment of route between positions"""
    route.stop_list[pos1:pos2] = reversed(route.stop_list[pos1:pos2])

def optimize_routes(route_list: List[Route], prob: ProblemData) -> None:
    """Improve routes using 2-opt moves"""
    improved = True
    while improved:
        improved = False
        curr_cost = calc_total_latency(route_list, prob.dist_matrix)
        
        # Try 2-opt moves on each route
        for route in route_list:
            if len(route.stop_list) < 4:
                continue
                
            # Try all possible segment reversals
            for i in range(len(route.stop_list) - 2):
                for j in range(i + 2, len(route.stop_list)):
                    # Try move
                    apply_2opt_move(route, i, j)
                    new_cost = calc_total_latency(route_list, prob.dist_matrix)
                    
                    if new_cost < curr_cost:
                        curr_cost = new_cost
                        improved = True
                    else:
                        # Undo move
                        apply_2opt_move(route, i, j)

#------------------------ Visualization ------------------------#

def visualize_solution(prob: ProblemData, route_list: List[Route], 
                      title: str = None, show: bool = True, 
                      save_path: Optional[str] = None) -> None:
    """Visualize problem instance and solution routes"""
    if not prob.has_coordinates:
        print("Solo se pueden visualizar instancias con coordenadas")
        return
        
    plt.figure(figsize=(10, 10))
    
    # Plot depot
    depot_coords = prob.depot.loc.as_tuple()
    plt.scatter([depot_coords[0]], [depot_coords[1]], 
                c='red', s=100, marker='s', label='Depósito')
    
    # Plot clients
    x_coords = [stop.loc.x for stop in prob.stops.values()]
    y_coords = [stop.loc.y for stop in prob.stops.values()]
    plt.scatter(x_coords, y_coords, c='blue', s=50, label='Clientes')
    
    # Add client labels
    for stop in prob.stops.values():
        plt.annotate(f'{stop.id}', (stop.loc.x, stop.loc.y),
                    xytext=(5, 5), textcoords='offset points')
    
    # Plot routes
    if route_list:
        colors = plt.cm.rainbow(np.linspace(0, 1, len(route_list)))
        for route_idx, (route, color) in enumerate(zip(route_list, colors)):
            route_x = [prob.depot.loc.x]  # Start at depot
            route_y = [prob.depot.loc.y]
            
            for stop in route.stop_list:
                route_x.append(stop.loc.x)
                route_y.append(stop.loc.y)
                
            route_x.append(prob.depot.loc.x)  # Return to depot
            route_y.append(prob.depot.loc.y)
            
            plt.plot(route_x, route_y, c=color, alpha=0.5, linewidth=2,
                    label=f'Ruta {route_idx + 1}')
    
    # Setup plot
    plt.title(title or f'Instancia de MTVRP con {prob.n_stops} clientes')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()

def write_solution(route_list: List[Route], stats: Dict, fname: str) -> None:
    """Write solution to file"""
    with open(fname, 'w') as f:
        # Write summary block
        f.write('='*50 + '\n')
        f.write('Solucion MTVRP\n')
        f.write('='*50 + '\n\n')
        
        # Write statistics
        f.write('Rendimiento:\n')
        f.write(f"  Costo Inicial:  {stats['initial_cost']}\n")
        f.write(f"  Costo Final:    {stats['final_cost']}\n")
        f.write(f"  Mejora relativa:   {stats['improvement']:.1f}%\n")
        f.write(f"  Tiempo Solucion:    {stats['solve_time']:.2f}s\n\n")
        
        # Write routes
        f.write('Rutas:\n')
        for i, route in enumerate(route_list, 1):
            stops = ','.join(str(s.id) for s in route.stop_list)
            f.write(f"  {i}: {stops}\n")
            f.write(f"     Carga Total: {route.total_load}\n")
            
        f.write('\n' + '='*50 + '\n')

#------------------------ Main Program ------------------------#

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Resolver MTVRP',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument('input_file', 
                       help='entrada')
    
    parser.add_argument('--output',
                       help='salida')
                       
    parser.add_argument('--visualizar', action='store_true',
                       help='visualizar solucion')
                       
    parser.add_argument('--plot-output',
                       help='guardar solucion a archivo')
    
    return parser.parse_args()

def main():
    args = parse_args()
    args.quiet = False
    print("\nLeyendo problema...")
    prob = read_problem_file(args.input_file)
    
    print(f"Con {prob.n_stops} clientes")
    print("\nConstruyendo solucion inicial...")
    
    # Build and measure initial solution
    start_time = time.time()
    route_list = build_nearest_neighbor(prob)
    initial_cost = calc_total_latency(route_list, prob.dist_matrix)
    
    print("Aplicando busqueda local...")
    
    # Improve solution
    optimize_routes(route_list, prob)
    final_cost = calc_total_latency(route_list, prob.dist_matrix)
    solve_time = time.time() - start_time
    
    # Collect statistics
    stats = {
        'initial_cost': initial_cost,
        'final_cost': final_cost,
        'improvement': 100 * (initial_cost - final_cost) / initial_cost,
        'solve_time': solve_time
    }
    
    # Output results
    print('Rendimiento:\n')
    print(f"  Costo Inicial:      {stats['initial_cost']}\n")
    print(f"  Costo Final:        {stats['final_cost']}\n")
    print(f"  Mejora relativa:    {stats['improvement']:.4f}%\n")
    print(f"  Tiempo Solucion:    {stats['solve_time']:.4f}s\n\n")
    
    if args.output:
        write_solution(route_list, stats, args.output)
        print(f"\nSolucion escrita {args.output}")
            
    if args.visualizar or args.plot_output:
        visualize_solution(prob, route_list, 
                         show=args.visualizar,
                         save_path=args.plot_output)

if __name__ == '__main__':
    main()