#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_DISTANCE -1
#define NOT_VISITED_VERTEX 0
#define THRESHOLD 0.25
#define DYNAMIC_CHUNK 2048

void vertex_set_clear(vertex_set *list) {
    list->count = 0;
}

void vertex_set_init(vertex_set *list, int count) {
    list->max_vertices = count;
    list->vertices = (int *) calloc(list->max_vertices, sizeof(int));
    vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(
        Graph &g,
        vertex_set *frontier,
        int *distances,
        int &current_frontier) {
    int num_of_frontiers = 0;

    #pragma omp parallel for reduction (+:num_of_frontiers) schedule (dynamic, DYNAMIC_CHUNK)
    for (int node = 0; node < g->num_nodes; node++) {
        // If the vertex contains current frontier,
        // then we need to add its neighbors.
        if (frontier->vertices[node] == current_frontier) {
            const int start_edge = g->outgoing_starts[node];
            const int end_edge = (node == g->num_nodes - 1)
                                 ? g->num_edges
                                 : g->outgoing_starts[node + 1];

            // Attempt to add all neighbors to the new frontier
            for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
                const int outgoing = g->outgoing_edges[neighbor];

                if (frontier->vertices[outgoing] == NOT_VISITED_VERTEX) {
                    num_of_frontiers++;
                    distances[outgoing] = distances[node] + 1;
                    frontier->vertices[outgoing] = current_frontier + 1;
                }
            }
        }
    }
    frontier->count = num_of_frontiers;
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution *sol) {

    vertex_set list;
    vertex_set_init(&list, graph->num_nodes);

    vertex_set *frontier = &list;

    // Initialize all nodes to NOT_VISITED
    #pragma omp parallel for schedule (dynamic, DYNAMIC_CHUNK)
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_DISTANCE;

    // Setup frontier with the root node
    // Number of hops to the root node
    int num_of_hops = 1;
    frontier->vertices[frontier->count++] = num_of_hops;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0) {

        #ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
        #endif

        vertex_set_clear(frontier);

        top_down_step(graph, frontier, sol->distances, num_of_hops);

        #ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
        #endif

        num_of_hops++;
    }
}

// Take one step of "bottom-up" BFS.  For each vertex, check whether
// it should be added to the new_frontier.
void bottom_up_step(
        Graph &g,
        vertex_set *frontier,
        int *distances,
        int &num_of_hops) {
    int num_of_frontiers = 0;

    #pragma omp parallel for reduction (+:num_of_frontiers) schedule (dynamic, DYNAMIC_CHUNK)
    for (int node = 0; node < g->num_nodes; node++) {
        if (frontier->vertices[node] == NOT_VISITED_VERTEX) {
            const int start_edge = g->incoming_starts[node];
            const int end_edge = (node == g->num_nodes - 1)
                                 ? g->num_edges
                                 : g->incoming_starts[node + 1];

            // Attempt to add all neighbors to the new frontier
            for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
                const int incoming = g->incoming_edges[neighbor];

                // If the vertex contains the target number of hops,
                // then it is the parent of current node.
                if (frontier->vertices[incoming] == num_of_hops) {
                    num_of_frontiers++;
                    distances[node] = distances[incoming] + 1;
                    frontier->vertices[node] = num_of_hops + 1;
                    break;
                }
            }
        }
    }
    frontier->count = num_of_frontiers;
}

void bfs_bottom_up(Graph graph, solution *sol) {
    // For PP students:
    //
    // You will need to implement the "bottom up" BFS here as
    // described in the handout.
    //
    // As a result of your code's execution, sol.distances should be
    // correctly populated for all nodes in the graph.
    //
    // As was done in the top-down case, you may wish to organize your
    // code by creating subroutine bottom_up_step() that is called in
    // each step of the BFS process.
    vertex_set list;
    vertex_set_init(&list, graph->num_nodes);

    vertex_set *frontier = &list;

    // Initialize all nodes to NOT_VISITED
    #pragma omp parallel for schedule (dynamic, DYNAMIC_CHUNK)
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_DISTANCE;

    // Setup frontier with the root node
    // Number of hops to the root node
    int num_of_hops = 1;
    frontier->vertices[frontier->count++] = num_of_hops;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0) {

        #ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
        #endif

        vertex_set_clear(frontier);

        bottom_up_step(graph, frontier, sol->distances, num_of_hops);

        #ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
        #endif

        num_of_hops++;
    }
}

void bfs_hybrid(Graph graph, solution *sol) {
    // For PP students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.

    vertex_set list;
    vertex_set_init(&list, graph->num_nodes);

    vertex_set *frontier = &list;

    // Initialize all nodes to NOT_VISITED
    #pragma omp parallel for schedule (dynamic, DYNAMIC_CHUNK)
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_DISTANCE;

    // Setup frontier with the root node
    // Number of hops to the root node
    int num_of_hops = 1;
    frontier->vertices[frontier->count++] = num_of_hops;
    sol->distances[ROOT_NODE_ID] = 0;

    // Compute threshold count
    const double threshold_count = graph->num_nodes * THRESHOLD;

    while (frontier->count != 0) {

        #ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
        #endif

        if (frontier->count > threshold_count) {
            vertex_set_clear(frontier);
            bottom_up_step(graph, frontier, sol->distances, num_of_hops);
        } else {
            vertex_set_clear(frontier);
            top_down_step(graph, frontier, sol->distances, num_of_hops);
        }

        #ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
        #endif

        num_of_hops++;
    }
}
