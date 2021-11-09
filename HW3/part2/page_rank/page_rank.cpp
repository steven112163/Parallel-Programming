#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double *solution, double damping, double convergence) {
    /*
       For PP students: Implement the page rank algorithm here.  You
       are expected to parallelize the algorithm using openMP.  Your
       solution may need to allocate (and free) temporary arrays.

       Basic page rank pseudocode is provided below to get you started:

       // initialization: see example code above
       score_old[vi] = 1/numNodes;

       while (!converged) {

         // compute score_new[vi] for all nodes vi:
         score_new[vi] = sum over all nodes vj reachable from incoming edges
                            { score_old[vj] / number of edges leaving vj  }
         score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

         score_new[vi] += sum over all nodes v in graph with no outgoing edges
                            { damping * score_old[v] / numNodes }

         // compute how much per-node scores have changed
         // quit once algorithm has converged

         global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
         converged = (global_diff < convergence)
       }

     */

    // Initialize vertex weights to uniform probability. Double
    // precision scores are used to avoid underflow for large graphs
    int numNodes = num_nodes(g);
    double equal_prob = 1.0 / numNodes;
    #pragma omp parallel for
    for (int i = 0; i < numNodes; ++i) {
        solution[i] = equal_prob;
    }

    // Declare old solution
    double *old_solution = (double *) malloc(numNodes * sizeof(double));

    // Declare sum of no outgoing nodes and global difference
    double sum_of_no_outgoing, global_diff;

    // Declare dummy constant
    double constant = (1.0 - damping) / numNodes;

    bool converged = false;
    while (!converged) {
        // Copy solution to old_solution
        memcpy(old_solution, solution, numNodes * sizeof(double));

        sum_of_no_outgoing = constant;
        global_diff = 0.0;

        #pragma omp parallel
        {
            // Compute sum of no outgoing nodes
            #pragma omp for reduction (+:sum_of_no_outgoing)
            for (int no_outgoing = 0; no_outgoing < numNodes; no_outgoing++) {
                if (outgoing_size(g, no_outgoing) == 0)
                    sum_of_no_outgoing += damping * old_solution[no_outgoing] / numNodes;
            }

            // Compute solution[vi] for all nodes vi
            #pragma omp for reduction (+:global_diff)
            for (int vi = 0; vi < numNodes; vi++) {
                const Vertex *start = incoming_begin(g, vi);
                const Vertex *end = incoming_end(g, vi);
                double sum = 0.0;
                for (const Vertex *incoming = start; incoming != end; incoming++) {
                    sum += old_solution[*incoming] / outgoing_size(g, *incoming);
                }
                solution[vi] = (damping * sum) + sum_of_no_outgoing;

                // Compute how much per-node scores have changed
                global_diff += fabs(old_solution[vi] - solution[vi]);
            }
        }

        // Quit once algorithm has converged
        converged = (global_diff < convergence);
    }
    delete old_solution;
}
