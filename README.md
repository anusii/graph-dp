# Graph Differential Privacy

This repository provides implementations of differentially private release mechanisms for graph statistics.
These implementations were originally produced in support of the paper "Private Graph Data Release: A Survey" by
Yang Li, Michael Purcell, Thierry Rakotoarivelo, David Smith, Thilina Ranbaduge, and Kee Siong Ng.

## Edge Differential Privacy
To demonstrate techniques designed to provide edge differential privacy, we implemented some of the algorithms described in the paper
[Smooth Sensitivity and Sampling in Private Data Analysis](https://cs-people.bu.edu/ads22/pubs/NRS07/NRS07-full-draft-v1.pdf)
by Kobbi Nissim, Sofya Raskhodnikova, and Adam Smith.
In this paper, the authors discuss several graph statistics with large global sensitivity.  For typical graphs, however,
the local sensitivity of these statistics can be much smaller. The authors prove that, by adding noise scaled according to a smooth
upper bound to the local sensitivity, it is possible to create practical differentially private release mechanisms for these statistics.

The primary difficulty with this approach is computing the value of the smooth upper bound for the local sensitivity of the various graph statistics.
Doing so efficiently requires bespoke algorithms for each statistic of interest. 
We provide reference implementations of the algorithms that the authors describe for computing smooth upper bounds for the local sensitivity
of the number of triangles in a graph and for the cost of a minimum spanning tree for a graph.

Our implementation, [edge_privacy.ipynb](./edge_privacy.ipynb), uses [networkx](https://networkx.org) to perform the required graph computations.
We use the implementation of the Cauchy mechanism provided by the
[RelM](https://github.com/anusii/RelM)
library to release the differentially private query responses.

## Node Differential Privacy
To demonstrate techniques designed to provide node differential privacy, we implemented some of the algorithms described in the paper
[Analyzing Graphs with Node Differential Privacy](https://privacytools.seas.harvard.edu/files/privacytools/files/chp3a10.10072f978-3-642-36594-2_26.pdf)
by Shiva Prasad Kasiviswanathan, Kobbi Nissim, Sofya Raskhodnikova, and Adam Smith.
In this paper the authors discuss a technique for computing low-sensitivity approximations
to several classes of graph statistics with large global sensitivity. These approximate graph statistics can then be perturbed by
adding noise scaled according to their global sensitivity to create practical differentially private release mechanisms for these statistics. 

The authors describe two methods for computing these low-sensitivity approximations.
The first method applies to a class of graph statistics which can be written as the solution to a particular convex optimisation problem.
As such, the primary difficulty in this method is formulating and then solving that problem.
We provide a reference implementation of the algorithm that the authors describe for computing approximations for the number of edges in a graph, the number of nodes in a graph, and the number of k-stars in a graph.

The second method applies to a class of graph statistics which can be written as the solution to a particular linear programming problem.
Again, the primary difficulty in this method is formulating and then solving that problem.
We provide a reference implementation of the algorithm that the authors describe for computing approximations for number of triangles in a graph.

Our implementation, [node_privacy.ipynb](./node_privacy.ipynb), uses [networkx](https://networkx.org) to perform the required graph computations
and [scipy](https://www.scipy.org) (in particular the optimisation algorithms provided by
[scipy.optimize](https://docs.scipy.org/doc/scipy/reference/optimize.html))
to solve the required optimisation problems. We use the implementation of the Laplace mechanism provided by the
[RelM](https://github.com/anusii/RelM) library to release the differentially private query responses.
  
