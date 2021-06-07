# Graph Differential Privacy

This repository provides implementations of differentially private release mechanisms for graph statistics.
These implementations were originally produced in support of the paper "Private Graph Data Release: A Survey" by
Yang Li, Michael Purcell, Thierry Rakotoarivelo, David Smith, Thilina Ranbaduge, and Kee Siong Ng.

## Edge Differential Privacy
To demonstrate techniques designed to provide edge differential privacy, we implemented some of the algorithms described in the paper
"Smooth Sensitivity and Sampling in Private Data Analysis" by Kobbi Nissim, Sofya Raskhodnikova, and Adam Smith.
In this paper, the authors discuss several graph statistics with large global sensitivity.  For typical graphs, however,
the local sensitivity of these statistics can be much smaller. The authors prove that, by adding noise scaled according to a smooth
upper bound to the local sensitivity, it is possible to create practical differentially private release mechanisms for these statistics.

The primary difficulty with this approach is computing the value of the smooth upper bound for the local sensitivity of the various graph statistics.
Doing so efficiently requires bespoke algorithms for each statistic of interest. 
We provide reference implementations for the algorithms that the authors describe for computing smooth upper bounds for the local sensitivity
of the number of triangles in a graph and for the cost of a minimum spanning tree for a graph.

We have produced two versions of our implementations for these algorithms. The first, [edge_privacy_networkx.py](./edge_privacy_networkx.py), 
uses networkx to perform the required graph computations.
The second, [edge_privacy_igraph.py](./edge_privacy_igraph.py), uses igraph for this purpose.
Both versions use the implementation of the Cauchy mechanism provided by the [RelM](https://github.com/anusii/RelM) 
library to release the differentially private query responses.
 

## Node Differential Privacy
To demonstrate techniques designed to provide node differential privacy, we implemented some of the algorithms described in the paper
"Analyzing Graphs with Node Differential Privacy" by Shiva Prasad Kasiviswanathan, Kobbi Nissim, Sofya Raskhodnikova, and Adam Smith.
