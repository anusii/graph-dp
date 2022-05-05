# Graph Differential Privacy

This repository provides implementations of differentially private release mechanisms for graph statistics.
These implementations were originally produced in support of the paper "Private Graph Data Release: A Survey" by
Yang Li, Michael Purcell, Thierry Rakotoarivelo, David Smith, Thilina Ranbaduge, and Kee Siong Ng.

We decided to create this repository in order to address the natural question of what methods should be used in various situations. This turns out to be a rather difficult question to answer in any way other than by saying, "it depends". For any system, the privacy and utility requirements must be taken into account. Furthermore, the nature of the underlying data and the nature queries that the system is designed to answer will determine what techniques are available for consideration.

As such, rather than trying to provide a comprehensive analysis we instead provide some tools that readers can use to perform the kinds of analyses required to determine the techniques that are suitable for their needs.

In this repository are ipython notebooks that demonstrate how to use existing graph analytics and differential privacy packages, to implement differentially private graph analytics. We discuss each of these in the following sections of this document.

## Generating Random Graphs
It is often useful to use synthetic data to for testing purposes.  As such, generating various types of random graphs is a fundamental task when experimenting with graph differential privacy.

In notebook [generating_random_graphs.ipynb](./generating_random_graphs.ipynb) we demonstrate how to use two popular graph analytics python packages, [NetworkX](https://networkx.org) and [igraph](https://igraph.org), to generate random graphs.  For both packages, we show how to generate random graphs drawn from the [Erdős-Rényi model](https://en.wikipedia.org/wiki/Erdős–Rényi_model) and the [Barabási-Albert model](https://en.wikipedia.org/wiki/Barabási–Albert_model). We also show how to interrogate the resulting graphs to determine/verify that they have the expected number of nodes and edges.

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

Our implementation, [edge_differential_privacy.ipynb](./edge_differential_privacy.ipynb), uses [NetworkX](https://networkx.org) to perform the required graph computations.
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

Our implementation, [node_differential_privacy.ipynb](./node_differential_privacy.ipynb), uses [NetworkX](https://networkx.org) to perform the required graph computations
and [scipy](https://www.scipy.org) (in particular the optimisation algorithms provided by
[scipy.optimize](https://docs.scipy.org/doc/scipy/reference/optimize.html))
to solve the required optimisation problems. We use the implementation of the Laplace mechanism provided by the
[RelM](https://github.com/anusii/RelM) library to release the differentially private query responses.

## Differentially Private Two-party Egocentric Betweenness Centrality
To provide an example of a more complicated type of query, we implemented the algorithms described in the paper [Differentially Private Two-Party Egocentric Betweenness Centrality](https://arxiv.org/pdf/1901.05562.pdf) by Leyla Roohi, Benjamin I.P. Rubinstein, and Vanessa Teague.

Many notions of the centrality of a node in a graph depend strongly on the global structure of the graph.  As such, queries that compute such quantities tend to have large sensitivity.  So, it is difficult to design differentially private release mechanisms that produce useful estimates of these quantities.

By focusing on an "egocentric" model of centrality, the authors effectively reduce the sensitivity of their queries by only considering a small neighborhood of the original graph.

One aspect that makes this algorithm challenging to use is that it uses a bespoke stratified sampling algorithm to implement a version of the exponential mechanism. This makes it difficult to use existing differential privacy libraries to apply the required perturbations in this step of the algorithm. In our example, we implement this bespoke sampling algorithm but have made no attempt to ensure that our implementation is secure.

Our implementation, [egocentric_betweenness_centrality.ipynb](./egocentric_betweenness_centrality.ipynb) uses [NetworkX](https://networkx.org) to perform the required graph computations. We use the implementation of the Laplace mechanism provided by the
[RelM](https://github.com/anusii/RelM) library to release the differentially private query responses.
