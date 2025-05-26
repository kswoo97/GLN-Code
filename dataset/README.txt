# Note. Please download datasets from https://www.dropbox.com/scl/fo/xyhn4mnnw311lof02f3tw/ALGcrX9Er4c2sFgW6BI4AMA?rlkey=5gtzq154fd8r2xzn3l2ubwncl&st=sy4o47rb&dl=0
# Locate the downloaded datasets in the folder "./dataset".
# This file includes detailed information for the datasets used for GLN.

## About {data_name}_node.pickle

- This file is a Python dictionary containing the following keys:
  - target_nodes: This is a 1-D numpy array containing the sampled nodes that are used for node classification.
  - sampled_edges: This is a Python dictionary where key is each node and the corresponding value is a Python list containing its 1-hop neighbors. These neighbors are not full neighbors, but sampled neighbors.
  - initial_node_attributes: This is a 1-D numpy array containing the initial text attributes of each node. k-th element indicates the attribute of node with k index.
  - node_labels: This is a 1-D numpy array containing the class of each node. k-th element indicates the class of node with k index.

## About {data_name}_edge.pickle

- This file is a Python dictionary containing the following keys:
  - target_nodes: This is a 1-D numpy array containing the sampled nodes that are used for link prediction.
  - sampled_edges: This is a Python dictionary where key is each node and the corresponding value is a Python list containing its 1-hop neighbors. These neighbors are not full neighbors, but sampled neighbors.
  - initial_node_attributes: This is a 1-D numpy array containing the initial text attributes of each node. k-th element indicates the attribute of node with k index.
  - ground_truth_pairs: This is a 2-D numpy array containing the ground-truth edges that are used as labels of link prediction. Specifically, (0,k)-th element node is forming link with (1,k)-th element node.