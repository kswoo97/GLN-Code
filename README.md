# Graph Lingual Network (GLN)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) [![arXiv](https://img.shields.io/badge/arXiv-2505.20742-b31b1b.svg)](https://arxiv.org/abs/2505.20742)

--------

### TL;DR: Imagine GNNs where hidden layers talk in language—ours does exactly that with an LLM!
![Result image](https://github.com/kswoo97/GLN/blob/main/GLN_figures/GLN_Image.png)

--------

### Paper overview

- **[Paper]** This is the official implementation of the paper titled "***‘Hello, World!’: Making GNNs Talk with LLMs***" (arXiv link here).
- **[Authors]** The work is a collaboration between ***Sunwoo Kim, Soo Yong Lee, Jaemin Yoo, Kijung Shin***, all from ***Korea Advanced Institute of Science and Technology (KAIST)***.
- **[Brief overview]** We propose Graph Lingual Network (GLN), a graph neural network where all hidden node representations are expressed as text.
  - To enable this, a large language model (LLM) acts as the message passing module.
  - The LLM is prompted to update each target node's representation by aggregating information from its neighbors.
- **[Figure]** An example figure showing the generation of node A’s representation by GLN.
![Result image](https://github.com/kswoo97/GLN/blob/main/GLN_figures/new_overview.png)

--------

### Code overview

#### Datasets

- **[Overview]** We support the following three datasets:
  - Co-citation network: arXiv (arxiv)
  - Co-purchase network: History-Book (book)
  - Hyperlink network: Wiki-CS (page)
- **[Storage]** Please download the datasets from [LINK](https://www.dropbox.com/scl/fo/xyhn4mnnw311lof02f3tw/ALGcrX9Er4c2sFgW6BI4AMA?rlkey=5gtzq154fd8r2xzn3l2ubwncl&st=sy4o47rb&dl=0), and put the datasets one aims to use in the folder named `./dataset`.
  - Refer to the ***README file*** in the [LINK](https://www.dropbox.com/scl/fo/xyhn4mnnw311lof02f3tw/ALGcrX9Er4c2sFgW6BI4AMA?rlkey=5gtzq154fd8r2xzn3l2ubwncl&st=sy4o47rb&dl=0).
 
#### Details regarding code

- **[Overview]** We support both (1) the encoding process and (2) the downstream task execution of GLN.
  - The encoding process is implemented in `GLN_encoding.py`
  - The downstream inference is implemented in `GLN_downstream.py`
  - The backbone code for the encoding and downstream tasks is implemented in `GLN_src.py`

- **[Running]** The Detailed code guideline is as follows.
  - For the ***encoding process***, run as follows:
    ```
    python3 GLN_encoding.py -data book -task node -llm gpt
    ```
    - `-data` indicates the dataset one aims to use. Possible options are: `{arxiv, paper, page}`, representing arXiv, History-Book, and Wiki-CS, respectively.
    - `-task` indicates the target downstream task one aims to perform. Possible options are: `{node, edge}`, representing node classification and link prediction, respectively.
    - `-llm` indicates the backbone large language model one aims to use for GLN. Possible options are: `{gpt, claude}`, representing GPT-4o-mini and Claude-3-Haiku, respectively.
      - However, one can easily change the LLM by changing the model name!
    - This code saves the node representations in the `./gen_results` folder, which is automatically made.

  - For the ***downstream tasks***, run as follows:
    - **[NOTE]** Before running this code, one should locate the generated node representations in the `./gen_results` folder.
    ```
    python3 GLN_downstream.py -data book -task node -llm gpt
    ```
    - `-data` indicates the dataset one aims to use. Possible options are: `{arxiv, paper, page}`, representing arXiv, History-Book, and Wiki-CS, respectively.
    - `-task` indicates the target downstream task one aims to perform. Possible options are: `{node, edge}`, representing node classification and link prediction, respectively.
    - `-llm` indicates the backbone large language model one aims to use for GLN. Possible options are: `{gpt, claude}`, representing GPT-4o-mini and Claude-3-Haiku, respectively.
      - However, one can easily change the LLM by changing the model name!
    - This code prints the target downstream task performance.

#### Generated results

- The sampled nodes' GLN representations, which can be used for downstream tasks, are provided in [LINK](https://www.dropbox.com/scl/fo/whvxkp4if3zfmenftg5gs/AKl6_L3i5A1klI4BJACU5Lg?rlkey=1zik6ycivp1w7ynd5sfhomujs&st=0w423zig&dl=0).
