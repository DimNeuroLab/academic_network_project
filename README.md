## Academic Network Project

### ANP Core

The `anp_core` module is the foundation of the ANP system: it manages datasets, builds and extends the academic (hindight) infosphere, and offers a range of utilities for data processing and analysis within the Academic Network Project. Key functions include loading datasets, assembling the infosphere, expanding its scope, and other essential processing tools.

A standard execution flow is:

1. Import the AMiner dataset into PyG and parse its contents.
2. Use `anp_infosphere_creation` to construct the infosphere, with options to split the work into multiple parts for parallel processing.
3. For each part, call `anp_infosphere_expansion_caller` to run the `anp_expansion` routines.
4. Combine all expanded segments by running `anp_infosphere_builder`.

---

### ANP NN

The `anp_nn` package contains Graph Neural Network models designed for prediction within the ANP framework. It currently includes:

* **Co-Author Prediction**: A model based on Heterogeneous Graph Transformers (HGT) to predict future collaborative links among researchers.

* **Synthetic Ground-Truth Generation**: Creates simulated interaction datasets using predefined recommender strategies. It:

  1. Trains a Recommender-Neutral User (RNU) model on historical network data.
  2. Simulates various recommendation approaches (e.g., no infosphere, hindsight infosphere, top-paper, top-paper × topic, LightGCN).
  3. Generates labeled interaction pairs according to each approach’s logic.

This addition allows for controlled benchmarking of models and recommender detection methods against known synthetic ground-truths.

For the experiments in the paper, a learning rate of 0.00001 and 50 sampled edges for minibatch creation were used as hyperparameters. The other hyperparameters remain fixed and can be found in the code.


#### Experimental Procedure

To run the experiments, after parsing the dataset and generating the *hindsight infosphere*, follow the steps below.

1. Train the RNU using the script `anp_link_prediction_co_author_hgt.py` and generate the corresponding weights.
2. Create a Ground Truth (GT) for each available infosphere by running `create_GT/create_gt_opt.py`.
3. Run the experiments for each generated GT and each infosphere using the script `anp_link_prediction_co_author_hgt_GT.py`, applying the parameters specified here and in the paper.

---
