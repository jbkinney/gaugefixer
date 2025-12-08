- New organization in different sections parallel to GaugeFixer code organization: features, models, gauges and projection into gauge subspaces.
- Notation problem with orbits and sets of orbits. 
- Marginalizatio property
- Length
- Mention Feature encoding?
- Apendices
- Points in discussion
  - Model fitting vs gauge fixing. Relationships between them but independent. Relationship with regularization and priors. - short looking forward statement citing Sam's paper
  - Applicable to any model fit by different means e.g. GP NN
  - Applicable in product GPs with veyr high number of parameters. 
  - zoom in regions of seq space
- Justin likes to start with the all order models, which is easier to grasp
- Remove some equations. 
- Difference between model fitting and gaugefixing in the introduction.

Library
- get_gauge_fixed_parameters
- Have GaugeFixer be used to fix the gauge of a pd.DataFrame of parameters
- Avoid __call__ method in general. Use more explicit names. 
- Change theta to params throughout.



- Heatmaps for additive component
- Labels for theta as additive, pairwise and cosntant theta
- A just show gauge instead of zero-sum gauge. 
- Merge branches into master, clean up code and ping justin for uploading to pypi. 
- Work on the text to get a workable manuscript.



- Add Sam's new refererences if possible: background averaged and reference-free (was it the zero-sum?)
- function to turn theta series into the matrix for logomaker/heatmap?
- function to turn subsequence into pi_lc values?
- Implement calcution of the dense projection matrix for benchmarking comparisons
- Improve algorith efficiency by removing columns from the linear operators representing the projection matrices for already visited orbits to avoid all the extra computation with the zeros. Doing this could facilitate filling in the dense/sparse projection matrix easily.
- Publish package on PyPI and documentation on RTD

25.11.24 JBK TODOs:
- DONE: `model.get_fixed_params()` now returns gauge-fixed parameters without altering model.theta. Users must call `model.set_params()` to update the model.
- One the other hand, the user should have to pass theta to `model.fixer()`, and that this should return the gauge-fixed thetas rather than alterning an internal state. 
- It would be nice if the user could pass multiple thetas at once to `model.fixer()` a dataframe, and get all of the gauge-fixed parameters back. This would be useful, e.g., for monte carlo sampling.
- I think it would also make sense for model.theta to be able to be a dataframe, rather than just a series, with columns indicating multiple instances of theta. Then when evaluating on a list of sequences, the model it would return a dataframe of predictions instead of just a vector. 
- model.set_random_theta() seems unecessary; won't be used outside of demos. 
- I think the name `model.size` is too ambiguous. Perhaps instead we should use `model.num_seqs`, as well as `model.num_params`. If we allow the model to have multiple instances of theta, also have `model.num_instances`. 
- I think GaugeFixer should not have a theta attribute. 