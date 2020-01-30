# Fast Subclass Discriminant Analysis

This repository provides the codes for the paper "Speed-up and Multi-view extensions to Subclass Discriminant Analysis", preprint available [here].

For single-view fastSDA the following files are used:
* *calculate_targets_singleview.m* - calculates target vector matrix T given class and subclass labels
* *get_fastSDA_kernel_results.m* - applies kernel fastSDA given data matrix X and labels. Note that X should be sorted according to classes/subclasses	
* *get_fastSDA_linear_results.m* 	- applies linear fastSDA given the kernel matrix K and labels. Note that data should be sorted according to classes/subclasses

The same logic is followed for multi-view case:
* *calculate_target_vectors_multiview.m* 
* *get_fastSDA_multiview_kernel_results.m* 	
* *get_fastSDA_multiview_linear_results.m*
### 
If you find our work useful, please cite it as:
```
@article{chumachenko2019speed,
  title={Speed-up and multi-view extensions to Subclass Discriminant Analysis},
  author={Chumachenko, Kateryna and Raitoharju, Jenni and Iosifidis, Alexandros and Gabbouj, Moncef},
  journal={arXiv preprint arXiv:1905.00794},
  year={2019}
}
```
[here]: <https://arxiv.org/abs/1905.00794>
