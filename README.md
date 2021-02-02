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

Additionally, there is now a helper function fastSDA.m to which you can just supply your train data, labels and desired number of dimensions and obtain the projection matrix.

### 
If you find our work useful, please cite it as:
```
@article{CHUMACHENKO2021107660,
title = "Speed-up and multi-view extensions to subclass discriminant analysis",
journal = "Pattern Recognition",
volume = "111",
pages = "107660",
year = "2021",
issn = "0031-3203",
doi = "https://doi.org/10.1016/j.patcog.2020.107660",
url = "http://www.sciencedirect.com/science/article/pii/S0031320320304635",
author = "Kateryna Chumachenko and Jenni Raitoharju and Alexandros Iosifidis and Moncef Gabbouj"
}
```
[here]: <https://arxiv.org/abs/1905.00794>
