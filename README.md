# Runtime Optimization of Widrow-Haff Classification Algorithm Using Proper Learning Samples 

### Link to the paper:
- https://www.researchgate.net/publication/319015521_Runtime_Optimization_of_Widrow-Hoff_Classification_Algorithm_Using_Proper_Learning_Samples
- Or
- https://civilica.com/doc/668873/
### Please cite:
- Mirinezhad, S. Younes, et al. "Runtime Optimization of Widrow-Haff Classification Algorithm Using Proper Learning Samples." 4th National Conference on Information Technology, Computer & Telecommunication. Mashhad, Iran. 2017.
- Or
- Dezfoulian, Mir-Hossein and Mirinezhad, S. Yunes and Mousavi, S. M. Hussein and Shafeii Mosleh, Mehrdad,1396,Runtime Optimization of Widrow-Haff Classification Algorithm Using Proper Learning Samples,4th National Conference on Information Technology?Computer & Telecommunication ,Mashhad,https://civilica.com/doc/668873

## Overview
This repository provides an implementation of the methodology described in the paper: **"Runtime Optimization of Widrow-Hoff Classification Algorithm Using Proper Learning Samples."** The paper introduces a novel combination of the Widrow-Hoff classification algorithm and the Multi-Class Instance Selection (MCIS) method to optimize the classification runtime and accuracy.

---

## Key Contributions of the Paper

1. **Widrow-Hoff Algorithm**:
   - A fast and efficient linear classification algorithm.
   - Sensitive to outliers and noisy data, which can impact its performance.

2. **MCIS (Multi-Class Instance Selection)**:
   - Pre-selects the most relevant data samples for classification.
   - Reduces the dataset size by removing noisy or outlier samples.
   - Improves runtime efficiency and accuracy when used with Widrow-Hoff.

3. **Combination**:
   - Integrating MCIS with Widrow-Hoff significantly improves runtime performance while maintaining or enhancing classification accuracy.

## Methodology

1. **MCIS Algorithm**:
   - Clusters positive class samples using K-Means.
   - Filters outlier samples from the negative class based on their distance from cluster centers.
   - The filtered dataset is passed to the classifier.

2. **Widrow-Hoff Algorithm**:
   - A linear classifier that iteratively updates weights using the Widrow-Hoff learning rule:
     \( w = w + \eta (y - \hat{y})x \)
   - Trained on the filtered dataset from MCIS.



![image](https://github.com/user-attachments/assets/a1b1a98b-020e-4724-b5b8-d470dac5d4e5)
![image](https://github.com/user-attachments/assets/a542cb8a-e843-4c99-9470-a87b6fd0f20a)
![image](https://github.com/user-attachments/assets/97f0a081-a7a3-4491-9a1e-278d571fcebe)
