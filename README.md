<!-- #region -->
# Perfect Erasure Functions

[![License: MIT](https://img.shields.io/badge/License-MIT-green``.svg)](https://opensource.org/licenses/MIT)

We present the implementation of the AISTATS 2025 paper:
> [**Fundamental Limits of Perfect Concept Erasure**](https://arxiv.org/abs/2503.20098), <br>
[Somnath Basu Roy Chowdhury](https://www.cs.unc.edu/~somnath/)<sup>1</sup>, [Kumar Avinava Dubey](https://scholar.google.co.in/citations?user=tBbUAfsAAAAJ&hl=en)<sup>1</sup>, [Ahmad Beirami](https://sites.google.com/view/beirami)<sup>3</sup>, [Nicholas Monath](https://scholar.google.co.in/citations?user=PTfhfCQAAAAJ&hl=en)<sup>3</sup>, [Amr Ahmed](https://scholar.google.co.in/citations?user=ivUi2T0AAAAJ&hl=en)<sup>2</sup>, and [Snigdha Chaturvedi](https://sites.google.com/site/snigdhac/)<sup>1</sup>. <br>
<sup>1</sup>UNC Chapel Hill, <sup>2</sup>Google Research, <sup>3</sup>Google DeepMind.


## Overview

Concept erasure is the task of erasing information about a concept (e.g., gender or race) from a representation set while retaining the maximum possible utility -- information from original representations. Concept erasure is useful in several applications, such as removing sensitive concepts to achieve fairness or understanding the impact of a specific concept on a model's performance for interpretability research.
  Previous concept erasure techniques have prioritized robustly erasing concepts without considering the utility of the resultant representations. However, there appears to be an inherent tradeoff between erasure and retaining utility (Chowdhury et al., 2024), and determining the optimal approach for achieving perfect concept erasure while maintaining high utility remains unclear. In this paper, we offer a fresh perspective toward solving this problem by quantifying the fundamental limits of concept erasure through an information-theoretic lens and formulating erasure functions that achieve these limits. We begin by formalizing the concept erasure task and present the optimal mutual information bounds achievable during concept erasure. We then analytically derive conditions on data distribution and formulate erasure functions that achieve the optimal bounds for perfect concept erasure. Empirically, we demonstrate that our proposed approach attains the optimal theoretical bounds. Additionally, we also show that our approach outperforms existing methods on a range of synthetic and real-world datasets using GPT-4 representations.
<!-- #endregion -->

  ## Installation
The simplest way to run our implementation is to create with a new conda environment.
```
conda create -n pef python=3.10
source activate pef
pip install -r requirements.txt
```

## Running PEF

To run PEF, use the following command:

```
cd src/
python main.py --dataset <name>
```

The dataset names can be `glove` or `synthetic`. 


## Reference

If you use our work, please cite our paper as:
```
@inproceedings{
    chowdhury2025fundamental,
    title={Fundamental Limits of Perfect Concept Erasure},
    author={
        Somnath Basu Roy Chowdhury and
        Kumar Avinava Dubey and 
        Ahmad Beirami and 
        Rahul Kidambi and 
        Nicholas Monath and 
        Amr Ahmed and 
        Snigdha Chaturvedi},
    booktitle={The 28th International Conference on Artificial Intelligence and Statistics},
    year={2025},
    url={https://openreview.net/forum?id=bppVexkY5N}
}
```
