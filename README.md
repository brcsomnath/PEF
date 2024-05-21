# Perfect Erasure Functions

This repository contains the implementation of NeurIPS 2024 submission tited:

> Fundamental Limits of Perfect Concept Erasure

## Overview

Concept erasure is the task of erasing information about a concept (e.g., gender or race) from a representation set while retaining the maximum possible utility -- information from original representations. Concept erasure is useful in several applications, such as removing sensitive concepts to achieve fairness or understanding the impact of a specific concept on a model's performance for interpretability research.
  Previous concept erasure techniques have prioritized robustly erasing concepts without considering the utility of the resultant representations. However, there appears to be an inherent tradeoff between erasure and retaining utility~\citep{kram}, and determining the optimal approach for achieving perfect concept erasure while maintaining high utility remains unclear. In this paper, we offer a fresh perspective toward solving this problem by quantifying the fundamental limits of concept erasure through an information-theoretic lens and formulating erasure functions that achieve these limits. We begin by formalizing the concept erasure task and present the optimal mutual information bounds achievable during concept erasure. We then analytically derive conditions on data distribution and formulate erasure functions that achieve the optimal bounds for perfect concept erasure. Empirically, we demonstrate that our proposed approach attains the optimal theoretical bounds. Additionally, we also show that our approach outperforms existing methods on a range of synthetic and real-world datasets using GPT-4 representations.


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
