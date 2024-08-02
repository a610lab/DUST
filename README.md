# DUST
这是论文Self-training with dual uncertainty for semi-supervised MRI image segmentation(DUST)的代码

摘要：
The goal of semi-supervised MRI image segmentation is to train high-accuracy segmentation models using a
small amount of labeled data and a large amount of unlabeled data, thereby alleviating the burden of manual
annotation by physicians. Self-training is a traditional semi-supervised learning method aimed at acquiring
additional image features by generating pseudo-labels for unlabeled data. However, this method may lead
to an unstable training process due to differences between samples. Additionally, the predictions of model
may contain erroneous noise, leading to low-confidence pseudo-labels. To address these issues, we proposed
a dual uncertainty-guided self-training framework (DUST) for semi-supervised MRI image segmentation.
Specifically, the two uncertainties consist of sample-level uncertainty and pixel-level uncertainty. 
The sample-level uncertainty is intended to achieve a stable and smooth training process. The pixel-level uncertainty is
intended to rectify pseudo-labels. We conducted a series of experiments on two public MRI image datasets,
ACDC2017 and Prostate2018. Compared to the baseline, we improved the Dice scores by 5.0% and 4.0% over
the two datasets, respectively. Furthermore, the experimental results indicate that the proposed method has
certain advantages compared to the comparative methods. This validates the feasibility and applicability of
our method.

![流程1](https://github.com/user-attachments/assets/04359f59-fe05-41b7-8254-9d5c544b8bfa)
