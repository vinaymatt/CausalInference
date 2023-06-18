# Causal Inference for Longitudinal Data
This repository contains the code, data, and results of a Causal Inference project I completed as part of my coursework under Professor Vasant Honavar. The project extensively applies concepts of counterfactual neural networks (CFRNet) and Gaussian processes for estimating Average Treatment Effects (ATE) in longitudinal data.

## Overview
In this project, a causal inference model was implemented to estimate the Average Treatment Effect (ATE) from longitudinal data, based on the counterfactual neural network (CFRNet). The model was verified on simulated data, following a Data Generating Process (DGP) given below. Since the CFRNet is a widely used estimand for causal effect estimation, the overarching idea was to model the temporal part of the data as a Gaussian Process to estimate causal effect. This is a small attempt to a very complex and important problem of estimating causal effects when dealing with longitudinal data. 

## Code Summary
The CFRNet is represented by the `CFRModel` class, a PyTorch implementation of a Counterfactual Regression Model introduced by Shalit et al. This model takes in treatment and control data, learns shared representations, and generates treated and control outcomes.

The core contribution of the code includes the implementation of a Gaussian process model (using the GPyTorch library) and the loss function for CFRNet. The loss function combines factual loss, balance term, Gaussian process regularisation, and a Wasserstein term, to handle the challenges posed by longitudinal data, confoudning variables, multilevel correlations, imbalanced treatment populations, and non-linearity and multi-dimensionality of treatment effects, respectively.

In addition, the Wasserstein distance (IPM) was used to measure the distributional distance between the predicted potential outcomes of treated and untreated groups, which plays a crucial role in regularizing the predicted treatment effects.

A complete simulation workflow is provided in the script, including data preparation, training, and evaluation stages. The performance of the model is evaluated on various metrics, including the Mean Squared Error (MSE), the Precision in Estimating Heterogeneous Effects (PEHE), and the error in estimated Average Treatment Effect (ATE).

The DGP as disccused below can be found in the `DGP.py` file in the directory.

## Data Generating Process
The synthetic longitudinal data was generated using a specific Data Generating Process (DGP). The following steps outline the process:

1. Randomly generated base features for the desired number of individuals and observations were fed into an encoder network, which is a sequence of linear and non-linear (Tanh) transformations with batch normalization and dropout for regularization. This network transformed the base features into covariates.
   
2. A propensity score was calculated for each individual using the sigmoid function applied to the first three covariates. A Bernoulli distribution based on these propensity scores was then used to assign the treatment status for each individual.

3. Two separate outcome networks were used to generate potential outcomes for the treated (Y1) and control (Y0) individuals. Each outcome network included a series of linear and Tanh non-linear transformations.

4. The observed outcomes were calculated based on the treatment assignment and the potential outcomes.

5. An autoregressive error term was added to the outcomes to simulate longitudinal correlation. The error term was generated based on an AR(1) process with a decay factor, and the time steps for each individual. The absolute difference in time steps determined the level of correlation in the error term.

6. To introduce multilevel correlation, individuals were randomly assigned to clusters, and an additional error term was added for individuals in the same cluster.

7. The error terms were combined into a covariance matrix, and the Cholesky decomposition of the covariance matrix was used to simulate correlated errors.

8. The correlated errors were added to the observed and potential outcomes.

9. The process was repeated for 1000 simulations, each resulting in an individual dataset.

The Average Treatment Effect (ATE) was calculated as the mean difference between the potential outcomes Y1 and Y0 for all individuals in each simulation. The Average Treatment Effect on the Treated (ATT) was also calculated for individuals who received the treatment. 

## Acknowledgements
I would like to express my gratitude to Professor Vasant Honavar for his guidance and the numerous insightful discussions we had throughout this project. Many of the strategies implemented in this work for dealing with longitudinal data were gleaned from our conversations.
