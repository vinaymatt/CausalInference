# Causal Inference Project
This repository contains the code, data, and results of a Causal Inference project I completed as part of my coursework under Professor Vasant Honavar. The project extensively applies concepts of counterfactual neural networks (CFRNet) and Gaussian processes for estimating Average Treatment Effects (ATE) in longitudinal data.

## Overview
In this project, a causal inference model was implemented to estimate the Average Treatment Effect (ATE) from longitudinal data, based on counterfactual neural networks (CFRNet) and Gaussian processes. The model was trained on simulated data, following a specific Data Generating Process (DGP).

## Code Summary
The core of the project is represented by the `CFRModel` class, a PyTorch implementation of a Counterfactual Regression Model. This model takes in treatment and control data, learns shared representations, and generates treated and control outcomes.

The code includes the implementation of a Gaussian process model (using the GPyTorch library) and the loss function for CFRNet. The loss function combines factual loss, balance term, Gaussian process regularisation, and a Wasserstein term, to handle the challenges posed by confounding variables, imbalanced treatment populations, and non-linearity and multi-dimensionality of treatment effects, respectively.

In addition, the Wasserstein distance was computed to measure the distributional distance between the predicted potential outcomes of treated and untreated groups, which plays a crucial role in regularizing the predicted treatment effects.

A complete simulation workflow is provided in the script, including data preparation, training, and evaluation stages. The performance of the model is evaluated on various metrics, including the Mean Squared Error (MSE), the Precision in Estimating Heterogeneous Effects (PEHE), and the error in estimated Average Treatment Effect (ATE).

## Data Generating Process
The data was generated following the process outlined below:
1. Generate base features B ∈ RN×D, where N is the number of individuals and D is the number of base features.
2. Compute the covariate matrix X ∈ RN×K using an encoder network, where K is the number of covariates.
3. Generate time steps using a Poisson distribution τ ∈ NN×T, where T is the number of time steps for each individual.
4. Assign treatment T ∈ {0, 1}N based on a propensity score function.
5. Generate potential outcomes Y1 and Y0 using two separate outcome networks.
6. Compute observed outcomes Y = T⊙Y1 +(1−T)⊙ Y0.
7. Simulate longitudinal and multilevel correlation in the data using the time steps and by splitting the individuals into clusters.
8. Add the error terms to the observed outcomes.

For the simulations in this project, ten base features were generated for 40 individuals, each with 30 covariates—the data spanned 20-time steps for each individual.

## Acknowledgements
I would like to express my gratitude to Professor Vasant Honavar for his guidance and the numerous insightful discussions we had throughout this project. Many of the strategies implemented in this work for dealing with longitudinal data were gleaned from our conversations.
