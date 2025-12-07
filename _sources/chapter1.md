# **Undergraduate Course: Food Oral Processing, Sensory Perception Modeling and Artificial Intelligence**

## Introduction

Food oral processing involves complex interactions between food structure, saliva, and oral tissues. 
Artificial intelligence offers powerful tools for predicting sensory outcomes from physical measurements.

## Key Concepts

### Tribology in Oral Processing

The friction coefficient $\mu$ between tongue and palate surfaces is influenced by:

$$
\mu = f(\eta, v, F_n, \sigma)
$$

Where:
- $\eta$ = viscosity
- $v$ = sliding velocity  
- $F_n$ = normal force
- $\sigma$ = surface roughness

### Neural Network Approach

TabTransformer and similar architectures can map physical parameters to sensory scores:

| Input Features | Output |
|---------------|--------|
| Viscosity | Thickness perception |
| Particle size | Grittiness |
| Fat content | Creaminess |
| Friction coefficient | Smoothness |

## Model Performance Metrics

Based on recent work with texture-modified plant-based smoothies:

- **MSE**: < 0.13
- **Test R²**: 0.91

These results demonstrate the potential for synthetic data augmentation in sensory science.

## References

1. Glumac, M. et al. (2025). Food Hydrocolloids - Synthetic data and neural networks for sensory prediction.