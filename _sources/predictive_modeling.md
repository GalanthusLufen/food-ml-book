# Predictive Modeling for Sensory Perception

This chapter explores the application of deep learning and synthetic data generation to predict sensory perception attributes in food systems. By integrating rheological, tribological, and physiological parameters, neural network models can learn complex relationships between instrumental measurements and human sensory responses.

## Introduction to predictive modeling

### The Challenge of Sensory Prediction

Traditional sensory evaluation faces several limitations:

- **Resource intensive** — trained panels, multiple sessions, statistical replication
- **Time consuming** — weeks to months for comprehensive studies
- **Limited scalability** — difficult to screen many formulations rapidly
- **Variability** — inter- and intra-individual differences complicate interpretation

```{figure} sensory.png
---
name: sensory-booths
---
Sensory perception experiments require exntesive preparation and are human and labour intensive.
```

### The Promise of AI/ML

Deep learning offers compelling advantages:

- **Pattern recognition** — captures nonlinear relationships in high-dimensional data
- **Transfer learning** — knowledge from one domain applied to related tasks
- **Scalability** — rapid prediction once trained
- **Integration** — combines multiple data types (rheology, tribology, physiology)

### Current Gaps

Despite potential, deep learning remains underutilized in sensory science due to:

1. **Data scarcity** — small, project-specific datasets
2. **Inconsistent protocols** — varied experimental designs across studies
3. **Reporting gaps** — incomplete documentation of methods and results
4. **Domain complexity** — multifactorial nature of sensory perception

## Synthetic Data for Sensory Science

### Rationale for Synthetic Data

When real-world data is scarce, noisy, or expensive, synthetic datasets provide:

- **Controlled testbed** for model development and validation
- **Privacy preservation** — no human participant data issues
- **Hypothesis testing** — explore theoretical relationships
- **Benchmarking** — evaluate model architectures before real data collection

### Dataset Engineering Principles

Effective synthetic datasets should incorporate:

**Literature-informed trends:**
- Rheological behavior of food systems
- Physiological parameters within normal ranges
- Sensory response patterns consistent with published studies

**Probabilistic modeling:**
- Realistic variability and noise
- Correlated variables where expected
- Session and learning effects

**Expert domain knowledge:**
- Food composition–texture relationships
- Oral physiology–perception linkages
- Ingredient functionality effects

## Dataset Structure for Sensory Prediction

### Input Feature Categories

A comprehensive dataset integrates multiple domains:

**1. Participant Metadata**
```
- Volunteer_ID
- Sex  
- Session (1 or 2)
- Repetition (1, 2, or 3)
```

**2. Food Model Information**
```
- Food_Model (e.g., Cucumber, Banana, Carrot)
- Modification_Type (None, Fungal Polysaccharide, Insect Protein)
- Ingredient_Variant (A, B, or C)
```

**3. Oral Physiology Parameters**
```
- Oral_Cavity_Size (cm³)
- Tongue_Strength (kPa)
- FOP_Time (seconds)
- Saliva_Protein (mg/dL)
- Saliva_Flow (mL/min)
```

**4. Rheological Properties**
```
- Shear viscosity at multiple rates (0.01–1000 s⁻¹)
- Maximum extensional viscosity (mPa·s)
```

**5. Tribological Properties**
```
- Friction_Coefficient
```

### Output Features (Sensory Attributes)

Target variables for prediction (0–10 intensity scale):

| Attribute | Description |
|-----------|-------------|
| Flowable | Ease of flow in mouth |
| Mouthcoating | Residual coating sensation |
| Stringiness | Thread-forming behavior |
| Adhesiveness | Sticking to oral surfaces |
| Cohesiveness | Internal binding strength |
| Creaminess | Smooth, rich sensation |
| Viscosity | Perceived thickness |
| Graininess | Particle sensation |
| Slipperiness | Ease of sliding |
| Ease of Swallowability | Swallowing comfort |

## Embedding Domain Knowledge

### Physiological Relationships

**Sex-based trends** (population-level):
```python
# Example physiological parameter distributions
if sex == 'Female':
    oral_cavity_size = uniform(65, 80)  # cm³
    tongue_strength = uniform(50, 60)    # kPa
    saliva_flow = uniform(0.2, 0.7)      # mL/min
else:  # Male
    oral_cavity_size = uniform(70, 100)
    tongue_strength = uniform(55, 70)
    saliva_flow = uniform(0.4, 1.0)
```

**Sensory–physiology correlations:**
- Higher saliva flow → increased flowability ratings
- Lower saliva protein → increased adhesiveness perception
- Smaller oral cavity → higher perceived viscosity
- Longer FOP time → enhanced mouthcoating

### Ingredient Effects on Rheology

**Fungal polysaccharides:**
- Reduce shear viscosity (60–70%)
- Increase extensional viscosity (25–75%)
- Progressive effect: FPA < FPB < FPC

**Insect proteins:**
- Increase shear viscosity (10–30%)
- Modest extensional viscosity increase (5–15%)
- Bulking effect: IPA < IPB < IPC

### Composition–Sensory Relationships

Food composition influences sensory ratings:

```python
# Conceptual relationships
flowability = f(water_content)  # Higher water → more flowable
mouthcoating = f(1/water_content, viscosity)
adhesiveness = f(sugar_content, viscosity)
graininess = f(fiber_content)
slipperiness = f(1/friction_coefficient)
```

## Neural Network Architecture

### Multi-Layer Perceptron Regressor (MLPR)

Architecture for sensory prediction:

```
Input Layer (64 features after encoding)
    │
    ▼
Hidden Layer 1 (256 neurons)
    ├── Fully Connected
    ├── Batch Normalization  
    ├── ReLU Activation
    ├── Dropout (0.2)
    └── Skip Connection
    │
    ▼
Hidden Layer 2 (128 neurons)
    ├── Fully Connected
    ├── Batch Normalization
    ├── ReLU Activation
    ├── Dropout (0.2)
    └── Skip Connection
    │
    ▼
Hidden Layer 3 (64 neurons)
    ├── Fully Connected
    ├── Batch Normalization
    ├── ReLU Activation
    ├── Dropout (0.1)
    └── Skip Connection
    │
    ▼
Hidden Layer 4 (32 neurons)
    ├── Fully Connected
    ├── Batch Normalization
    ├── ReLU Activation
    ├── Dropout (0.1)
    └── Skip Connection
    │
    ▼
Output Layer (10 sensory attributes)
    └── Sigmoid Activation
```

### Key Architecture Features

**Skip connections:**
- Link input directly to each hidden layer
- Enhance gradient flow during training
- Reduce vanishing gradient problem
- Enable learning of both direct and transformed features

**Batch normalization:**
- Stabilizes training
- Allows higher learning rates
- Provides regularization effect

**Dropout regularization:**
- Prevents overfitting
- Higher rate (0.2) in early layers
- Lower rate (0.1) in later layers

## Model Training

### Data Preprocessing

**One-Hot Encoding** for categorical variables:
```python
# Transform categorical to binary indicators
categorical_columns = [
    'Sex', 'Session', 'Food_Model', 
    'Modification_Type', 'Ingredient_Variant'
]
# Results in expanded feature space
# Original: 48 columns → Encoded: 74 columns
```

**Feature Scaling:**
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()  # Scale to [0, 1]
X_scaled = scaler.fit_transform(X)
```

### Training Configuration

```python
# Hyperparameters
config = {
    'loss_function': 'MSELoss',
    'optimizer': 'AdamW',
    'learning_rate': 0.0001,
    'weight_decay': 0.0005,
    'batch_size': 128,
    'epochs': 50,
    'train_split': 0.80,
    'val_split': 0.10,
    'test_split': 0.10
}
```

### Training Loop

```python
import torch
import torch.nn as nn

def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)
```

## Model Evaluation

### Performance Metrics

**Mean Squared Error (MSE):**
```
MSE = (1/n) Σ(y_true - y_pred)²
```

**Coefficient of Determination (R²):**
```
R² = 1 - (SS_res / SS_tot)
```

### Expected Performance

For well-trained models on comprehensive datasets:

| Metric | Training | Validation | Test |
|--------|----------|------------|------|
| MSE | < 0.05 | < 0.10 | < 0.13 |
| R² | > 0.95 | > 0.92 | > 0.91 |

### Convergence Behavior

Training typically shows:
- Rapid loss decrease in first 10 epochs
- Plateau reached around epoch 20
- Minimal improvement thereafter
- Training, validation, and test curves should converge similarly

## Predictions and Inference

### Known Input Validation

Test model on data from training distribution:

```python
model.eval()
with torch.no_grad():
    # Extract known sample
    sample_input = X_test[0:1]
    true_values = y_test[0]
    
    # Generate prediction
    predicted = model(sample_input)
    
    # Calculate error
    mse = torch.mean((predicted - true_values)**2)
```

Expected: MSE < 0.13 for individual samples

### Transfer Learning on Novel Inputs

Test generalization to unseen conditions:

```python
# Create novel input scenario
novel_input = {
    'Food_Model': 'Celery',
    'Sex': 'Female',
    'Tongue_Strength': 50,  # kPa
    'Viscosity_Profile': 'Newtonian_100mPas',
    'Modification': 'None',
    'Extensional_Viscosity': 1000,
    'Friction_Coefficient': 0.25
}

# Model predicts sensory ratings
# Even for combinations not in training data
```

### Interpretation of Predictions

The trained model captures:

1. **Ingredient effects** — FP reduces perceived viscosity, IP increases it
2. **Rheology–sensory links** — shear-thinning affects flowability
3. **Physiological modulation** — tongue strength impacts swallowability
4. **Food matrix interactions** — base food type influences all attributes

## Sensory Panel Validation

### Fisher F-Ratio Analysis

Assess synthetic panel discrimination:

```python
# ANOVA approach
F_ratio = MS_between / MS_within

# MS_between: variance due to sample differences
# MS_within: random error / intra-panelist variability
```

**Interpretation:**
- F > 2: Significant discrimination (p < 0.05)
- F > 5: Strong discrimination (p < 0.001)
- F > 100: Excellent discrimination

### Panel Consistency

Low within-group variance indicates:
- Reliable scoring across sessions
- Consistent panelist behavior
- Reproducible results

## Applications

### Ingredient Screening

Rapid evaluation of novel ingredients:
1. Input ingredient rheological profile
2. Predict sensory impact across attributes
3. Prioritize candidates for physical testing
4. Reduce experimental burden

### Formulation Optimization

Iterative improvement:
```
While sensory_target not met:
    Adjust formulation parameters
    Predict sensory profile
    Evaluate against targets
    Refine parameters
```

### Consumer Response Simulation

Model different consumer segments:
- Age-related physiological variation
- Sex-based differences
- Cultural/regional preferences

### Dysphagia Food Design

Predict safe-swallowing attributes:
- Ease of swallowability scores
- Cohesiveness requirements
- Optimal viscosity ranges

## Limitations and Future Directions

### Current Limitations

1. **Synthetic data basis** — requires validation with real experiments
2. **Hidden biases** — imposed probabilistic rules may not reflect reality
3. **Generalization uncertainty** — cross-dataset performance unknown
4. **Mechanistic opacity** — neural networks as black boxes

### Path Forward

**Experimental validation:**
- Collect real sensory data with matched instrumental measurements
- Compare model predictions to actual panel ratings
- Refine synthetic data generation rules

**Transfer learning:**
- Pre-train on synthetic data
- Fine-tune on limited real data
- Evaluate domain adaptation strategies

**Model interpretability:**
- SHAP values for feature importance
- Attention mechanisms for transparency
- Partial dependence plots

**Integration opportunities:**
- Combine with chemometrics (taste, aroma)
- Include visual attributes
- Link to consumer liking models

## Summary

Deep learning offers a powerful framework for predicting sensory perception from instrumental and physiological data. Key contributions of this approach include:

1. **Synthetic data methodology** — literature-informed dataset construction
2. **Multi-modal integration** — rheology, tribology, physiology, sensory
3. **Neural network architecture** — MLPR with skip connections for regression
4. **Transfer learning potential** — generalization to novel conditions

This framework provides a proof-of-concept for accelerating food product development through:
- Rapid ingredient screening
- Virtual prototyping
- Consumer response simulation
- Reduced experimental burden

## References

1. Chen, J. (2020). It is important to differentiate sensory property from the material property. *Trends in Food Science & Technology*, 96, 268–270.
2. Glumac, M., Avila-Sierra, A., & Mishyna, M. (2026). Synthetic data and deep neural networks enable prediction of sensory perception attributes in texture-modified plant-based smoothies. *Food Hydrocolloids*, 174, 112324.