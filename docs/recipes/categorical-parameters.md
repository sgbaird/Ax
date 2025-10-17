# Categorical Parameters in Ax and BoTorch

This document explains how Ax and BoTorch implement and treat categorical parameters from a modeling perspective.

## Introduction

Categorical parameters represent discrete, unordered choices in an optimization problem. Unlike ordered choices (e.g., small/medium/large) or continuous parameters, categorical parameters don't have a natural ordering or numerical representation.

Examples of categorical parameters include:
- Algorithm choices (e.g., "adam", "sgd", "rmsprop")
- Model architectures (e.g., "resnet", "vgg", "inception")
- Activation functions (e.g., "relu", "tanh", "sigmoid")

## Key Finding: Sophisticated Categorical Kernels Are the Default

**Contrary to common assumption, one-hot encoding is NOT the default** for categorical parameters in Ax. Instead, Ax uses a more sophisticated approach with **categorical kernels** (via `MixedSingleTaskGP` with `CategoricalKernel`) when the number of categorical options is reasonable.

One-hot encoding is only used as a fallback when there are many categorical parameters that would be expensive to handle with categorical kernels.

## Parameter Representation

In Ax, categorical parameters are represented by the [`ChoiceParameter`](https://github.com/sgbaird/Ax/blob/copilot/analyze-categorical-parameters/ax/core/parameter.py#L611-L880) class with `is_ordered=False`.

```python
from ax.core.parameter import ChoiceParameter, ParameterType

categorical_param = ChoiceParameter(
    name="optimizer",
    parameter_type=ParameterType.STRING,
    values=["adam", "sgd", "rmsprop"],
    is_ordered=False,  # This makes it categorical
)
```

### Ordered vs. Unordered Choice Parameters

- **Ordered** (`is_ordered=True`): Values have a natural ordering and can be treated as ordinal (e.g., [0, 1, 2, ...])
- **Unordered/Categorical** (`is_ordered=False`): Values have no inherent ordering and require special handling with categorical kernels or encoding

Note: Choice parameters with exactly 2 values are automatically treated as ordered, regardless of the `is_ordered` setting.

## Automatic Strategy Selection

Ax automatically selects the best modeling strategy based on the search space characteristics. The selection logic is in [`ax/generation_strategy/dispatch_utils.py`](https://github.com/sgbaird/Ax/blob/copilot/analyze-categorical-parameters/ax/generation_strategy/dispatch_utils.py#L134-L260).

### Strategy 1: BO_MIXED (Default for Few Categorical Parameters)

**Used when:**
- Number of ordered parameters < number of unordered categorical choices, AND
- Total discrete combinations ≤ 65, OR
- All parameters are discrete AND total combinations < 10,000

**Benefits:**
- Uses [`MixedSingleTaskGP`](https://github.com/pytorch/botorch/blob/main/botorch/models/gp_regression_mixed.py) with **categorical kernels** (e.g., Hamming distance)
- No continuous relaxation needed - parameters stay discrete
- More efficient acquisition function optimization via enumeration
- Better performance for categorical variables than one-hot encoding

**Transform:** [`ChoiceToNumericChoice`](https://github.com/sgbaird/Ax/blob/copilot/analyze-categorical-parameters/ax/adapter/transforms/choice_encode.py#L25-L149) - maps categorical values to integers without one-hot expansion

**Example:** 3 categorical parameters with 5 options each (125 combinations) → Uses BO_MIXED

### Strategy 2: BOTORCH_MODULAR with OneHot (Fallback for Many Categorical Parameters)

**Used when:**
- Number of ordered parameters ≥ number of unordered categorical choices, OR
- At least one ordered parameter AND number of one-hot encodings < 33

**Transform:** [`OneHot`](https://github.com/sgbaird/Ax/blob/copilot/analyze-categorical-parameters/ax/adapter/transforms/one_hot.py#L58-L241) - expands categorical parameters into continuous parameters

**Limitation:**
- Treats categorical parameters as continuous via one-hot encoding
- Can be less efficient with many categorical options
- Used when categorical kernel approach would be too expensive

**Example:** 40 categorical parameters → Uses BOTORCH_MODULAR with OneHot encoding

### Strategy 3: Sobol (Fallback for Too Many Combinations)

When the search space has >65 discrete combinations and >33 one-hot parameters, Ax falls back to Sobol sampling instead of Bayesian optimization.

## Transformation Pipeline Details

### ChoiceToNumericChoice (Used by BO_MIXED)

The [`ChoiceToNumericChoice`](https://github.com/sgbaird/Ax/blob/copilot/analyze-categorical-parameters/ax/adapter/transforms/choice_encode.py#L25-L149) transform maps categorical values to integers (0, 1, ..., n-1) **without** one-hot expansion. This preserves the discrete nature of the parameter and allows BoTorch to use categorical kernels.

**Example:**
```python
# Original parameter
ChoiceParameter(name="method", values=["a", "b", "c"])

# After ChoiceToNumericChoice becomes a single ChoiceParameter:
# method: ChoiceParameter with values [0, 1, 2] (type INT)
```

The model then uses `MixedSingleTaskGP` with a `CategoricalKernel` that can properly handle discrete categorical variables.

### OneHot Encoding (Used by BOTORCH_MODULAR)

The [`OneHot`](https://github.com/sgbaird/Ax/blob/copilot/analyze-categorical-parameters/ax/adapter/transforms/one_hot.py#L58-L241) transform converts categorical parameters into continuous parameters via one-hot encoding:

1. **For binary choices** (2 values): Creates a single `RangeParameter` with values in [0, 1]
2. **For multi-way choices** (3+ values): Creates multiple `RangeParameter`s, one for each value, forming a one-hot encoding

**Example:**
```python
# Original parameter
ChoiceParameter(name="method", values=["a", "b", "c"])

# After OneHot transform becomes three RangeParameters:
# method_OH_PARAM_0: [0, 1]  # encodes "a"
# method_OH_PARAM_1: [0, 1]  # encodes "b"
# method_OH_PARAM_2: [0, 1]  # encodes "c"
```

### Inverse Transform: Rounding (OneHot Only)

When models using one-hot encoding generate continuous suggestions, the `OneHot` transform uses rounding to convert back to categorical values:

- **Strict rounding** (default): Choose the category with the maximum value, breaking ties randomly
- **Randomized rounding**: Sample from the distribution defined by the continuous values

See the implementation in [`ax/adapter/transforms/rounding.py`](https://github.com/sgbaird/Ax/blob/copilot/analyze-categorical-parameters/ax/adapter/transforms/rounding.py).

### Transform Configurations

Different models use different transform chains, defined in [`ax/adapter/registry.py`](https://github.com/sgbaird/Ax/blob/copilot/analyze-categorical-parameters/ax/adapter/registry.py):

1. **`Mixed_transforms`** (lines 135-140) - Used by BO_MIXED:
   - `RemoveFixed`
   - `ChoiceToNumericChoice` ← Keeps parameters discrete
   - `Log`
   - `Logit`

2. **`MBM_X_trans`** (lines 101-108) - Used by BOTORCH_MODULAR:
   - `RemoveFixed`
   - `OrderedChoiceToIntegerRange`
   - `OneHot` ← Expands to continuous parameters
   - `Log`
   - `Logit`

## Communication with Models

After transformation, the encoded parameters are communicated to models via [`SearchSpaceDigest`](https://github.com/sgbaird/Ax/blob/copilot/analyze-categorical-parameters/ax/core/search_space.py#L1205-L1252), which contains:

- **`feature_names`**: List of parameter names
- **`bounds`**: Lower and upper bounds for each parameter
- **`categorical_features`**: List of indices indicating which features are categorical
- **`ordinal_features`**: List of indices for ordinal parameters
- **`discrete_choices`**: Mapping of parameter indices to their discrete values

The digest is created by [`extract_search_space_digest`](https://github.com/sgbaird/Ax/blob/copilot/analyze-categorical-parameters/ax/adapter/adapter_utils.py#L179-L290) which:
1. Identifies categorical parameters as `ChoiceParameter`s that are **not** ordered and **not** task parameters
2. Records their indices in `categorical_features`
3. Adds their numeric choices to `discrete_choices`

## BoTorch Model Integration

The choice of transform determines which BoTorch model is used. Model selection logic is in [`ax/generators/torch/botorch_modular/utils.py`](https://github.com/sgbaird/Ax/blob/copilot/analyze-categorical-parameters/ax/generators/torch/botorch_modular/utils.py#L290-L303).

### When BO_MIXED is Selected (Categorical Features Present)

When a BoTorch model is constructed and `categorical_features` are present in the `SearchSpaceDigest`, Ax uses [`MixedSingleTaskGP`](https://github.com/pytorch/botorch/blob/main/botorch/models/gp_regression_mixed.py):

```python
# From ax/generators/torch/botorch_modular/utils.py, lines 295-296
elif search_space_digest.categorical_features:
    model_class = MixedSingleTaskGP
```

`MixedSingleTaskGP` uses a **categorical kernel** (such as a Hamming distance-based kernel) that properly respects the discrete, unordered nature of categorical variables. This is more principled than treating one-hot encoded parameters as continuous.

### When BOTORCH_MODULAR is Used (OneHot Encoding)

When one-hot encoding is used, the `categorical_features` indices are passed to the model as kwargs. See [`ax/generators/torch/botorch_modular/surrogate.py`](https://github.com/sgbaird/Ax/blob/copilot/analyze-categorical-parameters/ax/generators/torch/botorch_modular/surrogate.py#L141-L143):

```python
if len(search_space_digest.categorical_features) > 0:
    kwargs["categorical_features"] = search_space_digest.categorical_features
```

However, with continuous relaxation via one-hot encoding, the model is typically `SingleTaskGP` treating the parameters as continuous.

## Understanding the Categorical Kernel (Hamming Distance)

When `MixedSingleTaskGP` is used, BoTorch employs a **product kernel** that combines different kernel types for different parameter types:

### Kernel Structure

The `MixedSingleTaskGP` model uses a composite kernel structure:

1. **Continuous Parameters**: Standard kernel (e.g., Matérn or RBF) for continuous/ordinal dimensions
2. **Categorical Parameters**: Hamming distance-based kernel for categorical dimensions
3. **Product Kernel**: The overall covariance is the product of these kernels

### Hamming Distance Kernel

The Hamming distance kernel measures similarity between categorical values:

- **Same category** (distance = 0): High covariance (values are considered similar)
- **Different category** (distance = 1): Low covariance (values are considered dissimilar)

**Mathematical form:**
```
k_categorical(x, x') = exp(-d_H(x, x'))
```

Where `d_H(x, x')` is the Hamming distance:
- `d_H(x, x') = 0` if x and x' have the same categorical value
- `d_H(x, x') = 1` if x and x' have different categorical values

### Why Hamming Distance is Better Than One-Hot

**One-hot encoding problems:**
- Creates artificial geometric relationships between unrelated categories
- Increases dimensionality unnecessarily
- Treats the continuous relaxation of discrete variables as meaningful

**Hamming distance advantages:**
- Respects the discrete, unordered nature of categorical variables
- No artificial geometric structure imposed
- More statistically principled for Gaussian Process modeling
- Better handles the correlation structure of categorical parameters

### Implementation Details

From the test code in [`ax/generators/torch/tests/test_surrogate.py`](https://github.com/sgbaird/Ax/blob/copilot/analyze-categorical-parameters/ax/generators/torch/tests/test_surrogate.py#L1731-L1760), we can see:

```python
# When categorical_features=[0] is specified:
surrogate.model.covar_module.kernels[0].base_kernel.kernels[1].active_dims  # [0] - categorical
surrogate.model.covar_module.kernels[0].base_kernel.kernels[0].active_dims  # [1,2] - continuous
```

This shows the kernel is split:
- One kernel component operates on categorical dimensions (e.g., dimension 0)
- Another kernel component operates on continuous dimensions (e.g., dimensions 1, 2)
- These are combined via a product kernel

### Example: How It Works

Consider a parameter with 3 categories: ["method_a", "method_b", "method_c"]

**After ChoiceToNumericChoice transform:** `[0, 1, 2]`

**Kernel evaluation:**
- `k_cat(0, 0) = exp(0) = 1.0` (same method)
- `k_cat(0, 1) = exp(-1) ≈ 0.37` (different methods)
- `k_cat(1, 2) = exp(-1) ≈ 0.37` (different methods)

The model learns that observations with the same categorical value are highly correlated, while different categorical values have reduced (but non-zero) correlation.

### BoTorch Implementation

The actual implementation is in BoTorch's [`MixedSingleTaskGP`](https://github.com/pytorch/botorch/blob/main/botorch/models/gp_regression_mixed.py) class, which:

1. Accepts `categorical_features` as a list of dimension indices
2. Constructs separate kernel components for categorical vs. continuous dimensions
3. Uses `active_dims` to specify which dimensions each kernel operates on
4. Combines kernels via multiplication to get the final covariance function

## Multi-Task Learning

Categorical parameters can also be used as **task parameters** for multi-task learning. Task parameters are a special type of categorical parameter where:

- `is_task=True` is set on the `ChoiceParameter`
- A `target_value` must be specified
- The parameter index is added to `task_features` instead of `categorical_features` in the `SearchSpaceDigest`

For more information on multi-task learning, see the [multi-task tutorial](https://honegumi.readthedocs.io/en/latest/curriculum/concepts/multitask/multitask.html) referenced in the original issue.

## Key Code References

- **Parameter definition**: [`ax/core/parameter.py`](https://github.com/sgbaird/Ax/blob/copilot/analyze-categorical-parameters/ax/core/parameter.py)
  - `ChoiceParameter` class (lines 611-880)
- **One-hot encoding**: [`ax/adapter/transforms/one_hot.py`](https://github.com/sgbaird/Ax/blob/copilot/analyze-categorical-parameters/ax/adapter/transforms/one_hot.py)
  - `OneHot` transform class (lines 58-241)
  - `OneHotEncoder` helper class (lines 32-56)
- **Alternative encodings**: [`ax/adapter/transforms/choice_encode.py`](https://github.com/sgbaird/Ax/blob/copilot/analyze-categorical-parameters/ax/adapter/transforms/choice_encode.py)
  - `ChoiceToNumericChoice` transform (lines 25-149)
  - `OrderedChoiceToIntegerRange` transform (lines 151-242)
- **Search space digest**: 
  - [`ax/core/search_space.py`](https://github.com/sgbaird/Ax/blob/copilot/analyze-categorical-parameters/ax/core/search_space.py) - `SearchSpaceDigest` class (lines 1205-1252)
  - [`ax/adapter/adapter_utils.py`](https://github.com/sgbaird/Ax/blob/copilot/analyze-categorical-parameters/ax/adapter/adapter_utils.py) - `extract_search_space_digest` function (lines 179-290)
- **Model integration**: [`ax/generators/torch/botorch_modular/surrogate.py`](https://github.com/sgbaird/Ax/blob/copilot/analyze-categorical-parameters/ax/generators/torch/botorch_modular/surrogate.py)
  - Categorical features passed to model (lines 141-143)

## Summary

Ax treats categorical parameters with a sophisticated, context-aware approach:

1. **Default (Few Categorical Parameters):** Uses `BO_MIXED` with `MixedSingleTaskGP` and **categorical kernels**
   - Transform: `ChoiceToNumericChoice` (keeps parameters discrete as integers)
   - Model: `MixedSingleTaskGP` with `CategoricalKernel`
   - Acquisition optimization: Enumerates discrete combinations
   - **This is more sophisticated than one-hot encoding**

2. **Fallback (Many Categorical Parameters):** Uses `BOTORCH_MODULAR` with **one-hot encoding**
   - Transform: `OneHot` (expands to continuous parameters)
   - Model: `SingleTaskGP` (treats as continuous)
   - Used when categorical kernel approach would be too expensive

3. **Last Resort (Too Many Combinations):** Falls back to Sobol sampling

The key insight: **One-hot encoding is NOT the default**. Ax preferentially uses categorical kernels when practical, which provides better modeling of categorical structure than continuous relaxation.

### Decision Boundaries

- BO_MIXED (categorical kernels): ≤65 discrete combinations OR ≤33 one-hot parameters with ordered parameters
- BOTORCH_MODULAR (one-hot): More categorical parameters than these thresholds
- Sobol: Exceeds both thresholds
