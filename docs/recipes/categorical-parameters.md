# Categorical Parameters in Ax and BoTorch

This document explains how Ax and BoTorch implement and treat categorical parameters from a modeling perspective.

## Introduction

Categorical parameters represent discrete, unordered choices in an optimization problem. Unlike ordered choices (e.g., small/medium/large) or continuous parameters, categorical parameters don't have a natural ordering or numerical representation.

Examples of categorical parameters include:
- Algorithm choices (e.g., "adam", "sgd", "rmsprop")
- Model architectures (e.g., "resnet", "vgg", "inception")
- Activation functions (e.g., "relu", "tanh", "sigmoid")

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
- **Unordered/Categorical** (`is_ordered=False`): Values have no inherent ordering and require special encoding

Note: Choice parameters with exactly 2 values are automatically treated as ordered, regardless of the `is_ordered` setting.

## Transformation Pipeline

Before categorical parameters are passed to models, they go through a transformation pipeline. The key transformation is **one-hot encoding**.

### One-Hot Encoding

The [`OneHot`](https://github.com/sgbaird/Ax/blob/copilot/analyze-categorical-parameters/ax/adapter/transforms/one_hot.py#L58-L241) transform converts categorical parameters into continuous parameters suitable for Gaussian Process models:

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

### Inverse Transform: Rounding

When models generate continuous suggestions, the `OneHot` transform uses rounding to convert back to categorical values:

- **Strict rounding** (default): Choose the category with the maximum value, breaking ties randomly
- **Randomized rounding**: Sample from the distribution defined by the continuous values

See the implementation in [`ax/adapter/transforms/rounding.py`](https://github.com/sgbaird/Ax/blob/copilot/analyze-categorical-parameters/ax/adapter/transforms/rounding.py).

### Alternative Encodings

Besides one-hot encoding, Ax provides other transformation options:

1. **[`ChoiceToNumericChoice`](https://github.com/sgbaird/Ax/blob/copilot/analyze-categorical-parameters/ax/adapter/transforms/choice_encode.py#L25-L149)**: Maps non-numeric choices to integers (0, 1, ..., n-1)
2. **[`OrderedChoiceToIntegerRange`](https://github.com/sgbaird/Ax/blob/copilot/analyze-categorical-parameters/ax/adapter/transforms/choice_encode.py#L151-L242)**: Converts ordered choices to integer `RangeParameter`s

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

When a BoTorch model is constructed, the `categorical_features` indices from `SearchSpaceDigest` are passed as a model argument. See [`ax/generators/torch/botorch_modular/surrogate.py`](https://github.com/sgbaird/Ax/blob/copilot/analyze-categorical-parameters/ax/generators/torch/botorch_modular/surrogate.py#L141-L143):

```python
if len(search_space_digest.categorical_features) > 0:
    kwargs["categorical_features"] = search_space_digest.categorical_features
```

### Categorical Kernels in BoTorch

BoTorch uses specialized kernels for categorical parameters:

- Models can use **categorical kernels** (e.g., Hamming distance-based kernels) for better performance with categorical variables
- The generation strategy may automatically select appropriate kernels based on the presence of categorical features

See the dispatch logic in [`ax/generation_strategy/dispatch_utils.py`](https://github.com/sgbaird/Ax/blob/copilot/analyze-categorical-parameters/ax/generation_strategy/dispatch_utils.py) which mentions "Using Bayesian optimization with a categorical kernel for improved performance."

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

Ax treats categorical parameters through a well-defined pipeline:

1. **Define** categorical parameters using `ChoiceParameter` with `is_ordered=False`
2. **Transform** via one-hot encoding (default) or alternative numeric encodings
3. **Communicate** to models through `SearchSpaceDigest` with `categorical_features` indices
4. **Model** using BoTorch with specialized categorical kernels
5. **Inverse transform** continuous model outputs back to categorical values via rounding

This approach allows Gaussian Process-based Bayesian optimization to handle categorical parameters effectively while maintaining the mathematical rigor required for optimization.
