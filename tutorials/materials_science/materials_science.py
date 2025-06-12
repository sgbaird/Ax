#!/usr/bin/env python
# coding: utf-8

# # Ax for Materials Science
# 
# Some optimization experiments, like the one described in this [tutorial](../getting_started), can be conducted in a completely automated manner.
# Other experiments may require a human in the loop, for instance a scientist manually conducting and evaluating each trial in a lab.
# In this tutorial we demonstrate this ask-tell optimization in a human-in-the-loop setting by imagining the task of maximizing the strength of a 3D printed part using compression testing (i.e., crushing the part) where different print settings will have to be manually tried and evaluated.
# 
# 
# ### Background
# 
# In 3D printing, several parameters can significantly affect the mechanical properties of the printed object:
# 
# - **Infill Density**: The percentage of material used inside the object. Higher infill density generally increases strength but also weight and material usage.
# - **Layer Height**: The thickness of each layer of material. Smaller layer heights can improve surface finish and detail but increase print time.
# - **Infill Type**: The pattern used to fill the interior of the object. Different patterns (e.g., honeycomb, gyroid, lines, rectilinear) offer various balances of strength, speed, and material efficiency.
# 
# - **Strength Measurement**: In this tutorial, we assume the strength of the 3D printed part is measured using compression testing, which evaluates how the object performs under compressive stress.
# 
# ### Learning Objectives
# - Understand black box optimization concepts
# - Define an optimization problem using Ax
# - Configure and run an experiment using Ax's `Client`
# - Analyze the results of the optimization
# 
# ### Prerequisites
# - Familiarity with Python and basic programming concepts
# - Understanding of [adaptive experimentation](../../intro-to-ae.mdx) and [Bayesian optimization](../../intro-to-bo.mdx)

# ## Step 1: Import Necessary Modules

# In[ ]:


from ax.api.client import Client
from ax.api.configs import  RangeParameterConfig, ChoiceParameterConfig

# ## Step 2: Initialize Client
# 
# Create an instance of the `Client` to manage the state of your experiment.

# In[ ]:


client = Client()

# ## Step 3: Configure Experiment
# 
# Define the parameters for the 3D printing optimization problem.
# The infill density and layer height can take on any value within their respective bounds so we will configure both using `RangeParameterConfig`s.
# On the other hand, infill type be either have one of four distinct values: "honeycomb", "gyroid", "lines", or "rectilinear".
# We will use a `ChoiceParameterConfig` to represent it in the optimization.

# In[ ]:


infill_density = RangeParameterConfig(name="infill_density", parameter_type="float", bounds=(0, 100))
layer_height = RangeParameterConfig(name="layer_height", parameter_type="float", bounds=(0.1, 0.4))
infill_type = ChoiceParameterConfig(name="infill_type", parameter_type="str", values=["honeycomb", "gyroid", "lines", "rectilinear"])

client.configure_experiment(
    parameters=[infill_density, layer_height, infill_type],
    # The following arguments are only necessary when saving to the DB
    name="3d_print_strength_experiment",
    description="Maximize strength of 3D printed parts",
    owner="developer",
)

# ## Step 4: Configure Optimization
# We want to maximize the compressive strength of our part, so we will set the objective to `compressive_strength`.
# However, we know that modifying the infill density, layer height, and infill type will affect the weight of the part as well.
# We'll include a requirement that the part must not weigh more than 10 grams by setting an outcome constraint when we call `configure_experiment`.
# 
# The following code will tell the `Client` that we intend to maximize compressive strength while keeping the weight less than 10 grams.

# In[ ]:


client.configure_optimization(objective="compressive_strength", outcome_constraints=["weight <= 10"])

# ## Step 4.1: Configure Generation Strategy (Important for Human-in-the-Loop)
# 
# When working with existing data in human-in-the-loop experiments, it's important to configure the generation strategy properly to ensure smooth transitions between different optimization phases.
# 
# **The Problem**: If you attach existing data and restart your script, the client doesn't know that the existing data came from a particular sampling strategy (like Sobol). This can cause the client to stay in the initial sampling phase indefinitely instead of transitioning to Bayesian optimization.
# 
# **The Solution**: Configure the generation strategy to properly account for existing trials and set appropriate transition criteria.

# In[ ]:


# Configure the generation strategy to handle existing data properly
client.configure_generation_strategy(
    method="fast",  # Use the default fast method (Sobol + BO)
    initialization_budget=6,  # We expect ~6 initialization trials (including existing data)
    use_existing_trials_for_initialization=True,  # Count attached trials toward initialization
    min_observed_initialization_trials=3,  # Minimum completed trials before BO
    allow_exceeding_initialization_budget=False,  # Enforce transition to BO after budget
)

# ## Step 5: Run Trials
# 
# Now the `Client` has been configured we can begin conducting the experiment.
# Use `attach_trial` to attach any existing data, use `get_next_trials` to generate parameter suggestions, and use `complete_trial` to report manually observed results.

# ## Step 5.1: Attach Preexisting Trials
# 
# Sometimes in our optimization experiments we may already have some previously collected data from manual "trials" conducted before the Ax experiment began.
# This can be incredibly useful!
# If we attach this data as custom trials, Ax will be able to use the data points in its optimization algorithm and improve performance.
# 
# **Note**: The generation strategy we configured in Step 4.1 will count these attached trials toward the initialization budget, ensuring proper transition to Bayesian optimization.

# In[ ]:


# Pairs of previously evaluated parameterizations and associated metric readings
preexisting_trials = [
    (
        {"infill_density": 10.43, "layer_height": 0.3, "infill_type": "gyroid"},
        {"compressive_strength": 1.74, "weight": 0.52},
    ),
    (
        {"infill_density": 55.54, "layer_height": 0.12, "infill_type": "lines"},
        {"compressive_strength": 4.63, "weight": 2.31},
    ),
    (
        {"infill_density": 99.43, "layer_height": 0.35, "infill_type": "rectilinear"},
        {"compressive_strength": 5.68, "weight": 2.84},
    ),
    (
        {"infill_density": 41.44, "layer_height": 0.21, "infill_type": "rectilinear"},
        {"compressive_strength": 3.95, "weight": 1.97},
    ),
    (
        {"infill_density": 27.23, "layer_height": 0.37, "infill_type": "honeycomb"},
        {"compressive_strength": 7.36, "weight": 3.31},
    ),
    (
        {"infill_density": 33.57, "layer_height": 0.24, "infill_type": "honeycomb"},
        {"compressive_strength": 13.99, "weight": 6.29},
    ),
]

for parameters, data in preexisting_trials:
    # Attach the parameterization to the Client as a trial and immediately complete it with the preexisting data
    trial_index = client.attach_trial(parameters=parameters)
    client.complete_trial(trial_index=trial_index, raw_data=data)

# ### Ask for trials
# 
# Now, let's have Ax suggest which trials to evaluate so that we can find the optimal configuration more efficiently.
# We'll do this by calling `get_next_trials`.
# We'll make use of Ax's support for parallelism, i.e. suggesting more than one trial at a time -- this can allow us to conduct our experiment much faster!
# If our lab had three identical 3D printers, we could ask Ax for a batch of three trials and evaluate three different infill density, layer height, and infill types at once.
# 
# Note that there will always be a tradeoff between "parallelism" and optimization performance since the quality of a suggested trial is often proportional to the amount of data Ax has access to.

# In[ ]:


trials = client.get_next_trials(max_trials=3)
trials

# ### Tell Ax the results
# 
# In a real-world scenerio we would print parts using the three suggested parameterizations and measure the compressive strength and weight manually, though in this tutorial we will simulate by calling a function.
# Once the data is collected we will tell Ax the result by calling `complete_trial`.

# In[ ]:


def evaluate(
    infill_density: float, layer_height: float, infill_type: str
) -> dict[str, float]:
    strength_map = {"lines": 1, "rectilinear": 2, "gyroid": 5, "honeycomb": 10}
    weight_map = {"lines": 1, "rectilinear": 2, "gyroid": 3, "honeycomb": 9}

    return {
        "compressive_strength": (
            infill_density / layer_height * strength_map[infill_type]
        )
        / 100,
        "weight": (infill_density / layer_height * weight_map[infill_type]) / 200,
    }


for trial_index, parameters in trials.items():
    client.complete_trial(trial_index=trial_index, raw_data=evaluate(**parameters))

# We'll repeat this process a number of times.
# Typically experimentation will continue until a satisfactory combination has been found, experimentation resources (in this example our 3D printing filliment) have been exhausted, or we feel we have spent enough time on optimization.

# In[ ]:


# Ask Ax for the next trials
trials = client.get_next_trials(max_trials=3)
trials

# In[ ]:


# Tell Ax the result of those trials
for trial_index, parameters in trials.items():
    client.complete_trial(trial_index=trial_index, raw_data=evaluate(**parameters))

# In[ ]:


# Ask Ax for the next trials
trials = client.get_next_trials(max_trials=3)
trials

# In[ ]:


# Tell Ax the result of those trials
for trial_index, parameters in trials.items():
    client.complete_trial(trial_index=trial_index, raw_data=evaluate(**parameters))

# ## Step 6: Analyze Results
# 
# At any time during the experiment you may analyze the results of the experiment.
# Most commonly this means extracting the parameterization from the best performing trial you conducted.
# The best trial will have the optimal objective value **without violating any outcome constraints**.

# In[ ]:


best_parameters, prediction, index, name = client.get_best_parameterization()
print("Best Parameters:", best_parameters)
print("Prediction (mean, variance):", prediction)

# ## Step 7: Compute Analyses
# 
# Ax can also produce a number of analyses to help interpret the results of the experiment via `client.compute_analyses`.
# Users can manually select which analyses to run, or can allow Ax to select which would be most relevant.
# In this case Ax selects the following:
# * **Parrellel Coordinates Plot** shows which parameterizations were evaluated and what metric values were observed -- this is useful for getting a high level overview of how thoroughly the search space was explored and which regions tend to produce which outcomes
# * **Scatter Plot** shows the effects of each trial on two metrics, and is useful for understanding the trade-off between the two outcomes
# * **Sensitivity Analysis Plot** shows which parameters have the largest affect on the objective using [Sobol Indicies](https://en.wikipedia.org/wiki/Variance-based_sensitivity_analysis)
# * **Slice Plot** shows how the model predicts a single parameter effects the objective along with a confidence interval
# * **Contour Plot** shows how the model predicts a pair of parameters effects the objective as a 2D surface
# * **Summary** lists all trials generated along with their parameterizations, observations, and miscellaneous metadata
# * **Cross Validation** helps to visualize how well the surrogate model is able to predict out of sample points 

# In[ ]:


# display=True instructs Ax to sort then render the resulting analyses
cards = client.compute_analyses(display=True)

# ## Alternative Approach: Save/Load for Persistent State
# 
# While the `attach_trial` approach shown above works well with proper generation strategy configuration, another robust approach for human-in-the-loop experiments is to use Ax's save/load functionality. This approach preserves the complete experiment state including the generation strategy.
# 
# ### Benefits of Save/Load Approach:
# - **Preserves Generation Strategy State**: The saved state includes information about which trials were generated by which strategy (Sobol, BO, etc.)
# - **Seamless Resumption**: When you load the experiment, it continues exactly where it left off
# - **No Configuration Needed**: You don't need to manually configure generation strategy parameters
# 
# ### Example Save/Load Pattern:
# ```python
# # After running some trials, save the experiment
# client.save_to_json_file(filepath="my_experiment.json")
# 
# # Later, in a new script or session, load the experiment
# client = Client.load_from_json_file(filepath="my_experiment.json")
# 
# # Continue with more trials - generation strategy state is preserved
# new_trials = client.get_next_trials(max_trials=3)
# ```
# 
# ### When to Use Which Approach:
# - **Use attach_trial + configure_generation_strategy**: When you have existing data from external sources or want fine control over generation strategy
# - **Use save/load**: When you want to pause and resume Ax experiments with full state preservation
# - **Hybrid approach**: Start with attach_trial for initial data, then use save/load for subsequent sessions
# 
# Both approaches are valid for human-in-the-loop optimization. Choose based on your workflow and data source requirements.

# ## Conclusion
# 
# This tutorial demonstrates how to use Ax's `Client` for optimizing the strength of 3D printed parts in a human-in-the-loop setting. Key takeaways:
# 
# 1. **Configure Generation Strategy**: When attaching existing data, properly configure the generation strategy to ensure smooth transitions between sampling and optimization phases.
# 
# 2. **Handle Restart Scenarios**: Use either proper generation strategy configuration with `attach_trial` or the save/load approach to handle script restarts gracefully.
# 
# 3. **Choose the Right Approach**: Select between attach_trial (for external data) and save/load (for state persistence) based on your workflow needs.
# 
# By iteratively collecting data and refining parameters with these best practices, you can effectively apply black box optimization to real-world experiments while avoiding common pitfalls in human-in-the-loop settings.
