#!/usr/bin/env python
# coding: utf-8

# <div style="background-color: #c1f2a5">
# A model of time discounting in value-based choice.
# </div>

# In[38]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt


# 
# ## Q1. Time discount choice data. 
# 
# <div class="alert alert-success">
# In this experiment, participants chose between taking \$20 now vs. accepting an offer of value  \$V after a delay of D days.
#     
# - `offers` is an n × 2 array, where each row (e.g., `[30., 1.]`) shows an offer V (column 0: `30.`) and a delay D (column 1: `1.`).
# - `subjects` is a 1D array of length n, indicating the participant ID in a given row.
# - `choices` is an n × 10 array, which contains 10 distinct choices made by the same participant (same ID in `subjects`) in the same condition (same combination of offer V and delay D in `offers`). "1" indicates accepting the delayed offer and "0" indicates taking \$20 now.
# 
# A look into the data is provided below.
#     
# </div>
# 

# In[39]:


# Load experiment data
# Experiment loaded in by instructor.
data = np.load("TimeDiscountExperiment.npz")


# In[40]:


# Look into the data provided by instructor.
# Offers and delays
offers = data["offers"]
print(f"- first 20 offers: [offer   delay]:\n {offers[:42, :]} \n")

# Participant ID in each row
subjects = data["participants"]

# Choices made by participants
choices = data["choices"]
print(f"- first 0 conditions' choices:\n {choices[:10, :]} \n")

# Unique offer values
Vs = np.unique(offers[:, 0])
print(f"- list of unique offer values: {Vs} \n")

# Unique delay values
Ds = np.unique(offers[:, 1])
print(f"- list of unique delay values: {Ds} \n")

# Unique participants
nsubj = len(np.unique(subjects))
print(f"- number of unique subjects: {str(nsubj)}\n")

# Number of iterations per participant per condition + total number of conditions
trialnum = choices.shape
print(f"- each participant has: {str(trialnum[1])} iterations per condition \n")
print(f"- there are  {str(int(trialnum[0] / nsubj))} conditions")


# In[41]:


choices[30:36, :]


# ## Behavior
# <div class="alert alert-success">
# 1.In two subplots of the same figure, the overall proportion of delayed choices is presented as a function of a) offer value (on the left) and b) delay value (on the right).
#     
# 2.In a separate figure, a heatmap is displayed where the x-axis is the offer value V, the y-axis is the delay value D, and colors indicate the proportion of **delayed choices** (i.e., accepting the delayed offer).
# 
# We see that participants 1) are less likely to accept offers with longer delays and 2) more likely to accept delayed offers of higher values.
# </div>
# 

# In[42]:


# YOUR CODE HERE
figure, axis = plt.subplots(1, 2, figsize = [10, 5])

figure.suptitle("Proportion of Delayed Choices vs. Offer & Delay Value", y = 1.00, fontsize = 14)

axis_1 = axis[0]

offers_1 = [0,0,0,0,0,0]
offer_count = -1
for i in np.arange(0, len(choices)):
    if i%6 == 0:
        offer_count += 1
        if offer_count == 6:
            offer_count = 0
    offers_1[offer_count] += sum(choices[i])

offers_1 = np.array(offers_1) / 600
axis_1.plot(Vs, offers_1, linestyle = '', marker = 'o')
axis_1.set_xlabel("Offer Value")
axis_1.set_ylabel("Proportion of Delayed Choices")
axis_1.set_title("Proportion of Delayed Choices vs. Offer Value")

axis_2 = axis[1]
delays_1 = [0,0,0,0,0,0]
delay_count = 0
for i in np.arange(0, len(choices)):
    delays_1[delay_count] += sum(choices[i])
    delay_count += 1
    if delay_count == 6:
        delay_count = 0

delays_1 = np.array(delays_1) / 600
axis_2.plot(Ds, delays_1, linestyle = '', marker = 'o')
axis_2.set_xlabel("Delay Value")
axis_2.set_ylabel("Proportion of Delayed Choices")
axis_2.set_title("Proportion of Delayed Choices vs. Delay Value")


# In[43]:


# YOUR CODE HERE
figure, axis = plt.subplots(1, 1)
matrix = np.zeros((6,6))
offer_count = 0
delay_count = -1
for i in np.arange(0, len(choices)):
    delay_count += 1
    if delay_count == 6 and offer_count == 5:
        offer_count = 0
        delay_count = 0
    if delay_count == 6 and offer_count != 5:
        offer_count += 1
        delay_count = 0
    matrix[offer_count][delay_count] += sum(choices[i])

matrix = matrix / 100
im = plt.imshow(matrix.T[::-1])
plt.locator_params(nbins=6)
plt.xticks(np.arange(0, 6), Vs, rotation=45, fontsize = 10)
plt.yticks(np.arange(0, 6), Ds[::-1], rotation=45, fontsize = 10)
plt.xlabel("Offer Value")
plt.ylabel("Delay Value")
plt.title("Heatmap of Offer Value and Delay Value")
plt.colorbar(im)


# ## Individual differences
# 
# <div class="alert alert-success">
#     
# Different participants can behave very differently in this type of experiment, so we look at each person individually. 
#     
# In 10 subplots (one per participant) of the same figure, each participant's behavior is plotted as a heatmap where the x-axis is the delay value D, the y-axis is the offer value V, and the colors indicate the probability of accepting the delayed offer given V and D, $P(delay|V,D)$. 
#     
# We create an `individual_behavior` list for each of the 10 participants, where each element is a 6 × 6 array that contains each participant's $P(delay|V,D)$. We will need this list later when comparing people's actual behavior with model predictions. 
#             
# </div>

# In[44]:


figure, axis = plt.subplots(2, 5, figsize=[30, 10])
plt.setp(axis, xticks=np.arange(0, 6), xticklabels=Vs,yticks=np.arange(0, 6), yticklabels=Ds[::-1])
figure.suptitle("Heatmaps of Offer and Delay Value for each Participant", y = 1.00, fontsize = 20)

# YOUR CODE HERE
first_index = 0
second_index = 0
count = -1
individual_behavior = []
for i in np.arange(0, len(choices), 36):
    offer_count = 0
    delay_count = -1
    curr_matrix = np.zeros((6,6))
    count += 1
    for j in np.arange(i, i+36):
        delay_count += 1
        if delay_count == 6 and offer_count == 5:
            offer_count = 0
            delay_count = 0
        if delay_count == 6 and offer_count != 5:
            offer_count += 1
            delay_count = 0
        curr_matrix[offer_count][delay_count] += sum(choices[j])
    individual_behavior.append(curr_matrix)
    curr_matrix = curr_matrix / 10
    
    if second_index == 5:
        first_index = 1
        second_index = 0
    curr_axis = axis[first_index][second_index]
    curr_axis.imshow(curr_matrix.T[::-1], aspect = "auto")
    curr_axis.set_xlabel("Offer Value")
    curr_axis.set_ylabel("Delay Value")
    curr_axis.set_title(str(count))
    second_index += 1


# 
# ## Individual differences. 
# 
# <div class="alert alert-success">
# 
# <li>How are subject 4 and 6 different?</li>
#     Subject 4 chose all offers of value 100, regardless of the delay. They also chose most offers of value 50, except when the delay was 100, in which they chose it about half the time. The same goes for value 30, except when the delay was 50 days or greater. As the offer value decreases, Subject 4 chose the lower values less and less, only choosing them when the delay time was less than 5 days(they chose these around half the time, more 23 and 25 than 21). They did not choose lower values with high delays for the most part, very rarely choosing delay 30 with 23 and 25. Subject 6 was different in that they only chose all offer of value 100 when the delay value was 1 or 2 days. They chose some of value 50, with the lowest delay value. While subject 4 chose a lot of options, mostly in the higher values regardless of delay time, Subject 6 limited their options mainly to the 100 choices, with the lowest 2 delay times. Subject 6 also very, very rarely tried offers 30 or below, despite the delay value and if they did so, only in the lowest 2 delay values, which differs to Subject 4 who chose a good amount of lower values with low delays.
# <li>How are subject 1 and 8they different? </li>
#     Subject 1 and Subject 8 were both similar in that they both stuck to choosing offer values of 50 and 100 all the time, whose delays were 10 days or lower. Subject 8 differed, in that they chose offer value 30 all the time with a delay of 1 day, whereas subject 1 only chose this option about half the time. Subject 1 also chose some options in the lower offer values(21, 23, 25), doing so about 30 percent or more of the time with the 2 lowest delay values. Subject 8 did not do this for the second lowest delay value, only doing so for the lowest delay value. Overall, they picked very similarly, with Subject 1 choosing some of the time on the lower value side and mainly in the two highest values with delays lower than 10, while Subject 8 chose for the lower value side less than Subject 1.
# </div>
# 
# 

# 
# ## Modeling the data
# 
# We will use a hyperbolic discount model to model the data. As a reminder, the model assumes that the non-delayed offer of $V_0 =20$ is compared to a subjective value offer of $V_1 = \frac{V}{1+k \dot D}$.
# 
# Then, we assume that participants choose between delay (1) or no delay (0) according to a softmax choice rule: $P(delay) = \frac{1}{1+exp(-\beta \cdot (V_1 - V_0))}$. This formula gives you the likelihood of an observed choice given A) the chosen model, B) the offer value V and the delay D, as well as C) parameters $k$ (discount factor) and $\beta$ (noise parameter). To avoid numerical issues where the probability is $0$, **we will use a slightly modified version of the original equation**: 
# 
# $$P(delay) = \frac{\epsilon}{2} + (1-\epsilon)\frac{1}{1+exp(-\beta \cdot(V_1 - V_0))},$$ 
# 
# where $\epsilon = .0001$ is a very small value preventing 0 or 1 probabilities.
# 
# 
# ### Hyperbolic time discount likelihood
# <div class="alert alert-success">
# 
# We wrote a function called `likelihood_one_choice` that takes an observed choice C (0 or 1), an 1 × 2 offer vector ([V, D]), and a 1 × 2 parameter vector ([k, beta]), and returns the likelihood of choice C. This function first computes the subjective value of the new offer and then uses it to compute the probability of the choice. 
# 
# </div>

# In[45]:


def likelihood_one_choice(C, offer, parameters):
    """
    Returns the likelihood of choice C give C, the offer ([V, D]), 
    and parameters ([k, beta]) in the softmax choice model.
    
    Parameters
    ----------
    
    C: integer
        Observed choice (either 0 or 1)
    offer : NumPy array of shape (2,)
        Combination of offer value and delay 
    parameters: NumPy array of shape (2,)
        Parameter values in the softmax function 
 
    Returns
    -------
    a float corresponding to the probability of C 

    """

    V = offer[0]
    D = offer[1]
    k = parameters[0]
    beta = parameters[1]
    eps = 0.0001

    # YOUR CODE HERE
    V_1 = V/(1 + k*D)
    V_0 = 20
    p_c_1 = eps/2 + (1-eps)*(1/(1 + np.exp(-beta * (V_1 - V_0))))
    if C == 0:
        return 1 - p_c_1
    else:
        return p_c_1


# In[46]:


# Add your own test cases


# In[47]:


"""Check if likelihood_one_choice computes the correct values"""
#Test cases provided by instructor.
from numpy.testing import assert_allclose

assert_allclose(likelihood_one_choice(1, np.array([100, 1]), np.array([1, 0])), 0.5)
assert_allclose(
    likelihood_one_choice(1, np.array([100, 1]), np.array([1, 1])), 0.9999499999999065
)
assert_allclose(
    likelihood_one_choice(0, np.array([100, 1]), np.array([1, 1])),
    5.000000009347527e-05,
)
assert_allclose(
    likelihood_one_choice(1, np.array([60, 30]), np.array([0.1, 0.5])),
    0.07590059420324143,
)

print("Success!")


# In[48]:


# Plot for Q3.1
# Plotting Code Provided by instructor
offer = np.array([100, 20])
beta = 0.5

ks = np.arange(0, 1, 0.01)
ls = np.empty(len(ks))

for i, k in enumerate(ks):
    ls[i] = likelihood_one_choice(1, offer, np.array([k, beta]))


figure, axis = plt.subplots()
axis.plot(ks, ls)
axis.set_xlabel("k")
axis.set_ylabel("P(delay)")
axis.set_title("likelihood for V=100, D=50, beta=0.5")


# 
# ### Hyperbolic time discount likelihood - multiple trials 
# <div class="alert alert-success">
# 
# We wrote a function called `log_likelihood_choice` that takes an n × 10 array of observed choices, an n × 2 offer array, and a 1 × 2 parameter vector ([k, beta]), and returns the log-likelihood of the given sequence of n × 10 choices. We assume that all choices are independent from each other such that the log-likelihood of the sequence is equal to the sum of the log-likelihood of each choice independently.
# 
# </div>

# In[49]:


def log_likelihood_choices(choices, offers, parameters):   
    """
    Returns the log-likelihood of the sequence of choices, given an array of choices, 
    an array of corresponding offers, and parameters ([k, beta]) in the softmax choice model.
    
    Parameters
    ----------
    
    choices: NumPy array of shape (n, 10)
        A sequence of choices made by participants
    offer: NumPy array of shape (n, 2)
        A sequence of offers upon which choices were based
    parameters: NumPy array of shape (2,)
        Parameter values in the softmax function 
 
    Returns
    -------
    a float corresponding to the log-likelihood of the given sequence of choices 

    """

    #YOUR CODE HERE
    log_sum = 0
    for i in np.arange(0, len(choices)):
        for j in np.arange(0, 10):
            log_sum += np.log(likelihood_one_choice(choices[i][j], offers[i], parameters))
    return log_sum


# In[50]:


# Add your own test cases here


# In[51]:


from numpy.testing import assert_allclose

data = np.load("TimeDiscountExperiment.npz")
offers = data["offers"]
choices = data["choices"]


assert_allclose(
    log_likelihood_choices(choices, offers, np.array([0, 0.5])), -8217.927553085532
)
assert_allclose(
    log_likelihood_choices(choices, offers, np.array([0, 0])), -2495.329850015794
)
assert_allclose(
    log_likelihood_choices(choices, offers, np.array([0.2, 0.1])), -1663.772256824346
)


print("Success!")


# In[52]:


# Plotting code provided by instructor.
k = 0.1

bs = np.arange(0, 1, 0.01)
ls = np.empty(len(bs))

for i, b in enumerate(bs):
    ls[i] = log_likelihood_choices(choices, offers, np.array([k, b]))


figure, axis = plt.subplots()
axis.plot(bs, ls)
axis.set_xlabel("beta")
axis.set_ylabel("P(delay)")
axis.set_title("log-likelihood for V=100, D=50, k=.1")


# ## Fitting the parameters
# 
# Now, we are going to use this likelihood function to find the parameters that best fit each participant's choice data.
# 
# ### Maximum likelihood 
# <div class="alert alert-success">
# 
# We are going to use [grid search](https://en.wikipedia.org/wiki/Hyperparameter_optimization#Grid_search_2) to find the $k$ and the $\beta$ values that maximize each participant's log-likelihood of choices. The "grid" consists of exhaustive combinations of possible $k$ and $\beta$ values, which you can use the provided incements `ks` and `bs` to generate.
#     
# 1. In 10 subplots (one per participant), we plotted the log-likelihood as a heatmap where the x-axis is $\beta$ and the y-axis is $k$. 
#     
# 2. Apart from generating the heatmaps, in each subplot, we also found the $k$ and the $\beta$ values that provide the maximium log-likelihood and then use a red dot to mark this position on that heatmap.
#     
# 3. We stored the maximum positions of $\beta$ as a 1D array `map_b` and the maximum positions of $k$ as another 1D array `map_k`.
# </div>

# In[53]:


ks = np.arange(0.01, 1, 0.01)
bs = np.arange(0.1, 1, 0.1)

figure, axis = plt.subplots(2, 5, figsize=[30, 10])
figure.suptitle("Log Likelihood Heatmaps of Beta and k Value for each Participant", y = 1.00, fontsize = 20)
# YOUR CODE HERE
log_likelihoods = []
map_b = np.array([])
map_k = np.array([])
first_index = 0
second_index = 0
count = -1
for i in np.arange(0, len(choices), 36):
    curr_choices = choices[i:i+36]
    curr_likelihoods = []
    curr_k_b = []
    curr_matrix = np.zeros((len(bs),len(ks)))
    count += 1
    for k in np.arange(0, len(bs)):
        for j in np.arange(0, len(ks)):
            parameters = np.array([ks[j],bs[k]])
            curr_log_likelihood = log_likelihood_choices(curr_choices, offers[0:36], parameters)
            curr_likelihoods.append(curr_log_likelihood)
            curr_matrix[k][j] = curr_log_likelihood
            curr_k_b.append(parameters)
            
    curr_max_index = np.argmax(curr_likelihoods)
    curr_max = curr_likelihoods[curr_max_index]
    map_b = np.append(map_b, curr_k_b[curr_max_index][1])
    map_k = np.append(map_k, curr_k_b[curr_max_index][0])
    if second_index == 5:
        first_index = 1
        second_index = 0
    curr_axis = axis[first_index][second_index]
    curr_axis.imshow(curr_matrix.T[::-1], aspect = "auto", extent=[0.1,0.9,0.01,0.99])
    curr_axis.plot(curr_k_b[curr_max_index][1], curr_k_b[curr_max_index][0], color="r", marker = 'o')
    curr_axis.set_xlabel("Beta Value")
    curr_axis.set_ylabel("k Value")
    curr_axis.set_title(str(count))
    second_index += 1


# ### Maximum likelihood 
# <div class="alert alert-success">
# 
# The data provided to us was not real participant data but simulated data. The true $k$ and $\beta$ values for each participant are provided below. In two subplots (one per parameter), we plotted the maximum likelihood estimates (MLEs) of each parameter (x-axis) against their true values (y-axis). One subplot compared MLEs of $\beta$ against actual $\beta$ values and the other comparing MLEs of $k$ against actual $k$ values.
# <ol>
# <li>How well did you recover the true parameters?</li>
#     We recovered the true parameters with relatively high accuracy. The points of k were all clustered around the the unity line, which means all the MLES of k were almost equal to the true values of k. Of the beta points, 6 were clustered around the unity line, which means that these MLEs of beta were almost equal to the true values of k and 3 were not near the unity line, which means that these MLEs were not all equal to the true values of beta.
# </ol>
# </div>
# 
# 

# In[54]:


true_k = data["true_k"]
true_beta = data["true_beta"]

# YOUR CODE HERE
figure, axis = plt.subplots(1, 2, figsize = [10, 5])
figure.suptitle("True Values vs. MLEs", y = 1.00, fontsize = 14)
axis_1 = axis[0]
axis_1.plot(map_b, true_beta, linestyle = '', marker = 'o')
axis_1.set_xlabel("MLEs of beta")
axis_1.set_ylabel("True beta values")
axis_1.plot(np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1))
axis_1.set_title("True beta values vs. MLEs of beta")

axis_2 = axis[1]
axis_2.plot(map_k, true_k, linestyle = '', marker = 'o')
axis_2.plot(np.arange(0, 1, 0.01), np.arange(0, 1, 0.01))
axis_2.set_xlabel("MLEs of k")
axis_2.set_ylabel("True k values")
axis_2.set_title("True k values vs. MLEs of k")


# # Simulating the model

# 
# ### Simulating the model.
# 
# Now, we want to simulate people's choices using the model with fitted parameters to 1) see how well the simulation matches the actual data and 2) make predictions for different offer or delay values. You will reuse your `likelihood_one_choice` function in the simulation.
# 
# <div class="alert alert-success">
#     
# Earlier, we saved each participant's aggregate data in a list called <code>individual_behavior</code>. Each element in this list is a 6 × 6 array storing each participant's $P(delay|V,D)$ given each combination of V and D. We also saved each participant's estimated $k$ value in `map_k` and estimated $\beta$ value in `map_b`. Using these data, we generate a figure with 10 subplots (one per participant).
#     
# 1. In each subplot, for each offer value V , we plotted the model's predicted likelihood of accepting the delayed offer as a function of delay. 
#     
# 2. On top of these lines, for each offer , we plotted each participant's $P(delay|V,D)$ as a function of delay (using the actual delay values, which takes one of 6 discrete values in `Ds`). 
#     
# Since the lines in #1 are very close to the corresponding data points in #2, we know that this is done correctly.   
# </div>

# In[55]:


individual_behvaior = np.array(individual_behavior) / 10


# In[56]:


Vs = np.unique(offers[:, 0])
Dline = np.arange(min(Ds), max(Ds) + 1)
Nd = len(Dline)

# YOUR CODE HERE
figure, axis = plt.subplots(2, 5, figsize=[30, 10])
figure.suptitle("predicted likelihood of accepting the delayed offer as a function of delay for each Participant", y = 1.00, fontsize = 20)
first_index = 0
second_index = 0
individual_behavior = np.array(individual_behavior) / 10
colors = ["red", "orange", "yellow", "green", "blue", "indigo"]
for i in np.arange(0, 10):
    curr_axis = axis[first_index][second_index]
    for j in np.arange(0, len(Vs)):
        curr_likelihoods = []
        curr_delay = []
        for k in np.arange(0, len(Dline)):
            curr_likelihoods.append(likelihood_one_choice(1, np.array([Vs[j], Dline[k]]), np.array([map_k[i], map_b[i]])))
        for h in np.arange(0, len(Ds)):
            curr_delay.append(individual_behavior[i][j][h])
        curr_axis.plot(Dline, curr_likelihoods, color = colors[j], label = Vs[j])
        curr_axis.plot(Ds, curr_delay, color = colors[j], marker = 'o')
        curr_axis.set_xlabel("Delay Value")
        curr_axis.set_ylabel("Predicted Likelihood of Accepting Delayed Offer")
        curr_axis.set_title(i)
    curr_axis.legend()
        
    second_index += 1
    if second_index == 5:
        first_index = 1
        second_index = 0


# 
# ### Interpretation. 
# 
# <div class="alert alert-success">
# <ul>
# <li>How well does the model capture the experiment data? Explain where any observed differences might have come from.</li>
#     The model captures the experiment data relatively well as we see the dots roughly follow the shape of the curve. Any observed differences might have come from the person having a hunch that a certain delay value would work or they may come from the times when the participant chose 0 multiple times and this may have have shifted the probability value. 
#     </ul>
# </div>
# 
# ### Prediction. 
# 
# <div class="alert alert-success">
# The following questions are answered based on the above graph.
# <ol>
# <li>For participant 3 (index starts from 0), if I make a delayed offer of \$50, how long should the delay be so that they will be perfectly ambivalent between taking this delayed offer vs. \$20 now?</li>
# 30
# <li>What about participant 4?</li>
# 75
# <li>What about participant 5?</li>
# 3
# </ol>
# </div>
# 

# 
