# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3.10 (XPython)
#     language: python
#     name: xpython
# ---

# %% deletable=false editable=false tags=[]
# Initialization cell
try:  # for CS1302 JupyterLite pyodide kernel
    import piplite

    with open("requirements.txt") as f:
        for package in f:
            package = package.strip()
            print("Installing", package)
            await piplite.install(package)
except ModuleNotFoundError:
    pass

import random
import jupytext
import otter
from ipywidgets import interact

grader = otter.Notebook("main.ipynb")
# %reload_ext divewidgets

# %% [markdown] slideshow={"slide_type": "slide"}
# # Information Theory

# %% [markdown] slideshow={"slide_type": "-"} tags=["remove-cell"]
# **CS1302 Introduction to Computer Programming**
# ___

# %% [markdown] slideshow={"slide_type": "subslide"}
# As mentioned in previous lectures, the following two lists `coin_flips` and `dice_rolls` simulate the random coin flips and rollings of a dice:

# %% nbgrader={"grade": false, "grade_id": "random", "locked": true, "schema_version": 3, "solution": false, "task": false} slideshow={"slide_type": "-"}
# Do NOT modify any variables defined here because some tests rely on them
import random
import math

random.seed(0)  # for reproducible results.
num_trials = 200
coin_flips = ["H" if random.random() <= 1 / 2 else "T" for i in range(num_trials)]
dice_rolls = [random.randint(1, 6) for i in range(num_trials)]
print("coin flips: ", *coin_flips)
print("dice rolls: ", *dice_rolls)


# %% [markdown] slideshow={"slide_type": "subslide"}
# **What is the difference of the two random processes?  
# Can we say one process has more information content than the other?**

# %% [markdown] slideshow={"slide_type": "-"}
# In this Lab, we will use dictionaries to store their distributions and then compute their information content using information theory, which was introduced by *Claude E. Shannon*. It has [numerous applications](https://www.khanacademy.org/computing/computer-science/informationtheory): 
# - *compression* (to keep files small)
# - *communications* (to send data to mobile phones), and 
# - *machine learning* (to identify relevant features to learn from).

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Entropy

# %% [markdown] slideshow={"slide_type": "fragment"}
# Mathematically, we denote a distribution as $\mathbf{p}=[p_i]_{i\in \mathcal{S}}$, where 
# - $\mathcal{S}$ is the set of distinct outcomes, and
# - $p_i$ denotes the probability (chance) of seeing outcome $i$.

# %%

# %% [markdown] slideshow={"slide_type": "fragment"}
# The following code shown in the lecture uses a dictionary to store the distribution for a sequence efficiently without storing outcomes with zero counts:

# %% nbgrader={"grade": false, "grade_id": "dist", "locked": true, "schema_version": 3, "solution": false, "task": false} slideshow={"slide_type": "-"}
# Do NOT modify any variables defined here because some tests rely on them
def distribute(seq):
    """Returns a dictionary where each value in a key-value pair is
    the probability of the associated key occurring in the sequence.
    """
    p = {}
    for i in seq:
        p[i] = p.get(i, 0) + 1 / len(seq)
    return p


# tests
coin_flips_dist = distribute(coin_flips)
dice_rolls_dist = distribute(dice_rolls)
print("Distribution of coin flips:", coin_flips_dist)
print("Distribution of dice rolls:", dice_rolls_dist)

# %% [markdown] slideshow={"slide_type": "fragment"}
# For $\mathbf{p}$ to be a valid distribution, the probabilities $p_i$'s have to sum to $1$, i.e.,
#
# $$\sum_{i\in \mathcal{S}} p_i = 1, $$
# which can be verified as follows:

# %% slideshow={"slide_type": "-"}
assert math.isclose(sum(coin_flips_dist.values()), 1) and math.isclose(
    sum(dice_rolls_dist.values()), 1
)


# %% [markdown] slideshow={"slide_type": "subslide"}
# **How to measure the information content?**

# %% slideshow={"slide_type": "-"} tags=["hide-input"] language="html"
# <iframe width="800" height="450" src="https://www.youtube.com/embed/2s3aJfRr9gE" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

# %% [markdown] slideshow={"slide_type": "subslide"}
# ---
#
# **Definition** (Entropy)
#
# In information theory, the information content of a distribution is measured by its [*entropy*](https://en.wikipedia.org/wiki/Entropy_(information_theory)) defined as:
#
# $$ 
# \begin{aligned} 
# H(\mathbf{p}) &:= \sum_{i\in \mathcal{S}} p_i \overbrace{\log_2 \tfrac{1}{p_i}}^{\text{called surprise} } \\ 
# &= - \sum_{i\in \mathcal{S}} p_i \log_2 p_i \kern1em \text{(bits)} \end{aligned}  
# $$
#
# with $p_i \log_2 \frac{1}{p_i} = 0$ if $p_i = 0$ because $\lim_{x\downarrow 0} x \log_2 \frac1x = 0$.
#
# ---

# %% [markdown] slideshow={"slide_type": "fragment"}
# For instance, if $\mathbf{p}=(p_{H},p_{T})=(0.5,0.5)$, then
#
# $$
# \begin{aligned} H(\mathbf{p}) &= 0.5 \log_2 \frac{1}{0.5} + 0.5 \log_2 \frac{1}{0.5} \\ &= 0.5 + 0.5 = 1  \text{ bit,}\end{aligned} 
# $$
#
# i.e., an outcome of a fair coin flip has one bit of information content, as expected.

# %% [markdown] slideshow={"slide_type": "fragment"}
# On the other hand, if $\mathbf{p}=(p_{H},p_{T})=(1,0)$, then
#
# $$
# \begin{aligned} H(\mathbf{p}) &= 1 \log_2 \frac{1}{1} + 0 \log_2 \frac{1}{0} \\ &= 0 + 0 = 0  \text{ bits,}\end{aligned} 
# $$
#
# i.e., an outcome of a biased coin flip that always comes up head has no information content, again as expected.

# %% [markdown] slideshow={"slide_type": "subslide"}
# **Exercise** (entropy)
#
# Define a function `entropy` that
# - takes a distribution $\mathbf{p}$ as its argument, and
# - returns the entropy $H(\mathbf{p})$.
#
# Handle the case when $p_i=0$, e.g., using the short-circuit evaluation of logical `and`.

# %% nbgrader={"grade": false, "grade_id": "entropy", "locked": false, "schema_version": 3, "solution": true, "task": false} slideshow={"slide_type": "-"} tags=["remove-output"]
def entropy(dist):
    ans = 0
    for a in dist:
        ans += dist[a] and dist[a]*math.log2(1/dist[a])
    return ans


# %% deletable=false editable=false
grader.check("entropy")

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Uniform distribution maximizes entropy

# %% [markdown] slideshow={"slide_type": "fragment"}
# Intuitively,
# - for large enough numbers of fair coin flips, we should have $\mathcal{S}=\{H,T\}$ and $p_H=p_T=0.5$, i.e., equal chance for head and tail.
# - for large enough numbers of fair dice rolls, we should have $p_i=\frac16$ for all $i\in \mathcal{S}=\{1,2,3,4,5,6\}$.

# %% slideshow={"slide_type": "-"}
import matplotlib.pyplot as plt


def plot_distribution(seq):
    dist = distribute(seq)
    plt.stem(
        dist.keys(),  # set-like view of the keys
        dist.values(),  # view of the values
    )
    plt.xlabel("Outcomes")
    plt.title("Distribution")
    plt.ylim(0, 1)


import ipywidgets as widgets

n_widget = widgets.IntSlider(
    value=1,
    min=1,
    max=num_trials,
    step=1,
    description="n:",
    continuous_update=False,
)

widgets.interactive(lambda n: plot_distribution(coin_flips[:n]), n=n_widget)

# %% slideshow={"slide_type": "-"}
widgets.interactive(lambda n: plot_distribution(dice_rolls[:n]), n=n_widget)


# %% [markdown] slideshow={"slide_type": "fragment"}
# ---
#
# **Definition** (Uniform)
#
# A distribution is called a *uniform distribution* if all its distinct outcomes have the same probability of occurring, i.e.,
#
# $$ p_i = \frac{1}{|\mathcal{S}|}\kern1em  \text{for all }i\in \mathcal{S},  $$
#
# where $|\mathcal{S}|$ is the mathematical notation to denote the size of the set $\mathcal{S}$.
#
# ---

# %% [markdown] slideshow={"slide_type": "subslide"}
# **Exercise** (uniform)
#
# Define a function `uniform` that
# - takes a sequence of possibly duplicate outcomes, and
# - returns a uniform distribution of the distinct outcomes.

# %% nbgrader={"grade": false, "grade_id": "uniform", "locked": false, "schema_version": 3, "solution": true, "task": false} slideshow={"slide_type": "-"} tags=["remove-output"]
def uniform(outcomes):
    """Returns the uniform distribution (dict) over distinct items in outcomes."""
    dic1 = {i:1 for i in outcomes}
    return {i : 1/len(dic1) for i in outcomes}


# %% deletable=false editable=false
grader.check("uniform")

# %% [markdown] slideshow={"slide_type": "subslide"}
# **What is the entropy for uniform distributions?**

# %% [markdown] slideshow={"slide_type": "fragment"}
# By definition,
#
# $$ \begin{aligned} H(\mathbf{p}) &:= \sum_{i\in \mathcal{S}} p_i \log_2 \tfrac{1}{p_i} \\ &= \sum_{i\in \mathcal{S}} \frac{1}{|\mathcal{S}|} \log_2 |\mathcal{S}| = \log_2 |\mathcal{S}|  \kern1em \text{(bits)} \end{aligned}  $$
#
# Entropy reduces to the formula in Lecture 1 and Lab 1 regarding the number of bits required to encode a set of integers or characters. It is the maximum possible entropy for a given finite set of possible outcomes.

# %% [markdown] slideshow={"slide_type": "fragment"}
# Use this result to test whether you have implemented both `entropy` and `uniform` correctly:

# %% slideshow={"slide_type": "-"} tags=["remove-output"]
assert all(
    math.isclose(entropy(uniform(range(n))), math.log2(n)) for n in range(1, 100)
)


# %% [markdown] slideshow={"slide_type": "slide"} tags=[]
# ## Joint distribution and its entropy

# %% [markdown] slideshow={"slide_type": "fragment"}
# If we duplicate a sequence of outcomes, the total information content should remain unchanged, NOT doubled, because the duplicate contains the same information as the original. We will verify this fact by creating a [joint distribution](https://en.wikipedia.org/wiki/Joint_probability_distribution) 
#
# $$\mathbf{p}=[p_{ij}]_{i\in \mathcal{S},j\in \mathcal{T}}$$ 
# - where $\mathcal{S}$ and $\mathcal{T}$ are sets of outcomes; and
# - $p_{ij}$ is the chance we see outcomes $i$ and $j$ simultaneously. 
#
# The subscript $ij$ in $p_{ij}$ denotes a tuple $(i,j)$, NOT the multiplication $i\times j$. We also have
#
# $$\sum_{i\in \mathcal{S}} \sum_{j\in \mathcal{T}} p_{ij} = 1.$$

# %% [markdown] slideshow={"slide_type": "subslide"}
# **Exercise** (jointly-distribute)
#
# Define a function `jointly_distribute` that 
# - takes two sequences, `seq1` and `seq2`, of outcomes with the same length, and
# - returns the joint distribution represented as a dictionary where each key-value pair has the key being a tuple `(i,j)` associated with the probability $p_{ij}$ of seeing `(i,j)` in `zip(seq1,seq2)`.

# %% nbgrader={"grade": false, "grade_id": "jointly_distribute", "locked": false, "schema_version": 3, "solution": true, "task": false} slideshow={"slide_type": "-"} tags=["remove-output"]
def jointly_distribute(seq1, seq2):
    """Returns the joint distribution of the tuple (i,j) of outcomes from zip(seq1,seq2)."""
    lis_sim = list(zip(seq1, seq2))
    di = {}
    for i in lis_sim:
        if i not in di:
            di[i] = lis_sim.count(i)/len(lis_sim)
    return di


# %% deletable=false editable=false
grader.check("jointly-distribute")

# %% [markdown] slideshow={"slide_type": "fragment"}
# If you have implemented `entropy` and `jointly_distribute` correctly, you can verify that duplicating a sequence will give the same entropy.

# %% slideshow={"slide_type": "-"} tags=["remove-output"]
assert math.isclose(
    entropy(jointly_distribute(coin_flips, coin_flips)), entropy(distribute(coin_flips))
)
assert math.isclose(
    entropy(jointly_distribute(dice_rolls, dice_rolls)), entropy(distribute(dice_rolls))
)

# %% [markdown] slideshow={"slide_type": "fragment"}
# However, for two sequences generated independently, the joint entropy is roughly the sum of the individual entropies.

# %% slideshow={"slide_type": "-"} tags=["remove-output"]
coin_flips_entropy = entropy(distribute(coin_flips))
dice_rolls_entropy = entropy(distribute(dice_rolls))
cf_dr_entropy = entropy(jointly_distribute(coin_flips, dice_rolls))
print(
    f"""Entropy of coin flip: {coin_flips_entropy}
Entropy of dice roll: {dice_rolls_entropy}
Sum of the above entropies: {coin_flips_entropy + dice_rolls_entropy}
Joint entropy: {cf_dr_entropy}"""
)

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Conditional distribution and entropy

# %% [markdown] slideshow={"slide_type": "fragment"}
# Mathematically, we denote a [conditional distribution](https://en.wikipedia.org/wiki/Conditional_probability_distribution) as $\mathbf{q}:=[q_{j|i}]_{i\in \mathcal{S}, j\in \mathcal{T}}$, where 
# - $\mathcal{S}$ and $\mathcal{T}$ are two sets of distinct outcomes, and
# - $q_{j|i}$ denotes the probability (chance) of seeing outcome $j$ given the condition that outcome $i$ is observed.
#
# For $\mathbf{q}$ to be a valid distribution, the probabilities $q_{j|i}$'s have to sum to $1$ for every $i$, i.e.,
#
# $$\sum_{j\in \mathcal{T}} q_{j|i} = 1 \kern1em \text{for all }i\in \mathcal{S} $$

# %% [markdown] slideshow={"slide_type": "fragment"}
# For example, suppose we want to compute the distribution of coin flips given dice rolls, then the following assign `q_H_1` and `q_T_1` to the values $q_{H|1}$ and $q_{T|1}$ respectively:

# %% slideshow={"slide_type": "-"}
coin_flips_1 = [j for i, j in zip(dice_rolls, coin_flips) if i == 1]
q_H_1 = coin_flips_1.count("H") / len(coin_flips_1)
q_T_1 = coin_flips_1.count("T") / len(coin_flips_1)
print("Coin flips given dice roll is 1:", coin_flips_1)
print(
    'Distribution of coin flip given dice roll is 1: {{ "H": {}, "T": {}}}'.format(
        q_H_1, q_T_1
    )
)
assert math.isclose(q_H_1 + q_T_1, 1)

# %% [markdown] slideshow={"slide_type": "fragment"}
# Note that `q_H_1 + q_T_1` is 1 as expected. Similarly, we can assign `q_H_2` and `q_T_2` to the values $q_{H|2}$ and $q_{T|2}$ respectively.

# %% slideshow={"slide_type": "-"}
coin_flips_2 = [j for i, j in zip(dice_rolls, coin_flips) if i == 2]
q_H_2 = coin_flips_2.count("H") / len(coin_flips_2)
q_T_2 = coin_flips_2.count("T") / len(coin_flips_2)
print("Coin flips given dice roll is 2:", coin_flips_2)
print(
    'Distribution of coin flip given dice roll is 2: {{ "H": {}, "T": {}}}'.format(
        q_H_2, q_T_2
    )
)
assert math.isclose(q_H_2 + q_T_2, 1)

# %% [markdown] slideshow={"slide_type": "fragment"}
# Finally, we want to store the conditional distribution as a nested dictionary so that `q[i]` points to the distribution 
#
# $$[q_{j|i}]_{j\in \mathcal{T}} \kern1em \text{for }i\in \mathcal{S}.$$

# %% slideshow={"slide_type": "-"}
q = {}
q[1] = dict(zip("HT", (q_H_1, q_T_1)))
q[2] = dict(zip("HT", (q_H_2, q_T_2)))
q


# %% [markdown] slideshow={"slide_type": "fragment"}
# Of course, the above dictionary is missing the entries for other possible outcomes of the dice rolls.

# %% [markdown] slideshow={"slide_type": "subslide"}
# **Exercise** (conditionally-distribute)
#
# Define a function `conditionally_distribute` that
# - takes two sequences `seq1` and `seq2` of outcomes of the same length, and
# - returns the conditional distribution of `seq2` given `seq1` in the form of a nested dictionary efficiently without storing the unobserved outcomes.
#
# In the above example, `seq1` is `dice_rolls` while `seq2` is `coin_flips`.
#
# ````{hint}
# For an efficient implementation without traversing the input sequences too many times, consider using the following solution template and the `setdefault` method.
# ```Python
# def conditionally_distribute(seq1, seq2):
#     q, count = {}, {}  # NOT q = count = {}
#     for i in seq1:
#         count[i] = count.get(i, 0) + 1
#     for i, j in zip(seq1, seq2):
#         q[i][j] = ____________________________________________________
#     return q
# ```
# ````

# %% nbgrader={"grade": false, "grade_id": "conditionally_distribute", "locked": false, "schema_version": 3, "solution": true, "task": false} slideshow={"slide_type": "-"} tags=["remove-output"]
def conditionally_distribute(seq1, seq2):
    """Returns the conditional distribution q of seq2 given seq1 such that
    q[i] is a dictionary for observed outcome i in seq1 and
    q[i][j] is the probability of observing j in seq2 given the
    corresponding outcome in seq1 is i."""
    q, count = {}, {}  # NOT q = count = {}
    for i in seq1:
        count[i] = count.get(i, 0) + 1
    for i, j in zip(seq1, seq2):
        if i not in q:
            q[i] = {}
        q[i][j] = list(zip(seq1, seq2)).count((i, j))/count[i]
    return q


# %% deletable=false editable=false
grader.check("(conditionally-distribute)")


# %% [markdown] slideshow={"slide_type": "fragment"}
# ---
#
# **Definition** (Conditional entropy)
#
# The [*conditional entropy*](https://en.wikipedia.org/wiki/Conditional_entropy) is defined for a conditional distribution $\mathbf{q}=[q_{j|i}]_{i\in \mathcal{S},j\in\mathcal{T}}$ and a distribution $\mathbf{p}=[p_i]_{i\in \mathcal{S}}$ as follows:
#
# $$ H(\mathbf{q}|\mathbf{p}) = \sum_{i\in \mathcal{S}} p_i \sum_{j\in \mathcal{T}} q_{j|i} \log_2 \frac{1}{q_{j|i}}, $$
# where, by convention,  
# - the summand of the outer sum is 0 if $p_i=0$ (regardless of the values of $q_{j|i}$), and
# - the summand of the inner sum is 0 if $q_{j|i}=0$.
#
# ---

# %% [markdown] slideshow={"slide_type": "subslide"}
# **Exercise** (conditional-entropy)
#
# Define a function `conditional_entropy` that
# - takes 
#   - a distribution p as its first argument,
#   - a conditional distribution q as its second argument, and
# - returns the conditional entropy $H(\mathbf{q}|\mathbf{p})$.
#
# Handle the cases when $p_i=0$ and $q_{j|i}=0$.

# %% nbgrader={"grade": false, "grade_id": "conditional_entropy", "locked": false, "schema_version": 3, "solution": true, "task": false} slideshow={"slide_type": "-"} tags=["remove-output"]
def conditional_entropy(p, q):
    """Returns the conditional entropy of the conditional distribution q given
    the distribution p."""
    result = 0
    for i in q:
        result += p[i] and p[i] * entropy(q[i])
    return result


# %% nbgrader={"grade": true, "grade_id": "test-conditional_entropy", "locked": true, "points": 1, "schema_version": 3, "solution": false, "task": false} slideshow={"slide_type": "-"} tags=["remove-output", "hide-input"]
# tests
cf_given_dr_dist = {
    4: {"T": 0.5588235294117647, "H": 0.4411764705882353},
    1: {"T": 0.46511627906976744, "H": 0.5348837209302325},
    3: {"H": 0.5135135135135135, "T": 0.4864864864864865},
    6: {"H": 0.5454545454545454, "T": 0.45454545454545453},
    2: {"T": 0.7586206896551724, "H": 0.2413793103448276},
    5: {"T": 0.5416666666666666, "H": 0.4583333333333333}}
assert (
    conditional_entropy(
        {"H": 0.5, "T": 0.5}, {"H": {"H": 0.5, "T": 0.5}, "T": {"H": 0.5, "T": 0.5}})
    == 1)
assert (
    conditional_entropy(
        {"H": 0, "T": 1}, {"H": {"H": 0.5, "T": 0.5}, "T": {"H": 0.5, "T": 0.5}})
    == 1)
assert (
    conditional_entropy(
        {"H": 0.5, "T": 0.5}, {"H": {"H": 1, "T": 0}, "T": {"H": 0, "T": 1}})
    == 0)
assert (
    conditional_entropy(
        {"H": 0.5, "T": 0.5}, {"H": {"H": 1, "T": 0}, "T": {"H": 0.5, "T": 0.5}})
    == 0.5)
assert math.isclose(
    conditional_entropy(dice_rolls_dist, cf_given_dr_dist), 0.9664712793722372)


# %% [markdown] slideshow={"slide_type": "subslide"}
# The joint probability $p_{ij}$ over $i\in \mathcal{S}$ and $j\in \mathcal{T}$ can be calculated as follows
#
# $$p_{ij} = p_{i} q_{j|i}$$
# where $p_i$ is the probability of $i$ and $q_{j|i}$ is the conditional probability of $j$ given $i$.

# %% [markdown]
# **Exercise** (joint-distribution)
#
# Define a function `joint_distribution` that
# - takes the distribution $p$ and conditional distribution $q$ as arguments, and
# - returns their joint distribution.

# %% nbgrader={"grade": false, "grade_id": "joint_distribution", "locked": false, "schema_version": 3, "solution": true, "task": false} tags=[]
def joint_distribution(p, q):
    ans = {}
    for i in p:
        for j in q[i]:
            ans[(i, j)] = p[i] * q[i][j]
    return ans


# %% deletable=false editable=false
grader.check("joint-distribution")


# %% [markdown] slideshow={"slide_type": "subslide"}
# Finally, a fundamental information identity relating the joint and conditional entropies is the [*chain rule*](https://en.wikipedia.org/wiki/Conditional_entropy#Chain_rule):

# %% [markdown]
# ---
#
# **Proposition**
#
# The joint entropy is equal to
#
# $$ H(\mathbf{p}) + H(\mathbf{q}|\mathbf{p})$$
#
# for any distribution $\mathbf{p}$ over outcome $i\in \mathcal{S}$ and conditional distribution $\mathbf{q}$ over outcome $j\in \mathcal{T}$ given outcome $i \in \mathcal{S}$. 
#
# ---

# %% [markdown] slideshow={"slide_type": "fragment"}
# If you have implemented `jointly_distribute`, `conditionally_distribute`, `entropy`, and `conditional_entropy` correctly, we can verify the identity as follows.

# %% slideshow={"slide_type": "-"}
def validate_chain_rule(seq1, seq2):
    p = distribute(seq1)
    q = conditionally_distribute(seq1, seq2)
    pq = jointly_distribute(seq1, seq2)
    H_pq = entropy(pq)
    H_p = entropy(p)
    H_q_p = conditional_entropy(p, q)
    print(
        f"""Entropy of seq1: {H_p}
Conditional entropy of seq2 given seq1: {H_q_p}
Sum of the above entropies: {H_p + H_q_p}
Joint entropy: {H_pq}"""
    )
    assert math.isclose(H_pq, H_p + H_q_p)


# %% slideshow={"slide_type": "-"} tags=["remove-output"]
validate_chain_rule(coin_flips, coin_flips)

# %% slideshow={"slide_type": "-"} tags=["remove-output"]
validate_chain_rule(dice_rolls, coin_flips)

# %% [markdown] editable=false tags=["remove-cell"]
# ## Submission
#
# Make sure you have run all cells in your notebook in order before running the cell below, so that all images/graphs appear in the output. The cell below will generate a zip file for you to submit. **Please save before exporting!**

# %% deletable=false editable=true tags=[]
# extra files to submit
extra_files = []

# %% deletable=false editable=false tags=[]
# Generate the source main.py necessary for grading and similarity check.
jupytext.write(jupytext.read("main.ipynb"), "main.py", fmt="py:percent")

# %% deletable=false editable=false tags=[]
# Generate the zip file to submit.
grader.export(pdf=False, run_tests=False, files=["main.py", *extra_files])

# %%
