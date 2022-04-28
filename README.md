## Installation 
This project requires python>= 3.6

First clone the repo with"
```
git clone --recurse-submodules https://github.com/nicofirst1/MAMBRL.git
```

We are using a custom version of the PettingZoo library, in order to install it execute:

```
pip install PettingZoo/
pip install .

```
If you wish to modify the library while coding use `pip install -e .` instead.

# Training
The pipeline is made of serveral architectures that can be trained separately.

## ModelFree and EnvModel
The [model free](src/model/ModelFree.py) and [env model](src/model/EnvModel.py) can be trained with 
```shell
python src/trainer/ModelFreeTrainer.py
or
python src/trainer/EnvModelTrainer.py
```
You can also pass arguements to the trained pipeline such as:
```shell
python src/trainer/ModelFreeTrainer.py -use_wandb --horizon 128 --lr 0.0001
```
For a complete list of parameters check the [Parameter class](src/common/Params.py).

## Full Training
If you want to train the whole architecture use the [FullTrainer](src/trainer/FullTrainer.py) with:
```shell
python src/trainer/FullTrainer.py
```

## Training on colab
You can train the model on
 [![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nicofirst1/MAMBRL)

The notebook uses [weight and biases](https://wandb.ai/) in order to keep track of the experiment.

# Problem Statement

## Description of the Problem:

<b><u>Abstract Definition:</b></u>

Consider 'N' points on a 2D plane which need to be visited at least once by 'M' bots.
Provide a path for all of the 'M' bots such that the total distance travelled by the bots is minimum.

<b><u>Consider the following </b></u>

<b> Target Points </b>
P1, a set of ‘N’ points
P1 = { p1, p2, p3, … pN }

<b> Bots </b>
P2, a set of 'M' points 
P2  = { b1, b2, b3, ... bM }

<b> <u> M<<N </b> </u>

Return a matrix Y which gives the sequence of points needed by each of the bots to cover all the points minimizzing the total distace travelled.

<b> Y[i,j] = ith point in the path of bot j.

## Attempt to Solution
This project aims to design and test a customized model of Markov Decision Process and attempt to solve it using Cooperative Q-Learning.

<b> Reasons for Multi Agent Reinforcement Learning to be the best approach for solution </b>

<b> <u> M<<N </b> </u>

1.  Consider the use case where M=50 (50 bots) and N=1000 (1000 Target Points)
2. The total number of possible paths is in order of 10^90 which is much greater than the generic use cases of 20 possible paths
3.  The environment is unexplored, meaning that location of all the 1000 points is unknown, The bots need to explore and simultaneously learn to move efficiently in the environment. Absence of exact knowledge of environment rules out the validity of deterministic algorithms.

In this project, the notion of multi-agent envronment is implemented using a novel method which we call <b> ConnectedQ </b>


## Model based RL 
- Check [this](https://bair.berkeley.edu/blog/2019/12/12/mbpo/) out for an overview on model based RL. Specifically we refer to the data generation technique.

## TODOs


- [] add tune configurations
- [] dockerize
- 