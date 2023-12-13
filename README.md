
# Options Framework for the Four-Rooms Domain

<p align="center">
    <img src="https://github.com/TristanBester/options/blob/main/results/options_clip.gif" alt="Your GIF"/>
</p>

## Ovierview
- This repository contains the code used to train and evaluate the options framework on the four-rooms domain. The options framework is a hierarchical reinforcement learning framework that allows agents to learn temporally extended actions, called options, in addition to primitive actions. 
- The agent is trained and evaluated on the four-rooms domain. The four-rooms domain is a simple grid-world environment that consists of four rooms connected by narrow corridors. The agent is tasked with navigating to a goal location in the environment. The agent is provided with a set of primitive actions that allow it to move in the four cardinal directions.
- The agent is also provided with a set of options that allow it to move to the goal location in each of the four rooms. The agent is rewarded for reaching the goal location.

<p align="center">
    <img src="https://github.com/TristanBester/options/blob/main/results/results.png" alt="Your GIF" width="500" />
</p>

## Prerequisites
- Python: 3.11+
- Poetry: 
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

## Getting Started
Clone the repository:

```bash
git clone https://github.com/TristanBester/options.git
cd options
``` 

Install dependencies:

```bash
poetry install
```

Run the simulation:

```bash
cd src && poetry run main.py
```



