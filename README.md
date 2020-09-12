---

<div align="center">    
 
# Improved Model Based RL using Action Narration and Group Discussion    

[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/ICLR-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)  
<!--
ARXIV   
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->


<!--  
Conference   
-->   
</div>
 
## Description   
In this project we demonstrate the ability of reinforcement learning agents to produce meaninful 'narration' of actions that can be used for the instruction of other agents and for planning in latent space.

## How to run   
First, install dependencies   
```bash
# clone MBAN 
git clone https://github.com/Gregory-Eales/MBAN

# install MBAN   
cd MBAN
pip install -e .   
pip install -r requirements.txt
 ```   
 Next, navigate to any file and run it.   
 ```bash
# module folder
cd mban

# run module (example: mnist as your main contribution)   
python train.py    
```

## Imports
This project is setup as a package which means you can now easily import any file into any other file like so:
```python
from mban.envs.env import Env
from mban.agents import Agent
from pytorch_lightning import Trainer

# agent
agent = Agent()

# environment
env = Env()

# train
agent.train(env)

# test
agent.run(env)
```

### Citation   
```
@article{Gregory Eales,
  title={Improved Model Based RL using Action Narration and Group Discussion},
  author={independent},
  journal={arxiv},
  year={2020}
}
```   
