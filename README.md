# 🚕 Taxi RL Agent

A professional Reinforcement Learning project that trains an agent to navigate a taxi in a grid world, pick up passengers, and deliver them to their destinations.

## 🎯 Project Overview

The agent learns to:
- Navigate a 5x5 grid world
- Pick up passengers from one of 4 locations
- Deliver passengers to their destination
- Minimize steps and avoid illegal actions

## 🏗️ Project Structure

```
taxi-rl-agent/
├── config/          # Configuration files
├── src/             # Source code modules
├── scripts/         # Executable scripts
├── models/          # Saved model checkpoints
└── logs/            # Training logs and metrics
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ziadsalama95/taxi-rl-agent.git
cd taxi-rl-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
python scripts/train.py
```

### Evaluation

```bash
python scripts/evaluate.py --model models/q_table_final.npy
```

### Interactive Play

```bash
python scripts/play.py --model models/q_table_final.npy
```

## 📊 Results

After training for 10,000 episodes:
- **Average Reward:** ~8.0
- **Success Rate:** ~99%
- **Average Steps:** ~13