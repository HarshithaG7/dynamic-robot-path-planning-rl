\# Dynamic Robot Path Planning Using Reinforcement Learning



\## 1. Project Overview



This project explores dynamic robot path planning using Reinforcement Learning (RL). A robot must navigate a 10 × 10 grid from a starting position to a goal while avoiding fixed walls and moving obstacles.



We plan to implement and compare two reinforcement learning algorithms:



\- \*\*Q-Learning\*\*

\- \*\*SARSA\*\*



The project will evaluate their performance using success rate, average steps, average cumulative reward, and collision rate. A Streamlit dashboard will be used to visualize the environment and display the evaluation results.



\## 2. Team Responsibilities



\### Member 1 — Environment Development

Responsible for:

\- Implementing the 10 × 10 grid environment.

\- Defining the robot's starting position and goal.

\- Adding fixed walls and moving obstacles.

\- Implementing state transitions, valid actions, rewards, and collision detection.

\- Providing a consistent environment interface for the RL agents.



\*\*Expected integration file:\*\* `environment.py`



\### Member 2 — Reinforcement Learning Algorithms

Responsible for:

\- Implementing the Q-Learning agent.

\- Implementing the SARSA agent.

\- Training the agents in the environment.

\- Recording episode results, including success, steps, cumulative reward, and collisions.

\- Providing a consistent interface for training and testing both algorithms.



\*\*Expected integration file:\*\* `agents.py`



Additional training utilities may be organized in `training.py`.



\### Member 3 — Dashboard, Evaluation, and Integration

\*\*Current owner: Member 3\*\*



Work completed so far:



\- Created the initial Streamlit dashboard in `app.py`.

\- Added sidebar controls for selecting Q-Learning, SARSA, or comparing both algorithms.

\- Added configuration controls for training episodes, learning rate (alpha), discount factor (gamma), and exploration rate (epsilon).

\- Created a visual 10 × 10 grid showing the robot, goal, fixed walls, and moving obstacles.

\- Added metric cards, charts, and a comparison table layout.

\- Created `requirements.txt` with the required Python libraries:

&#x20; - Streamlit

&#x20; - NumPy

&#x20; - Matplotlib

&#x20; - Pandas

\- Installed the dependencies and successfully launched the dashboard locally.

\- Pushed the initial dashboard and dependency file to the shared GitHub repository.



\*\*Evaluation work:\*\*



\- Preparing `evaluation.py` to calculate success rate, average steps, average cumulative reward, and collision rate.

\- Adding utilities to validate episode results, generate per-episode DataFrames, print reports, and compare Q-Learning with SARSA.

\- The evaluation module needs to be tested and connected to the actual outputs from Members 1 and 2.



\*\*Important status note:\*\* The current dashboard contains illustrative demonstration values and a sample environment visualization. These are placeholders, not experimental results from trained agents. They must be replaced with real environment and training outputs during integration.



\## 3. Current Project Structure



```text

dynamic-robot-path-planning-r1/

├── app.py

├── requirements.txt

├── README.md

├── environment.py       # Member 1

├── agents.py            # Member 2

├── training.py          # Member 2 / agreed team ownership

├── evaluation.py        # Member 3

└── visualization.py     # Shared responsibility / to be agreed

```



Files such as `environment.py`, `agents.py`, `training.py`, and `visualization.py` may still need to be created or integrated. Confirm file ownership and interfaces with the team before adding duplicate implementations.



\## 4. Technology Stack



\- Python

\- Streamlit — dashboard

\- NumPy — numerical operations

\- Matplotlib — plots and visualization

\- Pandas — evaluation tables and data processing

\- Git and GitHub — version control and collaboration



\## 5. Setup Instructions



Make sure Python and Git are installed.



Clone the repository:



```bash

git clone https://github.com/HarshithaG7/dynamic-robot-path-planning-rl.git

cd dynamic-robot-path-planning-rl

```



Install dependencies:



```bash

python -m pip install -r requirements.txt

```



Run the dashboard:



```bash

python -m streamlit run app.py

```



Run the evaluation utility after `evaluation.py` has been added:



```bash

python evaluation.py

```



The evaluation script's built-in test data is only for checking that the calculations work. It is not evidence of the trained agents' performance.



\## 6. Integration Requirements



To connect the environment, algorithms, evaluation, and dashboard, the team should agree on a common interface.



Each episode result should contain at least:



```python

{

&#x20;   "success": True,       # Whether the robot reached the goal

&#x20;   "steps": 18,           # Number of steps in the episode

&#x20;   "total\_reward": 75.0,  # Cumulative episode reward

&#x20;   "collision": False     # Whether a collision occurred

}

```



These values are an example of the expected format, not actual experiment results.



Members 1 and 2 should coordinate with Member 3 on:



1\. How the environment is initialized and reset.

2\. How the agent selects an action and receives the next state and reward.

3\. How moving obstacles are represented in the state.

4\. How training results and episode histories are returned.

5\. How the dashboard obtains real evaluation metrics and the final robot path.



\*\*Important:\*\* The exact function names and return formats should be agreed upon before integration. Please share your interfaces with Member 3 so the dashboard and evaluation code can be connected without unnecessary rewriting.



\## 7. Evaluation Metrics



| Metric | Meaning |

|---|---|

| Success Rate (%) | Percentage of episodes in which the robot reaches the goal |

| Average Steps | Mean number of steps per episode |

| Average Cumulative Reward | Mean total reward per episode |

| Collision Rate (%) | Percentage of episodes involving a collision |



Q-Learning and SARSA should be evaluated under comparable environment conditions and training settings.



\## 8. Current Status and Next Steps



\- \[x] Create initial Streamlit dashboard.

\- \[x] Add dashboard configuration controls and visualization layout.

\- \[x] Add Python dependencies.

\- \[x] Launch the dashboard locally.

\- \[x] Push the initial dashboard files to GitHub.

\- \[x] Finish and test `evaluation.py`.

\- \[ ] Integrate Member 1's environment implementation.

\- \[ ] Integrate Member 2's Q-Learning and SARSA implementations.

\- \[ ] Replace demonstration values with actual training results.

\- \[ ] Connect real evaluation metrics and robot paths to the dashboard.

\- \[ ] Run experiments and compare both algorithms.

\- \[ ] Test the complete application and prepare the final demonstration.



\## 9. Collaboration Notes



Please commit your work regularly and communicate any changes to shared function names or data formats before merging them.



Before pushing changes to `main`, coordinate with the team to avoid overwriting or conflicting with another member's work. If possible, use separate feature branches and merge tested changes.



The project is considered complete only after the environment, both RL agents, evaluation utilities, and dashboard work together using actual experimental results.



