
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ==================================================
# PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="Dynamic Robot Path Planning",
    page_icon="🤖",
    layout="wide"
)

# ==================================================
# TEMPORARY DEMO ENVIRONMENT
# This will later be connected to the actual
# environment developed by Member 1.
# ==================================================
GRID_SIZE = 10

START = (0, 0)
GOAL = (9, 9)

FIXED_WALLS = [
    (1, 3), (2, 3), (3, 3),
    (5, 1), (5, 2), (5, 3),
    (6, 7), (7, 7), (8, 7)
]

MOVING_OBSTACLES = [
    (2, 6), (6, 4), (8, 2)
]


# ==================================================
# DRAW THE 2D GRID
# ==================================================
def draw_grid(robot_position=START, path=None):
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

    # Fixed walls
    for row, col in FIXED_WALLS:
        grid[row, col] = 1

    # Moving obstacles
    for row, col in MOVING_OBSTACLES:
        grid[row, col] = 2

    # Start and goal
    grid[START] = 3
    grid[GOAL] = 4

    # Optional robot path
    if path:
        for row, col in path:
            if (
                (row, col) not in [START, GOAL]
                and (row, col) not in FIXED_WALLS
                and (row, col) not in MOVING_OBSTACLES
            ):
                grid[row, col] = 5

    # Robot position
    grid[robot_position] = 6

    colors = [
        "white",       # 0: Empty cell
        "dimgray",     # 1: Fixed wall
        "tomato",      # 2: Moving obstacle
        "limegreen",   # 3: Start
        "royalblue",   # 4: Goal
        "lightyellow", # 5: Path
        "gold"         # 6: Robot
    ]

    cmap = ListedColormap(colors)

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.imshow(grid, cmap=cmap, vmin=0, vmax=6)

    ax.set_xticks(range(GRID_SIZE))
    ax.set_yticks(range(GRID_SIZE))

    ax.set_xticks(
        np.arange(-0.5, GRID_SIZE, 1),
        minor=True
    )
    ax.set_yticks(
        np.arange(-0.5, GRID_SIZE, 1),
        minor=True
    )

    ax.grid(
        which="minor",
        color="black",
        linewidth=0.7
    )

    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_title("2D Robot Grid — Demo Preview")
    ax.invert_yaxis()

    fig.tight_layout()
    return fig


# ==================================================
# GENERATE ILLUSTRATIVE DEMO DATA
# These are NOT real RL experiment results.
# ==================================================
def generate_demo_results(episodes):
    rng = np.random.default_rng(42)

    # Illustrative values for testing the interface only
    q_success = min(
        95,
        45 + int(12 * np.log10(max(episodes, 1)))
    )
    sarsa_success = max(0, q_success - 5)

    return pd.DataFrame({
        "Algorithm": ["Q-learning", "SARSA"],
        "Success Rate (%)": [
            q_success,
            sarsa_success
        ],
        "Average Steps": [
            int(rng.integers(18, 28)),
            int(rng.integers(20, 32))
        ],
        "Average Reward": [
            int(rng.integers(40, 80)),
            int(rng.integers(30, 70))
        ],
        "Collision Rate (%)": [
            int(rng.integers(5, 20)),
            int(rng.integers(8, 25))
        ]
    })


# ==================================================
# SIDEBAR: TRAINING SETTINGS
# ==================================================
st.sidebar.title("Training Settings")

algorithm = st.sidebar.selectbox(
    "Select algorithm",
    ["Q-learning", "SARSA", "Compare both"]
)

episodes = st.sidebar.slider(
    "Training episodes",
    min_value=100,
    max_value=2000,
    value=1000,
    step=100
)

learning_rate = st.sidebar.slider(
    "Learning rate (alpha)",
    min_value=0.01,
    max_value=1.0,
    value=0.1,
    step=0.01
)

gamma = st.sidebar.slider(
    "Discount factor (gamma)",
    min_value=0.1,
    max_value=0.99,
    value=0.95,
    step=0.01
)

epsilon = st.sidebar.slider(
    "Initial exploration rate",
    min_value=0.0,
    max_value=1.0,
    value=1.0,
    step=0.05
)

st.sidebar.caption(
    "These settings will be connected to the actual "
    "training module during integration."
)


# ==================================================
# MAIN DASHBOARD HEADER
# ==================================================
st.title("🤖 Dynamic Robot Path Planning")

st.subheader(
    "Reinforcement Learning with Dynamic Obstacles"
)

st.write(
    "Explore how reinforcement learning can help a robot "
    "navigate a grid, reach its goal, and avoid obstacles."
)

st.info(
    "Dashboard development preview. The grid and metrics "
    "are placeholders until the real environment, agents, "
    "and evaluation modules are connected."
)


# ==================================================
# DASHBOARD BUTTONS
# ==================================================
button_col1, button_col2 = st.columns(2)

with button_col1:
    preview_clicked = st.button(
        "Generate Dashboard Preview",
        use_container_width=True
    )

with button_col2:
    clear_clicked = st.button(
        "Clear Preview",
        use_container_width=True
    )

if "demo_results" not in st.session_state:
    st.session_state.demo_results = None

if preview_clicked:
    st.session_state.demo_results = generate_demo_results(
        episodes
    )

if clear_clicked:
    st.session_state.demo_results = None


# ==================================================
# PERFORMANCE METRICS
# ==================================================
st.subheader("Performance Overview")

results = st.session_state.demo_results

if results is None:
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Success Rate", "—")
    col2.metric("Average Steps", "—")
    col3.metric("Average Reward", "—")
    col4.metric("Collision Rate", "—")

    st.caption(
        "Click Generate Dashboard Preview to test the "
        "dashboard using illustrative sample values."
    )

else:
    if algorithm == "Compare both":
        selected_results = results
    else:
        selected_results = results[
            results["Algorithm"] == algorithm
        ]

    if not selected_results.empty:
        row = selected_results.iloc[0]

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Success Rate",
            f"{row['Success Rate (%)']}%"
        )
        col2.metric(
            "Average Steps",
            int(row["Average Steps"])
        )
        col3.metric(
            "Average Reward",
            int(row["Average Reward"])
        )
        col4.metric(
            "Collision Rate",
            f"{row['Collision Rate (%)']}%"
        )


# ==================================================
# ROBOT GRID AND ENVIRONMENT DETAILS
# ==================================================
st.subheader("2D Robot Simulation")

grid_col, details_col = st.columns([3, 2])

with grid_col:
    fig = draw_grid()
    st.pyplot(fig)
    plt.close(fig)

with details_col:
    st.markdown("### Grid Legend")

    st.markdown(
        """
        - 🟩 **Green:** Start position
        - 🟦 **Blue:** Goal position
        - 🟨 **Yellow:** Robot position
        - ⬛ **Gray:** Fixed wall
        - 🟥 **Red:** Moving obstacle
        - ⬜ **White:** Free cell
        """
    )

    st.markdown("### Environment Information")

    st.write(f"Grid size: {GRID_SIZE} × {GRID_SIZE}")
    st.write(f"Fixed walls: {len(FIXED_WALLS)}")
    st.write(
        f"Moving obstacles: {len(MOVING_OBSTACLES)}"
    )
    st.write(f"Start position: {START}")
    st.write(f"Goal position: {GOAL}")


# ==================================================
# Q-LEARNING VS SARSA COMPARISON
# ==================================================
st.subheader("Algorithm Comparison")

results = st.session_state.demo_results

if results is not None:
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("**Success Rate (%)**")

        st.bar_chart(
            results.set_index("Algorithm")[
                ["Success Rate (%)"]
            ]
        )

    with chart_col2:
        st.markdown("**Average Steps**")

        st.bar_chart(
            results.set_index("Algorithm")[
                ["Average Steps"]
            ]
        )

    st.markdown("### Comparison Table")

    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True
    )

    st.warning(
        "These numbers are illustrative demo values, "
        "not results from trained reinforcement learning "
        "agents. Replace them with actual evaluation data."
    )

else:
    st.write(
        "Comparison charts will appear after you generate "
        "the preview or connect the actual evaluation module."
    )


# ==================================================
# FOOTER
# ==================================================
st.divider()

st.caption(
    "Academic Project | Dynamic Robot Path Planning "
    "Using Reinforcement Learning"
)
