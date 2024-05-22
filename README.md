# Football Match Data Analysis

## Overview

This project involves the analysis of a football match dataset. The dataset contains various features related to football matches, including player actions and outcomes. The goal is to explore the data, derive insights, and visualize key aspects such as goal distributions, shooting distances, and player performance.

## Requirements

- Python 3.x
- Libraries:
  - pandas
  - numpy
  - matplotlib

## Dataset

The dataset used for this analysis is `train.csv`, which contains information about football matches. The main features include:
- `matchId`: Identifier for the match
- `playerId`: Identifier for the player
- `playType`: Type of play
- `bodyPart`: Part of the body used for the action
- `x`: X-coordinate on the pitch
- `y`: Y-coordinate on the pitch
- `distance`: Calculated distance from the origin (0,0) on the pitch
- `interveningOpponents`: Number of opponents intervening
- `interveningTeammates`: Number of teammates intervening
- `interferenceOnShooter`: Interference on the shooter
- `minute`: Minute of the play
- `second`: Second of the play
- `outcome`: Outcome of the play (e.g., goal)

## Steps

1. **Import Libraries and Load Data**

   Import necessary libraries and load the dataset into a pandas DataFrame.

2. **Data Exploration**

   - Display the first few rows of the dataset.
   - Determine the shape of the dataset.
   - Count unique players.
   - Filter data to include only goals.

3. **Player Goal Analysis**

   - Create a dictionary to count the number of goals per player.
   - Identify the player with the maximum number of goals.
   - Calculate goal statistics using pandas.

4. **Goal and Shot Rates**

   - Calculate the goal rate for each player by dividing the number of goals by the number of shots.
   - Identify players with the highest and lowest goal rates.

5. **Distance Calculation**

   - Calculate the distance from the origin (0,0) for each play using the Pythagorean theorem.
   - Update the DataFrame to include this distance.

6. **Visualization**

   - Scatter plot of goals and non-goals based on coordinates (x, y).
   - Histogram and box plot visualizations for the x-coordinates of goals and non-goals.

## Running the Code

1. **Setup**

   Ensure you have the required libraries installed. You can install them using pip:
   ```bash
   pip install pandas numpy matplotlib
   ```

2. **Execution**

   Place the `train.csv` file in the appropriate directory and run the code in a Python environment.

3. **Output**

   The code will generate several plots that visualize the relationships between different features in the dataset. It will also print statistical insights to the console.

## Code

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv(r'C:\MachineLearning\4\352f1626-6710-4ac7-b1c7-4f4b6ee3cf68230721-0609481\train.csv')

# Display the first few rows of the dataset
df.head()

# Determine the shape of the dataset
df.shape

# Count unique players
len(set(df.playerId))
len(df['playerId'].unique())
df['playerId'].nunique()

# Filter data to include only goals
df_goal = df[df['outcome'] == 'گُل']
df_goal.head()

# Count the number of goals per player
d = {}
for p in df_goal.playerId:
    d.setdefault(p,0)
    d[p] = d[p] + 1

# Identify the player with the maximum number of goals
m = max(d.values())
for k,v in d.items():
    if v == m:
        print(k)

# Calculate goal statistics using pandas
p_goal = df_goal['playerId'].value_counts()
p_goal.index[0]
p_goal.idxmax()
p_goal.idxmin()
p_goal[p_goal.idxmin()]
p_goal[p_goal.idxmax()]
p_goal.index.max()

# Calculate goal and shot rates
p_shoot = df['playerId'].value_counts()
rate = p_goal/p_shoot
rate
rate.idxmax()
rate[rate.idxmax()]
rate.idxmin()
rate[rate.idxmin()]

# Calculate the distance from the origin
df['distance'] = np.sqrt(df['x']**2 + df['y']**2)

# Update the DataFrame to include relevant features
df = df[['matchId', 'playerId', 'playType', 'bodyPart', 'x', 'y', 'distance',
       'interveningOpponents', 'interveningTeammates', 'interferenceOnShooter',
       'minute', 'second', 'outcome']]
df.head()

# Separate data into goals and non-goals
not_goal = df[df['outcome'] != 'گُل']
goal = df[df['outcome'] == 'گُل']

# Scatter plot of goals and non-goals
plt.scatter(not_goal.x,not_goal.y,label = 'not_goal' , color = 'blue', marker='^')
plt.scatter(goal.x,goal.y,label = 'goal', color = 'red', marker='o'  )
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

# Histogram and box plot visualizations
plt.subplot(221)
plt.hist(not_goal.x)
plt.xlabel('x')
plt.subplot(222)
plt.hist(goal.x)
plt.xlabel('x')
plt.subplot(223)
plt.boxplot(not_goal.x, vert= False)
plt.subplot(224)
plt.boxplot(goal.x, vert= False)
plt.show()
```

## Conclusion

The analysis of the football match dataset reveals significant insights into player performance, goal distribution, and shooting distances. These insights can help in understanding player behaviors and improving strategies for future matches.
