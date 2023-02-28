import pandas as pd
import numpy as np

######  Preprocessing  ######
## First we will process the training data and mix it with the original data

# Load training data
train_df = pd.read_csv('./Data/train.csv')
train_df = train_df.drop(columns='id')

# We will load the original data since it helps with the predicitoj
original_df = pd.read_csv("./Data/cubic_zirconia.csv")

# Impute data and then merge them
original_df['depth'] = original_df['depth'].fillna(2*original_df['z']/(original_df['x']+original_df['y']))
original_df = original_df.drop(columns='Unnamed: 0')

# concat the two dataframes
train_df = pd.concat([train_df, original_df])

# reset index of train_df
train_df.reset_index(drop=True)

# preprocessing
cut_labeling = {col: val for val, col in enumerate(['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])}
color_labeling = {col: val for val, col in enumerate(['J', 'I', 'H', 'G', 'F', 'E', 'D'])}
clarity_labeling = {col: val for val, col in enumerate(
    ['I3', 'I2', 'I1', 'SI2', 'SI1', 'VVS2', 'VVS1', 'VS2', 'VS1', 'IF', 'FL'])}
# to preprocess label features, map label to integer number.
def preprocessing(df):
    df['cut'] = df['cut'].map(cut_labeling)
    df['color'] = df['color'].map(color_labeling)
    df['clarity'] = df['clarity'].map(clarity_labeling)
    return df

# Process data
train_df = preprocessing(train_df)

# We can impute several values, but given that we have 19000 data points, we can just drop those 
for key in ['x', 'y', 'z']:
    train_df = train_df.drop(train_df[train_df[key] == 0].index)

# Create a bunch of new features
def create_features(df):
    # Create new features for learning
    df['volume'] = df['x'] * df['y'] * df['z']  # Volumen of gems
    df['density'] = df['carat'] / df['volume']  # Density of gems
    df['table_percentage'] = (df['table'] / ((df['x'] + df['y']) / 2)) * 100    # The table percentage (of the gem face)
    df['depth_percentage'] = (df['depth'] / ((df['x'] + df['y']) / 2)) * 100    # The depth percentage
    # df['symmetry'] = (abs(df['x'] - df['z']) + abs(df['y'] - df['z'])) / (df['x'] + df['y'] + df['z'])  # Symmetry of the gem
    df['surface_area'] = 2 * ((df['x'] * df['y']) + (df['x'] * df['z']) + (df['y'] * df['z']))  # Surface area of gem
    df['depth_to_table_ratio'] = df['depth'] / df['table']  # ratio

    return df

# Create new features
train_df = create_features(train_df)

# So it makes sense to drop density that are far away from the average
train_df.drop(train_df[(train_df['density'] < 0.004)].index, inplace=True)
train_df.drop(train_df[(train_df['density'] > 0.01)].index, inplace=True)

# Depth to table ration 
train_df.drop(train_df[(train_df['depth_to_table_ratio'] < 0.25)].index, inplace=True)

# Change to log of price since it is skewed
train_df['price'] = np.log(train_df['price'])

# Generate training datasets
train_df.to_csv('./Data/training_data.csv', index=False)



##### Prepare test data #####
# Here we will use the most similar data relating the table and the depth to the test data

# Load test data from processing
test_df = pd.read_csv('./Data/test.csv')

# Make a copy of test data which only includes columns depth and table
test_df_copy = test_df[['id', 'depth', 'table']].copy()

# Find most similar data to those with ids_x in test_data
# We know that if x is missing then so to is y
ids_x = test_df[(test_df['x'] == 0)]['id']

# Loop through all the ids
for ix in ids_x.values:
    # Get the row and only keep the depth and table columns
    row = test_df[test_df['id'] == ix][['id', 'depth', 'table']]

    # Find the most similar data
    diff_df = test_df_copy[['depth', 'table']] - row[['depth', 'table']].values
    norm_df = diff_df.apply(lambda x: np.linalg.norm(x), axis=1)
    similar = test_df_copy.loc[norm_df.idxmin()]['id']

    # impute x and y of the similar data into the test data
    test_df.loc[test_df['id'] == ix, 'x'] = test_df[test_df['id'] == similar]['x'].values
    test_df.loc[test_df['id'] == ix, 'y'] = test_df[test_df['id'] == similar]['y'].values


# Now we can impute the z values
inds = test_df['z'][test_df['z']==0].index
test_df.loc[inds, 'z'] = test_df['depth'][inds] * (test_df['x'][inds] + test_df['y'][inds]) /200

# So it looks like we have a lot of outliers in the test data in z, so we will replace it with the formula z = depth * (x + y) / 200
args = test_df['z'].idxmax()
test_df.loc[args, 'z'] = test_df.iloc[args]['depth'] * (test_df.iloc[args]['x'] + test_df.iloc[args]['y']) / 200

# Now we can preprocess the data similar to the training data
test_df = preprocessing(test_df)
test_df = create_features(test_df)

# Drop the id column since we don't want it for training
test_df = test_df.drop(columns='id')

# Save the test data
test_df.to_csv('./Data/testing_data.csv', index=False)