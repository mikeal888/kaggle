import pandas as pd

# Load training data
train_df = pd.read_csv('train.csv')
train_df = train_df.drop(columns='id')

# Load test data from processing
test_df = pd.read_csv('test.csv')
test_df = test_df.drop(columns='id')

# There are some zero values in the test data
# This is the most important part: we need to accurately impute these 
test_df[test_df['z'] == 0] = test_df['z'].mean()
test_df[test_df['x'] == 0] = test_df['x'].mean()
test_df[test_df['y'] == 0] = test_df['y'].mean()

# We will load the original data since it helps with the predicitoj
original_df = pd.read_csv("cubic_zirconia.csv")

# Impute data and then merge them
original_df['depth'] = original_df['depth'].fillna(2*original_df['z']/(original_df['x']+original_df['y']))
original_df = original_df.drop(columns='Unnamed: 0')

# concat the two dataframes
train_df = pd.concat([train_df, original_df])

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
test_df = preprocessing(test_df)

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
test_df = create_features(test_df)

# So it makes sense to drop density that are far away from the average
train_df.drop(train_df[(train_df['density'] < 0.004)].index, inplace=True)
train_df.drop(train_df[(train_df['density'] > 0.01)].index, inplace=True)

# Depth to table ration 
train_df.drop(train_df[(train_df['depth_to_table_ratio'] < 0.25)].index, inplace=True)

# Generate training datasets
train_df.to_csv('training_data.csv', index=False)
test_df.to_csv('testing_data.csv', index=False)