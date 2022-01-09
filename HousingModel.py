import pandas as pd                                             #libraries to be used
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model

test = pd.read_csv('jtest.csv', index_col='Id')                 #importing data
test.drop(labels='Unnamed: 0', axis=1, inplace=True)            #drop Unnamed: 0, duplicate primary key

#set threshold for correlation minimum value
threshold = 0.6

# Exterior Quality, Exterior Condition, HeatingQC, KitchenQual replacements are all the same scale
test['ExterQual'].replace(to_replace=['Ex','Gd','TA','Fa','Po'], value=[10,8,6,4,2], inplace=True)
test['ExterCond'].replace(to_replace=['Ex','Gd','TA','Fa','Po'], value=[10,8,6,4,2], inplace=True)
test['HeatingQC'].replace(to_replace=['Ex','Gd','TA','Fa','Po'], value=[10,8,6,4,2], inplace=True)
test['KitchenQual'].replace(to_replace=['Ex','Gd','TA','Fa','Po'], value=[10,8,6,4,2], inplace=True)

# These rated categories also leave room for N/A options, which I have replaced with NaNs 
test['BsmtQual'].replace(to_replace=['Ex','Gd','TA','Fa','Po','NA'], value=[10,8,6,4,2,np.NaN], inplace=True)
test['BsmtCond'].replace(to_replace=['Ex','Gd','TA','Fa','Po','NA'], value=[10,8,6,4,2,np.NaN], inplace=True)
test['FireplaceQu'].replace(to_replace=['Ex','Gd','TA','Fa','Po','NA'], value=[10,8,6,4,2,np.NaN], inplace=True)
test['GarageQual'].replace(to_replace=['Ex','Gd','TA','Fa','Po','NA'], value=[10,8,6,4,2,np.NaN], inplace=True)
test['GarageCond'].replace(to_replace=['Ex','Gd','TA','Fa','Po','NA'], value=[10,8,6,4,2,np.NaN], inplace=True)
test['PoolQC'].replace(to_replace=['Ex','Gd','TA','Fa','NA'], value=[10,8,6,4,np.NaN], inplace=True)

# Create neighborhood scale column
neighborhoodValues = test.groupby('Neighborhood', as_index=False)['SalePrice'].mean().round(0).sort_values('SalePrice')
neighborhoodValues['SalePrice'] = neighborhoodValues['SalePrice']/50000
neighborhoodNames = np.asarray(neighborhoodValues['Neighborhood']).transpose()
neighborhoodScale = np.asarray(neighborhoodValues['SalePrice']).transpose()
test['NeighborhoodValue'] = test['Neighborhood']
test['NeighborhoodValue'].replace(to_replace=neighborhoodNames, value=neighborhoodScale, inplace=True)

# Create aggregate colums for total bathrooms from divided counts
# Bathroom columns: BsmtFullBath, BsmtHalfBath, FullBath, HalfBath
total_bathrooms = test['BsmtFullBath'] + test['FullBath'] + (test['BsmtHalfBath']/2) + (test['HalfBath']/2)
test['Bathrooms'] = total_bathrooms

# Create aggregate of total square footage by combining 
# GrLivArea, GarageArea, TotalBsmtSF, OpenPorchSF, WoodDeckSF
# EnclosedPorch, 3SsnPorch, ScreenPorch, PoolArea
total_sf = test['GrLivArea'] + test['GarageArea'] + test['TotalBsmtSF'] + test['OpenPorchSF'] + test['WoodDeckSF'] + test['EnclosedPorch'] + test['3SsnPorch'] + test['ScreenPorch'] + test['PoolArea']
test['TotalSpace'] = total_sf

#assigning which columns to check for null values, as those will be used in correlation calculations
test_corr = pd.DataFrame(test.select_dtypes(include=[np.number]).corr()['SalePrice'].sort_values(ascending=False))
testcols = np.array(test_corr[test_corr>threshold].dropna().index)

# Check important columns for null values
nullcols_test = test[testcols].isnull().sum()>0
ncols_test = np.array(nullcols_test[nullcols_test>0].index)
nullvals_test = np.array(test[ncols_test])

#view null data
nulldata_test = pd.isnull(nullvals_test)

#fill null data with mean value
for col in ncols_test:
    mean_value=test[col].mean()
    test[col].fillna(value=mean_value, inplace=True)

#run final test
X_test = test[testcols]
Y_test = test['SalePrice']
X_test = X_test.drop(['SalePrice'], axis=1)
lr = linear_model.LinearRegression()
testmodel = lr.fit(X_test, Y_test)
testpredictions = testmodel.predict(X_test)
print(f'R-squared is: {testmodel.score(X_test,Y_test)}')
sns.histplot(Y_test - testpredictions, kde=True, color='purple', label='Final')
plt.title('Complete Model, Verification Data')
plt.xlabel('Sale Price Variation')