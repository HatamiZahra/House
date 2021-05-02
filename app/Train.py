{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Feature Engineering"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train Dataset \n",
      " Variables:\t79\n",
      " Inputs:\t2917\n"
     ]
    }
   ],
   "source": [
    "# Checking the dimensions of the Dataframe\n",
    "print(\"Train Dataset \\n Variables:\\t{}\\n Inputs:\\t{}\".format(preprocessing_Merged_df.shape[1], preprocessing_Merged_df.shape[0]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Total House Area\n",
    "preprocessing_Merged_df['TotalHouseArea'] =  (preprocessing_Merged_df['1stFlrSF'] + preprocessing_Merged_df['2ndFlrSF'] + preprocessing_Merged_df['GrLivArea'] +\n",
    "                                              preprocessing_Merged_df['TotalBsmtSF'] - preprocessing_Merged_df['LowQualFinSF'] - preprocessing_Merged_df['BsmtUnfSF'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Total Bathrooms\n",
    "preprocessing_Merged_df['TotalBathrooms'] = (preprocessing_Merged_df['BsmtFullBath'] + (0.5 * preprocessing_Merged_df['BsmtHalfBath']) + preprocessing_Merged_df['FullBath'] + ( 0.5 * preprocessing_Merged_df['HalfBath']))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Total Porch Area\n",
    "preprocessing_Merged_df['TotalPorchArea'] =  (preprocessing_Merged_df['OpenPorchSF'] + preprocessing_Merged_df['EnclosedPorch'] + preprocessing_Merged_df['3SsnPorch'] + preprocessing_Merged_df['ScreenPorch'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Swimming Pool\n",
    "\n",
    "preprocessing_Merged_df['SwimmingPool'] = preprocessing_Merged_df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Year Built Plus Remold\n",
    "min_max_scaler = preprocessing.MinMaxScaler()\n",
    "\n",
    "scaled_YearBuilt = preprocessing_Merged_df[['YearBuilt']].values.astype(float)\n",
    "scaled_YearRemodAdd = preprocessing_Merged_df[['YearRemodAdd']].values.astype(float)\n",
    "\n",
    "scaled_YearBuilt = min_max_scaler.fit_transform(scaled_YearBuilt)\n",
    "scaled_YearRemodAdd = min_max_scaler.fit_transform(scaled_YearRemodAdd)\n",
    "\n",
    "preprocessing_Merged_df['YearBuiltPlusRemold'] = (scaled_YearBuilt + scaled_YearRemodAdd)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Basement\n",
    "\n",
    "preprocessing_Merged_df['Basement'] = preprocessing_Merged_df['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Dropping the features we used to build our new features\n",
    "\n",
    "preprocessing_Merged_df = preprocessing_Merged_df.drop(['1stFlrSF', '2ndFlrSF', 'GrLivArea', 'TotalBsmtSF', 'LowQualFinSF', 'BsmtUnfSF',\n",
    "                                                        'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath',\n",
    "                                                        'OpenPorchSF', 'EnclosedPorch', '3SsnPorch','ScreenPorch',\n",
    "                                                        'PoolArea','PoolQC','YearBuilt','YearRemodAdd',\n",
    "                                                        'BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtFinType2','BsmtFinSF2'], axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Dropping GarageArea\n",
    "preprocessing_Merged_df = preprocessing_Merged_df.drop(['GarageArea'], axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Dropping MoSold\n",
    "preprocessing_Merged_df = preprocessing_Merged_df.drop(['MoSold','Street'], axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [],
   "source": [
    "dummies_preprocessing_Merged_df = preprocessing_Merged_df.copy()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Creating Dummy Variables\n",
    "\n",
    "dummies_preprocessing_Merged_df = pd.get_dummies(preprocessing_Merged_df, drop_first=True).reset_index(drop=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>MSSubClass</th>\n",
       "      <th>LotFrontage</th>\n",
       "      <th>LotArea</th>\n",
       "      <th>OverallQual</th>\n",
       "      <th>OverallCond</th>\n",
       "      <th>MasVnrArea</th>\n",
       "      <th>BedroomAbvGr</th>\n",
       "      <th>KitchenAbvGr</th>\n",
       "      <th>TotRmsAbvGrd</th>\n",
       "      <th>Fireplaces</th>\n",
       "      <th>...</th>\n",
       "      <th>SaleType_ConLI</th>\n",
       "      <th>SaleType_ConLw</th>\n",
       "      <th>SaleType_New</th>\n",
       "      <th>SaleType_Oth</th>\n",
       "      <th>SaleType_WD</th>\n",
       "      <th>SaleCondition_AdjLand</th>\n",
       "      <th>SaleCondition_Alloca</th>\n",
       "      <th>SaleCondition_Family</th>\n",
       "      <th>SaleCondition_Normal</th>\n",
       "      <th>SaleCondition_Partial</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>6.499672</td>\n",
       "      <td>18.780783</td>\n",
       "      <td>13.833054</td>\n",
       "      <td>7</td>\n",
       "      <td>3.991517</td>\n",
       "      <td>19.433176</td>\n",
       "      <td>3</td>\n",
       "      <td>0.750957</td>\n",
       "      <td>2.261968</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>4.249693</td>\n",
       "      <td>21.443059</td>\n",
       "      <td>14.117918</td>\n",
       "      <td>6</td>\n",
       "      <td>6.000033</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>3</td>\n",
       "      <td>0.750957</td>\n",
       "      <td>1.996577</td>\n",
       "      <td>0.903334</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>6.499672</td>\n",
       "      <td>19.331291</td>\n",
       "      <td>14.476512</td>\n",
       "      <td>7</td>\n",
       "      <td>3.991517</td>\n",
       "      <td>17.768841</td>\n",
       "      <td>3</td>\n",
       "      <td>0.750957</td>\n",
       "      <td>1.996577</td>\n",
       "      <td>0.903334</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>6.862540</td>\n",
       "      <td>17.840335</td>\n",
       "      <td>14.106197</td>\n",
       "      <td>7</td>\n",
       "      <td>3.991517</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>3</td>\n",
       "      <td>0.750957</td>\n",
       "      <td>2.137369</td>\n",
       "      <td>0.903334</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>6.499672</td>\n",
       "      <td>22.118469</td>\n",
       "      <td>15.022008</td>\n",
       "      <td>8</td>\n",
       "      <td>3.991517</td>\n",
       "      <td>25.404165</td>\n",
       "      <td>4</td>\n",
       "      <td>0.750957</td>\n",
       "      <td>2.373753</td>\n",
       "      <td>0.903334</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2912</th>\n",
       "      <td>9.030083</td>\n",
       "      <td>8.936631</td>\n",
       "      <td>10.765922</td>\n",
       "      <td>4</td>\n",
       "      <td>5.348041</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>3</td>\n",
       "      <td>0.750957</td>\n",
       "      <td>1.834659</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2913</th>\n",
       "      <td>9.030083</td>\n",
       "      <td>8.936631</td>\n",
       "      <td>10.723223</td>\n",
       "      <td>4</td>\n",
       "      <td>3.991517</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>3</td>\n",
       "      <td>0.750957</td>\n",
       "      <td>1.996577</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2914</th>\n",
       "      <td>4.249693</td>\n",
       "      <td>33.173868</td>\n",
       "      <td>15.820339</td>\n",
       "      <td>5</td>\n",
       "      <td>5.348041</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>4</td>\n",
       "      <td>0.750957</td>\n",
       "      <td>2.137369</td>\n",
       "      <td>0.903334</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2915</th>\n",
       "      <td>7.337374</td>\n",
       "      <td>18.220106</td>\n",
       "      <td>14.307159</td>\n",
       "      <td>5</td>\n",
       "      <td>3.991517</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>3</td>\n",
       "      <td>0.750957</td>\n",
       "      <td>1.996577</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2916</th>\n",
       "      <td>6.499672</td>\n",
       "      <td>20.404197</td>\n",
       "      <td>14.124224</td>\n",
       "      <td>7</td>\n",
       "      <td>3.991517</td>\n",
       "      <td>13.690163</td>\n",
       "      <td>3</td>\n",
       "      <td>0.750957</td>\n",
       "      <td>2.373753</td>\n",
       "      <td>0.903334</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>2917 rows × 215 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "      MSSubClass  LotFrontage    LotArea  OverallQual  OverallCond  \\\n",
       "0       6.499672    18.780783  13.833054            7     3.991517   \n",
       "1       4.249693    21.443059  14.117918            6     6.000033   \n",
       "2       6.499672    19.331291  14.476512            7     3.991517   \n",
       "3       6.862540    17.840335  14.106197            7     3.991517   \n",
       "4       6.499672    22.118469  15.022008            8     3.991517   \n",
       "...          ...          ...        ...          ...          ...   \n",
       "2912    9.030083     8.936631  10.765922            4     5.348041   \n",
       "2913    9.030083     8.936631  10.723223            4     3.991517   \n",
       "2914    4.249693    33.173868  15.820339            5     5.348041   \n",
       "2915    7.337374    18.220106  14.307159            5     3.991517   \n",
       "2916    6.499672    20.404197  14.124224            7     3.991517   \n",
       "\n",
       "      MasVnrArea  BedroomAbvGr  KitchenAbvGr  TotRmsAbvGrd  Fireplaces  ...  \\\n",
       "0      19.433176             3      0.750957      2.261968    0.000000  ...   \n",
       "1       0.000000             3      0.750957      1.996577    0.903334  ...   \n",
       "2      17.768841             3      0.750957      1.996577    0.903334  ...   \n",
       "3       0.000000             3      0.750957      2.137369    0.903334  ...   \n",
       "4      25.404165             4      0.750957      2.373753    0.903334  ...   \n",
       "...          ...           ...           ...           ...         ...  ...   \n",
       "2912    0.000000             3      0.750957      1.834659    0.000000  ...   \n",
       "2913    0.000000             3      0.750957      1.996577    0.000000  ...   \n",
       "2914    0.000000             4      0.750957      2.137369    0.903334  ...   \n",
       "2915    0.000000             3      0.750957      1.996577    0.000000  ...   \n",
       "2916   13.690163             3      0.750957      2.373753    0.903334  ...   \n",
       "\n",
       "      SaleType_ConLI  SaleType_ConLw  SaleType_New  SaleType_Oth  SaleType_WD  \\\n",
       "0                  0               0             0             0            1   \n",
       "1                  0               0             0             0            1   \n",
       "2                  0               0             0             0            1   \n",
       "3                  0               0             0             0            1   \n",
       "4                  0               0             0             0            1   \n",
       "...              ...             ...           ...           ...          ...   \n",
       "2912               0               0             0             0            1   \n",
       "2913               0               0             0             0            1   \n",
       "2914               0               0             0             0            1   \n",
       "2915               0               0             0             0            1   \n",
       "2916               0               0             0             0            1   \n",
       "\n",
       "      SaleCondition_AdjLand  SaleCondition_Alloca  SaleCondition_Family  \\\n",
       "0                         0                     0                     0   \n",
       "1                         0                     0                     0   \n",
       "2                         0                     0                     0   \n",
       "3                         0                     0                     0   \n",
       "4                         0                     0                     0   \n",
       "...                     ...                   ...                   ...   \n",
       "2912                      0                     0                     0   \n",
       "2913                      0                     0                     0   \n",
       "2914                      0                     0                     0   \n",
       "2915                      0                     0                     0   \n",
       "2916                      0                     0                     0   \n",
       "\n",
       "      SaleCondition_Normal  SaleCondition_Partial  \n",
       "0                        1                      0  \n",
       "1                        1                      0  \n",
       "2                        1                      0  \n",
       "3                        0                      0  \n",
       "4                        1                      0  \n",
       "...                    ...                    ...  \n",
       "2912                     1                      0  \n",
       "2913                     0                      0  \n",
       "2914                     0                      0  \n",
       "2915                     1                      0  \n",
       "2916                     1                      0  \n",
       "\n",
       "[2917 rows x 215 columns]"
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dummies_preprocessing_Merged_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Normalizing GarageYearBlt Values\n",
    "scaled_GarageYearBlt = dummies_preprocessing_Merged_df[['GarageYrBlt']].values.astype(float)\n",
    "scaled_GarageYearBlt = min_max_scaler.fit_transform(scaled_GarageYearBlt)\n",
    "dummies_preprocessing_Merged_df['GarageYrBlt'] = scaled_GarageYearBlt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Splitting the datasets using the indexes we saved\n",
    "\n",
    "preprocessed_train_df = dummies_preprocessing_Merged_df.iloc[:(train_idx-2)]\n",
    "preprocessed_test_df = dummies_preprocessing_Merged_df.iloc[(train_idx-2):]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train Dataset \n",
      " Variables:\t215\n",
      " Inputs:\t1458\n",
      "\n",
      "\n",
      "Test Dataset \n",
      " Variables:\t215\n",
      " Inputs:\t1459\n"
     ]
    }
   ],
   "source": [
    "# Checking the dimensions of the Datasets\n",
    "\n",
    "print(\"Train Dataset \\n Variables:\\t{}\\n Inputs:\\t{}\".format(preprocessed_train_df.shape[1], preprocessed_train_df.shape[0]))\n",
    "print(\"\\n\")\n",
    "print(\"Test Dataset \\n Variables:\\t{}\\n Inputs:\\t{}\".format(preprocessed_test_df.shape[1], preprocessed_test_df.shape[0]))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Modeling"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [],
   "source": [
    "class Stats:\n",
    "    \n",
    "    def __init__(self, X, y, y_hat):\n",
    "        self.data = X\n",
    "        self.target = y\n",
    "        self.predicted = y_hat\n",
    "        ## degrees of freedom population dep. variable variance\n",
    "        self._dft = X.shape[0] - 1   \n",
    "        ## degrees of freedom population error variance\n",
    "        self._dfe = X.shape[0] - X.shape[1] - 1  \n",
    "    \n",
    "    def sse(self):\n",
    "        #returns sum of squared errors (model vs actual)\n",
    "        squared_errors = (self.target - self.predicted) ** 2\n",
    "        return np.sum(squared_errors)\n",
    "        \n",
    "    def sst(self):\n",
    "        #returns total sum of squared errors (actual vs avg(actual))\n",
    "        avg_y = np.mean(self.target)\n",
    "        squared_errors = (self.target - avg_y) ** 2\n",
    "        return np.sum(squared_errors)\n",
    "    \n",
    "    def r_squared(self):\n",
    "        #returns calculated value of r^2\n",
    "        return 1 - self.sse()/self.sst()\n",
    "    \n",
    "    def adj_r_squared(self):\n",
    "        #returns calculated value of adjusted r^2\n",
    "        return 1 - (self.sse()/self._dfe) / (self.sst()/self._dft)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [],
   "source": [
    "def report_stats(stats_obj):\n",
    "    #returns report of statistics for the model predicted values\n",
    "    print(\"Key Four Statistics for this Model \\n\")\n",
    "    items = ( ('sse:', stats_obj.sse()), ('sst:', stats_obj.sst()), \n",
    "             ('r^2:', stats_obj.r_squared()), ('adj_r^2:', stats_obj.adj_r_squared()) )\n",
    "    for item in items:\n",
    "        print('{0:8} {1:.4f}'.format(item[0], item[1]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.metrics import mean_squared_error, mean_absolute_error\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.linear_model import Ridge\n",
    "from sklearn.linear_model import Lasso\n",
    "from sklearn.metrics import mean_squared_log_error"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Splitting our train test into two: train e test sets\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    preprocessed_train_df, target, test_size=0.1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Ridge Regression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [],
   "source": [
    "import decimal\n",
    "def drange(x, y, jump):\n",
    "  while x < y:\n",
    "    yield float(x)\n",
    "    x += decimal.Decimal(jump)\n",
    "\n",
    "alpha_ridge = list(drange(0, 100, '0.1'))\n",
    "\n",
    "# Create variables to store SSE and MSE for all the alpha values\n",
    "alpha_ridge_rss = {}\n",
    "alpha_ridge_mse = {}\n",
    "\n",
    "# Regressions for all the values of alpha\n",
    "for i in alpha_ridge:\n",
    "    ## Assigin each model. \n",
    "    ridge = Ridge(alpha= i, normalize=True)\n",
    "    ## fit the model. \n",
    "    ridge.fit(X_train, y_train)\n",
    "    ## Predicting the target value\n",
    "    y_hat_ridge = ridge.predict(X_test)\n",
    "\n",
    "    mse = mean_squared_error(y_test, y_hat_ridge)\n",
    "    rss = sum((y_hat_ridge-y_test)**2)\n",
    "    alpha_ridge_mse[i] = mse\n",
    "    alpha_ridge_rss[i] = rss"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Alpha with the minimum MSE , minimun MSE:\t (0.1, 0.01525623131242704)\n",
      "\n",
      "Alpha with the minimum RSS , minimun RSS:\t (0.1, 2.227409771614347)\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# Finding out the best alpha for our model\n",
    "min_mse_ridge = min(alpha_ridge_mse.items(), key=lambda x: x[1])\n",
    "min_rss_ridge = min(alpha_ridge_rss.items(), key=lambda x: x[1])\n",
    "\n",
    "print(\"Alpha with the minimum MSE , minimun MSE:\\t {}\\n\".format(min_mse_ridge))\n",
    "print(\"Alpha with the minimum RSS , minimun RSS:\\t {}\\n\".format(min_rss_ridge))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### We can see here that alpha = 0.1 is the best option for our regression model."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Key Four Statistics for this Model \n",
      "\n",
      "sse:     2.2274\n",
      "sst:     22.2117\n",
      "r^2:     0.8997\n",
      "adj_r^2: 1.2077\n"
     ]
    }
   ],
   "source": [
    "#Running again the rigde regression only with the best alpha value\n",
    "best_alpha_ridge = 0.1\n",
    "ridge = Ridge(alpha= best_alpha_ridge, normalize=True)\n",
    "ridge.fit(X_train, y_train)\n",
    "y_hat_ridge = ridge.predict(X_test)\n",
    "\n",
    "# Four Key Statistics for the Multiple Linear Regression\n",
    "stats_linear_ridge = Stats(X_test,y_test,y_hat_ridge)\n",
    "report_stats(stats_linear_ridge)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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9pgbUERE7I7KHxERETt1+PBhvvzyS9q2bUv58uUf3h4bC61awYULsGoVFCxoYkgRsTmLxVjfKyICjh797+1VqvTCzc2DH38cRmzsffMCioiI3VH5IyIiYodiYmIICgoiX7589PufxXz69ze2ep47FypXNimgiJiqfHnIlQs2boTEROM2J6eMvPfeSG7fPs+ePdPMDSgiInZF5Y+IiIgdmj17NqdPnyYkJARXV9eHty9YANOmwSefQIcOJgYUEVM5OoK3N/z1F/z5539vf+mlSrz2WksOHfqCS5eOmRdQRETsisofERERO/PHH38wb948fHx8qF69+sPbd++G7t2N//CNG2diQBGxC2+9Zaz79cMPj95erdoAMmfOxebNQcTHx5oTTkRE7IrKHxERETsSFxdHYGAg2bJlIyAg4OHt585B06bg4QFLl4KTk4khRcQuODtDzZrGzn/nz//39owZs1GrVjDXrv3BgQMLzAsoIiJ2Q+WPiIiIHQkLC+P48eMMHToUd3d3AKKioEkTuH8f1q2DHDlMDikidqNGDXBxeXz2T7FitXj55brs2zeT69dPmxNORETshsofERERO3H27FmmTp1KzZo1qVevHmAs5NqpE/z6KyxZAl5eJocUEbvi6grVq8OBA3D16qPH3n13KE5Omfnxx2ASExPMCSgiInZB5Y+IiIgdSExMJDg4GCcnJ4KDg7FYLACMHg3Ll8OYMdCggckhRcQuvfeesQD0pk2P3u7qmosaNQZz8eIhjhxZYk44ERGxCyp/RERE7MA333zD/v378ff3J2/evACsXg1BQfD+++Dvb3JAEbFbbm5QpYqxKPzNm48ee/XVxhQuXJWff57EnTsXzQkoIiKmU/kjIiJissuXLzN+/HjefPNNWrZsCcCxY+DnBxUrwrx58J+JQCIiT1S3LiQkPD77x2KxUKvWcBITE/nppxASExPNCSgiIqZS+SMiImKixMREhg8fTlxcHMOHD8disXDtGjRqBNmywZo1kCmT2SlFxN7lygVvvgk7d8KdO48ey569IFWr9uHvv3eydet6cwKKiIipVP6IiIiYaMOGDWzdupXevXtTuHBhYmOhRQuIiDCKnwIFzE4oIqlFvXoQFwebNz9+7PXX25EvXxlmz/6UGzdu2D6ciIiYSuWPiIiISW7evMno0aMpXbo0fn5+APTuDdu3w/z5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\n",
      "text/plain": [
       "<Figure size 1440x576 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Residuals\n",
    "ridge_residuals = y_test - y_hat_ridge\n",
    "\n",
    "plt.figure(figsize=(20,8));plt.title(' Distribution of the Residuals ')\n",
    "sns.distplot(ridge_residuals, color='blue', fit=stats.norm, norm_hist=True, axlabel='Residuals')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Root-Mean-Squared-Error (RMSE):\t 0.009580025522862924\n",
      "\n"
     ]
    }
   ],
   "source": [
    "#Root mean squared error\n",
    "rmse_ridge = np.sqrt(mean_squared_log_error( y_test, y_hat_ridge ))\n",
    "\n",
    "print(\"Root-Mean-Squared-Error (RMSE):\\t {}\\n\".format(rmse_ridge))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##  Evaluating"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_predicted = ridge.predict(preprocessed_test_df)\n",
    "\n",
    "submission_df = pd.DataFrame ({\"Id\": houseId,\"SalePrice\": y_predicted})\n",
    "\n",
    "submission_df.to_csv('./submission_houseprices.csv', index=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
