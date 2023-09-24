from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score as f1_score_, roc_auc_score as roc_auc_score_

from required_functions import *
pd.set_option('display.max_columns', None)
pd.set_option("display.width", 500)

df_ = pd.read_csv("datasets/high_diamond_ranked_10min.csv")

# Explonatory Data Analysis
# DataFrame where "red" and "blue" are present. These lists contain the column names containing "red" and "blue," respectively.
check_df(df_)
red, blue = [col for col in df_.columns if "red" in col], [col for col in df_.columns if "blue" in col]

cat_cols, num_cols, cat_but_car = grab_col_names(df_,cat_th=3)

"""cat_cols (categorical_variables): 7 - ['blueWins', 'blueFirstBlood', 'blueDragons', 'blueHeralds', 'redFirstBlood', 'redDragons', 'redHeralds']
num_cols (numerical_variables): 32 - ['blueWardsPlaced', 'blueWardsDestroyed', 'blueKills', 'blueDeaths', 'blueAssists', 'blueEliteMonsters', 'blueTowersDestroyed', 'blueTotalGold', 'blueAvgLevel', 'blueTotalExperience', 'blueTotalMinionsKilled', 'blueTotalJungleMinionsKilled', 'blueGoldDiff', 'blueExperienceDiff', 'blueCSPerMin', 'blueGoldPerMin', 'redWardsPlaced', 'redWardsDestroyed', 'redKills', 'redDeaths', 'redAssists', 'redEliteMonsters', 'redTowersDestroyed', 'redTotalGold', 'redAvgLevel', 'redTotalExperience', 'redTotalMinionsKilled', 'redTotalJungleMinionsKilled', 'redGoldDiff', 'redExperienceDiff', 'redCSPerMin', 'redGoldPerMin']
"""

cat_cols = ['blueWins','redFirstBlood', 'redDragons', 'redHeralds']


gold_cols = ["blueTotalGold","blueGoldDiff", "blueKills", "blueWardsDestroyed", "blueAssists",
                     "blueTotalMinionsKilled", "blueTotalJungleMinionsKilled", "blueCSPerMin", "blueGoldPerMin",
                     "redTowersDestroyed"]
corr_check=["blueWins","blueTotalGold",'blueWardsPlaced',"blueWardsDestroyed","blueTotalMinionsKilled","redTowersDestroyed",'blueAvgLevel',
            'blueTotalExperience']

# Summary
for col in cat_cols:
    cat_summary(df_,col,True)

for col in num_cols:
    num_summary(df_, col, True)

for col in cat_cols:
    target_summary_with_cat(df_, "blueWins", col, True)

for col in num_cols:
    target_summary_with_num(df_, "blueWins",col, True)

# Correlation Summary
correlation_matrix(df_, gold_cols)

correlation_matrix(df_, corr_check)

correlation_matrix(df_, cat_cols)

high_correlated_cols(df_)

#Data Preprocessing
df =df_.drop(["gameId",'blueEliteMonsters','blueHeralds','blueWardsPlaced', 'blueWardsDestroyed','redWardsPlaced',
                          'redWardsDestroyed','blueTotalExperience','blueCSPerMin','blueGoldPerMin', 'redCSPerMin',
                          'redGoldPerMin','redExperienceDiff','blueExperienceDiff','blueDeaths', 'redDeaths',"redFirstBlood",
                        "redGoldDiff","redTotalExperience","blueGoldDiff",'redDragons','blueTotalJungleMinionsKilled',
              'redEliteMonsters','redTotalJungleMinionsKilled','redHeralds'],axis=1)

high_correlated_cols(df)

# Outliers
cat_cols_df,num_cols_df, cat_but_car  = grab_col_names(df,cat_th=3)

for col in num_cols_df:
    print(col, check_outlier(df, col, .25, .75))

for col in num_cols_df:
    show_outliers1(df, col,0.25, 0.75)


replace_with_thresholds(df, 'blueTotalGold', q1=0.05, q3=0.90, low_threshold=True)
replace_with_thresholds(df, 'blueKills', q1=0.0, q3=0.80, low_threshold=False)
replace_with_thresholds(df, 'blueAssists', q1=0.0, q3=0.8, low_threshold=False)

replace_with_thresholds(df, 'blueAvgLevel', q1=0.00, q3=0.95, low_threshold=False)
replace_with_thresholds(df, 'blueTotalMinionsKilled', q1=0.30, q3=0.95, low_threshold=True)


replace_with_thresholds(df, 'redKills', q1=0.0, q3=0.80, low_threshold=True)
replace_with_thresholds(df, 'redAssists', q1=0.0, q3=0.80, low_threshold=False)

replace_with_thresholds(df, 'redAvgLevel', q1=0.00, q3=0.95, low_threshold=False)
replace_with_thresholds(df, 'redTotalMinionsKilled', q1=0.30, q3=0.95, low_threshold=True)
replace_with_thresholds(df, 'redTotalGold', q1=0.05, q3=0.90, low_threshold=True)


warnings.simplefilter(action='ignore', category=Warning)

y = df["blueWins"]
X = df.drop("blueWins",axis=1)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.20)

base_models(X, y, cv=3) # LogisticRegression() 8093
best_models = hyperparameter_optimization(X, y, 3)
dff_voting = voting_model(best_models, X, y,cv=3)


########################### Model With Preprocessor

preprocessor = ColumnTransformer(
    transformers=[
        ('num', RobustScaler(), num_cols_df),
    ],
    remainder='passthrough'
)
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor',VotingClassifier(voting="soft",estimators=list(best_models.items())))
]).fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test,y_pred) # 0.2768218623481781
acc_score = accuracy_score(y_test, y_pred) # 0.7231781376518218
f1_score = f1_score_(y_test, y_pred) # 0.7241553202218859
roc_auc_score = roc_auc_score_(y_test, y_pred) # 0.7231787312817393

random_user1 = X_test.sample(100, random_state=50)
pipeline_samp1 = pipeline.predict(random_user1)
"""
array([1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0,
       1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0,
       1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0,
       1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1,
       0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1], dtype=int64)
"""

joblib.dump(pipeline, "lol_prediction.joblib")

model = joblib.load("lol_prediction.joblib")

model.predict(random_user1) #0






