import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score
from deap import base, creator, tools, algorithms
import random
from joblib import dump, load
import numpy as np
import math

# Load the datasets
data_2022 = pd.read_csv("2022_input_space.csv")
data_2021 = pd.read_csv("2021_input_space.csv")
data_2020 = pd.read_csv("2020_input_space.csv")
data_2017 = pd.read_csv("2017_input_space.csv")
data_2016 = pd.read_csv("2016_input_space.csv")
data_2015 = pd.read_csv("2015_input_space.csv")
data = pd.concat([data_2022, data_2021, data_2020, data_2017, data_2016, data_2015], ignore_index=True)

# Preprocess the data
data['Timestamp'] = pd.to_datetime(data['Timestamp'])
data['Month'] = data['Timestamp'].dt.month
data['Hour'] = data['Timestamp'].dt.hour
data.set_index('Timestamp', inplace=True)
y_ = data["llj_indic"]
y_ = y_.apply(lambda x: 0 if x == 0 else 1)  # Convert non-zero to 1
X_ = data[["u10_value", "diff_data", "LEG_H062_Wd", "Month", "Hour"]]
# Add a new column by multiplying existing columns
X_['richardson'] = 10 * (X_['diff_data']) / (X_['u10_value'] * X_['u10_value'])

#-------------------------------
# prediction için zaman kaydırma
#-------------------------------
forecast_interval = 1
future_y = y_.shift(0)  
now_x = X_
X = now_x[:-forecast_interval]    
y = future_y[1:]

# Reset indices of X and y before boolean indexing
X.reset_index(drop=True, inplace=True)
y.reset_index(drop=True, inplace=True)

# Define the evaluation function
def evaluate_model(individual):
    n_estimators = individual[0]
    max_depth = individual[1]
    min_samples_split = individual[2]
    min_samples_leaf = individual[3]

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test_scaled, y_pred_proba)
    
    return auc,

# Set up the DEAP toolbox
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_int_n_estimators", random.randint, 10, 200)  # for n_estimators
toolbox.register("attr_int_max_depth", random.randint, 1, 20)       # for max_depth
toolbox.register("attr_int_min_samples_split", random.randint, 2, 20)  # for min_samples_split
toolbox.register("attr_int_min_samples_leaf", random.randint, 1, 20)  # for min_samples_leaf

toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_int_n_estimators, toolbox.attr_int_max_depth, 
                  toolbox.attr_int_min_samples_split, toolbox.attr_int_min_samples_leaf), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate_model)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=[10, 1, 2, 1], up=[200, 20, 20, 20], indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Ratios to downsample
# ratios =  [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]
ratios = [ 0.02]

for ratio in ratios:
    # Downsample the majority class
    X_downsampled, y_downsampled = resample(X[y == 0], y[y == 0], n_samples=int(ratio * len(X[y == 0])), random_state=42)
    X_balanced = pd.concat([X_downsampled, X[y != 0]], ignore_index=True)
    y_balanced = pd.concat([y_downsampled, y[y != 0]], ignore_index=True)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.1, random_state=42)
    X_not_downsampled = X[y == 0].drop(X_downsampled.index)
    y_not_downsampled = y[y == 0].drop(y_downsampled.index)
    X_not_used = pd.concat([X_not_downsampled, X_test], ignore_index=True)
    y_not_used = pd.concat([y_not_downsampled, y_test], ignore_index=True)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_not_used_scaled = scaler.transform(X_not_used)
    X_test_scaled = X_not_used_scaled
    y_test_scaled = y_not_used

    # Run the evolutionary algorithm
    population = toolbox.population(n=50)
    ngen = 40
    cxpb = 0.5
    mutpb = 0.2
    
    # Use a closure to pass data to the evaluation function
    toolbox.register("evaluate", lambda ind: evaluate_model(ind))
    
    result, log = algorithms.eaSimple(population, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen,
                                      stats=None, halloffame=None, verbose=True)

    # After running the evolutionary algorithm loop
    # Extract and print the best individual
    best_individual = tools.selBest(population, k=1)[0]
    print(f"Best individual for ratio {ratio} is: ", best_individual)

    # Train the best model on the entire balanced dataset
    best_model = RandomForestClassifier(
        n_estimators=best_individual[0],
        max_depth=best_individual[1],
        min_samples_split=best_individual[2],
        min_samples_leaf=best_individual[3],
        random_state=42,
        n_jobs=-1
    )
    best_model.fit(X_train_scaled, y_train)

    # Predict using the best model on the test set
    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]

    # Calculate AUC score for the best model
    auc = roc_auc_score(y_test_scaled, y_pred_proba)
    print(f"With AUC Score: {auc}")

    # Save the best model
    dump(best_model, f"auc_best_random_forest_model_ratio_{ratio}.joblib") 

"""
# Modeli yükleme
loaded_model = load('auc_best_random_forest_model_ratio_0.02.joblib')


#-------------------------------------------------------------------------------------
# Test veri seti üzerinde tahminler yapma
# test_predictions = loaded_model.predict(X_test_scaled)
# results_df = pd.DataFrame({'Prediction': test_predictions, 'y_test': y_test_binary})

# test_predictions = loaded_model.predict(X_wide_space)
# results_df = pd.DataFrame({'Prediction': test_predictions, 'y_test': y_wide_space})

test_predictions = loaded_model.predict(X_test_scaled)
results_df = pd.DataFrame({'Prediction': test_predictions, 'y_test': y_test_scaled})

#-------------------------------------------------------------------------------------


# y_test'in 1'e eşit olduğu durumlarda predictions'ın da 1'e eşit olduğu durumları sayma
matching_true_alarms = len(results_df[(results_df['y_test'] == 1) & (results_df['Prediction'] == 1)])
matching_false_no_alarms = len(results_df[(results_df['y_test'] == 0) & (results_df['Prediction'] == 1)])
matching_true_no_alarm = len(results_df[(results_df['y_test'] == 0) & (results_df['Prediction'] == 0)])
matching_false_alarm = len(results_df[(results_df['y_test'] == 1) & (results_df['Prediction'] == 0)])

# Sonuçları CSV dosyasına yazma
results_df.to_csv('predictions_vs_y_test.csv', index=False)

print("CSV dosyası oluşturuldu: 'predictions_vs_y_test.csv'", X_train.shape, X_test.shape)
print("matching_catch_alarms:", matching_true_alarms)
print("matching_false_no_alarms:", matching_false_no_alarms)
print("matching_catch_no_alarm", matching_true_no_alarm)
print("matching_false_alarm", matching_false_alarm)

# SEDI hesaplama
cr = ((results_df['y_test'] == 0) & (results_df['Prediction'] == 0)).sum()
fa = ((results_df['y_test'] == 0) & (results_df['Prediction'] == 1)).sum()
h = ((results_df['y_test'] == 1) & (results_df['Prediction'] == 1)).sum()
m = ((results_df['y_test'] == 1) & (results_df['Prediction'] == 0)).sum()

H = h / (h + m)
F = fa / (cr + fa)

if F == 0:
    F = 0.01
if H == 0:
    H = 0.01

SEDI = (math.log(F) - math.log(H) - math.log(1 - F) + math.log(1 - H)) / (math.log(F) + math.log(H) + math.log(1 - F) + math.log(1 - H))

print("SEDI:", SEDI)

"""










