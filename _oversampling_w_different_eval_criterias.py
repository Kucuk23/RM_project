import random
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import math
import numpy as np
from deap import base, creator, tools, algorithms

#------------------
# Veri setini yükle
#------------------
data_2022 = pd.read_csv("2022_input_space.csv")
data_2021 = pd.read_csv("2021_input_space.csv")
data_2020 = pd.read_csv("2020_input_space.csv")
data_2017 = pd.read_csv("2017_input_space.csv")
data_2016 = pd.read_csv("2016_input_space.csv")
data_2015 = pd.read_csv("2015_input_space.csv")
data = pd.concat([data_2022, data_2021, data_2020, data_2017, data_2016, data_2015], ignore_index=True)

data['Timestamp'] = pd.to_datetime(data['Timestamp'])
data['Month'] = data['Timestamp'].dt.month
data['Hour'] = data['Timestamp'].dt.hour
data.set_index('Timestamp', inplace=True)
y_ = data["llj_level"]
X_ = data[["u10_value", "diff_data", "LEG_H062_Wd", "Month", "Hour"]]

#-------------------------------
# prediction için zaman kaydırma
#-------------------------------
forecast_interval = 1
future_y = y_.shift(0)
now_x = X_
X__ = now_x[:-forecast_interval]
y__ = future_y[1:]
y__ = y__.apply(lambda x: 0 if x <= 0 else 1)

# Reset indices of X__ and y__ before boolean indexing
X__.reset_index(drop=True, inplace=True)
y__.reset_index(drop=True, inplace=True)

# Veriyi ölçeklendirme
scaler = StandardScaler()
X__ = scaler.fit_transform(X__)

# Veri setini eğitim ve test olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(X__, y__, test_size=0.2, random_state=42)

# SMOTE ile veri çoğaltma
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# DEAP ayarları
def create_toolbox(weights):
    creator.create("FitnessMulti", base.Fitness, weights=weights)  # SEDI'yi maksimize, fa'yı minimize
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    toolbox.register("attr_int_n_estimators", random.randint, 10, 200)
    toolbox.register("attr_int_max_depth", random.randint, 1, 20)
    toolbox.register("attr_int_min_samples_split", random.randint, 2, 20)
    toolbox.register("attr_int_min_samples_leaf", random.randint, 1, 20)

    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_int_n_estimators, toolbox.attr_int_max_depth,
                      toolbox.attr_int_min_samples_split, toolbox.attr_int_min_samples_leaf), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def g_fitness(individual):
        n_estimators, max_depth, min_samples_split, min_samples_leaf = individual

        # RandomForestClassifier'ı oluştur
        rf = RandomForestClassifier(n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    min_samples_split=min_samples_split,
                                    min_samples_leaf=min_samples_leaf,
                                    random_state=42)
        # Modeli eğit
        rf.fit(X_train_resampled, y_train_resampled)
        # Test seti üzerinde tahmin yap
        predictions = rf.predict(X_test)

        # Sonuçları dataframe'e ekle
        results_df = pd.DataFrame({'y_test': y_test, 'Prediction': predictions})

        # SEDI ve fa hesaplama
        cr = ((results_df['y_test'] == 0) & (results_df['Prediction'] == 0)).sum()
        fa = ((results_df['y_test'] == 0) & (results_df['Prediction'] == 1)).sum()
        h = ((results_df['y_test'] == 1) & (results_df['Prediction'] == 1)).sum()
        m = ((results_df['y_test'] == 1) & (results_df['Prediction'] == 0)).sum()

        H = h / (h + m) if (h + m) != 0 else 0.01
        F = fa / (cr + fa) if (cr + fa) != 0 else 0.01

        SEDI = (math.log(F) - math.log(H) - math.log(1 - F) + math.log(1 - H)) / (
                    math.log(F) + math.log(H) + math.log(1 - F) + math.log(1 - H))
        
        sensitivity = h/(h+m)
        specifity = cr / (fa+cr)

        precision = h / (h + fa)
        recall = h / (h + m)

        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        auc = (sensitivity + specifity) / 2
        g_means = math.sqrt(sensitivity*specifity)

        return SEDI, auc, f1, g_means

    toolbox.register("evaluate", g_fitness)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=[10, 1, 2, 1], up=[200, 20, 20, 20], indpb=0.2)
    toolbox.register("select", tools.selNSGA2)  # Use NSGA-II for multi-objective optimization

    return toolbox

# GA parametreleri
population_size = 30
num_generations = 25
crossover_prob = 0.7
mutation_prob = 0.2

weights_list = [(1.0, 0.1, 0.1, 0.1), (0.1, 1.0, 0.1, 0.1), (0.1, 0.1, 1.0, 0.1),(0.1, 0.1, 0.1, 1.0)]  # Ağırlık kombinasyonları (-1.0, 1.0),(-2.0, 1.0),(-3.0, 1.0),(-1.0, 2.0),

for weights in weights_list:
    toolbox = create_toolbox(weights)

    # Popülasyonu başlat
    pop = toolbox.population(n=population_size)

    # Genetik algoritmayı çalıştır
    algorithms.eaMuPlusLambda(pop, toolbox, mu=population_size, lambda_=population_size,
                              cxpb=crossover_prob, mutpb=mutation_prob, ngen=num_generations,
                              stats=None, halloffame=None, verbose=True)

    # En iyi bireyi seç
    best_individual = tools.selBest(pop, 1)[0]
    best_params = {
        'n_estimators': best_individual[0],
        'max_depth': best_individual[1],
        'min_samples_split': best_individual[2],
        'min_samples_leaf': best_individual[3]
    }
    print(f"Ağırlıklar: {weights}, En iyi parametreler:", best_params)

    # En iyi modeli eğit ve kaydet
    best_model = RandomForestClassifier(**best_params, random_state=42)
    best_model.fit(X_train_resampled, y_train_resampled)
    filename = f'2nd_trial_best_genetic_model_weights_{weights}.joblib'
    dump(best_model, filename)
    print(f"Optimize edilmiş model '{filename}' dosyasına kaydedildi.")

"""

# Modeli yükleme
# SEDI, auc, f1, g_means

loaded_model = load('2nd_trial_best_genetic_model_weights_(0.1, 0.1, 0.1, 1.0).joblib')
test_predictions = loaded_model.predict(X_test)
results_df = pd.DataFrame({'Prediction': test_predictions, 'y_test': y_test})

#-------------------------------------------------------------------------------------


# y_test'in 1'e eşit olduğu durumlarda predictions'ın da 1'e eşit olduğu durumları sayma
matching_true_alarms = len(results_df[(results_df['y_test'] == 1) & (results_df['Prediction'] == 1)])
matching_false_no_alarms = len(results_df[(results_df['y_test'] == 0) & (results_df['Prediction'] == 1)])
matching_true_no_alarm = len(results_df[(results_df['y_test'] == 0) & (results_df['Prediction'] == 0)])
matching_false_alarm = len(results_df[(results_df['y_test'] == 1) & (results_df['Prediction'] == 0)])

total = matching_true_alarms + matching_false_no_alarms + matching_true_no_alarm + matching_false_alarm

print("CSV dosyası oluşturuldu: 'predictions_vs_y_test.csv'", X_train.shape, X_test.shape)
print("hits:", matching_true_alarms )
print("false_alarms:", matching_false_no_alarms)
print("correct_rejections:", matching_true_no_alarm )
print("misses:", matching_false_alarm )

# SEDI hesaplama
cr = ((results_df['y_test'] == 0) & (results_df['Prediction'] == 0)).sum()
fa = ((results_df['y_test'] == 0) & (results_df['Prediction'] == 1)).sum()
h = ((results_df['y_test'] == 1) & (results_df['Prediction'] == 1)).sum()
m = ((results_df['y_test'] == 1) & (results_df['Prediction'] == 0)).sum()



H = h / (h + m)
F = fa / (cr + fa)

print("--------------", h,m,cr,fa)

if F == 0:
    F = 0.01
if H == 0:
    H = 0.01

SEDI = (math.log(F) - math.log(H) - math.log(1 - F) + math.log(1 - H)) / (math.log(F) + math.log(H) + math.log(1 - F) + math.log(1 - H))

print("--------------", cr, fa, h, m, H, F, SEDI)

print("SEDI:", SEDI)

"""
"""
data_ERA5 = {
    'ERA5': (matching_true_alarms, matching_false_no_alarms, matching_true_no_alarm, matching_false_alarm)
}

print("Sonuçları data_ERA5 formatında:")
print(data_ERA5)
print("SEDI:", SEDI)

"""



























