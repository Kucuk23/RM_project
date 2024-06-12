import cfgrib
import  numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dosya_yolu = 'era5_variable_3.grib'
all_variables = []
time = None

with cfgrib.open_dataset(dosya_yolu) as ds:
    print(ds.variables)

datasets = cfgrib.open_datasets(dosya_yolu)

all_variables = []
for dataset in datasets:
    all_variables.append(dataset.data_vars)
    if time is None:
        time = pd.to_datetime(dataset["time"].values)

for dataset in datasets:
    dataset.close()

#-----------------------------------2022----------------------------------
llj_csv_2022 = pd.read_csv("hourly__indexed_datas_2022.csv")
llj_csv_2022['Timestamp'] = pd.to_datetime(llj_csv_2022['Timestamp(UTC)'])
#-----------------------------------2021----------------------------------
llj_csv_2021 = pd.read_csv("hourly__indexed_datas_2021.csv")
llj_csv_2021['Timestamp'] = pd.to_datetime(llj_csv_2021['Timestamp(UTC)'])
#-----------------------------------2020----------------------------------
llj_csv_2020 = pd.read_csv("hourly__indexed_datas_2020.csv")
llj_csv_2020['Timestamp'] = pd.to_datetime(llj_csv_2020['Timestamp(UTC)'])
#-----------------------------------2017----------------------------------
llj_csv_2017 = pd.read_csv("hourly__indexed_datas_2017.csv")
llj_csv_2017['Timestamp'] = pd.to_datetime(llj_csv_2017['Timestamp(UTC)'])
#-----------------------------------2016----------------------------------
llj_csv_2016 = pd.read_csv("hourly__indexed_datas_2016.csv")
llj_csv_2016['Timestamp'] = pd.to_datetime(llj_csv_2016['Timestamp(UTC)'])
#-----------------------------------2015----------------------------------
llj_csv_2015 = pd.read_csv("hourly__indexed_datas_2015.csv")
llj_csv_2015['Timestamp'] = pd.to_datetime(llj_csv_2015['Timestamp(UTC)'])

# merging datas
#2015de SIKINTI VAR GIBI ONU CIKARTTIM BI OYLE BAKCAM
merged_data = pd.concat([llj_csv_2022, llj_csv_2021, llj_csv_2020, llj_csv_2017, llj_csv_2016], ignore_index=True)
# sorting
merged_data = merged_data.sort_values(by='Timestamp')


for dataset in all_variables:
    if "u10" in dataset:
        u10_data = dataset["u10"].values[:, 0, 2]

llj_u10 = [] 
for date in merged_data['Timestamp']:
    idx = np.where(time == date)[0][0]
    u10_llj_value = u10_data[idx]
    llj_u10.append((u10_llj_value))


print("look.................\n", u10_data, "\n", len(u10_data),"\n", llj_u10,"\n", len(llj_u10))


# Tek bir figürde iki histogram çizimi
fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # 1 satır, 2 sütunluk bir figür oluştur

# İlk box plot
axs[0].boxplot(u10_data)
axs[0].set_title('All Cases')  # Grafik başlığı
axs[0].set_ylabel('Values')  # Y ekseninin etiketi
axs[0].set_ylim(-20, 20)  # Y ekseninin aralığını belirle

# İkinci box plot
axs[1].boxplot(llj_u10)
axs[1].set_title('LLJ=1 Cases')
axs[1].set_ylabel('Values')
axs[1].set_ylim(-20, 20)    # Y ekseninin aralığını belirle

fig.suptitle('V10 @10m (m/s)', fontsize=14, fontweight='bold')
plt.tight_layout()  # Grafiklerin sıkışmasını önlemek için
plt.show()


























