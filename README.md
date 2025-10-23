# Respiratory infant mortality rate in Mexico by month of birth

Daniel E. Noyola, Héctor A. Milán, Juan C. Cuevas-Tello

Universidad Autónoma de San Luis Potosí


```python
%pip install pandas matplotlib seaborn numpy statsmodels
```

    Requirement already satisfied: pandas in ./.venv/lib/python3.12/site-packages (2.3.3)
    Requirement already satisfied: matplotlib in ./.venv/lib/python3.12/site-packages (3.10.7)
    Requirement already satisfied: seaborn in ./.venv/lib/python3.12/site-packages (0.13.2)
    Requirement already satisfied: numpy in ./.venv/lib/python3.12/site-packages (2.3.4)
    Requirement already satisfied: statsmodels in ./.venv/lib/python3.12/site-packages (0.14.5)
    Requirement already satisfied: python-dateutil>=2.8.2 in ./.venv/lib/python3.12/site-packages (from pandas) (2.9.0.post0)
    Requirement already satisfied: pytz>=2020.1 in ./.venv/lib/python3.12/site-packages (from pandas) (2025.2)
    Requirement already satisfied: tzdata>=2022.7 in ./.venv/lib/python3.12/site-packages (from pandas) (2025.2)
    Requirement already satisfied: contourpy>=1.0.1 in ./.venv/lib/python3.12/site-packages (from matplotlib) (1.3.3)
    Requirement already satisfied: cycler>=0.10 in ./.venv/lib/python3.12/site-packages (from matplotlib) (0.12.1)
    Requirement already satisfied: fonttools>=4.22.0 in ./.venv/lib/python3.12/site-packages (from matplotlib) (4.60.1)
    Requirement already satisfied: kiwisolver>=1.3.1 in ./.venv/lib/python3.12/site-packages (from matplotlib) (1.4.9)
    Requirement already satisfied: packaging>=20.0 in ./.venv/lib/python3.12/site-packages (from matplotlib) (25.0)
    Requirement already satisfied: pillow>=8 in ./.venv/lib/python3.12/site-packages (from matplotlib) (12.0.0)
    Requirement already satisfied: pyparsing>=3 in ./.venv/lib/python3.12/site-packages (from matplotlib) (3.2.5)
    Requirement already satisfied: scipy!=1.9.2,>=1.8 in ./.venv/lib/python3.12/site-packages (from statsmodels) (1.16.2)
    Requirement already satisfied: patsy>=0.5.6 in ./.venv/lib/python3.12/site-packages (from statsmodels) (1.0.2)
    Requirement already satisfied: six>=1.5 in ./.venv/lib/python3.12/site-packages (from python-dateutil>=2.8.2->pandas) (1.17.0)
    Note: you may need to restart the kernel to use updated packages.



```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from itertools import product
```

**Change this variable to disable Gray Scale plotting**

```
grayscale = False
```


```python

grayscale = True
```


```python
# Helper variables
datasets_folder = 'Datasets'
plots_folder = 'Plots'
image_format = 'eps'
```


```python
month_mapping = {
    1: 'January', 2: 'February', 3: 'March', 4: 'April',
    5: 'May', 6: 'June', 7: 'July', 8: 'August',
    9: 'September', 10: 'October', 11: 'November', 12: 'December'
}

# Define a custom order for sorting, starting from July
custom_order = ['July', 'August', 'September', 'October', 'November', 'December',
                'January', 'February', 'March', 'April', 'May', 'June']

# Define a custom order for sorting, starting from April
occurence_custom_order = ['April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December',
                'January', 'February', 'March']
```

## Registered Deaths
The following dataset contains statistics on registered deaths from the [Subsystem of Demographic and Social Information](https://en.www.inegi.org.mx/programas/edr/#open_data). 

It focuses exclusively on deaths of children under the age of one, occurring between April 2014 and March 2021. However, children born after March 30, 2020, are excluded from this analysis.


```python
mortality_records_csv = f'{datasets_folder}/mortality_records.csv'

mortality_records_df = pd.read_csv(mortality_records_csv)
mortality_records_df = mortality_records_df.sort_values(by=['PERIODO_DEF', 'PERIODO_NAC'])
```

Total infants who died before their first birthday (from the children born between April 2014 and March 2020).


```python

mortality_df = mortality_records_df[(mortality_records_df['ANIO_NACIM'] >= 2014) & (mortality_records_df['ANIO_NACIM'] <= 2020)]
mortality_df = mortality_df[((mortality_df['ANIO_NACIM'] == 2014) & (mortality_df['MES_NACIM'] > 3)) | (mortality_df['ANIO_NACIM'] > 2014)]
mortality_df = mortality_df[((mortality_df['ANIO_NACIM'] == 2020) & (mortality_df['MES_NACIM'] < 4)) | (mortality_df['ANIO_NACIM'] < 2020)]

print(f'Total deaths:  {len(mortality_df):,}')
```

    Total deaths:  144,243


### Dataset Overview

This dataset contains 144,243 records, with each entry including the following features:

- **ENT_RESID**: State of residence (Identifier).
- **PERIODO_DEF**: Date of death, recorded in Year/Month format.
- **ANIO_OCUR**: Year of death.
- **MES_OCUR**: Month of death.
- **PERIODO_NAC**: Date of birth, recorded in Year/Month format.
- **ANIO_NACIM**: Year of birth.
- **MES_NACIM**: Month of birth.
- **CAUSA_DEF**: Cause of death, represented by the ICD-10 code.
- **EDAD**: Registered age at the time of death.


```python
mortality_df.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ENT_RESID</th>
      <th>PERIODO_DEF</th>
      <th>ANIO_OCUR</th>
      <th>MES_OCURR</th>
      <th>PERIODO_NAC</th>
      <th>ANIO_NACIM</th>
      <th>MES_NACIM</th>
      <th>CAUSA_DEF</th>
      <th>EDAD</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>146862</th>
      <td>12</td>
      <td>2021/03</td>
      <td>2021</td>
      <td>3</td>
      <td>2020/03</td>
      <td>2020</td>
      <td>3</td>
      <td>B199</td>
      <td>3011</td>
    </tr>
    <tr>
      <th>146863</th>
      <td>21</td>
      <td>2021/03</td>
      <td>2021</td>
      <td>3</td>
      <td>2020/03</td>
      <td>2020</td>
      <td>3</td>
      <td>J988</td>
      <td>3011</td>
    </tr>
    <tr>
      <th>146870</th>
      <td>26</td>
      <td>2021/03</td>
      <td>2021</td>
      <td>3</td>
      <td>2020/03</td>
      <td>2020</td>
      <td>3</td>
      <td>Q243</td>
      <td>3011</td>
    </tr>
    <tr>
      <th>146872</th>
      <td>23</td>
      <td>2021/03</td>
      <td>2021</td>
      <td>3</td>
      <td>2020/03</td>
      <td>2020</td>
      <td>3</td>
      <td>J189</td>
      <td>3011</td>
    </tr>
    <tr>
      <th>146877</th>
      <td>30</td>
      <td>2021/03</td>
      <td>2021</td>
      <td>3</td>
      <td>2020/03</td>
      <td>2020</td>
      <td>3</td>
      <td>Q210</td>
      <td>3011</td>
    </tr>
  </tbody>
</table>
</div>



Of the total recorded deaths, 8,805 were attributed to respiratory illnesses, classified under ICD-10 codes **J** and **U**.


```python
respiratory_illness_df1 = mortality_df[(mortality_df['CAUSA_DEF'].str.startswith('J')) | (mortality_df['CAUSA_DEF'].str.startswith('U'))]

```


```python
print(f'Deaths attributed to respiratory illnesses: {len(respiratory_illness_df1):,}')
```

    Deaths attributed to respiratory illnesses: 8,805


## Registered Births

The following dataset contains statistics on registered births from the [Subsystem of Demographic and Social Information](https://www.inegi.org.mx/programas/natalidad/#datos_abiertos). 

It focuses exclusively on bitrhs occurring between April 2014 and March 2020.


```python
birth_records_csv = f'{datasets_folder}/birth_records.csv'

birth_records_df = pd.read_csv(birth_records_csv)
```

Total births between April 2014 and March 2020 in Mexico


```python
print(f'Total births: {len(birth_records_df):,}')
```

    Total births: 12,604,902


### Dataset overview

This dataset contains 12,604,902 records, with each entry including the following features:

- **ENT_RESID**: State of residence (Identifier).
- **PERIODO_NAC**: Date of birth, recorded in Year/Month format.
- **ANO_NAC**: Year of birth.
- **MES_NAC**: Month of birth.


```python
birth_records_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ENT_RESID</th>
      <th>PERIODO_NAC</th>
      <th>ANO_NAC</th>
      <th>MES_NAC</th>
      <th>SEXO</th>
      <th>EDAD_MADN</th>
      <th>EDAD_PADN</th>
      <th>ORDEN_PART</th>
      <th>TIPO_NAC</th>
      <th>LUGAR_PART</th>
      <th>EDOCIV_MAD</th>
      <th>ESCOL_MAD</th>
      <th>ESCOL_PAD</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2014/04</td>
      <td>2014</td>
      <td>4</td>
      <td>2</td>
      <td>32</td>
      <td>36</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>7</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2014/04</td>
      <td>2014</td>
      <td>4</td>
      <td>2</td>
      <td>26</td>
      <td>25</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>2014/04</td>
      <td>2014</td>
      <td>4</td>
      <td>2</td>
      <td>29</td>
      <td>29</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>7</td>
      <td>7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>2014/04</td>
      <td>2014</td>
      <td>4</td>
      <td>1</td>
      <td>28</td>
      <td>29</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>7</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>2014/04</td>
      <td>2014</td>
      <td>4</td>
      <td>1</td>
      <td>30</td>
      <td>26</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>7</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



## Statistics on registered deaths

Overall, there were 12,604,902 births between April 2014 and March 2020 in Mexico for which maternal residence was one of the 32 Mexican states. 
Of these infants, 144,244 died before their first birthday, corresponding to an infant mortality rate of 11.44 deaths/1,000 births; 8,805 deaths were caused by respiratory illness, corresponding to a respiratory-infant mortality rate of 0.7 deaths/per 1,000 births. As such, respiratory deaths accounted for 6.1% of all demises in infants during the first year of life. Most respiratory deaths (6,531/8,805; 74.1%) corresponded to acute respiratory infections, including COVID-19 (Table 1).


```python
print(f'Infant Mortality Rate: {(len(mortality_df) /  len(birth_records_df) *  1000):.2f} deaths / 1,000 births')
print(f'Infant mortality Rate (Respiratory Ilness): {(len(respiratory_illness_df1) /  len(birth_records_df) *  1000):.2f} per 1,000 births')
print(f'Respiratory deaths (percentage): {(len(respiratory_illness_df1) /  len(mortality_df) *  100):.2f}%')
```

    Infant Mortality Rate: 11.44 deaths / 1,000 births
    Infant mortality Rate (Respiratory Ilness): 0.70 per 1,000 births
    Respiratory deaths (percentage): 6.10%



```python
# Ranges and corresponding diagnoses from the provided table
ranges = {
    "J00X-J069": "Upper respiratory tract infections",
    "J09X-J118": "Influenza",
    "J120-J129": "Viral pneumonia",
    "J13X-J159": "Bacterial pneumonia",
    "J168-J189": "Pneumonia by other agents and non-specified",
    "J200-J219": "Bronchitis and bronchiolitis",
    "J22X": "Acute lower respiratory tract infection, non-specified",
    "J304-J348": "Rhinitis and other nasal disorders",
    "J380-J399": "Laryngeal, pharyngeal and other upper respiratory tract disorders",
    "J42X-J449": "Bronchitis, emphysema, and other chronic lung diseases",
    "J450-J46X": "Asthma",
    "J677-J849": "Pneumonitis associated to diverse conditions and other lung disorders",
    "J852-J948": "Pulmonary abscess, pneumothorax and other pleural space disorders",
    "J960-J969": "Acute and chronic respiratory failure",
    "J980-J989": "Diverse tracheal, bronchial, and lung disorders",
    "U071-U072": "COVID-19"
}

# Function to extract the numeric part of ICD-10 code
def extract_numeric_part(code):
    return ''.join(filter(str.isdigit, code))

# Function to categorize the ICD-10 codes based on ranges or single values
def categorize_code(code):
    for range_key in ranges:
        if '-' in range_key:  # This is a range of codes
            start_range, end_range = range_key.split('-')
            start_numeric = extract_numeric_part(start_range)
            end_numeric = extract_numeric_part(end_range)
            code_numeric = extract_numeric_part(code)

            # Check if the numeric part of the code falls within the range
            if start_range[:1] == code[:1] and start_numeric <= code_numeric <= end_numeric:
                return range_key
        else:  # This is a single ICD-10 code (e.g., J22X)
            if code.startswith(range_key[:3]):  # Match the first part of the code (e.g., J22)
                return range_key
    return None

grouped_df = respiratory_illness_df1.groupby('CAUSA_DEF').size().reset_index(name='Deaths')

# Add a column to categorize the ICD-10 codes
grouped_df['Category'] = grouped_df['CAUSA_DEF'].apply(categorize_code)

# Group by the ICD-10 categories and sum the deaths
grouped_df = grouped_df.groupby('Category')['Deaths'].sum().reset_index()

# Merge the grouped data with the corresponding diagnoses
grouped_df['Diagnoses'] = grouped_df['Category'].map(ranges)

# Calculate the percentage of total deaths (8805 deaths)
total_deaths = grouped_df['Deaths'].sum()
grouped_df['Percentage'] = (grouped_df['Deaths'] / total_deaths * 100).apply(lambda x: round(x, 2))

# Rearrange columns for the final table format
grouped_df = grouped_df[['Category', 'Diagnoses', 'Deaths', 'Percentage']].sort_values(by='Category')

```

**Table  1.** Respiratory mortality diagnoses for Mexican infants born between April 2014 and March 2020 who died during the first year


```python
grouped_df.head(16)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Category</th>
      <th>Diagnoses</th>
      <th>Deaths</th>
      <th>Percentage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>J00X-J069</td>
      <td>Upper respiratory tract infections</td>
      <td>199</td>
      <td>2.26</td>
    </tr>
    <tr>
      <th>1</th>
      <td>J09X-J118</td>
      <td>Influenza</td>
      <td>93</td>
      <td>1.06</td>
    </tr>
    <tr>
      <th>2</th>
      <td>J120-J129</td>
      <td>Viral pneumonia</td>
      <td>54</td>
      <td>0.61</td>
    </tr>
    <tr>
      <th>3</th>
      <td>J13X-J159</td>
      <td>Bacterial pneumonia</td>
      <td>282</td>
      <td>3.20</td>
    </tr>
    <tr>
      <th>4</th>
      <td>J168-J189</td>
      <td>Pneumonia by other agents and non-specified</td>
      <td>4932</td>
      <td>56.01</td>
    </tr>
    <tr>
      <th>5</th>
      <td>J200-J219</td>
      <td>Bronchitis and bronchiolitis</td>
      <td>545</td>
      <td>6.19</td>
    </tr>
    <tr>
      <th>6</th>
      <td>J22X</td>
      <td>Acute lower respiratory tract infection, non-s...</td>
      <td>338</td>
      <td>3.84</td>
    </tr>
    <tr>
      <th>7</th>
      <td>J304-J348</td>
      <td>Rhinitis and other nasal disorders</td>
      <td>8</td>
      <td>0.09</td>
    </tr>
    <tr>
      <th>8</th>
      <td>J380-J399</td>
      <td>Laryngeal, pharyngeal and other upper respirat...</td>
      <td>30</td>
      <td>0.34</td>
    </tr>
    <tr>
      <th>9</th>
      <td>J42X-J449</td>
      <td>Bronchitis, emphysema, and other chronic lung ...</td>
      <td>74</td>
      <td>0.84</td>
    </tr>
    <tr>
      <th>10</th>
      <td>J450-J46X</td>
      <td>Asthma</td>
      <td>119</td>
      <td>1.35</td>
    </tr>
    <tr>
      <th>11</th>
      <td>J677-J849</td>
      <td>Pneumonitis associated to diverse conditions a...</td>
      <td>624</td>
      <td>7.09</td>
    </tr>
    <tr>
      <th>12</th>
      <td>J852-J948</td>
      <td>Pulmonary abscess, pneumothorax and other pleu...</td>
      <td>95</td>
      <td>1.08</td>
    </tr>
    <tr>
      <th>13</th>
      <td>J960-J969</td>
      <td>Acute and chronic respiratory failure</td>
      <td>171</td>
      <td>1.94</td>
    </tr>
    <tr>
      <th>14</th>
      <td>J980-J989</td>
      <td>Diverse tracheal, bronchial, and lung disorders</td>
      <td>1153</td>
      <td>13.09</td>
    </tr>
    <tr>
      <th>15</th>
      <td>U071-U072</td>
      <td>COVID-19</td>
      <td>88</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
</div>



**Figure 1.** Monthly respiratory infant mortality rate in Mexico according to date of demise and birthdate between April 2014 and March 2020

The following code creates a dataset containig mortality rates based on both month of occurence and month of birth



```python
# Registered deaths grouped by year/month of birth 
grouped_deaths_df1 = respiratory_illness_df1.groupby('PERIODO_NAC').size().reset_index(name='Deaths_Birth')
# Register births grouped by year/month
grouped_births_df = birth_records_df.groupby('PERIODO_NAC').size().reset_index(name='Births')

# Registered respiratory illness deaths grouped by year/month of occurence 
respiratory_illness_df2 = mortality_records_df[(mortality_records_df['CAUSA_DEF'].str.startswith('J')) | (mortality_records_df['CAUSA_DEF'].str.startswith('U'))]
grouped_deaths_df2 = respiratory_illness_df2.groupby('PERIODO_DEF').size().reset_index(name='Deaths_Occurence')

grouped_df = pd.merge(grouped_births_df, grouped_deaths_df1, on='PERIODO_NAC', how='inner')
grouped_df = pd.merge(grouped_df, grouped_deaths_df2, left_on='PERIODO_NAC', right_on='PERIODO_DEF', how='inner')

grouped_df.drop(columns=['PERIODO_DEF'], inplace=True)
grouped_df['Mortality_Rate_Birth'] =  grouped_df['Deaths_Birth'] / grouped_df['Births'] *  1000
grouped_df['Mortality_Rate_Occurence'] =  grouped_df['Deaths_Occurence'] / grouped_df['Births'] *  1000

grouped_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PERIODO_NAC</th>
      <th>Births</th>
      <th>Deaths_Birth</th>
      <th>Deaths_Occurence</th>
      <th>Mortality_Rate_Birth</th>
      <th>Mortality_Rate_Occurence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2014/04</td>
      <td>176479</td>
      <td>102</td>
      <td>97</td>
      <td>0.577972</td>
      <td>0.549640</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2014/05</td>
      <td>182031</td>
      <td>142</td>
      <td>110</td>
      <td>0.780087</td>
      <td>0.604293</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2014/06</td>
      <td>178619</td>
      <td>132</td>
      <td>95</td>
      <td>0.739003</td>
      <td>0.531858</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2014/07</td>
      <td>191022</td>
      <td>159</td>
      <td>117</td>
      <td>0.832365</td>
      <td>0.612495</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2014/08</td>
      <td>200111</td>
      <td>165</td>
      <td>85</td>
      <td>0.824542</td>
      <td>0.424764</td>
    </tr>
  </tbody>
</table>
</div>




```python

# Plotting the data
plt.figure(figsize=(12, 6))

# Plotting Mortality Rate based on Birth
plt.plot(grouped_df['PERIODO_NAC'], grouped_df['Mortality_Rate_Birth'], label='Mortality Rate (J + U) month of birth',  linestyle='-', color='gray' if grayscale else 'orange')

# Plotting Mortality Rate based on Occurrence
plt.plot(grouped_df['PERIODO_NAC'], grouped_df['Mortality_Rate_Occurence'], label='Mortality Rate (J + U) month of occurrence',  linestyle=(0, (1, 3)), color='gray' if grayscale else 'blue')


plt.legend()

plt.xticks(rotation=90, ha='center')

plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6])  # You can specify any list of values for the y-ticks


# Show grid
plt.grid(axis='y')

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2) 

# Remove grid borders (spines)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)  

# Adjust layout
plt.tight_layout()

plt.savefig(f'{plots_folder}/Fig1.{image_format}', format=image_format, dpi=1200)


# Display the plot
plt.show()
```
    
![png](nbconvert_output/output_29_1.png)
    


**Figure 2.** Average monthly respiratory (J and U ICD-10 codes) infant mortality rates based on month of occurrence and month of birth in Mexico (April 2014-March 2020).


```python
def rate_and_ci(deaths, births, scale=1000.0, z=1.96):
    deaths = deaths.astype(float)
    births = births.astype(float)
    rate = np.round((deaths / births) * scale, 2)
    se = np.sqrt(deaths) / births * scale
    lo = np.round(np.maximum(0.0, rate - z * se), 2)
    hi = np.round(rate + z * se, 2)
    return rate, lo, hi
```


```python
# Registered deaths grouped by year/month of birth 
grouped_deaths_df1 = respiratory_illness_df1.groupby('MES_NACIM').size().reset_index(name='Deaths_Birth')
# Register births grouped by year/month
grouped_births_df = birth_records_df.groupby('MES_NAC').size().reset_index(name='Births')

# Registered respiratory illness deaths grouped by year/month of occurence 
respiratory_illness_df2 = mortality_records_df[(mortality_records_df['CAUSA_DEF'].str.startswith('J')) | (mortality_records_df['CAUSA_DEF'].str.startswith('U'))]
grouped_deaths_df2 = respiratory_illness_df2.groupby('MES_OCURR').size().reset_index(name='Deaths_Occurence')

grouped_df = pd.merge(grouped_births_df, grouped_deaths_df1, left_on='MES_NAC', right_on='MES_NACIM', how='inner')
grouped_df = pd.merge(grouped_df, grouped_deaths_df2, left_on='MES_NAC', right_on='MES_OCURR', how='inner')

grouped_df.drop(columns=['MES_NACIM', 'MES_OCURR'], inplace=True)

rate, lo, hi = rate_and_ci(grouped_df['Deaths_Birth'], grouped_df['Births'])
grouped_df['Mortality_Rate_Birth'] =  rate
grouped_df['Mortality_Rate_Birth_Lower'] =  lo
grouped_df['Mortality_Rate_Birth_Upper'] =  hi

rate, lo, hi = rate_and_ci(grouped_df['Deaths_Occurence'], grouped_df['Births'])
grouped_df['Mortality_Rate_Occurence'] =  rate
grouped_df['Mortality_Rate_Occurence_Lower'] =  lo
grouped_df['Mortality_Rate_Occurence_Upper'] =  hi

# Replace the 'MES_NAC' values with month names
grouped_df['MES_NAC'] = grouped_df['MES_NAC'].map(month_mapping)

# Convert the 'MES_NAC' column to a categorical type with the custom order
grouped_df['MES_NAC'] = pd.Categorical(grouped_df['MES_NAC'], categories=custom_order, ordered=True)

# Sort the dataframe based on the custom order
grouped_df = grouped_df.sort_values('MES_NAC')

grouped_df.head(12)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MES_NAC</th>
      <th>Births</th>
      <th>Deaths_Birth</th>
      <th>Deaths_Occurence</th>
      <th>Mortality_Rate_Birth</th>
      <th>Mortality_Rate_Birth_Lower</th>
      <th>Mortality_Rate_Birth_Upper</th>
      <th>Mortality_Rate_Occurence</th>
      <th>Mortality_Rate_Occurence_Lower</th>
      <th>Mortality_Rate_Occurence_Upper</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>July</td>
      <td>1076970</td>
      <td>741</td>
      <td>624</td>
      <td>0.69</td>
      <td>0.64</td>
      <td>0.74</td>
      <td>0.58</td>
      <td>0.53</td>
      <td>0.63</td>
    </tr>
    <tr>
      <th>7</th>
      <td>August</td>
      <td>1136385</td>
      <td>812</td>
      <td>517</td>
      <td>0.71</td>
      <td>0.66</td>
      <td>0.76</td>
      <td>0.45</td>
      <td>0.41</td>
      <td>0.49</td>
    </tr>
    <tr>
      <th>8</th>
      <td>September</td>
      <td>1181923</td>
      <td>992</td>
      <td>567</td>
      <td>0.84</td>
      <td>0.79</td>
      <td>0.89</td>
      <td>0.48</td>
      <td>0.44</td>
      <td>0.52</td>
    </tr>
    <tr>
      <th>9</th>
      <td>October</td>
      <td>1153400</td>
      <td>989</td>
      <td>688</td>
      <td>0.86</td>
      <td>0.81</td>
      <td>0.91</td>
      <td>0.60</td>
      <td>0.56</td>
      <td>0.64</td>
    </tr>
    <tr>
      <th>10</th>
      <td>November</td>
      <td>1056550</td>
      <td>874</td>
      <td>917</td>
      <td>0.83</td>
      <td>0.78</td>
      <td>0.88</td>
      <td>0.87</td>
      <td>0.81</td>
      <td>0.93</td>
    </tr>
    <tr>
      <th>11</th>
      <td>December</td>
      <td>1067736</td>
      <td>774</td>
      <td>1180</td>
      <td>0.72</td>
      <td>0.67</td>
      <td>0.77</td>
      <td>1.11</td>
      <td>1.05</td>
      <td>1.17</td>
    </tr>
    <tr>
      <th>0</th>
      <td>January</td>
      <td>1019441</td>
      <td>616</td>
      <td>1145</td>
      <td>0.60</td>
      <td>0.55</td>
      <td>0.65</td>
      <td>1.12</td>
      <td>1.05</td>
      <td>1.19</td>
    </tr>
    <tr>
      <th>1</th>
      <td>February</td>
      <td>902558</td>
      <td>570</td>
      <td>906</td>
      <td>0.63</td>
      <td>0.58</td>
      <td>0.68</td>
      <td>1.00</td>
      <td>0.93</td>
      <td>1.07</td>
    </tr>
    <tr>
      <th>2</th>
      <td>March</td>
      <td>986549</td>
      <td>523</td>
      <td>803</td>
      <td>0.53</td>
      <td>0.48</td>
      <td>0.58</td>
      <td>0.81</td>
      <td>0.75</td>
      <td>0.87</td>
    </tr>
    <tr>
      <th>3</th>
      <td>April</td>
      <td>994946</td>
      <td>567</td>
      <td>708</td>
      <td>0.57</td>
      <td>0.52</td>
      <td>0.62</td>
      <td>0.71</td>
      <td>0.66</td>
      <td>0.76</td>
    </tr>
    <tr>
      <th>4</th>
      <td>May</td>
      <td>1024246</td>
      <td>662</td>
      <td>650</td>
      <td>0.65</td>
      <td>0.60</td>
      <td>0.70</td>
      <td>0.63</td>
      <td>0.58</td>
      <td>0.68</td>
    </tr>
    <tr>
      <th>5</th>
      <td>June</td>
      <td>1004198</td>
      <td>685</td>
      <td>518</td>
      <td>0.68</td>
      <td>0.63</td>
      <td>0.73</td>
      <td>0.52</td>
      <td>0.48</td>
      <td>0.56</td>
    </tr>
  </tbody>
</table>
</div>




```python

months = grouped_df['MES_NAC']
mortality_rate_birth = grouped_df['Mortality_Rate_Birth']
ci95_Lower_birth = grouped_df['Mortality_Rate_Birth_Lower']
ci95_Upper_birth = grouped_df['Mortality_Rate_Birth_Upper']
mortality_rate_occurrence = grouped_df['Mortality_Rate_Occurence']
ci95_Lower_occurrence = grouped_df['Mortality_Rate_Occurence_Lower']
ci95_Upper_occurrence = grouped_df['Mortality_Rate_Occurence_Upper']

# Set up the figure and axis
plt.figure(figsize=(12, 6))

# Set bar width
bar_width = 0.35

# Set positions for each bar
index = np.arange(len(months)) *  1.2

# Create bars for Month of Birth Mortality Rate
plt.bar(index - bar_width / 3, 
        mortality_rate_birth, 
        bar_width, 
        label='Month of birth J+U mortality', 
        color='lightgray' if grayscale else 'orange', 
        edgecolor='white')

# Error bars
plt.errorbar(index - bar_width / 3, 
        mortality_rate_birth,
        yerr=[mortality_rate_birth-ci95_Lower_birth,
                ci95_Upper_birth-mortality_rate_birth],
        color='black', 
        fmt='none', 
        elinewidth=1, 
        capsize=3)

# Create bars for Month of Occurrence Mortality Rate, offset by bar_width
plt.bar(index + bar_width, 
        mortality_rate_occurrence, 
        bar_width, label='Month of occurrence J+U mortality', 
        hatch='///', 
        color='lightgray' if grayscale else 'blue', 
        edgecolor='white')

# Error bars
plt.errorbar(index + bar_width, 
        mortality_rate_occurrence,
        yerr=[mortality_rate_occurrence-ci95_Lower_occurrence,
                ci95_Upper_occurrence-mortality_rate_occurrence],
        color='black', 
        fmt='none', 
        elinewidth=1, 
        capsize=3)


plt.legend()

plt.xticks(index + bar_width / 3, months, rotation=45, ha='center')

plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])  # You can specify any list of values for the y-ticks


# Show grid
plt.grid(axis='y')

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2) 

# Remove grid borders (spines)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)  

# Adjust layout
plt.tight_layout()

plt.savefig(f'{plots_folder}/Fig2.{image_format}', format=image_format, dpi=1200)


# Display the plot
plt.show()
```
    
![png](nbconvert_output/output_33_1.png)
    


### Chi-square test of homogeneity to evaluate overall seasonal variation

1. Mortality rates based on the month of occurrence
2. Mortality rates based on the month of birth


```python
from scipy.stats import chi2_contingency

# Example for months
table = np.array([grouped_df['Deaths_Occurence'], grouped_df['Births'] - grouped_df['Deaths_Occurence']])
chi2, p, dof, expected = chi2_contingency(table)


print(f"Chi² = {chi2:.3f}, df = {dof}, p-value = {p:.3e}")
```

    Chi² = 907.086, df = 11, p-value = 1.858e-187



```python
from scipy.stats import chi2_contingency

# Example for months
table = np.array([grouped_df['Deaths_Birth'], grouped_df['Births'] - grouped_df['Deaths_Birth']])
chi2, p, dof, expected = chi2_contingency(table)


print(f"Chi² = {chi2:.3f}, df = {dof}, p-value = {p:.3e}")
```

    Chi² = 188.850, df = 11, p-value = 1.520e-34


### Respiratory Mortality Rates by State and Region

To precisely calculate respiratory mortality rates, we first need to retrieve the list of Mexican states along with their corresponding regions. The dataset **State.csv** provides the complete list of Mexican states as of 2024, categorized by their respective regions. This data will serve as the foundation for calculating and analyzing the respiratory mortality rates across different geographic areas.


```python
states_df = pd.read_csv(f'{datasets_folder}/State.csv')

#Let's print the first records.
states_df.head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ENTIDAD</th>
      <th>NOM_ENTIDAD</th>
      <th>REGION</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Aguascalientes</td>
      <td>Centro Occidente</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Baja California</td>
      <td>Noroeste</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Baja California Sur</td>
      <td>Noroeste</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Campeche</td>
      <td>Yucatan</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Coahuila de Zaragoza</td>
      <td>Norte</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>Colima</td>
      <td>Centro Occidente</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>Chiapas</td>
      <td>Sur</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>Chihuahua</td>
      <td>Norte</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>Ciudad de Mexico</td>
      <td>Centro Este</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>Durango</td>
      <td>Norte</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>Guanajuato</td>
      <td>Centro Occidente</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>Guerrero</td>
      <td>Sur</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>Hidalgo</td>
      <td>Centro Este</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>Jalisco</td>
      <td>Centro Occidente</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>Mexico</td>
      <td>Centro Este</td>
    </tr>
    <tr>
      <th>15</th>
      <td>16</td>
      <td>Michoacan de Ocampo</td>
      <td>Centro Occidente</td>
    </tr>
    <tr>
      <th>16</th>
      <td>17</td>
      <td>Morelos</td>
      <td>Centro Este</td>
    </tr>
    <tr>
      <th>17</th>
      <td>18</td>
      <td>Nayarit</td>
      <td>Noroeste</td>
    </tr>
    <tr>
      <th>18</th>
      <td>19</td>
      <td>Nuevo Leon</td>
      <td>Noreste</td>
    </tr>
    <tr>
      <th>19</th>
      <td>20</td>
      <td>Oaxaca</td>
      <td>Sur</td>
    </tr>
  </tbody>
</table>
</div>




```python
grouped_deaths = respiratory_illness_df1.groupby('ENT_RESID').size().reset_index(name='Deaths')
grouped_deaths.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ENT_RESID</th>
      <th>Deaths</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>80</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>209</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>36</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>58</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>169</td>
    </tr>
  </tbody>
</table>
</div>




```python
grouped_births = birth_records_df.groupby('ENT_RESID').size().reset_index(name='Births')
grouped_births.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ENT_RESID</th>
      <th>Births</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>154704</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>318748</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>71916</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>94852</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>335568</td>
    </tr>
  </tbody>
</table>
</div>



**Supplementary Table 1.** Respiratory infant mortality rate by state.


```python
grouped_df = pd.merge(states_df, grouped_births, left_on='ENTIDAD', right_on='ENT_RESID', how='inner')
grouped_df = pd.merge(grouped_df, grouped_deaths, left_on='ENTIDAD', right_on='ENT_RESID', how='inner')
# Drop unused columns and rename existing ones
grouped_df.drop(columns=['ENT_RESID_x', 'ENT_RESID_y'], inplace=True)
grouped_df.rename(columns={'ENTIDAD': 'State_Code', 'NOM_ENTIDAD': 'State'}, inplace=True)

# Calculate and append the totals
totals = grouped_df[['Births', 'Deaths']].sum()
totals_row = pd.DataFrame([totals], columns=grouped_df.columns)
totals_row['State'] = 'Mexico (Country)'
totals_row['REGION'] = 'TOTAL'
grouped_df = pd.concat([grouped_df, totals_row], ignore_index=True)

r, lo, hi = rate_and_ci(grouped_df['Deaths'].values, grouped_df['Births'].values)

grouped_df['Respiratory_Mortality_Rate'] = r
grouped_df['CI95_Lower'] = lo
grouped_df['CI95_Upper'] = hi
grouped_df.head(33)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State_Code</th>
      <th>State</th>
      <th>REGION</th>
      <th>Births</th>
      <th>Deaths</th>
      <th>Respiratory_Mortality_Rate</th>
      <th>CI95_Lower</th>
      <th>CI95_Upper</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>Aguascalientes</td>
      <td>Centro Occidente</td>
      <td>154704.0</td>
      <td>80.0</td>
      <td>0.52</td>
      <td>0.41</td>
      <td>0.63</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>Baja California</td>
      <td>Noroeste</td>
      <td>318748.0</td>
      <td>209.0</td>
      <td>0.66</td>
      <td>0.57</td>
      <td>0.75</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.0</td>
      <td>Baja California Sur</td>
      <td>Noroeste</td>
      <td>71916.0</td>
      <td>36.0</td>
      <td>0.50</td>
      <td>0.34</td>
      <td>0.66</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.0</td>
      <td>Campeche</td>
      <td>Yucatan</td>
      <td>94852.0</td>
      <td>58.0</td>
      <td>0.61</td>
      <td>0.45</td>
      <td>0.77</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>Coahuila de Zaragoza</td>
      <td>Norte</td>
      <td>335568.0</td>
      <td>169.0</td>
      <td>0.50</td>
      <td>0.42</td>
      <td>0.58</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6.0</td>
      <td>Colima</td>
      <td>Centro Occidente</td>
      <td>66398.0</td>
      <td>27.0</td>
      <td>0.41</td>
      <td>0.26</td>
      <td>0.56</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7.0</td>
      <td>Chiapas</td>
      <td>Sur</td>
      <td>780847.0</td>
      <td>1093.0</td>
      <td>1.40</td>
      <td>1.32</td>
      <td>1.48</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8.0</td>
      <td>Chihuahua</td>
      <td>Norte</td>
      <td>353264.0</td>
      <td>314.0</td>
      <td>0.89</td>
      <td>0.79</td>
      <td>0.99</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9.0</td>
      <td>Ciudad de Mexico</td>
      <td>Centro Este</td>
      <td>682342.0</td>
      <td>497.0</td>
      <td>0.73</td>
      <td>0.67</td>
      <td>0.79</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10.0</td>
      <td>Durango</td>
      <td>Norte</td>
      <td>204847.0</td>
      <td>165.0</td>
      <td>0.81</td>
      <td>0.69</td>
      <td>0.93</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11.0</td>
      <td>Guanajuato</td>
      <td>Centro Occidente</td>
      <td>669729.0</td>
      <td>429.0</td>
      <td>0.64</td>
      <td>0.58</td>
      <td>0.70</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12.0</td>
      <td>Guerrero</td>
      <td>Sur</td>
      <td>425281.0</td>
      <td>242.0</td>
      <td>0.57</td>
      <td>0.50</td>
      <td>0.64</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13.0</td>
      <td>Hidalgo</td>
      <td>Centro Este</td>
      <td>283658.0</td>
      <td>151.0</td>
      <td>0.53</td>
      <td>0.45</td>
      <td>0.61</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14.0</td>
      <td>Jalisco</td>
      <td>Centro Occidente</td>
      <td>854794.0</td>
      <td>445.0</td>
      <td>0.52</td>
      <td>0.47</td>
      <td>0.57</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15.0</td>
      <td>Mexico</td>
      <td>Centro Este</td>
      <td>1615543.0</td>
      <td>902.0</td>
      <td>0.56</td>
      <td>0.52</td>
      <td>0.60</td>
    </tr>
    <tr>
      <th>15</th>
      <td>16.0</td>
      <td>Michoacan de Ocampo</td>
      <td>Centro Occidente</td>
      <td>535308.0</td>
      <td>317.0</td>
      <td>0.59</td>
      <td>0.52</td>
      <td>0.66</td>
    </tr>
    <tr>
      <th>16</th>
      <td>17.0</td>
      <td>Morelos</td>
      <td>Centro Este</td>
      <td>177192.0</td>
      <td>57.0</td>
      <td>0.32</td>
      <td>0.24</td>
      <td>0.40</td>
    </tr>
    <tr>
      <th>17</th>
      <td>18.0</td>
      <td>Nayarit</td>
      <td>Noroeste</td>
      <td>117962.0</td>
      <td>66.0</td>
      <td>0.56</td>
      <td>0.43</td>
      <td>0.69</td>
    </tr>
    <tr>
      <th>18</th>
      <td>19.0</td>
      <td>Nuevo Leon</td>
      <td>Noreste</td>
      <td>540412.0</td>
      <td>276.0</td>
      <td>0.51</td>
      <td>0.45</td>
      <td>0.57</td>
    </tr>
    <tr>
      <th>19</th>
      <td>20.0</td>
      <td>Oaxaca</td>
      <td>Sur</td>
      <td>447348.0</td>
      <td>349.0</td>
      <td>0.78</td>
      <td>0.70</td>
      <td>0.86</td>
    </tr>
    <tr>
      <th>20</th>
      <td>21.0</td>
      <td>Puebla</td>
      <td>Centro Este</td>
      <td>741317.0</td>
      <td>766.0</td>
      <td>1.03</td>
      <td>0.96</td>
      <td>1.10</td>
    </tr>
    <tr>
      <th>21</th>
      <td>22.0</td>
      <td>Queretaro</td>
      <td>Centro Este</td>
      <td>234345.0</td>
      <td>97.0</td>
      <td>0.41</td>
      <td>0.33</td>
      <td>0.49</td>
    </tr>
    <tr>
      <th>22</th>
      <td>23.0</td>
      <td>Quintana Roo</td>
      <td>Yucatan</td>
      <td>171679.0</td>
      <td>120.0</td>
      <td>0.70</td>
      <td>0.57</td>
      <td>0.83</td>
    </tr>
    <tr>
      <th>23</th>
      <td>24.0</td>
      <td>San Luis Potosi</td>
      <td>Norte</td>
      <td>287471.0</td>
      <td>161.0</td>
      <td>0.56</td>
      <td>0.47</td>
      <td>0.65</td>
    </tr>
    <tr>
      <th>24</th>
      <td>25.0</td>
      <td>Sinaloa</td>
      <td>Noroeste</td>
      <td>293436.0</td>
      <td>187.0</td>
      <td>0.64</td>
      <td>0.55</td>
      <td>0.73</td>
    </tr>
    <tr>
      <th>25</th>
      <td>26.0</td>
      <td>Sonora</td>
      <td>Noroeste</td>
      <td>265796.0</td>
      <td>202.0</td>
      <td>0.76</td>
      <td>0.66</td>
      <td>0.86</td>
    </tr>
    <tr>
      <th>26</th>
      <td>27.0</td>
      <td>Tabasco</td>
      <td>Centro Sur</td>
      <td>266569.0</td>
      <td>228.0</td>
      <td>0.86</td>
      <td>0.75</td>
      <td>0.97</td>
    </tr>
    <tr>
      <th>27</th>
      <td>28.0</td>
      <td>Tamaulipas</td>
      <td>Noreste</td>
      <td>331702.0</td>
      <td>157.0</td>
      <td>0.47</td>
      <td>0.40</td>
      <td>0.54</td>
    </tr>
    <tr>
      <th>28</th>
      <td>29.0</td>
      <td>Tlaxcala</td>
      <td>Centro Este</td>
      <td>143860.0</td>
      <td>119.0</td>
      <td>0.83</td>
      <td>0.68</td>
      <td>0.98</td>
    </tr>
    <tr>
      <th>29</th>
      <td>30.0</td>
      <td>Veracruz de Ignacio de la Llave</td>
      <td>Centro Sur</td>
      <td>739486.0</td>
      <td>570.0</td>
      <td>0.77</td>
      <td>0.71</td>
      <td>0.83</td>
    </tr>
    <tr>
      <th>30</th>
      <td>31.0</td>
      <td>Yucatan</td>
      <td>Yucatan</td>
      <td>209705.0</td>
      <td>211.0</td>
      <td>1.01</td>
      <td>0.87</td>
      <td>1.15</td>
    </tr>
    <tr>
      <th>31</th>
      <td>32.0</td>
      <td>Zacatecas</td>
      <td>Norte</td>
      <td>188823.0</td>
      <td>105.0</td>
      <td>0.56</td>
      <td>0.45</td>
      <td>0.67</td>
    </tr>
    <tr>
      <th>32</th>
      <td>NaN</td>
      <td>Mexico (Country)</td>
      <td>TOTAL</td>
      <td>12604902.0</td>
      <td>8805.0</td>
      <td>0.70</td>
      <td>0.69</td>
      <td>0.71</td>
    </tr>
  </tbody>
</table>
</div>



**Supplementary Table 2.** States included in each of the eight regions analyzed in the study.


```python
pd.set_option('display.max_colwidth', 150)
grouped_df = states_df.groupby('REGION')['NOM_ENTIDAD'].apply(lambda x: ', '.join(x)).reset_index()
grouped_df.columns = ['Region', 'States']
grouped_df.head(8)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Region</th>
      <th>States</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Centro Este</td>
      <td>Ciudad de Mexico, Hidalgo, Mexico, Morelos, Puebla, Queretaro, Tlaxcala</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Centro Occidente</td>
      <td>Aguascalientes, Colima, Guanajuato, Jalisco, Michoacan de Ocampo</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Centro Sur</td>
      <td>Tabasco, Veracruz de Ignacio de la Llave</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Noreste</td>
      <td>Nuevo Leon, Tamaulipas</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Noroeste</td>
      <td>Baja California, Baja California Sur, Nayarit, Sinaloa, Sonora</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Norte</td>
      <td>Coahuila de Zaragoza, Chihuahua, Durango, San Luis Potosi, Zacatecas</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Sur</td>
      <td>Chiapas, Guerrero, Oaxaca</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Yucatan</td>
      <td>Campeche, Quintana Roo, Yucatan</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Restore default settings after customization
pd.reset_option('display.max_colwidth')
```

#### Monthly Mortality Rates (National)


```python
# Registered deaths grouped by year/month of birth 
grouped_deaths_df1 = respiratory_illness_df1.groupby('MES_NACIM').size().reset_index(name='Deaths_Birth')
# Register births grouped by year/month
grouped_births_df = birth_records_df.groupby('MES_NAC').size().reset_index(name='Births')


grouped_df = pd.merge(grouped_births_df, grouped_deaths_df1, left_on='MES_NAC', right_on='MES_NACIM', how='inner')

grouped_df.drop(columns=['MES_NACIM'], inplace=True)
grouped_df['Mortality_Rate_Birth'] =  (grouped_df['Deaths_Birth'] / grouped_df['Births'] *  1000).apply(lambda x: round(x, 2))

# Replace the 'MES_NAC' values with month names
grouped_df['MES_NAC'] = grouped_df['MES_NAC'].map(month_mapping)

# Convert the 'MES_NAC' column to a categorical type with the custom order
grouped_df['MES_NAC'] = pd.Categorical(grouped_df['MES_NAC'], categories=custom_order, ordered=True)

# Sort the dataframe based on the custom order
grouped_df = grouped_df.sort_values('MES_NAC')

grouped_df.rename(columns={'MES_NAC': 'Month', 'Mortality_Rate_Birth': 'Mortality_Rate'}, inplace=True)
monthly_mortality_rates = grouped_df[['Month', 'Mortality_Rate']]
monthly_mortality_rates.head(12)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Month</th>
      <th>Mortality_Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>July</td>
      <td>0.69</td>
    </tr>
    <tr>
      <th>7</th>
      <td>August</td>
      <td>0.71</td>
    </tr>
    <tr>
      <th>8</th>
      <td>September</td>
      <td>0.84</td>
    </tr>
    <tr>
      <th>9</th>
      <td>October</td>
      <td>0.86</td>
    </tr>
    <tr>
      <th>10</th>
      <td>November</td>
      <td>0.83</td>
    </tr>
    <tr>
      <th>11</th>
      <td>December</td>
      <td>0.72</td>
    </tr>
    <tr>
      <th>0</th>
      <td>January</td>
      <td>0.60</td>
    </tr>
    <tr>
      <th>1</th>
      <td>February</td>
      <td>0.63</td>
    </tr>
    <tr>
      <th>2</th>
      <td>March</td>
      <td>0.53</td>
    </tr>
    <tr>
      <th>3</th>
      <td>April</td>
      <td>0.57</td>
    </tr>
    <tr>
      <th>4</th>
      <td>May</td>
      <td>0.65</td>
    </tr>
    <tr>
      <th>5</th>
      <td>June</td>
      <td>0.68</td>
    </tr>
  </tbody>
</table>
</div>




```python
def fit_count_glm(formula, data, offset_col, family='poisson'):
    if family == 'poisson':
        fam = sm.families.Poisson()
    else:
        fam = sm.families.NegativeBinomial()
        
    model = smf.glm(formula=formula,
                    data=data,
                    family=fam,
                    offset=np.log(data[offset_col])).fit()
    return model

def dispersion_ratio(model):
    return model.pearson_chi2 / model.df_resid

```

#### Respiratory Mortality Rates (Grouped by Region)


```python
# Registered deaths grouped by year/month of birth 
grouped_deaths_df = respiratory_illness_df1.groupby(['ENT_RESID','MES_NACIM']).size().reset_index(name='Deaths')
grouped_deaths_df = pd.merge(grouped_deaths_df, states_df, left_on='ENT_RESID', right_on='ENTIDAD', how='inner')
#grouped_deaths_df = grouped_deaths_df.groupby(['REGION','MES_NACIM']).size().reset_index(name='Deaths')

# Register births grouped by year/month
grouped_births_df = birth_records_df.groupby(['ENT_RESID','MES_NAC']).size().reset_index(name='Births')
grouped_births_df = pd.merge(grouped_births_df, states_df, left_on='ENT_RESID', right_on='ENTIDAD', how='inner')
grouped_births_df.head()

grouped_df = pd.merge(grouped_births_df, grouped_deaths_df, left_on=['ENT_RESID', 'MES_NAC'], right_on=['ENT_RESID', 'MES_NACIM'], how='left')
grouped_df.drop(columns=['ENTIDAD_x','MES_NACIM','ENTIDAD_y','NOM_ENTIDAD_y', 'REGION_y'], inplace=True)
grouped_df.rename(columns={'REGION_x': 'Region', 'NOM_ENTIDAD_x': 'State', 'MES_NAC': 'Month'}, inplace=True)
grouped_df['Deaths'] = grouped_df['Deaths'].fillna(0)

regional_mortality_rates = grouped_df.groupby(['Region'])[['Births', 'Deaths']].sum().reset_index()
r, lo, hi = rate_and_ci(regional_mortality_rates['Deaths'].values, regional_mortality_rates['Births'].values)
regional_mortality_rates['Mortality_Rate'] =  r
regional_mortality_rates['CI95_Lower'] = lo
regional_mortality_rates['CI95_Upper'] = hi

regional_mortality_rates.drop(columns=['Births','Deaths'], inplace=True)

regional_mortality_rates.head(9)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Region</th>
      <th>Mortality_Rate</th>
      <th>CI95_Lower</th>
      <th>CI95_Upper</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Centro Este</td>
      <td>0.67</td>
      <td>0.64</td>
      <td>0.70</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Centro Occidente</td>
      <td>0.57</td>
      <td>0.54</td>
      <td>0.60</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Centro Sur</td>
      <td>0.79</td>
      <td>0.73</td>
      <td>0.85</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Noreste</td>
      <td>0.50</td>
      <td>0.45</td>
      <td>0.55</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Noroeste</td>
      <td>0.66</td>
      <td>0.61</td>
      <td>0.71</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Norte</td>
      <td>0.67</td>
      <td>0.63</td>
      <td>0.71</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Sur</td>
      <td>1.02</td>
      <td>0.97</td>
      <td>1.07</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Yucatan</td>
      <td>0.82</td>
      <td>0.74</td>
      <td>0.90</td>
    </tr>
  </tbody>
</table>
</div>




```python
national_mortality_rate = grouped_df['Deaths'].sum() / grouped_df['Births'].sum() *  1000
print(f'Respiratory infant mortality rate: {national_mortality_rate:.2f}%')
```

    Respiratory infant mortality rate: 0.70%


**Table 2.** Respiratory infant mortality rates by birth month in Mexican children born between April 2014 and March 2020.


```python
# Registered deaths grouped by year/month of birth 
grouped_deaths_df = respiratory_illness_df1.groupby(['ENT_RESID','MES_NACIM']).size().reset_index(name='Deaths')
grouped_deaths_df = pd.merge(grouped_deaths_df, states_df, left_on='ENT_RESID', right_on='ENTIDAD', how='inner')
#grouped_deaths_df = grouped_deaths_df.groupby(['REGION','MES_NACIM']).size().reset_index(name='Deaths')

# Register births grouped by year/month
grouped_births_df = birth_records_df.groupby(['ENT_RESID','MES_NAC']).size().reset_index(name='Births')
grouped_births_df = pd.merge(grouped_births_df, states_df, left_on='ENT_RESID', right_on='ENTIDAD', how='inner')

grouped_df = pd.merge(grouped_births_df, grouped_deaths_df, left_on=['ENT_RESID', 'MES_NAC'], right_on=['ENT_RESID', 'MES_NACIM'], how='left')
grouped_df.drop(columns=['ENTIDAD_x','MES_NACIM','ENTIDAD_y','NOM_ENTIDAD_y', 'REGION_y'], inplace=True)
grouped_df.rename(columns={'REGION_x': 'Region', 'NOM_ENTIDAD_x': 'State', 'MES_NAC': 'Month'}, inplace=True)
grouped_df['Deaths'] = grouped_df['Deaths'].fillna(0)

grouped_df = grouped_df.groupby(['Region', 'Month'])[['Births', 'Deaths']].sum().reset_index()
r, lo, hi = rate_and_ci(grouped_df['Deaths'].values, grouped_df['Births'].values)
grouped_df['Mortality_Rate'] =  r

# Calculate confidence intervals
ci_df = grouped_df.groupby(['Region', 'Month'])[['Births', 'Deaths']].sum().reset_index()
ci_df['Rate'], ci_df['CI_Lower'], ci_df['CI_Upper'] = rate_and_ci(ci_df['Deaths'], ci_df['Births'])

# merge with month names and order
ci_df['Month'] = ci_df['Month'].map(month_mapping)
ci_df['Month'] = pd.Categorical(ci_df['Month'], categories=custom_order, ordered=True)
ci_df = ci_df.sort_values(['Region', 'Month'])

pivot_df = grouped_df.pivot(index='Month', columns='Region', values='Mortality_Rate').reset_index()

# Replace the 'Month' values with month names
pivot_df['Month'] = pivot_df['Month'].map(month_mapping)

# Convert the 'Month' column to a categorical type with the custom order
pivot_df['Month'] = pd.Categorical(pivot_df['Month'], categories=custom_order, ordered=True)

# Sort the dataframe based on the custom order
pivot_df = pivot_df.sort_values('Month')

# Add national mortality rates
pivot_df['National'] = monthly_mortality_rates['Mortality_Rate']

# Add regional mortality rates
totals_row = pd.DataFrame({
    'Month': ['Yearly Avg'],
    'Centro Este': [regional_mortality_rates['Mortality_Rate'].iloc[0]],
    'Centro Occidente': [regional_mortality_rates['Mortality_Rate'].iloc[1]],
    'Centro Sur': [regional_mortality_rates['Mortality_Rate'].iloc[2]],
    'Noreste': [regional_mortality_rates['Mortality_Rate'].iloc[3]],
    'Noroeste': [regional_mortality_rates['Mortality_Rate'].iloc[4]],
    'Norte': [regional_mortality_rates['Mortality_Rate'].iloc[5]],
    'Sur': [regional_mortality_rates['Mortality_Rate'].iloc[6]],
    'Yucatan': [regional_mortality_rates['Mortality_Rate'].iloc[7]],
    'National': [national_mortality_rate]
})

pivot_df = pd.concat([pivot_df, totals_row], ignore_index=True)

# Rename columns (For visualization purposes only)
pivot_df.rename(columns={
    'Centro Este': 'East-central',
    'Centro Occidente': 'West-central',
    'Centro Sur': 'South-central',
    'Noreste': 'Northeast',
    'Noroeste': 'Northwest',
    'Norte': 'North',
    'Sur': 'South',
    'Yucatan': 'Yucatan peninsula',
    }, inplace=True)

# Rename ci_df Region values for consistency
ci_df['Region'] = ci_df['Region'].replace({
    'Centro Este': 'East-central',
    'Centro Occidente': 'West-central',
    'Centro Sur': 'South-central',
    'Noreste': 'Northeast',
    'Noroeste': 'Northwest',
    'Norte': 'North',
    'Sur': 'South',
    'Yucatan': 'Yucatan peninsula',
    })

pivot_df.loc[:, pivot_df.columns != 'Month'] = pivot_df.loc[:, pivot_df.columns != 'Month'].round(2)

pivot_df.head(13)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Month</th>
      <th>East-central</th>
      <th>West-central</th>
      <th>South-central</th>
      <th>Northeast</th>
      <th>Northwest</th>
      <th>North</th>
      <th>South</th>
      <th>Yucatan peninsula</th>
      <th>National</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>July</td>
      <td>0.69</td>
      <td>0.51</td>
      <td>0.85</td>
      <td>0.52</td>
      <td>0.67</td>
      <td>0.64</td>
      <td>0.95</td>
      <td>0.78</td>
      <td>0.69</td>
    </tr>
    <tr>
      <th>1</th>
      <td>August</td>
      <td>0.74</td>
      <td>0.59</td>
      <td>0.73</td>
      <td>0.36</td>
      <td>0.75</td>
      <td>0.77</td>
      <td>0.97</td>
      <td>0.70</td>
      <td>0.71</td>
    </tr>
    <tr>
      <th>2</th>
      <td>September</td>
      <td>0.80</td>
      <td>0.73</td>
      <td>1.02</td>
      <td>0.52</td>
      <td>0.91</td>
      <td>0.86</td>
      <td>1.02</td>
      <td>1.03</td>
      <td>0.84</td>
    </tr>
    <tr>
      <th>3</th>
      <td>October</td>
      <td>0.96</td>
      <td>0.71</td>
      <td>0.86</td>
      <td>0.52</td>
      <td>0.72</td>
      <td>0.79</td>
      <td>1.05</td>
      <td>1.22</td>
      <td>0.86</td>
    </tr>
    <tr>
      <th>4</th>
      <td>November</td>
      <td>0.84</td>
      <td>0.71</td>
      <td>0.92</td>
      <td>0.63</td>
      <td>0.79</td>
      <td>0.93</td>
      <td>0.93</td>
      <td>0.88</td>
      <td>0.83</td>
    </tr>
    <tr>
      <th>5</th>
      <td>December</td>
      <td>0.72</td>
      <td>0.62</td>
      <td>0.86</td>
      <td>0.58</td>
      <td>0.78</td>
      <td>0.60</td>
      <td>0.98</td>
      <td>0.60</td>
      <td>0.72</td>
    </tr>
    <tr>
      <th>6</th>
      <td>January</td>
      <td>0.49</td>
      <td>0.52</td>
      <td>0.83</td>
      <td>0.51</td>
      <td>0.52</td>
      <td>0.59</td>
      <td>0.95</td>
      <td>0.73</td>
      <td>0.60</td>
    </tr>
    <tr>
      <th>7</th>
      <td>February</td>
      <td>0.52</td>
      <td>0.45</td>
      <td>0.75</td>
      <td>0.39</td>
      <td>0.69</td>
      <td>0.64</td>
      <td>1.11</td>
      <td>0.84</td>
      <td>0.63</td>
    </tr>
    <tr>
      <th>8</th>
      <td>March</td>
      <td>0.43</td>
      <td>0.42</td>
      <td>0.45</td>
      <td>0.36</td>
      <td>0.32</td>
      <td>0.62</td>
      <td>1.08</td>
      <td>0.62</td>
      <td>0.53</td>
    </tr>
    <tr>
      <th>9</th>
      <td>April</td>
      <td>0.51</td>
      <td>0.50</td>
      <td>0.65</td>
      <td>0.45</td>
      <td>0.55</td>
      <td>0.41</td>
      <td>0.96</td>
      <td>0.45</td>
      <td>0.57</td>
    </tr>
    <tr>
      <th>10</th>
      <td>May</td>
      <td>0.57</td>
      <td>0.51</td>
      <td>0.77</td>
      <td>0.49</td>
      <td>0.44</td>
      <td>0.50</td>
      <td>1.21</td>
      <td>0.77</td>
      <td>0.65</td>
    </tr>
    <tr>
      <th>11</th>
      <td>June</td>
      <td>0.69</td>
      <td>0.47</td>
      <td>0.70</td>
      <td>0.58</td>
      <td>0.60</td>
      <td>0.58</td>
      <td>1.03</td>
      <td>1.03</td>
      <td>0.68</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Yearly Avg</td>
      <td>0.67</td>
      <td>0.57</td>
      <td>0.79</td>
      <td>0.50</td>
      <td>0.66</td>
      <td>0.67</td>
      <td>1.02</td>
      <td>0.82</td>
      <td>0.70</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Filter out the 'Yearly Avg' row
filtered_df = pivot_df[pivot_df['Month'] != 'Yearly Avg']

# Define the list of regions to be plotted
regions = [
    'East-central', 'West-central', 'South-central', 'Northeast',
    'Northwest', 'North', 'South', 'Yucatan peninsula'
]

# Set up the figure and axes for 4 rows and 2 columns of subplots
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15, 20))
axes = axes.flatten()

# Plot each region in a separate subplot
for i, region in enumerate(regions):
    ax = axes[i]
    sub = ci_df[ci_df['Region'] == region]

    # plot shaded 95% CI band
    ax.fill_between(
        sub['Month'],
        sub['CI_Lower'],
        sub['CI_Upper'],
        color='lightgray',
        alpha=0.3,
        label='95% CI'
    )

    # Plot the region's mortality rate
    ax.plot(
        filtered_df['Month'], 
        filtered_df[region], 
        label=f'{region} region', 
        linestyle='-', 
        color='black' if grayscale else 'orange'
        )
    
    # Plot the national average for comparison
    ax.plot(
        filtered_df['Month'], 
        filtered_df['National'], 
        label='National mean', 
        linestyle=(0, (1, 3)), 
        color='black' if grayscale else 'blue'
        )
    
    # Set title and labels for each subplot
    ax.set_title(f'{region} region')
    #ax.set_xlabel('Month')
    ax.set_ylabel('Mortality Rate (deaths per 1,000 births)')
    ax.set_xticks(filtered_df['Month'])
    ax.set_xticklabels(filtered_df['Month'], rotation=45, ha='right')
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add grid and legend
    ax.grid(axis='y')
    ax.legend()

# Adjust layout to prevent overlapping
plt.tight_layout()

plt.savefig(f'{plots_folder}/Fig3.{image_format}', format=image_format, dpi=1200)

# Show the plot
plt.show()
```
    
![png](nbconvert_output/output_54_1.png)
    


**Table 3.** Monthly mortality rates based on the occurence month.


```python
# Registered deaths grouped by year/month of birth 
grouped_deaths_df1 = respiratory_illness_df1.groupby('MES_OCURR').size().reset_index(name='Deaths_Birth')
# Register births grouped by year/month
grouped_births_df = birth_records_df.groupby('MES_NAC').size().reset_index(name='Births')


grouped_df = pd.merge(grouped_births_df, grouped_deaths_df1, left_on='MES_NAC', right_on='MES_OCURR', how='inner')

grouped_df.drop(columns=['MES_OCURR'], inplace=True)
grouped_df['Mortality_Rate_Birth'] =  (grouped_df['Deaths_Birth'] / grouped_df['Births'] *  1000).apply(lambda x: round(x, 2))

# Replace the 'MES_NAC' values with month names
grouped_df['MES_NAC'] = grouped_df['MES_NAC'].map(month_mapping)

# Convert the 'MES_NAC' column to a categorical type with the custom order
grouped_df['MES_NAC'] = pd.Categorical(grouped_df['MES_NAC'], categories=occurence_custom_order, ordered=True)

# Sort the dataframe based on the custom order
grouped_df = grouped_df.sort_values('MES_NAC')

grouped_df.rename(columns={'MES_NAC': 'Month', 'Mortality_Rate_Birth': 'Mortality_Rate'}, inplace=True)
monthly_mortality_rates = grouped_df[['Month', 'Mortality_Rate']]
monthly_mortality_rates.head(12)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Month</th>
      <th>Mortality_Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>April</td>
      <td>0.62</td>
    </tr>
    <tr>
      <th>4</th>
      <td>May</td>
      <td>0.54</td>
    </tr>
    <tr>
      <th>5</th>
      <td>June</td>
      <td>0.47</td>
    </tr>
    <tr>
      <th>6</th>
      <td>July</td>
      <td>0.53</td>
    </tr>
    <tr>
      <th>7</th>
      <td>August</td>
      <td>0.43</td>
    </tr>
    <tr>
      <th>8</th>
      <td>September</td>
      <td>0.46</td>
    </tr>
    <tr>
      <th>9</th>
      <td>October</td>
      <td>0.58</td>
    </tr>
    <tr>
      <th>10</th>
      <td>November</td>
      <td>0.85</td>
    </tr>
    <tr>
      <th>11</th>
      <td>December</td>
      <td>1.09</td>
    </tr>
    <tr>
      <th>0</th>
      <td>January</td>
      <td>1.11</td>
    </tr>
    <tr>
      <th>1</th>
      <td>February</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>March</td>
      <td>0.81</td>
    </tr>
  </tbody>
</table>
</div>



**Table 4.** Respiratory infant mortality rates by occurence month in Mexican children born between April 2014 and March 2020.


```python
# Registered deaths grouped by year/month
grouped_deaths_df = respiratory_illness_df1.groupby(['ENT_RESID','MES_OCURR']).size().reset_index(name='Deaths')
grouped_deaths_df = pd.merge(grouped_deaths_df, states_df, left_on='ENT_RESID', right_on='ENTIDAD', how='inner')

# Register births grouped by year/month
grouped_births_df = birth_records_df.groupby(['ENT_RESID','MES_NAC']).size().reset_index(name='Births')
grouped_births_df = pd.merge(grouped_births_df, states_df, left_on='ENT_RESID', right_on='ENTIDAD', how='inner')

grouped_df = pd.merge(grouped_births_df, grouped_deaths_df, left_on=['ENT_RESID', 'MES_NAC'], right_on=['ENT_RESID', 'MES_OCURR'], how='left')
grouped_df.drop(columns=['ENTIDAD_x','MES_OCURR','ENTIDAD_y','NOM_ENTIDAD_y', 'REGION_y'], inplace=True)
grouped_df.rename(columns={'REGION_x': 'Region', 'NOM_ENTIDAD_x': 'State', 'MES_NAC': 'Month'}, inplace=True)
grouped_df['Deaths'] = grouped_df['Deaths'].fillna(0)

grouped_df = grouped_df.groupby(['Region', 'Month'])[['Births', 'Deaths']].sum().reset_index()
r, lo, hi = rate_and_ci(grouped_df['Deaths'].values, grouped_df['Births'].values)
grouped_df['Mortality_Rate'] =  r

pivot_df = grouped_df.pivot(index='Month', columns='Region', values='Mortality_Rate').reset_index()

# Replace the 'Month' values with month names
pivot_df['Month'] = pivot_df['Month'].map(month_mapping)

# Convert the 'Month' column to a categorical type with the custom order
pivot_df['Month'] = pd.Categorical(pivot_df['Month'], categories=occurence_custom_order, ordered=True)

# Sort the dataframe based on the custom order
pivot_df = pivot_df.sort_values('Month')

# Add national mortality rates
pivot_df['National'] = monthly_mortality_rates['Mortality_Rate']

# Rename columns (For visualization purposes only)
pivot_df.rename(columns={
    'Centro Este': 'East-central',
    'Centro Occidente': 'West-central',
    'Centro Sur': 'South-central',
    'Noreste': 'Northeast',
    'Noroeste': 'Northwest',
    'Norte': 'North',
    'Sur': 'South',
    'Yucatan': 'Yucatan peninsula',
    }, inplace=True)

pivot_df.loc[:, pivot_df.columns != 'Month'] = pivot_df.loc[:, pivot_df.columns != 'Month'].round(2)

pivot_df.head(13)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Region</th>
      <th>Month</th>
      <th>East-central</th>
      <th>West-central</th>
      <th>South-central</th>
      <th>Northeast</th>
      <th>Northwest</th>
      <th>North</th>
      <th>South</th>
      <th>Yucatan peninsula</th>
      <th>National</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>April</td>
      <td>0.45</td>
      <td>0.56</td>
      <td>0.67</td>
      <td>0.54</td>
      <td>0.72</td>
      <td>0.53</td>
      <td>1.04</td>
      <td>0.80</td>
      <td>0.62</td>
    </tr>
    <tr>
      <th>4</th>
      <td>May</td>
      <td>0.39</td>
      <td>0.50</td>
      <td>0.75</td>
      <td>0.36</td>
      <td>0.67</td>
      <td>0.38</td>
      <td>0.96</td>
      <td>0.69</td>
      <td>0.54</td>
    </tr>
    <tr>
      <th>5</th>
      <td>June</td>
      <td>0.42</td>
      <td>0.35</td>
      <td>0.57</td>
      <td>0.25</td>
      <td>0.37</td>
      <td>0.34</td>
      <td>0.93</td>
      <td>0.68</td>
      <td>0.47</td>
    </tr>
    <tr>
      <th>6</th>
      <td>July</td>
      <td>0.46</td>
      <td>0.43</td>
      <td>0.57</td>
      <td>0.40</td>
      <td>0.38</td>
      <td>0.39</td>
      <td>1.03</td>
      <td>0.78</td>
      <td>0.53</td>
    </tr>
    <tr>
      <th>7</th>
      <td>August</td>
      <td>0.36</td>
      <td>0.33</td>
      <td>0.58</td>
      <td>0.37</td>
      <td>0.28</td>
      <td>0.36</td>
      <td>0.80</td>
      <td>0.48</td>
      <td>0.43</td>
    </tr>
    <tr>
      <th>8</th>
      <td>September</td>
      <td>0.40</td>
      <td>0.41</td>
      <td>0.55</td>
      <td>0.19</td>
      <td>0.36</td>
      <td>0.36</td>
      <td>0.77</td>
      <td>0.80</td>
      <td>0.46</td>
    </tr>
    <tr>
      <th>9</th>
      <td>October</td>
      <td>0.59</td>
      <td>0.39</td>
      <td>0.63</td>
      <td>0.37</td>
      <td>0.46</td>
      <td>0.53</td>
      <td>0.96</td>
      <td>0.90</td>
      <td>0.58</td>
    </tr>
    <tr>
      <th>10</th>
      <td>November</td>
      <td>0.92</td>
      <td>0.63</td>
      <td>0.88</td>
      <td>0.59</td>
      <td>0.68</td>
      <td>0.70</td>
      <td>1.28</td>
      <td>1.12</td>
      <td>0.85</td>
    </tr>
    <tr>
      <th>11</th>
      <td>December</td>
      <td>1.29</td>
      <td>0.77</td>
      <td>1.10</td>
      <td>0.70</td>
      <td>1.03</td>
      <td>1.36</td>
      <td>1.12</td>
      <td>0.92</td>
      <td>1.09</td>
    </tr>
    <tr>
      <th>0</th>
      <td>January</td>
      <td>1.09</td>
      <td>0.95</td>
      <td>1.20</td>
      <td>0.96</td>
      <td>1.17</td>
      <td>1.39</td>
      <td>1.18</td>
      <td>0.84</td>
      <td>1.11</td>
    </tr>
    <tr>
      <th>1</th>
      <td>February</td>
      <td>1.03</td>
      <td>0.90</td>
      <td>1.22</td>
      <td>0.85</td>
      <td>0.86</td>
      <td>1.06</td>
      <td>1.07</td>
      <td>0.87</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>March</td>
      <td>0.69</td>
      <td>0.71</td>
      <td>1.04</td>
      <td>0.52</td>
      <td>1.00</td>
      <td>0.75</td>
      <td>1.14</td>
      <td>0.96</td>
      <td>0.81</td>
    </tr>
  </tbody>
</table>
</div>




```python


# Calculate confidence intervals
ci_df = grouped_df.groupby(['Region', 'Month'])[['Births', 'Deaths']].sum().reset_index()
ci_df['Rate'], ci_df['CI_Lower'], ci_df['CI_Upper'] = rate_and_ci(ci_df['Deaths'], ci_df['Births'])

# merge with month names and order
ci_df['Month'] = ci_df['Month'].map(month_mapping)
ci_df['Month'] = pd.Categorical(ci_df['Month'], categories=occurence_custom_order, ordered=True)
ci_df = ci_df.sort_values(['Region', 'Month'])

# Rename ci_df Region values for consistency
ci_df['Region'] = ci_df['Region'].replace({
    'Centro Este': 'East-central',
    'Centro Occidente': 'West-central',
    'Centro Sur': 'South-central',
    'Noreste': 'Northeast',
    'Noroeste': 'Northwest',
    'Norte': 'North',
    'Sur': 'South',
    'Yucatan': 'Yucatan peninsula',
    })

ci_df.head(12)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Region</th>
      <th>Month</th>
      <th>Births</th>
      <th>Deaths</th>
      <th>Rate</th>
      <th>CI_Lower</th>
      <th>CI_Upper</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>East-central</td>
      <td>April</td>
      <td>318596</td>
      <td>144</td>
      <td>0.45</td>
      <td>0.38</td>
      <td>0.52</td>
    </tr>
    <tr>
      <th>4</th>
      <td>East-central</td>
      <td>May</td>
      <td>327003</td>
      <td>129</td>
      <td>0.39</td>
      <td>0.32</td>
      <td>0.46</td>
    </tr>
    <tr>
      <th>5</th>
      <td>East-central</td>
      <td>June</td>
      <td>315566</td>
      <td>131</td>
      <td>0.42</td>
      <td>0.35</td>
      <td>0.49</td>
    </tr>
    <tr>
      <th>6</th>
      <td>East-central</td>
      <td>July</td>
      <td>331212</td>
      <td>153</td>
      <td>0.46</td>
      <td>0.39</td>
      <td>0.53</td>
    </tr>
    <tr>
      <th>7</th>
      <td>East-central</td>
      <td>August</td>
      <td>340441</td>
      <td>124</td>
      <td>0.36</td>
      <td>0.30</td>
      <td>0.42</td>
    </tr>
    <tr>
      <th>8</th>
      <td>East-central</td>
      <td>September</td>
      <td>347675</td>
      <td>139</td>
      <td>0.40</td>
      <td>0.33</td>
      <td>0.47</td>
    </tr>
    <tr>
      <th>9</th>
      <td>East-central</td>
      <td>October</td>
      <td>341278</td>
      <td>202</td>
      <td>0.59</td>
      <td>0.51</td>
      <td>0.67</td>
    </tr>
    <tr>
      <th>10</th>
      <td>East-central</td>
      <td>November</td>
      <td>316282</td>
      <td>292</td>
      <td>0.92</td>
      <td>0.81</td>
      <td>1.03</td>
    </tr>
    <tr>
      <th>11</th>
      <td>East-central</td>
      <td>December</td>
      <td>325198</td>
      <td>418</td>
      <td>1.29</td>
      <td>1.17</td>
      <td>1.41</td>
    </tr>
    <tr>
      <th>0</th>
      <td>East-central</td>
      <td>January</td>
      <td>315165</td>
      <td>345</td>
      <td>1.09</td>
      <td>0.97</td>
      <td>1.21</td>
    </tr>
    <tr>
      <th>1</th>
      <td>East-central</td>
      <td>February</td>
      <td>284037</td>
      <td>293</td>
      <td>1.03</td>
      <td>0.91</td>
      <td>1.15</td>
    </tr>
    <tr>
      <th>2</th>
      <td>East-central</td>
      <td>March</td>
      <td>315804</td>
      <td>219</td>
      <td>0.69</td>
      <td>0.60</td>
      <td>0.78</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Filter out the 'Yearly Avg' row
filtered_df = pivot_df[pivot_df['Month'] != 'Yearly Avg']

# Define the list of regions to be plotted
regions = [
    'East-central', 'West-central', 'South-central', 'Northeast',
    'Northwest', 'North', 'South', 'Yucatan peninsula'
]

# Set up the figure and axes for 4 rows and 2 columns of subplots
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15, 20))
axes = axes.flatten()

# Plot each region in a separate subplot
for i, region in enumerate(regions):
    ax = axes[i]
    sub = ci_df[ci_df['Region'] == region]

    # plot shaded 95% CI band
    ax.fill_between(
        sub['Month'],
        sub['CI_Lower'],
        sub['CI_Upper'],
        color='lightgray',
        alpha=0.3,
        label='95% CI'
    )

    # Plot the region's mortality rate
    ax.plot(
        filtered_df['Month'], 
        filtered_df[region], 
        label=f'{region} region', 
        linestyle='-', 
        color='black' if grayscale else 'orange'
        )
    
    # Plot the national average for comparison
    ax.plot(
        filtered_df['Month'], 
        filtered_df['National'], 
        label='National mean', 
        linestyle=(0, (1, 3)), 
        color='black' if grayscale else 'blue'
        )
    
    # Set title and labels for each subplot
    ax.set_title(f'{region} region')
    #ax.set_xlabel('Month')
    ax.set_ylabel('Mortality Rate (deaths per 1,000 births)')
    ax.set_xticks(filtered_df['Month'])
    ax.set_xticklabels(filtered_df['Month'], rotation=45, ha='right')
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add grid and legend
    ax.grid(axis='y')
    ax.legend()

# Adjust layout to prevent overlapping
plt.tight_layout()

plt.savefig(f'{plots_folder}/Fig9.{image_format}', format=image_format, dpi=1200)

# Show the plot
plt.show()
```
    
![png](nbconvert_output/output_60_1.png)
    


## Characteristics of infants born between April 2014 and March 2021


```python
births_df = birth_records_df

births_df['EDAD_MADN'] = births_df['EDAD_MADN'].astype(float)
births_df['EDAD_MADN'] = births_df['EDAD_MADN'].replace(99.0, np.nan)

births_df['EDAD_PADN'] = births_df['EDAD_PADN'].astype(float)
births_df['EDAD_PADN'] = births_df['EDAD_PADN'].replace(99.0, np.nan)

births_df['ORDEN_PART'] = births_df['ORDEN_PART'].astype(float)
births_df['ORDEN_PART'] = births_df['ORDEN_PART'].replace(99.0, np.nan)

births_df['TIPO_NAC'] = births_df['TIPO_NAC'].astype(float)
births_df['TIPO_NAC'] = births_df['TIPO_NAC'].replace(9.0, np.nan)

births_df['LUGAR_PART'] = births_df['LUGAR_PART'].astype(float)
births_df['LUGAR_PART'] = births_df['LUGAR_PART'].replace(9.0, np.nan)

births_df['EDOCIV_MAD'] = births_df['EDOCIV_MAD'].astype(float)
births_df['EDOCIV_MAD'] = births_df['EDOCIV_MAD'].replace(9.0, np.nan)

births_df['ESCOL_MAD'] = births_df['ESCOL_MAD'].astype(float)
births_df['ESCOL_MAD'] = births_df['ESCOL_MAD'].replace(9.0, np.nan)

births_df['ESCOL_PAD'] = births_df['ESCOL_PAD'].astype(float)
births_df['ESCOL_PAD'] = births_df['ESCOL_PAD'].replace(9.0, np.nan)
```

Helper function: Summarize the total Xs being part of the Y. It can return the total matches or just the percentage.


```python
def Is_Equal_To(x, y, usePercentage):
    if isinstance(y, (list, set, pd.Series)):  # Check if y is a list, set, or Series
        counter = x.isin(y).sum()
    elif pd.isna(y):  # Check if y is NaN
        counter = x.isna().sum()
    else:  # For scalar y
        counter = (x == y).sum()

    if usePercentage:
        total_count = len(x)
        return (counter / total_count) * 100 if total_count > 0 else 0
    else:
        return counter
```


```python
aggregations = {
    'Total': ('MES_NAC', lambda x: x.notna().sum()),
    'Maternal_Age_Median': ('EDAD_MADN', lambda x: x.median(skipna=True)),
    'Paternal_Age_Median': ('EDAD_PADN', lambda x: x.median(skipna=True)),
    'Pregnancy_Number_Median': ('ORDEN_PART', lambda x: x.median(skipna=True)),
    'Singleton_Pregnancy': ('TIPO_NAC', lambda x: Is_Equal_To(x, 1, False)),
    'Singleton_Pregnancy_Percentage': ('TIPO_NAC', lambda x: round(Is_Equal_To(x, 1, True), 2)),
    'Hospital_Birth': ('LUGAR_PART', lambda x: Is_Equal_To(x, {1, 2}, False)),
    'Hospital_Birth_Percentage': ('LUGAR_PART', lambda x: round(Is_Equal_To(x, {1, 2}, True), 2)),
    'All_Specified_Birth_Place': ('LUGAR_PART', lambda x: Is_Equal_To(x, {1, 2, 3, 4}, False)),
    'Single_Mother': ('EDOCIV_MAD', lambda x: Is_Equal_To(x, 1., False)),
    'Single_Mother_Percentage': ('EDOCIV_MAD', lambda x: round(Is_Equal_To(x, 1, True), 2)),
    'Single_Divorced_Separated_Widow': ('EDOCIV_MAD', lambda x: Is_Equal_To(x, {1, 4, 5, 6}, False)),
    'Single_Divorced_Separated_Widow_Percentage': ('EDOCIV_MAD', lambda x: round(Is_Equal_To(x, {1, 4, 5, 6}, True), 2)),
    'All_Specified_Marital_Status': ('EDOCIV_MAD', lambda x: Is_Equal_To(x, {1, 2, 3, 4, 5, 6}, False))
}

grouped_df = births_df.groupby('MES_NAC').agg(**aggregations).reset_index()

grouped_df['Hospital_Birth_From_Specified'] = (grouped_df['Hospital_Birth'] / grouped_df['All_Specified_Birth_Place'] *  100).apply(lambda x: round(x, 2))
grouped_df['Single_Mother_From_Specified'] = (grouped_df['Single_Mother'] / grouped_df['All_Specified_Marital_Status'] *  100).apply(lambda x: round(x, 2))
grouped_df['Single_Divorced_Separated_Widow_From_Specified'] = (grouped_df['Single_Divorced_Separated_Widow'] / grouped_df['All_Specified_Marital_Status'] *  100).apply(lambda x: round(x, 2))
# Convert Maternal Age, Paternal Age, and Pregnancy number to Integers.
grouped_df['Maternal_Age_Median'] = grouped_df['Maternal_Age_Median'].astype(int)
grouped_df['Paternal_Age_Median'] = grouped_df['Paternal_Age_Median'].astype(int)
grouped_df['Pregnancy_Number_Median'] = grouped_df['Pregnancy_Number_Median'].astype(int)

grouped_df.rename(columns={'MES_NAC': 'Month'}, inplace=True)

# Replace the 'Month' values with month names
grouped_df['Month'] = grouped_df['Month'].map(month_mapping)

# Convert the 'Month' column to a categorical type with the custom order
grouped_df['Month'] = pd.Categorical(grouped_df['Month'], categories=custom_order, ordered=True)

# Sort the dataframe based on the custom order
grouped_df = grouped_df.sort_values('Month')
```


```python
not_specified_birth_place =  (grouped_df['Total'].sum()) - (grouped_df['All_Specified_Birth_Place'].sum())
not_specified_birth_place_precentage = round(not_specified_birth_place / grouped_df['Total'].sum() *  100, 2)

print(f'Birth place not specified:  {not_specified_birth_place:,} ({not_specified_birth_place_precentage}%)')
```

    Birth place not specified:  615,775 (4.89%)



```python
not_specified_marital_status =  (grouped_df['Total'].sum()) - (grouped_df['All_Specified_Marital_Status'].sum())
not_specified_marital_status_precentage = round(not_specified_marital_status / grouped_df['Total'].sum() *  100, 2)

print(f'Marital status not specified:  {not_specified_marital_status:,} ({not_specified_marital_status_precentage}%)')
```

    Marital status not specified:  737,473 (5.85%)



```python
grouped_df = grouped_df[['Month', 'Total', 'Maternal_Age_Median', 'Paternal_Age_Median', 'Pregnancy_Number_Median', 'Singleton_Pregnancy',
                         'Singleton_Pregnancy_Percentage', 'Hospital_Birth', 'Hospital_Birth_Percentage', 'Hospital_Birth_From_Specified',
                         'Single_Mother', 'Single_Mother_Percentage', 'Single_Mother_From_Specified', 'Single_Divorced_Separated_Widow',
                         'Single_Divorced_Separated_Widow_Percentage', 'Single_Divorced_Separated_Widow_From_Specified']]

grouped_df.head(12)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Month</th>
      <th>Total</th>
      <th>Maternal_Age_Median</th>
      <th>Paternal_Age_Median</th>
      <th>Pregnancy_Number_Median</th>
      <th>Singleton_Pregnancy</th>
      <th>Singleton_Pregnancy_Percentage</th>
      <th>Hospital_Birth</th>
      <th>Hospital_Birth_Percentage</th>
      <th>Hospital_Birth_From_Specified</th>
      <th>Single_Mother</th>
      <th>Single_Mother_Percentage</th>
      <th>Single_Mother_From_Specified</th>
      <th>Single_Divorced_Separated_Widow</th>
      <th>Single_Divorced_Separated_Widow_Percentage</th>
      <th>Single_Divorced_Separated_Widow_From_Specified</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>July</td>
      <td>1076970</td>
      <td>25</td>
      <td>28</td>
      <td>2</td>
      <td>1061802</td>
      <td>98.59</td>
      <td>981287</td>
      <td>91.12</td>
      <td>95.80</td>
      <td>128979</td>
      <td>11.98</td>
      <td>12.73</td>
      <td>133179</td>
      <td>12.37</td>
      <td>13.15</td>
    </tr>
    <tr>
      <th>7</th>
      <td>August</td>
      <td>1136385</td>
      <td>25</td>
      <td>28</td>
      <td>2</td>
      <td>1120335</td>
      <td>98.59</td>
      <td>1037757</td>
      <td>91.32</td>
      <td>95.97</td>
      <td>136162</td>
      <td>11.98</td>
      <td>12.73</td>
      <td>140630</td>
      <td>12.38</td>
      <td>13.15</td>
    </tr>
    <tr>
      <th>8</th>
      <td>September</td>
      <td>1181923</td>
      <td>25</td>
      <td>28</td>
      <td>2</td>
      <td>1166299</td>
      <td>98.68</td>
      <td>1079203</td>
      <td>91.31</td>
      <td>95.94</td>
      <td>139359</td>
      <td>11.79</td>
      <td>12.51</td>
      <td>144003</td>
      <td>12.18</td>
      <td>12.92</td>
    </tr>
    <tr>
      <th>9</th>
      <td>October</td>
      <td>1153400</td>
      <td>25</td>
      <td>28</td>
      <td>2</td>
      <td>1137830</td>
      <td>98.65</td>
      <td>1051994</td>
      <td>91.21</td>
      <td>95.99</td>
      <td>136668</td>
      <td>11.85</td>
      <td>12.58</td>
      <td>141181</td>
      <td>12.24</td>
      <td>13.00</td>
    </tr>
    <tr>
      <th>10</th>
      <td>November</td>
      <td>1056550</td>
      <td>25</td>
      <td>28</td>
      <td>2</td>
      <td>1043065</td>
      <td>98.72</td>
      <td>960163</td>
      <td>90.88</td>
      <td>95.84</td>
      <td>127611</td>
      <td>12.08</td>
      <td>12.86</td>
      <td>131599</td>
      <td>12.46</td>
      <td>13.26</td>
    </tr>
    <tr>
      <th>11</th>
      <td>December</td>
      <td>1067736</td>
      <td>25</td>
      <td>28</td>
      <td>2</td>
      <td>1053562</td>
      <td>98.67</td>
      <td>968285</td>
      <td>90.69</td>
      <td>95.63</td>
      <td>128483</td>
      <td>12.03</td>
      <td>12.83</td>
      <td>132436</td>
      <td>12.40</td>
      <td>13.23</td>
    </tr>
    <tr>
      <th>0</th>
      <td>January</td>
      <td>1019441</td>
      <td>25</td>
      <td>28</td>
      <td>2</td>
      <td>1005850</td>
      <td>98.67</td>
      <td>926037</td>
      <td>90.84</td>
      <td>95.50</td>
      <td>124676</td>
      <td>12.23</td>
      <td>13.02</td>
      <td>128456</td>
      <td>12.60</td>
      <td>13.41</td>
    </tr>
    <tr>
      <th>1</th>
      <td>February</td>
      <td>902558</td>
      <td>25</td>
      <td>28</td>
      <td>2</td>
      <td>890437</td>
      <td>98.66</td>
      <td>822025</td>
      <td>91.08</td>
      <td>95.62</td>
      <td>110659</td>
      <td>12.26</td>
      <td>13.01</td>
      <td>114130</td>
      <td>12.65</td>
      <td>13.42</td>
    </tr>
    <tr>
      <th>2</th>
      <td>March</td>
      <td>986549</td>
      <td>25</td>
      <td>28</td>
      <td>2</td>
      <td>972836</td>
      <td>98.61</td>
      <td>897612</td>
      <td>90.99</td>
      <td>95.56</td>
      <td>119649</td>
      <td>12.13</td>
      <td>12.88</td>
      <td>123348</td>
      <td>12.50</td>
      <td>13.28</td>
    </tr>
    <tr>
      <th>3</th>
      <td>April</td>
      <td>994946</td>
      <td>25</td>
      <td>28</td>
      <td>2</td>
      <td>981828</td>
      <td>98.68</td>
      <td>904015</td>
      <td>90.86</td>
      <td>95.42</td>
      <td>117510</td>
      <td>11.81</td>
      <td>12.51</td>
      <td>121299</td>
      <td>12.19</td>
      <td>12.92</td>
    </tr>
    <tr>
      <th>4</th>
      <td>May</td>
      <td>1024246</td>
      <td>25</td>
      <td>28</td>
      <td>2</td>
      <td>1010753</td>
      <td>98.68</td>
      <td>932777</td>
      <td>91.07</td>
      <td>95.56</td>
      <td>121184</td>
      <td>11.83</td>
      <td>12.53</td>
      <td>125060</td>
      <td>12.21</td>
      <td>12.93</td>
    </tr>
    <tr>
      <th>5</th>
      <td>June</td>
      <td>1004198</td>
      <td>25</td>
      <td>28</td>
      <td>2</td>
      <td>990491</td>
      <td>98.64</td>
      <td>915144</td>
      <td>91.13</td>
      <td>95.71</td>
      <td>119410</td>
      <td>11.89</td>
      <td>12.61</td>
      <td>123340</td>
      <td>12.28</td>
      <td>13.03</td>
    </tr>
  </tbody>
</table>
</div>




```python
aggregations = {
    'Total': ('MES_NAC', lambda x: x.notna().sum()),
    'Mother_Without_Education_Level': ('ESCOL_MAD', lambda x: Is_Equal_To(x, 1, False)),
    'Percentage_Mother_Without_Education_Level': ('ESCOL_MAD', lambda x: round(Is_Equal_To(x, 1, True), 2)),
    'Mother_With_Elementary_School': ('ESCOL_MAD', lambda x: Is_Equal_To(x, {2, 3, 4}, False)),
    'Percentage_Mother_With_Elementary_School': ('ESCOL_MAD', lambda x: round(Is_Equal_To(x, {2, 3, 4}, True), 2)),
    'Mother_With_Elementary_Or_Without_Education': ('ESCOL_MAD', lambda x: Is_Equal_To(x, {1, 2, 3, 4}, False)),
    'Percentage_Mother_With_Elementary_Or_Without_Education': ('ESCOL_MAD', lambda x: round(Is_Equal_To(x, {1, 2, 3, 4}, True), 2)),
    'Mother_All_Specified_Education_Level': ('ESCOL_MAD', lambda x: Is_Equal_To(x, {1, 2, 3, 4, 5, 6, 7, 8}, False)),

    'Father_Without_Education_Level': ('ESCOL_PAD', lambda x: Is_Equal_To(x, 1, False)),
    'Percentage_Father_Without_Education_Level': ('ESCOL_PAD', lambda x: round(Is_Equal_To(x, 1, True), 2)),
    'Father_With_Elementary_School': ('ESCOL_PAD', lambda x: Is_Equal_To(x, {2, 3, 4}, False)),
    'Percentage_Father_With_Elementary_School': ('ESCOL_PAD', lambda x: round(Is_Equal_To(x, {2, 3, 4}, True), 2)),
    'Father_With_Elementary_Or_Without_Education': ('ESCOL_PAD', lambda x: Is_Equal_To(x, {1, 2, 3, 4}, False)),
    'Percentage_Father_With_Elementary_Or_Without_Education': ('ESCOL_PAD', lambda x: round(Is_Equal_To(x, {1, 2, 3, 4}, True), 2)),
    'Father_All_Specified_Education_Level': ('ESCOL_PAD', lambda x: Is_Equal_To(x, {1, 2, 3, 4, 5, 6, 7, 8}, False))
}

grouped_df = births_df.groupby('MES_NAC').agg(**aggregations).reset_index()

grouped_df['Mother_Without_Education_From_Specified'] = (grouped_df['Mother_Without_Education_Level'] / grouped_df['Mother_All_Specified_Education_Level'] *  100).apply(lambda x: round(x, 2))
grouped_df['Father_Without_Education_From_Specified'] = (grouped_df['Father_Without_Education_Level'] / grouped_df['Father_All_Specified_Education_Level'] *  100).apply(lambda x: round(x, 2))

grouped_df['Mother_With_Elementary_From_Specified'] = (grouped_df['Mother_With_Elementary_School'] / grouped_df['Mother_All_Specified_Education_Level'] *  100).apply(lambda x: round(x, 2))
grouped_df['Father_With_Elementary_From_Specified'] = (grouped_df['Father_With_Elementary_School'] / grouped_df['Father_All_Specified_Education_Level'] *  100).apply(lambda x: round(x, 2))


grouped_df['Mother_With_Elementary_Or_Without_Education_From_Specified'] = (grouped_df['Mother_With_Elementary_Or_Without_Education'] / grouped_df['Mother_All_Specified_Education_Level'] *  100).apply(lambda x: round(x, 2))
grouped_df['Father_With_Elementary_Or_Without_Education_From_Specified'] = (grouped_df['Father_With_Elementary_Or_Without_Education'] / grouped_df['Father_All_Specified_Education_Level'] *  100).apply(lambda x: round(x, 2))

grouped_df.rename(columns={'MES_NAC': 'Month'}, inplace=True)

# Replace the 'Month' values with month names
grouped_df['Month'] = grouped_df['Month'].map(month_mapping)

# Convert the 'Month' column to a categorical type with the custom order
grouped_df['Month'] = pd.Categorical(grouped_df['Month'], categories=custom_order, ordered=True)

# Sort the dataframe based on the custom order
grouped_df = grouped_df.sort_values('Month')
```


```python
not_specified_education_level =  (grouped_df['Total'].sum()) - (grouped_df['Mother_All_Specified_Education_Level'].sum())
not_specified_education_level_precentage = round(not_specified_education_level / grouped_df['Total'].sum() *  100, 2)

print(f'No information regarding mother education level:  {not_specified_education_level:,} ({not_specified_education_level_precentage}%)')
```

    No information regarding mother education level:  838,813 (6.65%)



```python
not_specified_education_level =  (grouped_df['Total'].sum()) - (grouped_df['Father_All_Specified_Education_Level'].sum())
not_specified_education_level_precentage = round(not_specified_education_level / grouped_df['Total'].sum() *  100, 2)

print(f'No information regarding father education level:  {not_specified_education_level:,} ({not_specified_education_level_precentage}%)')
```

    No information regarding father education level:  1,855,792 (14.72%)



```python
grouped_df = grouped_df[['Month', 'Total', 'Mother_Without_Education_Level', 'Percentage_Mother_Without_Education_Level', 'Mother_Without_Education_From_Specified', 
                         'Mother_With_Elementary_School', 'Percentage_Mother_With_Elementary_School', 'Mother_With_Elementary_From_Specified',
                         'Mother_With_Elementary_Or_Without_Education', 'Percentage_Mother_With_Elementary_Or_Without_Education', 
                         'Mother_With_Elementary_Or_Without_Education_From_Specified',
                         'Father_Without_Education_Level', 'Percentage_Father_Without_Education_Level', 'Father_Without_Education_From_Specified',
                         'Father_With_Elementary_School', 'Percentage_Father_With_Elementary_School', 'Father_With_Elementary_From_Specified',
                         'Father_With_Elementary_Or_Without_Education', 'Percentage_Father_With_Elementary_Or_Without_Education', 
                         'Father_With_Elementary_Or_Without_Education_From_Specified']]

grouped_df.head(12)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Month</th>
      <th>Total</th>
      <th>Mother_Without_Education_Level</th>
      <th>Percentage_Mother_Without_Education_Level</th>
      <th>Mother_Without_Education_From_Specified</th>
      <th>Mother_With_Elementary_School</th>
      <th>Percentage_Mother_With_Elementary_School</th>
      <th>Mother_With_Elementary_From_Specified</th>
      <th>Mother_With_Elementary_Or_Without_Education</th>
      <th>Percentage_Mother_With_Elementary_Or_Without_Education</th>
      <th>Mother_With_Elementary_Or_Without_Education_From_Specified</th>
      <th>Father_Without_Education_Level</th>
      <th>Percentage_Father_Without_Education_Level</th>
      <th>Father_Without_Education_From_Specified</th>
      <th>Father_With_Elementary_School</th>
      <th>Percentage_Father_With_Elementary_School</th>
      <th>Father_With_Elementary_From_Specified</th>
      <th>Father_With_Elementary_Or_Without_Education</th>
      <th>Percentage_Father_With_Elementary_Or_Without_Education</th>
      <th>Father_With_Elementary_Or_Without_Education_From_Specified</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>July</td>
      <td>1076970</td>
      <td>20401</td>
      <td>1.89</td>
      <td>2.03</td>
      <td>183036</td>
      <td>17.00</td>
      <td>18.21</td>
      <td>203437</td>
      <td>18.89</td>
      <td>20.24</td>
      <td>19625</td>
      <td>1.82</td>
      <td>2.14</td>
      <td>187230</td>
      <td>17.38</td>
      <td>20.40</td>
      <td>206855</td>
      <td>19.21</td>
      <td>22.54</td>
    </tr>
    <tr>
      <th>7</th>
      <td>August</td>
      <td>1136385</td>
      <td>21375</td>
      <td>1.88</td>
      <td>2.01</td>
      <td>190590</td>
      <td>16.77</td>
      <td>17.96</td>
      <td>211965</td>
      <td>18.65</td>
      <td>19.97</td>
      <td>20751</td>
      <td>1.83</td>
      <td>2.14</td>
      <td>195167</td>
      <td>17.17</td>
      <td>20.14</td>
      <td>215918</td>
      <td>19.00</td>
      <td>22.28</td>
    </tr>
    <tr>
      <th>8</th>
      <td>September</td>
      <td>1181923</td>
      <td>22567</td>
      <td>1.91</td>
      <td>2.04</td>
      <td>197797</td>
      <td>16.74</td>
      <td>17.90</td>
      <td>220364</td>
      <td>18.64</td>
      <td>19.94</td>
      <td>21984</td>
      <td>1.86</td>
      <td>2.17</td>
      <td>203421</td>
      <td>17.21</td>
      <td>20.10</td>
      <td>225405</td>
      <td>19.07</td>
      <td>22.28</td>
    </tr>
    <tr>
      <th>9</th>
      <td>October</td>
      <td>1153400</td>
      <td>22212</td>
      <td>1.93</td>
      <td>2.06</td>
      <td>191135</td>
      <td>16.57</td>
      <td>17.75</td>
      <td>213347</td>
      <td>18.50</td>
      <td>19.82</td>
      <td>21631</td>
      <td>1.88</td>
      <td>2.19</td>
      <td>196657</td>
      <td>17.05</td>
      <td>19.95</td>
      <td>218288</td>
      <td>18.93</td>
      <td>22.15</td>
    </tr>
    <tr>
      <th>10</th>
      <td>November</td>
      <td>1056550</td>
      <td>20782</td>
      <td>1.97</td>
      <td>2.11</td>
      <td>176192</td>
      <td>16.68</td>
      <td>17.89</td>
      <td>196974</td>
      <td>18.64</td>
      <td>20.00</td>
      <td>20197</td>
      <td>1.91</td>
      <td>2.24</td>
      <td>181877</td>
      <td>17.21</td>
      <td>20.22</td>
      <td>202074</td>
      <td>19.13</td>
      <td>22.46</td>
    </tr>
    <tr>
      <th>11</th>
      <td>December</td>
      <td>1067736</td>
      <td>21404</td>
      <td>2.00</td>
      <td>2.15</td>
      <td>176522</td>
      <td>16.53</td>
      <td>17.77</td>
      <td>197926</td>
      <td>18.54</td>
      <td>19.92</td>
      <td>20685</td>
      <td>1.94</td>
      <td>2.28</td>
      <td>182922</td>
      <td>17.13</td>
      <td>20.15</td>
      <td>203607</td>
      <td>19.07</td>
      <td>22.42</td>
    </tr>
    <tr>
      <th>0</th>
      <td>January</td>
      <td>1019441</td>
      <td>20114</td>
      <td>1.97</td>
      <td>2.12</td>
      <td>170321</td>
      <td>16.71</td>
      <td>17.92</td>
      <td>190435</td>
      <td>18.68</td>
      <td>20.04</td>
      <td>19445</td>
      <td>1.91</td>
      <td>2.24</td>
      <td>174394</td>
      <td>17.11</td>
      <td>20.10</td>
      <td>193839</td>
      <td>19.01</td>
      <td>22.34</td>
    </tr>
    <tr>
      <th>1</th>
      <td>February</td>
      <td>902558</td>
      <td>17534</td>
      <td>1.94</td>
      <td>2.08</td>
      <td>150097</td>
      <td>16.63</td>
      <td>17.81</td>
      <td>167631</td>
      <td>18.57</td>
      <td>19.89</td>
      <td>17072</td>
      <td>1.89</td>
      <td>2.22</td>
      <td>154890</td>
      <td>17.16</td>
      <td>20.12</td>
      <td>171962</td>
      <td>19.05</td>
      <td>22.34</td>
    </tr>
    <tr>
      <th>2</th>
      <td>March</td>
      <td>986549</td>
      <td>18890</td>
      <td>1.91</td>
      <td>2.05</td>
      <td>165146</td>
      <td>16.74</td>
      <td>17.93</td>
      <td>184036</td>
      <td>18.65</td>
      <td>19.98</td>
      <td>18172</td>
      <td>1.84</td>
      <td>2.16</td>
      <td>169659</td>
      <td>17.20</td>
      <td>20.19</td>
      <td>187831</td>
      <td>19.04</td>
      <td>22.35</td>
    </tr>
    <tr>
      <th>3</th>
      <td>April</td>
      <td>994946</td>
      <td>19461</td>
      <td>1.96</td>
      <td>2.09</td>
      <td>175816</td>
      <td>17.67</td>
      <td>18.91</td>
      <td>195277</td>
      <td>19.63</td>
      <td>21.00</td>
      <td>18373</td>
      <td>1.85</td>
      <td>2.17</td>
      <td>179252</td>
      <td>18.02</td>
      <td>21.13</td>
      <td>197625</td>
      <td>19.86</td>
      <td>23.30</td>
    </tr>
    <tr>
      <th>4</th>
      <td>May</td>
      <td>1024246</td>
      <td>19804</td>
      <td>1.93</td>
      <td>2.07</td>
      <td>179654</td>
      <td>17.54</td>
      <td>18.75</td>
      <td>199458</td>
      <td>19.47</td>
      <td>20.82</td>
      <td>18735</td>
      <td>1.83</td>
      <td>2.14</td>
      <td>182518</td>
      <td>17.82</td>
      <td>20.88</td>
      <td>201253</td>
      <td>19.65</td>
      <td>23.02</td>
    </tr>
    <tr>
      <th>5</th>
      <td>June</td>
      <td>1004198</td>
      <td>19428</td>
      <td>1.93</td>
      <td>2.07</td>
      <td>173344</td>
      <td>17.26</td>
      <td>18.47</td>
      <td>192772</td>
      <td>19.20</td>
      <td>20.54</td>
      <td>18270</td>
      <td>1.82</td>
      <td>2.13</td>
      <td>175962</td>
      <td>17.52</td>
      <td>20.54</td>
      <td>194232</td>
      <td>19.34</td>
      <td>22.67</td>
    </tr>
  </tbody>
</table>
</div>




```python
def Is_Equal_To(x, y, usePercentage):
    if isinstance(y, (list, set, pd.Series)):  # Check if y is a list, set, or Series
        counter = x.isin(y).sum()
    elif pd.isna(y):  # Check if y is NaN
        counter = x.isna().sum()
    else:  # For scalar y
        counter = (x == y).sum()
    
    if usePercentage:
        total_count = len(x)
        return (counter / total_count) * 100 if total_count > 0 else 0
    else:
        return counter
```


```python
def determine_age_in_months(age):
    if age < 3001:
        return 0
    elif age == 3001:
        return 1
    elif age == 3002:
        return 2
    elif age == 3003:
        return 3
    elif age == 3004:
        return 4
    elif age == 3005:
        return 5
    elif age == 3006:
        return 6
    elif age == 3007:
        return 7
    elif age == 3008:
        return 8
    elif age == 3009:
        return 9
    elif age == 3010:
        return 10
    elif age == 3011:
        return 11
    else:
        return None
```


```python
respiratory_illness_df1['EDAD_MESES'] = respiratory_illness_df1['EDAD'].apply(determine_age_in_months)
```

    /tmp/ipykernel_3379182/1722330210.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      respiratory_illness_df1['EDAD_MESES'] = respiratory_illness_df1['EDAD'].apply(determine_age_in_months)



```python
grouped_births_df = births_df.groupby('MES_NAC').size().reset_index(name='Nacimientos')
grouped_births_df.rename(columns={'MES_NAC': 'MES_NACIM'}, inplace=True)
grouped_births_df.head(12)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MES_NACIM</th>
      <th>Nacimientos</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1019441</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>902558</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>986549</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>994946</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>1024246</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>1004198</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>1076970</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>1136385</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>1181923</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>1153400</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>1056550</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>1067736</td>
    </tr>
  </tbody>
</table>
</div>




```python
aggregations = {
    str(i): ('EDAD_MESES', lambda x, i=i: Is_Equal_To(x, i, False)) for i in range(12)
}

grouped_df = respiratory_illness_df1.groupby('MES_NACIM').agg(**aggregations).reset_index()
grouped_df = pd.merge(grouped_births_df, grouped_df, left_on='MES_NACIM', right_on='MES_NACIM')
grouped_df = grouped_df.set_index('MES_NACIM')
grouped_df = grouped_df.loc[[4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3]].reset_index()

# Create a dictionary to map numeric values to month names
month_map = {
    1: 'January', 2: 'February', 3: 'March', 4: 'April',
    5: 'May', 6: 'June', 7: 'July', 8: 'August',
    9: 'September', 10: 'October', 11: 'November', 12: 'December'
}

# Map the month names to 'MES_NACIM'
grouped_df['MES_NACIM'] = grouped_df['MES_NACIM'].map(month_map)

# Update the columns '1' ~ '11' to calculate death rates
for col in map(str, range(0, 12)):
    grouped_df[col] = ((grouped_df[col] / grouped_df['Nacimientos']) * 1000).apply(lambda x: round(x, 2))

grouped_df.drop(columns=['Nacimientos'], inplace=True)


# Set the index to 'Mes de nacimiento'
grouped_df.set_index('MES_NACIM', inplace=True)

new_columns = list(range(1, 13))  # Creating a list from 1 to 12
grouped_df.columns = new_columns  # Assigning the new list of columns
```


```python
grouped_df.head(12)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
    </tr>
    <tr>
      <th>MES_NACIM</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>April</th>
      <td>0.02</td>
      <td>0.09</td>
      <td>0.08</td>
      <td>0.07</td>
      <td>0.05</td>
      <td>0.04</td>
      <td>0.05</td>
      <td>0.05</td>
      <td>0.03</td>
      <td>0.04</td>
      <td>0.03</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>May</th>
      <td>0.02</td>
      <td>0.09</td>
      <td>0.09</td>
      <td>0.06</td>
      <td>0.06</td>
      <td>0.06</td>
      <td>0.05</td>
      <td>0.07</td>
      <td>0.05</td>
      <td>0.03</td>
      <td>0.04</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>June</th>
      <td>0.01</td>
      <td>0.11</td>
      <td>0.09</td>
      <td>0.07</td>
      <td>0.08</td>
      <td>0.07</td>
      <td>0.08</td>
      <td>0.06</td>
      <td>0.03</td>
      <td>0.03</td>
      <td>0.03</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>July</th>
      <td>0.02</td>
      <td>0.09</td>
      <td>0.10</td>
      <td>0.10</td>
      <td>0.10</td>
      <td>0.09</td>
      <td>0.06</td>
      <td>0.04</td>
      <td>0.04</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>August</th>
      <td>0.02</td>
      <td>0.09</td>
      <td>0.13</td>
      <td>0.11</td>
      <td>0.10</td>
      <td>0.08</td>
      <td>0.04</td>
      <td>0.03</td>
      <td>0.04</td>
      <td>0.04</td>
      <td>0.01</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>September</th>
      <td>0.03</td>
      <td>0.13</td>
      <td>0.18</td>
      <td>0.17</td>
      <td>0.10</td>
      <td>0.06</td>
      <td>0.05</td>
      <td>0.03</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.03</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>October</th>
      <td>0.02</td>
      <td>0.18</td>
      <td>0.24</td>
      <td>0.11</td>
      <td>0.08</td>
      <td>0.06</td>
      <td>0.06</td>
      <td>0.03</td>
      <td>0.03</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>November</th>
      <td>0.02</td>
      <td>0.25</td>
      <td>0.18</td>
      <td>0.10</td>
      <td>0.08</td>
      <td>0.04</td>
      <td>0.03</td>
      <td>0.03</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>December</th>
      <td>0.04</td>
      <td>0.18</td>
      <td>0.15</td>
      <td>0.08</td>
      <td>0.04</td>
      <td>0.05</td>
      <td>0.04</td>
      <td>0.04</td>
      <td>0.03</td>
      <td>0.03</td>
      <td>0.03</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>January</th>
      <td>0.02</td>
      <td>0.15</td>
      <td>0.10</td>
      <td>0.07</td>
      <td>0.05</td>
      <td>0.03</td>
      <td>0.04</td>
      <td>0.02</td>
      <td>0.04</td>
      <td>0.03</td>
      <td>0.04</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>February</th>
      <td>0.02</td>
      <td>0.14</td>
      <td>0.09</td>
      <td>0.08</td>
      <td>0.04</td>
      <td>0.05</td>
      <td>0.03</td>
      <td>0.04</td>
      <td>0.04</td>
      <td>0.03</td>
      <td>0.05</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>March</th>
      <td>0.02</td>
      <td>0.09</td>
      <td>0.08</td>
      <td>0.05</td>
      <td>0.06</td>
      <td>0.04</td>
      <td>0.03</td>
      <td>0.04</td>
      <td>0.04</td>
      <td>0.03</td>
      <td>0.03</td>
      <td>0.03</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Plot the heatmap with custom colormap
plt.figure(figsize=(14, 6))
sns.heatmap(grouped_df, annot=True, cmap='Greys' if grayscale else 'RdYlGn_r', linewidths=.5, fmt='.2f')
plt.ylabel('Month of Birth')
plt.xlabel('Age in months postbirth')


plt.savefig(f'{plots_folder}/Fig8.{image_format}', format=image_format, dpi=1200)

plt.show()
```


    
![png](nbconvert_output/output_79_0.png)
    



```python
# Registered deaths grouped by year/month of birth 
grouped_deaths_df = respiratory_illness_df1.groupby(['ENT_RESID','MES_NACIM']).size().reset_index(name='Deaths')
grouped_deaths_df = grouped_deaths_df[grouped_deaths_df['ENT_RESID'] == 24]
grouped_deaths_df = pd.merge(grouped_deaths_df, states_df, left_on='ENT_RESID', right_on='ENTIDAD', how='inner')
#grouped_deaths_df = grouped_deaths_df.groupby(['REGION','MES_NACIM']).size().reset_index(name='Deaths')

# Register births grouped by year/month
grouped_births_df = birth_records_df.groupby(['ENT_RESID','MES_NAC']).size().reset_index(name='Births')
grouped_births_df = grouped_births_df[grouped_births_df['ENT_RESID'] == 24]
grouped_births_df = pd.merge(grouped_births_df, states_df, left_on='ENT_RESID', right_on='ENTIDAD', how='inner')

grouped_df = pd.merge(grouped_births_df, grouped_deaths_df, left_on=['ENT_RESID', 'MES_NAC'], right_on=['ENT_RESID', 'MES_NACIM'], how='left')
grouped_df.drop(columns=['ENTIDAD_x','MES_NACIM','ENTIDAD_y','NOM_ENTIDAD_y', 'REGION_y'], inplace=True)
grouped_df.rename(columns={'REGION_x': 'Region', 'NOM_ENTIDAD_x': 'State', 'MES_NAC': 'Month'}, inplace=True)
grouped_df['Deaths'] = grouped_df['Deaths'].fillna(0)

grouped_df = grouped_df.groupby(['State', 'Month'])[['Births', 'Deaths']].sum().reset_index()
grouped_df['Mortality_Rate'] =  grouped_df['Deaths'] / grouped_df['Births'] *  1000

pivot_df = grouped_df.pivot(index='Month', columns='State', values='Mortality_Rate').reset_index()

# Replace the 'Month' values with month names
pivot_df['Month'] = pivot_df['Month'].map(month_mapping)

# Convert the 'Month' column to a categorical type with the custom order
pivot_df['Month'] = pd.Categorical(pivot_df['Month'], categories=custom_order, ordered=True)

# Sort the dataframe based on the custom order
pivot_df = pivot_df.sort_values('Month')


pivot_df.head(13)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>State</th>
      <th>Month</th>
      <th>San Luis Potosi</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>July</td>
      <td>0.649746</td>
    </tr>
    <tr>
      <th>7</th>
      <td>August</td>
      <td>0.802108</td>
    </tr>
    <tr>
      <th>8</th>
      <td>September</td>
      <td>0.794425</td>
    </tr>
    <tr>
      <th>9</th>
      <td>October</td>
      <td>0.559451</td>
    </tr>
    <tr>
      <th>10</th>
      <td>November</td>
      <td>0.794680</td>
    </tr>
    <tr>
      <th>11</th>
      <td>December</td>
      <td>0.363534</td>
    </tr>
    <tr>
      <th>0</th>
      <td>January</td>
      <td>0.474772</td>
    </tr>
    <tr>
      <th>1</th>
      <td>February</td>
      <td>0.938828</td>
    </tr>
    <tr>
      <th>2</th>
      <td>March</td>
      <td>0.450065</td>
    </tr>
    <tr>
      <th>3</th>
      <td>April</td>
      <td>0.090506</td>
    </tr>
    <tr>
      <th>4</th>
      <td>May</td>
      <td>0.390422</td>
    </tr>
    <tr>
      <th>5</th>
      <td>June</td>
      <td>0.352144</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Registered deaths grouped by year/month of demise
grouped_deaths_df = respiratory_illness_df1.groupby(['ENT_RESID','MES_OCURR']).size().reset_index(name='Deaths')
grouped_deaths_df = grouped_deaths_df[grouped_deaths_df['ENT_RESID'] == 24]
grouped_deaths_df = pd.merge(grouped_deaths_df, states_df, left_on='ENT_RESID', right_on='ENTIDAD', how='inner')

# Register births grouped by year/month
grouped_births_df = birth_records_df.groupby(['ENT_RESID','MES_NAC']).size().reset_index(name='Births')
grouped_births_df = grouped_births_df[grouped_births_df['ENT_RESID'] == 24]
grouped_births_df = pd.merge(grouped_births_df, states_df, left_on='ENT_RESID', right_on='ENTIDAD', how='inner')

grouped_df = pd.merge(grouped_births_df, grouped_deaths_df, left_on=['ENT_RESID', 'MES_NAC'], right_on=['ENT_RESID', 'MES_OCURR'], how='left')
grouped_df.drop(columns=['ENTIDAD_x','MES_OCURR','ENTIDAD_y','NOM_ENTIDAD_y', 'REGION_y'], inplace=True)
grouped_df.rename(columns={'REGION_x': 'Region', 'NOM_ENTIDAD_x': 'State', 'MES_NAC': 'Month'}, inplace=True)
grouped_df['Deaths'] = grouped_df['Deaths'].fillna(0)

grouped_df = grouped_df.groupby(['State', 'Month'])[['Births', 'Deaths']].sum().reset_index()
grouped_df['Mortality_Rate'] =  grouped_df['Deaths'] / grouped_df['Births'] *  1000



pivot_df = grouped_df.pivot(index='Month', columns='State', values='Mortality_Rate').reset_index()

# Replace the 'Month' values with month names
pivot_df['Month'] = pivot_df['Month'].map(month_mapping)

# Convert the 'Month' column to a categorical type with the custom order
pivot_df['Month'] = pd.Categorical(pivot_df['Month'], categories=occurence_custom_order, ordered=True)

# Sort the dataframe based on the custom order
pivot_df = pivot_df.sort_values('Month')

pivot_df.head(13)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>State</th>
      <th>Month</th>
      <th>San Luis Potosi</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>April</td>
      <td>0.724047</td>
    </tr>
    <tr>
      <th>4</th>
      <td>May</td>
      <td>0.173521</td>
    </tr>
    <tr>
      <th>5</th>
      <td>June</td>
      <td>0.220090</td>
    </tr>
    <tr>
      <th>6</th>
      <td>July</td>
      <td>0.203046</td>
    </tr>
    <tr>
      <th>7</th>
      <td>August</td>
      <td>0.420152</td>
    </tr>
    <tr>
      <th>8</th>
      <td>September</td>
      <td>0.361102</td>
    </tr>
    <tr>
      <th>9</th>
      <td>October</td>
      <td>0.484858</td>
    </tr>
    <tr>
      <th>10</th>
      <td>November</td>
      <td>0.669204</td>
    </tr>
    <tr>
      <th>11</th>
      <td>December</td>
      <td>1.332956</td>
    </tr>
    <tr>
      <th>0</th>
      <td>January</td>
      <td>0.690578</td>
    </tr>
    <tr>
      <th>1</th>
      <td>February</td>
      <td>0.840004</td>
    </tr>
    <tr>
      <th>2</th>
      <td>March</td>
      <td>0.675098</td>
    </tr>
  </tbody>
</table>
</div>


