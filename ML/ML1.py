import numpy as np
import pandas as pd
import statistics
from scipy.stats import skew, norm, expon
import matplotlib.pyplot as plt

df = pd.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/datasets/mtcars.csv")

mpg = df['mpg'].values
print(mpg)


mode = statistics.mode(mpg)
mediana = statistics.median(mpg)
asymmetry_coefficient = skew(mpg)


print('Mode: ',mode, '\n Median: ', mediana, '\nasymmetry_coefficient: ',asymmetry_coefficient, )
if asymmetry_coefficient >0:print('right-handed')
elif  asymmetry_coefficient ==0: print('symmetry')
else: print('left-handed')


#3
plt.hist(mpg)
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.show()

#4
power=[]

for i in mpg:
    if i>120:
        power.append('High')
    else:
        power.append('Low')

#5
mu = np.mean(mpg)
sigma = np.std(mpg)

# EDF
data_sort = np.sort(mpg)
print('    h ', len(data_sort)+1), ' gg ', len(data_sort)
ecdf = np.arange(1, len(data_sort)+1) / len(data_sort)

# NDF
x = np.linspace(min(data_sort), max(data_sort), 1000)
cdf_normal = norm.cdf(x, loc=mu, scale=sigma)

plt.figure(figsize=(10, 5))
plt.step(data_sort, ecdf, label='Empirical Distribution Function')
plt.plot(x, cdf_normal, label=f'Normal Distribution\nmu={mu:.3f}, sigma={sigma:.3f}')
plt.xlabel('Value')
plt.ylabel('Probability')
plt.title('EDF vs. NDF')
plt.legend()
plt.show()

#6
data_exp = np.random.exponential(scale=1, size=200)

plt.figure(figsize=(10, 5))
count, bins, _ = plt.hist(data_exp, bins=20, density=True, alpha=0.6, label='Relative Frequencies')

x = np.linspace(0, max(data_exp), 1000)
pdf_exp = expon.pdf(x, scale=1)
plt.plot(x, pdf_exp, 'r-', label=f'Density\nλ=1')

plt.title('Histogram and Density of Exponential Distribution')
plt.xlabel('Value')
plt.ylabel('Probability')
plt.legend()
plt.show()

#7
new_dataframe={
'StudentID':[1,2,3,4,5,6],
'Name':['Lowri','Molly','Ria','Irene ','Hazel','Yasmin'],
'Gender':['m','f','f','f','m','f'],
'MathScore':[12,5,4,9,11,6],
'HistoryScore':[1,8,7,3,12,5]}

df = pd.DataFrame(new_dataframe)

math=0
history=0
for i in df['MathScore'].values:
    math +=i
math=math/df['MathScore'].values.size

for i in df['HistoryScore'].values:
    history +=i
math=math/df['HistoryScore'].values.size

print('Average value of History: ',history)
print('Average value of History: ',history)


average=[]
k=0
for i in df['MathScore']:
    average.append((i + df['HistoryScore'].loc[k])/2)
    k+=1

df['Avarage'] = average

print('New DF', df)

min_rate=12
max_rate=-1
for i in df['Avarage']:
    if i <min_rate:
        min_rate=i
    if i >max_rate:
        max_rate=i

print('Min ',min_rate, '\nMax ', max_rate)

indx_min = []
indx_max = []

for i in range(len(df.Name)):
    if min_rate == df.Avarage[i]:
        indx_min.append(i)
    if max_rate == df.Avarage[i]:
        indx_max.append(i)

print('\nMax : ')
for i in indx_max:
    print(df.Name.loc[i])
print('\n\nMin : ')
for i in indx_min:
    print(df.Name.loc[i])        

plt.hist(average)
plt.title("Average")
plt.show()