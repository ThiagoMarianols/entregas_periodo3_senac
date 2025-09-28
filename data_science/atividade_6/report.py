import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro, ttest_ind, mannwhitneyu


df = pd.read_csv('titanic.xls')

df = df[['Survived','Pclass','Sex','Age','Fare','Embarked']]
df['Age'] = df['Age'].fillna(df['Age'].median())
df = df.dropna(subset=['Embarked'])

print(df.describe())

sns.barplot(x='Sex', y='Survived', data=df)
plt.title('Taxa de sobrevivência por sexo')
plt.show()

sns.boxplot(x='Pclass', y='Fare', data=df)
plt.title('Distribuição das tarifas por classe')
plt.show()

sns.boxplot(x='Survived', y='Age', data=df)
plt.title('Distribuição de idade - sobreviventes vs não')
plt.show()

male = df[df['Sex']=='male']['Survived']
female = df[df['Sex']=='female']['Survived']

print('Shapiro male:', shapiro(male))
print('Shapiro female:', shapiro(female))

u, p = mannwhitneyu(male, female)
print('Mann-Whitney sexo vs sobrevivência: U=%.3f p=%.3f' % (u,p))

fare1 = df[df['Pclass']==1]['Fare']
fare3 = df[df['Pclass']==3]['Fare']

print('Shapiro fares1:', shapiro(fare1))
print('Shapiro fares3:', shapiro(fare3))

t, p = ttest_ind(fare1, fare3, equal_var=False)
print('Teste t tarifas classe1 vs classe3: t=%.3f p=%.3f' % (t,p))

age_surv = df[df['Survived']==1]['Age']
age_nosurv = df[df['Survived']==0]['Age']

print('Shapiro age_surv:', shapiro(age_surv))
print('Shapiro age_nosurv:', shapiro(age_nosurv))

t, p = ttest_ind(age_surv, age_nosurv, equal_var=False)
print('Teste t idade sobreviventes vs não: t=%.3f p=%.3f' % (t,p))

numeric_cols = ['Age', 'Fare']
df_numeric = df[numeric_cols].dropna()

corr_matrix = df_numeric.corr(method='pearson')
print("Matriz de correlação de Pearson:")
print(corr_matrix)

corr_matrix_spearman = df_numeric.corr(method='spearman')
print("\nMatriz de correlação de Spearman:")
print(corr_matrix_spearman)

plt.figure(figsize=(6, 4))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlação de Pearson entre variáveis numéricas')
plt.show()
