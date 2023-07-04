# EXPLORANDO SALÁRIOS NA ÁREA DE CIÊNCIA DE DADOS: UMA ANÁLISE COM APRENDIZADO DE MÁQUINA E MINERAÇÃO DE DADOS

# Instalação / Importação das bibliotecas utilizadas

import warnings
warnings.filterwarnings('ignore')

# Import Neccessary libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Import country code libraries
!pip install pycountry -q
import pycountry

# Importação do dataset
df = pd.read_csv('salaries.csv')

# Remoção dos campos que não serão utilizados na visualização dos dados
df.drop(df[['salary','salary_currency']], axis = 1, inplace = True)

# Verificação se existe valores nulos para tratar 

df.isnull().sum()

# Quantidade de Vagas por níveis de experiência, substituição para facilitar o entendimento 

df['experience_level'] = df['experience_level'].replace('EN','Nível Básico / Júnior')
df['experience_level'] = df['experience_level'].replace('MI','Nível Médio / Intermediário')
df['experience_level'] = df['experience_level'].replace('SE','Nível Sênior / Especialista')
df['experience_level'] = df['experience_level'].replace('EX','Nível Executivo / Diretor')

ex_level = df['experience_level'].value_counts()
fig = px.treemap(ex_level, path = [ex_level.index], values = ex_level.values, 
                title = 'Nível de Experiência')
fig.show()

# Diferentes cargos disponíveis no total

print('Diferentes descrições dos cargos: ', len(set(df['job_title'])))

# Ranking com as principais 10 funções 

top10_job_titles = df['job_title'].value_counts()[:10]
fig = px.bar(y = top10_job_titles.values, x = top10_job_titles.index, 
            text = top10_job_titles.values, title = 'Top 10 Cargos')
fig.update_layout(xaxis_title = "Descrição dos Cargos", yaxis_title = "Quantidade de Vagas")
fig.show()

# Quantidade de vagas por tipos de emprego

group = df['employment_type'].value_counts()
emp_type = ['Período Integral', 'Meio Período', 'Contrato', 'Freelancer']

fig = px.bar(x = group.values, y = emp_type, 
       color = group.index, text = group.values, 
       title = 'Tipos de Contrato')

fig.update_layout( xaxis_title = "Quantidade", yaxis_title = "Tipos")
fig.show()

# Função para converter em nome do país o código (ISO3166)
def country_code_to_name(country_code):
    try:
        return pycountry.countries.get(alpha_2=country_code).name
    except:
        return country_code
    # Function to convert country code to full name
def country_code_to_name(code):
    try:
        country = pycountry.countries.get(alpha_2=code)
        return country.name
    except:
        return None

# Converte o código em nomes
df['company_location'] = df['company_location'].apply(country_code_to_name)

# Média Salarial por Localização das Empresas
avg_salary_by_location = df.groupby('company_location', as_index=False)['salary_in_usd'].mean()

fig = px.choropleth(avg_salary_by_location,
                     locations='company_location',
                     locationmode='country names',
                     color='salary_in_usd',
                     hover_name='company_location',
                     color_continuous_scale=px.colors.sequential.Plasma,
                     title='Média Salarial por Localização das Empresas',
                     labels={'salary_in_usd': 'Salário Médio em Dólares'},
                     projection='natural earth')

fig.show()

# Dimensão das empresas

df['company_size'] = df['company_size'].replace('S','Pequeno')
df['company_size'] = df['company_size'].replace('M','Médio')
df['company_size'] = df['company_size'].replace('L','Grande')

group = df['company_size'].value_counts()
fig = px.pie(values = group.values, names = group.index, 
            title = 'Tamanho das Empresas')
fig.show()


#evolução salarial

avg_salaries_based_onyear=np.array(df['salary_in_usd'].groupby(df['work_year']).mean())

plt.title('Evolução da Média Salarial')
plt.xlabel('Anos')
plt.ylabel('Salário (US$)')
sns.set_style("darkgrid")   

sns.lineplot(x=['2020', '2021', '2022','2023'],y=avg_salaries_based_onyear)

# Top 10 maiores médias anuais do salário por função

top_salary =  df.groupby('job_title').agg({'salary_in_usd':'mean'}).round(2).sort_values('salary_in_usd', ascending=False).head(10)

plt.figure(figsize=(8,4))

sns.set(style="whitegrid")
ax= sns.barplot(y = top_salary.index, x ='salary_in_usd', 
            data = top_salary,           
            palette = "viridis",
            width=0.9)
            
plt.title('Top 10 Salários por Cargos\n', fontsize=16, fontweight="bold", loc="center")
plt.suptitle("\n***Média dos Salários Anual\n", fontsize = 10, color="gray")
plt.xlabel('\nSalário (US$)', color="black", fontsize=10)
plt.ylabel('Cargos', color="black", fontsize=10)
plt.xticks(fontsize=10, color="black")
plt.yticks(fontsize=10, color="black")

for i in ax.containers:
    ax.bar_label(i, size=10, label_type = "center", color="white", fontweight="bold")

plt.show() 


# Ajustes para melhor visualização dos gráficos

#

df['Nível Experiência'] = df ['experience_level']

df = df.drop("experience_level", axis=1)

df['Nível Experiência'] = df['Nível Experiência'].replace('Nível Básico / Júnior','Júnior')
df['Nível Experiência'] = df['Nível Experiência'].replace('Nível Médio / Intermediário', 'Pleno')
df['Nível Experiência'] = df['Nível Experiência'].replace('Nível Sênior / Especialista','Sênior')
df['Nível Experiência'] = df['Nível Experiência'].replace('Nível Executivo / Diretor', 'Gerencial')

#

df['Tipo'] = df ['employment_type']

df = df.drop("employment_type", axis=1)

df['Tipo'] = df['Tipo'].replace('PT','Meio Período')
df['Tipo'] = df['Tipo'].replace('FT', 'Período Integral')
df['Tipo'] = df['Tipo'].replace('CT','Contrato')
df['Tipo'] = df['Tipo'].replace('FL', 'Freelancer')

#

df['Porte Empresa'] = df ['company_size']

df = df.drop("company_size", axis=1)

#

df['Modelo'] = df ['remote_ratio']

df = df.drop("remote_ratio", axis=1)


df = df.astype({"Modelo": object})


df['Modelo'] = df['Modelo'].replace(0,'Presencial')
df['Modelo'] = df['Modelo'].replace(50, 'Híbrido')
df['Modelo'] = df['Modelo'].replace(100,'Remoto')

# Modelo de trabalho e a oferta de vagas

temp = df['Modelo'].value_counts().to_frame()
temp.columns = ['Quantidade']
temp['Percentual(%)'] =round(100 * df['Modelo'].value_counts(normalize= True),2)

fig = plt.figure(figsize=(9,4))
ax1 = plt.subplot(121)
ax1.axis('off')
bbox =[0,0,0.8,0.8]
ax1.table(cellText = temp.values, rowLabels=temp.index, colLabels=temp.columns,bbox=bbox)
ax2 = fig.add_subplot(122)
ax2 = sns.barplot(data = temp, x=temp.index,y=temp['Percentual(%)'])
ax2.bar_label(ax2.containers[0])
# plt.xticks(rotation= 30)
plt.xlabel('Modelo Trabalho', fontdict={'fontsize':14})
plt.ylabel('Percentual (%)', fontdict={'fontsize':14})

plt.suptitle('Modelo de Trabalho')
plt.show()




# Comparativos

a = 2  
b = 2  
c = 1  
columns= ['Nível Experiência', 'Tipo', 'Porte Empresa', 'Modelo']
fig = plt.figure(figsize = (15, 8))

for col in columns:
    dft = df.groupby([col], as_index=False).agg({'salary_in_usd': 'mean'}).sort_values(by='salary_in_usd').reset_index(drop=True)
    
    plt.subplot(a, b, c)
    ax = sns.barplot(data=dft, x=col, y='salary_in_usd', 
                     palette=["#4d92d4", "#13609d", "#004f8a", "#001350"], 
                     errorbar=('ci', False));

    for i in ax.patches:    
        ax.text(x = i.get_x() + i.get_width()/2, y = i.get_height()/2,
                s = f"{round(i.get_height(),2)}", 
                ha = 'center', va ='center', size = 10, weight = 'bold', color = 'white')
        
    plt.title(f"Salário Médio por {col}")
    plt.ylabel("Salário Médio")
    plt.xlabel(f"{col}");
    c = c + 1

plt.tight_layout()
plt.show()



# Processamento dos dados

df = pd.read_csv('salaries.csv')

#Função genérica que nos ajudará a dividir nosso conjunto de dados em conjunto de dados de treinamento, teste e validação.

def encode_categorical_columns(dataframe):
    cat_cols = ['experience_level', 'employment_type', 'job_title', 'employee_residence', 'remote_ratio', 'company_location', 'company_size']
    cols_to_drop = ['salary', 'salary_currency']
    
    # Criação de um novo dataset com as outras colunas
    new_dataframe = dataframe.drop(cat_cols + cols_to_drop, axis=1)
    
    # Criação de um dicionário para cada codificador
    encoders = {}
    
    #  Fazer uma iteração para cada uma das colunas, codificar
    for col in cat_cols:
        encoder = LabelEncoder()
        encoders[col] = encoder
        new_dataframe[col] = encoder.fit_transform(dataframe[col])
    return new_dataframe, encoders
    
clean_df, encoders = encode_categorical_columns(df)
clean_df.head()    

#Matriz de Correlação

def plot_correlation_matrix(dataframe):
    # Calcula a matriz de correlação
    corr_matrix = dataframe.corr()

    # Criação de uma máscara para o triângulo inferior
    mask = np.tril(np.ones_like(corr_matrix, dtype=bool))

    # Configurar a figura matplotlib
    fig, ax = plt.subplots(figsize=(10, 8))

    # Gerar um mapa de cores divergente personalizado
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Imprimir a matriz de correlação
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                annot=True, fmt=".2f", annot_kws={"size": 10})

    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    
    plt.title("Matriz de Correlação")

    
    plt.show()
    
plot_correlation_matrix(clean_df)

# Data Splitting

def split_dataset(dataframe, target_column, test_size=0.2, validation_size=0.2, random_state=None):
    # Separar as características e as variáveis de objetivo
    X = dataframe.drop(target_column, axis=1)
    y = dataframe[target_column]

    # Dividir os dados em conjuntos de treino e de teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Calcular a porcentagem restante para o conjunto de validação
    remaining_size = 1 - test_size
    validation_size_adjusted = validation_size / remaining_size

    # Dividir o conjunto de treino em conjuntos de treino e validação
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_size_adjusted, random_state=random_state)

    # Devolver os conjuntos de dados divididos
    return X_train, X_val, X_test, y_train, y_val, y_test
    
    
    
# Função genérica para normalização

def standardize_data(train_X, train_y):
    # Criação de um objeto StandardScaler
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    # Ajustar o escalonador ao conjunto de treino e transformar esses dados
    train_X_standardized = scaler_x.fit_transform(train_X)
    train_y_array = np.array(train_y)
    train_y_standardized = scaler_y.fit_transform(train_y_array.reshape((-1,1)))

    # Retornar o conjunto de treino padronizado a variável de destino padronizada e o escalonador ajustado
    return train_X_standardized, train_y_standardized, scaler_x, scaler_y


def standardize_test_data(test_X, test_y, scaler_X, scaler_y):
    # Transformar test_X utilizando o escalonador_X ajustado
    test_X_standardized = scaler_X.transform(test_X)

    # Converter test_y numa matriz NumPy e remodelar se necessário
    test_y_array = np.array(test_y)
    if len(test_y_array.shape) == 1:
        test_y_array = test_y_array.reshape(-1, 1)

    # Transformar test_y utilizando o escalonador_y ajustado
    test_y_standardized = scaler_y.transform(test_y_array)

    # Nivelar a variável de destino padronizada, se necessário
    test_y_standardized = test_y_standardized.flatten()

    # Retornar as características do conjunto de teste padronizado e a variável de destino padronizada
    return test_X_standardized, test_y_standardized

# Realização das previsões

# Dividir os Dados
train_x, val_x, test_x, train_y, val_y, test_y = split_dataset(clean_df, 'salary_in_usd', random_state=42)
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)
print(val_x.shape)
print(val_y.shape)

# Padronizar o conjunto de treino
train_x_std, train_y_std, scaler_x, scaler_y = standardize_data(train_x, train_y)
test_x_std, test_y_std = standardize_test_data(test_x, test_y, scaler_x, scaler_y)
val_x_std, val_y_std = standardize_test_data(val_x, val_y, scaler_x, scaler_y)

#  Construir e treinar o modelo
lin_reg_model = LinearRegression()
lin_reg_model.fit(train_x_std, train_y_std)

# Predição e Teste
y_pred = lin_reg_model.predict(test_x_std)

# Avaliação do Modelo
mse = mean_squared_error(test_y_std, y_pred)
print(mse)

# Gráfico com as informações

def predict_and_plot_difference(model, X_val, y_val, scaler):
   # Fazer previsões no conjunto de validação
    y_val_pred = model.predict(X_val)

     # Desfazer a normalização dos valores previstos
    y_val_pred_original = scaler.inverse_transform(y_val_pred.reshape(-1, 1))
    y_val_original = scaler.inverse_transform(y_val.reshape(-1, 1))
    
    # Criar um DataFrame com valores previstos e reais
    df_predictions = pd.DataFrame({'Actual': y_val_original.flatten(), 'Predicted': y_val_pred_original.flatten()})

    # Calcular a diferença entre os valores previstos e reais
    difference = df_predictions['Actual'] - df_predictions['Predicted']
    df_predictions['Difference'] = difference
    df_predictions['Difference in %'] = df_predictions['Difference']/df_predictions['Actual'] * 100

    # Imprimir a diferença
    plt.figure(figsize=(8, 6))
    plt.scatter(np.arange(len(difference)), difference, color='b')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Índice')
    plt.ylabel('Diferença')
    plt.title('Diferença entre Valores Reais e Previstos')
    plt.show()

    return df_predictions, difference

df_predictions, difference = predict_and_plot_difference(lin_reg_model, val_x_std, val_y_std, scaler_y)

# Tabela com os valores reais vs predições
df_predictions


