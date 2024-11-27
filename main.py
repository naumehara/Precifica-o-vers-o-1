import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import lognorm
import matplotlib.pyplot as plt

st.title('Precificação Versão 1')
st.subheader('Estimativa de produtividade e conclusão de taxa comercial.')

st.text('Para começar, indique a produtividade em sacas nos últimos anos')
col1, col2, col3, col4 , col5= st.columns(5) #criar coluna
# Cabeçalhos da "tabela"
col1.write("Ano 1")
col2.write("Ano 2")
col3.write("Ano 3")
col4.write("Ano 4")
col5.write("Ano 5")

# Inputs de produtividade para cada ano
p1 = col1.number_input('Produtividade Ano 1', value=0, min_value=0, label_visibility="collapsed")
p2 = col2.number_input('Produtividade Ano 2', value=0, min_value=0, label_visibility="collapsed")
p3 = col3.number_input('Produtividade Ano 3', value=0, min_value=0,  label_visibility="collapsed")
p4 = col4.number_input('Produtividade Ano 4', value=0, min_value=0,  label_visibility="collapsed")
p5 = col5.number_input('Produtividade Ano 5', value=0, min_value=0,  label_visibility="collapsed")
cob = st.selectbox('Escolha o nível de cobertura', options=('60%', '65%', '70%', '75%','80%' ))

x = st.checkbox('Desconsiderar Desvio Padrão', value=False)
# Ajustar o valor de x
x_vl = 0 if x else 1
 
md = np.mean([p1,p2,p3,p4,p5])
dv = np.std([p1,p2,p3,p4,p5])*x_vl
produtividade_segurada =  float(0 if round((md - dv) * (int(cob.strip('%')) / 100),2) <0 else round((md - dv) * (int(cob.strip('%')) / 100),2))
st.text('A produtividade segurada será de:')
st.markdown(f"**{produtividade_segurada}**"+' sacas')

li =float(st.selectbox('Insira o limite inferior de faixa que deseja', options=( '0','5', '10', '15', '20','25' )))

st.text('O número de sacas que serão seguradas é de:')
sacas = round(0 if produtividade_segurada - int(li) < 0 else produtividade_segurada - int(li),2)
st.markdown(f"**{sacas}**"+' sacas')


# PDF (função densidade de probabilidade) da distribuição Log-Normal
mu = np.log(md)           # Logaritmo da média (LN)
sigma = np.log(dv) 
x_values = np.linspace(0.01, 3 * md, 100) 
pdf_values = lognorm.pdf(x_values, s=sigma, scale=np.exp(mu))
pdf_produtividade = lognorm.pdf(float(produtividade_segurada), s=sigma, scale=np.exp(mu))
pdf_li= lognorm.pdf(float(li), s=sigma, scale=np.exp(mu))

plt.figure(figsize=(10, 6))
plt.plot(x_values, pdf_values, label='Distribuição da Produtividade', color='blue')

x_fill = np.linspace(li, produtividade_segurada, 100)
pdf_fill = lognorm.pdf(x_fill, s=sigma, scale=np.exp(mu))
plt.fill_between(x_fill, pdf_fill, color='lightpink', alpha=0.5, label='Área de cobertura')

# Marcar os pontos de produtividade_segurada e li
plt.scatter(produtividade_segurada, pdf_produtividade, color='green', label=f'Produtividade Segurada = {produtividade_segurada}')
plt.scatter(li, pdf_li, color='red', label=f'LI = {li}')

# Adicionar linhas verticais nos pontos
plt.axvline(produtividade_segurada, color='green', linestyle='--')
plt.axvline(li, color='red', linestyle='--')

# Adicionar título e legendas
plt.title('Distribuição de Produtividade')
plt.xlabel('Sacas')
plt.ylabel('Densidade de Probabilidade')
plt.legend()

# Mostrar o gráfico
st.pyplot(plt)

area = lognorm.cdf(produtividade_segurada, s=sigma, scale=np.exp(mu)) - lognorm.cdf(li, s=sigma, scale=np.exp(mu))
area_per= round(area*100,2)
st.markdown('Taxa pura de Risco '+f"**{area_per}**"+'%')

st.text('Taxas e Despesas:')
carr_seg = 0.04
da = 0.03
st.write("""
| Carregamento Segurança | Despesas Administrativas | 
| ---------------------- | ------------------------ |
|          4%            |            3%            |
""")

tx_comercial = round(area*(1+carr_seg)/(1-da),4)
st.text('Portanto, a taxa comercial será de:')
st.markdown(f"**{tx_comercial}**")

#st.write("Produtividades Inseridas")
#st.table({
    #'Ano 1': [p1],
   # 'Ano 2': [p2],
    #'Ano 3': [p3],
   # 'Ano 4': [p4]
#})
