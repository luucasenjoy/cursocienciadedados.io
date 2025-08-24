import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from io import BytesIO

# Configuração da página - DEVE SER O PRIMEIRO COMANDO STREAMLIT
st.set_page_config(
    page_title='Análise RFV - Segmentação de Clientes',
    page_icon='📊',
    layout="wide",
    initial_sidebar_state='expanded'
)

# Funções auxiliares
@st.cache_data
def to_excel(df):
    """Converte DataFrame para Excel"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='RFV_Analysis')
    return output.getvalue()

def recencia_class(x, r, q_dict):
    """Classifica a recência (menor valor = melhor)"""
    if x <= q_dict[r][0.25]: return 'A'
    elif x <= q_dict[r][0.50]: return 'B'
    elif x <= q_dict[r][0.75]: return 'C'
    else: return 'D'

def freq_val_class(x, fv, q_dict):
    """Classifica frequência/valor (maior valor = melhor)"""
    if x <= q_dict[fv][0.25]: return 'D'
    elif x <= q_dict[fv][0.50]: return 'C'
    elif x <= q_dict[fv][0.75]: return 'B'
    else: return 'A'

def main():
    # Cabeçalho da aplicação
    st.title('📊 Análise RFV - Segmentação de Clientes')
    
    st.markdown("""
    **RFV (Recência, Frequência, Valor)** é uma metodologia poderosa para segmentação de clientes baseada no 
    comportamento de compras. Esta análise ajuda a direcionar ações de marketing e CRM de forma mais eficiente.
    
    ### 📋 Como funciona:
    - **Recência (R)**: Dias desde a última compra (quanto menor, melhor)
    - **Frequência (F)**: Número total de compras (quanto maior, melhor)  
    - **Valor (V)**: Total gasto pelo cliente (quanto maior, melhor)
    """)
    
    # Upload do arquivo
    st.sidebar.header("📤 Upload de Dados")
    uploaded_file = st.sidebar.file_uploader(
        "Carregue seu arquivo de compras (CSV ou Excel)",
        type=['csv', 'xlsx'],
        help="O arquivo deve conter: ID_cliente, DiaCompra, CodigoCompra, ValorTotal"
    )
    
    # Exemplo de download
    with st.sidebar.expander("📝 Não tem um arquivo? Use nosso exemplo"):
        exemplo_data = {
            'ID_cliente': [1, 1, 2, 2, 3, 4, 4, 4, 5, 5, 6, 7, 8, 9, 10],
            'DiaCompra': ['2023-01-15', '2023-02-20', '2023-01-10', '2023-03-05', 
                         '2023-02-28', '2023-01-05', '2023-02-15', '2023-03-25',
                         '2023-03-01', '2023-03-20', '2023-02-10', '2023-01-20',
                         '2023-03-15', '2023-02-05', '2023-03-10'],
            'CodigoCompra': [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 
                           1009, 1010, 1011, 1012, 1013, 1014, 1015],
            'ValorTotal': [150.50, 200.00, 75.25, 300.00, 50.00, 500.00, 250.00, 
                         100.00, 75.00, 125.00, 300.00, 80.00, 200.00, 150.00, 400.00]
        }
        df_exemplo = pd.DataFrame(exemplo_data)
        csv_exemplo = df_exemplo.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="⬇️ Baixar Arquivo Exemplo (CSV)",
            data=csv_exemplo,
            file_name="dados_exemplo_rfv.csv",
            mime="text/csv"
        )
    
    if uploaded_file is not None:
        try:
            # Leitura do arquivo
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, parse_dates=['DiaCompra'])
            else:
                df = pd.read_excel(uploaded_file, parse_dates=['DiaCompra'])
            
            st.success(f"✅ Arquivo '{uploaded_file.name}' carregado com sucesso!")
            
            # Verificar colunas necessárias
            colunas_necessarias = ['ID_cliente', 'DiaCompra', 'CodigoCompra', 'ValorTotal']
            if not all(col in df.columns for col in colunas_necessarias):
                st.error("❌ O arquivo não contém todas as colunas necessárias!")
                st.write("Colunas encontradas:", list(df.columns))
                st.write("Colunas necessárias:", colunas_necessarias)
                return
            
            # Análise RFV
            with st.spinner('🔄 Processando análise RFV...'):
                # Recência
                dia_atual = df['DiaCompra'].max()
                df_recencia = df.groupby('ID_cliente')['DiaCompra'].max().reset_index()
                df_recencia.columns = ['ID_cliente', 'DiaUltimaCompra']
                df_recencia['Recencia'] = (dia_atual - df_recencia['DiaUltimaCompra']).dt.days
                
                # Frequência
                df_frequencia = df.groupby('ID_cliente')['CodigoCompra'].count().reset_index()
                df_frequencia.columns = ['ID_cliente', 'Frequencia']
                
                # Valor
                df_valor = df.groupby('ID_cliente')['ValorTotal'].sum().reset_index()
                df_valor.columns = ['ID_cliente', 'Valor']
                
                # Merge RFV
                df_rfv = df_recencia.merge(df_frequencia, on='ID_cliente').merge(df_valor, on='ID_cliente')
                df_rfv.set_index('ID_cliente', inplace=True)
                
                # Segmentação por quartis
                quartis = df_rfv.quantile(q=[0.25, 0.5, 0.75])
                
                df_rfv['R_quartil'] = df_rfv['Recencia'].apply(recencia_class, args=('Recencia', quartis))
                df_rfv['F_quartil'] = df_rfv['Frequencia'].apply(freq_val_class, args=('Frequencia', quartis))
                df_rfv['V_quartil'] = df_rfv['Valor'].apply(freq_val_class, args=('Valor', quartis))
                
                df_rfv['RFV_Score'] = df_rfv['R_quartil'] + df_rfv['F_quartil'] + df_rfv['V_quartil']
                
                # Ações de marketing
                acoes_marketing = {
                    'AAA': '🎯 Clientes Premium - Ofertas exclusivas e programas VIP',
                    'AAB': '⭐ Clientes Fiéis - Programas de fidelidade',
                    'ABB': '🚀 Clientes em Ascensão - Ofertas personalizadas',
                    'BBB': '👍 Clientes Regulares - Manter engajamento',
                    'CCC': '📧 Clientes Ocasionais - Campanhas de remarketing',
                    'DDD': '💤 Clientes Inativos - Campanhas de reativação',
                    'DAA': '⚡ Clientes em Risco - Ofertas especiais para retenção',
                    'CAA': '📞 Clientes Valiosos - Contato personalizado'
                }
                
                df_rfv['Acao_Marketing'] = df_rfv['RFV_Score'].map(acoes_marketing)
                df_rfv['Acao_Marketing'] = df_rfv['Acao_Marketing'].fillna('📋 Análise personalizada necessária')
            
            st.success('✅ Análise RFV concluída com sucesso!')
            
            # Métricas principais
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total de Clientes", len(df_rfv))
            with col2:
                st.metric("Faturamento Total", f"R$ {df_rfv['Valor'].sum():,.2f}")
            with col3:
                avg_value = df_rfv['Valor'].mean()
                st.metric("Valor Médio por Cliente", f"R$ {avg_value:,.2f}")
            
            # Resultados
            st.subheader("📊 Tabela RFV Completa")
            st.dataframe(df_rfv, use_container_width=True)
            
            # Estatísticas dos scores
            st.subheader("📈 Distribuição dos Scores RFV")
            score_counts = df_rfv['RFV_Score'].value_counts()
            st.bar_chart(score_counts)
            
            # Ações de marketing
            st.subheader("🎯 Ações de Marketing por Segmento")
            acao_counts = df_rfv['Acao_Marketing'].value_counts()
            st.dataframe(acao_counts)
            
            # Download dos resultados
            st.subheader("📥 Download dos Resultados")
            excel_data = to_excel(df_rfv.reset_index())
            
            st.download_button(
                label="💾 Baixar Análise RFV (Excel)",
                data=excel_data,
                file_name="analise_rfv_completa.xlsx",
                mime="application/vnd.ms-excel"
            )
            
            # Clientes premium
            st.subheader("🏆 Top 10 Clientes Premium (AAA)")
            clientes_premium = df_rfv[df_rfv['RFV_Score'] == 'AAA'].sort_values('Valor', ascending=False).head(10)
            st.dataframe(clientes_premium)
            
        except Exception as e:
            st.error(f"❌ Erro ao processar o arquivo: {str(e)}")
            st.write("Detalhes do erro:", e)
    
    else:
        st.info("👆 Faça upload de um arquivo CSV ou Excel para iniciar a análise RFV")

if __name__ == '__main__':
    main()