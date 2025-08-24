import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from io import BytesIO
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuração da página - DEVE SER O PRIMEIRO COMANDO STREAMLIT
st.set_page_config(
    page_title='RFV + Clusterização - Segmentação Avançada',
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
        df.to_excel(writer, index=False, sheet_name='RFV_Cluster_Analysis')
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

def find_optimal_clusters(data, max_k=10):
    """Encontra o número ideal de clusters usando o método do cotovelo"""
    inertias = []
    silhouette_scores = []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
        
        if len(set(kmeans.labels_)) > 1:  # Silhouette score precisa de pelo menos 2 clusters
            silhouette_scores.append(silhouette_score(data, kmeans.labels_))
        else:
            silhouette_scores.append(0)
    
    return inertias, silhouette_scores, list(k_range)

def main():
    # Cabeçalho da aplicação
    st.title('📊 Análise RFV + Clusterização - Segmentação Avançada')
    
    st.markdown("""
    **Análise RFV combinada com Clusterização** para segmentação mais precisa de clientes.
    Esta abordagem combina a simplicidade do RFV com o poder de machine learning não supervisionado.
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
    
    # Configurações de clusterização
    st.sidebar.header("⚙️ Configurações de Clusterização")
    n_clusters = st.sidebar.slider("Número de Clusters", 2, 8, 5)
    
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
            
            # --------------------------------------------------
            # ANÁLISE DE CLUSTERIZAÇÃO
            # --------------------------------------------------
            st.header("🎯 Análise de Clusterização")
            
            # Preparar dados para clusterização
            X = df_rfv[['Recencia', 'Frequencia', 'Valor']].copy()
            
            # Normalizar os dados
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Encontrar número ideal de clusters
            st.subheader("📊 Encontrando o Número Ideal de Clusters")
            
            inertias, silhouette_scores, k_range = find_optimal_clusters(X_scaled)
            
            # Gráfico do método do cotovelo
            fig_elbow = go.Figure()
            fig_elbow.add_trace(go.Scatter(x=k_range, y=inertias, mode='lines+markers', name='Inércia'))
            fig_elbow.update_layout(
                title='Método do Cotovelo - Inércia por Número de Clusters',
                xaxis_title='Número de Clusters',
                yaxis_title='Inércia'
            )
            
            # Gráfico do silhouette score
            fig_silhouette = go.Figure()
            fig_silhouette.add_trace(go.Scatter(x=k_range, y=silhouette_scores, mode='lines+markers', name='Silhouette Score'))
            fig_silhouette.update_layout(
                title='Silhouette Score por Número de Clusters',
                xaxis_title='Número de Clusters',
                yaxis_title='Silhouette Score'
            )
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_elbow, use_container_width=True)
            with col2:
                st.plotly_chart(fig_silhouette, use_container_width=True)
            
            # Aplicar K-Means com o número selecionado de clusters
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            df_rfv['Cluster'] = kmeans.fit_predict(X_scaled)
            
            # Calcular silhouette score final
            silhouette_avg = silhouette_score(X_scaled, df_rfv['Cluster'])
            st.metric("Silhouette Score", f"{silhouette_avg:.3f}")
            
            # Análise dos clusters
            st.subheader("📈 Análise dos Clusters")
            
            # Estatísticas por cluster - CÓDIGO CORRIGIDO
            cluster_stats = df_rfv.groupby('Cluster').agg({
                'Recencia': ['mean', 'std', 'min', 'max'],
                'Frequencia': ['mean', 'std', 'min', 'max'],
                'Valor': ['mean', 'std', 'min', 'max', 'sum']
            })
            
            # Adicionar contagem de clientes
            cluster_stats['Qtd_Clientes'] = df_rfv.groupby('Cluster').size()
            
            # Renomear colunas multi-level
            cluster_stats.columns = ['_'.join(col).strip() for col in cluster_stats.columns.values]
            cluster_stats = cluster_stats.round(2)
            
            st.dataframe(cluster_stats)
            
            # Visualização 3D dos clusters
            st.subheader("👁️ Visualização 3D dos Clusters")
            
            fig_3d = px.scatter_3d(
                df_rfv.reset_index(),
                x='Recencia',
                y='Frequencia',
                z='Valor',
                color='Cluster',
                hover_name='ID_cliente',
                title='Visualização 3D dos Clusters RFV',
                labels={'Recencia': 'Recência (dias)', 'Frequencia': 'Frequência', 'Valor': 'Valor (R$)'}
            )
            
            st.plotly_chart(fig_3d, use_container_width=True)
            
            # Distribuição dos clusters
            st.subheader("📊 Distribuição dos Clusters")
            
            fig_dist = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Recência por Cluster', 'Frequência por Cluster', 
                               'Valor por Cluster', 'Clientes por Cluster')
            )
            
            # Recência
            fig_dist.add_trace(
                go.Box(x=df_rfv['Cluster'], y=df_rfv['Recencia'], name='Recência'),
                row=1, col=1
            )
            
            # Frequência
            fig_dist.add_trace(
                go.Box(x=df_rfv['Cluster'], y=df_rfv['Frequencia'], name='Frequência'),
                row=1, col=2
            )
            
            # Valor
            fig_dist.add_trace(
                go.Box(x=df_rfv['Cluster'], y=df_rfv['Valor'], name='Valor'),
                row=2, col=1
            )
            
            # Contagem de clientes
            cluster_counts = df_rfv['Cluster'].value_counts().sort_index()
            fig_dist.add_trace(
                go.Bar(x=cluster_counts.index, y=cluster_counts.values, name='Clientes'),
                row=2, col=2
            )
            
            fig_dist.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Perfil de cada cluster
            st.subheader("👥 Perfil dos Clusters")
            
            for cluster_id in sorted(df_rfv['Cluster'].unique()):
                with st.expander(f"Cluster {cluster_id} - {len(df_rfv[df_rfv['Cluster'] == cluster_id])} clientes"):
                    cluster_data = df_rfv[df_rfv['Cluster'] == cluster_id]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Recência Média", f"{cluster_data['Recencia'].mean():.1f} dias")
                    with col2:
                        st.metric("Frequência Média", f"{cluster_data['Frequencia'].mean():.1f}")
                    with col3:
                        st.metric("Valor Médio", f"R$ {cluster_data['Valor'].mean():,.2f}")
                    with col4:
                        st.metric("Valor Total", f"R$ {cluster_data['Valor'].sum():,.2f}")
                    
                    st.write("**Top 5 RFV Scores neste cluster:**")
                    st.write(cluster_data['RFV_Score'].value_counts().head())
                    
                    st.write("**Ações de Marketing mais comuns:**")
                    st.write(cluster_data['Acao_Marketing'].value_counts().head())
            
            # Comparação RFV vs Clusterização
            st.header("🔄 Comparação: RFV vs Clusterização")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Segmentação RFV")
                rfv_counts = df_rfv['RFV_Score'].value_counts()
                fig_rfv = px.pie(values=rfv_counts.values, names=rfv_counts.index, 
                                title='Distribuição por Score RFV')
                st.plotly_chart(fig_rfv, use_container_width=True)
            
            with col2:
                st.subheader("Segmentação por Cluster")
                cluster_counts = df_rfv['Cluster'].value_counts()
                fig_cluster = px.pie(values=cluster_counts.values, 
                                    names=[f'Cluster {i}' for i in cluster_counts.index],
                                    title='Distribuição por Cluster')
                st.plotly_chart(fig_cluster, use_container_width=True)
            
            # Cruzamento RFV x Cluster
            st.subheader("📋 Cruzamento RFV x Cluster")
            cross_tab = pd.crosstab(df_rfv['RFV_Score'], df_rfv['Cluster'])
            st.dataframe(cross_tab.style.background_gradient(cmap='Blues'))
            
            # Download dos resultados
            st.header("📥 Download dos Resultados")
            
            excel_data = to_excel(df_rfv.reset_index())
            
            st.download_button(
                label="💾 Baixar Análise RFV + Clusters (Excel)",
                data=excel_data,
                file_name="analise_rfv_clusters.xlsx",
                mime="application/vnd.ms-excel"
            )
            
        except Exception as e:
            st.error(f"❌ Erro ao processar o arquivo: {str(e)}")
            st.write("Detalhes do erro:", e)
    
    else:
        st.info("👆 Faça upload de um arquivo CSV ou Excel para iniciar a análise")

if __name__ == '__main__':
    main()