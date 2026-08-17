# sheaf_financial_graph_analysis
Research project exploring sheaves and Connection Laplacian for financial time series. Includes preprocessing, similarity matrices, graph construction, scalar and vector sheaf Laplacians, spectral clustering, node classification by sector, and risk propagation analysis.

## Модули

| файл | назначение |
|---|---|
| `fetch_data.py` | загрузка котировок (MOEX ISS + Tiingo) |
| `preprocess.py` | парсинг Investing-CSV, дедупликация дат, единая сетка, лог-доходности |
| `similarity.py` | 5 мер близости, включая частные корреляции (precision matrix) |
| `build_graph.py` | корреляционный (порог) и kNN-граф, метрики узлов |
| `build_sheaf.py` | скалярный весовой пучок, restriction = √\|ρ\|·sign, спектр L₀, H⁰/H¹ |
| `connection_sheaf.py` | векторный пучок / Connection Laplacian: локальный PCA + Прокруст |
| `node_classify.py` | классификация секторов (признаки, label spreading, пучок) |
| `sheaf_cluster.py` | H1 (пучок vs граф по периодам), H2 (Фидлер → ядро/периферия), H3 (признаки) |
