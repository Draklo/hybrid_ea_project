# MVP do Algoritmo Evolutivo Híbrido – Versão Final (Multi-Output Surrogate)

Este repositório contém a versão mais refinada do MVP do Algoritmo Evolutivo Híbrido, com:
- **Multi-Output Surrogate**: Um único modelo que gera todos os objetivos.
- **Conversão para Core ML**: Uso do Apple Neural Engine (ANE) em macOS.
- **NSGA-II com pymoo**
- **Multiprocessamento**
- **UI em Streamlit**
- **Deploy via Docker**

## Estrutura
- `main.py`: Entrada do programa.
- `config.yaml`: Configurações gerais.
- `auto_evolutive_agent.py`: Orquestra a otimização.
- `evolution.py`: Operadores genéticos.
- `surrogate.py` e `neural_model.py`: Modelo substituto multi-output.
- `multiobj_nsga2.py`: NSGA-II com pymoo.
- `benchmarks.py`: Testes e benchmarks.
- `streamlit_app.py`: Interface em Streamlit.
- `requirements.txt`: Dependências.
- `Dockerfile`: Deploy via Docker.

## Uso
1. Instale as dependências:
   ```bash
   python3 -m pip install -r requirements.txt