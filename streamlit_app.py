# streamlit_app.py

import streamlit as st
import yaml
import time
from auto_evolutive_agent import AutoEvolutiveAgent

def main():
    st.title("MVP Evolutivo - Ensemble Surrogate + Métricas Avançadas")
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # Opções de UI para teste rápido
    is_multi = st.checkbox("MultiObj?", value=cfg["problem"].get("multiobj", False))
    use_surrogate = st.checkbox("Usar Surrogate?", value=cfg["surrogate"]["use"])
    pop_size = st.slider("População", 10, 300, cfg["evolution"]["population_size"])
    n_gen = st.slider("Gerações", 1, 200, cfg["evolution"]["generations"])

    if st.button("Iniciar"):
        # Atualiza config
        cfg["problem"]["multiobj"] = is_multi
        cfg["surrogate"]["use"] = use_surrogate
        cfg["evolution"]["population_size"] = pop_size
        cfg["evolution"]["generations"] = n_gen

        st.write("### Rodando AutoEvolutiveAgent...")

        agent = AutoEvolutiveAgent(cfg)
        result = agent.run()  # Pode demorar dependendo do Surrogate etc.

        # Extrai as métricas
        metrics_dict = agent.summary_metrics()

        st.write("#### Métricas Principais:")
        st.write("Tempo total (s):", metrics_dict["TimeTotal"])
        st.write("Real Evals:", metrics_dict["RealEvals"])
        st.write("Surrogate Evals:", metrics_dict["SurrogateEvals"])

        if metrics_dict["RealEvals"] > 0:
            ratio = metrics_dict["SurrogateEvals"] / metrics_dict["RealEvals"]
            st.write(f"Proporção Surrogate/Real: {ratio:.2f}")

        # Exibição diferenciada do erro Surrogate se == "N/A"
        err_surr = metrics_dict["AvgSurrogateError"]
        if err_surr == "N/A":
            st.write("Erro Médio Surrogate: N/A (Surrogate não usado ou sem medições)")
        else:
            st.write("Erro Médio Surrogate:", f"{err_surr:.4f}")

        # Se usou hypervolume
        if metrics_dict["FinalHV"] is not None:
            st.write("Hypervolume (Global):", metrics_dict["FinalHV"])

        # Caso seja multiobj e tenhamos a fronteira em result
        if is_multi and hasattr(result, "F"):
            st.write("População Final - shape:", result.F.shape)
            if hasattr(result, "nondomF"):
                st.write("Fronteira Pareto Não-Dominada - shape:", result.nondomF.shape)
                if hasattr(result, "hv_nondom") and (result.hv_nondom is not None):
                    st.write("Hipervolume (só ND):", result.hv_nondom)
                if hasattr(result, "igd_nondom") and (result.igd_nondom is not None):
                    st.write("IGD (só ND):", f"{result.igd_nondom:.4f}")
            else:
                st.write("Nenhuma info de fronteira não-dominada (nondomF).")
        else:
            st.write("Solução single-obj ou sem 'F' no result.")

if __name__ == "__main__":
    main()