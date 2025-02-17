# main.py
import sys
import yaml
import os
import subprocess

from auto_evolutive_agent import AutoEvolutiveAgent

def main():
    config_path="config.yaml"
    if len(sys.argv)>1:
        config_path=sys.argv[1]
    if not os.path.isfile(config_path):
        print("config.yaml nao encontrado.")
        sys.exit(1)
    with open(config_path,'r') as f:
        config=yaml.safe_load(f)

    if config["ui"].get("streamlit",False):
        cmd=["streamlit","run","streamlit_app.py"]
        subprocess.run(cmd)
    else:
        agent=AutoEvolutiveAgent(config)
        best=agent.run()
        if agent.is_multiobj:
            print("MultiObj => ver pymoo result.")
        else:
            print(f"Best fitness: {agent.best_fitness}")
        print("Real Evals:", agent.real_evals_count," Surrogate Evals:", agent.surrogate_evals_count)
        if len(agent.surrogate_errors)>0:
            avg_err = sum(agent.surrogate_errors)/len(agent.surrogate_errors)
            print(f"Avg Surrogate error: {avg_err:.4f}")

if __name__=="__main__":
    main()