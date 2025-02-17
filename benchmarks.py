# benchmarks.py
import time
import yaml
import numpy as np
from auto_evolutive_agent import AutoEvolutiveAgent

def test_single_rastrigin():
    with open("config.yaml","r") as f:
        cfg=yaml.safe_load(f)
    cfg["problem"]["name"]="rastrigin"
    cfg["problem"]["multiobj"]=False
    cfg["problem"]["maximize"]=False
    cfg["surrogate"]["use"]=True
    agent=AutoEvolutiveAgent(cfg)
    start=time.time()
    best=agent.run()
    elapsed=time.time()-start
    print("[Single Ras/Surrogate] best fitness=%.4f, time=%.2fs" % (best.fitness, elapsed))

def test_twoobj_surrogate():
    with open("config.yaml",'r') as f:
        cfg=yaml.safe_load(f)
    cfg["problem"]["name"]="twoobj_demo"
    cfg["problem"]["multiobj"]=True
    cfg["surrogate"]["use"]=True
    agent=AutoEvolutiveAgent(cfg)
    start=time.time()
    res=agent.run()
    elapsed=time.time()-start
    print("[MultiObj - twoobj_demo Surrogate] time=%.2fs" % elapsed)
    print("Resultado pymoo:", res.F.shape if res else "No result")

if __name__=="__main__":
    test_single_rastrigin()
    test_twoobj_surrogate()