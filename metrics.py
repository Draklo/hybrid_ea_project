# metrics.py

import time
import numpy as np
from pymoo.indicators.hv import Hypervolume

class MetricsLogger:
    """
    Armazena e exibe métricas de execução:
      - tempo total,
      - contagem real evals e surrogate evals,
      - erro médio Surrogate,
      - hipervolume do front (opcional).
    """
    def __init__(self, ref_point=None, surrogate_enabled=False):
        """
        :param ref_point: ponto de referência p/ cálculo de hipervolume
        :param surrogate_enabled: indica se o Surrogate está ou não ativo
        """
        self.start_time = None
        self.end_time = None
        self.real_evals = 0
        self.surrogate_evals = 0
        self.surrogate_errors = []
        self.ref_point = ref_point
        self.final_hv = None
        self.hypervol_history = []
        self.surrogate_enabled = surrogate_enabled  # <--- novo campo

    def start_timer(self):
        self.start_time = time.time()

    def stop_timer(self, device="cpu"):
        """
        Para o cronômetro. Se o device for GPU/MPS, faz sincronização
        para garantir que todo o processamento terminou antes de pegar o tempo final.
        """
        if device != "cpu":
            try:
                import torch
                # Sincroniza se for cuda ou mps
                if "mps" in str(device).lower():
                    torch.mps.synchronize()
                elif "cuda" in str(device).lower():
                    torch.cuda.synchronize()
            except:
                pass

        self.end_time = time.time()

    def compute_time_total(self):
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

    def log_surrogate_error(self, err):
        self.surrogate_errors.append(err)

    def compute_avg_surrogate_error(self):
        """
        Se Surrogate não estiver habilitado ou não houver medições,
        retorna 'N/A'. Caso contrário, retorna média dos erros.
        """
        if not self.surrogate_enabled or len(self.surrogate_errors) == 0:
            return "N/A"
        return sum(self.surrogate_errors) / len(self.surrogate_errors)

    def compute_hypervolume(self, F):
        """
        Calcula o hipervolume (se ref_point e F existirem) e armazena em self.final_hv.
        Requer pymoo.
        """
        if self.ref_point is None or F is None or len(F) == 0:
            return None
        hv = Hypervolume(ref_point=self.ref_point)
        val = hv.do(F)
        self.final_hv = val
        return val

    def record_hypervolume(self, F):
        """
        Se quiser registrar HV a cada geração (não obrigatório).
        """
        val = self.compute_hypervolume(F)
        if val is not None:
            self.hypervol_history.append(val)

    def summary(self):
        """
        Retorna dict com as principais métricas.
        """
        d = {}
        d["TimeTotal"] = self.compute_time_total()
        d["RealEvals"] = self.real_evals
        d["SurrogateEvals"] = self.surrogate_evals
        d["AvgSurrogateError"] = self.compute_avg_surrogate_error()
        d["FinalHV"] = self.final_hv
        return d