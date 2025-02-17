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
    def __init__(self, ref_point=None):
        """
        :param ref_point: ponto de referência para cálculo do hipervolume 
                          (ex.: [pior_obj1, pior_obj2]).
        """
        self.start_time = None
        self.end_time = None
        self.real_evals = 0
        self.surrogate_evals = 0
        self.surrogate_errors = []
        self.ref_point = ref_point
        self.final_hv = None
        self.hypervol_history = []  # se quiser registrar HV a cada geração

    def start_timer(self):
        self.start_time = time.time()

    def stop_timer(self):
        self.end_time = time.time()

    def compute_time_total(self):
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

    def log_surrogate_error(self, err):
        self.surrogate_errors.append(err)

    def compute_avg_surrogate_error(self):
        if len(self.surrogate_errors) == 0:
            return 0.0
        return sum(self.surrogate_errors) / len(self.surrogate_errors)

    def compute_hypervolume(self, F):
        """
        Calcula o hipervolume (se ref_point e F existirem).
        Salva em self.final_hv.
        Necessita: pip install pymoo
        """
        if self.ref_point is None or F is None or len(F) == 0:
            return None
        hv = Hypervolume(ref_point=self.ref_point)
        val = hv.do(F)
        self.final_hv = val
        return val

    def record_hypervolume(self, F):
        """
        Se quiser registrar o HV a cada geração.
        """
        val = self.compute_hypervolume(F)
        if val is not None:
            self.hypervol_history.append(val)

    def summary(self):
        """
        Retorna dict com as principais métricas.
        Exemplo: {
           "TimeTotal": 26.7,
           "RealEvals": 1500,
           "SurrogateEvals": 1446,
           "AvgSurrogateError": 12.3,
           "FinalHV": 42.1
        }
        """
        d = {}
        d["TimeTotal"] = self.compute_time_total()
        d["RealEvals"] = self.real_evals
        d["SurrogateEvals"] = self.surrogate_evals
        d["AvgSurrogateError"] = self.compute_avg_surrogate_error()
        d["FinalHV"] = self.final_hv
        return d