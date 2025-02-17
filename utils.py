import math
import numpy as np
import pandas as pd
import torch

# Variáveis globais que definem o problema em uso
PROBLEM_NAME = None
MAXIMIZE = True
IS_MULTI = False

BTC_DF_1H = None  # armazena CSV para o caso "btc_demo"
NORM_OBJS = False
NORM_FACTOR = 1.0


def load_btc_data_once():
    """
    Carrega apenas 1 vez o CSV de BTC (por ex. "Binance_BTCUSDT_1h.csv"),
    limpando NaNs e mantendo no global BTC_DF_1H.
    Ajuste o caminho e skiprows se necessário.
    """
    global BTC_DF_1H
    if BTC_DF_1H is None:
        BTC_DF_1H = pd.read_csv("btc_formatted.csv")
        BTC_DF_1H.dropna(inplace=True)


def twoobj_demo(x: np.ndarray) -> np.ndarray:
    """
    Exemplo de função multiobjetivo com 2 objetivos:
      f1 = (Sphere) soma dos quadrados
      f2 = (Rastrigin-like) ou outro critério
    Retorna array [f1, f2].
    """
    x = x.flatten()
    # f1 = soma dos quadrados
    f1 = np.sum(x ** 2)

    # Exemplo f2: "rastrigin" (simplificado)
    A = 10
    n = len(x)
    f2 = A * n + np.sum(x * x - A * np.cos(2 * math.pi * x))

    return np.array([f1, f2], dtype=float)


def evaluate_solution(x: np.ndarray) -> float:
    """
    Exemplo de função single-objective (p. ex. 'sphere').
    Retorna escalar.
    """
    x = x.flatten()
    # Sphere
    val = np.sum(x ** 2)
    return val


def evaluate_solution_multi_btc(x: np.ndarray) -> np.ndarray:
    """
    x => 5 parâmetros [p0..p4], cada um em [-1..1].
    Em vez de valores aleatórios, calculamos TAKE_PROFIT e DRAWDOWN baseados nos preços.
    Exemplo simplificado:
      - Pegamos o 1o e último valor do dataset p/ simular "holding" do ativo
      - TAKE_PROFIT (%) = (Close - Open) / Open
      - DRAWDOWN (%)    = (High - Low) / High
    Se 'problem.normalize_objs' for true, multiplicamos ou dividimos pelos 'norm_factor'.

    Ajuste conforme desejar (ex.: sortear período, usar janelas, etc.).
    """
    global BTC_DF_1H, NORM_OBJS, NORM_FACTOR

    x = x.flatten()
    if BTC_DF_1H is None:
        load_btc_data_once()

    # Exemplo: Vamos usar o 1o e o último valor do CSV
    open_val = BTC_DF_1H["X0"].iloc[0]
    close_val = BTC_DF_1H["X1"].iloc[-1]

    # Para "drawdown", pegamos o maior High e o menor Low no período
    high_val = BTC_DF_1H["X2"].max()
    low_val = BTC_DF_1H["X3"].min()

    # TAKE_PROFIT => retorno percentual
    take_profit = (close_val - open_val) / open_val
    # DRAWDOWN => (High - Low)/High
    drawdown = (high_val - low_val) / high_val

    # Normaliza se user pediu (ex.: multiplicar por 100 ou outro fator)
    if NORM_OBJS:
        take_profit *= NORM_FACTOR
        drawdown *= NORM_FACTOR

    # Retornamos [take_profit, drawdown]
    return np.array([take_profit, drawdown], dtype=float)


def set_objective(name: str, maximize: bool, multi: bool):
    """
    Define o problema global: "twoobj_demo", "btc_demo", etc.
    Se for "btc_demo" e multi, carrega CSV BTC.
    """
    global PROBLEM_NAME, MAXIMIZE, IS_MULTI, NORM_OBJS, NORM_FACTOR
    PROBLEM_NAME = name
    MAXIMIZE = maximize
    IS_MULTI = multi

    if name == "btc_demo" and multi:
        load_btc_data_once()

def apply_config_for_norm(problem_cfg: dict):
    """
    Lê do config se iremos normalizar os objetivos e define no escopo global.
    Ex.: problem_cfg["normalize_objs"] e problem_cfg["norm_factor"].
    """
    global NORM_OBJS, NORM_FACTOR
    NORM_OBJS = problem_cfg.get("normalize_objs", False)
    NORM_FACTOR = problem_cfg.get("norm_factor", 1.0)


def evaluate_solution_multi(x: np.ndarray) -> np.ndarray:
    """
    Para problemas multi-objetivo, decide qual problema está ativo
    e chama a função apropriada para obter [obj1, obj2].
    """
    if PROBLEM_NAME == "twoobj_demo":
        return twoobj_demo(x)
    elif PROBLEM_NAME == "btc_demo":
        return evaluate_solution_multi_btc(x)
    else:
        # Default
        return twoobj_demo(x)


def init_worker(name, maximize, multi):
    """
    Inicializa variáveis globais no multiprocesso (usado em parallel),
    definindo PROBLEM_NAME, etc.
    """
    global PROBLEM_NAME, MAXIMIZE, IS_MULTI
    PROBLEM_NAME = name
    MAXIMIZE = maximize
    IS_MULTI = multi


class Individual:
    """
    Representa um indivíduo na população evolutiva.
    """
    def __init__(self, genome, fitness=None, mutation_rate=None):
        self.genome = genome
        self.fitness = fitness
        self.objective_val = None
        self.mutation_rate = mutation_rate

    def __repr__(self):
        return f"Individual(g={self.genome}, fit={self.fitness})"


def get_device(config):
    """
    Retorna 'mps' se device.use_mps=True e disponível,
    senão 'cuda' se disponível, senão 'cpu'.
    """
    use_mps = config["device"].get("use_mps", False)
    if use_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def worker_evaluate(genome_list):
    """
    Usado em multiprocessamento no modo single-objective.
    Converte genome_list -> np.array e chama evaluate_solution(...).
    """
    arr = np.array(genome_list, dtype=float)
    return evaluate_solution(arr)

class Individual:
    """
    Representa um indivíduo na população evolutiva.
    """
    def __init__(self, genome, fitness=None, mutation_rate=None):
        self.genome = genome
        self.fitness = fitness
        self.objective_val = None
        self.mutation_rate = mutation_rate

    def __repr__(self):
        return f"Individual(g={self.genome}, fit={self.fitness})"