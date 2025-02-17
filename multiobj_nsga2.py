# multiobj_nsga2.py

import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.optimize import minimize

# ATENÇÃO: agora precisamos do NonDominatedSorting de outro lugar?
# Se estiver funcionando, mantenha como está. 
# Se deu erro "no module named non_dominated_sort", faça as adaptações.
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

# Indicadores de performance
from pymoo.indicators.hv import Hypervolume
from pymoo.indicators.igd import IGD
# Se quiser GD, IGD+, etc., importe também

import utils


##################################
# Definição local de CallbackList
##################################
class Callback:
    """Interface base para callbacks."""
    def notify(self, algorithm):
        pass


class CallbackList:
    """Recebe uma lista de callbacks e chama cada um deles ao final de cada geração."""
    def __init__(self, callbacks):
        self.callbacks = callbacks if callbacks else []

    def __call__(self, algorithm):
        for cb in self.callbacks:
            cb.notify(algorithm)


##################################
# Surrogate Problem
##################################
class SurrogateMOProblem(Problem):
    """
    Problem multiobjetivo com Surrogate + fallback, usando 'twoobj_demo' ou 'btc_demo'.
    """

    def __init__(self, config, surrogate_model):
        self.config = config
        self.surrogate_model = surrogate_model

        prob = config["problem"]
        self.n_var = prob["dimension"]
        bmin, bmax = prob["bounds"]
        # Supondo 2 objetivos. Ajuste caso precise de outro n_obj:
        self.n_obj = 2

        self.real_evals_count = 0
        self.surrogate_evals_count = 0

        # Buffer para re-treino do Surrogate
        self.X_reals = []
        self.Y_reals = []

        # Lista interna para armazenar erros do Surrogate
        self.surrogate_errors = []

        super().__init__(
            n_var=self.n_var,
            n_obj=self.n_obj,
            n_constr=0,
            xl=np.array([bmin] * self.n_var),
            xu=np.array([bmax] * self.n_var),
            vectorized=False
        )

    def _evaluate(self, x, out, *args, **kwargs):
        # Avalia 1 ou vários indivíduos
        if x.ndim == 1:
            out["F"] = self._evaluate_one(x)
        else:
            F = []
            for row in x:
                F.append(self._evaluate_one(row))
            out["F"] = np.array(F)

    def _evaluate_one(self, x_1d):
        """Avalia um único indivíduo (x_1d) de forma real ou via Surrogate."""
        # Se Surrogate não está ativo, faz avaliação real
        if not self.surrogate_model:
            real_val = utils.evaluate_solution_multi(x_1d)
            self.real_evals_count += 1
            self.X_reals.append(x_1d)
            self.Y_reals.append(real_val)
            return real_val

        # Surrogate ativo
        X = np.array([x_1d])
        mean_pred, std_pred = self.surrogate_model.predict(X)
        conf_mask = self.surrogate_model.is_confident(std_pred)

        if conf_mask[0]:
            # Surrogate confiante => compara com real (fallback se erro > threshold)
            pred_val = mean_pred[0]
            real_val = utils.evaluate_solution_multi(x_1d)
            self.real_evals_count += 1
            self.X_reals.append(x_1d)
            self.Y_reals.append(real_val)

            err = np.linalg.norm(pred_val - real_val)
            self.surrogate_errors.append(err)

            if self.surrogate_model.check_fallback(pred_val, real_val):
                # fallback => usa valor real
                return real_val
            else:
                # Surrogate ok => conta Surrogate e registra erro
                self.surrogate_evals_count += 1
                return pred_val
        else:
            # Surrogate não confia => avaliação real
            real_val = utils.evaluate_solution_multi(x_1d)
            self.real_evals_count += 1
            self.X_reals.append(x_1d)
            self.Y_reals.append(real_val)
            return real_val


##################################
# SurrogateCallback
##################################
class SurrogateCallback(Callback):
    """
    Re-treina o Surrogate a cada 'retrain_interval' gerações,
    limpando buffers X_reals/Y_reals após.
    """
    def __init__(self, problem, surrogate_model, retrain_interval=1):
        super().__init__()
        self.problem = problem
        self.surrogate_model = surrogate_model
        self.retrain_interval = retrain_interval

    def notify(self, algorithm):
        gen = algorithm.n_gen
        if self.surrogate_model and gen % self.retrain_interval == 0:
            X_ = np.array(self.problem.X_reals)
            Y_ = np.array(self.problem.Y_reals)
            if len(X_) > 0:
                self.surrogate_model.update_data(X_, Y_)
                self.surrogate_model.retrain()
                # limpa buffer
                self.problem.X_reals.clear()
                self.problem.Y_reals.clear()


##################################
# LocalSearchCallback (Opcional)
##################################
class LocalSearchCallback(Callback):
    """
    Exemplo de busca local leve (Memetic).
    """
    def __init__(self, ls_top_k=3, ls_steps=3, step_size=0.02):
        super().__init__()
        self.ls_top_k = ls_top_k
        self.ls_steps = ls_steps
        self.step_size = step_size

    def local_search_one(self, x_1d):
        """
        Tenta pequenas perturbações p/ ver se melhora soma de objetivos.
        Exemplo p/ 2 objs => minimização da soma(F).
        """
        from utils import evaluate_solution_multi
        best_x = x_1d.copy()
        best_val = evaluate_solution_multi(best_x)
        best_fit = np.sum(best_val)

        for _ in range(self.ls_steps):
            trial = best_x + np.random.uniform(-self.step_size, self.step_size, size=best_x.shape)
            t_val = evaluate_solution_multi(trial)
            t_fit = np.sum(t_val)
            if t_fit < best_fit:
                best_fit = t_fit
                best_x = trial
        return best_x

    def notify(self, algorithm):
        pop = algorithm.pop
        F = pop.get("F")  # shape (N, n_obj)
        sums = np.sum(F, axis=1)
        sorted_idx = np.argsort(sums)  # ascending => top first

        top_k_idx = sorted_idx[:self.ls_top_k]
        for idx in top_k_idx:
            x_before = pop[idx].X
            new_x = self.local_search_one(x_before)
            pop[idx].X = new_x
            new_F = utils.evaluate_solution_multi(new_x)
            pop[idx].F = new_F


##################################
# Função principal: run_nsga2_surrogate
##################################
def run_nsga2_surrogate(config, population, surrogate_model):
    """
    Executa NSGA-II (pymoo) usando Surrogate (multiobjetivo).
    :param config: dict com configurações
    :param population: população inicial (não necessariamente usada se iremos
                       deixar o pymoo criar a pop interna).
    :param surrogate_model: Surrogate ou None
    :return: objeto `res` do pymoo (res.F, etc.), acrescido de:
             - nondomF / nondomX,
             - hv_nondom (hipervolume),
             - igd_nondom, spread_nondom (exemplo de outras métricas).
    """

    # Cria a instância de Problem
    problem = SurrogateMOProblem(config, surrogate_model)

    # Configura NSGA-II
    algo = NSGA2(
        pop_size=config["evolution"]["population_size"],
        eliminate_duplicates=False
    )

    # Termination
    termination = get_termination("n_gen", config["evolution"]["generations"])

    # Callbacks
    callbacks = []
    retrain_int = config.get("retrain_interval", 1)
    cb_surro = SurrogateCallback(problem, surrogate_model, retrain_interval=retrain_int)
    callbacks.append(cb_surro)

    # Local Search (opcional)
    ls_top_k = config.get("local_search_top_k", 0)
    ls_steps = config.get("local_search_steps", 0)
    step_size = config.get("local_search_step_size", 0.01)
    if ls_top_k > 0 and ls_steps > 0:
        cb_ls = LocalSearchCallback(ls_top_k=ls_top_k, ls_steps=ls_steps, step_size=step_size)
        callbacks.append(cb_ls)

    # Minimização (pymoo)
    res = minimize(
        problem,
        algo,
        termination,
        seed=42,
        callback=CallbackList(callbacks),
        verbose=False
    )

    # Copiamos contadores do Surrogate
    res.real_evals_count = problem.real_evals_count
    res.surrogate_evals_count = problem.surrogate_evals_count
    res.surrogate_errors = problem.surrogate_errors

    # Descobrimos a parte não-dominada na população final
    pop = res.pop
    F = pop.get("F")
    nds = NonDominatedSorting().do(F, only_non_dominated_front=True)
    nondomF = F[nds]
    nondomX = pop.get("X")[nds]
    res.nondomF = nondomF
    res.nondomX = nondomX

    # Exemplo: computar HV somente nessa fronteira
    ref_point = config.get("hypervolume_ref", [100.0,100.0])
    hv_calc = Hypervolume(ref_point=ref_point)
    hv_val = hv_calc.do(nondomF)
    res.hv_nondom = hv_val

    # Exemplo: IGD => Precisamos de 'pf=...' (o PF de referência real).
    # Se não existir PF real conhecido, use outro array ou remova
    # Abaixo, só calculo IGD usando a própria nondomF como "pf".
    # (Vai dar zero ou muito perto de zero, mas serve de demo)
    igd_calc = IGD(pf=nondomF)
    igd_val = igd_calc.do(nondomF)
    res.igd_nondom = igd_val

    # Se quiser Spread, ou GD, etc., importe e calcule:
    # from pymoo.indicators.spacing import Spacing
    # sp = Spacing(nondomF)
    # spread_val = sp.do(nondomF)
    # res.spread_nondom = spread_val

    return res



