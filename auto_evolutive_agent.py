import numpy as np
import random
import multiprocessing as mp
from copy import deepcopy

import evolution
import utils
# REMOVER: from utils import Individual
from multiobj_nsga2 import run_nsga2_surrogate
import metrics


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


class AutoEvolutiveAgent:
    def __init__(self, config):
        self.config = config
        prob_cfg = config["problem"]
        self.dim = prob_cfg["dimension"]
        self.bounds = prob_cfg["bounds"]
        self.maximize = prob_cfg["maximize"]
        self.is_multiobj = prob_cfg.get("multiobj", False)

        # Se quiser passar as configs para o utils, descomente se necessário:
        # utils.set_problem_config(prob_cfg)

        # Define objetivo
        utils.set_objective(prob_cfg["name"], self.maximize, self.is_multiobj)
        self.n_objs = 2 if self.is_multiobj else 1

        evo_cfg = config["evolution"]
        self.pop_size = evo_cfg["population_size"]
        self.generations = evo_cfg["generations"]
        self.selection_type = evo_cfg["selection"]["type"]
        self.tournament_size = evo_cfg["selection"].get("tournament_size", 3)
        self.crossover_rate = evo_cfg["crossover"]["rate"]
        self.crossover_type = evo_cfg["crossover"].get("type", "uniform")
        mut_cfg = evo_cfg["mutation"]
        self.mutation_adaptation = mut_cfg.get("adaptation", False)
        self.mutation_tau = mut_cfg.get("tau", 0.1)
        self.mutation_min = mut_cfg.get("min_rate", 0.001)
        self.mutation_max = mut_cfg.get("max_rate", 0.5)
        self.initial_mutation_rate = mut_cfg["initial_rate"]
        self.elitism = evo_cfg.get("elitism", 1)

        res_cfg = config["restart"]
        self.restart_enabled = res_cfg["enabled"]
        self.stagnation_gens = res_cfg["stagnation_generations"]
        self.restart_fraction = res_cfg["restart_fraction"]
        self.restart_reuse_best = res_cfg["reuse_best"]

        par_cfg = config["parallel"]
        self.num_workers = par_cfg["evaluate_workers"]

        self.device = utils.get_device(config)

        # Surrogate settings
        self.surrogate_use = config["surrogate"]["use"]
        self.surrogate_start_gen = config.get("start_generation", 1)
        self.surrogate_retrain_interval = config.get("retrain_interval", 2)
        self.real_phase = config.get("real_phase_generations", 3)

        if self.surrogate_use:
            from advanced_surrogate import AdvancedEnsembleSurrogate
            self.surrogate_model = AdvancedEnsembleSurrogate(self.dim, self.n_objs, config, self.device)
            pretrain_csv = config["surrogate"].get("pretrain_csv", None)
            if pretrain_csv:
                self.surrogate_model.pretrain_from_csv(pretrain_csv)
                self.surrogate_model.train_ensemble()
        else:
            self.surrogate_model = None

        # Inicializa população
        self.population = []
        for _ in range(self.pop_size):
            genome = np.random.uniform(self.bounds[0], self.bounds[1], size=self.dim)
            self.population.append(Individual(genome=genome, mutation_rate=self.initial_mutation_rate))

        # Best tracking
        if not self.is_multiobj:
            self.best_fitness = -float('inf') if self.maximize else float('inf')
            self.best_individual = None
        else:
            self.best_fitness = None
            self.best_individual = None

        self.real_evals_count = 0
        self.surrogate_evals_count = 0
        self.surrogate_errors = []

        # Métricas
        ref_point = config.get("hypervolume_ref", [100.0, 100.0])
        self.logger = metrics.MetricsLogger(ref_point=ref_point)
        self.logger.start_timer()


    def run(self):
        if self.is_multiobj:
            # Multi-objetivo => NSGA-II
            res = run_nsga2_surrogate(self.config, self.population, self.surrogate_model)

            # Copia contadores
            self.real_evals_count = getattr(res, "real_evals_count", 0)
            self.surrogate_evals_count = getattr(res, "surrogate_evals_count", 0)
            if hasattr(res, "surrogate_errors"):
                self.surrogate_errors = res.surrogate_errors
                for err in self.surrogate_errors:
                    self.logger.log_surrogate_error(err)

            self.logger.real_evals = self.real_evals_count
            self.logger.surrogate_evals = self.surrogate_evals_count

            if hasattr(res, "F") and res.F is not None:
                self.logger.compute_hypervolume(res.F)

            self.logger.stop_timer()
            return res
        else:
            # Single-objective (não exibido por completo aqui)
            best = self._run_single()
            self.logger.real_evals = self.real_evals_count
            self.logger.surrogate_evals = self.surrogate_evals_count
            for err in self.surrogate_errors:
                self.logger.log_surrogate_error(err)

            self.logger.stop_timer()
            return best

    def summary_metrics(self):
        return self.logger.summary()

    # -------------- Métodos Single-Objective omitidos --------------
    # -------------- Em multi-objetivo, usamos run_nsga2_surrogate --------------

    def _run_single(self):
        """
        Placeholder do run single-objective. (omisso)
        """
        pass

    def evaluate_population(self, pop, gen):
        """
        Avalia a população usando Surrogate se já estivermos além da fase real-only.
        Caso contrário, avalia com a função real.
        """
        # Se estiver na fase real-only ou Surrogate desativado
        if (not self.surrogate_model) or (gen < self.surrogate_start_gen + self.real_phase):
            self._evaluate_real(pop)
            return

        X = np.array([ind.genome for ind in pop])
        mean_pred, std_pred = self.surrogate_model.predict(X)

        if self.n_objs == 1:
            mean_pred = mean_pred.flatten()
            std_pred = std_pred.flatten()

        conf_mask = self.surrogate_model.is_confident(std_pred)
        for i, ind in enumerate(pop):
            if conf_mask[i]:
                # Surrogate confiante => avaliamos real 1 vez para medir erro
                if self.n_objs == 1:
                    pred_val = mean_pred[i]
                    real_obj = self._evaluate_one_real(ind.genome)
                    real_val = real_obj.fitness
                    err = abs(pred_val - real_val)
                    print(f"[DEBUG] SurrogateUsed => pred={pred_val:.4f}, real={real_val:.4f}, err={err:.4f}")

                    if self.surrogate_model.check_fallback(pred_val, real_val):
                        # Fallback se erro > threshold
                        ind.fitness = real_val
                        ind.objective_val = real_val if self.maximize else -real_val
                        self.real_evals_count += 1
                        print(f"[DEBUG] Fallback triggered => real={real_val:.4f}, err={err:.4f}")
                    else:
                        ind.fitness = pred_val
                        ind.objective_val = pred_val if self.maximize else -pred_val
                        self.surrogate_evals_count += 1
                        self.surrogate_errors.append(err)
                        self.logger.log_surrogate_error(err)
                else:
                    # Multi-objetivo
                    pred_val = mean_pred[i]
                    real_arr = self._evaluate_one_real_multi(ind.genome)
                    err = np.linalg.norm(pred_val - real_arr)
                    print(f"[DEBUG] SurrogateUsed(multi) => pred={pred_val}, real={real_arr}, err={err:.4f}")

                    if self.surrogate_model.check_fallback(pred_val, real_arr):
                        ind.fitness = np.sum(real_arr)
                        ind.objective_val = real_arr
                        self.real_evals_count += 1
                        print(f"[DEBUG] Fallback triggered (multi) => err={err:.4f}")
                    else:
                        ind.fitness = np.sum(pred_val)
                        ind.objective_val = pred_val
                        self.surrogate_evals_count += 1
                        self.surrogate_errors.append(err)
                        self.logger.log_surrogate_error(err)
            else:
                # Surrogate não confia => avaliação real
                if self.n_objs == 1:
                    real_obj = self._evaluate_one_real(ind.genome)
                    ind.fitness = real_obj.fitness
                    ind.objective_val = real_obj.objective_val
                    self.real_evals_count += 1
                else:
                    real_arr = self._evaluate_one_real_multi(ind.genome)
                    ind.fitness = np.sum(real_arr)
                    ind.objective_val = real_arr
                    self.real_evals_count += 1

    def _run_single(self):
        # Geração 0 => avalia de forma normal
        self.evaluate_population(self.population, 0)
        self._update_best()

        # Se temos Surrogate, alimenta com dados iniciais e treina
        if self.surrogate_model:
            X = np.array([ind.genome for ind in self.population])
            Y = np.array([ind.fitness for ind in self.population]).reshape(-1, 1)
            self.surrogate_model.update_data(X, Y)
            self.surrogate_model.retrain()
            if hasattr(self.surrogate_model, "X_train"):
                print(f"[DEBUG] Surrogate initial dataset size={len(self.surrogate_model.X_train)}")

        stagnation_count = 0
        last_best = self.best_fitness

        for gen in range(1, self.generations + 1):
            # Gera nova população
            new_pop = []
            sorted_pop = sorted(self.population, key=lambda i: i.fitness, reverse=self.maximize)
            elites = sorted_pop[:self.elitism]

            while len(new_pop) < (self.pop_size - len(elites)):
                p1 = self._select_parent()
                p2 = self._select_parent()
                child = self._crossover(p1, p2)
                child_gen, child_mut = self._mutate(child, p1, p2)
                new_pop.append(Individual(genome=child_gen, mutation_rate=child_mut))

            new_pop.extend(elites)
            self.population = new_pop

            # Avalia pop: Surrogate ou real
            self.evaluate_population(self.population, gen)

            # Retraining Surrogate em intervalos
            if self.surrogate_model and gen % self.surrogate_retrain_interval == 0:
                reals = [r for r in self.population if r.objective_val is not None]
                if reals:
                    X_ = np.array([r.genome for r in reals])
                    Y_ = np.array([r.fitness for r in reals]).reshape(-1, 1)
                    self.surrogate_model.update_data(X_, Y_)
                    self.surrogate_model.retrain()

                    if hasattr(self.surrogate_model, "X_train"):
                        print(f"[DEBUG] Surrogate dataset size={len(self.surrogate_model.X_train)} after gen={gen}")

            self._update_best()

            # Checa se houve melhora
            if not self._improved(last_best, self.best_fitness):
                stagnation_count += 1
            else:
                stagnation_count = 0
                last_best = self.best_fitness

            # Reinício parcial se estagnado
            if self.restart_enabled and stagnation_count >= self.stagnation_gens:
                self._partial_restart()
                stagnation_count = 0
                self.evaluate_population(self.population, gen)
                self._update_best()

                if self.surrogate_model:
                    reals2 = [r for r in self.population if r.objective_val is not None]
                    if reals2:
                        X2_ = np.array([r.genome for r in reals2])
                        Y2_ = np.array([r.fitness for r in reals2]).reshape(-1, 1)
                        self.surrogate_model.update_data(X2_, Y2_)
                        self.surrogate_model.retrain()
                        if hasattr(self.surrogate_model, "X_train"):
                            print(f"[DEBUG] Surrogate dataset size={len(self.surrogate_model.X_train)} after partial restart")

        return self.best_individual

    def summary_metrics(self):
        return self.logger.summary()

    def _evaluate_real(self, population):
        """
        Avalia a pop em paralelo (se num_workers>1) ou no modo single-thread.
        """
        if self.num_workers > 1:
            data = [ind.genome.tolist() for ind in population]
            with mp.Pool(self.num_workers) as pool:
                results = pool.map(utils.worker_evaluate, data)
            for i, r in enumerate(results):
                population[i].fitness = r
                population[i].objective_val = r if self.maximize else -r
            self.real_evals_count += len(population)
        else:
            for ind in population:
                obj = self._evaluate_one_real(ind.genome)
                ind.fitness = obj.fitness
                ind.objective_val = obj.objective_val
            self.real_evals_count += len(population)

    def _evaluate_one_real(self, genome):
        """
        Avalia single-objective (apenas p/ placeholder).
        """
        val = utils.evaluate_solution(genome)
        if self.maximize:
            obj_val = val
        else:
            obj_val = -val
        class Obj:
            def __init__(self, f, o):
                self.fitness = f
                self.objective_val = o
        return Obj(val, obj_val)

    def _evaluate_one_real_multi(self, genome):
        """
        Avalia multi-objetivo.
        """
        return utils.evaluate_solution_multi(genome)

    def _select_parent(self):
        # placeholder, usando self.population
        pass

    def _crossover(self, p1, p2):
        # placeholder
        pass

    def _mutate(self, child, p1, p2):
        # placeholder
        pass

    def _update_best(self):
        if not self.is_multiobj:
            # single-objective
            pass
        else:
            # multiobj => optional
            pass

    def _improved(self, old_best, new_best):
        if not self.is_multiobj:
            if self.maximize:
                return (new_best > old_best)
            else:
                return (new_best < old_best)
        else:
            # multi-obj => ex.: check se hipervolume subiu?
            return False

    def _partial_restart(self):
        # placeholder
        pass