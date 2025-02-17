# evolution.py
import logging
import random
import numpy as np
from neural_model import NeuralModel
from utils import evaluate_solution, evaluate_solution_multi, twoobj_demo

# Configuração de logging básico (caso não haja outra configuração definida)
logger = logging.getLogger(__name__)
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

class Agent:
    """Representa um agente com um modelo neural e um valor de fitness associado."""
    _id_counter = 0  # contador para atribuir IDs únicos a cada agente

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Inicializa um novo agente com um modelo neural aleatório e fitness zero.
        :param input_size: Número de entradas do modelo neural.
        :param hidden_size: Número de neurônios na camada oculta do modelo.
        :param output_size: Número de saídas do modelo neural.
        """
        # Atribui ID único ao agente
        self.id = Agent._id_counter
        Agent._id_counter += 1

        # Cria o modelo neural do agente
        self.model = NeuralModel(input_size, hidden_size, output_size)
        self.fitness = 0.0  # fitness inicial do agente (0 até ser avaliado)

        logger.debug(f"Created Agent {self.id} with a new NeuralModel.")

    def evaluate(self, environment) -> float:
        """
        Avalia o agente no ambiente fornecido, executando ações até o término do episódio.
        Atualiza e retorna o fitness obtido.
        O ambiente deve prover métodos reset() e step(action), onde step retorna a recompensa por ação.
        """
        logger.debug(f"Evaluating Agent {self.id} in the given environment.")
        # Reinicia o ambiente e obtém o estado inicial, se o método reset existir
        state = None
        if hasattr(environment, 'reset'):
            state = environment.reset()
        if state is None and hasattr(environment, 'get_state'):
            # Se não há reset, tenta obter um estado inicial de outro modo
            state = environment.get_state()

        self.fitness = 0.0
        # Loop de interação agente-ambiente até o fim do episódio
        done = False
        step_count = 0
        while not done:
            # Obtém a ação do agente para o estado atual
            action = self.model.forward(state)
            # Executa a ação no ambiente
            if hasattr(environment, 'step'):
                result = environment.step(action)
            else:
                # Se o ambiente não tem step, tenta chamar um método genérico (por exemplo, simulate_one_step)
                result = environment.simulate_one_step(action)
            # O retorno pode ser (next_state, reward, done, info) como em OpenAI Gym
            if isinstance(result, tuple) and len(result) >= 2:
                # Suporta retorno padrão (state, reward, done, info)
                state = result[0]
                reward = result[1]
                done = result[2] if len(result) > 2 else False
            else:
                # Se step() retorna somente a recompensa acumulada ou objeto, tenta extrair
                reward = result
                # Não temos próxima state/done; assume-se que uma iteração representa o episódio inteiro
                done = True

            # Atualiza o fitness acumulando a recompensa obtida
            try:
                self.fitness += float(reward)
            except Exception as e:
                logger.error(f"Agent {self.id}: error converting reward to float -> {e}")
            step_count += 1
            # Se o ambiente não fornece explicitamente o término, interrompe após um número seguro de iterações (para evitar loop infinito)
            if step_count > 10000 and not done:
                logger.warning(f"Agent {self.id}: stopping evaluation after 10000 steps (possible infinite loop).")
                break

        logger.debug(f"Agent {self.id} evaluation completed with fitness = {self.fitness}.")
        return self.fitness

    def mutate(self, rate: float):
        """
        Aplica mutação no modelo neural do agente com a taxa especificada.
        :param rate: Taxa de mutação (probabilidade de mutar cada peso).
        """
        # Delegação da mutação para o modelo neural do agente
        self.model.mutate(rate)
        logger.debug(f"Agent {self.id}: model mutated with rate {rate}.")

    def crossover(self, other):
        """
        Realiza crossover (combinação) entre este agente e outro, gerando um novo agente filho.
        O crossover é uniforme: cada peso do filho vem aleatoriamente de um dos pais.
        :param other: Outro agente (pai 2) para cruzamento.
        :return: Novo Agent resultante do crossover dos pais.
        """
        if not isinstance(other, Agent):
            logger.error("Crossover requires another Agent instance.")
            raise TypeError("Other must be an Agent.")
        # Verifica se os modelos têm a mesma arquitetura
        if (self.model.input_size != other.model.input_size or 
                self.model.hidden_size != other.model.hidden_size or 
                self.model.output_size != other.model.output_size):
            logger.error("Cannot crossover agents with different neural model architectures.")
            raise ValueError("Incompatible agents for crossover (different network sizes).")

        # Cria um novo agente filho (com modelo neural inicial aleatório que será sobrescrito)
        child = Agent(self.model.input_size, self.model.hidden_size, self.model.output_size)
        # Combinação dos pesos dos pais no filho
        # Máscaras aleatórias para decidir de qual pai herdar cada peso
        mask1 = np.random.rand(*self.model.weights1.shape) < 0.5
        mask2 = np.random.rand(*self.model.weights2.shape) < 0.5
        # Aplica crossover uniforme em cada matriz de pesos
        child.model.weights1 = np.where(mask1, self.model.weights1, other.model.weights1)
        child.model.weights2 = np.where(mask2, self.model.weights2, other.model.weights2)
        # Fitness do filho começa em 0 (não avaliado ainda)
        child.fitness = 0.0

        # Loga informações sobre o crossover
        total_weights = child.model.weights1.size + child.model.weights2.size
        inherited_from_self = int(mask1.sum() + mask2.sum())
        inherited_from_other = total_weights - inherited_from_self
        logger.debug(f"Crossover between Agent {self.id} and Agent {other.id} -> Child Agent {child.id}: " +
                     f"{inherited_from_self} weights from Agent {self.id}, {inherited_from_other} from Agent {other.id}.")

        return child

    def clone(self):
        """
        Clona este agente, criando um novo agente com a mesma configuração de pesos.
        :return: Um novo Agent idêntico (mesmos pesos do modelo neural).
        """
        # Cria um novo agente com a mesma arquitetura
        clone_agent = Agent(self.model.input_size, self.model.hidden_size, self.model.output_size)
        # Copia os pesos do modelo neural para o clone
        clone_agent.model.weights1 = np.copy(self.model.weights1)
        clone_agent.model.weights2 = np.copy(self.model.weights2)
        # Opcionalmente, copia o fitness atual (pode ser reavaliado se necessário)
        clone_agent.fitness = self.fitness
        logger.debug(f"Cloned Agent {self.id} to new Agent {clone_agent.id} with identical neural network weights.")
        return clone_agent

    def __repr__(self):
        return f"<Agent id={self.id}, fitness={self.fitness}>"


class AutoEvolution:
    """
    Gerencia o algoritmo evolutivo para otimizar agentes neurocontrolados.
    Cria uma população de agentes e itera através de gerações aplicando seleção, crossover e mutação.
    """
    def __init__(self, population_size: int, input_size: int, hidden_size: int, output_size: int,
                 mutation_rate: float = 0.1, environment=None):
        """
        Inicializa o processo evolutivo com uma população de agentes.
        :param population_size: Tamanho da população de agentes.
        :param input_size: Número de entradas do modelo neural de cada agente.
        :param hidden_size: Número de neurônios na camada oculta do modelo neural.
        :param output_size: Número de saídas do modelo neural.
        :param mutation_rate: Taxa de mutação a ser utilizada em cada geração (padrão 0.1).
        :param environment: (Opcional) ambiente de simulação a ser usado para avaliar os agentes.
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generation = 0
        self.environment = environment  # pode ser definido agora ou fornecido ao chamar run()
        # Inicializa a população de agentes
        self.population = [Agent(input_size, hidden_size, output_size) for _ in range(population_size)]
        # Variáveis para acompanhar o melhor agente encontrado
        self.best_agent = None
        self.best_fitness = float('-inf')

        logger.info(f"Initialized AutoEvolution: population={population_size}, "
                    f"network_architecture=({input_size}, {hidden_size}, {output_size}), "
                    f"mutation_rate={mutation_rate}.")
        if environment is not None:
            logger.info(f"Environment set at initialization: {environment}")

    def run(self, generations: int, environment=None):
        """
        Executa o algoritmo evolutivo por um número especificado de gerações.
        :param generations: Quantidade de gerações a evoluir.
        :param environment: (Opcional) Ambiente de simulação a ser utilizado. 
                             Se não fornecido, usa o ambiente definido no __init__.
        :return: O melhor agente encontrado ao final das gerações.
        """
        if environment is None:
            # Usa o ambiente armazenado na instância, se disponível
            environment = self.environment
        if environment is None:
            logger.error("No environment provided for evolution.")
            raise ValueError("An environment must be provided to run evolution.")

        for gen in range(1, generations + 1):
            self.generation = gen
            logger.info(f"\n=== Generation {gen}/{generations} ===")

            # Etapa 1: Avaliação de todos os agentes da população
            logger.debug("Evaluating all agents in the population.")
            for agent in self.population:
                agent.evaluate(environment)

            # Ordena a população pelo valor de fitness (decrescente, melhor primeiro)
            self.population.sort(key=lambda ag: ag.fitness, reverse=True)
            current_best = self.population[0]
            logger.info(f"Generation {gen}: Best fitness = {current_best.fitness} (Agent {current_best.id})")

            # Atualiza o melhor global se o da geração atual for superior
            if current_best.fitness > self.best_fitness:
                self.best_fitness = current_best.fitness
                self.best_agent = current_best
                logger.info(f"New overall best found: Agent {current_best.id} with fitness {current_best.fitness} in generation {gen}.")

            # Etapa 2: Seleção e Reprodução (gerar nova população), exceto na última geração
            if gen < generations:
                logger.debug("Selecting top agents and generating offspring for next generation.")
                # Seleção: mantém os 2 melhores (elitismo)
                parents = self.population[:2]
                new_population = [parents[0], parents[1]]  # elitismo: copia os melhores diretamente
                # Reprodução: preenche o restante da nova população
                while len(new_population) < self.population_size:
                    # Escolhe aleatoriamente dois pais dentre os melhores
                    parent1 = random.choice(parents)
                    parent2 = random.choice(parents)
                    # Gera um filho por crossover dos pais selecionados
                    child = parent1.crossover(parent2)
                    # Aplica mutação no filho
                    child.mutate(self.mutation_rate)
                    new_population.append(child)
                # Define a nova população para a próxima geração
                self.population = new_population
                logger.debug(f"Generation {gen} completed. Population for next generation prepared (size {len(self.population)}).")

        logger.info(f"Evolution completed after {generations} generations. "
                    f"Best fitness achieved = {self.best_fitness} by Agent {self.best_agent.id}.")
        return self.best_agent