# neural_model.py
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Configuração de logging básico (caso não haja outra configuração definida)
logger = logging.getLogger(__name__)
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

class NeuralModel:
    """
    Modelo de Rede Neural simples (perceptron multicamada) usado pelo agente (single-output).
    Consiste em uma camada oculta com ativação e uma camada de saída linear.
    (Mantido para compatibilidade com seu 'Agent' atual, se ele usa esse.)
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Inicializa o modelo neural com pesos aleatórios.
        :param input_size: Número de neurônios de entrada.
        :param hidden_size: Número de neurônios na camada oculta.
        :param output_size: Número de neurônios de saída.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # Inicialização aleatória (aqui usamos NumPy mas sem PyTorch)
        self.weights1 = np.random.rand(input_size, hidden_size)
        self.weights2 = np.random.rand(hidden_size, output_size)

        logger.debug(f"NeuralModel (single-output) init: input={input_size}, hidden={hidden_size}, output={output_size}.")
        logger.debug(f"Initial weights1 stats -> mean: {self.weights1.mean():.4f}, std: {self.weights1.std():.4f}")
        logger.debug(f"Initial weights2 stats -> mean: {self.weights2.mean():.4f}, std: {self.weights2.std():.4f}")

    def forward(self, inputs):
        """
        Forward pass (1 hidden layer tanh) + linear output.
        (Implementação NumPy pura - compatível com o Agent single-output)
        """
        x = np.array(inputs, dtype=float)
        if x.size != self.input_size:
            logger.error(f"Input size {x.size} != expected {self.input_size} for NeuralModel.")
            raise ValueError(f"NeuralModel expected input size {self.input_size}, got {x.size}.")
        x = x.flatten()
        hidden = np.tanh(np.dot(x, self.weights1))
        output = np.dot(hidden, self.weights2)
        logger.debug(f"Single-output forward pass: input={x}, hidden={hidden}, output={output}")
        return output

    def mutate(self, rate: float):
        """
        Aplica mutação aleatória nos pesos, com probabilidade 'rate' por peso.
        """
        mask1 = np.random.rand(*self.weights1.shape) < rate
        mask2 = np.random.rand(*self.weights2.shape) < rate
        changes1 = np.random.uniform(-0.5, 0.5, size=self.weights1.shape)
        changes2 = np.random.uniform(-0.5, 0.5, size=self.weights2.shape)
        self.weights1 += changes1 * mask1
        self.weights2 += changes2 * mask2
        mutated_count = int(mask1.sum() + mask2.sum())
        total_weights = self.weights1.size + self.weights2.size
        logger.debug(f"NeuralModel mutate: mutated {mutated_count}/{total_weights} weights (rate={rate}).")

# ------------------------------------------------------------------------------
# ABAIXO ESTÃO AS DEFINIÇÕES PARA MULTI-OUTPUT (usadas por surrogate.py)
# ------------------------------------------------------------------------------

class MultiOutputSurrogateNet(nn.Module):
    """
    Rede neural multi-output em PyTorch, com camadas definidas via nn.Module.
    Atende às necessidades do SurrogateEnsemble, prevendo n_objs saídas de uma só vez.
    """
    def __init__(self, input_dim, hidden_layers, activation, n_outputs):
        """
        :param input_dim: Dimensão de entrada.
        :param hidden_layers: Lista de tamanhos para as camadas ocultas (ex.: [64,32]).
        :param activation: 'relu', 'tanh' etc. para as camadas ocultas.
        :param n_outputs: Número de saídas (n_objs).
        """
        super(MultiOutputSurrogateNet, self).__init__()
        # Cria sequencialmente as camadas [Linear -> Ativação -> ... -> Linear -> (sem ativ. na saída)]
        layer_list = []
        act_map = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid
        }
        act_cls = act_map.get(activation.lower(), nn.ReLU)

        prev_dim = input_dim
        for hdim in hidden_layers:
            layer_list.append(nn.Linear(prev_dim, hdim))
            layer_list.append(act_cls())
            prev_dim = hdim

        # Camada final (n_outputs)
        layer_list.append(nn.Linear(prev_dim, n_outputs))

        self.net = nn.Sequential(*layer_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass do MLP.
        x shape: (batch_size, input_dim)
        Retorna shape (batch_size, n_outputs).
        """
        return self.net(x)


def train_multioutput_surrogate(X, Y, config, device, n_objs):
    """
    Treina a rede MultiOutputSurrogateNet com dados (X, Y).
    :param X: np.array shape (N, input_dim)
    :param Y: np.array shape (N, n_objs)
    :param config: dicionário com parâmetros (learning_rate, epochs, etc.)
    :param device: 'cpu', 'cuda' ou 'mps'
    :param n_objs: número de saídas
    :return: instância treinada de MultiOutputSurrogateNet
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim

    if X.shape[0]<2:
        # Dados insuficientes => retorna None
        print("[DEBUG] train_multioutput_surrogate: Not enough data.")
        return None

    hidden_layers = config["surrogate"].get("hidden_layers",[64,32])
    activation = config["surrogate"].get("activation","relu")
    lr = config["surrogate"].get("learning_rate",0.01)
    batch_size = config["surrogate"].get("batch_size",32)
    max_epochs = config["surrogate"].get("epochs",30)
    patience = config["surrogate"].get("early_stopping_patience",5)

    input_dim = X.shape[1]
    net = MultiOutputSurrogateNet(input_dim, hidden_layers, activation, n_objs).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    X_t = torch.tensor(X, dtype=torch.float32)
    Y_t = torch.tensor(Y, dtype=torch.float32)
    # Split train/val
    n = X_t.shape[0]
    val_count = int(0.1*n) if n>10 else 0
    if val_count>0:
        X_val = X_t[-val_count:].to(device)
        Y_val = Y_t[-val_count:].to(device)
        X_train = X_t[:-val_count].to(device)
        Y_train = Y_t[:-val_count].to(device)
    else:
        X_train = X_t.to(device)
        Y_train = Y_t.to(device)
        X_val = None
        Y_val = None

    best_val = float("inf")
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        # shuffle
        perm = torch.randperm(X_train.shape[0])
        X_train_epoch = X_train[perm]
        Y_train_epoch = Y_train[perm]

        net.train()
        running_loss = 0.0
        batch_count = 0
        # mini-batch loop
        for i in range(0, X_train_epoch.shape[0], batch_size):
            xb = X_train_epoch[i:i+batch_size]
            yb = Y_train_epoch[i:i+batch_size]
            optimizer.zero_grad()
            pred = net(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_count += 1
        
        train_loss_avg = running_loss / (batch_count if batch_count>0 else 1)

        # Validation
        if X_val is not None and X_val.shape[0]>0:
            net.eval()
            with torch.no_grad():
                pv = net(X_val)
                val_loss = criterion(pv, Y_val).item()
        else:
            val_loss = train_loss_avg

        if val_loss< best_val -1e-7:
            best_val = val_loss
            epochs_no_improve=0
        else:
            epochs_no_improve+=1

        print(f"[DEBUG] MultiOutputSurrogate training epoch {epoch+1}/{max_epochs} -> train_loss={train_loss_avg:.6f}, val_loss={val_loss:.6f}")

        if epochs_no_improve>=patience:
            print("[DEBUG] EarlyStopping triggered.")
            break

    return net