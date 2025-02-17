import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import coremltools as ct
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from neural_model import MultiOutputSurrogateNet, train_multioutput_surrogate

class SurrogateEnsemble:
    """
    Surrogate multi-output.
    Se config['surrogate']['convert_coreml']==True, tenta converter p/ CoreML.
    """
    def __init__(self, input_dim, n_objs, config, device):
        self.config = config
        self.device = device
        self.multi_output = config["surrogate"].get("multi_output", False)
        self.n_objs = n_objs

        # Exemplo de kernel p/ um GP (opcional, se quiser usar)
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-6)
        self.gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=2)

        self.X_train = None
        self.y_train = None
        # Limiar de fallback
        self.fallback_error_thresh = config["surrogate"].get("fallback_error_thresh", 2.0)

        self.convert_coreml = config["surrogate"].get("convert_coreml", False)
        self.coreml_model = None
        self.input_dim = input_dim

        self.net = None  # A rede neural (MLP) propriamente dita

    def update_data(self, X, Y):
        """
        Adiciona (X, Y) ao buffer de treinamento (para re-treino posterior).
        X: shape (N, input_dim)
        Y: shape (N, n_objs)
        """
        if self.X_train is None:
            self.X_train = X
            self.y_train = Y
        else:
            self.X_train = np.vstack([self.X_train, X])
            self.y_train = np.vstack([self.y_train, Y])

    def retrain(self):
        """
        Re-treina o Surrogate (rede MLP multi-output), usando
        train_multioutput_surrogate(...). Converte p/ Core ML se configurado.
        """
        if self.X_train is None or len(self.X_train) < 2:
            print("[DEBUG] SurrogateEnsemble: No data to retrain.")
            return

        print(f"[DEBUG] SurrogateEnsemble: retraining on {len(self.X_train)} samples.")

        # Chama a função de treinamento do neural_model.py
        self.net = train_multioutput_surrogate(
            self.X_train,   # shape (N, input_dim)
            self.y_train,   # shape (N, n_objs)
            self.config, 
            self.device, 
            self.n_objs
        )
        if self.convert_coreml and (self.net is not None):
            self.coreml_model = self._convert_to_coreml_multi(self.net)

    def _convert_to_coreml_multi(self, net, name="MultiOutSurrogate"):
        """
        Converte a rede PyTorch para Core ML (MLProgram), para rodar no ANE se possível.
        """
        net.eval()
        example_in = torch.rand(1, self.input_dim, dtype=torch.float32, device=self.device)
        traced = torch.jit.trace(net, example_in)
        mlmodel = ct.convert(
            traced,
            inputs=[ct.TensorType(shape=example_in.shape, dtype=float, name="features")],
            convert_to="mlprogram",
            compute_units=ct.ComputeUnit.ALL
        )
        mlmodel.short_description = "MultiOutput Surrogate for Evolutionary GA"
        return mlmodel

    def predict(self, X):
        """
        Retorna (mean, std) => shape(N, n_objs).
        Se não tiver rede treinada, devolve zero + std grande (9999).
        """
        if self.net is None:
            n = len(X) if (X.ndim>1) else 1
            return (np.zeros((n, self.n_objs)), np.ones((n, self.n_objs))*9999.0)

        if self.convert_coreml and (self.coreml_model is not None):
            return self._predict_coreml(X)
        else:
            return self._predict_pytorch(X)

    def _predict_pytorch(self, X):
        """
        Faz inferência pela rede PyTorch no device especificado.
        """
        X = np.array(X, dtype=float)
        single = False
        if X.ndim == 1:
            X = X[np.newaxis, :]
            single = True

        with torch.no_grad():
            xt = torch.tensor(X, dtype=torch.float32, device=self.device)
            preds = self.net(xt).cpu().numpy()  # shape (N, n_objs)

        # Exemplo: definimos std fixo 0.05 => "confiança" alta
        std = np.ones_like(preds) * 0.05

        if single:
            return preds[0], std[0]
        else:
            return preds, std

    def _predict_coreml(self, X):
        """
        Se convert_coreml==True e já convertemos a rede,
        faz inferência via coreml_model.predict(...).
        """
        X = np.array(X, dtype=float)
        single = (X.ndim == 1)
        if single:
            X = X[np.newaxis, :]

        results = []
        for row in X:
            inp = {"features": row[np.newaxis,:]}
            outdict = self.coreml_model.predict(inp)
            # assumimos output => "output" shape(n_objs,)
            val = outdict.get("output", [0]*self.n_objs)
            results.append(np.array(val, dtype=float))

        arr = np.array(results)
        # Exemplo: definimos std fixo 0.01 aqui
        std = np.ones_like(arr)*0.01

        if single:
            return arr[0], std[0]
        else:
            return arr, std

    def check_fallback(self, predicted_vals, real_vals):
        """
        Se erro > fallback_error_thresh => fallback
        """
        err = np.linalg.norm(predicted_vals - real_vals)
        return (err > self.fallback_error_thresh)

    def is_confident(self, std_array):
        """
        Se media do std <= confidence_threshold => conf=True
        """
        threshold = self.config["surrogate"].get("confidence_threshold", 0.2)
        if std_array.ndim == 1:
            return (np.mean(std_array) <= threshold)
        else:
            mean_std = np.mean(std_array, axis=1)
            return (mean_std <= threshold)