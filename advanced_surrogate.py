import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import coremltools as ct
import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from neural_model import MultiOutputSurrogateNet, train_multioutput_surrogate

class AdvancedEnsembleSurrogate:
    """
    Ensemble de M redes + (opcional) GP multioutput.
    Evita reconverter a cada re-treino se self.already_converted==True.
    """

    def __init__(self, input_dim, n_objs, config, device):
        self.config = config
        self.device = device
        self.n_objs = n_objs
        self.num_models = config["surrogate"].get("ensemble_size", 3)
        self.models = []
        # Reduzimos fallback default no config.yaml para 5.0
        self.fallback_error_thresh = config["surrogate"].get("fallback_error_thresh", 50.0)
        self.dynamic_fallback = config["surrogate"].get("dynamic_fallback", False)
        self.confidence_threshold = config["surrogate"].get("confidence_threshold", 0.95)
        self.convert_coreml = config["surrogate"].get("convert_coreml", False)

        # Flag para não converter toda hora
        self.already_converted = False

        # Se iremos usar GP
        self.use_gp = config["surrogate"].get("use_gp", False)
        self.gp = None
        if self.use_gp:
            # Em multi-obj, criamos uma lista de GPs (um por objetivo)
            self.gp_list = []
            for _ in range(n_objs):
                kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-6)
                gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=2)
                self.gp_list.append(gp)
        else:
            self.gp_list = []

        self.coreml_models = []

        self.input_dim = input_dim
        self.X_train = None
        self.y_train = None

    def pretrain_from_csv(self, csv_path):
        df = pd.read_csv(csv_path)
        X_cols = [f"X{i}" for i in range(self.input_dim)]
        Y_cols = [f"Y{i}" for i in range(self.n_objs)]
        missing_x = set(X_cols) - set(df.columns)
        missing_y = set(Y_cols) - set(df.columns)
        if missing_x or missing_y:
            raise KeyError(f"CSV missing columns: {missing_x} {missing_y}")

        X_data = df[X_cols].values
        Y_data = df[Y_cols].values

        if self.X_train is None:
            self.X_train = X_data
            self.y_train = Y_data
        else:
            self.X_train = np.vstack([self.X_train, X_data])
            self.y_train = np.vstack([self.y_train, Y_data])

        print(f"[DEBUG] pretrain_from_csv: loaded {len(self.X_train)} samples.")

    def train_ensemble(self):
        """
        Treina M redes e (se use_gp==True) GPs.
        Evita converter p/ CoreML repetidamente (somente 1x se not self.already_converted).
        """
        if self.X_train is None or len(self.X_train) < 2:
            print("[DEBUG] Not enough data to train ensemble.")
            return

        self.models = []
        for _ in range(self.num_models):
            net = train_multioutput_surrogate(
                self.X_train, 
                self.y_train, 
                self.config, 
                self.device, 
                self.n_objs
            )
            self.models.append(net)

        print(f"[DEBUG] train_ensemble => trained {len(self.models)} models (MLP).")

        # Treina GP se use_gp == True
        if self.use_gp and len(self.gp_list) == self.n_objs:
            for j, gp in enumerate(self.gp_list):
                y_j = self.y_train[:, j]
                gp.fit(self.X_train, y_j)
            print(f"[DEBUG] train_ensemble => trained {len(self.gp_list)} GaussianProcess(es).")

        # Converte para CoreML só se for pedido e ainda não tiver sido feito
        if self.convert_coreml and not self.already_converted:
            self.coreml_models = []
            for i, net in enumerate(self.models):
                mlm = self._convert_to_coreml(net, f"EnsembleNet_{i}")
                self.coreml_models.append(mlm)
            self.already_converted = True
            print("[DEBUG] Core ML conversion done ONCE for the ensemble.")
        else:
            print("[DEBUG] Skipped Core ML conversion to avoid overhead (already converted or convert_coreml=False).")

    def _convert_to_coreml(self, net, name):
        net.eval()
        example_in = torch.rand(1, self.input_dim, dtype=torch.float32, device=self.device)
        traced = torch.jit.trace(net, example_in)
        mlmodel = ct.convert(
            traced,
            inputs=[ct.TensorType(shape=example_in.shape, dtype=float, name="features")],
            convert_to="mlprogram",
            compute_units=ct.ComputeUnit.ALL
        )
        mlmodel.short_description = "MultiOutput Surrogate MLP"
        return mlmodel

    def update_data(self, X, Y):
        if self.X_train is None:
            self.X_train = X
            self.y_train = Y
        else:
            self.X_train = np.vstack([self.X_train, X])
            self.y_train = np.vstack([self.y_train, Y])

    def retrain(self):
        """
        Re-treina o ensemble e GPs (se use_gp).
        Ajusta fallback se dynamic_fallback==True.
        """
        if self.X_train is None or len(self.X_train) < 2:
            return

        print(f"[DEBUG] AdvancedEnsembleSurrogate: retraining on {len(self.X_train)} samples.")
        self.train_ensemble()

        if self.dynamic_fallback:
            dd = np.std(self.y_train, axis=0).mean()
            self.fallback_error_thresh = dd * 2.0
            print(f"[DEBUG] dynamic_fallback => fallback_error_thresh = {self.fallback_error_thresh:.4f}")

    def predict(self, X):
        """
        Retorna (mean, std) => shape(N, n_objs).
        Se MLP ensemble + GP => comitê (média do ensemble e do GP).
        """
        if not self.models:
            # Se não tem modelos treinados, retorna zero + std alto
            n = X.shape[0] if X.ndim > 1 else 1
            return np.zeros((n, self.n_objs)), np.ones((n, self.n_objs)) * 9999

        # 1) Inferência MLP ensemble
        if (self.convert_coreml and self.already_converted and
                len(self.coreml_models) == len(self.models)):
            mean_mlp, std_mlp = self._predict_coreml_ensemble(X)
        else:
            mean_mlp, std_mlp = self._predict_pytorch_ensemble(X)

        # 2) Se use_gp => também faz predição do GP
        if self.use_gp and len(self.gp_list) == self.n_objs:
            X_ = np.array(X, dtype=float)
            single = (X_.ndim == 1)
            if single:
                X_ = X_[np.newaxis, :]

            gp_means_arr = []
            gp_stds_arr = []
            for j, gp in enumerate(self.gp_list):
                m_j, std_j = gp.predict(X_, return_std=True)
                gp_means_arr.append(m_j)
                gp_stds_arr.append(std_j)

            gp_means_arr = np.column_stack(gp_means_arr)
            gp_stds_arr = np.column_stack(gp_stds_arr)

            # Combinar MLP e GP => média
            final_mean = 0.5 * (mean_mlp + gp_means_arr)
            final_std = np.maximum(std_mlp, gp_stds_arr)
            return final_mean, final_std
        else:
            return mean_mlp, std_mlp

    def _predict_pytorch_ensemble(self, X):
        X = np.array(X, dtype=float)
        single = False
        if X.ndim == 1:
            X = X[np.newaxis, :]
            single = True

        preds_list = []
        with torch.no_grad():
            for net in self.models:
                xt = torch.tensor(X, dtype=torch.float32, device=self.device)
                p = net(xt).cpu().numpy()  # shape(N, n_objs)
                preds_list.append(p)

        all_preds = np.stack(preds_list, axis=0)  # (num_models, N, n_objs)
        mean_preds = np.mean(all_preds, axis=0)
        ensemble_std = np.std(all_preds, axis=0)

        if single:
            return mean_preds[0], ensemble_std[0]
        else:
            return mean_preds, ensemble_std

    def _predict_coreml_ensemble(self, X):
        X = np.array(X, dtype=float)
        single = (X.ndim == 1)
        if single:
            X = X[np.newaxis, :]

        ensemble_preds = []
        for mlm in self.coreml_models:
            local_preds = []
            for row in X:
                inp = {"features": row[np.newaxis, :]}
                outdict = mlm.predict(inp)
                val = outdict.get("output", [0] * self.n_objs)
                local_preds.append(np.array(val, dtype=float))
            ensemble_preds.append(np.array(local_preds))

        all_preds = np.stack(ensemble_preds, axis=0)  # (num_models, N, n_objs)
        mean_preds = np.mean(all_preds, axis=0)
        ensemble_std = np.std(all_preds, axis=0)
        return mean_preds, ensemble_std

    def check_fallback(self, predicted_vals, real_vals):
        err = np.linalg.norm(predicted_vals - real_vals)
        return (err > self.fallback_error_thresh)

    def is_confident(self, std_array):
        thr = self.confidence_threshold
        if std_array.ndim == 1:
            return (np.mean(std_array) <= thr)
        else:
            mean_std = np.mean(std_array, axis=1)
            return (mean_std <= thr)