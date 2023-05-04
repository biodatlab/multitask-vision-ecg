import os.path as op
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy
from typing import Union

import torchvision.models as models
from pytorch_lightning import LightningModule
import timm


class SaveLoadMixin:
    """
    Mixin class for saving and loading models
    """

    def save_configs(self, save_path: str):
        json.dump(
            self.configs,
            open(op.join(save_path, "configs.json"), "w", encoding="utf-8"),
        )

    @classmethod
    def from_configs(
        cls,
        model_path: str,
        train: bool = False,
        device: str = "cpu",
    ):
        """
        Loads model from configs.json and model.ckpt.

        Parameters
        ==========
        model_path: str
            Path to the folder containing configs.json and model.ckpt
        train: bool
            Whether to load model in training mode
        device: str
            Device to load model on

        Returns
        =======
        model: LightningModule
        """
        if isinstance(model_path, str):
            configs = json.load(open(op.join(model_path, "configs.json"), "r", encoding="utf-8"))
        else:
            raise ValueError("configs must be a string path to a folder containing configs.json and model.ckpt")

        configs["device"] = device

        # Load model.
        model = cls.load_from_checkpoint(op.join(model_path, "model.ckpt"), map_location=device, **configs)
        # Load model to device.
        model.to(device)

        # # Try to convert model to half precision when using GPU.
        # if "cuda" in device:
        #     try:
        #         model.half()
        #     except:
        #         print("Model cannot be converted to half precision. Continuing with full precision.")

        # Set model to eval mode if not training.
        if not train:
            model.eval()
        return model


class SingleTaskModel(LightningModule, SaveLoadMixin):
    """
    Single-task learning model with Lightning module
    """

    def __init__(
        self,
        in_channels: int = 1,
        learning_rate: float = 1e-3,
        use_timm: bool = False,
        pretrained: bool = False,
        backbone: str = "resnet18",
        latent_dim: int = 512,
        num_classes: int = 2,
        bias_head: bool = False,
        load_state_dict: str = None,
        device: str = "cpu",
        **kwargs,
    ):
        """
        Parameters
        ==========
        in_channels: int
            Number of input channels. Default is 1.
        learning_rate: float
            Learning rate for optimizer. Default is 1e-3.
        use_timm: bool
            Use timm pretrained model. Default is True.
        pretrained: bool
            Use pretrained model. Default is True.
        backbone: str
            Backbone model name. Default is "resnet18".
        latent_dim: int
            Latent dimension for the classification head. Default is 512.
        num_classes: int
            Number of classes for classification head. Default is 2.
        bias_head: bool
            Use bias for classification head. Default is False.
        load_state_dict: str
            Path to load state dict. Default is None.
        device: str
            Device to use. Default is "cpu".
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.accuracy = Accuracy()
        self.loss_fn = nn.CrossEntropyLoss()

        self.configs = {
            "in_channels": in_channels,
            "learning_rate": learning_rate,
            "use_timm": use_timm,
            "pretrained": pretrained,
            "backbone": backbone,
            "latent_dim": latent_dim,
            "num_classes": num_classes,
            "bias_head": bias_head,
            "load_state_dict": load_state_dict,
        }

        if use_timm:
            self.model = timm.create_model(
                backbone,
                pretrained=pretrained,
                in_chans=in_channels,
                num_classes=latent_dim,
            )
            num_in_features = self.model.get_classifier().in_features

        else:
            if backbone == "resnet18":
                self.model = models.resnet18(pretrained=pretrained)
                self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
                num_in_features = self.model.fc.in_features
            else:
                raise ValueError(
                    f"Backbone {backbone} is not defined in the torchvision.models backends. Consider setting use_timm=True."
                )

        self.model.fc = nn.Linear(in_features=num_in_features, out_features=latent_dim, bias=False)
        self.head = nn.Linear(in_features=latent_dim, out_features=num_classes, bias=bias_head)
        # for wandb logging
        self.save_hyperparameters()

    def forward(self, x):
        embeddings = self.model(x)
        embeddings = F.relu(embeddings)
        out = self.head(embeddings)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        # wandb.log({"loss":loss})
        return loss

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.2, patience=3, min_lr=1e-6, verbose=True
        )
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": self.scheduler,
            "monitor": "val_loss",
        }

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)


class SingleTaskClinicalCNNModel(SingleTaskModel, SaveLoadMixin):
    """
    Singletask learning model for ECG image & Clinical Features with Lightning module
    """

    def __init__(
        self,
        in_channels: int = 1,
        learning_rate: float = 1e-3,
        use_timm: bool = True,
        pretrained: bool = True,
        backbone: str = "resnet18",
        latent_dim: int = 512,
        num_classes: int = 2,
        bias_head: bool = False,
        num_categorical_features: int = 5,
        num_numerical_features: int = 1,
        embedding_size: int = 5,
        rnn_output_size: int = 10,
        rnn_type: str = "rnn",
        num_rnn_layers: int = 1,
        pretrained_backbone_path: str = None,
        load_state_dict: str = None,
        device: str = "cpu",
        **kwargs,
    ):
        """
        Parameters
        ==========
        in_channels: int
            Number of input channels. Default is 1.
        learning_rate: float
            Learning rate for optimizer. Default is 1e-3.
        use_timm: bool
            Use timm pretrained model. Default is True.
        pretrained: bool
            Use pretrained model. Default is True.
        backbone: str
            Backbone model name. Default is "resnet18".
        latent_dim: int
            Latent dimension for the classification head. Default is 512.
        num_classes: int
            Number of classes for the classification head. Default is 2.
        bias_head: bool
            Use bias for classification head. Default is False.
        num_categorical_features: int
            Number of categorical features. Default is 5.
        num_numerical_features: int
            Number of numerical features. Default is 1.
        embedding_size: int
            Embedding size for categorical features. Default is 5.
        rnn_output_size: int
            RNN output size. Default is 10.
        rnn_type: str
            RNN type. Default is "rnn".
        num_rnn_layers: int
            Number of RNN layers. Default is 1.
        pretrained_backbone_path: str
            Path to pretrained backbone model. Default is None.
        load_state_dict: str
            Path to load state dict. Default is None.
        device: str
            Device to use. Default is "cpu".
        """
        super(SingleTaskClinicalCNNModel, self).__init__(
            in_channels=in_channels,
            learning_rate=learning_rate,
            use_timm=use_timm,
            pretrained=pretrained,
            backbone=backbone,
            latent_dim=latent_dim,
            num_classes=num_classes,
            bias_head=bias_head,
            load_state_dict=load_state_dict,
            device=device,
            **kwargs,
        )

        self._device = device
        self.loss_fn = nn.CrossEntropyLoss()

        ## add num_categorical_features, num_numerical_features, use_embedding, and embedding_size to the config
        self.configs.update(
            {
                "num_categorical_features": num_categorical_features,
                "num_numerical_features": num_numerical_features,
                "embedding_size": embedding_size,
                "rnn_output_size": rnn_output_size,
                "rnn_type": rnn_type,
            }
        )

        self.rnn_type = rnn_type
        self.embedding_size = embedding_size
        rnn_input_size = embedding_size + num_numerical_features
        self.rnn_output_size = rnn_output_size
        self.num_rnn_layers = num_rnn_layers

        ### load model; if pretrained_backbone_path is provided
        if pretrained_backbone_path:
            # get base model from loaded SingleTaskModel() class
            self.model = SingleTaskModel.from_configs(pretrained_backbone_path).model

        self.embedding_layer = nn.Embedding(num_embeddings=2, embedding_dim=self.embedding_size)

        if self.rnn_type == "rnn":
            self.clinical_rnn_layer = nn.RNN(
                input_size=rnn_input_size, hidden_size=self.rnn_output_size, num_layers=num_rnn_layers
            )
        elif self.rnn_type == "lstm":
            self.clinical_rnn_layer = nn.LSTM(
                input_size=rnn_input_size, hidden_size=self.rnn_output_size, num_layers=num_rnn_layers
            )

        elif self.rnn_type == "birnn":
            self.clinical_rnn_layer = nn.RNN(
                input_size=rnn_input_size,
                hidden_size=self.rnn_output_size,
                num_layers=num_rnn_layers,
                bidirectional=True,
            )

        elif self.rnn_type == "bilstm":
            self.clinical_rnn_layer = nn.LSTM(
                input_size=rnn_input_size,
                hidden_size=self.rnn_output_size,
                num_layers=num_rnn_layers,
                bidirectional=True,
            )

        self.classification_head = nn.Linear(
            in_features=self.rnn_output_size * num_rnn_layers + latent_dim,
            out_features=num_scar_class,
            bias=bias_head,
        )

        self.save_hyperparameters()

    def get_rnn_hidden_state(self, size):
        return torch.zeros(size=size, device=self._device)

    def forward(self, x):
        """
        x is a tuple with 3 elements:
        1. x[0]: image array.
        2. x[1]: numerical clinical features.
        3. x[2]: categorical clinical features.
        """
        image_array, numerical_features, categorical_features = x

        # Get image embeddings.
        image_embeddings = self.model(image_array)
        # Reshape numerical features.
        numerical_features = numerical_features.reshape(-1, 1)
        feature_embedding_list = []

        categorical_features = torch.swapaxes(
            categorical_features.view(categorical_features.size(0), categorical_features.size(-1)), 1, 0
        )
        for categorical_feature in categorical_features:
            # Pass categorical feature through embedding layer.
            categorical_embeddings = self.embedding_layer(categorical_feature)
            # Concatenate raw numerical feature to the categorical embeddings.
            feature_embeddings = torch.cat([categorical_embeddings, numerical_features], dim=1)
            feature_embeddings = torch.unsqueeze(feature_embeddings, dim=0)
            feature_embedding_list.append(feature_embeddings)

        preprocessed_feature_embeddings = torch.cat(feature_embedding_list, dim=0)
        if self.rnn_type == "rnn":
            rnn_hidden_state_matrix = self.get_rnn_hidden_state(
                (self.num_rnn_layers, preprocessed_feature_embeddings.size(1), self.rnn_output_size)
            )
            # Pass the preprocessed feature embeddings through the RNN layer
            _, summarized_feature_embeddings = self.clinical_rnn_layer(
                preprocessed_feature_embeddings, rnn_hidden_state_matrix
            )
        elif self.rnn_type == "lstm":
            rnn_hidden_state_matrix = self.get_rnn_hidden_state(
                (self.num_rnn_layers, preprocessed_feature_embeddings.size(1), self.rnn_output_size)
            )
            rnn_cell_state_matrix = self.get_rnn_hidden_state(
                (self.num_rnn_layers, preprocessed_feature_embeddings.size(1), self.rnn_output_size)
            )
            # Pass the preprocessed feature embeddings through the LSTM layer
            _, (summarized_feature_embeddings, _) = self.clinical_rnn_layer(
                preprocessed_feature_embeddings, (rnn_hidden_state_matrix, rnn_cell_state_matrix)
            )

        elif self.rnn_type == "birnn":
            rnn_hidden_state_matrix = self.get_rnn_hidden_state(
                (self.num_rnn_layers * 2, preprocessed_feature_embeddings.size(1), self.rnn_output_size)
            )
            # Pass the preprocessed feature embeddings through the RNN layer
            _, summarized_feature_embeddings = self.clinical_rnn_layer(
                preprocessed_feature_embeddings, rnn_hidden_state_matrix
            )

            summarized_feature_embeddings = torch.sum(summarized_feature_embeddings, dim=0, keepdim=True)

        elif self.rnn_type == "bilstm":
            rnn_hidden_state_matrix = self.get_rnn_hidden_state(
                (self.num_rnn_layers * 2, preprocessed_feature_embeddings.size(1), self.rnn_output_size)
            )

            rnn_cell_state_matrix = self.get_rnn_hidden_state(
                (self.num_rnn_layers * 2, preprocessed_feature_embeddings.size(1), self.rnn_output_size)
            )

            # Pass the preprocessed feature embeddings through the LSTM layer
            _, (summarized_feature_embeddings, _) = self.clinical_rnn_layer(
                preprocessed_feature_embeddings, (rnn_hidden_state_matrix, rnn_cell_state_matrix)
            )
            summarized_feature_embeddings = torch.sum(summarized_feature_embeddings, dim=0, keepdim=True)

        # Reshape the summarized feature embeddings.
        summarized_feature_embeddings = summarized_feature_embeddings.view(
            summarized_feature_embeddings.size(1), summarized_feature_embeddings.size(-1) * self.num_rnn_layers
        )
        # Concatenate image embeddings and summarized clinical features embeddings.([img_emb, num_feat, cat1_emb, cat2_emb, ..., catn_emb])
        embedding_list = [image_embeddings, summarized_feature_embeddings]
        embeddings = torch.cat(embedding_list, dim=1)
        out = self.classification_head(embeddings)
        return out


class MultiTaskModel(LightningModule, SaveLoadMixin):
    """
    Multitask learning model for ECG image with Lightning module
    """

    def __init__(
        self,
        in_channels: int = 1,
        learning_rate: float = 1e-3,
        use_timm: bool = False,
        pretrained: bool = False,
        backbone: str = "resnet18",
        latent_dim: int = 512,
        scar_class: int = 2,
        lvef_class: int = 2,
        loss_weights: dict = {"scar": [1, 1], "lvef": [1, 1]},
        scar_lvef_loss_ratio: list = [0.7, 0.3],
        bias_head: bool = False,
        load_state_dict: str = None,
        device: str = "cpu",
        **kwargs,
    ):
        """
        Parameters
        ==========
        in_channels: int
            Number of input channels. Default is 1.
        learning_rate: float
            Learning rate for optimizer. Default is 1e-3.
        use_timm: bool
            Use timm pretrained model. Default is True.
        pretrained: bool
            Use pretrained model. Default is True.
        backbone: str
            Backbone model name. Default is "resnet18".
        latent_dim: int
            Latent dimension for each classification head. Default is 512.
        num_scar_class: int
            Number of scar classes. Default is 2.
        num_lvef_class: int
            Number of lvef classes. Default is 2.
        loss_weights: dict
            Loss weights for each class. Default is {"scar": [1, 1], "lvef": [1, 1]}.
        scar_lvef_loss_ratio: list
            Loss ratio for scar and lvef. Default is [0.7, 0.3].
        bias_head: bool
            Use bias for classification head. Default is False.
        load_state_dict: str
            Path to load state dict. Default is None.
        device: str
            Device to use. Default is "cpu".
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.accuracy = Accuracy()
        self.scar_loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(loss_weights["scar"], dtype=torch.float).to(device))
        self.lvef_loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(loss_weights["lvef"], dtype=torch.float).to(device))
        self.scar_loss_w, self.lvef_loss_w = scar_lvef_loss_ratio

        self.configs = {
            "in_channels": in_channels,
            "learning_rate": learning_rate,
            "use_timm": use_timm,
            "pretrained": pretrained,
            "backbone": backbone,
            "latent_dim": latent_dim,
            "scar_class": scar_class,
            "lvef_class": lvef_class,
            "loss_weights": loss_weights,
            "scar_lvef_loss_ratio": scar_lvef_loss_ratio,
            "bias_head": bias_head,
            "load_state_dict": load_state_dict,
        }

        if use_timm:
            self.model = timm.create_model(
                backbone,
                pretrained=pretrained,
                in_chans=in_channels,
                num_classes=latent_dim,
            )
            num_in_features = self.model.get_classifier().in_features

        else:
            if backbone == "resnet18":
                self.model = models.resnet18(pretrained=pretrained)
                if load_state_dict:
                    print(f"Loading {backbone} weights from {load_state_dict}")
                    self.model.load_state_dict(torch.load(load_state_dict))
                # self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
                num_in_features = self.model.fc.in_features
            else:
                raise ValueError(
                    f"Backbone {backbone} is not defined in the torchvision.models backends. Consider setting use_timm=True."
                )

        self.model.fc = nn.Linear(in_features=num_in_features, out_features=latent_dim, bias=False)

        self.scar_head = nn.Linear(in_features=latent_dim, out_features=scar_class, bias=bias_head)
        self.lvef_head = nn.Linear(in_features=latent_dim, out_features=lvef_class, bias=bias_head)

        # for wandb logging
        self.save_hyperparameters()

    def forward(self, x):
        embeddings = self.model(x)
        embeddings = F.relu(embeddings)

        scar = self.scar_head(embeddings)
        lvef = self.lvef_head(embeddings)
        return {"scar": scar, "lvef": lvef}

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)

        loss_scar = self.scar_loss_fn(logits["scar"], y["scar"])
        loss_lvef = self.lvef_loss_fn(logits["lvef"], y["lvef"])

        preds_scar = torch.argmax(logits["scar"], dim=1)
        preds_lvef = torch.argmax(logits["lvef"], dim=1)

        total_loss = (self.scar_loss_w * loss_scar) + (self.lvef_loss_w * loss_lvef)
        mean_acc = self.scar_loss_w * self.accuracy(preds_scar, y["scar"]) + self.lvef_loss_w * self.accuracy(
            preds_lvef, y["lvef"]
        )

        self.log("train_loss", total_loss, prog_bar=True)
        self.log("train_acc", mean_acc, prog_bar=True)
        return total_loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)

        loss_scar = self.scar_loss_fn(logits["scar"], y["scar"])
        loss_lvef = self.lvef_loss_fn(logits["lvef"], y["lvef"])

        preds_scar = torch.argmax(logits["scar"], dim=1)
        preds_lvef = torch.argmax(logits["lvef"], dim=1)

        total_loss = (self.scar_loss_w * loss_scar) + (self.lvef_loss_w * loss_lvef)
        mean_acc = self.scar_loss_w * self.accuracy(preds_scar, y["scar"]) + self.lvef_loss_w * self.accuracy(
            preds_lvef, y["lvef"]
        )

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", total_loss, prog_bar=True)
        self.log("val_acc", mean_acc, prog_bar=True)
        self.log("val_scar_acc", self.accuracy(preds_scar, y["scar"]), prog_bar=True)
        self.log("val_lvef_acc", self.accuracy(preds_lvef, y["lvef"]), prog_bar=True)
        return total_loss

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.2, patience=3, min_lr=1e-6, verbose=True
        )
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": self.scheduler,
            "monitor": "val_loss",
        }

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)


class MultiTaskClinicalCNNModel(MultiTaskModel, SaveLoadMixin):
    """
    Multitask learning model for ECG image & Clinical Features with Lightning module
    """

    def __init__(
        self,
        in_channels: int = 1,
        learning_rate: float = 1e-3,
        use_timm: bool = True,
        pretrained: bool = True,
        backbone: str = "resnet18",
        latent_dim: int = 512,
        num_scar_class: int = 2,
        num_lvef_class: int = 2,
        loss_weights: dict = {"scar": [1, 1], "lvef": [1, 1]},
        scar_lvef_loss_ratio: list = [0.7, 0.3],
        bias_head: bool = False,
        num_categorical_features: int = 5,
        num_numerical_features: int = 1,
        embedding_size: int = 5,
        rnn_output_size: int = 10,
        rnn_type: str = "rnn",
        num_rnn_layers: int = 1,
        pretrained_backbone_path: str = None,
        load_state_dict: str = None,
        device: str = "cpu",
        **kwargs,
    ):
        """
        Parameters
        ==========
        in_channels: int
            Number of input channels. Default is 1.
        learning_rate: float
            Learning rate for optimizer. Default is 1e-3.
        use_timm: bool
            Use timm pretrained model. Default is True.
        pretrained: bool
            Use pretrained model. Default is True.
        backbone: str
            Backbone model name. Default is "resnet18".
        latent_dim: int
            Latent dimension for each classification head. Default is 512.
        num_scar_class: int
            Number of scar classes. Default is 2.
        num_lvef_class: int
            Number of lvef classes. Default is 2.
        loss_weights: dict
            Loss weights for each class. Default is {"scar": [1, 1], "lvef": [1, 1]}.
        scar_lvef_loss_ratio: list
            Loss ratio for scar and lvef. Default is [0.7, 0.3].
        bias_head: bool
            Use bias for classification head. Default is False.
        num_categorical_features: int
            Number of categorical features. Default is 5.
        num_numerical_features: int
            Number of numerical features. Default is 1.
        embedding_size: int
            Embedding size for categorical features. Default is 5.
        rnn_output_size: int
            RNN output size. Default is 10.
        rnn_type: str
            RNN type. Default is "rnn".
        num_rnn_layers: int
            Number of RNN layers. Default is 1.
        pretrained_backbone_path: str
            Path to pretrained backbone model. Default is None.
        load_state_dict: str
            Path to load state dict. Default is None.
        device: str
            Device to use. Default is "cpu".
        """
        super(MultiTaskClinicalCNNModel, self).__init__(
            in_channels=in_channels,
            learning_rate=learning_rate,
            use_timm=use_timm,
            pretrained=pretrained,
            backbone=backbone,
            latent_dim=latent_dim,
            num_scar_class=num_scar_class,
            num_lvef_class=num_lvef_class,
            loss_weights=loss_weights,
            scar_lvef_loss_ratio=scar_lvef_loss_ratio,
            bias_head=bias_head,
            load_state_dict=load_state_dict,
            device=device,
            **kwargs,
        )

        self._device = device
        self.scar_loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(loss_weights["scar"], dtype=torch.float).to(device))
        self.lvef_loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(loss_weights["lvef"], dtype=torch.float).to(device))
        ## add num_categorical_features, num_numerical_features, use_embedding, and embedding_size to the config
        self.configs.update(
            {
                "num_categorical_features": num_categorical_features,
                "num_numerical_features": num_numerical_features,
                "embedding_size": embedding_size,
                "rnn_output_size": rnn_output_size,
                "rnn_type": rnn_type,
            }
        )

        self.rnn_type = rnn_type
        self.embedding_size = embedding_size
        rnn_input_size = embedding_size + num_numerical_features
        self.rnn_output_size = rnn_output_size
        self.num_rnn_layers = num_rnn_layers

        ### load model; if pretrained_backbone_path is provided
        if pretrained_backbone_path:
            # get base model from loaded MultiTaskModel() class
            self.model = MultiTaskModel.from_configs(pretrained_backbone_path).model

        self.embedding_layer = nn.Embedding(num_embeddings=2, embedding_dim=self.embedding_size)

        if self.rnn_type == "rnn":
            self.clinical_rnn_layer = nn.RNN(
                input_size=rnn_input_size, hidden_size=self.rnn_output_size, num_layers=num_rnn_layers
            )
        elif self.rnn_type == "lstm":
            self.clinical_rnn_layer = nn.LSTM(
                input_size=rnn_input_size, hidden_size=self.rnn_output_size, num_layers=num_rnn_layers
            )

        elif self.rnn_type == "birnn":
            self.clinical_rnn_layer = nn.RNN(
                input_size=rnn_input_size,
                hidden_size=self.rnn_output_size,
                num_layers=num_rnn_layers,
                bidirectional=True,
            )

        elif self.rnn_type == "bilstm":
            self.clinical_rnn_layer = nn.LSTM(
                input_size=rnn_input_size,
                hidden_size=self.rnn_output_size,
                num_layers=num_rnn_layers,
                bidirectional=True,
            )

        self.scar_head = nn.Linear(
            in_features=self.rnn_output_size * num_rnn_layers + latent_dim,
            out_features=num_scar_class,
            bias=bias_head,
        )
        self.lvef_head = nn.Linear(
            in_features=self.rnn_output_size * num_rnn_layers + latent_dim,
            out_features=num_lvef_class,
            bias=bias_head,
        )

        self.save_hyperparameters()

    def get_rnn_hidden_state(self, size):
        return torch.zeros(size=size, device=self._device)

    def forward(self, x):
        """
        x is a tuple with 3 elements:
        1. x[0]: image array.
        2. x[1]: numerical clinical features.
        3. x[2]: categorical clinical features.
        """
        image_array, numerical_features, categorical_features = x

        # Get image embeddings.
        image_embeddings = self.model(image_array)
        # Reshape numerical features.
        numerical_features = numerical_features.reshape(-1, 1)
        feature_embedding_list = []

        categorical_features = torch.swapaxes(
            categorical_features.view(categorical_features.size(0), categorical_features.size(-1)), 1, 0
        )
        for categorical_feature in categorical_features:
            # Pass categorical feature through embedding layer.
            categorical_embeddings = self.embedding_layer(categorical_feature)
            # Concatenate raw numerical feature to the categorical embeddings.
            feature_embeddings = torch.cat([categorical_embeddings, numerical_features], dim=1)
            feature_embeddings = torch.unsqueeze(feature_embeddings, dim=0)
            feature_embedding_list.append(feature_embeddings)

        preprocessed_feature_embeddings = torch.cat(feature_embedding_list, dim=0)
        if self.rnn_type == "rnn":
            rnn_hidden_state_matrix = self.get_rnn_hidden_state(
                (self.num_rnn_layers, preprocessed_feature_embeddings.size(1), self.rnn_output_size)
            )
            # Pass the preprocessed feature embeddings through the RNN layer
            _, summarized_feature_embeddings = self.clinical_rnn_layer(
                preprocessed_feature_embeddings, rnn_hidden_state_matrix
            )
        elif self.rnn_type == "lstm":
            rnn_hidden_state_matrix = self.get_rnn_hidden_state(
                (self.num_rnn_layers, preprocessed_feature_embeddings.size(1), self.rnn_output_size)
            )
            rnn_cell_state_matrix = self.get_rnn_hidden_state(
                (self.num_rnn_layers, preprocessed_feature_embeddings.size(1), self.rnn_output_size)
            )
            # Pass the preprocessed feature embeddings through the LSTM layer
            _, (summarized_feature_embeddings, _) = self.clinical_rnn_layer(
                preprocessed_feature_embeddings, (rnn_hidden_state_matrix, rnn_cell_state_matrix)
            )

        elif self.rnn_type == "birnn":
            rnn_hidden_state_matrix = self.get_rnn_hidden_state(
                (self.num_rnn_layers * 2, preprocessed_feature_embeddings.size(1), self.rnn_output_size)
            )
            # Pass the preprocessed feature embeddings through the RNN layer
            _, summarized_feature_embeddings = self.clinical_rnn_layer(
                preprocessed_feature_embeddings, rnn_hidden_state_matrix
            )

            summarized_feature_embeddings = torch.sum(summarized_feature_embeddings, dim=0, keepdim=True)

        elif self.rnn_type == "bilstm":
            rnn_hidden_state_matrix = self.get_rnn_hidden_state(
                (self.num_rnn_layers * 2, preprocessed_feature_embeddings.size(1), self.rnn_output_size)
            )

            rnn_cell_state_matrix = self.get_rnn_hidden_state(
                (self.num_rnn_layers * 2, preprocessed_feature_embeddings.size(1), self.rnn_output_size)
            )

            # Pass the preprocessed feature embeddings through the LSTM layer
            _, (summarized_feature_embeddings, _) = self.clinical_rnn_layer(
                preprocessed_feature_embeddings, (rnn_hidden_state_matrix, rnn_cell_state_matrix)
            )
            summarized_feature_embeddings = torch.sum(summarized_feature_embeddings, dim=0, keepdim=True)

        # Reshape the summarized feature embeddings.
        summarized_feature_embeddings = summarized_feature_embeddings.view(
            summarized_feature_embeddings.size(1), summarized_feature_embeddings.size(-1) * self.num_rnn_layers
        )
        # Concatenate image embeddings and summarized clinical features embeddings.([img_emb, num_feat, cat1_emb, cat2_emb, ..., catn_emb])
        embedding_list = [image_embeddings, summarized_feature_embeddings]
        embeddings = torch.cat(embedding_list, dim=1)
        scar = self.scar_head(embeddings)
        lvef = self.lvef_head(embeddings)
        return {"scar": scar, "lvef": lvef}
