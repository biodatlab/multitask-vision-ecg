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

from transformers import get_linear_schedule_with_warmup
from .models import SingleTaskModel, MultiTaskModel, SingleTaskClinicalCNNModel, MultiTaskClinicalCNNModel

NUM_ECG_LEADS = 12

class LeadEmbeddingMixin:
    def get_lead_embeddings(self, leads_batch):
        # There are 12 leads in each ECG.
        lead_embedding_list = []
        for i in range(NUM_ECG_LEADS):
            lead_batch = leads_batch[:, i, :].unsqueeze(1).unsqueeze(1)
            lead_embedding = self.model(lead_batch)
            lead_embedding_list.append(lead_embedding)

        # Average the lead embeddings.
        lead_embeddings = torch.mean(torch.stack(lead_embedding_list), dim=0)
        return lead_embeddings


class ClinicalEmbeddingMixin:
    def get_feature_embeddings(self, numerical_features, categorical_features):
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
        return summarized_feature_embeddings


class SingleTaskModel1D(SingleTaskModel, LeadEmbeddingMixin):
    def __init__(self, num_leads: int = 12, **kwargs):
        super().__init__(**kwargs)
        self.num_leads = num_leads
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(1, 3), stride=1, padding=3)
        self.save_hyperparameters()

    def forward(self, leads_batch):
        lead_embeddings = self.get_lead_embeddings(leads_batch)
        lead_embeddings = F.relu(lead_embeddings)
        out = self.head(lead_embeddings)
        return out


class MultiTaskModel1D(MultiTaskModel, LeadEmbeddingMixin):
    def __init__(self, num_leads: int = 12, **kwargs):
        super().__init__(**kwargs)
        self.num_leads = num_leads
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(1, 3), stride=1, padding=3)
        self.save_hyperparameters()

    def forward(self, leads_batch):
        lead_embeddings = self.get_lead_embeddings(leads_batch)
        lead_embeddings = F.relu(lead_embeddings)
        scar = self.scar_head(lead_embeddings)
        lvef = self.lvef_head(lead_embeddings)
        return torch.cat([scar, lvef], dim=1)


class SingleTaskClinicalModel1D(SingleTaskClinicalCNNModel, LeadEmbeddingMixin, ClinicalEmbeddingMixin):
    def __init__(self, num_leads: int = 12, **kwargs):
        super().__init__(**kwargs)
        self.num_leads = num_leads
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(1, 3), stride=1, padding=3)

        self.model.fc = nn.Linear(in_features=num_in_features, out_features=latent_dim, bias=False)
        self.head = nn.Linear(
            in_features=self.rnn_output_size * num_rnn_layers + latent_dim,
            out_features=num_classes,
            bias=bias_head,
        )
        self.save_hyperparameters()

    def forward(self, x):
        """
        x is a tuple with 3 elements:
        1. x[0]: ECG lead arrays.
        2. x[1]: numerical clinical features.
        3. x[2]: categorical clinical features.
        """
        leads_batch, numerical_features, categorical_features = x

        lead_embeddings = self.get_lead_embeddings(leads_batch)
        summarized_feature_embeddings = self.get_feature_embeddings(numerical_features, categorical_features)

        # Concatenate lead embeddings and summarized clinical features embeddings.([lead_embeddings, num_feat, cat1_emb, cat2_emb, ..., catn_emb])
        embedding_list = [lead_embeddings, summarized_feature_embeddings]
        embeddings = torch.cat(embedding_list, dim=1)
        embeddings = F.relu(embeddings)
        out = self.head(embeddings)
        return out


class MultiTaskClinicalModel1D(SingleTaskClinicalCNNModel, LeadEmbeddingMixin, ClinicalEmbeddingMixin):
    def __init__(self, num_leads: int = 12, **kwargs):
        super().__init__(**kwargs)
        self.num_leads = num_leads
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(1, 3), stride=1, padding=3)

        self.model.fc = nn.Linear(in_features=num_in_features, out_features=latent_dim, bias=False)
        self.head = nn.Linear(
            in_features=self.rnn_output_size * num_rnn_layers + latent_dim,
            out_features=num_classes,
            bias=bias_head,
        )
        self.save_hyperparameters()

    def forward(self, x):
        """
        x is a tuple with 3 elements:
        1. x[0]: ECG lead arrays.
        2. x[1]: numerical clinical features.
        3. x[2]: categorical clinical features.
        """
        leads_batch, numerical_features, categorical_features = x

        lead_embeddings = self.get_lead_embeddings(leads_batch)
        summarized_feature_embeddings = self.get_feature_embeddings(numerical_features, categorical_features)

        # Concatenate lead embeddings and summarized clinical features embeddings.([lead_embeddings, num_feat, cat1_emb, cat2_emb, ..., catn_emb])
        embedding_list = [lead_embeddings, summarized_feature_embeddings]
        embeddings = torch.cat(embedding_list, dim=1)
        embeddings = F.relu(embeddings)
        out = self.head(embeddings)
        return out
