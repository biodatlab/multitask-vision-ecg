from typing import Union, List, Optional, Dict
import numpy as np
import os.path as op
import torch
import torch.nn.functional as F
import PIL
from PIL import Image
import albumentations as A

from mtecg import SingleTaskModel, SingleTaskClinicalCNNModel, MultiTaskModel, MultiTaskClinicalCNNModel
import mtecg.constants as constants

MODEL_STRING_TO_CLASS_MAP = {
    "single-task": SingleTaskModel,
    "single-task-clinical": SingleTaskClinicalCNNModel,
    "multi-task": MultiTaskModel,
    "multi-task-clinical": MultiTaskClinicalCNNModel,
}


class ECGClassifier:
    def __init__(
        self,
        model_path: str,
        model_class: Union[str, object] = "multi-task",
        device: str = "cpu",
        task: Optional[str] = ["scar"],
        round_probabilities: bool = True,
    ):
        if isinstance(model_class, str):
            model_class = MODEL_STRING_TO_CLASS_MAP[model_class]

        self.model = model_class.from_configs(model_path, device=device, train=False)
        self.input_transforms = A.load(op.join(model_path, "transform.json"))
        self.device = device
        self.round_probabilities = round_probabilities

        if isinstance(self.model, SingleTaskModel):
            if not isinstance(task, list):
                task = [task]
            self.task = task
        elif isinstance(self.model, MultiTaskModel):
            self.task = ["scar", "lvef"]

    @torch.no_grad()
    def predict(
        self,
        image: Union[str, PIL.Image.Image],
        clinical_features: Optional[Dict[str, Union[int, float]]] = None,
    ) -> Dict[List[float], str]:

        if isinstance(image, str):
            image = Image.open(image)
        image = image.convert("RGB")
        image = self.input_transforms(image=np.array(image))["image"]
        image = image.unsqueeze(0)

        if isinstance(self.model, SingleTaskModel) and clinical_features is None:
            return self._predict_singleclass(image)
        elif isinstance(self.model, MultiTaskModel) and clinical_features is None:
            return self._predict_multiclass(image)
        elif isinstance(self.model, SingleTaskClinicalCNNModel):
            assert (
                clinical_features is not None
            ), f"Clinical features must be provided for model type: {self.model.__class__}."
            clinical_features = self._prepare_clinical_features(clinical_features)
            return self._predict_singleclass(image, clinical_features)
        elif isinstance(self.model, MultiTaskClinicalCNNModel):
            assert (
                clinical_features is not None
            ), f"Clinical features must be provided for model type: {self.model.__class__}."
            clinical_features = self._prepare_clinical_features(clinical_features)
            return self._predict_multiclass(image, clinical_features)

    @torch.no_grad()
    def _predict_singleclass(self, image: torch.Tensor, clinical_features: Dict[str, torch.Tensor] = None):
        if clinical_features and isinstance(self.model, SingleTaskClinicalCNNModel):
            model_input = (
                image.to(self.device),
                clinical_features["numerical_features"].to(self.device),
                clinical_features["categorical_features"].to(self.device),
            )
        else:
            model_input = image.to(self.device)

        logits = self.model(model_input)
        return {self.task[0]: self._get_output_dict(logits)}

    @torch.no_grad()
    def _predict_multiclass(self, image: torch.Tensor, clinical_features: Dict[str, torch.Tensor] = None):
        if clinical_features and isinstance(self.model, MultiTaskClinicalCNNModel):
            model_input = (
                image.to(self.device),
                clinical_features["numerical_features"].to(self.device),
                clinical_features["categorical_features"].to(self.device),
            )
        else:
            model_input = image.to(self.device)

        logits = self.model(model_input)
        return {key: self._get_output_dict(class_logits) for key, class_logits in logits.items()}

    def _get_output_dict(self, logits):
        probability_tensor = F.softmax(logits, dim=1)
        probabilities = probability_tensor.tolist()[0]
        if self.round_probabilities:
            probabilities = [round(p, 2) for p in probabilities]
        predicted_probability_dict = {
            "negative": probabilities[0],
            "positive": probabilities[1],
        }

        prediction = probability_tensor.argmax(dim=1).tolist()[0]
        return {
            "prediction": prediction,
            "probability": predicted_probability_dict,
        }

    def _prepare_clinical_features(
        self, clinical_feature_dict: Dict[str, Union[int, float]]
    ) -> Dict[str, torch.Tensor]:
        categorical_features = [
            clinical_feature_dict[feature] for feature in constants.categorical_feature_column_names
        ]
        numerical_features = [clinical_feature_dict[feature] for feature in constants.numerical_feature_column_names]

        categorical_features = np.array(categorical_features).astype(int)
        numerical_features = np.array(numerical_features).astype(float)

        feature_tensor_dict = {
            "numerical_features": torch.tensor(numerical_features, dtype=torch.float32).view(1, 1, -1),
            "categorical_features": torch.tensor(categorical_features, dtype=torch.long).view(1, 1, -1),
        }
        return feature_tensor_dict
