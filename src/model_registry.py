#!/usr/bin/env python3
# Author: Joel Ye

from src.model import (
    NeuralDataTransformer,
)

from src.model_baselines import (
    RatesOracle,
    RandomModel,
)

LEARNING_MODELS = {
    "NeuralDataTransformer": NeuralDataTransformer,
}

NONLEARNING_MODELS = {
    "Oracle": RatesOracle,
    "Random": RandomModel
}

INPUT_MASKED_MODELS = {
    "NeuralDataTransformer": NeuralDataTransformer,
}

MODELS = {**LEARNING_MODELS, **NONLEARNING_MODELS, **INPUT_MASKED_MODELS}

def is_learning_model(model_name):
    return model_name in LEARNING_MODELS

def is_input_masked_model(model_name):
    return model_name in INPUT_MASKED_MODELS

def get_model_class(model_name):
    return MODELS[model_name]