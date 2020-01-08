from sklearn.pipeline import Pipeline

from img_modeling_pipeline.config import config
from img_modeling_pipeline.processing import preprocessors as pp
from img_modeling_pipeline import model

model_pipeline = Pipeline([
    ('datasets', pp.CreateDataset(config.IMAGE_SIZE)),
    ('model', model.cnn_clf)
])