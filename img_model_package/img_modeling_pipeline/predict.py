import logging

import pandas as pd

from img_modeling_pipeline import __version__ as _version
from img_modeling_pipeline.processing import data_management as dm

_logger = logging.getLogger(__name__)
KERAS_PIPELINE = dm.load_pipeline_keras()
ENCODER = dm.load_encoder()


def make_single_prediction(*, image_name: str, image_directory: str):

    image_df = dm.load_single_image(
        data_folder=image_directory,
        filename=image_name)

    prepared_df = image_df['image'].reset_index(drop=True)

    _logger.info(f'received input array: {prepared_df}, '
                 f'filename: {image_name}')

    predictions = KERAS_PIPELINE.predict(prepared_df)
    readable_predictions = ENCODER.encoder.inverse_transform(predictions)

    _logger.info(f'Made prediction: {predictions}'
                 f' with model version: {_version}')

    return dict(predictions=predictions,
                readable_predictions=readable_predictions,
                version=_version)


def make_bulk_prediction(*, images_df: pd.Series) -> dict:

    _logger.info(f'received input df: {images_df}')

    predictions = KERAS_PIPELINE.predict(images_df)
    readable_predictions = ENCODER.encoder.inverse_transform(predictions)

    _logger.info(f'Made predictions: {predictions}'
                 f' with model version: {_version}')

    return dict(predictions=predictions,
                readable_predictions=readable_predictions,
                version=_version)


# if __name__ == "__main__":
#     result = make_single_prediction(image_name='1.png', image_directory='./datasets/test_data/Black-grass')
#     print(result)
