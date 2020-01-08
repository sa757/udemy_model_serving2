from img_modeling_pipeline.processing import data_management as dm
import numpy as np


def test_make_single_prediction():
    # Given
    image_name = '1.png'
    image_directory = '../img_modeling_pipeline/datasets/test_data/Black-grass'

    image_df = dm.load_single_image(
        data_folder=image_directory,
        filename=image_name)

    prepared_df = image_df['image'].reset_index(drop=True)

    KERAS_PIPELINE = dm.load_pipeline_keras()
    ENCODER = dm.load_encoder()

    # When
    predictions = KERAS_PIPELINE.predict(prepared_df)
    readable_predictions = ENCODER.encoder.inverse_transform(predictions)

    # Then
    assert readable_predictions is not None
    assert isinstance(predictions[0], np.int32)
    assert isinstance(readable_predictions[0], str)



if __name__ == "__main__":
    test_make_single_prediction()