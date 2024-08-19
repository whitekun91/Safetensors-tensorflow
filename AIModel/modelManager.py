import os
from safetensors.numpy import save_file, load_file

"""

Packages of tensorflow model activation (training, evaluating)

"""


def train(x, y, epochs, model):
    """
    :param x: array
    :param y: array
    :param epochs: int
    :param model: tensorflow model object
    :return:
        - model fit
    """
    model.fit(x, y, epochs=epochs)


def model_weight_save(model, folder_path, file_name):
    """
    :param model: tensorflow model object
    :param folder_path: string
    :param file_name: string
    :return:
    """
    weights = model.get_weights()
    weights_dict = {f'layer_{i}': weights[i] for i in range(len(weights))}
    os.makedirs(f'./{folder_path}/', exist_ok=True)
    save_file(weights_dict, f'./{folder_path}/{file_name}')
    print('Safetensors file save complete!')


def model_weight_load_and_evaluate(folder_path, file_name, model):
    """
    :param folder_path: string
    :param file_name: string
    :param model: tensorflow model object
    :return:
    """
    loaded_weights_dict = load_file(f'./{folder_path}/{file_name}')
    print('Safetensors file load complete!')
    loaded_weights = [loaded_weights_dict[f'layer_{i}'] for i in range(len(model.get_weights()))]
    model.set_weights(loaded_weights)


def model_evaluate(x, y, model):
    """
    :param x: array
    :param y: array
    :param model: tensorflow model object
    :return:
    """
    loss, accuracy = model.evaluate(x, y)
    print(f'Loss: {loss}, Accuracy: {accuracy}')
