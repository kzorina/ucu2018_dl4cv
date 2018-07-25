from models.lenet_in1x28x28_out10 import LeNet1x28x28
from models.lenet_in3x32x32_out10 import LeNet3x32x32


class ModelFactory(object):
    """
    Model simple factory method
    """

    @staticmethod
    def create(params):
        """
        Creates Model based on detector type
        :param params: Model settings
        :return: Model instance. In case of unknown Model type throws exception.
        """

        if params['MODEL']['name'] == 'lenet_in1x28x28_out10':
            return LeNet1x28x28()
        elif params['MODEL']['name'] == 'lenet_in3x32x32_out10':
            return LeNet3x32x32()
        # elif ...

        raise ValueError("ModelFactory(): Unknown Model type: " + params['Model']['type'])
