from datasets.ds_mnist import DataSetMnist
from datasets.ds_cifar10 import DataSetCifar10
from datasets.ds_dtd import DataSetDTD
from datasets.ds_fashion  import DataSetDeepFashion

class DatasetFactory(object):
    """
    Dataset simple factory method
    """

    @staticmethod
    def create(params):
        """
        Creates Dataset based on detector type
        :param params: Dataset settings
        :return: Dataset instance. In case of unknown Dataset type throws exception.
        """

        # TODO: ?use logger
        if params['DATASET']['name'] == 'mnist':
            return DataSetMnist(params['DATASET']['path'],
                                batch_size_train=params['DATASET']['batch_size'],
                                batch_size_val=params['DATASET']['batch_size_val'],
                                download=params['DATASET']['download'])
        elif params['DATASET']['name'] == 'cifar10':
            return DataSetCifar10(params['DATASET']['path'],
                                  batch_size_train=params['DATASET']['batch_size'],
                                  batch_size_val=params['DATASET']['batch_size_val'],
                                  download=params['DATASET']['download'])
        elif params['DATASET']['name'] == 'dtd':
            return DataSetDTD(params['DATASET']['path'],
                              batch_size_train=params['DATASET']['batch_size'],
                              batch_size_val=params['DATASET']['batch_size_val'])
        elif params['DATASET']['name'] == 'fashion':

            if params['MODEL']['name'] == 'lenet_in3x32x32_out10':
                return DataSetDeepFashion(params['DATASET']['path'],
                                          batch_size_train=params['DATASET']['batch_size'],
                                          batch_size_val=params['DATASET']['batch_size_val'],
                                          fin_scale=32)
            elif params['MODEL']['name'] == 'inception':
                return DataSetDeepFashion(params['DATASET']['path'],
                                          batch_size_train=params['DATASET']['batch_size'],
                                          batch_size_val=params['DATASET']['batch_size_val'],
                                          fin_scale=299)
            else:
                return DataSetDeepFashion(params['DATASET']['path'],
                                          batch_size_train=params['DATASET']['batch_size'],
                                          batch_size_val=params['DATASET']['batch_size_val'],
                                          fin_scale=224)



        raise ValueError("DatasetFactory(): Unknown Dataset type: " + params['Dataset']['type'])
