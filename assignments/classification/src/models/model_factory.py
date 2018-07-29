from models.lenet_in1x28x28_out10 import LeNet1x28x28
from models.lenet_in3x32x32_out10 import LeNet3x32x32
#from models.vgg19 import Vgg19
from models.inception import Inception3
from torchvision.models import vgg19, resnet18
from torch.nn import DataParallel, Linear, Sequential


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
        elif params['MODEL']['name'] == 'vgg19':
            model = vgg19(pretrained=True)
            for param in model.parameters():
                param.requires_grad = False

            num_of_features = model.classifier[6].in_features

            features = list(model.classifier.children())[:-1]  # Remove last layer
            features.extend([Linear(num_of_features, 6)])  # Add our layer with 4 outputs
            model.classifier = Sequential(*features)  # Replace the model classifier
            print(model)
            return model
        elif params['MODEL']['name'] == 'resnet18':
            model = resnet18(pretrained=True)
            for param in model.parameters():
                param.requires_grad = False

            num_of_features = model.fc.in_features
            model.fc = Linear(num_of_features, 6)

            #return Vgg19()
            return model
        elif params['MODEL']['name'] == 'inception':
            return Inception3()
        # elif ...

        raise ValueError("ModelFactory(): Unknown Model type: " + params['Model']['type'])
