<!--- ###################################################### --->

# Dataset used in experiments
For experiments I used DeepFashion dataset
http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html

To create dataset for training I used separately ```dataset_create_my.py```. 
In this series of experiments I used next categories from dataset: 
```['Tee', 'Hoodie', 'Skirt', 'Shorts', 'Dress', 'Jeans']```

# Training LeNet-like model on DeepFashion dataset (baseline)

The goal here is to train simple model on cifar10 dataset without gpu support.

## Description

Experiment e0017

Training time [cpu]: ~ 205 min. (30 epochs), later on I run it on gpu, ~ 60 min (20 epoch)
For compasison I used only 20 epoch later to speed up the computation.

## Deliverables

- [x] model
  - `e0017_model_best.pth`  
- [x] log files
  - 'e0017_log*' 
  
## Interpretation

As we can see on the next plots, around 20 epoch our model started to overfit.

![Acc vs Epoch](fig/e_0017_train_acc.PNG?ra=true "Train Acc vs Epoch")
![Acc vs Epoch](fig/e_0017_test_acc.png?ra=true "Test Acc vs Epoch")
![Acc vs Epoch](fig/e_0017_train_loss.png?ra=true "Train Loss vs Epoch")
![Acc vs Epoch](fig/e_0017_test_loss.png?ra=true "Test Loss vs Epoch")

Next experiments will be compared to this basic, but 20 epochs:
 

![Acc vs Epoch](fig/e_0055_test_acc.png?ra=true "Test Acc vs Epoch")
![Acc vs Epoch](fig/e_0055_test_loss.png?ra=true "Test Loss vs Epoch")

## Conclusion

We get around 79% accuracy on baseline model. Let's try what will change if we conduct some changes. 

<!--- ###################################################### ---> 

# Data augmentation 1

I desided to try some augmentations. First let's try random affine transformations.

## Description

Experiment e0065

Training time [cpu]: ~ 60 min. (20 epochs)

```
init_transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.RandomAffine(15, (0.1, 0.1)),
            transforms.Resize(fin_scale),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean,
                                 std=imagenet_std)
        ])
```
        

## Deliverables

- [x] model
  - `e0065_model_best.pth`  
- [x] log files
  - 'e0065_log*' 
  
## Interpretation

On first we can see transformation applied. Second image shows that test accuracy went down for 3%. 



![Acc vs Epoch](fig/e_0065_images.png?ra=true "Images transformation")
![Acc vs Epoch](fig/e_0065_test_accuracy.png?ra=true "Test Accuracy vs Epoch")

 

## Conclusion

Maybe this model is vulnerable to this kind of augmentation. 
We should change model or change augmentation to get better results.  
<!--- ###################################################### ---> 

# Experiments with pre-trained model

I wanted to experiment with different models and compare them. 
First I tried to run ```vgg19``` on my local machine. 1/295 of epoch took 2 mins. 
Simple calculations lead me to UCU GPU. 
I tried to run pretrained resnet and pretrained vgg.
Unfortunatelly I keep getting some errors with connection to during this models were running there.
For LeNet I did not got any errors, so other experiments will be with LeNet.

I tried not once and one experiment with ResNet survived 8 epochs!

## Description

Experiment e0068

Training time [cpu]: ~ 40 min. (8 epochs)

```
model = resnet18(pretrained=True)
            for param in model.parameters():
                param.requires_grad = False

            num_of_features = model.fc.in_features
            model.fc = Linear(num_of_features, 6)
```
        

## Deliverables

- [x] model
  - `e0068_model_best.pth`  
  
## Interpretation

On next plots we can see how ResNet (gray one) overperforms LeNet from the first epoch!

![Acc vs Epoch](fig/e_0068_test_acc.png?ra=true "Test Acc vs Epoch")
![Acc vs Epoch](fig/e_0068_test_loss.png?ra=true "Test Loss vs Epoch")

 

## Conclusion

ResNet is good. But for ResNet, VGG and other complex models we need good computational power. 
As I did not manage to get UCU GPU to work with this models - I will continue with simple LeNet.
  
<!--- ###################################################### ---> 

# Manually cleaning

I decided to look, which category model is getting wrong.
![Acc vs Epoch](fig/e_0055_accuracy_dress.png?ra=true "Accuracy Dress")
![Acc vs Epoch](fig/e_0055_accuracy_hoodie.png?ra=true "Accuracy Hoodie")
![Acc vs Epoch](fig/e_0055_accuracy_jeans.png?ra=true "Accuracy Jeans")
![Acc vs Epoch](fig/e_0055_accuracy_scirt.png?ra=true "Accuracy Scirt")
![Acc vs Epoch](fig/e_0055_accuracy_shorts.png?ra=true "Accuracy Shorts")
![Acc vs Epoch](fig/e_0055_accuracy_tee.png?ra=true "Accuracy Tee")

As we can see, hoodie accuracy is drastically bad. 
I looked on the real images and I think, I understand neural net :)
![Acc vs Epoch](fig/Hoodie_strange.png?ra=true "Data")
I would also name some of them tee or scirt.
So I decided to manually clean data for Hoodies.

## Description

Experiment e0072

Training time [cpu]: ~ 60 min. (20 epochs)

Deleted about 200 images of fake hoodies.        

## Deliverables

- [x] model
  - `e0072_model_best.pth`  
  
## Interpretation

It seems that accuracy for Hoodie became better, but not significantly

![Acc vs Epoch](fig/e_0072_accuracy_hoodie.png?ra=true "Accuracy Hoodie")


## Conclusion

It is a good idea to find some better data for hoodie, or investigate more, 
why it misses to tune something.
  
<!--- ###################################################### --->

# Scheduler for learning rate

How schedule of learning rate can influence accuracy?

## Description

Experiment e0078

Training time [cpu]: ~ 60 min. (20 epochs)

```
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)
```
        

## Deliverables

- [x] model
  - `e0078_model_best.pth`  
  
## Interpretation
We can see that test accuracy and loss remain the same while train became less sensitive to overfit.

![Acc vs Epoch](fig/e_0078_train_acc.PNG?ra=true "Train Acc vs Epoch")
![Acc vs Epoch](fig/e_0078_test_acc.png?ra=true "Test Acc vs Epoch")
![Acc vs Epoch](fig/e_0078_train_loss.png?ra=true "Train Loss vs Epoch")
![Acc vs Epoch](fig/e_0078_test_loss.png?ra=true "Test Loss vs Epoch")

 

## Conclusion

Scheduling learning rate to be smaller saves us from overfitting.
  
<!--- ###################################################### --->

# Adam for optimization

What if we change optimization method?

## Description

Experiment e0079

Training time [cpu]: ~ 40 min. (14 epochs)

```
optimizer = torch.optim.Adam(model.parameters(),
                                    lr=params['TRAIN']['lr'])
```
        

## Deliverables

- [x] model
  - `e0079_model_best.pth`  
  
## Interpretation

Converging is faster and accuracy is slightly better.

![Acc vs Epoch](fig/e_0079_train_acc.PNG?ra=true "Train Acc vs Epoch")
![Acc vs Epoch](fig/e_0079_test_acc.png?ra=true "Test Acc vs Epoch")
![Acc vs Epoch](fig/e_0079_train_loss.png?ra=true "Train Loss vs Epoch")
![Acc vs Epoch](fig/e_0079_test_loss.png?ra=true "Test Loss vs Epoch")

 

## Conclusion

In my case Adam optimazer is better choice then SGD.
  
<!--- ###################################################### --->

# Grauscale

I am interestend, how important color information is for task of category classification. 
So I decided to try convert everything into grayscale and see what happens.

## Description

Experiment e0080

Training time [cpu]: ~ 40 min. (14 epochs)

```
        init_transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.Resize(fin_scale),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean,
                                 std=imagenet_std)
        ])
```
        

## Deliverables

- [x] model
  - `e0080_model_best.pth`  
  
## Interpretation

On next plots we can see how ResNet (gray one) overperforms LeNet from the first epoch!

![Acc vs Epoch](fig/e_0068_test_acc.png?ra=true "Test Acc vs Epoch")
![Acc vs Epoch](fig/e_0068_test_loss.png?ra=true "Test Loss vs Epoch")

 

## Conclusion

ResNet is good. But for ResNet, VGG and other complex models we need good computational power. 
As I did not manage to get UCU GPU to work with this models - I will continue with simple LeNet.
  
<!--- ###################################################### --->

# Ideas for improvement
1. Find GPU, or figure out how to work without crashing on UCU one. And try different complex models like VGG, Resnet, Inception etc.
2. Look carefully on dataset, maybe for some categories find other data.
3. Try to take small amount of proper data and fine-tune some model.
