# pedestro-detecto
CNN for binary image classification of pedestrians.

## Setup

*Tested on python 3.7.7 and 3.7.10*

Check python version:

```
python --version
```

Clone the repository:

```
git clone https://github.com/lieblius/pedestro-detecto.git
```

Install requirements:

```
pip install -r requirements.txt
```

If you would like to use custom test data, put the test data or any custom data in this format:

```
custom_data/
     |
     ----->not_pedestrian/
     |            |
     |            -----> *.jpg/png
     ------>pedestrian/
                |
                -------> *.jpg/png
```

Run `main.py`. Default is to use trained model, to train again change the path in `config.py` or delete `models/trained/model.pth`.

Some sample output with verbose flag set during testing:

```
Train Acc |100%|██████████| 316/316 [00:06<00:00, 48.92it/s]
Evaluation accuracy: 1.0
Val Acc |100%|██████████| 89/89 [00:01<00:00, 48.98it/s]
Evaluation accuracy: 0.9350282485875706
not_pedestrian/dark.jpg: pred=not_pedestrian, actu=not_pedestrian, correct
not_pedestrian/night_1.jpg: pred=not_pedestrian, actu=not_pedestrian, correct
not_pedestrian/night_2.jpg: pred=not_pedestrian, actu=not_pedestrian, correct
not_pedestrian/red.jpg: pred=not_pedestrian, actu=not_pedestrian, correct
pedestrian/bright_night.png: pred=pedestrian, actu=pedestrian, correct
pedestrian/crowded_crosswalk.png: pred=pedestrian, actu=pedestrian, correct
pedestrian/four_crosswalk.png: pred=pedestrian, actu=pedestrian, correct
pedestrian/girl_crosswalk.jpg: pred=pedestrian, actu=pedestrian, correct
pedestrian/night.png: pred=pedestrian, actu=pedestrian, correct
pedestrian/schoolbus_crossing.png: pred=pedestrian, actu=pedestrian, correct
pedestrian/sunrise_crosswalk.png: pred=pedestrian, actu=pedestrian, correct
pedestrian/umich_crosswalk.jpg: pred=pedestrian, actu=pedestrian, correct
pedestrian/umich_crosswalk_2.jpg: pred=pedestrian, actu=pedestrian, correct
Test Acc |100%|██████████| 4/4 [00:00<00:00,  6.92it/s]
Evaluation accuracy: 1.0
```

## Model Architecture
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 32, 100, 100]           2,432
         MaxPool2d-2           [-1, 32, 25, 25]               0
            Conv2d-3          [-1, 128, 13, 13]         102,528
         MaxPool2d-4            [-1, 128, 3, 3]               0
            Conv2d-5             [-1, 32, 2, 2]         102,432
            Linear-6                   [-1, 16]           2,064
            Linear-7                    [-1, 2]              34
================================================================
Total params: 209,490
```

## Current Results:
```
Training Accuracy: 100%
Validation Accuracy: 93.5%
Test Accuracy: 100% on "custom_data" (Only 12 or so photos)
```

## Dataset:
https://www.kaggle.com/tejasvdante/pedestrian-no-pedestrian
