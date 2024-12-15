## 1. UML
TODO

## 2. Project organization
```
├── documentation       <- UML diagrams
├── balancers           <- Package with balancers and utils
│   ├── __init__.py     <- Package identicator
│   ├── smote.py        <- SMOTE balancer (interpolation)
│   ├── adasyn.py       <- ADASYN balancer (interpolation)
│   ├── augmentation.py <- Augmentation balancer (augmenting images like rotations, etc.)
│   ├── autoencoder.py  <- Autoencoder balancer (learning needed!)
│   ├── dgan.py         <- DGAN balancer (learning needed!)
│   ├── balancer.py     <- General balancer with all balancers (aggregating all of the above)
│   ├── annotations.py  <- Annotations module
│   └── configuration_reader.py  <- Balancer configuration reader
├── maritime-flags-dataset    <- Source and balanced flags (A-Z)
│   ├── ADASYN_balanced_flags <- Balanced flags by using ADASYN balancer
│   ├── SMOTE_balanced_flags  <- Balanced flags by using SMOTE balancer
│   ├── AUGMENTATION_balanced_flags  <- Balanced flags by using Augmentation balancer
│   ├── DGAN_balanced_flags  <- Balanced flags by using DGAN balancer
│   ├── AE_balanced_flags    <- Balanced flags by using Autoencoder balancer
│   ├── combined_flags       <- Combined/test images 
│   ├── two_flags            <- Balanced two flags (A and B) per 1000 images
│   └── imbalanced_flags     <- Source folder with imbalanced flags
├── balance.py <- Balancing dataset by using balancers package (BALANCING)
├── balancer_configuration.json <- Balancer configuration
└── detection.py <- Training and testing image detectors (EVALUATING)
```
## 3. Balancing approaches
### 3.1 Augmentation
![image](https://github.com/user-attachments/assets/853a495e-1c16-4de4-8ad1-1334a6338bcd)

### 3.2 SMOTE
![image](https://github.com/user-attachments/assets/29c468ba-70f1-4650-8110-82f006c1075b)

### 3.3 ADASYN
![image](https://github.com/user-attachments/assets/7a004a3e-8bf9-468a-a375-4d30d2c98735)

### 3.4 Autoencoder
![image](https://github.com/user-attachments/assets/63f77f71-79b2-4879-b1e7-1dcc876de327)

### 3.5 Deep Convolutional GAN
![image](https://github.com/user-attachments/assets/cd73ea8b-2670-4db2-af29-7475bc267b35)
