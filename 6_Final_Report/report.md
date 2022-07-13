---
title: Image Exposure Correction Through the Application of Convolutional Neural Networks (CNNs) 
author: 1921983
date: 19th June 2022
toc: true
numbersections: true
geometry: margin=2.5cm
urlcolor: blue
bibliography: references.bib
csl: elsevier-harvard.csl
header-includes: |
    \usepackage{caption}         
    \usepackage{subcaption}
    \usepackage{fancyhdr}
    \pagestyle{fancy}
    \lfoot{Draft Prepared: 19th June 2022}
    \rfoot{Page \thepage}
    \usepackage{float}
    \let\origfigure\figure
    \let\endorigfigure\endfigure
    \renewenvironment{figure}[1][2] {
    \expandafter\origfigure\expandafter[H]
    } {
        \endorigfigure
    }
---

\newpage

# Abstract
In this Post-Module Assignment (PMA) the tasks performed will assess the understanding of Convolutional Neural Networks (CNNs) through its application for image exposure
correction. The PMA shall implement a CNN to take an input image (that is incorrectly exposed) and correct the image to be as close to ground truth as possible. This document shall detail the implementation and results yielded from said implementation. 

\newpage

# Introduction
As detailed in the brief, exposure refers to the total quantity of light energy incident on a sensitive material. In a digital camera it describes the process of controlling the light energy reaching a sensor. Despite digital cameras having sophisticated automated modes and scenery settings the complex nature of some scenes result in under or over-exposed images. 

# Input/Output Manipulations
All of the Input/Output (I/O) manipulations are performed in the dataset creation class. The program uses an abstract 'Dataset' class to define it's own bespoke data set object using object oriented principles like overriding and inheritance, . Functions that have names preceded by and followed by double underscores are internal functions that will have default functionality that can be overriden if necessary [@lott_2014].
```py
# dataset class inherited from torch dataset abstract class
class MyDataset(Dataset):
    # init takes root dir, 
    # assuming that folders below are INPUT_IMAGES and GT_IMAGES
    # i.e. root folder is training/ or validation/
    def __init__(self, root_dir):
        self.root_dir = root_dir
        # create variable to store path to input images
        self.input_dir = root_dir + "INPUT_IMAGES"
        # create variable to store path to GT images
        self.gt_dir = root_dir + "GT_IMAGES"
        # get list of files in directory of input images directory
        self.input_list_files = os.listdir(self.input_dir)
        # get list of files in directory of ground truth directory
        self.gt_list_files = os.listdir(self.gt_dir)
```
The `.init()`{.py} method is overriden to take the root directory when ```MyDataset```{.py} is initalised. It then allows the program to define the directories that contain the images of relevance.

```py
    def __len__(self):
        # override length method to show the number of input images
        return len(self.input_list_files)
```
The `.len()`{.py} method is overriden to tell the dataloader how many items are in the dataset. In this case, it would be the number of input images.

```py
    # override get item method to return input image 
    # and the ground truth image for a given index
    def __getitem__(self, index):
        # get image file name from list of images @ given index
        img_file = self.input_list_files[index]
        # create path using os.join using input dir and img file
        img_path = os.path.join(self.input_dir, img_file)
        # open image @ path
        image = Image.open(img_path)
        # create np array of image
        input_image = np.array(image)

        # there are 5 input images per GT image 
        # therefore floor division by 5 of index gives us GT index
        gt_index = index // 5
        # get image file name at index from GT list
        gt_image_file = self.gt_list_files[gt_index]
        # create path using os.join using gt dir and img file
        gt_image_path = os.path.join(self.gt_dir, gt_image_file)
        # open image from path
        gt_image = Image.open(gt_image_path)
        # create array from image
        target_image = np.array(gt_image)

        # transform both images - resize to 256 x 256
        augmentations = both_transform(image=input_image, image0=target_image)
        # get images from augmentations
        input_image = augmentations["image"]
        target_image = augmentations["image0"]

        # transform input image
        input_image = transform_only_input(image=input_image)["image"]
        # transform target image
        target_image = transform_only_mask(image=target_image)["image"]

        # return input and target
        return input_image, target_image
```
Overriding the `.getitem()`{.py} method is key to the functioning of the dataset and will need to be bespoke dependent on data format and preference of the developer. In this case, the program passes an index to the method (where the index is a number from 0 - Total Number of Images). It returns two numpy arrays, one being an input image to be corrected and another the target (ground truth) image.

To get both an input and ground truth (GT) image, the same process is performed:

1. Get index (For GT, floor division is used to get the corresponding index)
2. Get image file name from list of file names
3. Create image path by joining path and file name
4. Open image
5. Convert to numpy array
6. Perform transformations (Resize, Flip, Colour Jitter)
7. Return images

\newpage

# Network
The proposed architecture is a Generative Adversarial Network (GAN) composed of two neural networks:

1. A generator that will correct the exposure of an over or under-exposed image.
2. A discriminator which is trained to distinguish correctly exposed from incorrectly exposed images.

The aim of the generator is to trick the discriminator into thinking a generated (fake) image is a real image. The discriminator is only used during the training phase to optimise the generator [@steffens2018deep]. 

![Model Implementation Explanation](figures/example.png)

## Generator
### Architecture
The generator is designed using a typical encoder-decoder architecture that is improved with skip-connections. Skip connections can improve the gradient through the network. This architecture has been inspired by a number of sources [@steffens2018deep] & [@zhuimage2image] & [@persson]. 

![1D Diagram of Convolutional Block](figures/oneD_Conv_Rep.png)

In the proposed architecture; for each given input, there are four filters built where each involves a different set of pixels and then the outputs are merged together.

![Generator Network Architecture](figures/gen_arch.png)

The generator loss penalises itself for producing an image that the discriminator network classifies as fake. Backpropagation adjusts each weight in the right direction by calculating the weight's impact on the output — how the output would change if you changed the weight [@8478390]. Backpropagation starts at the output and flows back through the discriminator into the generator.

![Backpropagation in Generator Training](figures/gen_arch_2.png)

### Code
```py
class Block(nn.Module):
    # similar to CNNBlock
    # down is used for encoder/decoder
    # act is different for encoder and decoder
    def __init__(self, in_channels, out_channels, down=True, 
                act="relu", use_dropout=False):
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, 
                    bias=False, padding_mode="reflect")
            # if encoder we want a down sample, always a stride of 2 - similar to UNet
            if down
            # else in encoder phase
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.down = down

    def forward(self, x):
        x = self.conv(x)
        # return dropout if use dropout is true, else just x
        return self.dropout(x) if self.use_dropout else x


class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            # slope of 0.2, no batch norm in inital layer
            nn.LeakyReLU(0.2),
        ) # 128
        # encoder phases, leaky activation
        # copied for 6 layers
        self.down1 = Block(features, features * 2, 
            down=True, act="leaky", use_dropout=False) # 64
        self.down2 = Block(features * 2, features * 4, 
            down=True, act="leaky", use_dropout=False) # 32
        self.down3 = Block(features * 4, features * 8, 
            down=True, act="leaky", use_dropout=False) # 16
        self.down4 = Block(features * 8, features * 8, 
            down=True, act="leaky", use_dropout=False) # 8
        self.down5 = Block(features * 8, features * 8, 
            down=True, act="leaky", use_dropout=False) # 4
        self.down6 = Block(features * 8, features * 8, 
            down=True, act="leaky", use_dropout=False) # 2
        
        self.bottleneck = nn.Sequential(nn.Conv2d(features * 8, 
            features * 8, 4, 2, 1), nn.ReLU()) # 1x1

        # up decoder phase, transpose convolution
        # we then concatenate these 
        self.up1 = Block(features * 8, features * 8, 
            down=False, act="relu", use_dropout=True)
        self.up2 = Block(features * 8 * 2, features * 8, 
            down=False, act="relu", use_dropout=True)
        self.up3 = Block(features * 8 * 2, features * 8, 
            down=False, act="relu", use_dropout=True)
        # no dropout from here
        self.up4 = Block(features * 8 * 2, features * 8, 
            down=False, act="relu", use_dropout=False)
        self.up5 = Block(features * 8 * 2, features * 4, 
            down=False, act="relu", use_dropout=False)
        self.up6 = Block(features * 4 * 2, features * 2, 
            down=False, act="relu", use_dropout=False)
        self.up7 = Block(features * 2 * 2, features, 
            down=False, act="relu", use_dropout=False)
        
        # last conv transpose
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, in_channels, 
                kernel_size=4, stride=2, padding=1),
            # each pixel value to be between -1 and 1
            nn.Tanh(),
            # can use sigmoid for 0-1
        )

    def forward(self, x):
        # convolve of each layer, down phases
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.bottleneck(d7)
        # same in upward phase
        # but we concatenate in upward section
        up1 = self.up1(bottleneck)
        # concatentate with last part of down layer
        # 1st goes with last, 2nd with penultimate etc
        up2 = self.up2(torch.cat([up1, d7], 1))
        up3 = self.up3(torch.cat([up2, d6], 1))
        up4 = self.up4(torch.cat([up3, d5], 1))
        up5 = self.up5(torch.cat([up4, d4], 1))
        up6 = self.up6(torch.cat([up5, d3], 1))
        up7 = self.up7(torch.cat([up6, d2], 1))
        # very similar to UNet, lots more down sampling
        return self.final_up(torch.cat([up7, d1], 1))
```

## Discriminator
The discriminator implemented follows an architecture used by [@lindernoren_2022] with performance improvements using loops and less parameters. The discriminator uses back propogation to update its weights from the loss function [@google]. The discriminator loss penalises itself for being *fooled* by the generator.

![Backpropagation in Discriminator Training](figures/gan_arch_2.png)

In the figure above, the two "Sample" boxes represent INPUT_IMAGES and GT_IMAGES feeding into the discriminator. During discriminator training the generator does not train. Its weights remain constant while it produces examples for the discriminator to train on [@google].

### Code
```py
# disc class, we send in x and y
class Discriminator(nn.Module):
    # 3 channels for rgb
    # 4 features, we use CNNBlock 4 times
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        # call super
        # create block
        self.initial = nn.Sequential(
            # no batchnorm in inital block
            nn.Conv2d(
                in_channels * 2,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2),
        )

        # we concatenate x and y
        layers = []
        in_channels = features[0]
        # for each feature, and skip inital
        for feature in features[1:]:
            # add cnn block for each feature
            layers.append(
                CNNBlock(in_channels, feature, stride=1 if feature == features[-1] else 2),
            )
            in_channels = feature

        # we need to append one final layer otherwise errors occur
        layers.append(
            nn.Conv2d(
                in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"
            ),
        )
        # unpack into sequential
        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        # concat first x and y
        x = torch.cat([x, y], dim=1)
        # send through inital
        x = self.initial(x)
        # send through model
        x = self.model(x)
        return x
```

\newpage

# Configuration Parameters
All config parameters are defined in capital letter for ease of identification. This provides a simple and user-friendly way to change learning or reading parameters before or during training of the model

```py
# Used to send images to processing device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# location of images
TRAIN_FOLDER = "exposure_dataset\\training\\"
VALID_FOLDER = "exposure_dataset\\testing\\"
# loading / saving checkpoints
LOAD_MODEL = False
SAVE_MODEL = True
TRAIN_MODEL = True
# checkpoint save paths
CHECKPOINT_DISCRIM = "disc.pth.tar"
CHECKPOINT_GENER = "gen.pth.tar"
# Learning Rate
RATE_LEARN = 1.2e-4
# Number of images per iteration
BATCH = 32
TRAIN_BATCH = 16
# Number of threads
WORKERS = 12
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
EPOCHS = 250
```

\newpage

# Loss Function

$$ L = L1(G(A), B) + λLD([G(A), A, B], [fake, real]) $$ 

Where $A$ belongs to a set of improperly exposed images, $B$ belongs to a set of properly exposed images. $G(A)$ is the output the generator network produces for an $A$ input image, and $LD$ is the crossentropy in the discriminator network. On one hand, the mean absolute error $L1$ provides an intuitive loss function to force low-frequency correctness on the generator network. The pixel-wise error between the generated output for an improperly exposed image $G(A)$ and its properly exposed counterpart $B$ guides the network towards a coherent output [@isolaimage].

```py
BCE = nn.BCEWithLogitsLoss()
L1_LOSS = nn.L1Loss()
```

During training the loss values are shown using the `tdqm`{.py} library. The progress bar shows each iteration's `D_fake`{.py} and `D_real`{.py} values. This is done using the code below: 

```py
if idx % 10 == 0:
    loop.set_postfix(
        D_real=torch.sigmoid(D_real).mean().item(),
        D_fake=torch.sigmoid(D_fake).mean().item(),
    )
```

![Loss Values](figures/TQDM_LOSS.png)

\newpage

# Training and Testing
## Training Function
GANs contains two separately trained networks. Due to this, its training algorithm must address two issues: 1) GANs must train two networks, 2) GAN convergence is difficult to identify.

The process for training the generator is as follows:

1. Get input image
2. Produce generator output of input image
3. Get classification of output from discriminator
4. Calculate loss from discriminator classification
5. Backpropagate through both the discriminator and generator to obtain gradients
6. Use gradients to change only the generator weights

```py
# train function to train models
def train_fn(disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, g_scaler, d_scaler,):
    # use TQDM to show percentage completion
    loop = tqdm(loader, leave=True)

    # get images from dataloader
    for idx, (x, y) in enumerate(loop):
        # send to GPU
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        # Train Discriminator
        with torch.cuda.amp.autocast():
            # generate image from input image
            y_fake = gen(x)
            # apply discriminator on real images
            D_real = disc(x, y)
            # calculate loss
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            # apply discriminator on fake and GT image
            D_fake = disc(x, y_fake.detach())
            # caculate loss
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            # mean loss
            D_loss = (D_real_loss + D_fake_loss) / 2

        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train generator
        with torch.cuda.amp.autocast():
            # apply discriminator on fake and GT image
            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1_loss(y_fake, y) * L1_LAMBDA
            G_loss = G_fake_loss + L1

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )
```

## Testing Function
Below shows the function which takes a dataloader, the generator model and path to an output folder. This function puts all input images through the generator and outputs them in batches of `TEST_BATCH`{.py}. This images can then be manually inspected.
```py
def testing_gen(loader, gen, folder):
    # get input image and ground truth from data loader
    loop = tqdm(loader, leave=True)

    # get images from dataloader
    for idx, (x) in enumerate(loop):
        # send to GPU
        x = x.to(DEVICE)
        gen.eval()
        with torch.no_grad():
            # generate image from input image
            y_fake = gen(x)
            y_fake = y_fake * 0.5 + 0.5  # remove normalization#
            # save the generated image in evaluations folder with epoch number
            save_image(y_fake, folder + f"/y_gen_{idx}.png")
            # save the corresponding input image
            save_image(x * 0.5 + 0.5, folder + f"/input_{idx}.png")
        gen.train()
```

## Main Function
In the main function; both models, their optimisers, and the loss functions are initilised.

Using the configuration parameters, the main function can decide which logic flow to execute. This allows for simple changing of processes from training to testing and changing inputs. 
```py
def main():
    # create discriminator in GPU
    disc = Discriminator(in_channels=3).to(DEVICE)
    # create generator in GPU
    gen = Generator(in_channels=3, features=64).to(DEVICE)
    # create optimisers for discriminator and generator
    opt_disc = optim.Adam(disc.parameters(), lr=RATE_LEARN, betas=(0.5, 0.999),)
    opt_gen = optim.Adam(gen.parameters(), lr=RATE_LEARN, betas=(0.5, 0.999))
    # loss functions
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    # if user chooses to load saved model
    if LOAD_MODEL:
        # load GEN and DISC checkpoints
        load_checkpoint(CHECKPOINT_GENER, gen, opt_gen, RATE_LEARN,)
        load_checkpoint(CHECKPOINT_DISCRIM, disc, opt_disc, RATE_LEARN,)
    
    # data dataset and loader for testing function
    test_dataset = MyDataset(root_dir=TEST_FOLDER, test=True)
    # create dataloaders
    test_loader = DataLoader(
        test_dataset,
        batch_size=TEST_BATCH,
        shuffle=True,
        num_workers=WORKERS,
    )
    
    # if model is being trained    
    if TRAIN_MODEL:
        # create dataset
        train_dataset = MyDataset(root_dir=TRAIN_FOLDER)
        # create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH,
            shuffle=True,
            num_workers=WORKERS,
        )
        g_scaler = torch.cuda.amp.GradScaler()
        d_scaler = torch.cuda.amp.GradScaler()

        # create validation dataset
        val_dataset = MyDataset(root_dir=VALID_FOLDER)
        # create dataloader, of 1 batch to show same image every epoch
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

        # for each epoch in range 0 to epochs
        for epoch in range(EPOCHS):
            # call train function passing params defined above
            train_fn(
                disc, gen, train_loader, opt_disc, 
                opt_gen, L1_LOSS, BCE, g_scaler, d_scaler,
            )

            # user wants ot save model 
            if SAVE_MODEL and epoch % 5 == 0:
                # save disc and gen checkpoints
                save_checkpoint(gen, opt_gen, filename=CHECKPOINT_GENER)
                save_checkpoint(disc, opt_disc, filename=CHECKPOINT_DISCRIM)

            # save an example from validation loader
            save_examples(gen, val_loader, epoch, folder="evaluation")
    # not being trained
    else:
        # call testing function
        testing_gen(test_loader, gen, folder='test_results')


if __name__ == "__main__":
    # if called from command line, run main func
    main()
```

## Test Cases
Test case functions have been authored to check that all components of model implementation are working as intended.
```py
# function to test generator
def test_gen():
    # random image of correct size
    x = torch.randn((1, 3, 256, 256))
    # initialise model
    model = Generator(in_channels=3, features=64)
    # generate image from x
    preds = model(x)
    # shape should be the same
    print(preds.shape)

def test_disc():
    # create random x and y
    x = torch.randn((1, 3, 256, 256))
    y = torch.randn((1, 3, 256, 256))
    # init model
    model = Discriminator(in_channels=3)
    # pass and x and y into model
    preds = model(x, y)
    # check model and preds is correct
    print(model)
    print(preds.shape)

def test_dataset():
    # create dataset
    dataset = MyDataset("exposure_dataset/training/")
    # load dataset
    loader = DataLoader(dataset, batch_size=5)
    for x, y in loader:
        # get images from loader
        print(x.shape)
        save_image(x, "x.png")
        save_image(y, "y.png")
```

\newpage

# Utility Functions
## Transformations
Transformations are performed on all images put through the custom data loader. Firstly, the program resizes both the input and ground truth images. This is done using functions within the `albumenations`{.py} library.
```py
# resize images to 256 x 256 
both_transform = A.Compose(
    [A.Resize(width=256, height=256),], additional_targets={"image0": "image"},
)
```
Two more transformations are performed, one specific to the input image and one specific to the ground truth image. To add variety to the input images and train the generator more effectively, the program adds a horizontal flip and random colour jitter at a probibility of 0.5.
```py
transform_only_input = A.Compose(
    [   
        # flip image, with probabilty of 0.5
        A.HorizontalFlip(p=0.5),
        # Add colour jitter
        A.ColorJitter(p=0.2),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

transform_only_mask = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)
```
## Test Images
One test image is generated per epoch. The generated images can be inspected during or after training to evaluate the progress and effectiveness of the model. The generated images are saved in an evaluation folder. The code from this function is very similar to the testing function.
```py
# utility functions
# save an example for each epoch
def save_examples(gen, val_loader, epoch, folder):
    # get input image and ground truth from data loader
    x, y = next(iter(val_loader))
    # send to GPU
    x, y = x.to(DEVICE), y.to(DEVICE)
    gen.eval()
    with torch.no_grad():
        # generate image from input image
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5  # remove normalization#
        # save the generated image in evaluations folder with epoch number
        save_image(y_fake, folder + f"/y_gen_{epoch}.png")
        # save the corresponding input image
        save_image(x * 0.5 + 0.5, folder + f"/input_{epoch}.png")
        if epoch == 1:
            # for first epoch, save the ground truth image for this generated image
            save_image(y * 0.5 + 0.5, folder + f"/label_{epoch}.png")
    gen.train()
```

## Checkpoint Processing
Checkpoints are used to save the progress of the model's training. The model's weights in the form of a `state_dict`{.py} and saves to a `.tar`{.py} file. In the case of the program failing mid-train, checkpoints can be used to re-load to a pre-trained state. Checkpoints will be always be loaded after the intial training period.
```py
# function to save checkpoints for generator and discriminator
def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    # notify user of checkpoint save
    print("=> Saving checkpoint")
    # create checkpoint object of states of the model and corresponding optimiser
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    # save checkpoint object
    torch.save(checkpoint, filename)

# function to load checkpoints for generator and discrimiator
def load_checkpoint(checkpoint_file, model, optimizer, lr):
    # notify user
    print("=> Loading checkpoint")
    # load checkpoint from checkpoint file
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
    # load state dicts into model and optimiser
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
```

\newpage

# Model Evaluation
## Learning Rate 1
Using a learning rate of $2e^-4$ the following results were given. The model was trained for a total of 500 epochs using this learning rate. All other parameters were kept the same. 

![](figures/label_1.png){width=20%}
![](figures/input_1.png){width=20%}
![](figures/first_gen_2.png){width=20%}
![](figures/final_gen_2_good.png){width=20%}
![](figures/final_gen_2_bad.png){width=20%}
*5 Figures - The ground truth image (Far Left), An example input image (Middle Left), G(X) before any significant training (Middle), A good example of G(X) after 500 epochs of training (Middle Right), A bad example of G(X) after 300 epochs of training (Far Right).*

In the above images there is a significant improvement between image 2 and image 3, however in image 4 there is a serious degradation in quality. This is due to the learning rate being too high and over-training the model. The results above are an example of GAN convergence. Following this discovery, the learning rate was adjusted and the model was retrained. (See below for results)

## Learning Rate 2
Using a learning rate of $1.2e^-4$ the following results were given. The model was trained for a total of 250 epochs using this learning rate. All other parameters were kept the same. 

![](figures/label_1.png){width=25%}
![](figures/input_39.png){width=25%}
![](figures/first_gen_1.png){width=25%}
![](figures/final_gen_1_good.png){width=25%}
*4 Figures - The ground truth image (Far Left), An example input image (Middle Left), G(X) before any significant training (Middle Right), A good example of G(X) after 250 epochs of training (Right).*

In the above images, there is a highly improved result after only 250 epochs of training. Utilising a lower learning rate had a large impact on the results. This learning rate was then adopted to test on larger batches of images. 

\newpage

# Analysis and Conclusions
## Results
Using the function defined in section 7.2, the following images have been passed to the generator model.

![Batch of 16 unseen input images](figures/test_input_1.png)

Below are the results of the above images.

![Batch of 16 generated by generator](figures/test_y_gen_1.png)

Upon inspection, it can be seen that 60-70% of the generated images yield a useful result. Some images experience blurring, while others do not. The model performs well in a variety of conditions due to the variety of images using in the training phase. 

## Analysis
During the training process, both the discriminator and generator networks have been trained in conjunction with eachother to ensure the generator does not converge towards image that can pass the discriminator stage while not looking correctly exposed. It has been noted in this experiment and previous papers, that if training of both models is not completed simultaneously then the generator creates images that are unnatural (i.e. the image is entirely red, green, blue etc.) [@afifi2021learning]. Therefore for meaningful results to be obtained, the discrimiator must be able to differ between $G(A)$, $A$ and $B$ (where $A$ is an input image, $B$ is a ground truth image, and $G(A)$ is a generated version of $A$).

The implemented network is a viable alternative to traditional image-to-image translation problems (For example, CycleGAN), using at least two orders of magnitude less trainable parameters than other state-of-the-art networks [@raiunpaired].

As the generator improves with training, the discriminator performance gets continually worse because it can no longer easily tell the difference between real and fake images. If the generator succeeds perfectly, then the discriminator has a 50% accuracy. In effect, the discriminator flips a coin to make its prediction [@google]. This creates a problem of convergence of the GAN as a whole: the discriminator feedback gets less meaningful over time as it is essentially guessing. If the GAN continues to train past the point when the discriminator is giving completely random feedback, then the generator starts to train on bad feedback, and its own quality may collapse. This is precisely what we saw earlier in the report when using a $2e^-4$ learning rate.

## Conclusion
In conclusion, the implemented model yields acceptable results with minimal training time. It is clear that the model consistently struggles correcting the exposure high exposure images due to the distortion of the image. On the other hand, the lower exposure images or images with low exposure portions like the examples of the car. We can see that the exposure of the car and its interior is corrected successfully, while the surrouding area is not. 

\newpage

# Alternative Approaches
Histogram Equalization is one viable alternative. It is an image processing technique that adjusts the contrast of an image by using its histogram. To enhance the image’s contrast, it spreads out the most frequent pixel intensity values or stretches out the intensity range of the image. By accomplishing this, histogram equalization allows the image’s areas with lower contrast to gain a higher contrast [@poddar2013non].

Manual exposure using digital cameras can allow the user to manipulate images however they like and pick an exposure suitable for the current surroundings. Another alternative is to use RAW image format when shooting on a digital camera. This allows you to manipulate ISO and exposure in post production with no quality degradation.

Furthermore, Adobe Lightroom, used in conjunction with RAW images provides an ideal environment for image exposure manipulation and correction.

Finally, on a case by case basis, it is clear that using manual correction techniques would be preferred. However, in industry, or when using large batches of images; a trained neural network could be suitable to solve the problem given. 




\newpage

# References