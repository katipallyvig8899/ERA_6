# ERA_6
# part 1  
# Backpropogation in network:

input: 2 neurons

one hidden layer : 2 neurons

output : 2 neurons

# Network Flow 
<img width="717" alt="Screenshot 2023-07-24 at 5 44 28 AM" src="https://github.com/katipallyvig8899/ERA_6/assets/45558037/9f6deb74-39be-4b89-a4f9-7c435c9ac31f">

# Equations 

<img width="1027" alt="Screenshot 2023-07-24 at 5 45 36 AM" src="https://github.com/katipallyvig8899/ERA_6/assets/45558037/46bd64af-e790-45fb-b658-7b96ee86279e">
Initially  hidden layer it took the input from input layer and by using initial weights, hidden layer neurons get temporary hidden value, by applying activation function hidden layer outputs would be updated. Then pass to the output layer. After the output layer each output related to corresponding error(difference between target and network predicted output). By backpropogating loss value should be minimized, weights will be updated.By using different learning rates, we observed by increasing the learning rate convergence fast but for complex data when learning rate small then loss converge but slow process.

<img width="736" alt="Screenshot 2023-07-25 at 6 44 35 AM" src="https://github.com/katipallyvig8899/ERA_6/assets/45558037/a27bc0ee-5cea-4465-8b66-4202af1e1b8e">

# part2



# Network code 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.05)
        ) # output_size = 26 | RF 3

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.05)
        ) # output_size = 24 | RF 5

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 24 | RF 5
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 12 | RF 7

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.05)
        ) # output_size = 10 | RF 11
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout(0.05)
        ) # output_size = 8 | RF 15
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout(0.05)
        ) # output_size = 6 | RF 19
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=14, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14),
            #nn.Dropout(0.05)
        ) # output_size = 6 | RF 23

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        ) # output_size = 1 | RF 43

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            # nn.BatchNorm2d(10),
            # nn.ReLU(),
            # nn.Dropout(dropout_value)
        ) #RF 43


        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.gap(x)
        x = self.convblock8(x)

        x = x.view(-1, 10)
# Summary 
<img width="595" alt="Screenshot 2023-07-24 at 5 53 21 AM" src="https://github.com/katipallyvig8899/ERA_6/assets/45558037/15cf9e4b-dd52-46ff-87b0-c198e0b7e6d9">

# Trainning (Epochs=15)
EPOCH: 0
loss=0.07648270577192307 batch_id=468: 100%|██████████| 469/469 [00:19<00:00, 24.34it/s]

Test set: Average loss: 0.0784, Accuracy: 9800/10000 (98.00%)

EPOCH: 1
loss=0.11000124365091324 batch_id=468: 100%|██████████| 469/469 [00:19<00:00, 24.17it/s]

Test set: Average loss: 0.0590, Accuracy: 9831/10000 (98.31%)

EPOCH: 2
loss=0.014273762702941895 batch_id=468: 100%|██████████| 469/469 [00:20<00:00, 23.45it/s]

Test set: Average loss: 0.0304, Accuracy: 9915/10000 (99.15%)

EPOCH: 3
loss=0.07579246163368225 batch_id=468: 100%|██████████| 469/469 [00:21<00:00, 22.25it/s]

Test set: Average loss: 0.0291, Accuracy: 9910/10000 (99.10%)

EPOCH: 4
loss=0.0054457527585327625 batch_id=468: 100%|██████████| 469/469 [00:19<00:00, 23.70it/s]

Test set: Average loss: 0.0246, Accuracy: 9923/10000 (99.23%)

EPOCH: 5
loss=0.022834597155451775 batch_id=468: 100%|██████████| 469/469 [00:19<00:00, 24.44it/s]

Test set: Average loss: 0.0289, Accuracy: 9907/10000 (99.07%)

EPOCH: 6
loss=0.020172329619526863 batch_id=468: 100%|██████████| 469/469 [00:19<00:00, 23.92it/s]

Test set: Average loss: 0.0212, Accuracy: 9935/10000 (99.35%)

EPOCH: 7
loss=0.025972142815589905 batch_id=468: 100%|██████████| 469/469 [00:20<00:00, 23.06it/s]

Test set: Average loss: 0.0204, Accuracy: 9937/10000 (99.37%)

EPOCH: 8
loss=0.0317254438996315 batch_id=468: 100%|██████████| 469/469 [00:20<00:00, 23.31it/s]

Test set: Average loss: 0.0202, Accuracy: 9931/10000 (99.31%)

EPOCH: 9
loss=0.038027323782444 batch_id=468: 100%|██████████| 469/469 [00:20<00:00, 22.36it/s]

Test set: Average loss: 0.0206, Accuracy: 9931/10000 (99.31%)

EPOCH: 10
loss=0.006522379349917173 batch_id=468: 100%|██████████| 469/469 [00:19<00:00, 24.28it/s]

Test set: Average loss: 0.0203, Accuracy: 9935/10000 (99.35%)

EPOCH: 11
loss=0.04078394174575806 batch_id=468: 100%|██████████| 469/469 [00:19<00:00, 23.69it/s]

Test set: Average loss: 0.0203, Accuracy: 9930/10000 (99.30%)

EPOCH: 12
loss=0.04271587356925011 batch_id=468: 100%|██████████| 469/469 [00:20<00:00, 23.17it/s]

Test set: Average loss: 0.0197, Accuracy: 9933/10000 (99.33%)

EPOCH: 13
loss=0.052093639969825745 batch_id=468: 100%|██████████| 469/469 [00:20<00:00, 23.11it/s]

Test set: Average loss: 0.0205, Accuracy: 9928/10000 (99.28%)

EPOCH: 14
loss=0.03014208935201168 batch_id=468: 100%|██████████| 469/469 [00:20<00:00, 23.11it/s]

Test set: Average loss: 0.0189, Accuracy: 9938/10000 (99.38%)


We created the network like squeeze and expand (transition layer in between) to reduce the parameters and increase the performance. We trained the our model with a 15 epochs, we have not reached 99.4  




