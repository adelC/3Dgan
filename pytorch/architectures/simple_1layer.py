import torch
import torch.nn.functional as F

class CNNsimple(torch.nn.Module):

    def __init__(self):
        super(CNNsimple, self).__init__()

        self.conv1 = torch.nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=1) # 25 x 25                                                        
        self.pool = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=1) # 8 x 12 x 12                                                         
        self.fc1 = torch.nn.Linear(8 * 12 * 12, 1)

    def forward(self, x):

        #Computes the activation of the first convolution                                                                                        
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(-1, 8 * 12 * 12)

        #Computes the activation of the first fully connected layer                                                                              
        x = F.relu(self.fc1(x))

        return(x)
