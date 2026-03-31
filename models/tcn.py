import torch
import torch.nn as nn

class SimpleTemporalModel(nn.Module):
    def __init__(self, num_joints_in, in_features, output_dim, filter_widths, dropout=0.25, channels=1024, task='classification'):
        """
        A simplified temporal model with 1D convolutions and adaptive average pooling.
        
        Arguments:
        num_joints_in -- number of input joints (e.g., 17)
        in_features -- number of input features per joint (e.g., 3 for x, y, confidence or 2 for x, y)
        output_dim -- dimensionality of the output (e.g., num_classes for classification or 17 for regression)
        filter_widths -- list of filter widths for each convolutional layer
        dropout -- dropout probability
        channels -- number of channels for intermediate convolutional layers
        """
        super(SimpleTemporalModel, self).__init__()
        
        # Initialize attributes
        self.num_joints_in = num_joints_in
        self.in_features = in_features
        self.output_dim = output_dim
        self.filter_widths = filter_widths
        self.task = task
        
        # Dropout and ReLU layers
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
        
        # Initial expansion layer
        self.expand_conv = nn.Conv1d(num_joints_in * in_features, channels, filter_widths[0], padding=(filter_widths[0] // 2), bias=False)
        self.expand_bn = nn.BatchNorm1d(channels, momentum=0.1)
        
        # Intermediate convolutional layers
        layers_conv = []
        layers_bn = []
        
        for fw in filter_widths[1:]:
            layers_conv.append(nn.Conv1d(channels, channels, fw, padding=(fw // 2), bias=False))  # No strides
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
        
        # Store layers as ModuleLists
        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)
        
        # Adaptive pooling layer to reduce the temporal dimension to 1
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Final shrink layer
        if self.task == 'classification':
            self.shrink = nn.Conv1d(channels, output_dim, 1)
        elif self.task == 'regression':
            self.shrink = nn.Conv1d(channels, num_joints_in * 3, 1)

    def forward(self, x):
        # Reshape input to [batch_size, num_joints_in * in_features, sequence_length]
        x = x.view(x.shape[0], x.shape[1], -1)  # [batch_size, sequence_length, num_joints_in * in_features]
        x = x.permute(0, 2, 1)  # [batch_size, num_joints_in * in_features, sequence_length]
        
        # Expand channels
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))
        
        # Pass through convolutional layers
        for conv, bn in zip(self.layers_conv, self.layers_bn):
            x = self.drop(self.relu(bn(conv(x))))
        
        # Apply adaptive average pooling to reduce temporal dimension to 1
        x = self.pool(x)  # Shape after pooling: [batch_size, channels, 1]
        
        # Apply the final shrink layer to get the desired output shape
        x = self.shrink(x)  # Shape: [batch_size, output_dim, 1] or [batch_size, num_joints_in * 3, 1]
        
        # Final reshaping for classification or regression
        x = x.squeeze(-1)  # Remove the last dimension
        
        if self.task == 'regression':
            x = x.view(x.shape[0], self.num_joints_in, 3)  # Reshape for regression task: [batch_size, num_joints_in, 3]
        
        return x
