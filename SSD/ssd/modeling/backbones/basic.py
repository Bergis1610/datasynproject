import torch
from typing import Tuple, List
from torch import nn


class BasicModel(torch.nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
    """
    def __init__(self,
            output_channels: List[int],
            image_channels: int,
            output_feature_sizes: List[Tuple[int]]):
        super().__init__()
        self.out_channels = output_channels
        self.output_feature_shape = output_feature_sizes

        self.module_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, 
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1, 
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=[2,2], stride=1),
            nn.Conv2d(
                in_channels=16, 
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1, 
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=16, 
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1, 
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=16, 
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1, 
            ),
            nn.ReLU(),



            nn.Conv2d(
                in_channels=32, 
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1, 
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, 
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1, 
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, 
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1, 
            ),
            nn.ReLU(),


            

            nn.Conv2d(
                in_channels=64, 
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1, 
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, 
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1, 
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, 
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1, 
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128, 
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1, 
            ),
            nn.ReLU(),
        )

        

        self.module_2 = nn.Sequential( 
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128, 
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1, 
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128, 
                out_channels=256,
                kernel_size=3,
                stride=2,
                padding=1, 
            ),
            nn.ReLU(),
        )


        self.module_3 = nn.Sequential ( 
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256, 
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1, 
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256, 
                out_channels=256,
                kernel_size=3,
                stride=2,
                padding=1, 
            ),
            nn.ReLU()
        )


        self.module_4 = nn.Sequential ( 
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256, 
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1, 
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128, 
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1, 
            ),
            nn.ReLU()
        )

        self.module_5 = nn.Sequential ( 
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128, 
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1, 
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128, 
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1, 
            ),
            nn.ReLU()
        )

        self.module_6 = nn.Sequential ( 
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, 
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1, 
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128, 
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=0, 
            ),
            nn.ReLU()
        )
        
        self.modules = [self.module_1, self.module_2, self.module_3, self.module_4, self.module_5, self.module_6]

         

    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """
        

        out_features = []
        var = x
        for module in self.modules: 
            var = module(var)
            out_features.append(var)

        for idx, feature in enumerate(out_features):
            out_channel = self.out_channels[idx]
            h, w = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        assert len(out_features) == len(self.output_feature_shape),\
            f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"
        return tuple(out_features)

