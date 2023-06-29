import random

import torch
from torchvision.transforms import transforms, functional

from synthesizers.synthesizer import Synthesizer
from tasks.task import Task

transform_to_image = transforms.ToPILImage()
transform_to_tensor = transforms.ToTensor()

# 可以理解为给图片加一个小块，位置和小块内容自己定（随机矩阵）
class PatternSynthesizer(Synthesizer):
    pattern_tensor: torch.Tensor = torch.tensor([                           # shape = (5,3)
        [1., 0., 1.],
        [-10., 1., -10.],
        [-10., -10., 0.],
        [-10., 1., -10.],
        [1., 0., 1.]
    ])
    "Just some random 2D pattern."

    x_top = 3
    "X coordinate to put the backdoor into."
    y_top = 23
    "Y coordinate to put the backdoor into."

    mask_value = -10
    "A tensor coordinate with this value won't be applied to the image."

    resize_scale = (5, 10)
    "If the pattern is dynamically placed, resize the pattern."

    mask: torch.Tensor = None
    "A mask used to combine backdoor pattern with the original image."

    pattern: torch.Tensor = None
    "A tensor of the `input.shape` filled with `mask_value` except backdoor."

    def __init__(self, task: Task):
        super().__init__(task)
        self.make_pattern(self.pattern_tensor, self.x_top, self.y_top)

    # 加小块，可以定义位置和内容
    def make_pattern(self, pattern_tensor, x_top, y_top):
        full_image = torch.zeros(self.params.input_shape)                               # 生成一个5w*32*32的0张量
        full_image.fill_(self.mask_value)                                               # b.fill_(-10)就表示用-10填充b，是in_place操作
        x_bot = x_top + pattern_tensor.shape[0]                                         # x_bot = x_top(3) + 5
        y_bot = y_top + pattern_tensor.shape[1]                                         # y_bot = y_top(23) + 3
        if x_bot >= self.params.input_shape[1] or y_bot >= self.params.input_shape[2]:  # 后门不可大于图片尺寸
            raise ValueError(f'Position of backdoor outside image limits:image: {self.params.input_shape}, but backdoor ends at ({x_bot}, {y_bot})')
        full_image[:, x_top:x_bot, y_top:y_bot] = pattern_tensor                        # full_image[:, 3:8, 23:26]
        self.mask = 1 * (full_image != self.mask_value).to(self.params.device)          # self.mask应该是等于1的
        self.pattern = self.task.normalize(full_image).to(self.params.device)           # pattern加了个mask，还加了个小矩阵

    # 在图片中加入pattern
    def synthesize_inputs(self, batch, attack_portion=None):
        pattern, mask = self.get_pattern()                                                                              # 在attack的portion上加入pattern
        batch.inputs[:attack_portion] = (1 - mask) * batch.inputs[:attack_portion] + mask * pattern
        return

    def synthesize_labels(self, batch, attack_portion=None):
        batch.labels[:attack_portion].fill_(self.params.backdoor_label)                                                 # 在attack的portion上加入backdoor_label，这里backdoor_label是8
        return

    # 生成随机的pattern内容和位置
    def get_pattern(self):
        # pattern位置随机，内容大小随机
        if self.params.backdoor_dynamic_position:
            resize = random.randint(self.resize_scale[0], self.resize_scale[1])
            pattern = self.pattern_tensor
            if random.random() > 0.5:
                pattern = functional.hflip(pattern)
            image = transform_to_image(pattern)
            pattern = transform_to_tensor(functional.resize(image, resize, interpolation=0)).squeeze()

            x = random.randint(0, self.params.input_shape[1] - pattern.shape[0] - 1)
            y = random.randint(0, self.params.input_shape[2] - pattern.shape[1] - 1)
            self.make_pattern(pattern, x, y)

        return self.pattern, self.mask
