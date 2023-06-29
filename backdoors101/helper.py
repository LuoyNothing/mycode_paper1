import importlib
import logging
import os
import random
from collections import defaultdict
from copy import deepcopy
from shutil import copyfile
from typing import Union

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from attack import Attack
from synthesizers.synthesizer import Synthesizer
from tasks.fl.fl_task import FederatedLearningTask
from tasks.task import Task
from utils.parameters import Params
from utils.utils import create_logger, create_table

logger = logging.getLogger('logger')


class Helper:
    params: Params = None
    task: Union[Task, FederatedLearningTask] = None
    synthesizer: Synthesizer = None
    attack: Attack = None
    tb_writer: SummaryWriter = None

    def __init__(self, params):
        self.params = Params(**params)

        self.times = {'backward': list(), 'forward': list(), 'step': list(),
                      'scales': list(), 'total': list(), 'poison': list()}
        if self.params.random_seed is not None:
            self.fix_random(self.params.random_seed)

        self.make_folders()
        self.make_task()
        self.make_synthesizer()
        self.attack = Attack(self.params, self.synthesizer)

        if 'neural_cleanse' in self.params.loss_tasks:
            self.nc = True
        # if 'spectral_evasion' in self.params.loss_tasks:
        #     self.attack.fixed_model = deepcopy(self.task.model)

        self.best_acc = float(0)

    # 1.判断使用的的数据集以及需要调用数据集的位置；2.通过获取module的名字来引入这个类以及其内部的一些方法。
    # 3.self.task通过默认构造函数以及helper的params获取task
    def make_task(self):                                                        # 通过参数task来锁定模块中的类，并创建一个对象实例
        name_lower = self.params.task.lower()                                   # lower() 方法转换字符串中所有大写字符为小写。这里为name_lower = cifarfed
        name_cap = self.params.task                                             # name_cap = CifarFed
        if self.params.fl:                                                      # 如果是FL任务
            module_name = f'tasks.fl.{name_lower}_task'                         # module_name = tasks.fl.cifarfed_task
            path = f'tasks/fl/{name_lower}_task.py'                             # path = tasks/fl/cifarfed_task.py
        else:
            module_name = f'tasks.{name_lower}_task'                            # module_name = tasks.mnist_task，表示模块名字
            path = f'tasks/{name_lower}_task.py'                                # path = tasks/mnist_task.py
        try:
            task_module = importlib.import_module(module_name)                  # task_module = （动态地获取另一个py文件中定义好的变量/方法）tasks.fl.cifarfed_task
            task_class = getattr(task_module, f'{name_cap}Task')                # task_class = （从task_module中获取MNISTTask的属性值）是个CifarFedTask类
        except (ModuleNotFoundError, AttributeError):
            raise ModuleNotFoundError(f'Your task: {self.params.task} should '
                                      f'be defined as a class '
                                      f'{name_cap}'
                                      f'Task in {path}')
        self.task = task_class(self.params)                                     # self.task = CifarFedTask(参数)

    def make_synthesizer(self):                                                 # 根据参数锁定合成器对象，并返回实例
        name_lower = self.params.synthesizer.lower()                            # name_lower = pattern
        name_cap = self.params.synthesizer                                      # name_cap = Pattern
        module_name = f'synthesizers.{name_lower}_synthesizer'                  # module_name = synthesizers.pattern_synthesizer
        try:
            synthesizer_module = importlib.import_module(module_name)           # synthesizer_module = synthesizers.pattern_synthesizer
            task_class = getattr(synthesizer_module, f'{name_cap}Synthesizer')  # task_class = 是一个对象PatternSynthesizer
        except (ModuleNotFoundError, AttributeError):
            raise ModuleNotFoundError(
                f'The synthesizer: {self.params.synthesizer}'
                f' should be defined as a class '
                f'{name_cap}Synthesizer in '
                f'synthesizers/{name_lower}_synthesizer.py')
        self.synthesizer = task_class(self.task)                                # 返回一个PatternSynthesizer对象实例

    # 1.根据params的log是否为True判断是否要在params_folder_path创建文件夹；2.文件夹中还包含run.html，一些画图内容包含在里面；
    # 创建日志信息   3.还会根据tb是否为True来判断是否要使用Tensorboard作图。
    def make_folders(self):
        log = create_logger()                                                   # 创建日志环境，日志的基本设置
        if self.params.log:
            try:
                os.mkdir(self.params.folder_path)                               # saved_models/model_  os.mkdir() 方法用于以数字权限模式创建目录（单级目录），默认的模式为 0777 (八进制)。
            except FileExistsError:
                log.info('Folder already exists')
            with open('saved_models/runs.html', 'a') as f:
                f.writelines([f'<div><a href="https://github.com/ebagdasa/'
                              f'backdoors/tree/{self.params.commit}">GitHub'
                              f'</a>, <span> <a href="http://gpu/'
                              f'{self.params.folder_path}">{self.params.name}_'
                              f'{self.params.current_time}</a></div>'])

            fh = logging.FileHandler(filename=f'{self.params.folder_path}/log.txt')                                     # 将日志发送到磁盘，默认无限增长
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')                       # 输出格式
            fh.setFormatter(formatter)
            log.addHandler(fh)
            log.warning(f'Logging to: {self.params.folder_path}')
            log.error(
                f'LINK: <a href="https://github.com/ebagdasa/backdoors/tree/'
                f'{self.params.commit}">https://github.com/ebagdasa/backdoors'
                f'/tree/{self.params.commit}</a>')

            with open(f'{self.params.folder_path}/params.yaml.txt', 'w') as f:
                yaml.dump(self.params, f)                                                                               # yaml.dump()函数，就是将yaml文件一次性全部写入你创建的文件。
        if self.params.tb:
            wr = SummaryWriter(log_dir=f'runs/{self.params.name}')
            self.tb_writer = wr
            params_dict = self.params.to_dict()
            table = create_table(params_dict)
            self.tb_writer.add_text('Model Params', table)
    # 保存模型√
    def save_model(self, model=None, epoch=0, val_acc=0):
        if self.params.save_model:
            logger.info(f"Saving model to {self.params.folder_path}.")                                      # saved_models/model_{self.task}_{self.current_time}_{self.name}
            # model_name：构建的最终模型文件名，包含文件夹路径和文件名后缀。
            model_name = '{0}/model_last.pt.tar'.format(self.params.folder_path)                                        # 文件名,{0}是一个占位符，将在字符串中的该位置上被 self.params.folder_path 的值替换。
            saved_dict = {'state_dict': model.state_dict(),                                                             # 需保存的模型相关参数
                          'epoch': epoch,
                          'lr': self.params.lr,
                          'params_dict': self.params.to_dict()}
            self.save_checkpoint(saved_dict, False, model_name)                                                         # 保存断点
            if epoch in self.params.save_on_epochs:                                                                     # 规定几轮保存一次
                logger.info(f'Saving model on epoch {epoch}')
                self.save_checkpoint(saved_dict, False, filename=f'{model_name}.epoch_{epoch}')
            if val_acc >= self.best_acc:                                                                                # 如果性能最好，复制一份最好的
                self.save_checkpoint(saved_dict, False, f'{model_name}.best')
                self.best_acc = val_acc
    # 保存断点
    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        if not self.params.save_model:
            return False
        torch.save(state, filename)                                                                                     # 保存模型
        if is_best:
            copyfile(filename, 'model_best.pth.tar')                                                                    # 如果是最优的，就复制到model_best.pth.tar

    def flush_writer(self):
        if self.tb_writer:
            self.tb_writer.flush()

    def plot(self, x, y, name):
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(tag=name, scalar_value=y, global_step=x)
            self.flush_writer()
        else:
            return False

    def report_training_losses_scales(self, batch_id, epoch):
        if not self.params.report_train_loss or batch_id % self.params.log_interval != 0:                               # log_interval记录时间间隔
            return
        total_batches = len(self.task.train_loader)
        losses = [f'{x}: {np.mean(y):.2f}' for x, y in self.params.running_losses.items()]
        scales = [f'{x}: {np.mean(y):.2f}' for x, y in self.params.running_scales.items()]
        logger.info(f'Epoch: {epoch:3d}. Batch: {batch_id:5d}/{total_batches}.  Losses: {losses}. Scales: {scales}')
        for name, values in self.params.running_losses.items():
            self.plot(epoch * total_batches + batch_id, np.mean(values), f'Train/Loss_{name}')
        for name, values in self.params.running_scales.items():
            self.plot(epoch * total_batches + batch_id, np.mean(values), f'Train/Scale_{name}')
        self.params.running_losses = defaultdict(list)                                                                  # 这里表示清空数据吧
        self.params.running_scales = defaultdict(list)

    @staticmethod
    def fix_random(seed=1):
        from torch.backends import cudnn

        logger.warning('Setting random_seed seed for reproducible results.')
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = False
        cudnn.enabled = True
        cudnn.benchmark = True
        np.random.seed(seed)

        return True
