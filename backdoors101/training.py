# 1.通过参数来构成helper,从而搭建整个框架
# 2.run才是运行整个框架
'''
1.模型保存与加载
    1.保存模型：helper.py——>save_model
    2.最开始加载模型：cifar10_task.py——> build_model
    3.从30轮继续训练的加载模型：task.py——> resume_model
'''

# 读取命令行参数
import argparse
# 对目录和文件提供了复制、移动、删除、压缩、解压等操作
import shutil
# 获取时间
from datetime import datetime
# yaml是一个专门用来写配置文件的语言
import yaml
# prompt_toolkit 用于打造交互式命令行
from prompt_toolkit import prompt
# Tqdm 是一个快速，可扩展的Python进度条，可以在 Python 长循环中添加一个进度提示信息，用户只需要封装任意的迭代器 tqdm(iterator)
from tqdm import tqdm

# noinspection PyUnresolvedReferences
# 数据集处理
from dataset.pipa import Annotations  # legacy to correctly load dataset.
from helper import Helper
from utils.utils import *
# logging模块是Python内置的标准模块，主要用于输出运行日志，可以设置输出日志的等级、日志保存路径、日志文件回滚等
logger = logging.getLogger('logger')

# 训练方法
def train(hlpr: Helper, epoch, model, optimizer, train_loader, attack=True):
    criterion = hlpr.task.criterion                                                 # criterion指的是损失函数，这里返回交叉熵损失
    model.train()                                                                   # model.train() 让你的模型知道现在正在训练。像 dropout、batchnorm 层在训练和测试时的作用不同，所以需要使它们运行在对应的模式中。
    for i, data in enumerate(train_loader):
        batch = hlpr.task.get_batch(i, data)                                        # 获得当前的batch，把数据加载到设备上,
        model.zero_grad()                                                           # 把梯度设置成0，在计算反向传播的时候一般都会这么操作，为了防止梯度积累
        # 主要进行攻击的代码
        loss = hlpr.attack.compute_blind_loss(model, criterion, batch, attack)      # 这里表示训练时期加上攻击,attack.py，这里涉及多任务权重平衡
        loss.backward()                                                             # 计算梯度，这里别管，就是要这么写；下面是计算权值
        optimizer.step()

        hlpr.report_training_losses_scales(i, epoch)                                # 打印训练时期的精度和损失等info信息，helper.py
        if i == hlpr.params.max_batch_id:
            break
    return


def test(hlpr: Helper, epoch, backdoor=False):
    model = hlpr.task.model
    # 好像是提醒状态的，表示测试阶段
    model.eval()
    hlpr.task.reset_metrics()                                                       # 重置评估指标

    with torch.no_grad():                                                           # with torch.no_grad：所有计算得出的tensor的requires_grad都自动设置为False。
        for i, data in tqdm(enumerate(hlpr.task.test_loader)):

            # 将数据和标签加载到设备上
            batch = hlpr.task.get_batch(i, data)                                                                        # 加载到设备上
            if backdoor:
                batch = hlpr.attack.synthesizer.make_backdoor_batch(batch, test=True, attack=True)                      # 注入后门
            outputs = model(batch.inputs)                                                                               # 预测结果，输出为[-0.0113, -0.0098, -0.0933,..]十项
            hlpr.task.accumulate_metrics(outputs=outputs, labels=batch.labels)                                          # 计算预测和真实标签的指标(精度和损失),task.py

    metric = hlpr.task.report_metrics(epoch, prefix=f'Backdoor {str(backdoor):5s}. Epoch: ',                            # 打印的黄色提示，task.py
                             tb_writer=hlpr.tb_writer,
                             tb_prefix=f'Test_backdoor_{str(backdoor):5s}')

    return metric


def run(hlpr):
    acc = test(hlpr, 0, backdoor=False)
    for epoch in range(hlpr.params.start_epoch,
                       hlpr.params.epochs + 1):
        train(hlpr, epoch, hlpr.task.model, hlpr.task.optimizer,
              hlpr.task.train_loader)
        acc = test(hlpr, epoch, backdoor=False)
        test(hlpr, epoch, backdoor=True)
        hlpr.save_model(hlpr.task.model, epoch, acc)
        if hlpr.task.scheduler is not None:
            hlpr.task.scheduler.step(epoch)

def fl_run(hlpr: Helper):
    print("进入FL训练")
    for epoch in range(hlpr.params.start_epoch, hlpr.params.epochs + 1):                # start_epoch=1开始
        run_fl_round(hlpr, epoch)                                                       # 联邦学习训练，更新一次全局模型
        metric = test(hlpr, epoch, backdoor=False)                                      # 评估指标--干净数据的精度
        test(hlpr, epoch, backdoor=True)                                                # 测试--后门精度
        hlpr.save_model(hlpr.task.model, epoch, metric)                                 # 保存模型

def run_fl_round(hlpr, epoch):
    global_model = hlpr.task.model                                                      # 获得global的模型：global_model 在fl_task.py
    print(f"epoch:{epoch} 开始训练")
    local_model = hlpr.task.local_model                                                 # 获得local的模型;local_model在fl_task.py
    round_participants = hlpr.task.sample_users_for_round(epoch)                        # 选择客户端对象，选择恶意客户端和正常客户端，fl_task.py
    print("随机选择的10个客户端：", end=" ")
    for user in round_participants:
        if user.compromised:
            print(f'({user.user_id})', end=' ')
            continue
        print(user.user_id, end=' ')
    print()
    weight_accumulator = hlpr.task.get_empty_accumulator()                              # 一个空的权重参数字典

    # tqdm是python的进度条库，基本是基于对象迭代
    for user in tqdm(round_participants):                                               # 对于每个客户端进行训练
        hlpr.task.copy_params(global_model, local_model)                                # 1.将参数从global_model复制到local_model，来进行本地模型更新
        optimizer = hlpr.task.make_optimizer(local_model)                               # 选择优化器SGD
        for local_epoch in range(hlpr.params.fl_local_epochs):                          # 开始本地训练
            if user.compromised:                                                        # 如果是恶意的用户，则执行进攻的训练
                train(hlpr, local_epoch, local_model, optimizer, user.train_loader, attack=True)
            else:                                                                       # 如果是非恶意的用户，则执行非进攻的训练
                train(hlpr, local_epoch, local_model, optimizer, user.train_loader, attack=False)
        local_update = hlpr.task.get_fl_update(local_model, global_model)               # 这是本地模型-全局模型的变化，为需要上传的本地更新
        if user.compromised:                                                            # 如果用户是恶意用户，还会*一个提升因子
            hlpr.attack.fl_scale_update(local_update)
        hlpr.task.accumulate_weights(weight_accumulator, local_update)                  # 把所有选中客户端的更新加起来。其中进行了DP裁剪

    hlpr.task.update_global_model(weight_accumulator, global_model)                     # 所有用户完成之后，更新全局的模型

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Backdoors')
    parser.add_argument('--params', dest='params', default='utils/params.yaml')
    parser.add_argument('--name', dest='name', required=True)
    parser.add_argument('--commit', dest='commit', default=get_current_git_hash())

    args = parser.parse_args()

    with open(args.params) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    params['current_time'] = datetime.now().strftime('%b.%d_%H.%M.%S')
    params['commit'] = args.commit
    params['name'] = args.name

    helper = Helper(params)
    logger.warning(create_table(params))

    try:
        if helper.params.fl:
            fl_run(helper)

        else:
            run(helper)
    except (KeyboardInterrupt):
        if helper.params.log:
            answer = prompt('\nDelete the repo? (y/n): ')
            if answer in ['Y', 'y', 'yes']:
                logger.error(f"Fine. Deleted: {helper.params.folder_path}")
                shutil.rmtree(helper.params.folder_path)
                if helper.params.tb:
                    shutil.rmtree(f'runs/{args.name}')
            else:
                logger.error(f"Aborted training. "
                             f"Results: {helper.params.folder_path}. "
                             f"TB graph: {args.name}")
        else:
            logger.error(f"Aborted training. No output generated.")
