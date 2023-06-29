from tasks.batch import Batch
from tasks.task import Task
from utils.parameters import Params


class Synthesizer:
    params: Params
    task: Task

    def __init__(self, task: Task):
        self.task = task
        self.params = task.params

    # 实施后门以及注入后门的动作。1.确定攻击比例；2.确定后门批次并注入后门攻击
    def make_backdoor_batch(self, batch: Batch, test=False, attack=True) -> Batch:
        # Don't attack if only normal loss task.
        if (not attack) or (self.params.loss_tasks == ['normal'] and not test):
            return batch
        if test:
            attack_portion = batch.batch_size
        else:
            attack_portion = round(batch.batch_size * self.params.poisoning_proportion)     # attack_portion = 64 * 0.5 = 32攻击的位置来源于从数据集中随机取样，round() 方法返回浮点数x的四舍五入值
        backdoored_batch = batch.clone()
        # 传入batch和portion
        self.apply_backdoor(backdoored_batch, attack_portion)
        return backdoored_batch

    def apply_backdoor(self, batch, attack_portion):
        """
        Modifies only a portion of the batch (represents batch poisoning).

        :param batch:
        :return:
        """
        self.synthesize_inputs(batch=batch, attack_portion=attack_portion)
        self.synthesize_labels(batch=batch, attack_portion=attack_portion)

        return

    def synthesize_inputs(self, batch, attack_portion=None):
        raise NotImplemented

    def synthesize_labels(self, batch, attack_portion=None):
        raise NotImplemented
