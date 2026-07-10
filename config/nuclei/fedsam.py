from .base import BaseConfig


class Config(BaseConfig):
    def __init__(self, exp_name='fedsam'):
        super(Config, self).__init__(exp_name)
        self.CLIENT = 'FedSAMClient'
        self.COMM_TYPE = 'FedAvg'
