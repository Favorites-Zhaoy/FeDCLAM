from .fedclam import Config as FedCLAMConfig


class Config(FedCLAMConfig):
    def __init__(self, exp_name='fedclam_resenc_ag'):
        super(Config, self).__init__(exp_name)
        self.NETWORK = 'resattunet'
        self.NETWORK_PARAMS.update({'attention': 'gate', 'deep_supervision': False})
        self.DEEP_SUPERVISION = False
        self.DEEP_SUPERVISION_WEIGHTS = [1.0, 0.5, 0.25]
