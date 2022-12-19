
from algorithms.classes.ERM import ERM
# from algorithms.classes.IRM import IRM
#from algorithms.classes.DoYoJo import DoYoJoVAE as DoYoJo
# from algorithms.classes.Fish import Fish
# from algorithms.classes.VREx import VREx
# from algorithms.classes.Mixup import Mixup
# from algorithms.classes.GroupDRO import GroupDRO
# from algorithms.classes.MLDG import MLDG
# from algorithms.classes.AbstractMMD import MMD, CORAL
# from algorithms.classes.MTL import MTL
# from algorithms.classes.SagNet import SagNet
# from algorithms.classes.RSC import RSC
# from algorithms.classes.SD import SD
# from algorithms.classes.ANDMask import ANDMask
# from algorithms.classes.IGA import IGA
# from algorithms.classes.SelfReg import SelfReg
# from algorithms.classes.SANDMask import SANDMask
# from algorithms.classes.Fishr import Fishr
# from algorithms.classes.IB_ERM import IB_ERM
# from algorithms.classes.IB_IRM import IB_IRM
# from algorithms.classes.AbstractCAD import CAD, CondCAD
# from algorithms.classes.Transfer import Transfer
# from algorithms.classes.AbstractCausIRL import CausIRL_MMD,  CausIRL_CORAL
# from algorithms.classes.SCIRM import SCIRM

ALGORITHMS = [
    'ANDMask',
    'CausIRL_MMD',
    'CausIRL_CORAL',
    'CORAL',
    'CAD',
    'CondCAD',
    'DIFEX',
    'ERM',
    'Fish',
    'Fishr',
    'GroupDRO',
    'IB_ERM',
    'IB_IRM',
    'IRM',
    'IGA',
    'Mixup',
    'MMD',
    'MLDG',
    'MTL',
    'RSC',
    'SagNet',
    'SANDMask',
    'SelfReg',
    'SD',
    'Transfer',
    'VREx',
]


def get_subalgorithm_class(algorithm_name):
    if algorithm_name not in globals():
        raise NotImplementedError(
            "Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]
