# DO NOT MODIFY!!!

from dataclasses import dataclass
from math import sqrt

@dataclass
class Tunables:
    MU_ROUND: float    = 4.5 # 6.9  # 4.5
    SIGMA_ROUND: float = 30 # 20.0
    UCB_C: float = sqrt(2) #3.2 #5000 # 2000 to 10000
    UCB_DYN_C: bool = False
    UCB_DYN_C_K: float     = 0.28
    UCB_DYN_C_CAP: int     = 99999 # idc
    SMOOTH_GAMMA = 0.16 # lowerbound
    SMOOTH_ETA   = 0.9  # numerator scale
    SMOOTH_D     = 0.0015 # idk what is this
    LOSE_PENALTY: float = 2.1 # 2.0 VERY COOL
    MAX_RAISES_PER_NODE: int = 10
    ALLOW_RAISE_MAX: bool = False
    RAISE_GAP: str = 'linear'
    UCT_SMOOTH: bool = True
    #RAISE_LIMITER: float = 2/3
    USE_PROB_AGENT: bool = True
    MONTE_CARLO: bool = True

    VERBOSE: bool = False
