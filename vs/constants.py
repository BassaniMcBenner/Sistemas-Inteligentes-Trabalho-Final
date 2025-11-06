""" CONSTANTS
    @Author: Tacla (UTFPR)
    It has the constants used by the simulator and by the user of the simulator
"""


class VS:
    # States of the body of the agent
    ENDED = 2   # Successfully ended
    ACTIVE = 1  # still running
    IDLE = 0    # not active, but alive
    DEAD = -1   # fatal error

    # Possible results for the walk action
    BUMPED = -1         # agent bumped into a wall or the end of the grid
    TIME_EXCEEDED = -2  # agent reached the time limit - no more battery
    EXECUTED = 1        # action successfully executed

    # Possible result for check_victims
    NO_VICTIM = -1

    # Possible results for the check_obstacles method
    UNK = -1           # when the agent ignores (unknow)
    CLEAR = 0
    WALL = 1
    END = 2

    # Value for representing obstacles in the 2D grid (environment)
    OBST_WALL = 100  # a wall in the grid
    OBST_NONE = 1    # a position without obstacles
    
    # Colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GRAY = (128, 128, 128)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    RED = (255, 0, 0)
    CYAN = (0, 255, 255)
    YELLOW = (255, 255, 0)
    PINK = (255, 105, 180)

    # Specific colors for victim by injury severity -
    # SEV BLK is the most severe; SEV GRN is the least severe
    VIC_COLOR_BLK = (64, 64, 64)   # BLACK
    VIC_COLOR_RED = (255, 0, 0)     # RED
    VIC_COLOR_YEL = (255, 255, 0)   # YELLOW
    VIC_COLOR_GRN = (0, 255, 0)     # GREEN
    VIC_COLOR_LIST = [VIC_COLOR_GRN, VIC_COLOR_YEL, VIC_COLOR_RED, VIC_COLOR_BLK]
