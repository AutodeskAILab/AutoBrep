from enum import IntEnum

class MMTokenIndex(IntEnum):
    # Begin / End of Data Sequence
    BOS = 0
    EOS = 1
    # Begin / End of Text Sentence
    BOT = 2
    EOT = 3
    # Begin / End of a CAD B-Rep
    BOC = 4
    EOC = 5
    # Begin / End of a Level
    BOL = 6
    EOL = 7
    # Begin / End of a Face
    BOF = 8
    EOF = 9
    # Begin / End of Geometry Prompt
    BOGEOM = 10
    EOGEOM = 11
    # Begin / End of Meta
    BOM = 12
    EOM = 13
    # Complexity
    GEN_EASY = 14
    GEN_MID = 15
    GEN_HARD = 16
    GEN_UNCOND = 17
    # Begin / End of Point Cloud
    BOPC = 18
    EOPC = 19
    # Dummy ID for autocomplete
    DUMMYID = 20
