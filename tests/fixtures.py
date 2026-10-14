from pathlib import Path

from goethe.hw import Gpu

REPO = Path(__file__).resolve().parent.parent


def gpu(name="NVIDIA GeForce RTX 5090", total_gib=31.4, cc="12.0", index=0, free_gib=None, driver="580.178.04",
        width=16, width_max=16):
    return Gpu(index, name, int(total_gib * 1024), int((free_gib if free_gib is not None else total_gib) * 1024),
               driver, cc, width, width_max)


RTX_5090 = gpu()
PRO_6000 = gpu("NVIDIA RTX PRO 6000 Blackwell Workstation Edition", 95.0)
PRO_4000 = gpu("NVIDIA RTX PRO 4000 Blackwell", 23.5)
RTX_3090 = gpu("NVIDIA GeForce RTX 3090", 24.0, cc="8.6")
