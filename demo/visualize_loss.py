__author__ = "Jie Lei"
from tqdm import tqdm
import matplotlib.pyplot as plt


def parse_log(logfile):
    with open(logfile, "r") as f:
        lines = [l.strip("\n") for l in f.readlines()]
    loss_lines = [l for l in lines if "loss_classifier" in l]
    info_dict = dict(
        iter=[],
        loss=[],
        loss_box_reg=[],
        loss_rpn_box_reg=[],
        loss_classifier=[],
        loss_objectness=[],
        lr=[]
    )
    for l in tqdm(loss_lines):
        words = l.split()
        for k in info_dict.keys():
            convert_dtype = float if k != "iter" else int
            idx = words.index(k+":")
            info_dict[k].append(convert_dtype(words[idx+1]))
    return info_dict


def plot_curve(x, y, xlabel="", ylabel="", title=""):
    """
    Assuming already imported:
    import matplotlib.pyplot as plt
    %matplotlib inline

    """
    fig = plt.figure()
    fig.suptitle(title, fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.plot(x, y)
    plt.show()

