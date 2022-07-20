import matplotlib.pyplot as plt


def plot_prediction(known,
                    unknown,
                    pred,
                    bg_pred=None,
                    label='MT-GPRMachine',
                    msize=5,
                    facecolor='#EAEEF9',
                    ax=None,
                    bg_markers=None,
                    bg_labels=None):
    """ plot the prediction with the truth """
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 5))

    train_len = len(known)
    test_len = len(unknown)
    tot_len = train_len + test_len

    ax.axvspan(-.5, train_len - .5, facecolor=facecolor, zorder=-1)

    if bg_pred is not None:
        lines = []
        for i, m in enumerate(bg_markers):
            l = ax.plot(range(train_len, tot_len), bg_pred[:, i], marker=m, c='.8',
                        alpha=1, ms=msize, label=bg_labels[i])
            lines.append(l)

    ax.plot(range(train_len), known, marker='o', c='#1D21FB',
            ms=msize, label='Known')
    ax.plot(range(train_len, tot_len), unknown, marker='o',
            c='#00F7F0', ms=msize, label='Unknown')
    ax.plot(range(train_len, tot_len), pred, marker='d',
            c='#F80202', ms=msize * 1.2,
            # markerfacecolor='none',
            label=label)
    return ax
