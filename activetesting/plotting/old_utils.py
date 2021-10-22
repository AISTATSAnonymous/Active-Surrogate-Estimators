def KolmogorovSmirnov(A, B):
    statistic, p = ks(A, B)

    return statistic, p


def batch_test(a, b, step, which='wc'):
    # a, b: list of lists, each list has [vis, risk, acquisition]
    if which == 'ks':
        test = ks_2samp
    elif which == 'mw':
        test = partial(mannwhitneyu, alternative='less')
    elif which == 'wc':
        test = wilcoxon
        test = partial(wilcoxon, alternative='less')

    else:
        raise ValueError

    for ai in a:
        a_vals = ai[0].get_slice(ai[1], ai[2], step).values
        for bi in b:
            b_vals = bi[0].get_slice(bi[1], bi[2], step).values
            stat, p = test(a_vals, b_vals)
            print(p < 0.05, f'p={p:.3e}', f'stat={stat:.3e}', ai[2], bi[2])


def plot_risks_select_combinations(
            self, acquisition_risks, errors='std',
            fig=None, ax=None, alpha=0.3, i=0, labels=None, lw=None,
            white_bg=True, lw2=None):
    """Plot mean +- std error.

    Of the risk-estimator-acquisition-function combination against the
    true empirical test risk.

    """
    if errors == 'std_error':
        middle = self.means
        sqrtN = np.sqrt(self.n_runs)
        upper_base = middle + self.stds/sqrtN
        lower_base = middle - self.stds/sqrtN
    elif errors == 'std':
        middle = self.means
        upper_base = middle + self.stds
        lower_base = middle - self.stds
    elif errors == 'percentiles':
        middle = self.means
        lower_base, upper_base = self.percentiles
    else:
        raise ValueError(f'Do not recognize errors={errors}.')
    if fig is None or ax is None:
        fig, ax = plt.subplots(dpi=200)

    linestyles = itertools.cycle(['--', '-.', ':'])
    for acquisition, risk in acquisition_risks:
        acq_risk = f'{acquisition}_{risk}'
        color = acquisition_risks_to_color[acq_risk]
        m = middle.loc[acquisition][risk].values
        s_u = upper_base.loc[acquisition][risk].values
        s_l = lower_base.loc[acquisition][risk].values
        x = np.arange(1, s_l.size + 1)

        if white_bg:
            ax.fill_between(
                x, s_u, s_l,
                color='white', alpha=1)


        ax.fill_between(
            x, s_u, s_l,
            color=color, alpha=alpha)
        ax.plot(x, s_l, '--', color=color, zorder=100, lw=lw)
        ax.plot(x, s_u, '--', color=color, zorder=100, lw=lw)
        ax.plot(x, m, color=color,
                label=labels[i], zorder=100, lw=lw2)
        i += 1

    return fig, ax


def plot_equivalent_samples(
            self, acquisition_risks, baseline=None, errors='std',
            fig=None, ax=None, alpha=0.3, i=0, labels=None, zorders=None,
            colors=None, relative=True, rolling_before=False,
            rolling_after=False, inverse=False):
    """Plot mean +- std error.

    Of the risk-estimator-acquisition-function combination against the
    true empirical test risk.

    """
    if errors == 'percentiles':
        upper_base = self.quant_errors
    elif errors == 'std':
        upper_base = self.errors
    else:
        raise ValueError

    if fig is None or ax is None:
        fig, ax = plt.subplots(dpi=200)

    if baseline is None:
        baselines = ['RandomAcquisition', 'BiasedRiskEstimator']

    base_risk = upper_base.loc[baselines[0]][baselines[1]].values
    if zorders is None:
        zorders = 100 * [None]

    linestyles = itertools.cycle(['--', '-.', ':'])
    for acquisition, risk in acquisition_risks:
        acq_risk = f'{acquisition}_{risk}'
        if colors is None:
            color = acquisition_risks_to_color[acq_risk]
        else:
            color = colors[i]

        s_u = upper_base.loc[acquisition][risk].values
        if (R := rolling_before) is not False:
            s_u = np.convolve(
                s_u, np.ones(R)/R, mode='valid')
            base_risk = np.convolve(
                base_risk, np.ones(R)/R, mode='valid')

        diffs = s_u[:, np.newaxis] - base_risk
        diffs[diffs < 0] = 1e10
        idxs = np.argmin(diffs, axis=1) + 1
        x = range(1, len(idxs)+1)
        if relative:
            y = idxs/x
        else:
            y = idxs

        if (R := rolling_after) is not False:
            y = np.convolve(y, np.ones(R)/R, mode='valid')
            x = range(1, len(y)+1)

        if inverse:
            y = 1/y

        ax.plot(y, '-', color=color, label=labels[i],
                zorder=zorders[i])
        i += 1

    return fig, ax


class OOMFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(
            self, useOffset=offset, useMathText=mathText)

    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom
#     def _set_format(self, vmin=None, vmax=None):
#         self.format = self.fformat
#         if self._useMathText:
#             self.format = r'$\mathdefault{%s}$' % self.format


def plot_regression(self, run, fig, ax, legend):

    data = self.datasets[run]
    data['x'] = self.check_remove_feature_dim(data['x'])

    train = data['x'][data['train_idxs']], data['y'][data['train_idxs']]
    test = data['x'][data['test_idxs']], data['y'][data['test_idxs']]

    # also plot model predictions
    filtered = self.results[
        (self.results.run == run)
        & (self.results.acquisition == self.acquisitions[0])]
    y_preds, idx = filtered.y_preds.values, filtered.idx.values
    sorted_idxs = np.argsort(data['x'][idx])

    predicted_p = ax.plot(
        data['x'][idx][sorted_idxs], y_preds[sorted_idxs],
        zorder=-10, linewidth=1, color=pal[2], alpha=0.7,
        label='Model Predictions')

#     predicted_p = ax.scatter(
#         data['x'][idx][sorted_idxs], y_preds[sorted_idxs], s=5, marker='x',
#         zorder=10, linewidth=0.5, color=pal[2], alpha=1, label='Model Pred.')

    idxs = np.argsort(test[0])
    ax.scatter(test[0][idxs], test[1][idxs], label='Test Data', s=0.5,
               color=pal[0])
    ax.scatter(*train, s=3, label='Train Data', color=pal[1])

    if legend:
        ax.legend(fontsize=fs)

    return fig, ax


def plot_class(ax, run, self):

    data = self.datasets[run]

    train = data['x'][data['train_idxs']], data['y'][data['train_idxs']]
    test = data['x'][data['test_idxs']], data['y'][data['test_idxs']]

    # also plot model predictions
    filtered = self.results[
        (self.results.run == run)
        & (self.results.acquisition == self.acquisitions[0])]
    y_preds, idx = filtered.y_preds.values, filtered.idx.values

    cmap = sns.diverging_palette(220, 20, s=255, as_cmap=True)
    ax.scatter(
        data['x'][idx][:, 0],
        data['x'][idx][:, 1],
        c=y_preds,
        label='Model Predictions',
        marker='x',
        s=5,
        cmap=cmap,
        linewidth=0.5,
        zorder=10,
        )

    cmap = sns.diverging_palette(220, 20, s=50, as_cmap=True)
    ax.scatter(
        train[0][:, 0], train[0][:, 1], c=train[1],
        s=0.5, label='Train Data', cmap=cmap, linewidth=0.5)

    return ax


def plot_log_convergence(
            self, acquisition_risks, errors='std',
            fig=None, ax=None, alpha=0.3, i=0, names=None, labels=None,
            rolling=False, zorder=100, colors=None, with_errors=False,
            swapaxes=False, print_it=False, lw=None, error_type='quant',
            scale='default', lwfill=None
):
    """Plot mean +- std error.

    Of the risk-estimator-acquisition-function combination against the
    true empirical test risk.

    """
    if errors == 'percentiles':
        upper_base = self.quant_errors
    elif errors == 'std':
        upper_base = self.errors
    elif errors == 'log mean':
        upper_base = self.log_sq_diff
        if scale != 'manual log':
            raise ValueError('Log target!')
    else:
        raise ValueError

    if fig is None or ax is None:
        fig, ax = plt.subplots(dpi=200)

    linestyles = itertools.cycle(['--', '-.', ':'])
    for acquisition, risk in acquisition_risks:
        acq_risk = f'{acquisition}_{risk}'
        if print_it:
            print(acq_risk)

        if colors is None:
            color = acquisition_risks_to_color[acq_risk]
        else:
            color = colors[i]

        y = upper_base.loc[acquisition][risk].values


        if (R := rolling) is not False:
            y = np.convolve(
                y, np.ones(R)/R, mode='valid')

        if scale == 'manual log':
            plot = ax.plot
            if errors != 'log mean':
                y = np.log10(y)
        else:
            plot = ax.loglog

        x = np.arange(1, y.size+1)
        if swapaxes:
            plot(
                y, x, '-', color=color, label=labels[i],
                zorder=zorder, lw=lw)
        else:
            plot(
                x, y, '-', color=color, label=labels[i],
                zorder=zorder, lw=lw)

        if with_errors and error_type == 'quant':

            low, up = self.extra_quant_errors
            low = low.loc[acquisition][risk].values
            up = up.loc[acquisition][risk].values

            if (R := rolling) is not False:
                up = np.convolve(
                    up, np.ones(R)/R, mode='valid')
                low = np.convolve(
                    low, np.ones(R)/R, mode='valid')
                x = np.arange(0, len(s_u))

        elif with_errors and error_type == 'std_log_error':
            middle = y
            std = self.log_sq_diff_std.loc[acquisition][risk].values
            std = std/np.sqrt(self.n_runs)
            if scale != 'manual log':
                std = np.power(std, 10)
            low = middle - std
            up = middle + std

        # elif with_errors and error_type == 'log_std_error':
        #     middle = y
        #     std = self.stds.loc[acquisition][risk].values
        #     std = std**2 / self.n_runs
        #     if add_sqrt:
        #         std = np.sqrt(std)
        #     if scale == 'manual log':
        #         std = np.log10(std)
        #     low = middle - std
        #     up = middle + std

        # else:
            # raise

        if with_errors and swapaxes:
            raise

        if with_errors:
#             ax.fill_between(x, low, up, color='white', alpha=1, zorder=-100)
            ax.fill_between(
                x, low, up, color=color, alpha=0.3, zorder=-100, lw=lwfill)
            # plot(x, std, color=color, alpha=0.3, zorder=-100, lw=1)

        i += 1

    return fig, ax


def plot_ratios(
            self, acquisition_risks, errors='std', labels=None,
            fig=None, ax=None, alpha=0.3, i=0, smoothing=False):
    """Plot mean +- std error.

    Of the risk-estimator-acquisition-function combination against the
    true empirical test risk.

    """
    if errors == 'percentiles':
        upper_base = self.quant_errors
    elif errors == 'std':
        upper_base = self.errors
    else:
        raise ValueError

    if fig is None or ax is None:
        fig, ax = plt.subplots(dpi=200)

    x = np.arange(1, self.n_points+1)
    linestyles = itertools.cycle(['--', '-.', ':'])
    for acquisition, risk in acquisition_risks:
        acq_risk = f'{acquisition}_{risk}'

        color = acquisition_risks_to_color.get(acq_risk, 'b')

        selected = upper_base.loc[
            'RandomAcquisition']['BiasedRiskEstimator'].values
        random = upper_base.loc[acquisition][risk].values

        y = selected / random

        ax.plot(
            x, y, '-', color=color, label=labels[i],
            zorder=100)

        i += 1

    return fig, ax


def plot_bars_from_ratios(
            self, acquisition_risks, errors='std', labels=None,
            fig=None, ax=None, alpha=0.3, i=0, smoothing=False,
            x_min_max=None, colors=None,
            ):
    """Plot mean +- std error.

    Of the risk-estimator-acquisition-function combination against the
    true empirical test risk.

    """
    if errors == 'percentiles':
        upper_base = self.quant_errors
    elif errors == 'std':
        upper_base = self.errors
    else:
        raise ValueError

    if fig is None or ax is None:
        fig, ax = plt.subplots(dpi=200)

    x = np.arange(1, self.n_points+1)
    linestyles = itertools.cycle(['--', '-.', ':'])
    for acquisition, risk in acquisition_risks:
        acq_risk = f'{acquisition}_{risk}'
        if colors is None:
            color = acquisition_risks_to_color[acq_risk]
        else:
            color = cols[i]

        selected = upper_base.loc[
            'RandomAcquisition']['BiasedRiskEstimator'].values
        random = upper_base.loc[acquisition][risk].values

        ratios = selected / random

        if (a := x_min_max) is not None:
            ratios = ratios[a[0]:a[1]]

        mean_ratio = np.mean(ratios)
        std_ratio = np.std(ratios)

        ax.bar(
            i, mean_ratio, color=color, label=labels[i],
            alpha=0.7
            )
        ax.plot(
            [i, i], [mean_ratio+std_ratio, mean_ratio-std_ratio],
            color='grey', lw=2
        )
        i += 1

    # ax.plot([1, i+0.5], [1, 1], c=f'C1', label='random')
    return fig, ax


def loss_dist(self, acquisition, run=0, step=0, fig=None, ax=None,
              normalise=False, T=None, lazy_save=False, with_true=False,
              color='orange', label='acquisition', s=0.1):

    if self.pmfs is None:
        raise ValueError(f'No pmfs are loaded!')

    if fig is None or ax is None:
        fig, ax = plt.subplots(dpi=200)

    vals = self.pmfs[run][acquisition]
    pmfs = vals[step]
    acq = pmfs['pmf']
    # true = pmfs['true_pmf']
    true = self.pmfs[run]['TrueLossAcquisition'][step]['pmf']

    acq, true = np.array(acq), np.array(true)

    # Normalise to max val s.t. celluloid can keep constant axes.
    max_val = np.max([acq.max(), true.max()])
    min_val = np.min([acq.min(), true.min()])
    if normalise:
        acq = (acq-acq.min())/(acq.max() - acq.min())
        true = (true-true.min())/(true.max()-true.min())
        max_val = 1
        min_val = 0

    sorted_idxs = np.argsort(true)

    if with_true:
        true_p = ax.scatter(
            np.arange(len(true[sorted_idxs])),
            true[sorted_idxs], label='true', color=pal[4], s=0.1, alpha=0.1)

    acq_p = ax.scatter(
        np.arange(len(acq[sorted_idxs])),
        acq[sorted_idxs], label=label, color=color, s=s, marker='o', alpha=0.5)

    # ax.legend(
    #     (list(true_p)+list(acq_p)+list(acquired)),
    #     ['True Loss', 'Acquisition', 'Acquired'],
    #     fontsize=6, loc='upper left')

    return fig, ax


""" Now the actual figure-making code."""


def figure1(vis):

    small = True

    errors = 'std'
    alpha = 0.5

    if not small:
        fig, axs = plt.subplots(
            1, 1, figsize=(linewidth, .7*linewidth), sharex=True)
    else:
        fig, axs = plt.subplots(
            1, 1, figsize=(0.8*linewidth, .5*linewidth), sharex=True)

    acquisition_risks = [
        ['RandomAcquisition', 'BiasedRiskEstimator'],
        ['GPAcquisitionUncertainty', 'FancyUnbiasedRiskEstimator'],
        ]
    labels = [LABEL_RANDOM, 'Active Testing', 'Theoretical Maximum']

    plot_risks_select_combinations(
        vis, acquisition_risks, labels=labels, fig=fig, ax=axs)
    if not small:
        axs.legend(prop=dict(size=fs-1))
    else:
        axs.legend(**new_kwargs)

    axs.yaxis.set_major_formatter(OOMFormatter(-2, "%1.0f"))
    axs.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    if not small:
        axs.set_ylabel(diff_to_empircal_label)
    else:
        axs.set_ylabel('Difference to \n Full Test Loss')
        axs.set_ylim(-0.07, 0.07)

    axs.set_xlabel(LABEL_ACQUIRED_FULL)

    axs.set_xticks([1, 5, 10, 15, 20, 25, 30])
    axs.set_xticklabels([1, "", 10, "", 20, "", 30])

    axs.set_xlim(1, 30)

    plt.tight_layout()

    plt.savefig(
        'notebooks/plots/fig1_small.pdf', bbox_inches='tight', pad_inches=0.02)

    return fig, axs


def figure2(vis, select):
    self = vis
    fig, axs = plt.subplots(
        2, 1, figsize=(linewidth, 2*1.85*0.5+0.7),
        gridspec_kw={'height_ratios': [1, 1]})

    run = 6
    step = 4

    selected_color = pal[-2]
    predictions_color = pal[2]

    acquisition = 'GPSurrogateAcquisitionMSE'
    labels = [LABEL_RANDOM, 'Active Testing']
    labels = [r'$\hat{R}_i$ = '+i for i in labels]

    # First plot what people have already seen
    # ax = axs[0]
    # plot_risks_select_combinations(
    #   vis, acquisition_risks, labels=labels, fig=fig, ax=ax)
    # ax.set_ylabel(diff_to_empircal_label)
    # ax.set_xlabel(acquisition_step_label)
    # ax.plot([step+1, step+1], [-7, 7], '--', c=selected_color, linewidth=1,
    #         label='Current Step')
    # ax.set_ylim(-7.5e-2, 8.5e-2)
    # ax.legend(**new_kwargs)

    # #### Now plot the data
    pmfs = self.pmfs[run][acquisition]
    data = self.datasets[run]
    data['x'] = self.check_remove_feature_dim(data['x'])

    train = data['x'][data['train_idxs']], data['y'][data['train_idxs']]

    filtered = self.results[
        (self.results.run == run)
        & (self.results.acquisition == acquisition)]
    y_preds, idx = filtered.y_preds.values, filtered.idx.values

    N = len(data['x'])
    t = step
    pmf = pmfs[t]
    if select is not None:
        next_idx = select
    else:
        next_idx = pmf['test_idx']

    test_remain = (
        data['x'][pmf['remaining']], data['y'][pmf['remaining']])
    test_select = (
        data['x'][pmf['observed']], data['y'][pmf['observed']])

    ax = axs[0]
    # plot train and test data
    train_p = ax.plot(*train, 'o', markersize=3, c=pal[1])
    remaining_p = ax.plot(*test_remain, 'o', markersize=1, c=pal[0])
    selected_p = ax.plot(
        *test_select, 'x', markersize=5, c=pal[0], linewidth=0.1)
    # current choice
    current_p = ax.plot(
        data['x'][next_idx], data['y'][next_idx],
        'x', markersize=5, linewidth=0.1, c=selected_color, label='Up Next')

    # # also plot static predictions of model
    sorted_idxs = np.argsort(data['x'][idx])
    predicted_p = ax.plot(
        data['x'][idx][sorted_idxs], y_preds[sorted_idxs],
        zorder=-1, linewidth=1, color=pal[2], alpha=0.7)

    ax.legend(
        (list(train_p)+ list(predicted_p)+list(remaining_p)
         +list(selected_p)+list(current_p)),
        ['Train', 'Model Pred.', 'Test Unobserved', 'Test Observed', 'Up Next'],
        loc='lower right', **new_kwargs)

    # ### Now plot the acquisition distributions
    ax = axs[1]

    # true loss
    align_pmf = np.zeros(N)
    align_pmf[pmf['remaining']] = pmf['true_pmf']
    # align_pmf /= align_pmf.max()

    x_bar = np.linspace(0, data['x'].max(), len(align_pmf))
    bar_w = (x_bar[1]-x_bar[0])*0.8
    bar_p = ax.bar(
        x_bar, align_pmf, color=pal[4], width=0.005, alpha=0.7,
        label='True Loss', zorder=100)

    # predicted loss
    align_pmf = np.zeros(N)
    align_pmf[pmf['remaining']] = pmf['pmf']
    # align_pmf /= align_pmf.max()

    x_bar = np.linspace(0, data['x'].max(), len(align_pmf))
    bar_w = (x_bar[1]-x_bar[0])*0.8
    bar_p = ax.bar(
        x_bar, align_pmf, color=predictions_color, width=bar_w,
        label='Approximate Loss', alpha=0.5)

    # bar_p = ax.plot(x_bar, align_pmf, color=pal[4], label='True Loss')

    # current choice
    bar_select_p = ax.bar(
        x_bar[next_idx], align_pmf[next_idx],
        color=selected_color, width=bar_w, label='Up Next')

    ax.legend(**new_kwargs)

    # def plot_acquisition(self, run, fig, ax)
    pmfs = self.pmfs[run][acquisition]
    data = self.datasets[run]
    data['x'] = self.check_remove_feature_dim(data['x'])

    train = data['x'][data['train_idxs']], data['y'][data['train_idxs']]

    filtered = self.results[
        (self.results.run == run)
        & (self.results.acquisition == acquisition)]
    y_preds, idx = filtered.y_preds.values, filtered.idx.values

    N = len(data['x'])
    t = step
    pmf = pmfs[t]

    test_remain = (
        data['x'][pmf['remaining']], data['y'][pmf['remaining']])
    test_select = (
        data['x'][pmf['observed']], data['y'][pmf['observed']])

    # axs[0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    axs[0].set_ylabel(r'Outcome $y$')
    axs[0].set_yticks([-0.8, -1.3])
    axs[0].set_yticklabels([1, 0.5])
    axs[0].set_xticklabels([])

    axs[1].yaxis.set_major_formatter(OOMFormatter(-2, "%1.0f"))
    axs[1].ticklabel_format(axis="y", style="sci", scilimits=(0,0))

    # axs[1].set_yticks([0, .1, .2])
    axs[1].set_ylabel('Acquisition', labelpad=6)
    axs[1].set_xlabel(r'Input $x$')
    axs[1].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    # axs[1].set_xticklabels([0, "", 0.4, "", 0.8, 1])
    axs[0].set_xlim(-.03, 1.03)
    axs[1].set_xlim(-.03, 1.03)

    voffs = [1, 1.1]
    for i, title in enumerate('ab'):
        title = f'({title})'
        axs[i].text(
            -0.138, voffs[i], title, fontsize=label_fs, c='k',
            horizontalalignment='center',
            verticalalignment='center',
            transform=axs[i].transAxes)    

    fig.subplots_adjust(hspace=0.3)

    # # plt.tight_layout()
    plt.savefig(
        'notebooks/plots/illustrative.pdf', bbox_inches='tight',
        pad_inches=0.01)

    return fig, axs


def figure3(visualisers):
    class_idx = 2
    # viss = [visualisers[i] for i in [0, 3, 4]]
    viss = visualisers
    # namess = [
    #     'Gaussian Process / Gaussian Process Prior',
    #     'Linear Regression / Quadratic Data',
    #     'Random Forest / Two Moons Data'
    # ]
    namess = [
        'GP / GP / GP Prior',
        'Linear / GP / Quadratic',
        'RF / RF / Two Moons'
    ]

    fig = plt.figure(figsize=(linewidth, 3.8))  # , constrained_layout=True)
    gs = fig.add_gridspec(3, 2)   # , height_ratios=[0.6, 1])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)

    ax3 = fig.add_subplot(gs[0, 1])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, 1])

    axs = np.array([[ax0, ax3],[ax1, ax4], [ax2, ax5]])

    # plot convergence into first column
    acq_risk = [
        ['RandomAcquisition', 'BiasedRiskEstimator'],
        ['GPSurrogateAcquisitionMSE', 'FancyUnbiasedRiskEstimator'],
    ]
    import copy
    acq_risk = [copy.deepcopy(acq_risk) for i in range(3)]

    # called differently for classification
    acq_risk[-1][1][0] = \
        'RandomForestClassifierSurrogateAcquisitionEntropy_RFHighCapacity'

    labels = [LABEL_RANDOM, 'Active Testing', 'Theoretically Optimal']
    i = 0
    for col_ax, vis in zip(axs[:, 0], viss):
        plot_risks_select_combinations(
            vis, acq_risk[i], fig=fig, ax=col_ax, labels=labels, lw=0.5)
        col_ax.set_xticks([1, 20, 40])
        col_ax.set_xlim(0, 48)
        if i == 0:
            col_ax.legend(**new_kwargs, loc='upper right')

        i += 1

    ax1.set_ylabel('Difference to Full Test Loss', labelpad=10)
    ax2.set_xlabel(acquisition_step_label)
    ax5.set_xlabel('Example Data', labelpad=13)
    ax0.yaxis.set_major_formatter(OOMFormatter(-2, "%1.0f"))
    ax0.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax1.yaxis.set_major_formatter(OOMFormatter(-2, "%1.0f"))
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax2.yaxis.set_major_formatter(OOMFormatter(-0, "%1.0f"))
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax0.tick_params(labelbottom=False)
    ax1.tick_params(labelbottom=False)

    ax1.set_ylim([-3e-2, 3e-2])
    # ax2.set_ylim([-8e-2, 8e-2])

    # plot data into second column row
    i = 0
    # hoff = [0.1, 0, 0]
    hoff = [0, 0, 0]
    for col_ax, vis in zip(axs[:, 1], viss):
        if i != class_idx:
            plot_regression(vis, 9, ax=col_ax, fig=fig, legend=False)
        else:
            plot_class(col_ax, 3, vis)
        if i == 0:
            col_ax.set_ylim(0.8, 2)
        if i == class_idx:
            col_ax.set_ylim(-2, 3.4)
        if i == 0:
            col_ax.legend(loc='upper left', **new_kwargs)
        if i == class_idx:
            col_ax.legend(loc='upper left', **new_kwargs)

    #     ax.set_title(names[i], fontsize=fs)

        col_ax.text(
            hoff[i], 1.1, namess[i], fontsize=label_fs, c='k',
            horizontalalignment='center',
            verticalalignment='center',
            transform=col_ax.transAxes)

        col_ax.text(
            -1.2, 1.1, f"({'abc'[i]})", fontsize=label_fs, c='k',
            horizontalalignment='center',
            verticalalignment='center',
            transform=col_ax.transAxes)

        col_ax.set_xticks([])
        col_ax.set_yticks([])
        # col_ax.set_xticklabels([])
        # col_ax.set_yticklabels([])

        i += 1
    
    # plt.tight_layout()
    plt.subplots_adjust(hspace=.3, wspace=.05)

    plt.savefig(
        'notebooks/plots/synthetic_comparison.pdf', bbox_inches='tight',
        pad_inches=0.02)
    return fig, axs


def figure4(visualisers, names):
    fig, axs = plt.subplots(2, 1, figsize=(linewidth-0.071, 3), sharex=True)
    # axs = [axs[1], axs[0]]

    errors = 'percentiles'

    acquisition_risks = [
        ['RandomAcquisition', 'BiasedRiskEstimator'],
        ['RandomForestClassifierSurrogateAcquisitionEntropy_RFInfDepth',
         'FancyUnbiasedRiskEstimator'],
        ['ClassifierAcquisitionEntropy', 'FancyUnbiasedRiskEstimator'],
        ['SelfSurrogateAcquisitionEntropy', 'FancyUnbiasedRiskEstimator']
        ]

    labels = [
        LABEL_RANDOM, 'Random Forest Surrogate', 'Original Model',
        'BNN Surrogate']

    plot_log_convergence(
        visualisers[0], acquisition_risks, fig=fig, ax=axs[0], labels=labels,
        errors=errors,
        rolling=10,
    )

    acquisition_risks = [
        ['RandomAcquisition', 'BiasedRiskEstimator'],
        ['RandomForestClassifierSurrogateAcquisitionEntropy_RFInfDepth',
         'FancyUnbiasedRiskEstimator'],
        ['ClassifierAcquisitionEntropy', 'FancyUnbiasedRiskEstimator'],
        ['SelfSurrogateAcquisitionEntropy', 'FancyUnbiasedRiskEstimator'],
        ['AnySurrogateAcquisitionEntropy_LazySurrEnsemble',
         'FancyUnbiasedRiskEstimator'],
        ]

    labels = [
        LABEL_RANDOM, 'Random Forest Surrogate', 'Original Model',
        'ResNet Surrogate', 'ResNet Train Ensemble']

    new_order = [2, 0, 3, 4, 1]

    acquisition_risks = [acquisition_risks[i] for i in new_order]
    labels = [labels[i] for i in new_order]

    fig, ax = plot_log_convergence(
        visualisers[1],
        acquisition_risks,
        labels=labels,
        errors=errors,
        rolling=10,
        fig=fig,
        ax=axs[1],
        zorder=-100
    )

    axs[0].set_title('Radial BNN on MNIST', fontsize=label_fs)
    axs[0].set_xlim(1, 1000)
    axs[0].set_xscale('linear')
    axs[0].set_ylim(3e-5, 0.5e-2)
    axs[0].legend(**new_kwargs)
    # axs[0].yaxis.set_major_formatter(OOMFormatter(-2, "%1.0f"))
    # axs[0].ticklabel_format(axis='y', style='sci', scilimits=(-4,-4))

    axs[1].set_title('ResNet-18 on Fashion-MNIST', fontsize=label_fs)
    axs[1].legend(**new_kwargs, facecolor='white', framealpha=1, loc='upper right', fontsize=fs)

    axs[1].set_xlim(1, 1e3)
    axs[1].set_xscale('linear')
    # axs[1].set_xticks([1, 200, 400, 600, 800])
    axs[1].set_xticks([1, 200, 400, 600, 800, 1000])
    # axs[1].set_xticklabels([1, 200, "", 600, "", 1000])

    # axs[1].yaxis.set_major_formatter(OOMFormatter(-1, "%1.0f"))
    # axs[1].ticklabel_format(axis='y', style='sci', scilimits=(-4,-4))
    axs[1].set_ylim(1e-3, 0.8e0)

    for tick in axs[0].yaxis.get_minor_ticks():
        tick.label1.set_visible(False)

    for tick in axs[1].yaxis.get_minor_ticks():
        tick.label1.set_visible(False)

    # axs[0].set_yticks([0.02, 0.05, 0.1])
    # axs[1].set_yticks([10e-2, 100e-2])

    axs[0].set_ylabel(LABEL_STD)

    axs[1].set_ylabel(LABEL_STD)
    axs[1].set_xlabel(acquisition_step_label)

    # plt.tight_layout()
    # for ax in axs:
    #     ax.set_yscale('linear')

    axs[0].text(
        -0.23, 1.12, '(a)', fontsize=label_fs, c='k',
        horizontalalignment='center',
        verticalalignment='center',
        transform=axs[0].transAxes)

    axs[1].text(
        -0.23, 1.12, '(b)', fontsize=label_fs, c='k',
        horizontalalignment='center',
        verticalalignment='center',
        transform=axs[1].transAxes)

    plt.subplots_adjust(hspace=.4)

    plt.savefig(
        'notebooks/plots/img_small.pdf', bbox_inches='tight',
        pad_inches=0.01)

    return fig, axs


def figure5(vis):
    fig, axs = plt.subplots(
        1, 3, figsize=(linewidth, 0.3*linewidth), sharey=True)

    with_true = False
    cols = [pal[4], pal[2], pal[-1]]
    s = 0.1
    labels = ['True Loss', 'No Surrogate', 'Train Ensemble']
    loss_dist(
        vis, 'TrueLossAcquisition', run=0, step=0, fig=fig, ax=axs[0],
        with_true=with_true, color=cols[0], label=labels[0], s=s)
    loss_dist(
        vis, 'ClassifierAcquisitionEntropy', run=0, step=0, fig=fig, ax=axs[1],
        with_true=True, color=cols[1], label=labels[1], s=s)
    loss_dist(
        vis, 'AnySurrogateAcquisitionEntropy_LazySurrEnsemble', run=0, step=0,
        fig=fig, ax=axs[2],
        with_true=True, color=cols[2], label=labels[0], s=s)

    axs[1].set_xlabel('Sorted Indices')
    axs[0].set_ylabel('Cross-Entropy')

    for i, ax in enumerate(axs):
        ax.set_xticks([1, 500, 1000])
        ax.set_xlim(0, 1100)
        ax.set_title(labels[i], fontsize=fs, color=cols[i], loc='left')
        if i == 0:
            continue

    # CIFAR10
        # rect = matplotlib.patches.Rectangle(
        #     (930, 0.00005), 100, 0.032,
        #     linewidth=0.5,edgecolor=pal[3],facecolor='none')

    # CIFAR 100
        rect = matplotlib.patches.Rectangle(
            (800, 0.000001), 230, 0.01,
            linewidth=0.5, edgecolor=pal[3], facecolor='none')
        ax.add_patch(rect)

    #     a = 0.005
    #     ax.plot([0, 1200], [a, a], '--', c='grey', lw=1)
    #     ax.legend(loc='upper left', **new_kwargs)

    # axs[0].yaxis.set_major_formatter(OOMFormatter(-2, "%1.0f"))
    # axs[0].ticklabel_format(axis='y', style='sci', scilimits=(-4,-4))

    ax.set_yscale('log')
    ax.set_ylim(1e-8, 0.04)
    ax.set_yticks([1e-8, 1e-5, 1e-2])
    # ax.set_yticklabels([1e-8, 1e-5, 10^])

    for i, title in enumerate('abc'):
        title = f'({title})'
        # axs[i].set_title(
        #     title, fontsize=label_fs, c='grey', loc='right', pad=4)
        axs[i].text(
            -0.15, 1.16, title, fontsize=fs, c='k',
            horizontalalignment='center',
            verticalalignment='center',
            transform=axs[i].transAxes)

    plt.subplots_adjust(hspace=.4)

    plt.savefig(
        'notebooks/plots/dists.pdf', bbox_inches='tight', pad_inches=0.02)

    return fig, axs


def figure6(visualisers, names):
    inverse = True
    with_errors = False
    error = 'percentiles'
    fig = plt.figure(
        figsize=(linewidth-0.042, 3-0.01+0.2),)
        # constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[0.6, 1])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1], sharey=ax0)
    ax2 = fig.add_subplot(gs[0, 2], sharey=ax0)
    row = [ax0, ax1, ax2]
    ax3 = fig.add_subplot(gs[1, :])
    ax3.grid(b=True, which='major', linestyle='-', alpha=0.5, axis='y')
    # ax3.grid(b=True, which='minor', linestyle='-', alpha=0.5)
    axs = [row, [ax3]]

    acquisitions = [
        'AnySurrogateAcquisitionEntropy_LazySurrEnsembleLarge',
        'AnySurrogateAcquisitionEntropy_LazySurrEnsemble',
        'AnySurrogateAcquisitionEntropy_LazySurrEnsemble',
        'AnySurrogateAcquisitionEntropy_LazySurrEnsemble',
    ]
    # acquisitions = [a+'20' for a in acquisitions]

    acquisition_risks = [
        ['RandomAcquisition', 'BiasedRiskEstimator'],
        ['REPLACE THIS', 'FancyUnbiasedRiskEstimator'],
        ]

    labels = [LABEL_RANDOM, 'Active Testing']

    # plot data into first rows
    off = 0
    viss = visualisers[off:-1]
    colsdata = [pal[1], pal[0], pal[-2]]
    for i in range(len(viss)):
        cols = [pal[3], pal[2]]
        if inverse:
            pos = 2-i
        else:
            pos = 1

        acquisition_risks[1][0] = acquisitions[i+off]
        plot_log_convergence(
            viss[i], acquisition_risks, fig=fig, ax=row[pos], labels=labels,
            colors=cols, errors=error, with_errors=with_errors)
        lab = names[i+off].split(' ')[1]
        row[pos].set_title(lab, fontsize=fs, color=colsdata[i])

    # now plot equivalent samples into the other row

    # # first, just plot the data
    colsdata2 = [pal[1], pal[0], pal[-2], pal[-1]]

    # # new visualiser order
    if inverse:
        new_order = [0, 3, 1, 2]
        new_order = new_order[::-1] # legend order
        plot_equivalent_samples(
            visualisers[0], [acquisition_risks[0]], fig=fig, ax=ax3,
            labels=[labels[0]],
            errors=error, inverse=inverse)

    else:
        new_order = [0, 1, 3, 2]

    for i in new_order:
        vis = visualisers[i]

        acq_risk = [acquisitions[i], 'FancyUnbiasedRiskEstimator']
        lab = [' '.join(names[i].split(' ')[1:])]
        # print(i, lab, acquisitions[i], vis.acquisitions, acq_risk)
        plot_equivalent_samples(
            vis, [acq_risk], fig=fig, ax=ax3, labels=lab,
            colors=[colsdata2[i]], errors=error, inverse=inverse)

    if not inverse:
        # now also plot random
        plot_equivalent_samples(
            visualisers[0], [acquisition_risks[0]], fig=fig, ax=ax3,
            labels=[labels[0]], errors=error, inverse=inverse)

    for ax in row[1:]:
        ax.yaxis.set_tick_params(labelleft=False)

    # upper plots
    for ax in row:
    #     ax.set_xlim(0, 1000)
        ax.set_xscale('linear')
        ax.set_xlim(0, 500)
        ax.set_xticks([1, 200, 400])
        ax.set_xticklabels([1, 200, 400])
        ax.set_yticks([1e-1, 1e-3, 1e-5])

    ax1.set_xlabel(acquisition_step_label)
    ax0.set_ylim(1e-5, 0.3e-0)
    # ax0.set_ylim(3e-7, 0.6e-0)
    ax0.set_ylabel(LABEL_STD)
    if inverse:
        ax2.legend(**new_kwargs, loc='upper right')
    else:
        ax2.legend(**new_kwargs, loc='lower center')
    # ax0.legend(**new_kwargs, loc='upper center')

    # lower plots
    ax3.legend(**new_kwargs, loc='upper left', facecolor='white', framealpha=1)
    ax3.set_xlim(0, 200)
    ax3.set_xticks([1, 50, 100, 150, 200])
    # ax3.set_xticklabels([1, "", 100, "", 200])

    ax3.set_xlabel(acquisition_step_label)
    # ax3.set_yscale('log')

    if not inverse:
        ticks = np.array([1, 2, 4, 10, 20])
        ax3.set_yticks(ticks)
        ax3.set_yticklabels(ticks)
        ax3.set_ylim(0.7, 25)
        ax3.set_ylabel(sample_efficiency_label, labelpad=16)

    if inverse:
        # for tick in ax3.yaxis.get_minor_ticks():
        #     tick.set_visible(False)
        # for tick in ax3.yaxis.get_major_ticks():
        #     tick.set_visible(False)
        # ticks = np.array([0.1, 0.25, 0.5, 1])
        ax3.set_ylim(1/25, 1.3)
        ax3.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax3.set_yticklabels([0, '', 0.5, '', 1])
        ax3.set_ylabel('Relative \n Labeling Cost', labelpad=11)
        ax3.grid(False)
    #     ax3.yaxis.set_major_formatter(OOMFormatter(0, "%1.1f"))
    #     ax3.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        ax3.grid(True, axis='y')        

    ax0.text(
        -0.71, 1.16, '(a)', fontsize=label_fs, c='k',
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax0.transAxes)

    ax3.text(
        -0.235, 1.05, '(b)', fontsize=label_fs, c='k',
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax3.transAxes)

    plt.subplots_adjust(wspace=0, hspace=0.5)

    plt.savefig(
        'notebooks/plots/big_data_log_conv.pdf', bbox_inches='tight',
        pad_inches=0.01)

    return fig, axs


def figure7(vis):
    fig = plt.figure(
        figsize=(1.15*linewidth-0.04, 0.9*3.8/3*2-0.05+0.3),)

    error = 'percentiles'

    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    axs = [ax0, ax1]

    labels = [
        r'Naive Entropy',
        r'Bias-Corrected Entropy',
        r'Theoretically Optimal']

    acquisition_risks = [
    #     ['RandomAcquisition', 'BiasedRiskEstimator'],
        ['ClassifierAcquisitionEntropy', 'BiasedRiskEstimator'],
        ['ClassifierAcquisitionEntropy', 'FancyUnbiasedRiskEstimator'],
        ['TrueLossAcquisition', 'FancyUnbiasedRiskEstimator'],
    ]

    plot_risks_select_combinations(
        vis,
        acquisition_risks,
        labels=labels,
        errors=error,
        fig=fig,
        ax=axs[0],
        lw=0.5,
        # lw2=1
    )

    labels = [LABEL_RANDOM, 'Mutual Information', 'Predictive Entropy']

    acquisition_risks = [
        ['RandomAcquisition', 'BiasedRiskEstimator'],
        ['BNNClassifierAcquisitionMI', 'FancyUnbiasedRiskEstimator'],
        ['ClassifierAcquisitionEntropy', 'FancyUnbiasedRiskEstimator'],
    ]

    plot_log_convergence(
        vis,
        acquisition_risks,
        labels=labels,
        errors=error,
        fig=fig,
        ax=axs[1],
        zorder=-100,
        rolling=10,
    )

    axs[0].set_xlim(0.01, 400)
    # for tick in axs[0].xaxis.get_minor_ticks():
    #     tick.label1.set_visible(False)
    # for tick in axs[0].xaxis.get_major_ticks():
    #     tick.label1.set_visible(False)

    axs[0].set_ylim(-0.2, 0.8)
    axs[0].yaxis.set_major_formatter(OOMFormatter(-1, "%1.0f"))
    axs[0].ticklabel_format(axis='y', style='sci', scilimits=(-0,-0))
    axs[0].set_yticks([0, 0.3, 0.6])

    leg = axs[0].legend(
        **new_kwargs, loc='upper right', framealpha=1, facecolor='white')
    leg.set_zorder(10000)

    axs[0].set_ylabel('Difference to \n Full Test Loss', labelpad=18)

    # axs[0].set_xlabel(acquisition_step_label)
    # axs[0].set_title('Radial BNN on MNIST', fontsize=label_fs)
    axs[1].set_xscale('linear')
    axs[1].set_xticks([1, 100, 200, 300, 400])
    # axs[1].set_xticklabels([1, None, 200, None, 400])
    axs[1].set_xlim(0.01, 400)
    axs[1].set_ylim(3e-4, 1e-2)
    axs[1].set_yticks([1e-3, 1e-2])
    # axs[1].yaxis.set_major_formatter(OOMFormatter(-2, "%1.0f"))
    # axs[1].yaxis.set_minor_formatter(OOMFormatter(-2, "%1.0f"))
    # axs[1].ticklabel_format(axis='y', style='sci', scilimits=(-4,-4))

    for tick in axs[1].yaxis.get_minor_ticks():
        tick.label1.set_visible(False)

    axs[1].legend(**new_kwargs)
    axs[1].set_xlabel(acquisition_step_label)
    axs[1].set_ylabel(LABEL_STD)

    # -0.29
    axs[0].text(
        -0.35, 1.1, '(a)', fontsize=label_fs, c='k',
        horizontalalignment='center',
        verticalalignment='center',
        transform=axs[0].transAxes)

    axs[1].text(
        -0.35, 1.04, '(b)', fontsize=label_fs, c='k',
        horizontalalignment='center',
        verticalalignment='center',
        transform=axs[1].transAxes)

    plt.tight_layout()

    plt.subplots_adjust(hspace=.5, wspace=.001)

    plt.savefig(
        'notebooks/plots/bias_and_mi.pdf', bbox_inches='tight',
        pad_inches=0.01)

    return fig, axs


"""Appendix Plots."""


def figureA1(visualisers):

    # names = [
    #     'GP / GP Prior',
    #     'GP / Quadratic',
    #     'GP / Sinusoidal',
    #     'Linear / Quadratic',
    #     'RF / Two Moons'
    # ]

    names = [
        'GP / GP / GP Prior',
        'GP / GP / Quadratic',
        'GP / GP / Sinusoidal',
        'Linear / GP / Quadratic',
        'RF / RF / Two Moons'
    ]

    fig, axs = plt.subplots(
        2, 5, figsize=(textwidth, 3*0.8),
        gridspec_kw={'height_ratios': [0.8, 1]})

    for ax in axs[0]:
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axs[1]:
        ax.set_yticks([])
        ax.set_xticks([0, 20, 40])

    axs[0][0].set_ylabel('Model / Data')
    axs[1][0].set_ylabel('Difference to \n Full Test Loss')
    # for ax in axs[1]:
    #     ax.set_xlabel(acquisition_step_label)
    axs[1][2].set_xlabel(LABEL_ACQUIRED_FULL)

    # plot data into first row
    i = 0
    for col_ax, vis in zip(axs[0], visualisers):
        ax = col_ax
        if i != 4:
            plot_regression(vis, 9, ax=col_ax, fig=fig, legend=False)
        else:
            plot_class(col_ax, 3, vis)

        if i == 0:
            ax.set_ylim(0.8, 2.3)
        if i == 4:
            ax.set_ylim(-2, 3.4)

        if i == 0:
            ax.legend(loc='upper left', **new_kwargs)
        if i == 4:
            ax.legend(loc='upper right', **new_kwargs)

        ax.set_title(names[i], fontsize=fs)
        i += 1

    # plot convergence into second row
    acq_risk = [
        ['RandomAcquisition', 'BiasedRiskEstimator'],
        ['GPSurrogateAcquisitionMSE', 'FancyUnbiasedRiskEstimator'],
        ['TrueLossAcquisition', 'FancyUnbiasedRiskEstimator']
    ]
    import copy
    acq_risk = [copy.deepcopy(acq_risk) for i in range(5)]

    # called differently for classification
    acq_risk[-1][1][0] = \
        'RandomForestClassifierSurrogateAcquisitionEntropy_RFHighCapacity'

    labels = [LABEL_RANDOM, 'Active Testing', 'Theoretically Optimal']
    # labels = [r'$\hat{R}_i$ = '+i for i in labels]
    i = 0
    for col_ax, vis in zip(axs[1], visualisers):
        fig, ax = plot_risks_select_combinations(
            vis, acq_risk[i], fig=fig,
            ax=col_ax, labels=labels, lw=0.5)

    #     ax.set_xlim(0, 48)
        ax.set_xticks([1, 20, 40])
        if i == 4:
            ax.legend(**new_kwargs, loc='upper right')

        i += 1

    plt.tight_layout()

    plt.subplots_adjust(hspace=.05, wspace=.001)

    plt.savefig(
        'notebooks/plots/synthetic_comparison_extended.pdf',
        bbox_inches='tight', pad_inches=0.01)


def figureA2(vis):
    fig, axs = plt.subplots(1, 3, figsize=(textwidth, 0.8))

    inverse = True
    with_errors = False
    acquisition_risks = [
        ['RandomAcquisition', 'BiasedRiskEstimator'],
        ['ClassifierAcquisitionEntropy', 'FancyUnbiasedRiskEstimator'],
        ]

    labels = [LABEL_RANDOM, 'Active Testing', 'MI']

    # axs[0].set_title('BNN: Entropy Acquisition')

    plot_risks_select_combinations(
        vis, acquisition_risks, fig=fig, ax=axs[0], labels=labels,
        errors='percentiles', lw=0.5
    )

    plot_log_convergence(
        vis, acquisition_risks, fig=fig, ax=axs[1], labels=labels,
        errors='percentiles',
        with_errors=with_errors
    )

    plot_equivalent_samples(
        vis, acquisition_risks, fig=fig, ax=axs[2], labels=labels,
        errors='percentiles',
        relative=True, inverse=inverse
    )

    for ax in axs:
        ax.set_xlabel(acquisition_step_label)

    axs[0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    axs[0].set_xticks([1, 500, 1000])
    label = 'Difference to \n Full Test Loss'
    axs[0].set_ylabel(label, fontsize=label_fs)
    axs[0].set_ylim(-0.08, 0.08)
    axs[0].set_xlim(1, 1000)

    # axs[1].set_ylabel(r'$\mathrm{Std}(A_i) \, := B_i$', fontsize=label_fs)
    axs[1].set_xlim(1, 1000)
    axs[1].set_ylim(7e-6, 3e-3)
    axs[1].set_yticks([1e-5, 1e-4, 1e-3])
    axs[1].set_xscale('linear')
    axs[1].set_xticks([1, 500, 1000])

    label = LABEL_STD
    axs[1].set_ylabel(label, fontsize=label_fs)
    # axs[1].set_ylim(0.1e-2, 1e-1)
    # axs[1].set_yticks([.1e-2, 1e-2, 1e-1])
    # axs[1].yaxis.set_major_formatter(OOMFormatter(-3, "%1.0f"))
    # axs[1].ticklabel_format(axis='y', style='sci', scilimits=(-4,-4))

    axs[2].set_xlim(0, 200)
    axs[2].set_xticks([1, 100, 200])
    # axs[2].set_ylabel(r'$x \mid B_i = x B_{i.i.d.} $', fontsize=label_fs)

    if not inverse:
        axs[2].set_yscale('log')
        axs[2].set_ylabel(sample_efficiency_label, fontsize=label_fs)
        axs[2].set_yticks([1, 4, 10, 40])
        axs[2].set_yticklabels([1, 4, 10, 40])
        axs[2].set_ylim(0.7, 70)
        axs[2].legend(**new_kwargs, loc='upper right')
        axs[2].grid(True, axis='y', zorder=-100, alpha=0.3)

    if inverse:
        axs[2].set_ylabel(
            'Relative \n Labeling Cost', fontsize=label_fs, labelpad=5)    
        axs[2].set_yticks([0, 0.25, 0.5, 0.75, 1])
        axs[2].set_yticklabels([0, '', 0.5, '', 1])
        axs[2].set_ylim([0, 1.175])
    #     axs[2].set_yticks([1e-2, 10e-2, 25e-2, 100e-2])
        axs[2].legend(
            **new_kwargs, loc='upper left', framealpha=1, facecolor='white')
    #     axs[2].yaxis.set_major_formatter(OOMFormatter(0, "%1.2f"))
    #     axs[2].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        axs[2].grid(True, axis='y', zorder=-100, alpha=0.3)

    hoffs = [-0.55, -0.55, -0.55]
    for i, title in enumerate('abc'):
        title = f'({title})'
        axs[i].text(
            hoffs[i], 1.15, title, fontsize=label_fs, c='k',
            horizontalalignment='center',
            verticalalignment='center',
            transform=axs[i].transAxes)

    plt.subplots_adjust(wspace=0.62)

    # plt.savefig('notebooks/plots/img_mnist.pdf')
    plt.savefig(
        'notebooks/plots/figureA2.pdf', bbox_inches='tight', pad_inches=0.02)

    return fig, axs


def figureA3(visualisers, std_log_error=False):

    # figure 4 with errors
    fig, axs = plt.subplots(2, 1, figsize=(textwidth, 6), sharex=True)
    # axs = [axs[1], axs[0]]

    if std_log_error:
        plot_args = dict(
            lw=0.5, error_type='std_log_error', scale='manual log',
            errors='log mean', lwfill=0)
    else:
        plot_args = dict()

    acquisition_risks = [
        ['RandomAcquisition', 'BiasedRiskEstimator'],
        ['RandomForestClassifierSurrogateAcquisitionEntropy_RFInfDepth',
         'FancyUnbiasedRiskEstimator'],
        ['ClassifierAcquisitionEntropy', 'FancyUnbiasedRiskEstimator'],
        ['SelfSurrogateAcquisitionEntropy', 'FancyUnbiasedRiskEstimator']
        ]

    labels = [
        LABEL_RANDOM, 'Random Forest Surrogate', 'Original Model',
        'BNN Surrogate']

    if not std_log_error:
        new_order = [0, 3]
        acquisition_risks = [acquisition_risks[i] for i in new_order]
        labels = [labels[i] for i in new_order]

    plot_log_convergence(
        visualisers[0], acquisition_risks, fig=fig, ax=axs[0], labels=labels,
        # rolling=10,
        with_errors=True,
        **plot_args
    )

    acquisition_risks = [
        ['RandomAcquisition', 'BiasedRiskEstimator'],
        ['RandomForestClassifierSurrogateAcquisitionEntropy_RFInfDepth',
         'FancyUnbiasedRiskEstimator'],
        ['ClassifierAcquisitionEntropy', 'FancyUnbiasedRiskEstimator'],
        ['SelfSurrogateAcquisitionEntropy', 'FancyUnbiasedRiskEstimator'],
        ['AnySurrogateAcquisitionEntropy_LazySurrEnsemble',
         'FancyUnbiasedRiskEstimator'],
        ]

    labels = [
        LABEL_RANDOM, 'Random Forest Surrogate', 'Original Model',
        'ResNet Surrogate', 'ResNet Train Ensemble']

    if std_log_error:
        new_order = [2, 0, 3, 4, 1]
    else:
        new_order = [2, 1]

    acquisition_risks = [acquisition_risks[i] for i in new_order]
    labels = [labels[i] for i in new_order]

    fig, ax = plot_log_convergence(
        visualisers[1],
        acquisition_risks,
        labels=labels,
        # rolling=10,
        fig=fig,
        ax=axs[1],
        zorder=-100,
        with_errors=True,
        **plot_args
    )

    axs[0].set_title('Radial BNN on MNIST', fontsize=fs)
    axs[0].set_xlim(1, 1000)
    axs[0].set_xscale('linear')
    if std_log_error:
        axs[0].set_ylim(-5, np.log10(0.5e-2))
    else:
        axs[0].set_ylim(8e-6, 0.5e-2)
    axs[0].legend()
    # axs[0].yaxis.set_major_formatter(OOMFormatter(-2, "%1.0f"))
    # axs[0].ticklabel_format(axis='y', style='sci', scilimits=(-4,-4))

    axs[1].set_title('ResNet-18 on Fashion-MNIST', fontsize=fs)
    axs[1].legend(facecolor='white', framealpha=1, loc='upper right')

    axs[1].set_xlim(1, 1e3)
    axs[1].set_xscale('linear')
    # axs[1].set_xticks([1, 200, 400, 600, 800])
    axs[1].set_xticks([1, 200, 400, 600, 800, 1000])
    # axs[1].set_xticklabels([1, 200, "", 600, "", 1000])

    # axs[1].yaxis.set_major_formatter(OOMFormatter(-1, "%1.0f"))
    # axs[1].ticklabel_format(axis='y', style='sci', scilimits=(-4,-4))
    if std_log_error:
        axs[1].set_ylim(-3.2, 1.3)
    else:
        axs[1].set_ylim(1e-4, 1e1)

    for tick in axs[0].yaxis.get_minor_ticks():
        tick.label1.set_visible(False)

    for tick in axs[1].yaxis.get_minor_ticks():
        tick.label1.set_visible(False)

    # axs[0].set_yticks([0.02, 0.05, 0.1])
    # axs[1].set_yticks([10e-2, 100e-2])
    if std_log_error:
        y_label = LABEL_MEAN_LOG
    else:
        y_label = LABEL_STD
    y_label = y_label.replace('\n', '')

    axs[0].set_ylabel(y_label)
    axs[1].set_ylabel(y_label)
    axs[1].set_xlabel(acquisition_step_label)

    axs[0].set_yticks([-3, -4, -5])

    # plt.tight_layout()
    # for ax in axs:
    #     ax.set_yscale('linear')

    axs[0].text(
        -0.075, 1.055, '(a)', fontsize=label_fs, c='k',
        horizontalalignment='center',
        verticalalignment='center',
        transform=axs[0].transAxes)

    axs[1].text(
        -0.075, 1.055, '(b)', fontsize=label_fs, c='k',
        horizontalalignment='center',
        verticalalignment='center',
        transform=axs[1].transAxes)

    plt.subplots_adjust(hspace=.2)

    plt.savefig(
        'notebooks/plots/figureA3.pdf', bbox_inches='tight',
        pad_inches=0.01)

    return fig, axs


def figureA4(visualisers, names, std_log_error=False):
    inverse = True
    with_errors = True
    if std_log_error:
        plot_args = dict(
            lw=0.5, error_type='std_log_error', scale='manual log',
            errors='log mean', lwfill=0)
        # plot_args = dict(
        #     lw=0.5, error_type='std_log_error', scale='manual log',
        #     errors='std', lwfill=0)

    else:
        plot_args = dict()

    fig = plt.figure(
        figsize=(textwidth, 3-0.01),)
        # constrained_layout=True)

    gs = fig.add_gridspec(1, 3)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    row = [ax0, ax1, ax2]
    axs = [row]

    acquisitions = [
        'AnySurrogateAcquisitionEntropy_LazySurrEnsembleLarge',
        'AnySurrogateAcquisitionEntropy_LazySurrEnsemble',
        'AnySurrogateAcquisitionEntropy_LazySurrEnsemble',
        'AnySurrogateAcquisitionEntropy_LazySurrEnsemble',
    ]

    acquisition_risks = [
        ['RandomAcquisition', 'BiasedRiskEstimator'],
        ['ClassifierAcquisitionEntropy', 'FancyUnbiasedRiskEstimator'],
        ]

    labels = [LABEL_RANDOM, 'Active Testing']

    # plot data into first rows
    off = 0
    viss = visualisers[off:-1]
    colsdata = [pal[1], pal[0], pal[-2]]
    for i in range(len(viss)):
        cols = [pal[3], pal[2]]
        if inverse:
            pos = 2-i
        else:
            pos = 1

        acquisition_risks[1][0] = acquisitions[i+off]
        plot_log_convergence(
            viss[i], acquisition_risks, fig=fig, ax=row[pos], labels=labels,
            colors=cols, with_errors=with_errors,
            **plot_args
            )
        lab = names[i+off].split(' ')[1]
        row[pos].set_title(lab, fontsize=fs, color=colsdata[i])

    # upper plots
    for ax in row:
    #     ax.set_xlim(0, 1000)
        ax.set_xscale('linear')
        ax.set_xlim(0, 500)

    if std_log_error:
        row[0].set_ylim(np.log10(9e-5), -1)
        row[1].set_ylim(np.log10(5e-6), -1)
        row[2].set_ylim(np.log10(1e-6), -1)
        row[0].set_yticks([-1, -2, -3, -4])
        row[1].set_yticks([-1, -2, -3, -4, -5])
        row[2].set_yticks([-1, -2, -3, -4, -5, -6])
    else:
        row[0].set_ylim(9e-5, 1e-1)
        row[1].set_ylim(5e-6, 1e-1)
        row[2].set_ylim(1e-6, 1e-1)
    for ax in row:
        ax.set_xlabel(acquisition_step_label)
    # ax0.set_ylim(1e-5, 0.3e-0)
    # ax0.set_ylim(3e-7, 0.6e-0)
    if std_log_error:
        y_label = LABEL_MEAN_LOG
    else:
        y_label = LABEL_STD

    y_label = y_label.replace('\n', '')

    ax0.set_ylabel(y_label)
    if inverse:
        ax2.legend(loc='upper center', prop=dict(size=fs))
    else:
        ax2.legend(loc='lower center')
    # ax0.legend(**new_kwargs, loc='upper center')

    # lower plots
    plt.subplots_adjust(wspace=0.4, hspace=0.45)

    plt.savefig(
        'notebooks/plots/figureA4.pdf', bbox_inches='tight',
        pad_inches=0.01)

    return fig, axs


def figureA5(vis, std_log_error=False):
    if std_log_error:
        plot_args = dict(
            lw=0.5, error_type='std_log_error', scale='manual log',
            errors='log mean', lwfill=0)
    else:
        plot_args = dict()

    fig = plt.figure(
        figsize=(textwidth, 0.9*3.8/3*2-0.05),)

    gs = fig.add_gridspec(1, 1)
    ax0 = fig.add_subplot(gs[0, 0])
    axs = [ax0]

    labels = [LABEL_RANDOM, 'Mutual Information', 'Predictive Entropy']

    acquisition_risks = [
        ['RandomAcquisition', 'BiasedRiskEstimator'],
        ['BNNClassifierAcquisitionMI', 'FancyUnbiasedRiskEstimator'],
        ['ClassifierAcquisitionEntropy', 'FancyUnbiasedRiskEstimator'],
    ]

    if not std_log_error:
        new_order = [1, 2]
        acquisition_risks = [acquisition_risks[i] for i in new_order]
        labels = [labels[i] for i in new_order]

    plot_log_convergence(
        vis,
        acquisition_risks,
        labels=labels,
        fig=fig,
        ax=axs[0],
        zorder=-100,
        # rolling=10,
        with_errors=True,
        **plot_args
    )

    # axs[0].set_title('Radial BNN on MNIST', fontsize=label_fs)
    axs[0].set_xscale('linear')
    axs[0].set_xticks([1, 100, 200, 300, 400])
    # axs[0].set_xticklabels([1, None, 200, None, 400])
    axs[0].set_xlim(0.01, 400)
    if std_log_error:
        axs[0].set_ylim(np.log10(7e-5), np.log10(5e-2))
        axs[0].set_ylabel(LABEL_MEAN_LOG)
        axs[0].set_yticks([-2, -3, -4])
    else:
        axs[0].set_ylim(7e-5, 5e-2)
        axs[0].set_ylabel(LABEL_STD)
        axs[0].set_yticks([1e-3, 1e-2])
    # axs[0].yaxis.set_major_formatter(OOMFormatter(-2, "%1.0f"))
    # axs[0].yaxis.set_minor_formatter(OOMFormatter(-2, "%1.0f"))
    # axs[0].ticklabel_format(axis='y', style='sci', scilimits=(-4,-4))

        for tick in axs[0].yaxis.get_minor_ticks():
            tick.label1.set_visible(False)

    axs[0].legend()
    axs[0].set_xlabel(acquisition_step_label)

    plt.tight_layout()

    plt.savefig(
        'notebooks/plots/figureA5.pdf', bbox_inches='tight',
        pad_inches=0.01)

    return fig, axs


"""Graveyard."""


def figure5_swapaxes(visualisers, names):

    inverse = True
    with_errors = False
    swapaxes = True

    fig3 = plt.figure(figsize=(linewidth, 3), constrained_layout=True)
    gs = fig3.add_gridspec(2, 3, height_ratios=[0.6, 1])
    ax0 = fig3.add_subplot(gs[0, 0])
    ax1 = fig3.add_subplot(gs[0, 1], sharey=ax0)
    ax2 = fig3.add_subplot(gs[0, 2], sharey=ax0)
    row = [ax0, ax1, ax2]

    ax3 = fig3.add_subplot(gs[1, :])
    ax3.grid(b=True, which='major', linestyle='-', alpha=0.5, axis='y')
    # ax3.grid(b=True, which='minor', linestyle='-', alpha=0.5)

    acquisition_risks = [
        ['RandomAcquisition', 'BiasedRiskEstimator'],
        ['ClassifierAcquisitionEntropy', 'FancyUnbiasedRiskEstimator'],
        ]

    labels = [LABEL_RANDOM, 'Active Testing']

    # plot data into first rows
    off = 2
    viss = visualisers[off:-1]
    colsdata = [pal[1], pal[0], pal[-2]]
    for i in range(len(viss)):
        cols = [pal[3], pal[2]]
        if inverse:
            pos = 2-i
        else:
            pos = 1

        acquisition_risks[1][0] = acquisitions[i+off]
        plot_log_convergence(
            viss[i], acquisition_risks, fig=fig, ax=row[pos], labels=labels,
            colors=cols, errors='percentiles', with_errors=with_errors,
            swapaxes=True)
        lab = names[i+off].split(' ')[1]
        row[pos].set_title(lab, fontsize=fs, color=colsdata[i])

    # now plot equivalent samples into the other row
    # # first, just plot the data
    colsdata2 = [pal[2], None, pal[1], pal[0], pal[-2], pal[-1]]

    # # new visualiser order, 
    if inverse:
        new_order = [0, 1, 2, 5, 3, 4]
        new_order = new_order[::-1] # legend order
        plot_equivalent_samples(
            visualisers[0], [acquisition_risks[0]], fig=fig, ax=ax3,
            labels=[labels[0]],
            errors='percentiles', inverse=inverse)

    else:
        new_order = [0, 1, 2, 3, 5, 4]

    for i in new_order:
        vis = visualisers[i]
        if i < 2:
            continue

        acq_risk = [acquisitions[i], 'FancyUnbiasedRiskEstimator']
        lab = [' '.join(names[i].split(' ')[1:])]
        print(i, lab, acquisitions[i], vis.acquisitions, acq_risk)
        plot_equivalent_samples(
            vis, [acq_risk], fig=fig, ax=ax3, labels=lab,
            colors=[colsdata2[i]], errors='percentiles',
            inverse=inverse)

    if not inverse:
        # now also plot random
        plot_equivalent_samples(
            visualisers[0], [acquisition_risks[0]], fig=fig, ax=ax3,
            labels=[labels[0]],
            errors='percentiles', inverse=inverse)

    for ax in row[1:]:
        ax.yaxis.set_tick_params(labelleft=False)

    # upper plots
    for ax in row:
        # ax.set_xlim(0, 500)
        # ax.set_xlim(0, 1000)
        # ax.set_xscale('linear')
        # ax.set_xticks([1, 200, 400])
        # ax.set_xticklabels([1, 200, 400])
        ax.set_yticks([1e-1, 1e-3, 1e-5])

        ax.set_xscale('log')
        ax.set_yscale('linear')
        ax.invert_xaxis()
        ax.set_yticks([1, 100, 200])
    #     ax.set_xticks([1e-3, 1e-4, 1e-5])

    ax1.set_xlabel('Median Squared Error')

    # ax0.set_ylim(0, 400)
    # ax0.set_xlim(.1e-0, 1e-3)
    # ax1.set_xlim(.8e-1, 3e-4)
    # ax2.set_xlim(2e-2, 1e-4)

    ax0.set_ylim(0, 200)
    ax0.set_xlim(0.08, 3e-3)
    ax0.set_xticks([1e-1, 1e-2])

    ax1.set_xlim(.8e-1, 3e-4)
    ax1.set_xticks([1e-1, 1e-2, 1e-3])
    ax2.set_xlim(2e-2, .5e-4)
    ax2.set_xticks([1e-2, 1e-3, 1e-4])

    ax0.set_ylabel('No. Points \n Acquired', labelpad=9)
    # if inverse:
    #     ax2.legend(**new_kwargs, loc='upper right')
    # else:
    leg = ax0.legend(**new_kwargs, loc='upper left', framealpha=1, facecolor='white')
    leg.set_zorder(10000)

    # ax0.legend(**new_kwargs, loc='upper center')

    # lower plots
    ax3.legend(**new_kwargs, loc='upper left', facecolor='white', framealpha=1)
    ax3.set_xlim(0, 200)
    ax3.set_xticks([1, 50, 100, 150])

    ax3.set_xlabel(acquisition_step_label)
    # ax3.set_yscale('log')

    if not inverse:
        ticks = np.array([1, 2, 4, 10, 20])
        ax3.set_yticks(ticks)
        ax3.set_yticklabels(ticks)
        ax3.set_ylim(0.7, 25)
        ax3.set_ylabel(sample_efficiency_label, labelpad=17)

    if inverse:
        # for tick in ax3.yaxis.get_minor_ticks():
        #     tick.set_visible(False)
        # for tick in ax3.yaxis.get_major_ticks():
        #     tick.set_visible(False)
        # ticks = np.array([0.1, 0.25, 0.5, 1])
        ax3.set_ylim(1/25, 1.3)
        ax3.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax3.set_yticklabels([0, '', 0.5, '', 1])
        ax3.set_ylabel('Relative \n Labelling Cost', labelpad=12)
        ax3.grid(False)
    #     ax3.yaxis.set_major_formatter(OOMFormatter(0, "%1.1f"))
    #     ax3.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        ax3.grid(True, axis='y')

    ax0.text(
        -0.71, 1.16, '(a)', fontsize=label_fs, c='k',
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax0.transAxes)

    ax3.text(
        -0.235, 1.05, '(b)', fontsize=label_fs, c='k',
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax3.transAxes)

    plt.subplots_adjust(wspace=0, hspace=0.45)

    plt.savefig(
        'notebooks/plots/big_data_log_conv_swapped_axes.pdf',
        bbox_inches='tight', pad_inches=0.01)

    return fig, _


acquisition_risks_to_color = dict(
    RandomAcquisition_BiasedRiskEstimator=pal[3],
    GPAcquisitionUncertainty_FancyUnbiasedRiskEstimator=pal[2],
    GPAcquisitionUncertainty_with_clip_FancyUnbiasedRiskEstimator=pal[2],
    TrueLossAcquisition_FancyUnbiasedRiskEstimator=pal[4],
    GPSurrogateAcquisitionMSE_FancyUnbiasedRiskEstimator=pal[2],
    GPSurrogateAcquisitionMSE_with_clip_FancyUnbiasedRiskEstimator=pal[2],
    RandomForestClassifierSurrogateAcquisitionEntropy_RFHighCapacity_FancyUnbiasedRiskEstimator=pal[2],
    RandomForestClassifierSurrogateAcquisitionEntropy_RFInfDepth_FancyUnbiasedRiskEstimator=pal[1],
    ClassifierAcquisitionEntropy_FancyUnbiasedRiskEstimator=pal[2],
    SelfSurrogateAcquisitionEntropy_FancyUnbiasedRiskEstimator=pal[0],
    SelfSurrogateAcquisitionEntropy_LazySurr_FancyUnbiasedRiskEstimator=pal[-3],
    AnySurrogateAcquisitionEntropy_LazySurrEnsemble_FancyUnbiasedRiskEstimator=pal[-1],
    SelfSurrogateAcquisitionEntropy_LazyTrain_FancyUnbiasedRiskEstimator=pal[-3],
    RandomForestClassifierSurrogateAcquisitionEntropy_RFInfDepthTrain_FancyUnbiasedRiskEstimator=pal[-1],
    BNNClassifierAcquisitionMI_FancyUnbiasedRiskEstimator=pal[-2],
    ClassifierAcquisitionEntropy_BiasedRiskEstimator=pal[-1],
    SawadeOptimalAcquisition_SawadeAcquisitionCfg_ImportanceWeightedRiskEstimator=pal[0],
    SawadeOptimalAcquisition_SawadeAcquisitionCfgOptimal_ImportanceWeightedRiskEstimator=pal[0],
    SawadeOptimalAcquisition_SawadeAcquisitionCfgWOReplacement_ImportanceWeightedRiskEstimator=pal[-3],
    SawadeOptimalAcquisition_SawadeAcquisitionCfgWOReplacement_FancyUnbiasedRiskEstimator=pal[-2],
    SawadeOptimalAcquisition_SawadeAcquisitionCfgWOReplacement_BiasedRiskEstimator=pal[-2],
    SawadeAcquisition_SawadeAcquisitionCfg_ImportanceWeightedRiskEstimator=pal[0],
    SawadeAcquisition_SawadeAcquisitionCfgOptimal_ImportanceWeightedRiskEstimator=pal[0],
    SawadeOptimalAcquisition_SawadeAcquisitionCfgWOReplacementOptimal_ImportanceWeightedRiskEstimator=pal[-3],
    SawadeOptimalAcquisition_SawadeAcquisitionCfgWOReplacementOptimal_FancyUnbiasedRiskEstimator=pal[-2],
    SawadeOptimalAcquisition_SawadeAcquisitionCfgWOReplacementOptimal_BiasedRiskEstimator=pal[-2],
)
