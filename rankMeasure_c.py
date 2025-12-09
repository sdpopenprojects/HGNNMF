import numpy as np


# effort-aware performance measures
def rank_measure(predict_label, effort, test_label):
    length = len(test_label)
    if 0 in effort:
        # for avoiding effort has zero
        effort = effort + 1

    # defective
    idx_def = list(np.where(predict_label >= 0.5))[0]
    predict_label_def = predict_label[idx_def]
    effort_def = effort[idx_def]
    p_def = predict_label_def / effort_def
    test_label_def = test_label[idx_def]

    # combining defective
    data_def = np.zeros(shape=(len(predict_label_def), 3))
    data_def[:, 0] = p_def
    data_def[:, 1] = effort_def
    data_def[:, 2] = test_label_def
    data_def = sorted(data_def, key=lambda x: (-x[0]))
    data_def = np.array(data_def)

    # nondefective
    idx_nodef = list(set(range(length)) - set(idx_def))
    predict_label_nodef = predict_label[idx_nodef]
    effort_nodef = effort[idx_nodef]
    p_nodef = predict_label_nodef / effort_nodef
    test_label_nodef = test_label[idx_nodef]

    # combining nondefective
    data_nodef = np.zeros(shape=(len(predict_label_nodef), 3))
    data_nodef[:, 0] = p_nodef
    data_nodef[:, 1] = effort_nodef
    data_nodef[:, 2] = test_label_nodef
    data_nodef = sorted(data_nodef, key=lambda x: (-x[0]))
    data_nodef = np.array(data_nodef)

    # combining
    if data_def.size == 0:
        data = data_nodef
    elif data_nodef.size == 0:
        data = data_def
    else:
        data = np.vstack([data_def, data_nodef])

    # actual model(CBS+)
    data_mdl = data
    mdl = computeArea(data_mdl, length)

    # optimal model
    data_opt = sorted(data, key=lambda x: (-x[2], x[1]))  # x[2]:test_label  x[1]:effort
    data_opt = np.array(data_opt)
    opt = computeArea(data_opt, length)

    # worst model
    data_wst = sorted(data, key=lambda x: (x[2], -x[1]))  # x[2]:test_label  x[1]:effort
    data_wst = np.array(data_wst)
    wst = computeArea(data_wst, length)

    if opt - wst != 0:
        Popt = 1 - (opt - mdl) / (opt - wst)
    else:
        Popt = 0.5

    cErecall, cEprecision, cEfmeasure, cPMI, cIFA = computeMeasure(data_mdl, length)

    return Popt, cErecall, cEprecision, cEfmeasure, cPMI, cIFA


def computeMeasure(data, length):
    cumXs = np.cumsum(data[:, 1])  # effort
    cumYs = np.cumsum(data[:, 2])  # test_label
    Xs = cumXs / cumXs[length - 1]  # percent of effort

    idx = np.min(np.where(Xs >= 0.2))
    # pos=idx
    pos = idx + 1

    Erecall = cumYs[idx] / cumYs[length - 1]

    Eprecision = cumYs[idx] / pos

    if Erecall + Eprecision != 0:
        Efmeasure = 2 * Erecall * Eprecision / (Erecall + Eprecision)
    else:
        Efmeasure = 0

    PMI = pos / length

    # Iidx = next(iter(np.where(cumYs == 1)[0]), -1)
    Iidx = np.min(np.where(cumYs >= 1))
    IFA = Iidx + 1

    return Erecall, Eprecision, Efmeasure, PMI, IFA


def computeArea(data, length):
    data = np.array(data)
    cumXs = np.cumsum(data[:, 1])  # effort
    cumYs = np.cumsum(data[:, 2])  # test_label

    Xs = cumXs / cumXs[length - 1]
    Ys = cumYs / cumYs[length - 1]

    # Use the trapezoidal rule to calculate the area under the curve
    fix_subareas = [0] * len(Ys)
    fix_subareas[0] = 0.5 * Ys[0] * Xs[0]
    for i in range(1, len(Ys)):
        fix_subareas[i] = 0.5 * (Ys[i - 1] + Ys[i]) * abs(Xs[i - 1] - Xs[i])

    area = sum(fix_subareas)

    return area
