import logging

import numpy as np


def readable_name(drug_name):
    if drug_name == 'N-(2,3-dihydroxypropyl)-1-((2-fluoro-4-iodophenyl)amino)isonicotinamide':  # noqa
        drug_name = 'Pimasertib'
    elif drug_name == 'AZD 6244':
        drug_name = 'Selumetinib'
    elif drug_name == 'HM781-36B':
        drug_name = 'Poziotinib'
    elif drug_name == 'ARRY-334543':
        drug_name = 'Varlitinib'
    elif drug_name == 'EKB 569':
        drug_name = 'Pelitinib'

    return drug_name.capitalize()


def get_logger(log_file=None, log_level=logging.INFO, stream=True):

    logger = logging.getLogger(__file__)
    handlers = []
    if stream:
        stream_handler = logging.StreamHandler()
        handlers.append(stream_handler)

    if log_file is not None:
        file_handler = logging.FileHandler(str(log_file), 'w')
        handlers.append(file_handler)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    logger.setLevel(log_level)

    return logger


class Model:
    def __init__(self, model, word_to_id) -> None:
        self.model = model
        self.word_to_id = word_to_id
        self.shape = model.shape
        self.keys = set(word_to_id.keys())

    def __getitem__(self, word):
        id_ = self.word_to_id[word]
        return self.model[id_]

    def __contains__(self, word):
        return word in self.keys

    def items(self):
        for word, id_ in self.word_to_id.items():
            yield (word, self.model[id_])


def calc_acc(rows, top):
    tots = []
    for row in rows:
        ans_cpts = set(row['ans_cpts'])
        c = [1]*top
        for k in range(top):
            cand_cpt = row['cand_cpts'][k]
            if cand_cpt in ans_cpts:
                break
            else:
                c[k] = 0
        tots.append(c)
    accs = np.mean(np.array(tots), axis=0)
    return list(accs)


def calc_mrr(rows):
    return np.mean([row['rr'] for row in rows])


def get_year_result_row(year, query_direction, method, acc, mrr,
                        weighted_mean=None, centering=None):
    result_row = {'year': year,
                  'query_direction': query_direction,
                  'method': method,
                  'weighted_mean': weighted_mean,
                  'centering': centering}
    for ith, acc_ in enumerate(acc):
        result_row[f'top{ith+1}_acc'] = acc_
    result_row['mrr'] = mrr
    return result_row


class A:
    # cal set
    cal_D = r'$|\mathcal{D}|$'
    cal_G = r'$|\mathcal{G}|$'
    cal_R = r'$|\mathcal{R}|$'

    # set
    D = r'$|D|$'
    G = r'$|G|$'

    # quotient set
    E_d = r'$\mathrm{E}_{d\in D}\{|[d]|\}$'
    E_g = r'$\mathrm{E}_{g\in G}\{|[g]|\}$'

    all = [cal_D, cal_G, cal_R, D, G, E_d, E_g]
    Es = [E_d, E_g]

    @staticmethod
    def stats(key, query_direction):

        idx = int(query_direction == 'gene2drug')

        # cal set
        cal = [A.cal_D, A.cal_G]

        # set
        DG = [A.D, A.G]

        # quotient set
        E = [A.E_d, A.E_g]

        if key == 'cal_D':
            return cal[idx]
        elif key == 'cal_G':
            return cal[1-idx]
        elif key == 'cal_R':
            return A.cal_R

        elif key == 'D':
            return DG[idx]
        elif key == 'G':
            return DG[1-idx]

        elif key == 'E_d':
            return E[idx]

        raise ValueError(f'key: {key}')


class P:
    # cal set
    cal_P = r'$|\mathcal{P}|$'

    # set
    sum_Dp = r'$\sum_{p\in\mathcal{P}}|D_p|$'
    sum_Gp = r'$\sum_{p\in\mathcal{P}}|G_p|$'

    # cal cap set
    sum_cal_cap_Dp = r'$\sum_{p\in\mathcal{P}}|\mathcal{D}_p\cap D|$'
    sum_cal_cap_Gp = r'$\sum_{p\in\mathcal{P}}|\mathcal{G}_p\cap G|$'

    # quotient set
    E_dp = \
        r'$\mathrm{E}_{p\in\mathcal{P}}\{\mathrm{E}_{d\in D_p}\{|[d]_p|\}\}$'
    E_gp = \
        r'$\mathrm{E}_{p\in\mathcal{P}}\{\mathrm{E}_{g\in G_p}\{|[g]_p|\}\}$'
    E_d = r'$\mathrm{E}_{p\in\mathcal{P}}\{\mathrm{E}_{d\in \mathcal{D}_p\cap D}\{|[d]|\}\}$'  # noqa
    E_g = r'$\mathrm{E}_{p\in\mathcal{P}}\{\mathrm{E}_{g\in \mathcal{G}_p\cap G}\{|[g]|\}\}$'  # noqa

    all = [cal_P,
           sum_Dp, sum_Gp,
           sum_cal_cap_Dp, sum_cal_cap_Gp,
           E_dp, E_gp, E_d, E_g]
    Es = [E_dp, E_gp, E_d, E_g]

    @staticmethod
    def stats(key, query_direction):
        idx = int(query_direction == 'gene2drug')

        # set
        sum_ = [P.sum_Dp, P.sum_Gp]

        # cal cap set
        sum_cal_cap = [P.sum_cal_cap_Dp, P.sum_cal_cap_Gp]

        # quotient set
        E_p = [P.E_dp, P.E_gp]
        E = [P.E_d, P.E_g]

        if key == 'cal_P':
            return P.cal_P

        elif key == 'sum_Dp':
            return sum_[idx]

        elif key == 'sum_Gp':
            return sum_[1-idx]

        elif key == 'sum_cal_cap_Dp':
            return sum_cal_cap[idx]

        elif key == 'sum_cal_cap_Gp':
            return sum_cal_cap[1-idx]

        elif key == 'E_dp':
            return E_p[idx]
        elif key == 'E_d':
            return E[idx]

        raise ValueError(f'key: {key}')


class Y1:
    # cal set
    cal_D = r'$|\mathcal{D}^y|$'
    cal_G = r'$|\mathcal{G}^y|$'
    cal_R = r'$|\mathcal{R}^y|$'

    # set
    D = r'$|D^y|$'
    G = r'$|G^y|$'

    # quotient set
    E_d = r'$\mathrm{E}_{d\in D^y}\{|[d]|\}$'
    E_g = r'$\mathrm{E}_{g\in G^y}\{|[g]|\}$'

    all = [cal_D, cal_G, cal_R, D, G, E_d, E_g]
    Es = [E_d, E_g]

    @staticmethod
    def stats(key, query_direction):

        idx = int(query_direction == 'gene2drug')

        # cal set
        cal = [Y1.cal_D, Y1.cal_G]

        # set
        DG = [Y1.D, Y1.G]

        # quotient set
        E = [Y1.E_d, Y1.E_g]

        if key == 'cal_D':
            return cal[idx]
        elif key == 'cal_G':
            return cal[1-idx]
        elif key == 'cal_R':
            return Y1.cal_R

        elif key == 'D':
            return DG[idx]
        elif key == 'G':
            return DG[1-idx]

        elif key == 'E_d':
            return E[idx]

        raise ValueError(f'key: {key}')


class Y2:
    # cal set
    cal_R_L = r'$\left|\mathcal{R}^{y \mid L_y}\right|$'
    cal_R_U = r'$\left|\mathcal{R}^{y \mid U_y}\right|$'

    # set
    D_U = r'$\left|D^{y\mid U_y}\right|$'
    G_U = r'$\left|G^{y\mid U_y}\right|$'

    # quotient set
    E_d_L = r'$\mathrm{E}_{d\in D^{y\mid L_y}}\left\{\left|[d]^{y\mid L_y}\right|\right\}$'  # noqa
    E_g_L = r'$\mathrm{E}_{g\in G^{y\mid L_y}}\left\{\left|[g]^{y\mid L_y}\right|\right\}$'  # noqa
    E_d_U = r'$\mathrm{E}_{d\in D^{y\mid U_y}}\left\{\left|[d]^{y\mid U_y}\right|\right\}$'  # noqa
    E_g_U = r'$\mathrm{E}_{g\in G^{y\mid U_y}}\left\{\left|[g]^{y\mid U_y}\right|\right\}$'  # noqa

    all = [cal_R_L, cal_R_U,
           D_U, G_U,
           E_d_L, E_d_U, E_g_L, E_g_U]
    Es = [E_d_U, E_d_L, E_g_U, E_g_L]

    @staticmethod
    def stats(key, query_direction):

        idx = int(query_direction == 'gene2drug')

        # set
        DG_U = [Y2.D_U, Y2.G_U]

        # quotient set
        E_L = [Y2.E_d_L, Y2.E_g_L]
        E_U = [Y2.E_d_U, Y2.E_g_U]

        if key == 'cal_R_L':
            return Y2.cal_R_L
        elif key == 'cal_R_U':
            return Y2.cal_R_U

        elif key == 'D_U':
            return DG_U[idx]
        elif key == 'G_U':
            return DG_U[1-idx]

        elif key == 'E_d_L':
            return E_L[idx]
        elif key == 'E_d_U':
            return E_U[idx]

        raise ValueError(f'key: {key}')


class P1Y1_P2Y1:

    # set
    sum_Dp = r'$\sum_{p\in\mathcal{P}}|D_p^y|$'
    sum_Gp = r'$\sum_{p\in\mathcal{P}}|G_p^y|$'

    # quotient set
    E_dp = r'$\mathrm{E}_{p\in\mathcal{P}}\{\mathrm{E}_{d\in D_p^y}\{|[d]_p^y|\}\}$'  # noqa
    E_gp = r'$\mathrm{E}_{p\in\mathcal{P}}\{\mathrm{E}_{g\in G_p^y}\{|[g]_p^y|\}\}$'  # noqa
    E_d = r'$\mathrm{E}_{p\in\mathcal{P}}\{\mathrm{E}_{d\in \mathcal{D}_p^y\cap D^y}\{|[d]^y|\}\}$'  # noqa
    E_g = r'$\mathrm{E}_{p\in\mathcal{P}}\{\mathrm{E}_{g\in \mathcal{G}_p^y\cap G^y}\{|[g]^y|\}\}$'  # noqa

    # cal cap set
    sum_cal_cap_Dp = r'$\sum_{p\in\mathcal{P}}|\mathcal{D}_p^y\cap D^y|$'
    sum_cal_cap_Gp = r'$\sum_{p\in\mathcal{P}}|\mathcal{G}_p^y\cap G^y|$'

    all = [sum_Dp, sum_Gp,
           sum_cal_cap_Dp, sum_cal_cap_Gp,
           E_dp, E_gp, E_d, E_g]
    Es = [E_d, E_g, E_dp, E_gp]

    @staticmethod
    def stats(key, query_direction):
        idx = int(query_direction == 'gene2drug')

        # set
        sum_ = [P1Y1_P2Y1.sum_Dp, P1Y1_P2Y1.sum_Gp]

        # cal cap set
        sum_cal_cap = [P1Y1_P2Y1.sum_cal_cap_Dp, P1Y1_P2Y1.sum_cal_cap_Gp]

        # quotient set
        E = [P1Y1_P2Y1.E_d, P1Y1_P2Y1.E_g]
        E_p = [P1Y1_P2Y1.E_dp, P1Y1_P2Y1.E_gp]

        if key == 'sum_Dp':
            return sum_[idx]
        elif key == 'sum_Gp':
            return sum_[1-idx]

        elif key == 'sum_cal_cap_Dp':
            return sum_cal_cap[idx]

        elif key == 'E_d':
            return E[idx]
        elif key == 'E_dp':
            return E_p[idx]

        raise ValueError(f'key: {key}')


class P1Y2_P2Y2:
    # set
    sum_Dp_U = r'$\sum_{p\in\mathcal{P}}\left|D_p^{y \mid U_y}\right|$'
    sum_Gp_U = r'$\sum_{p\in\mathcal{P}}\left|G_p^{y \mid U_y}\right|$'

    # cal cap set
    sum_cal_cap_Dp_U = r'$\sum_{p\in\mathcal{P}}|\mathcal{D}_p^y\cap D^{y\mid U_y}|$'  # noqa
    sum_cal_cap_Gp_U = r'$\sum_{p\in\mathcal{P}}|\mathcal{G}_p^y\cap G^{y\mid U_y}|$'  # noqa

    # quotient set
    E_dp_L = r'$\mathrm{E}_{p\in\mathcal{P}}\left\{\mathrm{E}_{d\in D_p^{y \mid L_y}}\left\{\left|[d]_p^{y\mid L_y}\right|\right\}\right\}$'  # noqa
    E_gp_L = r'$\mathrm{E}_{p\in\mathcal{P}}\left\{\mathrm{E}_{g\in G_p^{y \mid L_y}}\left\{\left|[g]_p^{y\mid L_y}\right|\right\}\right\}$'  # noqa
    E_dp_U = r'$\mathrm{E}_{p\in\mathcal{P}}\left\{\mathrm{E}_{d\in D_p^{y \mid U_y}}\left\{\left|[d]_p^{y\mid U_y}\right|\right\}\right\}$'  # noqa
    E_gp_U = r'$\mathrm{E}_{p\in\mathcal{P}}\left\{\mathrm{E}_{g\in G_p^{y \mid U_y}}\left\{\left|[g]_p^{y\mid U_y}\right|\right\}\right\}$'  # noqa
    E_d_L = r'$\mathrm{E}_{p\in\mathcal{P}}\left\{\mathrm{E}_{d\in \mathcal{D}_p^y\cap D^{y\mid L_y}}\left\{\left|[d]^{y\mid L_y}\right|\right\}\right\}$'  # noqa
    E_g_L = r'$\mathrm{E}_{p\in\mathcal{P}}\left\{\mathrm{E}_{g\in \mathcal{G}_p^y\cap G^{y\mid L_y}}\left\{\left|[g]^{y\mid L_y}\right|\right\}\right\}$'  # noqa
    E_d_U = r'$\mathrm{E}_{p\in\mathcal{P}}\left\{\mathrm{E}_{d\in \mathcal{D}_p^y\cap D^{y\mid U_y}}\left\{\left|[d]^{y\mid U_y}\right|\right\}\right\}$'  # noqa
    E_g_U = r'$\mathrm{E}_{p\in\mathcal{P}}\left\{\mathrm{E}_{g\in \mathcal{G}_p^y\cap G^{y\mid U_y}}\left\{\left|[g]^{y\mid U_y}\right|\right\}\right\}$'  # noqa

    all = [sum_Dp_U, sum_Gp_U,
           sum_cal_cap_Dp_U, sum_cal_cap_Gp_U,
           E_dp_U, E_gp_U, E_d_U, E_g_U]
    Es = [E_dp_U, E_gp_U, E_d_U, E_g_U]

    @staticmethod
    def stats(key, query_direction):
        idx = int(query_direction == 'gene2drug')

        # set
        sum_U = [P1Y2_P2Y2.sum_Dp_U, P1Y2_P2Y2.sum_Gp_U]

        # cal cap set
        sum_cal_cap_U = [P1Y2_P2Y2.sum_cal_cap_Dp_U,
                         P1Y2_P2Y2.sum_cal_cap_Gp_U]

        # quotient set
        E_d_L = [P1Y2_P2Y2.E_d_L, P1Y2_P2Y2.E_g_L]
        E_d_U = [P1Y2_P2Y2.E_d_U, P1Y2_P2Y2.E_g_U]
        E_dp_L = [P1Y2_P2Y2.E_dp_L, P1Y2_P2Y2.E_gp_L]
        E_dp_U = [P1Y2_P2Y2.E_dp_U, P1Y2_P2Y2.E_gp_U]

        if key == 'sum_Dp_U':
            return sum_U[idx]

        elif key == 'sum_cal_cap_Dp_U':
            return sum_cal_cap_U[idx]

        elif key == 'E_d_L':
            return E_d_L[idx]
        elif key == 'E_d_U':
            return E_d_U[idx]
        elif key == 'E_dp_L':
            return E_dp_L[idx]
        elif key == 'E_dp_U':
            return E_dp_U[idx]

        raise ValueError(f'key: {key}')


expectations = A.Es + P.Es + Y1.Es + Y2.Es + P1Y1_P2Y1.Es + P1Y2_P2Y2.Es
