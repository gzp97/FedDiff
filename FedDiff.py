import numpy as np
from tqdm import trange, tqdm
import tensorflow as tf
import time
from copy import deepcopy

from .fedbase import BaseFedarated
from flearn.utils.tf_utils import process_grad, cosine_sim, softmax, norm_grad
from flearn.utils.model_utils import batch_data, gen_batch, gen_epoch
from flearn.attacker.untargeted.sign_flipping import sign_flipping, sign_flipping_p
from flearn.attacker.untargeted.additive_noise import additive_noise, additive_noise_inner, additive_noise_p
from flearn.attacker.untargeted.same_value import same_value, same_value_p
from flearn.attacker.targeted.backdoor import backdoor

from flearn.trainers.VAE import VAE
from sklearn import preprocessing
from flearn.utils.geomedian import geometric_median

from sklearn.cluster import SpectralClustering, KMeans
from sklearn.decomposition import KernelPCA
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM


class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        print('Using fair fed avg to Train')
        self.inner_opt = tf.train.GradientDescentOptimizer(params['learning_rate'])
        super(Server, self).__init__(params, learner, dataset)
        self.train_vae_epochs = int(np.ceil(len(self.clients) / params['clients_per_round']))
        self.params = params
        self.learner = learner
        self.dataset = dataset

    def train(self):
        error_rates = []
        start = time.time()
        print('Training with {} workers ---'.format(self.clients_per_round))
        print(self.train_vae_epochs)

        num_clients = len(self.clients)
        pk = np.ones(num_clients) * 1.0 / num_clients

        percent_of_attacker = 0.3
        X_dim = 100
        num_attackers = round(percent_of_attacker * len(self.clients))
        attackers = self.clients[:num_attackers]
        attack_type = self.params['attack']
        print(self.byz_percent)

        Xs = []
        for i in trange(self.train_vae_epochs):

            indices, selected_clients = self.select_clients(i, pk, num_clients=self.clients_per_round)
            np.random.seed(i)
            active_clients = np.random.choice(selected_clients, round(self.clients_per_round * (1 - self.drop_percent)),
                                              replace=False)
            #
            csolns = []
            #
            selected_clients = selected_clients.tolist()
            active_clients = active_clients.tolist()

            for idx, c in enumerate(active_clients):  # simply drop the slow devices
                # communicate the latest model
                c.set_params(self.latest_model)
                weights_before = c.get_params()
                # loss = c.get_loss()  # compute loss on the whole training data, with respect to the starting point (the global model)

                soln, stats = c.solve_inner(num_epochs=self.num_epochs, batch_size=self.batch_size)
                self.metrics.update(rnd=i - 1, cid=c.id, stats=stats)
                csolns.append(soln)

            self.latest_model = self.aggregate_fedavg(csolns, self.latest_model, self.learning_rate)

            weights = []

            if self.params['dataset'] == 'mnist':
                for (w, soln) in csolns:  # w is the number of local samples
                    weight = soln[-1].flatten()
                    weights.append(np.array(weight))

            elif self.params['dataset'] == 'vehicle':
                for (w, soln) in csolns:  # w is the number of local samples
                    weight = soln[0].flatten()
                    weights.append(np.array(weight))

            else:

                for (w, soln) in csolns:  # w is the number of local samples
                    weight = []
                    for idx, v in enumerate(soln):
                        f_w = v.flatten()
                        weight = np.append(weight, f_w)
                    weight = np.array(weight).flatten()
                    weights.append(np.array(weight))



            X = np.array(weights)
            X = (X - np.mean(X)) / np.std(X)

            X_20 = []

            np.random.seed(i)
            for _ in X:
                if self.params['dataset'] == 'mnist':
                    new = _
                elif self.params['dataset'] == 'vehicle':
                    new = _
                    # print(_.shape)
                else:
                    new = np.random.choice(_, X_dim)
                # input()
                X_20.append(new)

            X_20 = np.array(X_20)

            X = X_20

            X_dim = len(X[0])

            for _ in X:
                Xs.append(_)
        Xs = (Xs - np.mean(Xs)) / np.std(Xs)


        kpca = KernelPCA(kernel='rbf', n_components= 4)

        base = []

        for _ in Xs:  # w is the number of local samples
            base.append(_.astype(np.float64))
        avs = geometric_median(np.array(base))

        X_avs = [v - avs for i, v in enumerate(Xs)]

        Xs = X_avs

        new_Xs = kpca.fit_transform(Xs)

        # clf = IsolationForest()
        clf = OneClassSVM()

        clf.fit(new_Xs)



        if attack_type == 'label':
            for i, c in enumerate(self.clients):
                if i < self.byz_percent * len(self.clients):
                    c.label()

        for i in range(self.num_rounds + 1):
            start_epoch = time.time()
            if i % self.eval_every == 0:
                num_test, num_correct_test = self.test()  # have set the latest model for all clients
                num_train, num_correct_train = self.train_error()
                num_val, num_correct_val = self.validate()
                if i == 0:
                    num_correct_test = 0
                    num_correct_train = 0
                    num_correct_val = 0
                tqdm.write('At round {} testing accuracy: {}'.format(i,
                                                                     np.sum(np.array(num_correct_test)) * 1.0 / np.sum(
                                                                         np.array(num_test))))
                tqdm.write('At round {} training accuracy: {}'.format(i, np.sum(
                    np.array(num_correct_train)) * 1.0 / np.sum(np.array(num_train))))
                tqdm.write('At round {} validating accuracy: {}'.format(i, np.sum(
                    np.array(num_correct_val)) * 1.0 / np.sum(np.array(num_val))))
                num_test_backdoor, num_correct_test_backdoor = self.test_backdoor()
                tqdm.write('At round {} backdoor testing accuracy: {}'.format(i,
                                                                              np.sum(np.array(
                                                                                  num_correct_test_backdoor)) * 1.0 / np.sum(
                                                                                  np.array(num_test_backdoor))))

            indices, selected_clients = self.select_clients(i, pk, num_clients=self.clients_per_round)
            np.random.seed(i)
            active_clients = np.random.choice(selected_clients, round(self.clients_per_round * (1 - self.drop_percent)),
                                              replace=False)

            csolns = []

            selected_clients = selected_clients.tolist()
            active_clients = active_clients.tolist()

            losses = []

            for idx, c in enumerate(active_clients):  # simply drop the slow devices
                # communicate the latest model
                c.set_params(self.latest_model)
                # weights_before = c.get_params()
                loss = c.get_loss()  # compute loss on the whole training data, with respect to the starting point (the global model)
                losses.append(loss)
                soln, stats = c.solve_inner(num_epochs=self.num_epochs, batch_size=self.batch_size)

                csolns.append(soln)

                self.metrics.update(rnd=i - 1, cid=c.id, stats=stats)

            loss_mean = np.mean(losses)
            if i % self.eval_every == 0:
                tqdm.write('At round {} loss: {}'.format(i, loss_mean))

            percent_of_attacker = self.params['percent']
            num_byz = percent_of_attacker * len(active_clients)

            attack_type = self.params['attack']
            # print(attack_type, percent_of_attacker, self.learning_rate)
            if attack_type == 'add':

                csolns = additive_noise_p(csolns, percent_of_attacker)
            elif attack_type == 'sign':

                csolns = sign_flipping_p(csolns, percent_of_attacker)
            elif attack_type == 'same':

                csolns = same_value_p(csolns, percent_of_attacker)

            elif attack_type == 'backdoor':
                num_attackers = 0
                if percent_of_attacker == 1:
                    num_attackers = 1
                else:
                    num_attackers = percent_of_attacker * len(csolns)
                csolns = backdoor(csolns, num_attackers)





            weights = []

            if self.params['dataset'] == 'mnist':
                for (w, soln) in csolns:  # w is the number of local samples
                    weight = soln[-1].flatten()
                    weights.append(np.array(weight))

            elif self.params['dataset'] == 'vehicle':
                for (w, soln) in csolns:  # w is the number of local samples
                    weight = soln[0].flatten()
                    weights.append(np.array(weight))

            else:

                for (w, soln) in csolns:  # w is the number of local samples
                    weight = []
                    for idx, v in enumerate(soln):
                        f_w = v.flatten()
                        weight = np.append(weight, f_w)
                    weight = np.array(weight).flatten()
                    weights.append(np.array(weight))



            X = np.array(weights)
            X = (X - np.mean(X)) / np.std(X)

            X_20 = []

            np.random.seed(i)
            for _ in X:
                if self.params['dataset'] == 'mnist':
                    new = _
                elif self.params['dataset'] == 'vehicle':
                    new = _
                    # print(_.shape)
                else:
                    new = np.random.choice(_, X_dim)
                # input()
                X_20.append(new)

            X_20 = np.array(X_20)

            X = X_20

            X_dim = len(X[0])

            # print(X_dim)

            base = []

            for _ in X:  # w is the number of local samples
                base.append(_.astype(np.float64))
            avs = geometric_median(np.array(base))

            X_avs = [v - avs for i, v in enumerate(X)]
            X = X_avs

            kpca = KernelPCA(kernel='rbf', n_components=4)

            X_hat = kpca.fit_transform(X)

            # pred_y = clf.predict(X_hat)

            anomly_scores = clf.decision_function(X_hat)

            mean_anomly_score = np.mean(anomly_scores)

            pred_y = []

            for ii, _ in enumerate(anomly_scores):
                if _ < mean_anomly_score:
                    pred_y.append(-1)
                else:
                    pred_y.append(1)


            old_csolns = deepcopy(csolns)
            minidx = []


            # print(pred_y)

            # input()
            for iidx, _ in enumerate(pred_y):
                if _ == 1:
                    minidx.append(iidx)


            # print(minidx)
            # input()

            benign_solns = []
            malicious_solns = []
            boc = []
            moc = []
            for iidx in range(len(old_csolns)):
                if iidx in minidx:
                    benign_solns.append((old_csolns[iidx][0], X_hat[iidx]))
                    boc.append(old_csolns[iidx])
                else:
                    malicious_solns.append((old_csolns[iidx][0], X_hat[iidx]))
                    moc.append(old_csolns[iidx])


            flag = True
            new_malicious_csolns = deepcopy(malicious_solns)
            l_b = len(benign_solns)
            while flag == True and len(benign_solns) > 1 and len(malicious_solns) > 1:
                # print(1)
                flag = False

                distance = geo_loss(benign_solns, malicious_solns)
                # distance = mmd_loss(benign_solns, malicious_solns, 1).eval()

                # print(distance)
                # print("--")
                new_distances = []
                for iidx in range(len(benign_solns)):
                    if len(benign_solns) <= 1:
                        break
                    # print(iidx)
                    nm_wsolns = deepcopy(malicious_solns)
                    nm_wsolns.append(benign_solns[iidx])
                    nb_wsolns = deepcopy(benign_solns)
                    nb_wsolns.pop(iidx)

                    new_distance = geo_loss(nb_wsolns, nm_wsolns)
                    # new_distance = mmd_loss(nb_wsolns, nm_wsolns, 1).eval()
                    new_distances.append(new_distance)
                    # print(soln)
                    if new_distance >= distance:
                        # print(new_distance, distance)
                        new_malicious_csolns.append(benign_solns[iidx])
                        flag = True
                new_benign_csolns = []
                for _ in benign_solns:
                    fflag = False
                    for b_ in new_malicious_csolns:
                        if (np.array(_[1]) == np.array(b_[1])).all():
                            fflag = True
                    if not fflag:
                        new_benign_csolns.append(_)
                malicious_solns = new_malicious_csolns
                benign_solns = new_benign_csolns

                if l_b == len(benign_solns):
                    break

            final_csolns = []

            b_csolns = []
            for w, _ in benign_solns:
                b_csolns.append(_)

            new_pred_y = []
            for iidx, _ in enumerate(X_hat):
                fflag = False
                for b_ in b_csolns:
                    if (np.array(_) == np.array(b_)).all():
                        fflag = True
                if fflag:
                    new_pred_y.append(1)
                    final_csolns.append(old_csolns[iidx])
                else:
                    new_pred_y.append(-1)

            # print(pred_y)
            # print(np.array(new_pred_y))



            if i == 0:
                losses = []
                c = active_clients[-1]
                for idx, (w, soln) in enumerate(old_csolns):
                    c.set_params(soln)
                    loss = c.get_loss()
                    losses.append(loss)
                minidx = np.argmin(losses)
                w, self.latest_model = old_csolns[minidx]
            else:
                if len(final_csolns) > 1:
                    self.latest_model = self.aggregate_geomed(final_csolns, self.latest_model, self.learning_rate)


            errors = 0
            for idx in range(len(pred_y)):
                if pred_y[idx] == -1 and new_pred_y[idx] == 1:
                    errors += 1
            error_rate = errors / len(pred_y)
            # print(error_rate)
            error_rates.append(error_rate)
        end = time.time()

        print(end - start)
        print(np.mean(error_rates))


def get_geomed(wsolns):


    from flearn.utils.geomedian import geometric_median
    # total_weight = 0.0
    base = []
    for i in range(len(wsolns[0][1])):
        base.append([])
    for (w, soln) in wsolns:  # w is the number of local samples

        for i, v in enumerate(soln):
            base[i].append(v.astype(np.float64))

    average_update = []
    for _ in base:
        avs = geometric_median(_)
        average_update.append(avs)

    geomed = [v.astype(np.float64) for i, v in
                         enumerate(average_update)]

    return geomed


def geo_loss(benign_solns, malicious_solns):
    benign_geomed = get_geomed(benign_solns)
    malicious_geomed = get_geomed(malicious_solns)
    distance = 0
    for idx in range(len(benign_geomed)):
        distance += np.linalg.norm(np.array(benign_geomed[idx]) - np.array(malicious_geomed[idx]))
    return distance

def compute_pairwise_distances(b_solns, m_solns):

    b_mat = []
    for i in range(len(b_solns)):
        b_mat.append([])
        for _ in b_solns[i][1]:
            b_mat[i].append(_)
    # print(len(b_solns))
    m_mat = []
    for i in range(len(m_solns)):
        m_mat.append([])
        for _ in m_solns[i][1]:
            m_mat[i].append(_)

    norm = lambda x: tf.reduce_sum(tf.square(x), 1)

    return tf.transpose(norm(tf.expand_dims(b_mat, 2) - tf.transpose(m_mat)))

def gaussian_kernel_matrix(b_solns, m_solns, sigmas):
    beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))

    dist = compute_pairwise_distances(b_solns, m_solns)

    s = tf.matmul(beta, tf.reshape(dist, (1,-1)))

    return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))


def mmd(b_solns, m_solns, kernel = gaussian_kernel_matrix):

    with tf.name_scope("mmd"):
        cost = tf.reduce_mean(kernel(b_solns, b_solns))
        cost += tf.reduce_mean(kernel(m_solns, m_solns))
        cost -= 2 * tf.reduce_mean(kernel(b_solns, m_solns))

        cost = tf.where(cost > 0, cost, 0, name='value')

    return cost


def mmd_loss(b_solns, m_solns, weight):

    sigmas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e5, 1e6]
    from functools import partial
    gaussian_kernel = partial(gaussian_kernel_matrix, sigmas = sigmas)

    loss_value = mmd(b_solns, m_solns, kernel=gaussian_kernel)
    loss_value = tf.maximum(1e-4, loss_value) * weight

    return loss_value