# ref: https://github.com/laituan245/image-segmentation-GMM

from PIL import Image
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

COLORS = [
    (255, 0, 0),   # red
    (0, 255, 0),  # green
    (0, 0, 255),   # blue
    (255, 255, 0),  # yellow
    (255, 0, 255),  # magenta
]


class GaussianMixtureModel:

    def __init__(self, ncomp, initial_mus, initial_covs, initial_priors, epsilon=1e-10):
        self.ncomp = ncomp
        self.mus = np.asarray(initial_mus)
        self.covs = np.asarray(initial_covs)
        self.priors = np.asarray(initial_priors)
        self.epsilon = epsilon

    def e_step(self, datas):
        unnormalized_probs = []
        for i in range(self.ncomp):
            mu, cov, prior = self.mus[i, :], self.covs[i, :, :], self.priors[i]
            unnormalized_prob = prior * \
                multivariate_normal.pdf(datas, mean=mu, cov=cov)
            unnormalized_probs.append(np.expand_dims(unnormalized_prob, -1))
        preds = np.concatenate(unnormalized_probs, axis=1)
        log_likelihood = np.sum(preds, axis=1)
        log_likelihood = np.sum(np.log(log_likelihood))

        preds = preds / np.sum(preds, axis=1, keepdims=True)
        return np.asarray(preds), log_likelihood

    def m_step(self, datas, beliefs):
        new_mus, new_covs, new_priors = [], [], []
        soft_counts = np.sum(beliefs, axis=0)
        for i in range(self.ncomp):
            new_mu = np.sum(np.expand_dims(beliefs[:, i], -1) * datas, axis=0)
            new_mu /= soft_counts[i]
            new_mus.append(new_mu)

            data_shifted = np.subtract(datas, np.expand_dims(new_mu, 0))
            new_cov = np.matmul(np.transpose(np.multiply(
                np.expand_dims(beliefs[:, i], -1), data_shifted)), data_shifted)
            new_cov /= soft_counts[i]
            new_covs.append(new_cov)

            new_priors.append(soft_counts[i] / np.sum(soft_counts))

        self.mus = np.asarray(new_mus)
        self.covs = np.asarray(new_covs)
        self.priors = np.asarray(new_priors)

    def fit(self, data):
        prev_ll = None
        for i in range(1000):
            beliefs, log_likelihood = self.e_step(data)  # e-step
            self.m_step(data, beliefs)  # m-step
            print("Log Likelihood:{} at iteration {}".format(log_likelihood, i))
            if prev_ll != None and abs(log_likelihood - prev_ll) < self.epsilon:
                break
            prev_ll = log_likelihood


def load_image(infilename):
    img = Image.open(infilename)
    img.load()
    data = np.asarray(img, dtype="int32")
    return data


image_name = '33'
image_path = './data/train/{}.png'.format(image_name)
image = load_image(image_path)
image_height, image_width, image_channels = image.shape
image_pixels = np.reshape(image, (-1, image_channels))
plt.imshow(image)
plt.show()

ncomp = 2

# Apply K-Means to find the initial weights and covariance matrices for GMM
kmeans = KMeans(n_clusters=ncomp)
labels = kmeans.fit_predict(image_pixels)
initial_mus = kmeans.cluster_centers_
initial_priors, initial_covs = [], []
for i in range(ncomp):
    datas = np.array([image_pixels[j, :]
                      for j in range(len(labels)) if labels[j] == i]).T
    initial_covs.append(np.cov(datas))
    initial_priors.append(datas.shape[1] / float(len(labels)))

gmm = GaussianMixtureModel(ncomp, np.asarray(initial_mus), np.asarray(
    initial_covs), np.asarray(initial_priors))
gmm.fit(image_pixels)

beliefs, log_likelihood = gmm.e_step(image_pixels)
map_beliefs = np.reshape(beliefs, (image_height, image_width, ncomp))
segmented_map = np.zeros((image_height, image_width, 3))
for i in range(image_height):
    for j in range(image_width):
        hard_belief = np.argmax(map_beliefs[i, j, :])
        segmented_map[i, j, :] = np.asarray(COLORS[hard_belief]) / 255.0
plt.imshow(segmented_map)
plt.show()
