import torch
import numpy as np

from tqdm import tqdm
from scipy import linalg


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_activation_statistics(dataloader, model, classifier):
    classifier.eval()
    model.eval()
    device = next(model.parameters()).device

    real_act, fake_act = [], []

    for x_real, label_org in tqdm(dataloader):
        rand_idx = torch.randperm(label_org.size(0))
        label_trg = label_org[rand_idx]

        x_real = x_real.to(device)
        label_trg = label_trg.to(device)
        fake_image = model(x_real, label_trg)

        real_act.append(classifier(x_real).cpu().numpy())
        fake_act.append(classifier(fake_image).cpu().numpy())

    real_act = np.concatenate(real_act, axis=0)
    fake_act = np.concatenate(fake_act, axis=0)

    return np.mean(real_act, axis=0), np.cov(real_act.T), np.mean(fake_act, axis=0), np.cov(fake_act.T)


@torch.no_grad()
def calculate_fid(dataloader, model, classifier):
    m1, s1, m2, s2 = calculate_activation_statistics(dataloader, model, classifier)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value.item()
