# GP in 1D
---
##The procedures for simulating GP in 1D, random realisation, no specified point.

* set up the grid, `sites = 1:n`
* compute euclidean distances, `H = abs(loc1 - loc2)`
* compute covariance, `Sigma = (1+phiM*H)*exp(-phiM*H) # with matern kernel`
* find cholesky, `L = chol(Sigma)`
* simulate random generalisation, `r = prior + L * z # z here is random normal with mu = 0, tau = 1`
* sample with design matrix, `F = matrix(M, n) # M is number of samples to measure on the grid`
* sample from the true with measurement noise, `y = F*r + tau * randn(M)`
* predict the field covariance with measurements, `C = F*Sigma*F' + tau ** 2 * ones(M, M) # take care, otherwise, singular matrix`
* find posterior mean, `mu_post = prior + Sigma * F' * inv(C) * (y - F * prior)`
* find posterior covariance, `cov_post = Sigma - Sigma * F' * inv(C) * F * Sigma`




