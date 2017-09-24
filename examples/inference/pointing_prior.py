import numpy as np
import matplotlib.pyplot as pl
import healpy as hp
from perses.inference.PointingPrior import UniformPointingPrior,\
    GaussianPointingPrior

nsiden = 5
nside = 2 ** nsiden
npix = hp.pixelfunc.nside2npix(nside)

pointing_center = (0, 0)
#prior = UniformPointingPrior(pointing_center=pointing_center, high_theta=np.radians(90))
prior = GaussianPointingPrior(pointing_center=pointing_center, sigma=10,\
    degrees=True)
ndraws = int(1e6)
draws = np.array([prior.draw() for idraw in range(ndraws)])
lons = draws[:,1]
lats = draws[:,0]
alpha = 0.1

(data_lons, data_lats) =\
    hp.pixelfunc.pix2ang(nside, np.arange(npix), lonlat=True)
log_prior_values = np.array([prior.log_prior((data_lats[i], data_lons[i]))\
                                                         for i in range(npix)])
prior_values = np.exp(log_prior_values)
hp.mollview(prior_values, title='Prior')

pixels_of_draws = hp.pixelfunc.ang2pix(nside, lons, lats, lonlat=True)
histogram = np.zeros(npix, dtype=int)
unique, counts = np.unique(pixels_of_draws, return_counts=True)
for ipixel in range(len(unique)):
    histogram[unique[ipixel]] = counts[ipixel]
histogram = histogram / (4 * np.pi * np.mean(histogram))
hp.mollview(histogram, title='Histogram of draws')

hp.mollview(prior_values - histogram, 'Prior - Histogram')
#hp.projscatter(lons, lats, lonlat=True, alpha=alpha, color='b', marker='.',\
#    s=1)
#circle_lats = np.ones(360) * 89
#circle_lons = np.arange(360)
#rotated_circle_lons, rotated_circle_lats =\
#    prior.rotator(circle_lons, circle_lats, lonlat=True)
#hp.projplot(rotated_circle_lons, rotated_circle_lats, lonlat=True, color='k')
#circle_lats = np.ones(360) * 85
#circle_lons = np.arange(360)
#rotated_circle_lons, rotated_circle_lats =\
#    prior.rotator(circle_lons, circle_lats, lonlat=True)
#hp.projplot(rotated_circle_lons, rotated_circle_lats, lonlat=True, color='k')

print(prior.to_string())

pl.figure()
pl.hist(lons, bins=10)
pl.title('Longitude distribution')
pl.figure()
pl.hist(lats, bins=10)
pl.title('Latitude distribution')

pl.show()
