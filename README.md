# saturn_moons

from astropy.io import fits
data = fits.getdata("myimage.fits").astype(float)
hdr = fits.getheader("myimage.fits")
fits.writeto("myimage_float.fits", data, hdr, overwrite=True)
