import os
import datetime
import yaml

from astropy.io import fits

def write2fits(samples, filename, setup=None):
   """
   writes recarray containing all samples to fits file
   """
   
   #-- create Table data for fits file
   cols = []
   
   for (name, fmt) in samples.dtype.descr:
      col = fits.Column(name=name, format='D', array=samples[name])
      cols.append(col)
      
   cols = fits.ColDefs(cols)
   
   tbhdu = fits.BinTableHDU.from_columns(cols)
   
   #-- add date to header
   header = tbhdu.header
   header['date'] = str(datetime.datetime.now())
   
   #-- add complete setup as comment if provided
   if setup is not None:
      if not isinstance(setup, str):
         setup = yaml.safe_dump(setup, sort_keys=False)
      for line in setup.split('\n'):
         header['comment'] = line
   
   #-- delete file if it already exists
   if os.path.isfile(filename):
      os.remove(filename)
   
   tbhdu.writeto(filename)
   
def read_fits(filename):
   """
   Reads fits file and returns samples and the settings if they exist
   """
   
   hdu = fits.open(filename)
   
   samples = hdu[1].data
   
   setup = hdu[1].header['comment']
   
   return samples, str(setup)
