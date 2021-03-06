import os, re, sys, tarfile

# the try-except below allows urlretrieve to be used whether this is being
# called from Python2 or Python3.
try:
    from urllib import urlretrieve
except:
    from urllib.request import urlretrieve

options = sys.argv[1:]
         
# Auxiliary data downloads
# Format: [URL, file 1, file 2, ..., file to run when done]

aux_data = \
{
 #'gsm': ['http://xte.mit.edu/angelica/gsm', 
 #   'gsm.tar.gz',
 #   None],
 'haslam': ['http://lambda.gsfc.nasa.gov/data/foregrounds/haslam',\
    'lambda_haslam408_dsds.fits',\
    None],
 'guzman': ['https://bitbucket.org/ktausch/perses-dev/downloads',\
    'extrapolated_spectral_indices.pkl',\
    None]
}

if not os.path.exists('input'):
    os.mkdir('input')

os.chdir('input')

files = []
if (len(options) > 0) and ('clean' not in options):
    if 'clean' in options:
        to_download = list(aux_data.keys())
        files = [None] * len(to_download)
    else:
        to_download = []
        for key in options:
            if key == 'fresh':
                continue
                
            if re.search(':', key):
                pre, post = key.split(':')
                to_download.append(pre)
                files.append(int(post))
            else:
                to_download.append(key)
                files.append(None)
                
        # In this case, get all new files
        if to_download == [] and 'fresh' in options:
            to_download = list(aux_data.keys())
            files = [None] * len(to_download)
else:
    to_download = list(aux_data.keys())
    files = [None] * len(to_download)
        
for i, direc in enumerate(to_download):
                
    if not os.path.exists(direc):
        os.mkdir(direc)
    
    os.chdir(direc)
    
    web = aux_data[direc][0]
    
    if type(files[i]) is type(None):
        fns = aux_data[direc][1:-1]
    else:
        fns = [aux_data[direc][1:-1][files[i]]]
        
    for fn in fns:
            
        if os.path.exists(fn):
            if ('fresh' in options) or ('clean' in options):
                os.remove(fn)
            else:
                continue
            
        # 'clean' just deletes files, doesn't download new ones
        if 'clean' in options:
            continue
    
        print("Downloading {0!s}/{1!s}...".format(web, fn))
        urlretrieve('{0!s}/{1!s}'.format(web, fn), fn)
        
        # If it's not a tarball, move on
        if not re.search('tar', fn):
            continue
            
        # Otherwise, unpack it
        tar = tarfile.open(fn)
        tar.extractall()
        tar.close()
    
    # Run a script [optional]
    if type(aux_data[direc][-1]) is not type(None):
        with open(aux_data[direc][-1]) as f:
            exec(f.read(), globals(), locals())
    
    os.chdir('..')

