from setuptools import setup, find_packages

setup(name='eschallot',                # Package name
      version='0.1.4',                         # Package version
      author='Seokhwan Min',                      # Your name
      author_email='petermsh513@gmail.com',   # Your email address
      description='Optimization tool for light-scattering multi-shell spherical particles.',  # Short description
      long_description=open('README.md').read(),  # Reads the long description from README.md
      long_description_content_type='text/markdown',
      url='https://github.com/apmd-lab/Eschallot',  # URL to the package's GitHub repo
      packages=find_packages(exclude=['runfiles','results','test_features']),  # Automatically find packages; exclude tests/docs directories
      classifiers=['Development Status :: 4 - Beta',
                   'Intended Audience :: Science/Research',
                   'Programming Language :: Python :: 3',
                   'License :: OSI Approved :: GNU General Public License v3 (GPLv3)', 
                   'Operating System :: Unix',
                   'Topic :: Scientific/Engineering :: Physics',
                   ],

      # Specify additional files to include within the package
      package_data={'eschallot': ['material_data/*.txt'],
                    },
      include_package_data=True,  # Includes files specified in MANIFEST.in, if present
      )