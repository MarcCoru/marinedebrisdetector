from setuptools import setup, find_packages

with open('requirements.txt') as req_file:
    requirements = [req.strip() for req in req_file.read().splitlines()]

setup(name='marinedebrisdetector',
      version='0.0',
      description='Marine Debris Detector',
      url='http://github.com/marccoru/marinedebrisdetector',
      author='Marc Rußwurm',
      author_email='marc.russwurm@epfl.ch',
      license='MIT',
      packages=find_packages(),
      install_requires=requirements,
      entry_points={
            'console_scripts': [
                  'marinedebrisdetector=marinedebrisdetector:main'
            ]
      },
      zip_safe=False)
