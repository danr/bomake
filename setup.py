from setuptools import setup

requirements = '''
    metrohash
'''

console_scripts = '''
'''

setup(
    name='bomake',
    py_modules=['bomake'],
    version='0.1',
    description='A make system',
    url='https://github.com/danr/bomake',
    author='Dan Rosén',
    author_email='danr42@gmail.com',
    python_requires='>=3.7',
    license='MIT',
    install_requires=requirements.split(),
    entry_points={'console_scripts': console_scripts.split()},
)
