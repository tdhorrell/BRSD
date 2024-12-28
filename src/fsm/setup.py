from setuptools import setup

package_name = 'fsm'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='brsd',
    maintainer_email='cprlsugerkitten@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'brsd_fsm_node = fsm.brsd_fsm_node:main',
            'brsd_fsm_node_correction = fsm.brsd_fsm_node_correction:main',
            'brsd_fsm_node_path = fsm.brsd_fsm_node_path:main',
            'brsd_fsm_node_metric = fsm.brsd_fsm_node_metric:main',
            'brsd_fsm_node_expo = fsm.brsd_fsm_node_expo:main'
        ],
    },
)
