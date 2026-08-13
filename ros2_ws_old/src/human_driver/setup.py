from setuptools import find_packages, setup

package_name = 'human_driver'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='aman',
    maintainer_email='amannindra@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            "talker = human_driver.keyboard_input:main",
            "listener = human_driver.keyboard_output:main",
            'talker_pygame = human_driver.keyboard_input_pygame:main',
        ],
    },
)
