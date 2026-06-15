from setuptools import find_packages, setup

package_name = 'stanley_controler'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    package_data={
        package_name: ['centerline.csv'],
    },
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['stanley_controler/centerline.csv']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='aman',
    maintainer_email='amannindra@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            "talker = stanley_controler.publisher_stanley:main",
            
        ],
    },
)
