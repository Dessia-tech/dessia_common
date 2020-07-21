#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


"""
# from os import walk, remove
# from os.path import isdir, join, exists
import os
import sys
import re
import tempfile
from subprocess import CalledProcessError, check_output

from setuptools import setup

from Cython.Distutils import build_ext

from distutils.core import run_setup
from distutils.cmd import Command
from distutils.extension import Extension
import time
from datetime import timedelta
import calendar
import shutil
import tarfile

import hashlib
import netifaces


protected_files = ['dessia_common/core_protected.py']

ext_modules = []
for file in protected_files:
    module = file.replace('/', '.')
    module = module[:-3]
    file_to_compile = file[:-3] + '_protected.pyx'
    ext_modules.append(Extension(module,  [file_to_compile]))
    
    
    
class ClientDist(Command):
    description = 'Creating client distribution with compiled packages and license'
    user_options = [
        # The format is (long option, short option, description).
        ('exp-year=', None, 'Year of license expiration'),
        ('exp-month=', None, 'Month of year of license expiration'),
        ('exp-day=', None, 'Day of month of license expiration'),
        ('formats=', None,
         "formats for source distribution (comma-separated list)"),
        ('getnodes=', None, 'UUID given by getnode (comma-separated list)'),
        ('macs=', None, 'MACS of machine (comma-separated list)'),
        ('detect-macs', None, 'using this machine macs')
        ]
    
    addresses_determination_lines = ['import netifaces\n',
                                     'addrs = []\n',
                                     'for i in netifaces.interfaces():\n',
                                     "    if i != 'lo':\n",
                                     "        for k, v in netifaces.ifaddresses(i).items():\n",
                                     "            for v2 in v:\n",
                                     "                if 'addr' in v2:\n",
                                     "                    a = v2['addr']\n",
                                     "                    if len(a) == 17:\n",
                                     "                        addrs.append(a.replace(':',''))\n",
                                     "addrs = set(addrs)\n\n"
                                     ]
    
    def get_machine_macs(self):
        addrs = []
        for i in netifaces.interfaces():
            if i != 'lo':
                for k, v in netifaces.ifaddresses(i).items():
                    for v2 in v:
                        if 'addr' in v2:
                            a = v2['addr']
                            if len(a) == 17:
                                addrs.append(a.replace(':',''))
        return list(set(addrs))

    def initialize_options(self):
        """Set default values for options."""
        # Each user option must be listed here with their default value.
        self.exp_year = 2000
        self.exp_month = 1
        self.exp_day = 1
        self.formats = 'gztar'  
        self.getnodes = None
        self.macs = None
        self.detect_macs = None

    def finalize_options(self):
        """Post-process options."""

        self.detect_macs = self.detect_macs is not None
        if not self.detect_macs:
            
            macs = []
            if self.macs is not None:
                for mac in self.macs.split(','):
                    macs.append(mac)
                self.macs = macs        
                print('\nCompiling for macs: {}'.format(self.macs))
                
            getnodes = []
            if self.getnodes is not None:
                for getnode in self.getnodes.split(','):
                    getnodes.append(getnode)
                self.getnodes = getnodes        
                print('\nCompiling for getnodes: {}'.format([str(g for g in self.getnodes)]))

        else:   
            self.macs = self.get_machine_macs()
            print('Using detected mac of this machine: {}'.format(self.macs))

        if not self.detect_macs and self.getnodes is None and self.macs is None:
            raise ValueError('Define either a mac or a getnode to protect the code or use detect-macs option')

        if self.exp_year is not None:
            self.exp_year = int(self.exp_year)
        if self.exp_month is not None:
            self.exp_month = int(self.exp_month)
        if self.exp_day is not None:
            self.exp_day = int(self.exp_day)
            
        self.expiration = int(calendar.timegm(time.struct_time((self.exp_year,
                                                                self.exp_month,
                                                                self.exp_day,
                                                                0, 0, 0 ,0, 0, 0))))
        self.not_before = int(time.time())

        print('\nExpiration date: {}/{}/{}: duration of {} days'.format(self.exp_day,
                                                  self.exp_month,
                                                  self.exp_year,
                                                  timedelta(seconds=self.expiration-self.not_before).days))
        
            
        formats = [] 
        for s in self.formats.split(','):
            if not s in ['zip', 'tar', 'gztar', 'bztar']:
                raise NotImplementedError('Unsupported file type: {}'.format(s))
            formats.append(s)
        self.formats = formats

        
    def protection_lines(self):
        protection_lines = []
        if self.getnodes is not None:
            physical_token = hashlib.sha256(str(self.getnodes[0]).encode()).hexdigest()
        if self.macs is not None:
            physical_token = hashlib.sha256(str(self.macs[0]).encode()).hexdigest()

        error_msg_time_before = 'Invalid license Please report this error to DessIA support with this traceback token: TB{}'.format(physical_token)
        error_msg_time_after = 'Invalid license Please report this error to DessIA support with this traceback token: TA{}'.format(physical_token)
        error_msg_mac = 'Invalid license. Please report this error to DessIA support with this traceback token: M{}'.format(physical_token)
        protection_lines = ['valid_license = True\n',
                            't_execution = time_package.time()\n',
                            'if t_execution > {}:\n'.format(self.expiration), 
                            '    print("{}")\n'.format(error_msg_time_after),
                            '    raise RuntimeError\n\n',
                            'if t_execution < {}:\n'.format(self.not_before),
                            '    print("{}")\n'.format(error_msg_time_before),
                            '    raise RuntimeError\n\n']
        
        
        if self.macs is not None:
            protection_lines.extend(['if addrs.isdisjoint(set({})):\n'.format(self.macs),
                                     '    print("{}")\n'.format(error_msg_mac),
                                     '    raise RuntimeError\n\n'])
        elif self.getnodes is not None:
            protection_lines.extend(['if getnode() not in {}:\n'.format(self.getnodes),
                                     '    print("{}")\n'.format(error_msg_mac),
                                     '    raise RuntimeError\n\n'])
            
        return protection_lines

    def write_pyx_files(self):
        self.files_to_compile = []
        for file in protected_files:
            with open(file, 'r', encoding="utf-8") as f:
                # File Parsing
                new_file_lines = []        
                lines = f.readlines()
                line_index = 0
                # Inserting imports for uuid and time
                line = lines[line_index]
                while line.startswith('#!') or line.startswith('# -*-') or ('cython' in line):
                    new_file_lines.append(line)
                    line_index += 1
                    line = lines[line_index]
                    
                    
                new_file_lines.append('import time as time_package\n')
                if self.getnodes is not None:
                    new_file_lines.append('from uuid import getnode\n')
                if self.macs is not None:
                    new_file_lines.extend(self.addresses_determination_lines)
        
                while line_index < len(lines):
                    line = lines[line_index]
                    new_file_lines.append(line)
                    if line.startswith('def '):# Function
                        # counting parenthesis
                        op = line.count('(')
                        cp = line.count(')')
                        while op != cp:
                            line_index += 1
                            line = lines[line_index]
                            new_file_lines.append(line)
                            op += line.count('(')
                            cp += line.count(')')
                        # list of args is finished.
                        # Now trying to see if lines are docstrings
                        line_index += 1
                        line = lines[line_index]
        
                        if line.startswith('    """'):
                            new_file_lines.append(line)
                            line_index += 1
                            line = lines[line_index]
                            
                            while not line.startswith('    """'):
                                new_file_lines.append(line)
                                line_index += 1
                                line = lines[line_index]
                            new_file_lines.append(line)
                            line_index += 1
                            line = lines[line_index]
                            
        
                        for protection_line in self.protection_lines():
                            new_file_lines.append('    {}'.format(protection_line))
        
                        new_file_lines.append(line)
        
        
                    elif line.startswith('    def ') and not line.startswith('    def __init__('):# Method of class, not init
                        # counting parenthesis
                        op = line.count('(')
                        cp = line.count(')')
                        while op != cp:
                            line_index += 1
                            line = lines[line_index]
                            op += line.count('(')
                            cp += line.count(')')
                            new_file_lines.append(line)
                        # list of args is finished.
                        # Now trying to see if lines are docstrings
                        line_index += 1
                        line = lines[line_index]
        
                        if line.startswith('        """'):
                            new_file_lines.append(line)    
                            line_index += 1
                            line = lines[line_index]
                            while not line.startswith('        """'):
                                new_file_lines.append(line)
                                line_index += 1
                                line = lines[line_index]
                            new_file_lines.append(line)
                            line_index += 1
                            line = lines[line_index]
                                        
        
                        for protection_line in self.protection_lines():
                            new_file_lines.append('        {}'.format(protection_line))
                            
                        new_file_lines.append(line)
                       
                    line_index+=1
        
                            
                new_file_name = file[:-3]+'_protected.pyx'
                self.files_to_compile.append(new_file_name)
                with open(new_file_name, 'w+', encoding="utf-8") as nf:
                    nf.writelines(new_file_lines)
                    
                        
                
    def delete_compilation_files(self):
        # Remove _protected files and .c
        for file in self.files_to_compile:
            if os.path.exists(file):
                os.remove(file)

            file = file[:-3] + 'c'
            if os.path.exists(file):
                os.remove(file)

    

    def run(self):
        print('\n\nBeginning build')
        package_name = self.distribution.get_name()
        tmp_dir = tempfile.mkdtemp()
        # Creating sdist
        setup_result = run_setup('setup.py', script_args=['sdist',
                                                          '--formats=tar',
                                                          '--dist-dir={}'.format(tmp_dir)])
        sdist_filename = setup_result.dist_files[0][2]
        
        folder_path = sdist_filename[:-4]
        dist_name = os.path.basename(folder_path)
        if os.path.isdir(folder_path):
            shutil.rmtree(folder_path)
        tar = tarfile.open(sdist_filename)
        tar.extractall(path=tmp_dir)
        # compile_path = os.path.join(tmp_dir, os.path.commonprefix(tar.getnames()))
        tar.close()
        
        
        # Compiling
        self.write_pyx_files()
        print('Compiling files in {}'.format(tmp_dir))
        setup_result = run_setup('compile.py', script_args=['build_ext',
                                                            '--build-lib={}'.format(tmp_dir)])
        # Copying compiled files to sdist folder
        compiled_files_dir = os.path.join(tmp_dir, package_name)
#        destination_base = join(folder_path, package_name)
        # destination_base = folder_path
        
        
        for root_dir, _, files in os.walk(compiled_files_dir):
            for file in files:
                source = os.path.join(root_dir, file)
                # destination = source.replace('client_dist', destination_base)
                # print('root_dir')
                new_file_location = source.replace(tmp_dir, '').lstrip('/ ')
                destination = os.path.join(folder_path, new_file_location)
                print('copying file {} to {}'.format(source, destination))
                shutil.copy(source, destination)


        # Packaging
        print('Packaging')
        archive_names = []
        suffix = '-py{}{}'.format(sys.version_info.major, sys.version_info.minor)
        archive_name = os.path.join('client_dist', dist_name+suffix)            
        for packaging_format in self.formats:
            archive_name_with_extension = shutil.make_archive(archive_name,
                                                              root_dir=tmp_dir,
                                                              format=packaging_format,
                                                              base_dir=dist_name)
            archive_names.append(archive_name_with_extension)

            
        # Cleaning
        print('Cleaning')
        self.delete_compilation_files()
        shutil.rmtree(folder_path)
        shutil.rmtree(compiled_files_dir)
        
        # Cleaning sdist dir
        shutil.rmtree(tmp_dir)

        print('Client build finished, output is {}'.format(archive_names))
    
class ClientWheelDist(ClientDist):
    
    def run(self):
        print('\n\nBeginning build')
        package_name = self.distribution.get_name()
        
        # Compiling
        self.write_pyx_files()
        print('Compiling files')
        setup_result = run_setup('compile.py', script_args=['bdist_wheel'],
                                 )
            
        # Cleaning
        print('Cleaning')
        self.delete_compilation_files()

        print('Client build finished, output is {}'.format(setup_result.dist_files))
    
tag_re = re.compile(r'\btag: %s([0-9][^,]*)\b')
version_re = re.compile('^Version: (.+)$', re.M)
    
def version_from_git_describe(version):
    if version[0]=='v':
            version = version[1:]

    # PEP 440 compatibility
    number_commits_ahead = 0
    if '-' in version:
        version, number_commits_ahead, commit_hash = version.split('-')
        number_commits_ahead = int(number_commits_ahead)

    print('number_commits_ahead', number_commits_ahead)

    split_versions = version.split('.')
    if 'post' in split_versions[-1]:
        suffix = split_versions[-1]
        split_versions = split_versions[:-1]
    else:
        suffix = None

    for pre_release_segment in ['a', 'b', 'rc']:
        if pre_release_segment in split_versions[-1]:
            if number_commits_ahead > 0:
                split_versions[-1] = str(split_versions[-1].split(pre_release_segment)[0])
                if len(split_versions) == 2:
                    split_versions.append('0')
                if len(split_versions) == 1:
                    split_versions.extend(['0', '0'])

                split_versions[-1] = str(int(split_versions[-1])+1)
                future_version = '.'.join(split_versions)
                return '{}.dev{}'.format(future_version, number_commits_ahead)
            else:
                return '.'.join(split_versions)

    if number_commits_ahead > 0:
        if len(split_versions) == 2:
            split_versions.append('0')
        if len(split_versions) == 1:
            split_versions.extend(['0', '0'])
        split_versions[-1] = str(int(split_versions[-1])+1)
        split_versions = '.'.join(split_versions)
        return '{}.dev{}'.format(split_versions, number_commits_ahead)
    else:
        if suffix is not None:
            split_versions.append(suffix)

        return '.'.join(split_versions)

def get_version():
    # Return the version if it has been injected into the file by git-archive
    version = tag_re.search('$Format:%D$')
    if version:
        return version.group(1)

    d = os.path.dirname(__file__)

    if os.path.isdir(os.path.join(d, '.git')):
        cmd = 'git describe --tags'
        try:
            version = check_output(cmd.split()).decode().strip()[:]

        except CalledProcessError:
            raise RuntimeError('Unable to get version number from git tags')

        return version_from_git_describe(version)
    else:
        # Extract the version from the PKG-INFO file.
        with open(os.path.join(d, 'PKG-INFO')) as f:
            version = version_re.search(f.read()).group(1)

    # print('version', version)
    return version


setup(
    name = 'dessia_common',
    version=get_version(),
    description="Common tools for DessIA software",
    long_description='',
    keywords='',
    url='',
    author='Steven Masfaraud',
    author_email='masfaraud@dessia.tech',
    packages=['dessia_common'],
    install_requires=['typeguard', 'networkx', 'numpy', 'pandas', 'jinja2',
                      'mypy_extensions', 'scipy', 'pyDOE',
                      'netifaces'],
    python_requires='>=3.7',
    cmdclass = {'build_ext': build_ext, 'cdist': ClientDist,
                'cdist_wheel': ClientWheelDist},
    
    ext_modules = ext_modules,

)

# try:
#     from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
#     class bdist_wheel(_bdist_wheel):
#         def finalize_options(self):
#             _bdist_wheel.finalize_options(self)
#             self.root_is_pure = False
# except ImportError:
#     bdist_wheel = None

# setup(
#     # ...
#     cmdclass={'bdist_wheel': bdist_wheel},
# )
