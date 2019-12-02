#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


"""

from os import walk, remove
from os.path import isdir, join, exists

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


protected_files = ['dessia_common.core_protected.py]


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
        self.formats = 'zip'  
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
            print('Detecting mac, using: {}'.format(self.macs))

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
            if exists(file):
                remove(file)

            file = file[:-3] + 'c'
            if exists(file):
                remove(file)

    

    def run(self):
        print('\n\nBeginning build')
        
        package_name = self.distribution.get_name()
        
        # Creating sdist
        setup_result = run_setup('setup.py', script_args=['sdist', '--formats=tar', '--dist-dir=client_dist'])
        sdist_filename = setup_result.dist_files[0][2]
        folder_path = sdist_filename[:-4]
        if isdir(folder_path):
            shutil.rmtree(folder_path)
        tar = tarfile.open(sdist_filename)
        tar.extractall(path='client_dist')
        tar.close()
        
        # Clening sdist tar
        remove(sdist_filename)
        
        # Compiling
        self.write_pyx_files()
        print('Compiling files')
        setup_result = run_setup('compile.py', script_args=['build_ext', '--build-lib=client_dist'])

        # Copying compiled files to sdist folder
        compiled_files_dir = join('client_dist', package_name)
#        destination_base = join(folder_path, package_name)
        destination_base = folder_path
        for root_dir, _, files in walk(compiled_files_dir):
            for file in files:
                source = join(root_dir, file)
                destination = source.replace('client_dist', destination_base)
                print('copying file {} to {}'.format(source, destination))
                shutil.copy(source, destination)


        # Packaging
        print('Packaging')
        for packaging_format in self.formats:
            shutil.make_archive(folder_path, packaging_format, folder_path)
            
        # Cleaning
        print('Cleaning')
        self.delete_compilation_files()
        shutil.rmtree(folder_path)
        shutil.rmtree(compiled_files_dir)

        
        print('Client build finished, output is {} + {}'.format(folder_path, self.formats))
        
ext_modules = []
for file in protected_files:
    module = file.replace('/', '.')
    module = module[:-3]
    file_to_compile = file[:-3] + '_protected.pyx'
    ext_modules.append(Extension(module,  [file_to_compile]))
    
setup(
    name = 'agb',
    cmdclass = {'build_ext': build_ext, 'cdist': ClientDist},
    install_requires = ['netifaces'],
    ext_modules = ext_modules
)
            

