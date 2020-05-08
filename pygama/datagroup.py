#!/usr/bin/env python3
import os, json
import pandas as pd
from parse import parse
from pprint import pprint

import pygama.utils as pu


class DataGroup:
    """
    Easily look up groups of files according to the LEGEND data convention,
    for processing and metadata organization.  
    
    Source: https://docs.legend-exp.org/index.php/apps/files/?dir=/LEGEND%20Documents/Technical%20Documents/Analysis&fileid=9140#pdfviewer
    """
    def __init__(self, run=None, rhi=None, runlist=None, config=None, mode=None,
                 tslo=None, tshi=None, nfiles=None):
        """
        """
        # DataFrames to store file key attributes
        self.daq_files = None # daq & raw 
        self.hit_files = None # raw, dsp, hit
        self.evt_files = None # evt
        
        # active data takers (geds, spms, etc...)
        self.subsystems = []
        
        # limit number of files (debug, testing)
        self.nfiles = nfiles

        # typical usage: include a JSON config file to set data paths, etc.
        if config is not None:
            self.set_config(os.path.expandvars(config))

        # runs of interest (integers)
        self.runs = []
        if run!=None and rhi!=None:
            self.runs.extend([r for r in range(int(run), int(rhi+1))])
        elif run!=None:
            self.runs.append(int(run))
        elif runlist!=None:
            self.runs.extend(runlist)


    def set_config(self, config):
        """
        define paths to hit- and event-level data through self.config.
        """
        with open(config) as f:
            self.config = json.load(f)
            
        # this option is how we handle different filesystems, e.g. hades, cenpa
        self.experiment = self.config['experiment']

        # set the data tiers to consider (daq, raw, dsp, hit, evt, tmap ...)
        tiers = []
        if 'hit_dirs' in self.config:
            tiers.extend(self.config['hit_dirs'])
        if 'evt_dirs' in self.config:
            tiers.extend(self.config['evt_dirs'])
        
        # set file paths to each tier
        for tier in tiers:
            key = f'{tier}_dir'
            self.config[key] = os.path.expandvars(self.config[key])

        # load primary run lookup file (runDB)
        f_runDB = f"{self.config['meta_dir']}/{self.config['runDB']}"
        with open(os.path.expandvars(f_runDB)) as f:
            self.runDB = json.load(f)
        
        self.run_types = self.config['run_types']
        self.hit_dirs = self.config['hit_dirs']
        self.event_dirs = self.config['event_dirs']

        # get the active subsystems from the main config file
        self.subsystems = []
        for k, v in self.config['daq_to_raw']['ch_groups'].items():
            if v['sysn'] not in self.subsystems:
                self.subsystems.append(v['sysn'])
        
        
    def find_daq_files(self, ft=None):
        """
        Do an os.walk through the daq directory and build a list of files to 
        consider, including any file that matches the file template.
        
        If self.nfiles is set, we stop after adding this many files to the list.
        
        Finally, we convert to a DataFrame to make sorting by other attributes 
        (e.g. timestamp) easier. 
        
        Add the corresponding 'raw` file name(s) for each daq file, using the template string from the config file. Note that the subsystem label
        "sysn" in the file will be left unfilled, and handled by daq_to_raw
        to write to multiple output files:
            `raw_file` : /base/LPGTA/raw/{sysn}/phy/[prefix]_{sysn}_raw.lh5
        """
        # required: file template string.  can be from argument or config dict
        if ft is None:
            ft = self.config['daq_to_raw']['daq_filename_template']
        else:
            ft = ft
            
        # make sure directory exists
        if not os.path.isdir(self.config['daq_dir']):
            print('DAQ directory not found: ', self.config['daq_dir'])
            exit()

        # list of dictionaries, converted to DataFrame
        daq_files = []
        
        if self.experiment == 'LPGTA':
            
            # look for folders matching these strings
            self.run_labels = [f'run{r:0>4d}' for r in self.runs]
            
            stop_walk = False
            for path, folders, files in os.walk(self.config['daq_dir']):
                
                if any(rl in path for rl in self.run_labels):
                    
                    # get the run number
                    run = [rl for rl in self.run_labels if rl in path][0]
                    run = int(run.split('run')[-1])
                    
                    for f in sorted(files):
                        
                        # only accept files matching our template
                        finfo = parse(ft, f)
                        if finfo is not None:
                            finfo = finfo.named # convert to dict
                            finfo['daq_file'] = path + '/' + f
                            finfo['run'] = run
                            
                            # add a pd.datetime64 column for easier sorting
                            ts = finfo['YYYYmmdd'] + finfo['hhmmss']
                            dt = pd.to_datetime(ts, format='%Y%m%d%H%M%S')
                            finfo['date'] = dt
                            
                            daq_files.append(finfo)
                            
                        if self.nfiles is not None and len(daq_files)==self.nfiles:
                            stop_walk = True
                        if stop_walk:
                            break
                if stop_walk:
                    break

        elif self.experiment == "ORCA":
            
            # here we need to search by the ORCA run number, which is treated
            # here as a "subrun" number
            print('lol tbd')
            

        # convert found files to DataFrame
        self.daq_files = pd.DataFrame(daq_files)

        def get_raw_file(row):
            """
            add a 'raw_file' column to self.daq_files.
            subsytem {sysn} is left unspecified, that's handled in daq_to_raw
            """
            # load filename changes (convert daq label to pygama label)
            mods = self.config['daq_to_raw']['file_info_mods']['rtp']
            
            # map variables to the template
            raw_template = self.config['daq_to_raw']['raw_filename_template']
            sd = pu.SafeDict(row.to_dict())
            sd['rtp'] = mods[row['rtp']] # convert DAQ label to legend convention
            raw_file = raw_template.format_map(sd)
            
            # fill in the file path
            raw_path = f"{self.config['raw_dir']}/{{sysn}}/{sd['rtp']}/"
            row['raw_file'] = raw_path + raw_file
            
            return row

        self.daq_files = self.daq_files.apply(get_raw_file, axis=1)
        
        return self.daq_files
        
                
    def find_keys(self):
        """
        every file has a unique [cycle/run/timestamp] according to some format
        string.  if we have these keys, we can build all the files for the
        groups: [daq/raw/dsp/hit/evt] according to the LEGEND format.
        
        These file keys may be worth writing to the legend-metadata repository
        
        Note also that in PGT and L200, they intend to delete the daq files,
        so i might want to either scan over the raw data too, or stipulate that
        we scan over the daq files to build the list of keys, to ensure we
        don't lose anything
        
        NOTE: in the scan, assume there are nested directories, and we
        need to os.walk to get all the unique keys matching the given convention.
        
        the point is to build a table of filenames from a unique ID,
        so that the I/O of the processing is really easy to automate -- 
        we just loop over the table.
        row 1: [unique id] [daq file] [raw file] [dsp file] [hit file]
        
        LPGTA: {base_dir}/{run:0>4d}-{label}/{YYYYmmdd}-{hhmmss}-{rtp}.fcio
        - dir: /global/cfs/cdirs/legend/data/lngs/pgt
        - example: 20200404-031933-phys.fcio
        - unique key: [run, label, YYYYmmdd, hhmmss]
        
        LPGTA
            raw|dsp|hit 
                geds|spms|pmt
                    phy|cal
        DETCHAR
            geds
                scantype                    
        
        ORCA: {YYYY}-{mm}-{dd}-{prefix}Run{run}
        - dir: /global/cfs/cdirs/legend/data/cage/Data
        - example: 2020-3-17-CAGERun126
        - unique key: [YYYY, mm, dd, run]
        
        HADES: {prefix}-run{run:0>4d}-{YYmmdd}T{hhmmss}.fcio
        - dir: /global/cfs/cdirs/legend/data/hades/I02166B/tier0/ba_HS4_top_dlt
        
        - example: char_data-I02166B-ba_HS4_top_dlt-run0001-191025T153307.fcio
        - unique key: [run, YYmmdd, hhmmss]
        NOTE: how to handle different detectors?
        
        LEGEND: {expt}_r{run:0>4d}_{YYYYmmdd}T{hhmmss}Z_{rtp}_{sysn}_{tier}.lh5
        - dir: /global/cfs/cdirs/legend/users/wisecg/LPGTA/raw
        - example: LPGTA_r0021_20200404T083254Z_phy_geds_raw.lh5
        - unique key: [run, YYYYmmdd, hhmmss]
        
        """
        # self.keys = None
        # print('hi clint')
        pprint(self.config)
        
        base_dir = '/cori'
        os.walk(base_dir)
        find files matching expression