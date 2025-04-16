'''Process Raw (L0) data to L1A HDF5'''
import os
import json
from datetime import datetime, timedelta, date
import re
import numpy as np
import pandas as pd
import tables

from Source.MainConfig import MainConfig
from Source.HDFRoot import HDFRoot
from Source.HDFGroup import HDFGroup
from Source.Utilities import Utilities


class ProcessL1aTriOS:
    '''Process L1A for TriOS from MSDA-XE'''
    @staticmethod
    def processL1a(fp, outFilePath): 
        # fp is a list of all triplets

        configPath = MainConfig.settings['cfgPath']
        cal_path = configPath[0:configPath.rfind('.')] + '_Calibration/'
        # In case full path includes a '.'

        if '.mlb' in fp[0]:   # Multi frame
            # acq_time = []
            acq_name = []
            for file in fp:

                # ## Test filename for different date formating
                # match1 = re.search(r'\d{8}_\d{6}', file.split('/')[-1])
                # match2 = re.search(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}', file.split('/')[-1])
                # if match1 is not None:
                #     a_time = match1.group()
                # elif match2 is not None:
                #     a_time = match2.group()
                # else:
                #     print("  ERROR: no identifier recognized in TRIOS L0 file name" )
                #     print("  L0 filename should have a date to identify triplet instrument")
                #     print("  either 'yyymmdd_hhmmss' or 'yyy-mm-dd_hh-mm-ss' ")
                #     exit()

                # acq_time.append(a_time)


                ## Test filename for station/cast

                def parse_filename(data):
                    dates = []
                    for pattern in [
                        r'\d{8}.\d{6}', 
                        r'\d{4}.\d{2}.\d{2}.\d{2}.\d{2}.\d{2}',
                        r'\d{8}.\d{2}.\d{2}.\d{2}',
                        r'\d{4}.\d{2}.\d{2}.\d{6}',
                        r'\d{4}S', 
                    ]:
                        match = re.search(pattern, data)
                        if match is not None:
                            dates.append(match.group(0))
                    
                    if len(dates) == 0:
                        raise IndexError
                    
                    return dates[0]
                
                try:
                    a_name = parse_filename(file.split('/')[-1])
                except IndexError:
                    print("  ERROR: no identifier recognized in TRIOS L0 file name" )
                    print("  L0 filename should have a cast to identify triplet instrument")
                    print("  ending in 4 digits before S.mlb ")
                    return None,None

                # match1 = re.search(r'\d{8}_\d{6}', file.split('/')[-1])
                # match2 = re.search(r'\d{4}S', file.split('/')[-1])
                # match3 = re.search(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}', file.split('/')[-1])
                # # string except for serial number will be the same for a triplet
                # if match1 is not None:
                #     a_name = match1.group()
                # elif match2 is not None:
                #     a_name = match2.group()
                # elif match3 is not None:
                #     a_name = match3.group()
                # else:
                #     print("  ERROR: no identifier recognized in TRIOS L0 file name" )
                #     print("  L0 filename should have a cast to identify triplet instrument")
                #     print("  ending in 4 digits before S.mlb ")
                #     return None,None

                # acq_time.append(a_cast)
                acq_name.append(a_name)

            # acq_time = list(dict.fromkeys(acq_time)) # Returns unique timestamps
            acq_name = list(dict.fromkeys(acq_name)) # Returns unique names
            outFFP = []
            # for a_time in acq_time:
            for a_name in acq_name:
                print("")
                print("Generate the telemetric file...")
                # print('Processing: ' +a_time)
                print('Processing: ' +a_name)

                # hdfout = a_time + '_.hdf'

                tables.file._open_files.close_all() # Why is this necessary?

                # For each triplet, this creates an HDF
                root = HDFRoot()
                root.id = "/"
                root.attributes["WAVELENGTH_UNITS"] = "nm"
                root.attributes["LI_UNITS"] = "count"
                root.attributes["LT_UNITS"] = "count"
                root.attributes["ES_UNITS"] = "count"
                root.attributes["SATPYR_UNITS"] = "count"
                root.attributes["PROCESSING_LEVEL"] = "1a"

                # ffp = [s for s in fp if a_time in s]
                ffp = [s for s in fp if a_name in s]
                root.attributes["RAW_FILE_NAME"] = str(ffp)
                # root.attributes["TIME-STAMP"] = a_name
                root.attributes["CAST"] = a_name
                for file in ffp:
                    try:
                        # Regex accomodate both SAM_1234_ and SAM1234 conventions
                        name = re.findall(r'SAM_?(\d+)_', os.path.basename(file))[0]
                    except IndexError:
                        raise ValueError("ERROR : naming convention os not respected")

                    start,_ = ProcessL1aTriOS.formatting_instrument(name,cal_path,file,root,configPath)

                    if start is None:
                        return None, None
                    acq_datetime = datetime.strptime(start,"%Y%m%dT%H%M%SZ")
                    root.attributes["TIME-STAMP"] = datetime.strftime(acq_datetime,'%a %b %d %H:%M:%S %Y')


                # File naming convention on TriOS TBD depending on convention used in MSDA_XE
                #The D-6 requirements to add the timestamp manually on every acquisition is impractical.
                #Convert to using any unique name and append timestamp from data (start)
                try:
                    new_name = file.split('/')[-1].split('.mlb')[0].split(f'SAM_{name}_RAW_SPECTRUM_')[1]
                    if re.search(r'\d{4}S', file.split('/')[-1]) is not None:
                        new_name = new_name+'_'+str(start)  # I'm not sure what match2 was supposed to find in ACRI's code - AR
                except IndexError as err:
                    msg = "possibly an error in naming of Raw files"
                    Utilities.writeLogFile(msg)
                    try:
                        new_name = file.split('/')[-1].split('.mlb')[0].split(f'SAM_{name}_Spectrum_RAW_')[1]
                    except IndexError as e:
                        new_name = f'{start}'  # Needed for files with different naming conventions

                # new_name = outFilePath + '/' + 'Trios_' + str(start) + '_' + str(stop) + '_L1A.hdf'
                # outFFP.append(os.path.join(outFilePath,f'{new_name}_L1A.hdf'))
                outFFP.append(os.path.join(outFilePath,f'{new_name}_L1A.hdf'))
                root.attributes["L1A_FILENAME"] = outFFP[-1]

                root = ProcessL1aTriOS.fixChronology(root)

                try:
                    # root.writeHDF5(new_name)
                    root.writeHDF5(outFFP[-1])

                except Exception:
                    msg = 'Unable to write L1A file. It may be open in another program.'
                    Utilities.errorWindow("File Error", msg)
                    print(msg)
                    Utilities.writeLogFile(msg)
                    return None, None

                Utilities.checkOutputFiles(outFFP[-1])

            return root, outFFP
        else:
            print('Single Frame deprecated')

        return None, None
    
    # use namesList to define dtype for recarray
    @staticmethod
    def reshape_data(NAME,N,data):
        ds_dt = np.dtype({'names':[NAME],'formats':[(float)] })
        tst = np.array(data).reshape((1,N))
        rec_arr2 = np.rec.fromarrays(tst, dtype=ds_dt)
        return rec_arr2

    # def reshape_data_str(NAME,N,data):
    #     dt = h5py.special_dtype(vlen=str)
    #     ds_dt = np.dtype({'names':[NAME],'formats':[dt] })
    #     tst = np.array(data).reshape((1,N))
    #     rec_arr2 = np.rec.fromarrays(tst, dtype=ds_dt)
    #     return rec_arr2

    # Function for reading and formatting .dat data file
    @staticmethod
    def read_dat(inputfile):
        file_dat = open(inputfile,'r')
        flag = 0
        index = 0
        for line in file_dat:
            index = index + 1
            # checking end of attributes
            if '[END] of [Attributes]' in line:
                flag = 1
                break
        if flag == 0:
            print('PROBLEM WITH FILE .dat: Metadata not found')
            end_meta = None
        else:
            end_meta = index
        file_dat.close()
        metadata = pd.read_csv(inputfile, skiprows=1, nrows=end_meta-3, header=None, sep='=')
        meta = metadata[metadata[0].str.contains('Version|Date|PositionLatitude|PositionLongitude|IntegrationTime')][1]
        data = pd.read_csv(inputfile, skiprows=end_meta+2, nrows=255, header=None, sep=r'\s+')[1]
        meta = meta.to_numpy(dtype=str)
        data = data.to_numpy(dtype=str)
        date1 = datetime.strptime(meta[1], " %Y-%m-%d %H:%M:%S")
        time = meta[1].split(' ')[2]
        meta[0] = date1
        meta[1] = time
        return meta,data

    @staticmethod
    def read_mlb(filename):
        """
        Read TriOS .mlb file and return metadata (e.g. temperature, tilt, integration time),
            spectrum data, and timestamps (from IDData)
        """
        # Skip Header and Get Column Names
        with open(filename, 'r', encoding="utf-8") as f:
            start_index, column_names = 0, ''
            for l in f:
                if l == '\n' or l.startswith('%'):
                    start_index += 1
                    column_names = l
                    continue
                else:
                    break
            else:
                raise ValueError("No header found in file")
        column_names = re.split('\s+%', column_names[1:].strip())
        # Read Data
        data = pd.read_csv(filename, skiprows=start_index + 1, names=column_names, sep=r'\s+')
        # Format IDData to UTC datetime
        # dt = pd.to_datetime(data.IDData.str[6:], format='%Y-%m-%d_%H-%M-%S_%f', utc=True)
        if 'DateTime' not in data.columns:
            dt = pd.to_datetime(data.IDData.str[6:], format='%Y-%m-%d_%H-%M-%S_%f', utc=True)
            # Convert from seconds since 1970-01-01 to days since 1900-01-01
            dt = ((datetime(1970, 1, 1) - datetime(1900, 1, 1)).total_seconds() + dt.to_numpy(dtype=float)/10**9) / 86400
            # Add two days to match legacy format
            dt += 2
            # Insert Column at position 0
            data.insert(0, 'DateTime', dt)
            column_names = ['DateTime'] + column_names
        # Extract Spectrum Columns
        spec_cols = [idx for idx, h in enumerate(column_names) if h.startswith('c')]
        meta = data.iloc[:, 0:spec_cols[0]]
        specs = data.iloc[:, spec_cols]
        time = data.IDData
        return meta, specs, time

    # Function for reading cal files
    @staticmethod
    def read_cal(inputfile):
        file_dat = open(inputfile,'r', encoding="utf-8")
        flag_meta = 0
        flag_data = 0
        index = 0
        for line in file_dat:
            index = index + 1
            # checking end of attributes
            if '[END] of [Attributes]' in line:
                flag_meta = index
            if '[DATA]' in line:
                flag_data = index
                break

        if flag_meta == 0:
            print('PROBLEM WITH CAL FILE: Metadata not found')
            exit()
        if flag_data == 0:
            print('PROBLEM WITH CAL FILE: data not found')
            exit()

        file_dat.close()

        metadata = pd.read_csv(inputfile, skiprows=1, nrows=flag_meta-3, header=None, sep='=')
        metadata = metadata[~metadata[0].str.contains(r'\[')]
        metadata = metadata.reset_index(drop=True)
        data = pd.read_csv(inputfile, skiprows=flag_data+1, nrows=255, header=None, sep=r'\s+')

        # NAN filtering, set to zero
        for col in data:
            indnan = data[col].astype(str).str.contains('nan', case=False)
            data.loc[indnan, col] = '0.0'

        return metadata,data

    # Generic function for adding metadata from the ini file
    @staticmethod
    def get_attr(metadata, gp):
        for irow,_ in enumerate(metadata.iterrows()):
            gp.attributes[metadata[0][irow].strip()]=str(metadata[1][irow].strip())
        return None

    # Function for reading and getting metadata for config .ini files
    @staticmethod
    def attr_ini(ini_file, gp):
        ini = pd.read_csv(ini_file, skiprows=1, header=None, sep='=')
        ini = ini[~ini[0].str.contains(r'\[')]
        ini = ini.reset_index(drop=True)
        ProcessL1aTriOS.get_attr(ini,gp)
        return None


    # Function for data formatting
    @staticmethod
    def formatting_instrument(name, cal_path, input_file, root, configPath):
        print('Formatting ' + str(name) + ' Data')
        # Extract measurement type from config file
        with open(configPath, 'r', encoding="utf-8") as fc:
            text = fc.read()
            conf_json = json.loads(text)
        sensor = conf_json['CalibrationFiles']['SAM_'+name+'.ini']['frameType']
        print(sensor)

        if 'LT' not in sensor and 'LI' not in sensor and 'ES' not in sensor:
            print('Error in config file. Check frame type for calibration files')
            # exit()
            return None,None

        # A = f.create_group('SAM_'+name+'.dat')
        gp =  HDFGroup()
        gp.id = 'SAM_'+name+'.ini'
        root.groups.append(gp)

        # Configuration file
        ProcessL1aTriOS.attr_ini(cal_path + 'SAM_'+name+'.ini',gp)

        # Formatting data
        meta, data, time = ProcessL1aTriOS.read_mlb(input_file)

        ## if date is the first field "%yyy-mm-dd"
        if len(time[0].rsplit('_')[0]) == 11:
            dates = [i.rsplit('_')[0][1:] for i in time]
            datetag = [float(i.rsplit('-')[0] + str(date(int(i.rsplit('-')[0]), int(i.rsplit('-')[1]), int(i.rsplit('-')[2])).timetuple().tm_yday)) for i in dates]
            timetag = [float(i.rsplit('_')[1].replace('-','') + '000') for i in time]
        ## if not it is in second place
        else:
            dates = [i.rsplit('_')[1] for i in time]
            datetag = [float(i.rsplit('-')[0] + str(date(int(i.rsplit('-')[0]), int(i.rsplit('-')[1]), int(i.rsplit('-')[2])).timetuple().tm_yday)) for i in dates]
            timetag = [float(i.rsplit('_')[2].replace('-','') + '000') for i in time]

        # Reshape data and create HDF5 datasets
        gp.attributes['CalFileName'] = 'SAM_' + name + '.ini'
        n = len(meta)
        cfg = [
            # ('DATETAG', meta['DateTime'], 'NONE'),
            # ('DATETAG2', datetag, 'NONE'),
            ('DATETAG', datetag, 'NONE'),
            ('INTTIME', meta['IntegrationTime'], sensor),
            ('CHECK', np.zeros(n), 'NONE'),
            ('DARK_AVE', meta['DarkAvg'] if 'DarkAvg' in meta.columns else np.zeros(n), sensor),
            ('DARK_SAMP', np.zeros(n), sensor),
            ('FRAME', np.zeros(n), 'COUNTER'),
            ('POSFRAME', np.zeros(n), 'COUNT'),
            ('SAMPLE', np.zeros(n), 'DELAY'),
            ('SPECTEMP', meta['Temperature'] if 'Temperature' in meta.columns else np.zeros(n), 'NONE'),
            ('THERMAL_RESP', np.zeros(n), 'NONE'),
            ('TIMER', np.zeros(n), 'NONE'),
            ('TIMETAG2', timetag, 'NONE')
        ]
        if 'PreTilt' in meta.columns:
            cfg.append(('TILT_PRE', meta['PreTilt'], 'NONE'))
        if 'PostTilt' in meta.columns:
            cfg.append(('TILT_POST', meta['PostTilt'], 'NONE'))
        for k, v, t in cfg:
            # Reshape Data
            rec = ProcessL1aTriOS.reshape_data(t, n, data=v)
            # HDF5 Dataset Creation
            gp.addDataset(k)
            gp.datasets[k].data = np.array(rec, dtype=[('NONE', '<f8')])

        # Computing wavelengths
        c0 = float(gp.attributes['c0s'])
        c1 = float(gp.attributes['c1s'])
        c2 = float(gp.attributes['c2s'])
        c3 = float(gp.attributes['c3s'])
        wl = []
        for i in range(1,256):
        # for i in range(1,data.shape[1]+1):
            wl.append(str(round((c0 + c1*(i+1) + c2*(i+1)**2 + c3*(i+1)**3), 2)))

        #Create Data (LI,LT,ES) dataset
        ds_dt = np.dtype({'names': wl,'formats': [np.float64]*len(wl)})
        my_arr = np.array(data).transpose()
        try:
            rec_arr = np.rec.fromarrays(my_arr, dtype=ds_dt)
        except ValueError as err:
            if len(err.args) > 0 and err.args[0].startswith("mismatch between the number of fields "):
                rec_arr = np.rec.fromarrays(my_arr[1:], dtype=ds_dt)
            else:
                raise
        gp.addDataset(sensor)
        gp.datasets[sensor].data=np.array(rec_arr, dtype=ds_dt)

        # Calibrations files
        metacal,cal = ProcessL1aTriOS.read_cal(cal_path + 'Cal_SAM_'+name+'.dat')
        B1 = gp.addDataset('CAL_'+sensor)
        B1.columns["0"] = cal.values[:,1].astype(np.float64)
        B1.columnsToDataset()
        ProcessL1aTriOS.get_attr(metacal,B1)

        metaback,back = ProcessL1aTriOS.read_cal(cal_path + 'Back_SAM_'+name+'.dat')
        # C1 = gp.addDataset('BACK_'+sensor,data=back[[1,2]].astype(np.float64))
        C1 = gp.addDataset('BACK_'+sensor)
        C1.columns["0"] = back.values[:,1]
        C1.columns["1"] = back.values[:,2]
        C1.columnsToDataset()
        ProcessL1aTriOS.get_attr(metaback,C1)

        start_time = datetime.strftime(datetime(1900,1,1) + timedelta(days=meta['DateTime'].iloc[0]-2), "%Y%m%dT%H%M%SZ")
        stop_time = datetime.strftime(datetime(1900,1,1) + timedelta(days=meta['DateTime'].iloc[-1]-2), "%Y%m%dT%H%M%SZ")

        return start_time,stop_time

    # TriOS L0 exports are in reverse chronological order. Reorder all data fields
    @staticmethod
    def fixChronology(node):
        print('Sorting all datasets chronologically')
        for gp in node.groups:
            dateTime = []
            dateTagArray = gp.datasets['DATETAG'].data
            timeTagArray = gp.datasets['TIMETAG2'].data
            for i, dateTag in enumerate(dateTagArray):
                dt1 = Utilities.dateTagToDateTime(dateTag[0])
                dateTime.append(Utilities.timeTag2ToDateTime(dt1,timeTagArray[i][0]))

            for ds in gp.datasets:

                # BACK_ and CAL_ are nLambda x 2 and nLambda x 1, respectively, not timestamped to DATETAG, TIMETAG2
                if (not ds.startswith('BACK_')) and (not ds.startswith('CAL_')):
                    gp.datasets[ds].datasetToColumns()
                    gp.datasets[ds].data = np.array([x for _, x in sorted(zip(dateTime,gp.datasets[ds].data))])

        return node
