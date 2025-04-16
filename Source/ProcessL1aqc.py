''' Process L1A to L1AQC '''
import math
import datetime
import copy
import numpy as np
from pysolar.solar import get_azimuth, get_altitude

from Source.HDFDataset import HDFDataset
from Source.ProcessL1aqc_deglitch import ProcessL1aqc_deglitch
from Source.Utilities import Utilities
from Source.ConfigFile import ConfigFile

class ProcessL1aqc:
    ''' Process L1A to L1AQC '''

    @staticmethod
    def filterBadTimes(node,badTimes):
        ''' For each dataset in each group, find the badTimes to remove and delete those rows
         Keep Ancillary Data in tact. This may help in L1B to capture better ancillary data '''
        if len(badTimes) > 0:
            msg = "Eliminate combined filtered data from datasets.*****************************"
            print(msg)
            Utilities.writeLogFile(msg)

            for gp in node.groups:

                # SATMSG has an ambiguous timer POSFRAME.COUNT, cannot filter
                # Test: Keep Ancillary Data in tact. This may help in L1B to capture better ancillary data
                if gp.id != "SOLARTRACKER_STATUS" and gp.id != "ANCILLARY_METADATA":
                    fractionRemoved = ProcessL1aqc.filterData(gp, badTimes)

                    # Now test whether the overlap has eliminated all radiometric data
                    if fractionRemoved > 0.98 and (gp.id.startswith("ES") or gp.id.startswith("LI") or gp.id.startswith("LT")):
                        msg = "Radiometric data >98'%' eliminated. Aborting."
                        print(msg)
                        Utilities.writeLogFile(msg)
                        return None

                    gpTimeset  = gp.getDataset("TIMETAG2")

                    gpTime = gpTimeset.data["NONE"]
                    lenGpTime = len(gpTime)
                    msg = f'   Data end {lenGpTime} long, a loss of {round(100*(fractionRemoved))} %'
                    print(msg)
                    Utilities.writeLogFile(msg)

                    if 'FrameType' in gp.attributes and gp.attributes['FrameType'] == 'ShutterDark':
                        fractionRemoved = ProcessL1aqc.filterData_ADJUSTED(gp, badTimes)

                        # Now test whether the overlap has eliminated all radiometric data
                        if fractionRemoved > 0.98 and (gp.id.startswith("ES") or gp.id.startswith("LI") or gp.id.startswith("LT")):
                            msg = "Radiometric data >98'%' eliminated. Aborting."
                            print(msg)
                            Utilities.writeLogFile(msg)
                            return None

                        gpTimeset  = gp.getDataset("TIMETAG2")

                        gpTime = gpTimeset.data["NONE"]
                        lenGpTime = len(gpTime)
                        msg = f'   Data end {lenGpTime} long, a loss of {round(100*(fractionRemoved))} %'
                        print(msg)
                        Utilities.writeLogFile(msg)
        return node

    @staticmethod
    def filterData(group, badTimes):
        ''' Delete flagged records '''

        msg = f'Remove {group.id} Data'
        print(msg)
        Utilities.writeLogFile(msg)

        timeStamp = group.getDataset("DATETIME").data

        startLength = len(timeStamp)
        msg = f'   Length of dataset prior to removal {startLength} long'
        print(msg)
        Utilities.writeLogFile(msg)

        # Delete the records in badTime ranges from each dataset in the group
        finalCount = 0
        originalLength = len(timeStamp)
        for dateTime in badTimes:
            # Need to reinitialize for each loop
            startLength = len(timeStamp)
            newTimeStamp = []

            start = dateTime[0]
            stop = dateTime[1]

            # msg = f'Eliminate data between: {dateTime}'
            # print(msg)
            # Utilities.writeLogFile(msg)

            if startLength > 0:
                rowsToDelete = []
                for i in range(startLength):
                    if start <= timeStamp[i] and stop >= timeStamp[i]:
                        rowsToDelete.append(i)
                        finalCount += 1
                    else:
                        newTimeStamp.append(timeStamp[i])
                group.datasetDeleteRow(rowsToDelete)
            else:
                msg = 'Data group is empty. Continuing.'
                print(msg)
                Utilities.writeLogFile(msg)
            timeStamp = newTimeStamp.copy()

        msg = f'   Length of records removed from dataset: {finalCount}'
        print(msg)
        Utilities.writeLogFile(msg)

        return finalCount/originalLength

    @staticmethod
    def filterData_ADJUSTED(group, badTimes):
        ''' Delete additional flagged records for Sea-Bird Darks with adjusted 
            timestamps to match Lights '''

        msg = f'Remove {group.id} Data'
        print(msg)
        Utilities.writeLogFile(msg)

        timeStamp = group.getDataset("DATETIME_ADJUSTED").data

        startLength = len(timeStamp)
        msg = f'   Length of dataset prior to removal {startLength} long'
        print(msg)
        Utilities.writeLogFile(msg)

        # Delete the records in badTime ranges from each dataset in the group
        finalCount = 0
        originalLength = len(timeStamp)
        for dateTime in badTimes:
            # Need to reinitialize for each loop
            startLength = len(timeStamp)
            newTimeStamp = []

            start = dateTime[0]
            stop = dateTime[1]

            # msg = f'Eliminate data between: {dateTime}'
            # print(msg)
            # Utilities.writeLogFile(msg)

            if startLength > 0:
                rowsToDelete = []
                for i in range(startLength):
                    if start <= timeStamp[i] and stop >= timeStamp[i]:
                        rowsToDelete.append(i)
                        finalCount += 1
                    else:
                        newTimeStamp.append(timeStamp[i])
                group.datasetDeleteRow(rowsToDelete)
            else:
                msg = 'Data group is empty. Continuing.'
                print(msg)
                Utilities.writeLogFile(msg)
            timeStamp = newTimeStamp.copy()

        msg = f'   Length of records removed from dataset: {finalCount}'
        print(msg)
        Utilities.writeLogFile(msg)

        return finalCount/originalLength

    @staticmethod
    def renameGroup(gp, cf):
        ''' Rename the groups to more generic ids rather than the names of the cal files '''

        #NOTE: Attn. DALEC and SoRAD, you can update your group names here.
        if gp.id.startswith("GPRMC") or gp.id.startswith("GPGAA"):
            gp.id = "GPS"
        if ConfigFile.settings['SensorType'].lower() == 'seabird':
            if gp.id.startswith("UMTWR"):
                gp.id = "SunTracker_pySAS"
            if gp.id.startswith("SATNAV"):
                gp.id = "SunTracker_SOLARTRACKER"
            if gp.id.startswith("SATMSG"):
                gp.id = "SOLARTRACKER_STATUS"
            if gp.id.startswith("SATPYR"):
                gp.id = "PYROMETER"
            if gp.id.startswith("HED"):
                gp.id = "ES_DARK"
            if gp.id.startswith("HSE"):
                gp.id = "ES_LIGHT"
            if gp.id.startswith("HLD"):
                if cf.sensorType == "LI":
                    gp.id = "LI_DARK"
                if cf.sensorType == "LT":
                    gp.id = "LT_DARK"
            if gp.id.startswith("HSL"):
                if cf.sensorType == "LI":
                    gp.id = "LI_LIGHT"
                if cf.sensorType == "LT":
                    gp.id = "LT_LIGHT"
        else:
            gp.id = cf.sensorType

    @staticmethod
    def processL1aqc(node, calibrationMap, ancillaryData=None):
        '''
        Filters data for tilt, yaw, rotator, and azimuth. Deglitching
        '''

        node.attributes["PROCESSING_LEVEL"] = "1aqc"
        now = datetime.datetime.now()
        timestr = now.strftime("%d-%b-%Y %H:%M:%S")
        node.attributes["FILE_CREATION_TIME"] = timestr

        # Add configuration parameters (L1C) to root/node attributes
        if ConfigFile.settings['bL1aqcSunTracker']:
            node.attributes['SunTracker'] = 'YES'

        if ConfigFile.settings['bL1aqcCleanPitchRoll']:
            node.attributes['PITCH_ROLL_FILTER'] = ConfigFile.settings['fL1aqcPitchRollPitch']
        node.attributes['HOME_ANGLE'] = ConfigFile.settings['fL1aqcRotatorHomeAngle']
        node.attributes['ROTATOR_DELAY_FILTER'] = ConfigFile.settings['fL1aqcRotatorDelay']
        if ConfigFile.settings['bL1aqcRotatorAngle']:
            node.attributes['ROT_ANGLE_MIN'] = ConfigFile.settings['fL1aqcRotatorAngleMin']
            node.attributes['ROT_ANGLE_MAX'] = ConfigFile.settings['fL1aqcRotatorAngleMax']
        if ConfigFile.settings['bL1aqcCleanSunAngle']:
            node.attributes['RELATIVE_AZIMUTH_MIN'] = ConfigFile.settings['fL1aqcSunAngleMin']
            node.attributes['RELATIVE_AZIMUTH_MAX'] = ConfigFile.settings['fL1aqcSunAngleMax']

        msg = f"ProcessL1aqc.processL1aqc: {timestr}"
        print(msg)
        Utilities.writeLogFile(msg)

        # Reorganize groups in with new names
        # if ConfigFile.settings['SensorType'].lower() == 'seabird':
        for gp in node.groups:
            cf = calibrationMap[gp.attributes["CalFileName"]]
            ProcessL1aqc.renameGroup(gp,cf)

        # Add a dataset to each group for DATETIME, as defined by TIMETAG2 and DATETAG
        node  = Utilities.rootAddDateTime(node)

        # 2021-04-09: Include ANCILLARY_METADATA in all datasets, regardless of whether they are SunTracker or not
        # or if they have an ancillary file or not
        compass = None
        esDateTime = None
        gpsStatus = None
        ancTimeTag2 = None
        ancDateTag = None
        for gp in node.groups:
            if gp.id.startswith('GP'):

                # NOTE: Do So-Rad and DALEC use a group in L1A called GP*, like GPS or GPRMC?
                #   NOTE: If not, make changes here to pick up GPS data from the appropriate group.

                gpsDateTime = gp.getDataset('DATETIME').data
                gpsLat = gp.getDataset('LATPOS')
                latHemiData = gp.getDataset('LATHEMI')
                gpsLon = gp.getDataset('LONPOS')
                lonHemiData = gp.getDataset('LONHEMI')

                ancTimeTag2 = [Utilities.datetime2TimeTag2(dt) for dt in gpsDateTime]
                ancDateTag = [Utilities.datetime2DateTag(dt) for dt in gpsDateTime]

                latAnc = []
                lonAnc = []
                for i in range(gpsLat.data.shape[0]):
                    latDM = gpsLat.data["NONE"][i]
                    latDirection = latHemiData.data["NONE"][i]
                    latDD = Utilities.dmToDd(latDM, latDirection)
                    lonDM = gpsLon.data["NONE"][i]
                    lonDirection = lonHemiData.data["NONE"][i]
                    lonDD = Utilities.dmToDd(lonDM, lonDirection)
                    latAnc.append(latDD)
                    lonAnc.append(lonDD)

                if gp.attributes['CalFileName'].startswith('GPRMC'):
                    gpsStatus = gp.getDataset('STATUS')
                else:
                    gpsStatus = gp.getDataset('FIXQUAL')

            elif gp.id.startswith('ES'):
                esDateTime = gp.getDataset('DATETIME').data
            elif gp.id.startswith('SATTHS'):
                # Fluxgate on the THS:
                compass = gp.getDataset('COMP')

        # Solar geometry from GPS alone; No Tracker, no Ancillary
        relAzAnc = []
        if not ConfigFile.settings["bL1aqcSunTracker"] and not ancillaryData:
            # Only proceed if GPS is present
            if 'gpsDateTime' in locals():
                # Solar geometry is preferentially acquired from SunTracker or pySAS
                # Otherwise resorts to ancillary data. Otherwise processing fails.
                # Run Pysolar to obtain solar geometry.
                sunAzimuthAnc = []
                sunZenithAnc = []
                for i, dt_utc in enumerate(gpsDateTime):
                    sunAzimuthAnc.append(get_azimuth(latAnc[i],lonAnc[i],dt_utc,0)) # latAnc lonAnc from GPS, not ancillary file
                    sunZenithAnc.append(90 - get_altitude(latAnc[i],lonAnc[i],dt_utc,0))

                # SATTHS fluxgate compass on SAS
                if compass is None:
                    msg = 'Required ancillary data for sensor offset missing. Abort.'
                    print(msg)
                    Utilities.writeLogFile(msg)
                    return None
                else:
                    relAzAnc = compass - sunAzimuthAnc
            else:
                # TODO: Make sure DALEC, So-Rad can get the datetime from the GPS.
                msg = 'Required GPS data is missing. Check tdf files and ancillary data. Abort.'
                print(msg)
                Utilities.writeLogFile(msg)
                return None

        shipAzimuth = None
        station = None
        salt = None
        sst = None
        wind = None
        aod = None
        cloud = None
        wave = None
        speed_f_w = None
        pitch = None
        roll = None
        # If ancillary file is provided, use it. Otherwise fill in what you can using the datetime, lat, lon from GPS
        if ancillaryData is not None:

            # Remove all ancillary data that does not intersect GPS data
            # Change this to ES to work around set-ups with no GPS
            ancData = HDFDataset()
            ancData.attributes = copy.deepcopy(ancillaryData.attributes)#.copy()
            ancData.data = copy.deepcopy(ancillaryData.data)#.copy()
            # ancData.data = np.copy(ancillaryData.data)
            ancData.datasetToColumns()
            ancData.id = 'AncillaryData'

            ancDateTime = ancData.columns["DATETIME"][0].copy()

            print('Removing non-pertinent ancillary data.')
            # lower = bisect.bisect_left(ancDateTime, min(esDateTime))
            # First ancillary element during radiometry
            # lower = np.abs([d-min(esDateTime) for d in ancDateTime]).argmin()
            lower = Utilities.find_nearest(ancDateTime,min(esDateTime))
            lower = list(range(0,lower-1)) # keep one before that
            # upper = bisect.bisect_right(ancDateTime, max(esDateTime))+1
            # Last ancillary element during radiometry
            # upper = np.abs([d-max(esDateTime) for d in ancDateTime]).argmin()
            upper = Utilities.find_nearest(ancDateTime,max(esDateTime))
            upper = list(range(upper+2,len(ancDateTime))) # keep one after that
            ancData.colDeleteRow(upper)
            ancData.colDeleteRow(lower)

            # Test if any data remaining
            if not ancData.columns["DATETIME"][0]:
                msg = "No coincident ancillary data found. Check ancillary file. Aborting"
                print(msg)
                Utilities.writeLogFile(msg)
                return None

            # Reinitialize with new, smaller dataset
            # NOTE: Essential ancillary data for non-SunTracker file includes
            #    lat, lon, datetime, ship heading and offset between bow and
            #    SAS instrument from which SAS azimuth is calculated.
            #    Alternatively, can use the azimuth of the SAS from fluxgate
            #    compass (e.g., SATTHS) as a last resort

            timeStamp = ancData.columns["DATETIME"][0]
            ancTimeTag2 = [Utilities.datetime2TimeTag2(dt) for dt in timeStamp]
            ancDateTag = [Utilities.datetime2DateTag(dt) for dt in timeStamp]
            latAnc = ancData.columns["LATITUDE"][0]
            lonAnc = ancData.columns["LONGITUDE"][0]

            # Solar geometry is preferentially acquired from SunTracker or pySAS
            # Otherwise resorts to ancillary data. Otherwise processing fails.
            # Run Pysolar to obtain solar geometry.
            sunAzimuthAnc = []
            sunZenithAnc = []
            for i, dt_utc in enumerate(timeStamp):
                sunAzimuthAnc.append(get_azimuth(latAnc[i],lonAnc[i],dt_utc,0))
                sunZenithAnc.append(90 - get_altitude(latAnc[i],lonAnc[i],dt_utc,0))

            # relAzAnc either from ancillary relZz, ancillary sensorAz, (or THS compass above ^^)
            relAzAnc = []
            if "REL_AZ" in ancData.columns:
                relAzAnc = ancData.columns["REL_AZ"][0]

            sasAzAnc = []
            # This is a new SeaBASS field
            if "SENSOR_AZ" in ancData.columns:
                sasAzAnc = ancData.columns["SENSOR_AZ"][0]

            if ConfigFile.settings['SensorType'].lower() != 'trios es only':
                if not ConfigFile.settings["bL1aqcSunTracker"] and not relAzAnc and not sasAzAnc:
                    msg = 'Required ancillary sensor geometries missing or incorrect ancillary file used. Abort.'
                    print(msg)
                    Utilities.writeLogFile(msg)
                    return None
                elif not ConfigFile.settings["bL1aqcSunTracker"] and not relAzAnc:
                    # Corrected below for +/- solar-sensor orientation
                    relAzAnc = []
                    for i, sasAz in enumerate(sasAzAnc):
                        relAzAnc.append(sasAz - sunAzimuthAnc[i])

            if "HEADING" in ancData.columns:
                # HEADING/shipAzimuth comes from ancillary data file here (not GPS or SATNAV)
                shipAzimuth = ancData.columns["HEADING"][0]
            if "STATION" in ancData.columns:
                station = ancData.columns["STATION"][0]
            if "SALINITY" in ancData.columns:
                salt = ancData.columns["SALINITY"][0]
            if "SST" in ancData.columns:
                sst = ancData.columns["SST"][0]
            if "WINDSPEED" in ancData.columns:
                wind = ancData.columns["WINDSPEED"][0]
            if "AOD" in ancData.columns:
                aod = ancData.columns["AOD"][0]
            if "CLOUD" in ancData.columns:
                cloud = ancData.columns["CLOUD"][0]
            if "WAVE_HT" in ancData.columns:
                wave = ancData.columns["WAVE_HT"][0]
            if "SPEED_F_W" in ancData.columns:
                speed_f_w = ancData.columns["SPEED_F_W"][0]
            if "PITCH" in ancData.columns:
                pitch = ancData.columns["PITCH"][0]
            if "ROLL" in ancData.columns:
                roll = ancData.columns["ROLL"][0]

        else:
            # If no ancillary file is provided, create an ancillary group from GPS
            # Generate HDFDataset
            # NOTE: Where no ancillary data file is present, GPS data required.
            #   NOTE: Autonomous SUNTRACKER systems (pySAS, soRad, DALEC, etc.) should have gpsDateTime here.
            ancData = None
            ancillaryData = HDFDataset()
            ancillaryData.id = "AncillaryData"

            timeStamp = gpsDateTime
            ancillaryData.appendColumn("DATETIME", gpsDateTime)

            ancillaryData.appendColumn("LATITUDE", latAnc)
            ancillaryData.attributes["LATITUDE_UNITS"]='degrees'
            ancillaryData.appendColumn("LONGITUDE", lonAnc)
            ancillaryData.attributes["LONGITUDE_UNITS"]='degrees'

        if ConfigFile.settings['SensorType'].lower() == 'trios es only' and not ancillaryData:
            msg = "Required ancillary data for GPS location with sensor 'TriOS Es Only'. Abort."
            print(msg)
            Utilities.writeLogFile(msg, ConfigFile)
            return None

        if not ConfigFile.settings["bL1aqcSunTracker"] and not ancillaryData:
            msg = 'Required ancillary metadata for sensor offset missing. Abort.'
            print(msg)
            Utilities.writeLogFile(msg)
            return None

        if relAzAnc:
            # Correct relAzAnc to reflect an angle from the sun to the sensor, positive (+) clockwise
            relAzAnc = np.array(relAzAnc)
            relAzAnc[relAzAnc>180] = relAzAnc[relAzAnc>180] - 360
            relAzAnc[relAzAnc<-180] = relAzAnc[relAzAnc<-180] + 360
            relAzAnc.tolist()

        #############################################################################################
        # Begin Filtering
        #############################################################################################

        # For QC with badTimes when badTimes has only one record per interval (rather than a time span),
        #   L1AQC dark timestamps for Sea-Bird will not be captured by the exact badTimes. Therefore, set each dark
        #   sample timestamp to the nearest light sample.
        if ConfigFile.settings["SensorType"].lower() == "seabird":
            groupDict = {}
            for iGp, gp in enumerate(node.groups):
                groupDict[gp.id] = iGp
            for gp in node.groups:
                if "DARK" in gp.id:
                    if "LI" in gp.id:
                        lightGroup = node.groups[groupDict["LI_LIGHT"]]
                    elif "LT" in gp.id:
                        lightGroup = node.groups[groupDict["LT_LIGHT"]]
                    elif "ES" in gp.id:
                        lightGroup = node.groups[groupDict["ES_LIGHT"]]
                    else:
                        lightGroup = None

                    gp = Utilities.fixDarkTimes(gp,lightGroup)

        # Aggregate all bad times prior to removal (Restart badTimes after first filter,
        #       Suntracker-Es overlap, which uses a different datetime reference (Es instead of Ancillary))
        badTimes = []

        # Apply Filter for lack of Suntracker data while Es is collecting
        if node is not None and ConfigFile.settings["bL1aqcSunTracker"]:
            sunTrackerDateTime = None
            esDateTime = None
            for group in node.groups:
                if group.id.startswith("SunTracker"):
                    sunTrackerDateTime = group.datasets['DATETIME'].data
                if group.id.startswith("ES"):
                    esDateTime = group.datasets['DATETIME'].data

            msg = "Filtering file for Suntracker data outages"
            print(msg)
            Utilities.writeLogFile(msg)

            i = 0
            start = -1
            stop = None
            index = None
            # Threshold for an Es datetime distance from a Suntracker datetime:
            tThreshold = datetime.timedelta(seconds=30)

            for index, esTimeI in enumerate(esDateTime):

                tDiff = [abs(x - esTimeI) for x in sunTrackerDateTime]
                if min(tDiff) > tThreshold:
                    i += 1
                    if start == -1:
                        start = index
                    stop = index
                else:
                    if start != -1:
                        startstop = [esDateTime[start],esDateTime[stop]]
                        msg = f'   Flag data from {startstop[0]} to {startstop[1]}'
                        print(msg)
                        Utilities.writeLogFile(msg)
                        badTimes.append(startstop)
                        start = -1

            if start != -1 and stop == index: # Records from a mid-point to the end are bad
                startstop = [esDateTime[start],esDateTime[stop]]
                badTimes.append(startstop)
                msg = f'   Flag additional data from {startstop[0]} to {startstop[1]}'
                # print(msg)
                Utilities.writeLogFile(msg)

            msg = f'Percentage of data failed on Suntracker outage: {round(100*i/len(esDateTime))} %'
            print(msg)
            Utilities.writeLogFile(msg)

            if start==0 and stop==index: # All records are bad
                return None

        node = ProcessL1aqc.filterBadTimes(node,badTimes)
        if node is None:
            return None

        badTimes = []
        startstop = None

        # Apply GPS Status Filter
        # NOTE: I believe this is for an old dataset with GPGGA GPS with spotty reception.
        #   Does not appear to apply to current instruments (i.e., no gp.id=="GPS")
        gps = False
        for gp in node.groups:
            if gp.id == "GPS":
                gps = True
                timeStamp = gp.getDataset("DATETIME").data

        if gps:
            msg = "Filtering file for GPS status"
            print(msg)
            Utilities.writeLogFile(msg)

            i = 0
            start = -1
            stop = None
            index = None
            for index, status in enumerate(gpsStatus.data["NONE"]):
                # "V" for GPRMC, "0" for GPGGA
                if status == b'V' or status == 0:
                    i += 1
                    if start == -1:
                        start = index
                    stop = index
                else:
                    if start != -1:
                        startstop = [timeStamp[start],timeStamp[stop]]
                        msg = f'   Flag data from {startstop[0]} to {startstop[1]}'
                        # print(msg)
                        Utilities.writeLogFile(msg)
                        badTimes.append(startstop)
                        start = -1

            if start != -1 and stop == index: # Records from a mid-point to the end are bad
                startstop = [timeStamp[start],timeStamp[stop]]
                badTimes.append(startstop)
                msg = f'   Flag additional data from {startstop[0]} to {startstop[1]}'
                # print(msg)
                Utilities.writeLogFile(msg)
            msg = f'Percentage of data failed on GPS Status: {round(100*i/len(timeStamp))} %'
            print(msg)
            Utilities.writeLogFile(msg)

            if start==0 and stop==index: # All records are bad
                return None


        # Apply Pitch & Roll Filter
        # This has to record the time interval (in datetime) for the bad angles in order to remove these time intervals
        # rather than indexed values gleaned from SATNAV, since they have not yet been interpolated in time.
        # Interpolating them first would introduce error.

        if node is not None and int(ConfigFile.settings["bL1aqcCleanPitchRoll"]) == 1:
            msg = "Filtering file for high pitch and roll or pre/post-tilt"
            print(msg)
            Utilities.writeLogFile(msg)

            # Preferentially read PITCH and ROLL from SunTracker/pySAS THS sensor...
            pitch, roll = None, None
            tilt_pre, tilt_post = None, None
            for gp in node.groups:
                # NOTE: SOLARTRACKER (not pySAS) and DALEC use SunTracker group for PITCH/ROLL
                if gp.id.startswith("SunTracker") and "PITCH" in gp.datasets and "ROLL" in gp.datasets:
                        timeStamp = gp.getDataset("DATETIME").data
                        pitch = gp.getDataset("PITCH").data["SAS"]
                        roll = gp.getDataset("ROLL").data["SAS"]
                        break
                 # For SATTHS without SunTracker (i.e. with pySAS)
                if gp.id.startswith('SATTHS') and "PITCH" in gp.datasets and "ROLL" in gp.datasets:
                        timeStamp = gp.getDataset("DATETIME").data
                        pitch = gp.getDataset("PITCH").data["NONE"]
                        roll = gp.getDataset("ROLL").data["NONE"]
                        break
                if (ConfigFile.settings["SensorType"].lower() == "trios es only" and
                        gp.id.startswith('ES') and "TILT_PRE" in gp.datasets and "TILT_POST" in gp.datasets):
                    # TriOS RAMSES G2 have an integrated tilt sensor and
                    # each frame has tilt pre- and post-radiometric measurement
                    timeStamp = gp.getDataset("DATETIME").data
                    tilt_pre = gp.getDataset("TILT_PRE").data["NONE"]
                    tilt_post = gp.getDataset("TILT_POST").data["NONE"]

            # ...and failing that, try to pull from ancillary data
            if (pitch is None or roll is None) and (tilt_pre is None or tilt_post is None):
                if "PITCH" in ancData.columns:
                    msg = "Pitch data from ancillary file used."
                    print(msg)
                    Utilities.writeLogFile(msg)
                    pitch = ancData.columns["PITCH"][0]
                    if "ROLL" in ancData.columns:
                        msg = "Roll data from ancillary file used."
                        print(msg)
                        Utilities.writeLogFile(msg)
                        roll = ancData.columns["ROLL"][0]
                    timeStamp = ancData.columns["DATETIME"][0]
                else:
                    msg = "Pitch and roll data not found for tilt sensor or in Ancillary Data.\n"
                    msg = msg + " Try adding to Ancillary Data or turning off tilt filter. Aborting."
                    print(msg)
                    Utilities.writeLogFile(msg)
                    return None

            pitchMax = float(ConfigFile.settings["fL1aqcPitchRollPitch"])
            rollMax = float(ConfigFile.settings["fL1aqcPitchRollRoll"])

            if pitch is not None and roll is not None:
                i = 0
                start = -1
                stop = None
                for index, pitchi in enumerate(pitch):
                    if abs(pitchi) > pitchMax or abs(roll[index]) > rollMax:
                        i += 1
                        if start == -1:
                            # print('Pitch or roll angle outside bounds. Pitch: ' + str(round(pitch[index])) + ' Roll: ' +str(round(pitch[index])))
                            start = index
                        stop = index
                    else:
                        if start != -1:
                            # print('Pitch or roll angle passed. Pitch: ' + str(round(pitch[index])) + ' Roll: ' +str(round(pitch[index])))
                            startstop = [timeStamp[start],timeStamp[stop]]
                            msg = f'   Flag data from {startstop[0]} to {startstop[1]}'
                            # print(msg)
                            Utilities.writeLogFile(msg)
                            badTimes.append(startstop)
                            start = -1

                if start != -1 and stop == index: # Records from a mid-point to the end are bad
                    startstop = [timeStamp[start],timeStamp[stop]]
                    badTimes.append(startstop)
                    msg = f'   Flag data from {startstop[0]} to {startstop[1]} '
                    # print(msg)
                    Utilities.writeLogFile(msg)

                if start == 0 and stop == index:  # All records are bad
                    msg = 'All records are outside pitch/roll bounds. Aborting.'
                    print(msg)
                    Utilities.writeLogFile(msg)
                    return None

                msg = f'Percentage of data out of Pitch/Roll bounds: {round(100 * i / len(timeStamp))} %'
            elif tilt_pre is not None and tilt_post is not None:
                max_tilt = min(pitchMax, rollMax)
                bad_counter = 0
                for t, pre, post in zip(timeStamp, tilt_pre, tilt_post):
                    if abs(pre) > max_tilt or abs(post) > max_tilt:
                        startstop = [t, t]  # use edge case of ProcessL1aqc.filterData to remove this record
                        badTimes.append(startstop)
                        bad_counter += 1
                msg = f'Percentage of data out of pre/post-tilt bounds: {round(100 * bad_counter / len(timeStamp))} %'
            else:
                raise AssertionError('Neither pitch and roll or tilt_pre and tilt_post found.')

            print(msg)
            Utilities.writeLogFile(msg)

        # Apply Rotator Delay Filter (delete records within so many seconds of a rotation)
        # This has to record the time interval (TT2) for the bad angles in order to remove these time intervals
        # rather than indexed values gleaned from SATNAV, since they have not yet been interpolated in time.
        # Interpolating them first would introduce error.
        home = float(ConfigFile.settings["fL1aqcRotatorHomeAngle"])
        if node is not None and ConfigFile.settings["bL1aqcRotatorDelay"] and ConfigFile.settings["bL1aqcSunTracker"]:
            gp = None
            for group in node.groups:
                # NOTE: SOLARTRACKER and pySAS using POINTING dataset to get rotator movements
                #   NOTE: DALEC also uses POINTING. SoRad has no POINTING and is slower acquisition; maybe exclude rotator delay for SoRad.
                if group.id.startswith("SunTracker"):
                    gp = group
                    break

            if gp is not None:
                if gp.getDataset("POINTING"):
                    timeStamp = gp.getDataset("DATETIME").data
                    rotator = gp.getDataset("POINTING").data["ROTATOR"]
                    # Rotator Home Angle Offset is generally set in the .sat file when setting up the SunTracker
                    # It may also be set for when no SunTracker is present and it's not included in the
                    # ancillary data, but that's not relevant here...
                    delay = float(ConfigFile.settings["fL1aqcRotatorDelay"])

                    # if node is not None and int(ConfigFile.settings["bL1aqcRotatorDelay"]) == 1:
                    msg = "Filtering file for Rotator Delay"
                    print(msg)
                    Utilities.writeLogFile(msg)

                    kickout = 0
                    time = None
                    start = None
                    startIndex = None
                    i = 0
                    for index, rotatori in enumerate(rotator):
                        if index == 0:
                            lastAngle = rotatori
                        else:
                            # if rotatori > (lastAngle + 0.05) or rotatori < (lastAngle - 0.05):
                            if rotatori > (lastAngle + 1.0) or rotatori < (lastAngle - 1.0):
                                i += 1
                                # Detect angle changed
                                start = timeStamp[index] # Should restart with every rotator change
                                # print('Rotator delay kick-out. ' + str(timeInt) )
                                if kickout == 0:
                                    startIndex = index # Should only change for new delay interval
                                lastAngle = rotatori
                                kickout = 1

                            else:
                                # Test if this is fL1aqcRotatorDelay seconds past a kick-out start
                                time = timeStamp[index]
                                if kickout==1 and time > (start + datetime.timedelta(0,delay)):
                                    # startstop = [timeStampTuple[startIndex],timeStampTuple[index-1]]
                                    startstop = [timeStamp[startIndex],timeStamp[index-1]]
                                    msg = f'   Flag data from {startstop[0]} to {startstop[1]}'
                                    # print(msg)
                                    Utilities.writeLogFile(msg)
                                    badTimes.append(startstop)
                                    kickout = 0
                                    startIndex = None
                                elif kickout ==1:
                                    i += 1

                    msg = f'Percentage of Tracker data out of Rotator Delay bounds: {round(100*i/len(timeStamp))} %'
                    print(msg)
                    Utilities.writeLogFile(msg)

                else:
                    msg = 'No POINTING data found. Filtering on rotator delay failed.'
                    print(msg)
                    Utilities.writeLogFile(msg)
                    return None
            else:
                msg = 'No solar tracker data found. Filtering on rotator delay failed.'
                print(msg)
                Utilities.writeLogFile(msg)
                return None


        # Apply Absolute Rotator Angle Filter
        # This has to record the time interval (TT2) for the bad angles in order to remove these time intervals
        # rather than indexed values gleaned from SATNAV, since they have not yet been interpolated in time.
        # Interpolating them first would introduce error.
        if node is not None and ConfigFile.settings["bL1aqcSunTracker"] and ConfigFile.settings["bL1aqcRotatorAngle"]:
            msg = "Filtering file for bad Absolute Rotator Angle"
            print(msg)
            Utilities.writeLogFile(msg)

            gp = None
            for group in node.groups:
                # NOTE: SOLARTRACKER and pySAS using POINTING dataset to get rotator movements
                #   NOTE: DALEC also uses POINTING. SoRad has no POINTING but should still have option to filter here, or?
                if group.id.startswith("SunTracker"):
                    gp = group
                    break

            if gp is not None:
                if gp.getDataset("POINTING"):
                    timeStamp = gp.getDataset("DATETIME").data
                    rotator = gp.getDataset("POINTING").data["ROTATOR"]
                    # Rotator Home Angle Offset is generally set in the .sat file when setting up the SunTracker
                    # It may also be set for when no SunTracker is present and it's not included in the
                    # ancillary data, but that's not relevant here
                    home = float(ConfigFile.settings["fL1aqcRotatorHomeAngle"])

                    absRotatorMin = float(ConfigFile.settings["fL1aqcRotatorAngleMin"])
                    absRotatorMax = float(ConfigFile.settings["fL1aqcRotatorAngleMax"])

                    i = 0
                    start = -1
                    stop = None
                    for index, rotatori in enumerate(rotator):
                        if rotatori + home > absRotatorMax or rotatori + home < absRotatorMin or math.isnan(rotatori):
                            i += 1
                            if start == -1:
                                # print('Absolute rotator angle outside bounds. ' + str(round(rotatori + home)))
                                start = index
                            stop = index
                        else:
                            if start != -1:
                                # print('Absolute rotator angle passed: ' + str(round(rotatori + home)))
                                startstop = [timeStamp[start],timeStamp[stop]]
                                msg = ('   Flag data from ' + str(startstop[0]) + ' to ' + str(startstop[1]))
                                # print(msg)
                                Utilities.writeLogFile(msg)

                                badTimes.append(startstop)
                                start = -1

                    if start != -1 and stop == index: # Records from a mid-point to the end are bad
                        startstop = [timeStamp[start],timeStamp[stop]]
                        msg = f'   Flag data from {startstop[0]} to {startstop[1]}'
                        # print(msg)
                        Utilities.writeLogFile(msg)
                        badTimes.append(startstop)

                    msg = f'Percentage of Tracker data out of Absolute Rotator bounds: {round(100*i/len(timeStamp))} %'
                    print(msg)
                    Utilities.writeLogFile(msg)

                    if start==0 and stop==index: # All records are bad
                        return None
                else:
                    msg = 'No rotator data found. Filtering on absolute rotator angle failed.'
                    print(msg)
                    Utilities.writeLogFile(msg)
                    return None

        # General setup for ancillary or SunTracker data prior to Relative Solar Azimuth option
        if ConfigFile.settings["bL1aqcSunTracker"]:
            # Solar geometry is preferentially acquired from SunTracker or pySAS
            # Otherwise resorts to ancillary data. Otherwise processing fails.
            gp = None
            for group in node.groups:
                # NOTE: SOLARTRACKER and pySAS carry azimuth information in the SUNTRACKER group, but not RelAz
                #   NOTE: DALEC has RelAz in the SunTracker* group
                #       NOTE: SoRad has RelAz in the sorad group
                if group.id.startswith("SunTracker"):
                    gp = group
                    break

            if gp is not None:
                # TODO: Update datasets for DALEC/SoRAD to capture relAz
                if gp.getDataset("AZIMUTH") and gp.getDataset("HEADING") and gp.getDataset("POINTING"):
                    timeStamp = gp.getDataset("DATETIME").data
                    # Rotator Home Angle Offset is generally set in the .sat file when setting up the SunTracker
                    # It may also be set here for when no SunTracker is present and it's not included in the
                    # ancillary data. See below.
                    home = float(ConfigFile.settings["fL1aqcRotatorHomeAngle"])
                    sunAzimuth = gp.getDataset("AZIMUTH").data["SUN"]# strips off dtype name
                    gp.addDataset("SOLAR_AZ")
                    gp.datasets["SOLAR_AZ"].data = np.array(sunAzimuth, dtype=[('NONE', '<f8')])
                    # gp.datasets["SOLAR_AZ"].data = sunAzimuth
                    # sunAzimuth = sunAzimuth["SUN"]
                    del gp.datasets["AZIMUTH"]
                    sunZenith = 90 - gp.getDataset("ELEVATION").data["SUN"]
                    # sunZenith["None"] = 90 - sunZenith["SUN"]
                    gp.addDataset("SZA")
                    gp.datasets["SZA"].data = np.array(sunZenith, dtype=[('NONE', '<f8')])
                    # gp.datasets["SZA"].data = sunZenith
                    # sunZenith = sunZenith["SUN"] # strips off dtype name
                    del gp.datasets["ELEVATION"]
                    if gp.id == "SunTracker_SOLARTRACKER":
                        sasAzimuth = gp.getDataset("HEADING").data["SAS_TRUE"]
                    elif gp.id == "SunTracker_pySAS":
                        sasAzimuth = gp.getDataset("HEADING").data["SAS"]
                    else:
                        sasAzimuth = None

                    newRelAzData = gp.addDataset("REL_AZ")

                    relAz = sasAzimuth - sunAzimuth

                    # Correct relAzAnc to reflect an angle from the sun to the sensor, positive (+) clockwise
                    relAz[relAz>180] = relAz[relAz>180] - 360
                    relAz[relAz<-180] = relAz[relAz<-180] + 360
                else:
                    msg = "No rotator, solar azimuth, and/or ship'''s heading data found. Filtering on relative azimuth not added."
                    print(msg)
                    Utilities.writeLogFile(msg)
        else:
            relAz = relAzAnc

        # In case there is no SunTracker to provide sun/sensor geometries, Pysolar was used
        # to estimate sun zenith and azimuth using GPS position and time, and sensor azimuth will
        # come from ancillary data input or THS compass. For SunTracker and pySAS, SZA and solar azimuth go in the
        # SunTracker  group, otherwise in the ANCILLARY group.
        # REL_AZ will be pulled from SunTracker if available, otherwise from ANCILLARY
        # in ProcessL1bqc.

        # Initialize a new group to host the unconventional ancillary data
        ancGroup = node.addGroup("ANCILLARY_METADATA")
        # If using a SunTracker, add RelAz to the SunTracker group...
        if ConfigFile.settings["bL1aqcSunTracker"]:
            newRelAzData.columns["REL_AZ"] = relAz
            newRelAzData.columnsToDataset()
        else:
        #... otherwise populate the ancGroup
            if ConfigFile.settings['SensorType'].lower() != 'trios es only':
                ancGroup.addDataset("REL_AZ")
                ancGroup.datasets["REL_AZ"].data = np.array(relAzAnc, dtype=[('NONE', '<f8')])
                ancGroup.attributes["REL_AZ_UNITS"]='degrees'
            ancGroup.addDataset("SOLAR_AZ")
            ancGroup.attributes["SOLAR_AZ_UNITS"]='degrees'
            ancGroup.datasets["SOLAR_AZ"].data = np.array(sunAzimuthAnc, dtype=[('NONE', '<f8')])
            ancGroup.addDataset("SZA")
            ancGroup.datasets["SZA"].data = np.array(sunZenithAnc, dtype=[('NONE', '<f8')])
            ancGroup.attributes["SZA_UNITS"]='degrees'

        # Now include the remaining ancillary data in ancGroup with or w/out SunTracker
        ancGroup.addDataset("LATITUDE")
        ancGroup.datasets["LATITUDE"].data = np.array(latAnc, dtype=[('NONE', '<f8')])
        ancGroup.addDataset("LONGITUDE")
        ancGroup.datasets["LONGITUDE"].data = np.array(lonAnc, dtype=[('NONE', '<f8')])
        ancGroup.addDataset("TIMETAG2")
        ancGroup.datasets["TIMETAG2"].data = np.array(ancTimeTag2, dtype=[('NONE', '<f8')])
        ancGroup.addDataset("DATETAG")
        ancGroup.datasets["DATETAG"].data = np.array(ancDateTag, dtype=[('NONE', '<f8')])

        # Add datetime to Anc group
        dateTime = ancGroup.addDataset("DATETIME")
        timeData = ancGroup.getDataset("TIMETAG2").data["NONE"].tolist()
        dateTag = ancGroup.getDataset("DATETAG").data["NONE"].tolist()
        timeStampAnc = []
        for i, time in enumerate(timeData):
            # Converts from TT2 (hhmmssmss. UTC) and Datetag (YYYYDOY UTC) to datetime
            # Filter for aberrant Datetags
            if str(dateTag[i]).startswith("19") or str(dateTag[i]).startswith("20"):
                dt = Utilities.dateTagToDateTime(dateTag[i])
                timeStampAnc.append(Utilities.timeTag2ToDateTime(dt, time))
            else:
                ancGroup.datasetDeleteRow(i)
                msg = "Bad Datetag found in ancillary. Eliminating record"
                print(msg)
                Utilities.writeLogFile(msg)
        dateTime.data = timeStampAnc

        # For non-SunTracker datasets, define the timeStamp around the ancillary data
        # Otherwise, it was already defined above
        if not ConfigFile.settings['bL1aqcSunTracker']:
            # Convert datetimes
            timeStamp = timeStampAnc

        # Look for additional datasets in provided ancillaryData and populate the new ancillary group
        # if ancillaryData is not None:
        if ancData is not None:
            ancGroup.attributes = ancData.attributes.copy()
            if "HEADING" in ancData.columns:
                ancGroup.addDataset("HEADING")
                ancGroup.datasets["HEADING"].data = np.array(shipAzimuth, dtype=[('NONE', '<f8')])
            if "STATION" in ancData.columns:
                ancGroup.addDataset("STATION")
                ancGroup.datasets["STATION"].data = np.array(station, dtype=[('NONE', '<f8')])
            if "SALINITY" in ancData.columns:
                ancGroup.addDataset("SALINITY")
                ancGroup.datasets["SALINITY"].data = np.array(salt, dtype=[('NONE', '<f8')])
            if "SST" in ancData.columns:
                ancGroup.addDataset("SST")
                ancGroup.datasets["SST"].data = np.array(sst, dtype=[('NONE', '<f8')])
            if "WINDSPEED" in ancData.columns:
                ancGroup.addDataset("WINDSPEED")
                ancGroup.datasets["WINDSPEED"].data = np.array(wind, dtype=[('NONE', '<f8')])
            if "AOD" in ancData.columns:
                ancGroup.addDataset("AOD")
                ancGroup.datasets["AOD"].data = np.array(aod, dtype=[('NONE', '<f8')])
            if "CLOUD" in ancData.columns:
                ancGroup.addDataset("CLOUD")
                ancGroup.datasets["CLOUD"].data = np.array(cloud, dtype=[('NONE', '<f8')])
            if "WAVE_HT" in ancData.columns:
                ancGroup.addDataset("WAVE_HT")
                ancGroup.datasets["WAVE_HT"].data = np.array(wave, dtype=[('NONE', '<f8')])
            if "SPEED_F_W" in ancData.columns:
                ancGroup.addDataset("SPEED_F_W")
                ancGroup.datasets["SPEED_F_W"].data = np.array(speed_f_w, dtype=[('NONE', '<f8')])
            if "PITCH" in ancData.columns:
                ancGroup.addDataset("PITCH")
                ancGroup.datasets["PITCH"].data = np.array(ancData.columns["PITCH"][0], dtype=[('NONE', '<f8')])
            if "ROLL" in ancData.columns:
                ancGroup.addDataset("ROLL")
                ancGroup.datasets["ROLL"].data = np.array(ancData.columns["ROLL"][0], dtype=[('NONE', '<f8')])
        ancGroup.attributes["FrameType"] = "Not Required"
        ######################################################################################################

        # Apply Relative Azimuth filter
        # This has to record the time interval (TT2) for the bad angles in order to remove these time intervals
        # rather than indexed values gleaned from SATNAV, since they have not yet been interpolated in time.
        # Interpolating them first would introduce error.
        if node is not None and int(ConfigFile.settings["bL1aqcCleanSunAngle"]) == 1:
            msg = "Filtering file for bad Relative Solar Azimuth"
            print(msg)
            Utilities.writeLogFile(msg)

            relAzimuthMin = float(ConfigFile.settings["fL1aqcSunAngleMin"])
            relAzimuthMax = float(ConfigFile.settings["fL1aqcSunAngleMax"])

            i = 0
            start = -1
            stop = None
            # The length of relAz (and therefore the value of i) depends on whether ancillary
            #  data are used or SunTracker data
            # relAz and timeStamp are 1:1, but could be TRACKER or ANCILLARY
            for index, relAzi in enumerate(relAz):
                relAzimuthAngle = relAzi

                if abs(relAzimuthAngle) > relAzimuthMax or abs(relAzimuthAngle) < relAzimuthMin or math.isnan(relAzimuthAngle):
                    i += 1
                    if start == -1:
                        # print('Relative solar azimuth angle outside bounds. ' + str(round(relAzimuthAngle,2)))
                        start = index
                    stop = index # start is fixed and stop progresses until good data found.
                else:
                    # good data found
                    if start != -1:
                        # good data follows bad data
                        # print('Relative solar azimuth angle passed: ' + str(round(relAzimuthAngle,2)))
                        startstop = [timeStamp[start],timeStamp[stop]]
                        msg = f'   Flag data from: {startstop[0]}  to {startstop[1]}'
                        # print(msg)
                        Utilities.writeLogFile(msg)
                        badTimes.append(startstop)
                        start = -1

            if start != -1 and stop == index: # Records from a mid-point to the end are bad
                startstop = [timeStamp[start],timeStamp[stop]]
                msg = f'   Flag data from {startstop[0]} to {startstop[1]} '
                # print(msg)
                Utilities.writeLogFile(msg)
                badTimes.append(startstop)

            msg = f'Percentage of data out of Relative Solar Azimuth bounds: {round(100*i/len(relAz))} %'
            print(msg)
            Utilities.writeLogFile(msg)

            if start==0 and stop==index: # All records are bad
                msg = "All records out of bounds. Aborting."
                print(msg)
                Utilities.writeLogFile(msg)
                return None

        node = ProcessL1aqc.filterBadTimes(node,badTimes)
        if node is None:
            return None

        # Confirm that radiometry still overlaps in time
        if ConfigFile.settings["SensorType"].lower() != "trios es only":
            groupDict = {}
            for iGp, gp in enumerate(node.groups):
                groupDict[gp.id] = iGp

            esDateTime = None
            if ConfigFile.settings["SensorType"].lower() == "seabird":
                esDateTime = node.groups[groupDict["ES_LIGHT"]].datasets['DATETIME'].data
                liDateTime = node.groups[groupDict["LI_LIGHT"]].datasets['DATETIME'].data
                ltDateTime = node.groups[groupDict["LT_LIGHT"]].datasets['DATETIME'].data
            else:
                # Confirm for trios, sorad, dalec
                esDateTime = node.groups[groupDict["ES"]].datasets['DATETIME'].data
                liDateTime = node.groups[groupDict["LI"]].datasets['DATETIME'].data
                ltDateTime = node.groups[groupDict["LT"]].datasets['DATETIME'].data

            if (min(esDateTime) > max(liDateTime) or min(esDateTime) > max(ltDateTime) or
                max(esDateTime) < min(liDateTime) or max(esDateTime) < min(ltDateTime) or
                min(liDateTime) > max(ltDateTime) or min(liDateTime) > max(ltDateTime) or
                min(ltDateTime) > max(liDateTime) or min(ltDateTime) > max(liDateTime)):

                msg = 'Radiometry groups no longer overlap in time. Abort.'
                print(msg)
                Utilities.writeLogFile(msg)
                return None


        ###########################################################################
        # Now deglitch
        if ConfigFile.settings["SensorType"].lower() == "seabird":
            # Deglitching is handled seperately for lights and darks, so no need to filter on _ADJUSTED timestamps
            node = ProcessL1aqc_deglitch.processL1aqc_deglitch(node)
        else:
            node.attributes['L1AQC_DEGLITCH'] = 'OFF'

        # DATETIME is not supported in HDF5; remove
        if node is not None:
            for gp in node.groups:
                if 'DATETIME' in gp.datasets:
                    del gp.datasets["DATETIME"]
                if 'DATETIME_ADJUSTED' in gp.datasets:
                    del gp.datasets["DATETIME_ADJUSTED"]

        return node
