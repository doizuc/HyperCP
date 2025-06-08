''' Process L1BQC to L2 '''
import os
import collections
import warnings
import time

import datetime
import copy
import numpy as np
import scipy as sp

from PyQt5 import QtWidgets
from tqdm import tqdm

from Source.HDFRoot import HDFRoot
from Source.Utilities import Utilities
from Source.ConfigFile import ConfigFile
from Source.RhoCorrections import RhoCorrections
from Source.Uncertainty_Analysis import Propagate
from Source.Weight_RSR import Weight_RSR
from Source.ProcessL2OCproducts import ProcessL2OCproducts
from Source.ProcessL2BRDF import ProcessL2BRDF
from Source.ProcessInstrumentUncertainties import Trios, HyperOCR


class ProcessL2:
    ''' Process L2 '''

    @staticmethod
    def nirCorrectionSatellite(root, sensor, rrsNIRCorr, nLwNIRCorr):
        newReflectanceGroup = root.getGroup("REFLECTANCE")
        newRrsData = newReflectanceGroup.getDataset(f'Rrs_{sensor}')
        newnLwData = newReflectanceGroup.getDataset(f'nLw_{sensor}')

        # These will include all slices in root so far
        # Below the most recent/current slice [-1] will be selected for processing
        rrsSlice = newRrsData.columns
        nLwSlice = newnLwData.columns

        for k in rrsSlice:
            if (k != 'Datetime') and (k != 'Datetag') and (k != 'Timetag2'):
                rrsSlice[k][-1] -= rrsNIRCorr
        for k in nLwSlice:
            if (k != 'Datetime') and (k != 'Datetag') and (k != 'Timetag2'):
                nLwSlice[k][-1] -= nLwNIRCorr

        newRrsData.columnsToDataset()
        newnLwData.columnsToDataset()


    @staticmethod
    def nirCorrection(node, sensor, F0):
        # F0 is sensor specific, but ultimately, SimSpec can only be applied to hyperspectral data anyway,
        # so output the correction and apply it to satellite bands later.
        simpleNIRCorrection = int(ConfigFile.settings["bL2SimpleNIRCorrection"])
        simSpecNIRCorrection = int(ConfigFile.settings["bL2SimSpecNIRCorrection"])

        newReflectanceGroup = node.getGroup("REFLECTANCE")
        newRrsData = newReflectanceGroup.getDataset(f'Rrs_{sensor}')
        newnLwData = newReflectanceGroup.getDataset(f'nLw_{sensor}')
        newRrsUNCData = newReflectanceGroup.getDataset(f'Rrs_{sensor}_unc')
        newnLwUNCData = newReflectanceGroup.getDataset(f'nLw_{sensor}_unc')

        newNIRData = newReflectanceGroup.getDataset(f'nir_{sensor}')
        newNIRnLwData = newReflectanceGroup.getDataset(f'nir_nLw_{sensor}')

        # These will include all slices in node so far
        # Below the most recent/current slice [-1] will be selected for processing
        rrsSlice = newRrsData.columns
        nLwSlice = newnLwData.columns
        nirSlice = newNIRData.columns
        nirnLwSlice = newNIRnLwData.columns

        # # Perform near-infrared residual correction to remove additional atmospheric and glint contamination
        # if ConfigFile.settings["bL2PerformNIRCorrection"]:
        if simpleNIRCorrection:
            # Data show a minimum near 725; using an average from above 750 leads to negative reflectances
            # Find the minimum between 700 and 800, and subtract it from spectrum (spectrally flat)
            msg = "Perform simple residual NIR subtraction."
            print(msg)
            Utilities.writeLogFile(msg)

            # rrs correction
            NIRRRs = []
            for k in rrsSlice:
                if (k == 'Datetime') or (k == 'Datetag') or (k == 'Timetag2'):
                    continue
                if float(k) >= 700 and float(k) <= 800:
                    NIRRRs.append(rrsSlice[k][-1])
            rrsNIRCorr = min(NIRRRs)
            if rrsNIRCorr < 0:
                #   NOTE: SeaWiFS protocols for residual NIR were never intended to ADD reflectance
                #   This is most likely in blue, non-turbid waters not intended for NIR offset correction.
                #   Revert to NIR correction of 0 when this happens. No good way to update the L2 attribute
                #   metadata because it may only be on some ensembles within a file.
                msg = 'Bad NIR Correction. Revert to No NIR correction.'
                print(msg)
                Utilities.writeLogFile(msg)
                rrsNIRCorr = 0
            # Subtract average from each waveband
            for k in rrsSlice:
                if (k == 'Datetime') or (k == 'Datetag') or (k == 'Timetag2'):
                    continue

                rrsSlice[k][-1] -= rrsNIRCorr

            nirSlice['NIR_offset'].append(rrsNIRCorr)

            # nLw correction
            NIRRRs = []
            for k in nLwSlice:
                if (k == 'Datetime') or (k == 'Datetag') or (k == 'Timetag2'):
                    continue
                if float(k) >= 700 and float(k) <= 800:
                    NIRRRs.append(nLwSlice[k][-1])
            nLwNIRCorr = min(NIRRRs)
            # Subtract average from each waveband
            for k in nLwSlice:
                if (k == 'Datetime') or (k == 'Datetag') or (k == 'Timetag2'):
                    continue
                nLwSlice[k][-1] -= nLwNIRCorr

            nirnLwSlice['NIR_offset'].append(nLwNIRCorr)

        elif simSpecNIRCorrection:
            # From Ruddick 2005, Ruddick 2006 use NIR normalized similarity spectrum
            # (spectrally flat)
            msg = "Perform similarity spectrum residual NIR subtraction."
            print(msg)
            Utilities.writeLogFile(msg)

            # For simplicity, follow calculation in rho (surface reflectance), then covert to rrs
            ρSlice = copy.deepcopy(rrsSlice)
            for k,value in ρSlice.items():
                if (k == 'Datetime') or (k == 'Datetag') or (k == 'Timetag2'):
                    continue
                ρSlice[k][-1] = value[-1] * np.pi

            # These ratios are for rho = pi*Rrs
            α1 = 2.35 # 720/780 only good for rho(720)<0.03
            α2 = 1.91 # 780/870 try to avoid, data is noisy here
            threshold = 0.03

            # Retrieve TSIS-1s
            wavelength = [float(key) for key in F0.keys()]
            F0 = [value for value in F0.values()]

            # Rrs
            ρ720 = []
            x = []
            for k in ρSlice:
                if (k == 'Datetime') or (k == 'Datetag') or (k == 'Timetag2'):
                    continue
                if float(k) >= 700 and float(k) <= 750:
                    x.append(float(k))

                    # convert to surface reflectance ρ = π * Rrs
                    ρ720.append(ρSlice[k][-1]) # Using current element/slice [-1]

            # if not ρ720:
            #     print("Error: NIR wavebands unavailable")
            #     if os.environ["HYPERINSPACE_CMD"].lower() == 'false':
            #         QtWidgets.QMessageBox.critical("Error", "NIR wavebands unavailable")
            ρ1 = sp.interpolate.interp1d(x,ρ720)(720)
            F01 = sp.interpolate.interp1d(wavelength,F0)(720)
            ρ780 = []
            x = []
            for k in ρSlice:
                if k in ('Datetime', 'Datetag', 'Timetag2'):
                    continue
                if float(k) >= 760 and float(k) <= 800:
                    x.append(float(k))
                    ρ780.append(ρSlice[k][-1])
            if not ρ780:
                print("Error: NIR wavebands unavailable")
                if os.environ["HYPERINSPACE_CMD"].lower() == 'false':
                    QtWidgets.QMessageBox.critical("Error", "NIR wavebands unavailable")
            ρ2 = sp.interpolate.interp1d(x,ρ780)(780)
            F02 = sp.interpolate.interp1d(wavelength,F0)(780)
            ρ870 = []
            x = []
            for k in ρSlice:
                if k in ('Datetime', 'Datetag', 'Timetag2'):
                    continue
                if float(k) >= 850 and float(k) <= 890:
                    x.append(float(k))
                    ρ870.append(ρSlice[k][-1])
            if not ρ870:
                msg = 'No data found at 870 nm'
                print(msg)
                Utilities.writeLogFile(msg)
                ρ3 = None
                F03 = None
            else:
                ρ3 = sp.interpolate.interp1d(x,ρ870)(870)
                F03 = sp.interpolate.interp1d(wavelength,F0)(870)

            # Reverts to primary mode even on threshold trip in cases where no 870nm available
            if ρ1 < threshold or not ρ870:
                ε = (α1*ρ2 - ρ1)/(α1-1)
                εnLw = (α1*ρ2*F02 - ρ1*F01)/(α1-1)
                msg = f'offset(rrs) = {ε}; offset(nLw) = {εnLw}'
                print(msg)
                Utilities.writeLogFile(msg)
            else:
                msg = "SimSpec threshold tripped. Using 780/870 instead."
                print(msg)
                Utilities.writeLogFile(msg)
                ε = (α2*ρ3 - ρ2)/(α2-1)
                εnLw = (α2*ρ3*F03 - ρ2*F02)/(α2-1)
                msg = f'offset(rrs) = {ε}; offset(nLw) = {εnLw}'
                print(msg)
                Utilities.writeLogFile(msg)

            rrsNIRCorr = ε/np.pi
            nLwNIRCorr = εnLw/np.pi

            # Now apply to rrs and nLw
            # NOTE: This correction is also susceptible to a correction that ADDS to reflectance
            #   spectrally, depending on spectral shape (see test_SimSpec.m).
            #   This is most likely in blue, non-turbid waters not intended for SimSpec.
            #   Revert to NIR correction of 0 when this happens. No good way to update the L2 attribute
            #   metadata because it may only be on some ensembles within a file.
            if rrsNIRCorr < 0:
                msg = 'Bad NIR Correction. Revert to No NIR correction.'
                print(msg)
                Utilities.writeLogFile(msg)
                rrsNIRCorr = 0
                nLwNIRCorr = 0
                # L2 metadata will be updated

            for k in rrsSlice:
                if (k == 'Datetime') or (k == 'Datetag') or (k == 'Timetag2'):
                    continue

                rrsSlice[k][-1] -= float(rrsNIRCorr) # Only working on the last (most recent' [-1]) element of the slice
                nLwSlice[k][-1] -= float(nLwNIRCorr)


            nirSlice['NIR_offset'].append(rrsNIRCorr)
            nirnLwSlice['NIR_offset'].append(nLwNIRCorr)

        newRrsData.columnsToDataset()
        newnLwData.columnsToDataset()
        newRrsUNCData.columnsToDataset()
        newnLwUNCData.columnsToDataset()
        newNIRData.columnsToDataset()
        newNIRnLwData.columnsToDataset()

        return rrsNIRCorr, nLwNIRCorr


    @staticmethod
    def spectralReflectance(node, sensor, timeObj, xSlice, F0, F0_unc, rhoScalar, rhoVec, waveSubset, xUNC):
        ''' The slices, stds, F0, rhoVec here are sensor-waveband specific '''
        esXSlice = xSlice['es'] # mean
        esXmedian = xSlice['esMedian']
        esXRemaining = xSlice['esRemaining']
        esXstd = xSlice['esSTD']
        liXSlice = xSlice['li']
        liXmedian = xSlice['liMedian']
        liXRemaining = xSlice['liRemaining']
        liXstd = xSlice['liSTD']
        ltXSlice = xSlice['lt']
        ltXmedian = xSlice['ltMedian']
        ltXRemaining = xSlice['ltRemaining']
        ltXstd = xSlice['ltSTD']
        dateTime = timeObj['dateTime']
        dateTag = timeObj['dateTag']
        timeTag = timeObj['timeTag']

        threeCRho = int(ConfigFile.settings["bL23CRho"])
        ZhangRho = int(ConfigFile.settings["bL2ZhangRho"])

        # Root (new/output) groups:
        newReflectanceGroup = node.getGroup("REFLECTANCE")
        newRadianceGroup = node.getGroup("RADIANCE")
        newIrradianceGroup = node.getGroup("IRRADIANCE")

        # If this is the first ensemble spectrum, set up the new datasets
        if not f'Rrs_{sensor}' in newReflectanceGroup.datasets:
            newESData = newIrradianceGroup.addDataset(f"ES_{sensor}")
            newLIData = newRadianceGroup.addDataset(f"LI_{sensor}")
            newLTData = newRadianceGroup.addDataset(f"LT_{sensor}")
            newLWData = newRadianceGroup.addDataset(f"LW_{sensor}")

            newESDataMedian = newIrradianceGroup.addDataset(f"ES_{sensor}_median")
            newLIDataMedian = newRadianceGroup.addDataset(f"LI_{sensor}_median")
            newLTDataMedian = newRadianceGroup.addDataset(f"LT_{sensor}_median")

            newRrsData = newReflectanceGroup.addDataset(f"Rrs_{sensor}")
            newRrsUncorrData = newReflectanceGroup.addDataset(f"Rrs_{sensor}_uncorr") # Preserve uncorrected Rrs (= lt/es)
            newnLwData = newReflectanceGroup.addDataset(f"nLw_{sensor}")

            # September 2023. For clarity, drop the "Delta" nominclature in favor of
            # either STD (standard deviation of the sample) or UNC (uncertainty)
            newESSTDData = newIrradianceGroup.addDataset(f"ES_{sensor}_sd")
            newLISTDData = newRadianceGroup.addDataset(f"LI_{sensor}_sd")
            newLTSTDData = newRadianceGroup.addDataset(f"LT_{sensor}_sd")

            # No average (mean or median) or standard deviation values associated with Lw or reflectances,
            #   because these are calculated from the means of Lt, Li, Es

            newESUNCData = newIrradianceGroup.addDataset(f"ES_{sensor}_unc")
            newLIUNCData = newRadianceGroup.addDataset(f"LI_{sensor}_unc")
            newLTUNCData = newRadianceGroup.addDataset(f"LT_{sensor}_unc")
            newLWUNCData = newRadianceGroup.addDataset(f"LW_{sensor}_unc")
            newRrsUNCData = newReflectanceGroup.addDataset(f"Rrs_{sensor}_unc")
            newnLwUNCData = newReflectanceGroup.addDataset(f"nLw_{sensor}_unc")

            # Add standard deviation datasets for comparison
            newLWSTDData = newRadianceGroup.addDataset(f"LW_{sensor}_sd")
            newRrsSTDData = newReflectanceGroup.addDataset(f"Rrs_{sensor}_sd")
            newnLwSTDData = newReflectanceGroup.addDataset(f"nLw_{sensor}_sd")

            # For CV, use CV = STD/n

            if sensor == 'HYPER':
                newRhoHyper = newReflectanceGroup.addDataset(f"rho_{sensor}")
                newRhoUNCHyper = newReflectanceGroup.addDataset(f"rho_{sensor}_unc")
                if ConfigFile.settings["bL2PerformNIRCorrection"]:
                    newNIRData = newReflectanceGroup.addDataset(f'nir_{sensor}')
                    newNIRnLwData = newReflectanceGroup.addDataset(f'nir_nLw_{sensor}')
        else:
            newESData = newIrradianceGroup.getDataset(f"ES_{sensor}")
            newLIData = newRadianceGroup.getDataset(f"LI_{sensor}")
            newLTData = newRadianceGroup.getDataset(f"LT_{sensor}")
            newLWData = newRadianceGroup.getDataset(f"LW_{sensor}")

            newESDataMedian = newIrradianceGroup.getDataset(f"ES_{sensor}_median")
            newLIDataMedian = newRadianceGroup.getDataset(f"LI_{sensor}_median")
            newLTDataMedian = newRadianceGroup.getDataset(f"LT_{sensor}_median")

            newRrsData = newReflectanceGroup.getDataset(f"Rrs_{sensor}")
            newRrsUncorrData = newReflectanceGroup.getDataset(f"Rrs_{sensor}_uncorr")
            newnLwData = newReflectanceGroup.getDataset(f"nLw_{sensor}")

            newESSTDData = newIrradianceGroup.getDataset(f"ES_{sensor}_sd")
            newLISTDData = newRadianceGroup.getDataset(f"LI_{sensor}_sd")
            newLTSTDData = newRadianceGroup.getDataset(f"LT_{sensor}_sd")

            # No average (mean or median) or standard deviation values associated with Lw or reflectances,
            #   because these are calculated from the means of Lt, Li, Es

            newESUNCData = newIrradianceGroup.getDataset(f"ES_{sensor}_unc")
            newLIUNCData = newRadianceGroup.getDataset(f"LI_{sensor}_unc")
            newLTUNCData = newRadianceGroup.getDataset(f"LT_{sensor}_unc")
            newLWUNCData = newRadianceGroup.getDataset(f"LW_{sensor}_unc")
            newRrsUNCData = newReflectanceGroup.getDataset(f"Rrs_{sensor}_unc")
            newnLwUNCData = newReflectanceGroup.getDataset(f"nLw_{sensor}_unc")

            newLWSTDData = newRadianceGroup.getDataset(f"LW_{sensor}_sd")
            newRrsSTDData = newReflectanceGroup.getDataset(f"Rrs_{sensor}_sd")
            newnLwSTDData = newReflectanceGroup.getDataset(f"nLw_{sensor}_sd")

            if sensor == 'HYPER':
                newRhoHyper = newReflectanceGroup.getDataset(f"rho_{sensor}")
                newRhoUNCHyper = newReflectanceGroup.getDataset(f"rho_{sensor}_unc")
                if ConfigFile.settings["bL2PerformNIRCorrection"]:
                    newNIRData = newReflectanceGroup.getDataset(f'nir_{sensor}')
                    newNIRnLwData = newReflectanceGroup.addDataset(f'nir_nLw_{sensor}')

        # Add datetime stamps back onto ALL datasets associated with the current sensor
        # If this is the first spectrum, add date/time, otherwise append
        # Groups REFLECTANCE, IRRADIANCE, and RADIANCE are intiallized with empty datasets, but
        # ANCILLARY is not.
        if "Datetag" not in newRrsData.columns:
            for gp in node.groups:
                if gp.id == "ANCILLARY": # Ancillary is already populated. The other groups only have empty (named) datasets
                    continue
                else:
                    for ds in gp.datasets:
                        if sensor in ds: # Only add datetime stamps to the current sensor datasets
                            gp.datasets[ds].columns["Datetime"] = [dateTime] # mean of the ensemble datetime stamp
                            gp.datasets[ds].columns["Datetag"] = [dateTag]
                            gp.datasets[ds].columns["Timetag2"] = [timeTag]
        else:
            for gp in node.groups:
                if gp.id == "ANCILLARY":
                    continue
                else:
                    for ds in gp.datasets:
                        if sensor in ds:
                            gp.datasets[ds].columns["Datetime"].append(dateTime)
                            gp.datasets[ds].columns["Datetag"].append(dateTag)
                            gp.datasets[ds].columns["Timetag2"].append(timeTag)

        # Organise Uncertainty into wavebands
        lwUNC = {}
        rrsUNC = {}
        rhoUNC = {}
        esUNC = {}
        liUNC = {}
        ltUNC = {}

        # Only Factory - Trios has no uncertainty here
        if (ConfigFile.settings['bL1bCal'] >= 2 or ConfigFile.settings['SensorType'].lower() == 'seabird'):
            esUNC = xUNC[f'esUNC_{sensor}']  # should already be convolved to hyperspec
            liUNC = xUNC[f'liUNC_{sensor}']  # added reference to HYPER as band convolved uncertainties will no longer
            ltUNC = xUNC[f'ltUNC_{sensor}']  # overwite normal instrument uncertainties during processing
            rhoUNC = xUNC[f'rhoUNC_{sensor}']
            for i, wvl in enumerate(waveSubset):
                k = str(wvl)
                if (any([wvl == float(x) for x in esXSlice]) and
                        any([wvl == float(x) for x in liXSlice]) and
                        any([wvl == float(x) for x in ltXSlice])):  # More robust (able to handle sensor and hyper bands
                    if sensor == 'HYPER':
                        lwUNC[k] = xUNC['lwUNC'][i]
                        rrsUNC[k] = xUNC['rrsUNC'][i]
                    else:  # apply the sensor specific Lw and Rrs uncertainties
                        lwUNC[k] = xUNC[f'lwUNC_{sensor}'][i]
                        rrsUNC[k] = xUNC[f'rrsUNC_{sensor}'][i]
        else:
            # factory case
            for wvl in waveSubset:
                k = str(wvl)
                if (any([wvl == float(x) for x in esXSlice]) and
                        any([wvl == float(x) for x in liXSlice]) and
                        any([wvl == float(x) for x in ltXSlice])):  # old version had issues with '.0'
                    esUNC[k] = 0
                    liUNC[k] = 0
                    ltUNC[k] = 0
                    rhoUNC[k] = 0
                    lwUNC[k] = 0
                    rrsUNC[k] = 0

        deleteKey = []
        for i, wvl in enumerate(waveSubset):  # loop through wavebands
            k = str(wvl)
            if (any(wvl == float(x) for x in esXSlice) and
                    any(wvl == float(x) for x in liXSlice) and
                    any(wvl == float(x) for x in ltXSlice)):
                # Initialize the new dataset if this is the first slice
                if k not in newESData.columns:
                    newESData.columns[k] = []
                    newLIData.columns[k] = []
                    newLTData.columns[k] = []
                    newLWData.columns[k] = []
                    newRrsData.columns[k] = []
                    newRrsUncorrData.columns[k] = []
                    newnLwData.columns[k] = []

                    # No average (mean or median) or standard deviation values associated with Lw or reflectances,
                    #   because these are calculated from the means of Lt, Li, Es
                    newESDataMedian.columns[k] = []
                    newLIDataMedian.columns[k] = []
                    newLTDataMedian.columns[k] = []

                    newESSTDData.columns[k] = []
                    newLISTDData.columns[k] = []
                    newLTSTDData.columns[k] = []
                    newESUNCData.columns[k] = []
                    newLIUNCData.columns[k] = []
                    newLTUNCData.columns[k] = []
                    newLWUNCData.columns[k] = []
                    newRrsUNCData.columns[k] = []
                    newnLwUNCData.columns[k] = []

                    newLWSTDData.columns[k] = []
                    newRrsSTDData.columns[k] = []
                    newnLwSTDData.columns[k] = []

                    if sensor == 'HYPER':
                        newRhoHyper.columns[k] = []
                        newRhoUNCHyper.columns[k] = []
                        if ConfigFile.settings["bL2PerformNIRCorrection"]:
                            newNIRData.columns['NIR_offset'] = [] # not used until later; highly unpythonic
                            newNIRnLwData.columns['NIR_offset'] = []

                # At this waveband (k); still using complete wavelength set
                es = esXSlice[k][0] # Always the zeroth element; i.e. XSlice data are independent of past slices and node
                li = liXSlice[k][0]
                lt = ltXSlice[k][0]
                esRemaining = np.asarray(esXRemaining[k]) # array of remaining ensemble values in this band
                liRemaining = np.asarray(liXRemaining[k])
                ltRemaining = np.asarray(ltXRemaining[k])
                f0 = F0[k]
                f0UNC = F0_unc[k]

                esMedian = esXmedian[k][0]
                liMedian = liXmedian[k][0]
                ltMedian = ltXmedian[k][0]

                esSTD = esXstd[k][0]
                liSTD = liXstd[k][0]
                ltSTD = ltXstd[k][0]

                # Calculate the remote sensing reflectance
                nLwUNC = {}
                lwRemainingSD = 0
                rrsRemainingSD = 0
                nLwRemainingSD = 0
                if threeCRho:
                    lw = lt - (rhoScalar * li)
                    rrs = lw / es
                    nLw = rrs*f0

                    # Now calculate the std for lw, rrs
                    lwRemaining = ltRemaining - (rhoScalar * liRemaining)
                    rrsRemaining = lwRemaining / esRemaining
                    lwRemainingSD = np.std(lwRemaining)
                    rrsRemainingSD = np.std(rrsRemaining)
                    nLwRemainingSD = np.std(rrsRemaining*f0)

                elif ZhangRho:
                    # Only populate the valid wavelengths
                    if float(k) in waveSubset:
                        lw = lt - (rhoVec[k] * li)
                        rrs = lw / es
                        nLw = rrs*f0

                        # Now calculate the std for lw, rrs
                        lwRemaining = ltRemaining - (rhoVec[k] * liRemaining)
                        rrsRemaining = lwRemaining / esRemaining
                        lwRemainingSD = np.std(lwRemaining)
                        rrsRemainingSD = np.std(rrsRemaining)
                        nLwRemainingSD = np.std(rrsRemaining*f0)

                else:
                    lw = lt - (rhoScalar * li)
                    rrs = lw / es
                    nLw = rrs*f0

                    # Now calculate the std for lw, rrs
                    lwRemaining = ltRemaining - (rhoScalar * liRemaining)
                    rrsRemaining = lwRemaining / esRemaining
                    lwRemainingSD = np.std(lwRemaining)
                    rrsRemainingSD = np.std(rrsRemaining)
                    nLwRemainingSD = np.std(rrsRemaining*f0)
                

                # nLw uncertainty;
                nLwUNC[k] = np.power((rrsUNC[k]**2)*(f0**2) + (rrs**2)*(f0UNC**2), 0.5)

                newESData.columns[k].append(es)
                newLIData.columns[k].append(li)
                newLTData.columns[k].append(lt)

                rrs_uncorr = lt / es

                newESSTDData.columns[k].append(esSTD)
                newLISTDData.columns[k].append(liSTD)
                newLTSTDData.columns[k].append(ltSTD)

                newLWSTDData.columns[k].append(lwRemainingSD)
                newRrsSTDData.columns[k].append(rrsRemainingSD)
                newnLwSTDData.columns[k].append(nLwRemainingSD)

                newESDataMedian.columns[k].append(esMedian)
                newLIDataMedian.columns[k].append(liMedian)
                newLTDataMedian.columns[k].append(ltMedian)

                # Only populate valid wavelengths. Mark others for deletion
                if float(k) in waveSubset:  # should be redundant!
                    newRrsUncorrData.columns[k].append(rrs_uncorr)
                    newLWData.columns[k].append(lw)
                    newRrsData.columns[k].append(rrs)
                    newnLwData.columns[k].append(nLw)

                    newLWUNCData.columns[k].append(lwUNC[k])
                    newRrsUNCData.columns[k].append(rrsUNC[k])
                    # newnLwUNCData.columns[k].append(nLwUNC)
                    newnLwUNCData.columns[k].append(nLwUNC[k])
                    if ConfigFile.settings['bL1bCal']==1 and ConfigFile.settings["SensorType"].lower() in ["trios", "trios es only"]:
                    # Specifique case for Factory-Trios
                        newESUNCData.columns[k].append(esUNC[k])
                        newLIUNCData.columns[k].append(liUNC[k])
                        newLTUNCData.columns[k].append(ltUNC[k])
                    else:
                        newESUNCData.columns[k].append(esUNC[k][0])
                        newLIUNCData.columns[k].append(liUNC[k][0])
                        newLTUNCData.columns[k].append(ltUNC[k][0])

                    if sensor == 'HYPER':
                        if ZhangRho:
                            newRhoHyper.columns[k].append(rhoVec[k])
                            if xUNC is not None:  # TriOS factory does not require uncertainties
                                newRhoUNCHyper.columns[k].append(xUNC[f'rhoUNC_{sensor}'][k])
                            else:
                                newRhoUNCHyper.columns[k].append(np.nan)
                        else:
                            newRhoHyper.columns[k].append(rhoScalar)
                            if xUNC is not None:  # perhaps there is a better check for TriOS Factory branch?
                                try:
                                    # todo: explore why rho UNC is 1 index smaller than everything else
                                    # last wvl is missing
                                    newRhoUNCHyper.columns[k].append(xUNC[f'rhoUNC_{sensor}'][k])
                                except KeyError:
                                    newRhoUNCHyper.columns[k].append(0)
                            else:
                                newRhoUNCHyper.columns[k].append(np.nan)
                else:
                    deleteKey.append(k)

        # Eliminate reflectance keys/values in wavebands outside of valid set for the sake of Zhang model
        deleteKey = list(set(deleteKey))
        for key in deleteKey:
            # Only need to do this for the first ensemble in file
            if key in newRrsData.columns:
                del newLWData.columns[key]
                del newRrsUncorrData.columns[key]
                del newRrsData.columns[key]
                del newnLwData.columns[key]

                del newLWUNCData.columns[key]
                del newRrsUNCData.columns[key]
                del newnLwUNCData.columns[key]
                if sensor == 'HYPER':
                    del newRhoHyper.columns[key]

        newESData.columnsToDataset()
        newLIData.columnsToDataset()
        newLTData.columnsToDataset()
        newLWData.columnsToDataset()
        newRrsUncorrData.columnsToDataset()
        newRrsData.columnsToDataset()
        newnLwData.columnsToDataset()

        newESDataMedian.columnsToDataset()
        newLIDataMedian.columnsToDataset()
        newLTDataMedian.columnsToDataset()

        newESSTDData.columnsToDataset()
        newLISTDData.columnsToDataset()
        newLTSTDData.columnsToDataset()
        newLWSTDData.columnsToDataset()
        newRrsSTDData.columnsToDataset()
        newnLwSTDData.columnsToDataset()
        newESUNCData.columnsToDataset()
        newLIUNCData.columnsToDataset()
        newLTUNCData.columnsToDataset()
        newLWUNCData.columnsToDataset()
        newRrsUNCData.columnsToDataset()
        newnLwUNCData.columnsToDataset()

        if sensor == 'HYPER':
            newRhoHyper.columnsToDataset()
            newRhoUNCHyper.columnsToDataset()
            newRrsUncorrData.columnsToDataset()


    @staticmethod
    def spectralIrradiance(node, sensor, timeObj, xSlice, F0, F0_unc, waveSubset, xUNC):
        """
        Same as spectralReflectance, but only applies to irradiance data. Use for Es only processing
        """
        esXSlice = xSlice['es']  # mean
        esXmedian = xSlice['esMedian']
        esXRemaining = xSlice['esRemaining']
        esXstd = xSlice['esSTD']

        dateTime = timeObj['dateTime']
        dateTag = timeObj['dateTag']
        timeTag = timeObj['timeTag']

        # Root (new/output) groups:
        newIrradianceGroup = node.getGroup("IRRADIANCE")

        # If this is the first ensemble spectrum, set up the new datasets
        if not f'ES_{sensor}' in newIrradianceGroup.datasets:
            newESData = newIrradianceGroup.addDataset(f"ES_{sensor}")
            newESDataMedian = newIrradianceGroup.addDataset(f"ES_{sensor}_median")
            newESSTDData = newIrradianceGroup.addDataset(f"ES_{sensor}_sd")
            newESUNCData = newIrradianceGroup.addDataset(f"ES_{sensor}_unc")
        else:
            newESData = newIrradianceGroup.getDataset(f"ES_{sensor}")
            newESDataMedian = newIrradianceGroup.getDataset(f"ES_{sensor}_median")
            newESSTDData = newIrradianceGroup.getDataset(f"ES_{sensor}_sd")
            newESUNCData = newIrradianceGroup.getDataset(f"ES_{sensor}_unc")

        # Add datetime stamps back onto ALL datasets associated with the current sensor
        # If this is the first spectrum, add date/time, otherwise append
        # Groups REFLECTANCE, IRRADIANCE, and RADIANCE are intiallized with empty datasets, but
        # ANCILLARY is not.
        if "Datetag" not in newESData.columns:
            for gp in node.groups:
                if gp.id == "ANCILLARY":  # Ancillary is already populated. The other groups only have empty (named) datasets
                    continue
                else:
                    for ds in gp.datasets:
                        if sensor in ds:  # Only add datetime stamps to the current sensor datasets
                            gp.datasets[ds].columns["Datetime"] = [dateTime]  # mean of the ensemble datetime stamp
                            gp.datasets[ds].columns["Datetag"] = [dateTag]
                            gp.datasets[ds].columns["Timetag2"] = [timeTag]
        else:
            for gp in node.groups:
                if gp.id == "ANCILLARY":
                    continue
                else:
                    for ds in gp.datasets:
                        if sensor in ds:
                            gp.datasets[ds].columns["Datetime"].append(dateTime)
                            gp.datasets[ds].columns["Datetag"].append(dateTag)
                            gp.datasets[ds].columns["Timetag2"].append(timeTag)

        # Organise Uncertainty into wavebands
        esUNC = {}

        # Only Factory - Trios has no uncertainty here
        if (ConfigFile.settings['bL1bCal'] >= 2 or ConfigFile.settings['SensorType'].lower() == 'seabird'):
            esUNC = xUNC[f'esUNC_{sensor}']  # should already be convolved to hyperspec
        else:
            # factory case
            for wvl in waveSubset:
                k = str(wvl)
                if any([wvl == float(x) for x in esXSlice]):
                    esUNC[k] = 0

        for wvl in waveSubset:
            k = str(wvl)
            if any([wvl == float(x) for x in esXSlice]):
                # Initialize the new dataset if this is the first slice
                if k not in newESData.columns:
                    newESData.columns[k] = []
                    newESDataMedian.columns[k] = []
                    newESSTDData.columns[k] = []
                    newESUNCData.columns[k] = []

                # At this waveband (k); still using complete wavelength set
                es = esXSlice[k][0]  # Always the zeroth element; i.e. XSlice data are independent of past slices and node
                esRemaining = np.asarray(esXRemaining[k])  # array of remaining ensemble values in this band
                f0 = F0[k]
                f0UNC = F0_unc[k]

                esMedian = esXmedian[k][0]
                esSTD = esXstd[k][0]

                newESData.columns[k].append(es)
                newESSTDData.columns[k].append(esSTD)
                newESDataMedian.columns[k].append(esMedian)

                # Only populate valid wavelengths. Mark others for deletion
                if (ConfigFile.settings['bL1bCal'] == 1 and
                        ConfigFile.settings["SensorType"].lower() in ["trios", "trios es only"]):
                    # Specifique case for Factory-Trios
                    newESUNCData.columns[k].append(esUNC[k])
                else:
                    newESUNCData.columns[k].append(esUNC[k][0])

        newESData.columnsToDataset()
        newESDataMedian.columnsToDataset()
        newESSTDData.columnsToDataset()
        newESUNCData.columnsToDataset()


    @staticmethod
    def filterData(group, badTimes, sensor = None):
        ''' Delete flagged records. Sensor is only specified to get the timestamp.
            All data in the group (including satellite sensors) will be deleted. '''

        msg = f'Remove {group.id} Data'
        print(msg)
        Utilities.writeLogFile(msg)

        if sensor is None:
            if group.id == "ANCILLARY":
                timeStamp = group.getDataset("LATITUDE").data["Datetime"]
            if group.id == "IRRADIANCE":
                timeStamp = group.getDataset("ES").data["Datetime"]
            if group.id == "RADIANCE":
                timeStamp = group.getDataset("LI").data["Datetime"]
            if group.id == "SIXS_MODEL":
                timeStamp = group.getDataset("direct_ratio").data["Datetime"]
        else:
            if group.id == "IRRADIANCE":
                timeStamp = group.getDataset(f"ES_{sensor}").data["Datetime"]
            if group.id == "RADIANCE":
                timeStamp = group.getDataset(f"LI_{sensor}").data["Datetime"]
            if group.id == "REFLECTANCE":
                timeStamp = group.getDataset(f"Rrs_{sensor}").data["Datetime"]

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

            # msg = f'Eliminate data between: {dateTime}'
            # print(msg)
            # Utilities.writeLogFile(msg)

            start = dateTime[0]
            stop = dateTime[1]

            if startLength > 0:
                rowsToDelete = []
                for i in range(startLength):
                    if start <= timeStamp[i] and stop >= timeStamp[i]:
                        try:
                            rowsToDelete.append(i)
                            finalCount += 1
                        except Exception:
                            print('error')
                    else:
                        newTimeStamp.append(timeStamp[i])
                group.datasetDeleteRow(rowsToDelete)
            else:
                msg = 'Data group is empty. Continuing.'
                print(msg)
                Utilities.writeLogFile(msg)
            timeStamp = newTimeStamp.copy()

        if len(badTimes) == 0:
            startLength = 1 # avoids div by zero below when finalCount is 0

        for ds in group.datasets:
            # if ds != "STATION":
            try:
                group.datasets[ds].datasetToColumns()
            except Exception:
                print('error')

        msg = f'   Length of dataset after removal {originalLength-finalCount} long: {round(100*finalCount/originalLength)}% removed'
        print(msg)
        Utilities.writeLogFile(msg)
        return finalCount/originalLength


    @staticmethod
    def interpolateColumn(columns, wl):
        ''' Interpolate wavebands to estimate a single, unsampled waveband '''
        #print("interpolateColumn")
        # Values to return
        return_y = []

        # Column to interpolate to
        new_x = [wl]

        # Get wavelength values
        wavelength = []
        for k in columns:
            #print(k)
            wavelength.append(float(k))
        x = np.asarray(wavelength)

        # get the length of a column
        num = len(list(columns.values())[0])

        # Perform interpolation for each row
        for i in range(num):
            values = []
            for k in columns:
                #print("b")
                values.append(columns[k][i])
            y = np.asarray(values)

            new_y = sp.interpolate.interp1d(x, y)(new_x)
            return_y.append(new_y[0])

        return return_y


    @staticmethod
    def specQualityCheck(group, inFilePath, station=None):
        ''' Perform spectral filtering
        Calculate the STD of the normalized (at some max value) average ensemble.
        Then test each normalized spectrum against the ensemble average and STD and negatives (within the spectral range).
        Plot results'''

        # This is the range upon which the spectral filter is applied (and plotted)
        # It goes up to 900 to include bands used in NIR correction
        fRange = [350, 900]

        badTimes = []
        if group.id == 'IRRADIANCE':
            Data = group.getDataset("ES")
            timeStamp = group.getDataset("ES").data["Datetime"]
            badTimes = Utilities.specFilter(inFilePath, Data, timeStamp, station, filterRange=fRange,\
                filterFactor=ConfigFile.settings["fL2SpecFilterEs"], rType='Es')
            msg = f'{len(np.unique(badTimes))/len(timeStamp)*100:.1f}% of Es data flagged'
            print(msg)
            Utilities.writeLogFile(msg)
        else:
            Data = group.getDataset("LI")
            timeStamp = group.getDataset("LI").data["Datetime"]
            badTimes1 = Utilities.specFilter(inFilePath, Data, timeStamp, station, filterRange=fRange,\
                filterFactor=ConfigFile.settings["fL2SpecFilterLi"], rType='Li')
            msg = f'{len(np.unique(badTimes1))/len(timeStamp)*100:.1f}% of Li data flagged'
            print(msg)
            Utilities.writeLogFile(msg)

            Data = group.getDataset("LT")
            timeStamp = group.getDataset("LT").data["Datetime"]
            badTimes2 = Utilities.specFilter(inFilePath, Data, timeStamp, station, filterRange=fRange,\
                filterFactor=ConfigFile.settings["fL2SpecFilterLt"], rType='Lt')
            msg = f'{len(np.unique(badTimes2))/len(timeStamp)*100:.1f}% of Lt data flagged'
            print(msg)
            Utilities.writeLogFile(msg)

            badTimes = np.append(badTimes1,badTimes2, axis=0)

        if len(badTimes) == 0:
            badTimes = None
        return badTimes


    @staticmethod
    def ltQuality(sasGroup):
        ''' Perform Lt Quality checking '''

        ltData = sasGroup.getDataset("LT")
        ltData.datasetToColumns()
        ltColumns = ltData.columns
        # These get popped off the columns, but restored when filterData runs datasetToColumns
        ltColumns.pop('Datetag')
        ltColumns.pop('Timetag2')
        ltDatetime = ltColumns.pop('Datetime')

        badTimes = []
        for indx, dateTime in enumerate(ltDatetime):
            # If the Lt spectrum in the NIR is brighter than in the UVA, something is very wrong
            UVA = [350,400]
            NIR = [780,850]
            ltUVA = []
            ltNIR = []
            for wave in ltColumns:
                if float(wave) > UVA[0] and float(wave) < UVA[1]:
                    ltUVA.append(ltColumns[wave][indx])
                elif float(wave) > NIR[0] and float(wave) < NIR[1]:
                    ltNIR.append(ltColumns[wave][indx])

            if np.nanmean(ltUVA) < np.nanmean(ltNIR):
                badTimes.append(dateTime)

        badTimes = np.unique(badTimes)
        # Duplicate each element to a list of two elements in a list
        # BUG: This is not optimal as it creates one badTimes record for each bad
        #   timestamp, rather than span of timestamps from badtimes[i][0] to badtimes[i][1]
        badTimes = np.rot90(np.matlib.repmat(badTimes,2,1), 3)
        msg = f'{len(np.unique(badTimes))/len(ltDatetime)*100:.1f}% of spectra flagged'
        print(msg)
        Utilities.writeLogFile(msg)

        if len(badTimes) == 0:
            badTimes = None
        return badTimes


    @staticmethod
    def negReflectance(reflGroup, field, VIS = None):
        ''' Perform negative reflectance spectra checking '''
        # Run for entire file, not just one ensemble
        if VIS is None:
            VIS = [400,700]

        reflData = reflGroup.getDataset(field)
        # reflData.datasetToColumns()
        reflColumns = reflData.columns
        reflDate = reflColumns.pop('Datetag')
        reflTime = reflColumns.pop('Timetag2')
        # reflColumns.pop('Datetag')
        # reflColumns.pop('Timetag2')
        timeStamp = reflColumns.pop('Datetime')

        badTimes = []
        for indx, timeTag in enumerate(timeStamp):
            # If any spectra in the vis are negative, delete the whole spectrum
            reflVIS = []
            wavelengths = []
            for wave in reflColumns:
                wavelengths.append(float(wave))
                if float(wave) > VIS[0] and float(wave) < VIS[1]:
                    reflVIS.append(reflColumns[wave][indx])
                # elif float(wave) > NIR[0] and float(wave) < NIR[1]:
                #     ltNIR.append(ltColumns[wave][indx])

            # Flag entire record for removal
            if any(item < 0 for item in reflVIS):
                badTimes.append(timeTag)

            # Set negatives to 0
            NIR = [VIS[-1]+1,max(wavelengths)]
            UV = [min(wavelengths),VIS[0]-1]
            for wave in reflColumns:
                if ((float(wave) >= UV[0] and float(wave) < UV[1]) or \
                            (float(wave) >= NIR[0] and float(wave) <= NIR[1])) and \
                            reflColumns[wave][indx] < 0:
                    reflColumns[wave][indx] = 0

        badTimes = np.unique(badTimes)
        badTimes = np.rot90(np.matlib.repmat(badTimes,2,1), 3) # Duplicates each element to a list of two elements (start, stop)
        msg = f'{len(np.unique(badTimes))/len(timeStamp)*100:.1f}% of {field} spectra flagged'
        print(msg)
        Utilities.writeLogFile(msg)

        # # Need to add these at the beginning of the ODict
        reflColumns['Timetag2'] = reflTime
        reflColumns['Datetag'] = reflDate
        reflColumns['Datetime'] = timeStamp
        reflColumns.move_to_end('Timetag2', last=False)
        reflColumns.move_to_end('Datetag', last=False)
        reflColumns.move_to_end('Datetime', last=False)

        reflData.columnsToDataset()

        if len(badTimes) == 0:
            badTimes = None
        return badTimes


    @staticmethod
    def metQualityCheck(refGroup, sasGroup):
        ''' Perform meteorological quality control '''

        esFlag = float(ConfigFile.settings["fL2SignificantEsFlag"])
        dawnDuskFlag = float(ConfigFile.settings["fL2DawnDuskFlag"])
        humidityFlag = float(ConfigFile.settings["fL2RainfallHumidityFlag"])
        cloudFLAG = float(ConfigFile.settings["fL2CloudFlag"]) # Not to be confused with cloudFlag...

        esData = refGroup.getDataset("ES")
        esData.datasetToColumns()
        esColumns = esData.columns

        esColumns.pop('Datetag')
        esColumns.pop('Timetag2')
        esTime = esColumns.pop('Datetime')

        liData = sasGroup.getDataset("LI")
        liData.datasetToColumns()
        liColumns = liData.columns
        liColumns.pop('Datetag')
        liColumns.pop('Timetag2')
        liColumns.pop('Datetime')

        ltData = sasGroup.getDataset("LT")
        ltData.datasetToColumns()
        ltColumns = ltData.columns
        ltColumns.pop('Datetag')
        ltColumns.pop('Timetag2')
        ltColumns.pop('Datetime')

        li750 = ProcessL2.interpolateColumn(liColumns, 750.0)
        es370 = ProcessL2.interpolateColumn(esColumns, 370.0)
        es470 = ProcessL2.interpolateColumn(esColumns, 470.0)
        es480 = ProcessL2.interpolateColumn(esColumns, 480.0)
        es680 = ProcessL2.interpolateColumn(esColumns, 680.0)
        es720 = ProcessL2.interpolateColumn(esColumns, 720.0)
        es750 = ProcessL2.interpolateColumn(esColumns, 750.0)
        badTimes = []
        for indx, dateTime in enumerate(esTime):
            # Masking spectra affected by clouds (Ruddick 2006, IOCCG Protocols).
            # The alternative to masking is to process them differently (e.g. See Ruddick_Rho)
            # Therefore, set this very high if you don't want it triggered (e.g. 1.0, see Readme)
            if li750[indx]/es750[indx] >= cloudFLAG:
                msg = f"Quality Check: Li(750)/Es(750) >= cloudFLAG:{cloudFLAG}"
                print(msg)
                Utilities.writeLogFile(msg)
                badTimes.append(dateTime)

            # Threshold for significant es
            # Wernand 2002
            if es480[indx] < esFlag:
                msg = f"Quality Check: es(480) < esFlag:{esFlag}"
                print(msg)
                Utilities.writeLogFile(msg)
                badTimes.append(dateTime)

            # Masking spectra affected by dawn/dusk radiation
            # Wernand 2002
            #v = esXSlice["470.0"][0] / esXSlice["610.0"][0] # Fix 610 -> 680
            if es470[indx]/es680[indx] < dawnDuskFlag:
                msg = f'Quality Check: ES(470.0)/ES(680.0) < dawnDuskFlag:{dawnDuskFlag}'
                print(msg)
                Utilities.writeLogFile(msg)
                badTimes.append(dateTime)

            # Masking spectra affected by rainfall and high humidity
            # Wernand 2002 (940/370), Garaba et al. 2012 also uses Es(940/370), presumably 720 was developed by Wang...???
            # TODO: Follow up on the source of this flag
            if es720[indx]/es370[indx] < humidityFlag:
                msg = f'Quality Check: ES(720.0)/ES(370.0) < humidityFlag:{humidityFlag}'
                print(msg)
                Utilities.writeLogFile(msg)
                badTimes.append(dateTime)

        badTimes = np.unique(badTimes)
        badTimes = np.rot90(np.matlib.repmat(badTimes,2,1), 3) # Duplicates each element to a list of two elements in a list
        msg = f'{len(np.unique(badTimes))/len(esTime)*100:.1f}% of spectra flagged'
        print(msg)
        Utilities.writeLogFile(msg)

        if len(badTimes) == 0:
            # Restore timestamps to columns (since it's not going to filterData, where it otherwise happens)
            esData.datasetToColumns()
            liData.datasetToColumns()
            ltData.datasetToColumns()
            badTimes = None
        return badTimes


    @staticmethod
    def columnToSlice(columns, start, end):
        ''' Take a slice of a dataset stored in columns '''

        # Each column is a time series either at a waveband for radiometer columns, or various grouped datasets for ancillary
        # Start and end are defined by the interval established in the Config (they are indexes)
        newSlice = collections.OrderedDict()
        for col in columns:
            if start == end:
                newSlice[col] = columns[col][start:end+1] # otherwise you get nada []
            else:
                newSlice[col] = columns[col][start:end] # up to not including end...next slice will pick it up
        return newSlice


    @staticmethod
    # def interpAncillary(node, ancData, modRoot, radData):
    def includeModelDefaults(ancGroup, modRoot):
        ''' Include model data or defaults for blank ancillary fields '''
        print('Filling blank ancillary data with models or defaults from Configuration')

        epoch = datetime.datetime(1970, 1, 1,tzinfo=datetime.timezone.utc)
        # radData = referenceGroup.getDataset("ES") # From node, the input file

        # Convert ancillary date time
        if ancGroup is not None:
            ancGroup.datasets['LATITUDE'].datasetToColumns()
            ancTime = ancGroup.datasets['LATITUDE'].columns['Timetag2']
            ancSeconds = []
            ancDatetime = []
            for i, ancDate in enumerate(ancGroup.datasets['LATITUDE'].columns['Datetag']):
                ancDatetime.append(Utilities.timeTag2ToDateTime(Utilities.dateTagToDateTime(ancDate),ancTime[i]))
                ancSeconds.append((ancDatetime[i]-epoch).total_seconds())
        # Convert model data date and time to datetime and then to seconds for interpolation
        if modRoot is not None:
            modTime = modRoot.groups[0].datasets["Timetag2"].tolist()
            modSeconds = []
            modDatetime = []
            for i, modDate in enumerate(modRoot.groups[0].datasets["Datetag"].tolist()):
                modDatetime.append(Utilities.timeTag2ToDateTime(Utilities.dateTagToDateTime(modDate),modTime[i]))
                modSeconds.append((modDatetime[i]-epoch).total_seconds())

        # Model or default fills
        if 'WINDSPEED' in ancGroup.datasets:
            ancGroup.datasets['WINDSPEED'].datasetToColumns()
            windDataset = ancGroup.datasets['WINDSPEED']
            wind = windDataset.columns['NONE']
        else:
            windDataset = ancGroup.addDataset('WINDSPEED')
            wind = np.empty((1,len(ancSeconds)))
            wind[:] = np.nan
            wind = wind[0].tolist()
        if 'AOD' in ancGroup.datasets:
            ancGroup.datasets['AOD'].datasetToColumns()
            aodDataset = ancGroup.datasets['AOD']
            aod = aodDataset.columns['NONE']
        else:
            aodDataset = ancGroup.addDataset('AOD')
            aod = np.empty((1,len(ancSeconds)))
            aod[:] = np.nan
            aod = aod[0].tolist()
        # Default fills
        if 'SALINITY' in ancGroup.datasets:
            ancGroup.datasets['SALINITY'].datasetToColumns()
            saltDataset = ancGroup.datasets['SALINITY']
            salt = saltDataset.columns['NONE']
        else:
            saltDataset = ancGroup.addDataset('SALINITY')
            salt = np.empty((1,len(ancSeconds)))
            salt[:] = np.nan
            salt = salt[0].tolist()
        if 'SST' in ancGroup.datasets:
            ancGroup.datasets['SST'].datasetToColumns()
            sstDataset = ancGroup.datasets['SST']
            sst = sstDataset.columns['NONE']
        else:
            sstDataset = ancGroup.addDataset('SST')
            sst = np.empty((1,len(ancSeconds)))
            sst[:] = np.nan
            sst = sst[0].tolist()

        # Initialize flags
        windFlag = []
        aodFlag = []
        for i,ancSec in enumerate(ancSeconds):
            if np.isnan(wind[i]):
                windFlag.append('undetermined')
            else:
                windFlag.append('field')
            if np.isnan(aod[i]):
                aodFlag.append('undetermined')
            else:
                aodFlag.append('field')

        # Replace Wind, AOD NaNs with modeled data where possible.
        # These will be within one hour of the field data.
        if modRoot is not None:
            msg = 'Filling in field data with model data where needed.'
            print(msg)
            Utilities.writeLogFile(msg)

            for i,ancSec in enumerate(ancSeconds):

                if np.isnan(wind[i]):
                    # msg = 'Replacing wind with model data'
                    # print(msg)
                    # Utilities.writeLogFile(msg)
                    idx = Utilities.find_nearest(modSeconds,ancSec)
                    wind[i] = modRoot.groups[0].datasets['Wind'][idx]
                    windFlag[i] = 'model'
                if np.isnan(aod[i]):
                    # msg = 'Replacing AOD with model data'
                    # print(msg)
                    # Utilities.writeLogFile(msg)
                    idx = Utilities.find_nearest(modSeconds,ancSec)
                    aod[i] = modRoot.groups[0].datasets['AOD'][idx]
                    aodFlag[i] = 'model'

        # Replace Wind, AOD, SST, and Sal with defaults where still nan
        msg = 'Filling in ancillary data with default values where still needed.'
        print(msg)
        Utilities.writeLogFile(msg)

        saltFlag = []
        sstFlag = []
        for i, value in enumerate(wind):
            if np.isnan(value):
                wind[i] = ConfigFile.settings["fL2DefaultWindSpeed"]
                windFlag[i] = 'default'
        for i, value in enumerate(aod):
            if np.isnan(value):
                aod[i] = ConfigFile.settings["fL2DefaultAOD"]
                aodFlag[i] = 'default'
        for i, value in enumerate(salt):
            if np.isnan(value):
                salt[i] = ConfigFile.settings["fL2DefaultSalt"]
                saltFlag.append('default')
            else:
                saltFlag.append('field')
        for i, value in enumerate(sst):
            if np.isnan(value):
                sst[i] = ConfigFile.settings["fL2DefaultSST"]
                sstFlag.append('default')
            else:
                sstFlag.append('field')

        # Populate the datasets and flags with the InRad variables
        windDataset.columns["NONE"] = wind
        windDataset.columns["WINDFLAG"] = windFlag
        windDataset.columnsToDataset()
        aodDataset.columns["AOD"] = aod
        aodDataset.columns["AODFLAG"] = aodFlag
        aodDataset.columnsToDataset()
        saltDataset.columns["NONE"] = salt
        saltDataset.columns["SALTFLAG"] = saltFlag
        saltDataset.columnsToDataset()
        sstDataset.columns["NONE"] = sst
        sstDataset.columns["SSTFLAG"] = sstFlag
        sstDataset.columnsToDataset()

        # Convert ancillary seconds back to date/timetags ...
        ancDateTag = []
        ancTimeTag2 = []
        ancDT = []
        for i, sec in enumerate(ancSeconds):
            ancDT.append(datetime.datetime.utcfromtimestamp(sec).replace(tzinfo=datetime.timezone.utc))
            ancDateTag.append(float(f'{int(ancDT[i].timetuple()[0]):04}{int(ancDT[i].timetuple()[7]):03}'))
            ancTimeTag2.append(float( \
                f'{int(ancDT[i].timetuple()[3]):02}{int(ancDT[i].timetuple()[4]):02}{int(ancDT[i].timetuple()[5]):02}{int(ancDT[i].microsecond/1000):03}'))

        # Move the Timetag2 and Datetag into the arrays and remove the datasets
        for ds in ancGroup.datasets:
            ancGroup.datasets[ds].columns["Datetag"] = ancDateTag
            ancGroup.datasets[ds].columns["Timetag2"] = ancTimeTag2
            ancGroup.datasets[ds].columns["Datetime"] = ancDT
            ancGroup.datasets[ds].columns.move_to_end('Timetag2', last=False)
            ancGroup.datasets[ds].columns.move_to_end('Datetag', last=False)
            ancGroup.datasets[ds].columns.move_to_end('Datetime', last=False)

            ancGroup.datasets[ds].columnsToDataset()

    @staticmethod
    def sliceAveHyper(y, hyperSlice):
        ''' Take the slice mean of the lowest X% of hyperspectral slices '''
        xSlice = collections.OrderedDict()
        xSliceRemaining = collections.OrderedDict()
        xMedian = collections.OrderedDict()
        hasNan = False
        # Ignore runtime warnings when array is all NaNs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for k in hyperSlice: # each k is a time series at a waveband.
                v = hyperSlice[k] if y is None else [hyperSlice[k][i] for i in y]# selects the lowest 5% within the interval window...
                mean = np.nanmean(v) # ... and averages them
                median = np.nanmedian(v) # ... and the median spectrum
                xSlice[k] = [mean]
                xMedian[k] = [median]
                if np.isnan(mean):
                    hasNan = True

                # Retain remaining spectra for use in calculating Rrs_sd
                xSliceRemaining[k] = v

        return hasNan, xSlice, xMedian, xSliceRemaining


    @staticmethod
    def sliceAveOther(node, start, end, y, ancGroup, sixSGroup):
        ''' Take the slice AND the mean averages of ancillary and 6S data with X% '''        

        def _sliceAveOther(node, start, end, y, group):
            if node.getGroup(group.id):
                newGroup = node.getGroup(group.id)
            else:
                newGroup = node.addGroup(group.id)

            for dsID in group.datasets:
                if newGroup.getDataset(dsID):
                    newDS = newGroup.getDataset(dsID)
                else:
                    newDS = newGroup.addDataset(dsID)
                ds = group.getDataset(dsID)
                
                # Set relAz to abs(relAz) prior to averaging
                if dsID == 'REL_AZ':
                    ds.columns['REL_AZ'] = np.abs(ds.columns['REL_AZ']).tolist()

                ds.datasetToColumns()
                dsSlice = ProcessL2.columnToSlice(ds.columns,start, end)
                dsXSlice = None

                for subDScol in dsSlice: # each dataset contains columns (including date, time, data, and possibly flags)
                    if subDScol == 'Datetime':
                        timeStamp = dsSlice[subDScol]
                        # Stores the mean datetime by converting to (and back from) epoch second
                        if len(timeStamp) > 0:
                            epoch = datetime.datetime(1970, 1, 1,tzinfo=datetime.timezone.utc) #Unix zero hour
                            tsSeconds = []
                            for dt in timeStamp:
                                tsSeconds.append((dt-epoch).total_seconds())
                            meanSec = np.mean(tsSeconds)
                            dateTime = datetime.datetime.utcfromtimestamp(meanSec).replace(tzinfo=datetime.timezone.utc)
                            date = Utilities.datetime2DateTag(dateTime)
                            sliceTime = Utilities.datetime2TimeTag2(dateTime)
                    if subDScol not in ('Datetime', 'Datetag', 'Timetag2'):
                        v = [dsSlice[subDScol][i] for i in y] # y is an array of indexes for the lowest X%

                        if dsXSlice is None:
                            dsXSlice = collections.OrderedDict()
                            dsXSlice['Datetag'] = [date]
                            dsXSlice['Timetag2'] = [sliceTime]
                            dsXSlice['Datetime'] = [dateTime]

                        if subDScol not in dsXSlice:
                            dsXSlice[subDScol] = []
                        if (subDScol.endswith('FLAG')) or (subDScol.endswith('STATION')):
                            # Find the most frequest element
                            dsXSlice[subDScol].append(Utilities.mostFrequent(v))
                        else:
                            # Otherwise take a nanmean of the slice
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore", category=RuntimeWarning)
                                dsXSlice[subDScol].append(np.nanmean(v)) # Warns of empty when empty...

                # Just test a sample column to see if it needs adding or appending
                if subDScol not in newDS.columns:
                    newDS.columns = dsXSlice
                else:
                    for item in newDS.columns:
                        newDS.columns[item] = np.append(newDS.columns[item], dsXSlice[item])

                newDS.columns.move_to_end('Timetag2', last=False)
                newDS.columns.move_to_end('Datetag', last=False)
                newDS.columns.move_to_end('Datetime', last=False)
                newDS.columnsToDataset()

        _sliceAveOther(node, start, end, y, ancGroup)        
        _sliceAveOther(node, start, end, y, sixSGroup)

    @staticmethod
    def ensemblesReflectance(node, sasGroup, refGroup, ancGroup, uncGroup,
                             esRawGroup, liRawGroup, ltRawGroup, sixSGroup, start, end):

        # return ProcessL2.ensemblesReflectance_legacy(node, sasGroup, refGroup, ancGroup, uncGroup,
        #                                              esRawGroup, liRawGroup, ltRawGroup, sixSGroup, start, end)

        # %% Get dataset
        if ConfigFile.settings["SensorType"].lower() == "trios es only":
            # Only Irradiance
            groups = {'ES': refGroup}
            es_only = True
        else:
            # Radiances and Irradiance
            groups = {'ES': refGroup, 'LI': sasGroup, 'LT': sasGroup}
            es_only = False
        columns = {}
        for k, group in groups.items():
            d = group.getDataset(k)
            d.datasetToColumns()
            columns[k] = d.columns

        # %% Slice Data
        data_slice = {k: ProcessL2.columnToSlice(d, start, end) for k, d in columns.items()}

        # %% Check Length of Slice
        es_start_datetime, es_stop_datetime = data_slice['ES']['Datetime'][0], data_slice['ES']['Datetime'][-1]
        if (es_stop_datetime - es_start_datetime) < datetime.timedelta(seconds=60):
            Utilities.writeLogFileAndPrint("ProcessL2.ensemblesReflectance ensemble is less than 1 minute. Skipping.")
            return False

        # TODO Check why SIXS code used to be here but data manipulation is not used later on, hence dropped

        # %% Get active raw groups (based on data available in groups, required to get std)
        map_raw_groups = {'ES': esRawGroup, 'LI': liRawGroup, 'LT': ltRawGroup}
        if ConfigFile.settings['SensorType'].lower() == "seabird":
            raw_groups = {k: {t: map_raw_groups[k][t] for t in ['LIGHT', 'DARK']} for k in groups.keys()}
            raw_slices = {k: {t: {'datetime': grp.datasets['DATETIME'].data[start:end],
                                  'data': ProcessL2.columnToSlice(grp.datasets[k].columns, start, end)}
                              for t in ['LIGHT', 'DARK']} for k, grp in raw_groups.items()}
        else:
            raw_groups = {k: map_raw_groups[k] for k in groups.keys()}
            raw_slices = {k: {'data': ProcessL2.columnToSlice(grp.datasets[k].columns, start, end)}
                          for k, grp in raw_groups.items()}

        # %% Get Configuration
        enable_percent_lt = float(ConfigFile.settings["bL2EnablePercentLt"])
        percent_lt = float(ConfigFile.settings["fL2PercentLt"])
        zhang_rho = int(ConfigFile.settings["bL2ZhangRho"])
        if ConfigFile.settings["SensorType"].lower() in ["trios", "trios es only"]:
            sensor, sensor_type = Trios(), 'TriOS'
        elif ConfigFile.settings["SensorType"].lower() == "seabird":
            sensor, sensor_type = HyperOCR(), 'SeaBird'
        else:
            raise ValueError('Sensor type not supported.')
        # TODO check why Delete Datetime, Datetag, and Timetag2 from slices

        # %% Compute mean datetime of slice
        # Based on Es timestamp only
        timestamps = data_slice['ES']['Datetime']
        epoch = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
        mean_timestamp = np.mean(np.array(timestamps) - epoch).total_seconds()
        mean_datetime = datetime.datetime.fromtimestamp(mean_timestamp, tz=datetime.timezone.utc)
        timestamp_dict = {
            'dateTime': mean_datetime,
            'dateTag': Utilities.datetime2DateTag(mean_datetime),
            'timeTag': Utilities.datetime2TimeTag2(mean_datetime)
        }

        # %% Get standard deviation of slice (entire slice, not just the lowest X%)
        # Drop time info, for stats functions
        for k in data_slice.keys():
            del data_slice[k]['Datetime']
            del data_slice[k]['Datetag']
            del data_slice[k]['Timetag2']
        wavelengths = np.asarray(list(data_slice['ES'].keys()), dtype=float)
        stats = sensor.generateSensorStats(sensor_type, raw_groups, raw_slices, wavelengths)
        if ConfigFile.settings["SensorType"].lower() == "seabird":
            raw_groups = {k: d['LIGHT'] for k, d in raw_groups.items()}
            for key, group in raw_groups.items():
                group.id = f'{key}_L1AQC'
        slice_std = {k: {str(wl): [std_interp[0]] for wl, std_interp in stats[k]['std_Signal_Interpolated'].items()}
                     for k, slice in data_slice.items()}
        # Use wavelengths rather than keys from stats as stats is rounding wavelength to one decimal
        # which is inconsistent with other places in the code.

        # %% Convolve to satellite bands
        convolve_to_satellite, satellite_bands = {}, {}
        if ConfigFile.settings['bL2WeightMODISA']:
            convolve_to_satellite['MODISA'] = lambda slice: Weight_RSR.processMODISBands(slice, sensor='A')
            satellite_bands['MODIS'] = Weight_RSR.MODISBands()
        if ConfigFile.settings['bL2WeightMODIST']:
            convolve_to_satellite['MODIST'] = lambda slice: Weight_RSR.processMODISBands(slice, sensor='T')
            satellite_bands['MODIS'] = Weight_RSR.MODISBands()
        if ConfigFile.settings['bL2WeightVIIRSN']:
            convolve_to_satellite['VIIRSN'] = lambda slice: Weight_RSR.processVIIRSBands(slice, sensor='N')
            satellite_bands['VIIRS'] = Weight_RSR.VIIRSBands()
        if ConfigFile.settings['bL2WeightVIIRSJ']:
            convolve_to_satellite['VIIRSJ'] = lambda slice: Weight_RSR.processVIIRSBands(slice, sensor='J')
            satellite_bands['VIIRS'] = Weight_RSR.VIIRSBands()
        if ConfigFile.settings['bL2WeightSentinel3A']:
            convolve_to_satellite['Sentinel3A'] = lambda slice: Weight_RSR.processSentinel3Bands(slice, sensor='A')
            satellite_bands['Sentinel3'] = Weight_RSR.Sentinel3Bands()
        if ConfigFile.settings['bL2WeightSentinel3B']:
            convolve_to_satellite['Sentinel3A'] = lambda slice: Weight_RSR.processSentinel3Bands(slice, sensor='B')
            satellite_bands['Sentinel3'] = Weight_RSR.Sentinel3Bands()

        satellite_slice = {satellite: {k: convolve_to_satellite[satellite](slice) for k, slice in data_slice.items()}
                           for satellite in convolve_to_satellite}
        satellite_slice_std = {satellite: {k: convolve_to_satellite[satellite](slice_std)
                                           for k, slice in slice_std.keys()}
                               for satellite in convolve_to_satellite}

        # %% Get index of N lowest Lt frames => selection
        if enable_percent_lt and es_only:
            Utilities.writeLogFileAndPrint("Percent LT is not supported for Trios ES only. Disabled feature.")
            enable_percent_lt = False
        elif enable_percent_lt and 'LT' not in data_slice:
            Utilities.writeLogFileAndPrint("Percent LT is not available. No LT data found.")
            enable_percent_lt = False

        if 'LT' in data_slice:
            n = len(data_slice['LT'][list(data_slice['LT'].keys())[0]])
        else:
            n = len(timestamps)
        y = np.arange(n) # Default to all indexes, if no LT data or percent_lt is not enabled
        if enable_percent_lt:
            # Calculates the lowest X% (based on Hooker & Morel 2003; Hooker et al. 2002; Zibordi et al. 2002, IOCCG Protocols)
            # X will depend on FOV and integration time of instrument. Hooker cites a rate of 2 Hz.
            # It remains unclear to me from Hooker 2002 whether the recommendation is to take the average of the ir/radiances
            # within the threshold and calculate Rrs, or to calculate the Rrs within the threshold, and then average, however IOCCG
            # Protocols pretty clearly state to average the ir/radiances first, then calculate the Rrs...as done here.
            x = round(n * percent_lt / 100)
            # There are sometimes only a small number of spectra in the slice,
            #  so the percent Lt estimation becomes highly questionable and is overridden here.
            if n <= 5 or x == 0:
                x = n  # if only 5 or fewer records retained, use them all...
            if x > 1:
                lt780 = ProcessL2.interpolateColumn(data_slice['LT'], 780.0)
                index = np.argsort(lt780)
                y = index[:x]
                n = x  # Update n to the number of selected spectra
                Utilities.writeLogFileAndPrint(f"{n} spectra remaining in slice to average after filtering to lowest {percent_lt}%.")

        # %% Get Ensemble Size
        for grp in node.groups:
            if grp.id not in ['REFLECTANCE', 'IRRADIANCE', 'RADIANCE']:
                continue
            if es_only and grp.id != 'IRRADIANCE':
                continue
            if 'Ensemble_N' not in grp.datasets:
                grp.addDataset('Ensemble_N')
                grp.datasets['Ensemble_N'].columns['N'] = []
            grp.datasets['Ensemble_N'].columns['N'].append(n)
            grp.datasets['Ensemble_N'].columnsToDataset()

        # %% Mean of the slice selection (based on index of N lowest Lt frames)
        slice_mean, slice_median, slice_remaining = {}, {}, {}
        for k, slice in data_slice.items():
            has_nan, slice_mean[k], slice_median[k], slice_remaining[k] = ProcessL2.sliceAveHyper(y, slice)
            if has_nan:
                Utilities.writeLogFileAndPrint("ProcessL2.ensemblesReflectance: Slice X% average error: Dataset all NaNs.")
                return False

        # %% Mean of the slice selection convolved to satellite bands
        satellite_slice_mean, satellite_slice_median, satellite_slice_remaining = {}, {}, {}
        for satellite, data in satellite_slice.items():
            for k, slice in data.items():
                has_nan, satellite_slice_mean[satellite][k], satellite_slice_median[satellite][k], \
                    satellite_slice_remaining[satellite][k] = ProcessL2.sliceAveHyper(y, slice)
                if has_nan:
                    Utilities.writeLogFileAndPrint("ProcessL2.ensemblesReflectance: Slice X% average error: Dataset all NaNs.")
                    return False

        # %% Mean of the ancillary selection
        ProcessL2.sliceAveOther(node, start, end, y, ancGroup, sixSGroup)
        newAncGroup = node.getGroup("ANCILLARY")  # Just populated above
        newAncGroup.attributes['Ancillary_Flags (0, 1, 2, 3)'] = ['undetermined', 'field', 'model', 'default']

        anc_slice = {}
        for param in ['WINDSPEED', 'SZA', 'SST', 'SALINITY', 'REL_AZ', 'AOD']:
            if param in newAncGroup.datasets:
                l = newAncGroup.getDataset(param).data[param][-1].copy()
                anc_slice[param] = l[0] if isinstance(l, list) else l
            else:
                if param == 'AOD' and not zhang_rho:
                    continue   # Optional if don't use Zhang Rho
                if param == 'SALINITY':
                    continue   # Optional
                if param == 'REL_AZ' and es_only:
                    continue   # Optional for ES only
                Utilities.writeLogFileAndPrint(f"ProcessL2.ensemblesReflectance: Required {param} data absent in Ancillary. Aborting.")
                return False

        for param in ['CLOUD', 'WAVE_HT', 'STATION']:  # TODO CHECK If need second loop or could skip the [-1] for optional parameters
            if "WAVE_HT" in newAncGroup.datasets:
                l = newAncGroup.getDataset(param).data[param].copy()
                anc_slice[param] = l[0] if isinstance(l, list) else l
            else:
                anc_slice[param] = None

        # %% Calculate rho_sky for the ensemble
        if es_only:
            rho_vec, rho_scalar, rho_unc = None, None, None
        else:
            rho_vec, rho_scalar, rho_unc = ProcessL2.calculate_rho_sky_for_ensemble(wavelengths.tolist(), slice_mean, anc_slice)

        # %% Get TSIS-1 and convolve to satellite bands
        # NOTE: TSIS uncertainties reported as 1-sigma
        F0_hyper, F0_unc, F0_raw, F0_unc_raw, wv_raw = Utilities.TSIS_1(timestamp_dict['dateTag'], wavelengths.tolist())

        # Recycling _raw in TSIS_1 calls below prevents the dataset having to be reread
        if F0_hyper is None:
            Utilities.writeLogFileAndPrint(f"ProcessL2.ensemblesReflectance: No hyperspectral TSIS-1 F0. Aborting.")
            return False

        satellite_f0, satellite_f0_unc = {}, {}
        satellite_bands_subset = {}
        for sat, bands in satellite_bands.items():
            # Convolve TSIS-1 F0 to satellite bands
            satellite_f0[sat], satellite_f0_unc[sat] = Utilities.TSIS_1(timestamp_dict['dateTag'], bands, F0_raw, F0_unc_raw, wv_raw)[0:2]
            # Get bands for Zhang models
            b = np.array(bands)
            satellite_bands_subset[sat] = b[(350 <= b) & (b <= 1000)].to_list()


        # %% Format data and Propagate Uncertainties
        x_slice = {
            **{k.lower(): v for k, v in slice_mean.items()},
            **{k.lower() + 'Median': v for k, v in slice_median.items()},
            **{k.lower() + 'STD': v for k, v in slice_std.items()},
            **{k.lower() + 'STD_RAW': v['std_Signal'] for k, v in stats.items()}, # Check output is reliable
            **{k.lower() + 'Remaining': v for k, v in slice_remaining.items()},
        }
        x_unc = None
        tic = time.process_time()
        if ConfigFile.settings["bL1bCal"] <= 2:  # Factory Calibration or FRM-Class Specific
            l1b_unc = sensor.ClassBased(node, uncGroup, stats)
            if l1b_unc:
                x_slice.update(l1b_unc)
                # convert uncertainties back into absolute form using the signals recorded from ProcessL2
                for k, v in slice_mean.items():
                    x_slice[k.lower() + 'Unc'] = {u[0]: [u[1][0] * np.abs(s[0])] for u, s in
                                                  zip(x_slice[k.lower() + 'Unc'].items(), v.values())}
                if es_only:
                    x_unc = sensor.ClassBasedL2_ES_only(wavelengths, x_slice)
                    for k in list(x_unc.keys()):  # Delete LT and LI uncertainties as all infinity
                        if k.start_with('lt') or k.start_with('li'):
                            del x_unc[k]
                else:
                    x_unc = sensor.ClassBasedL2(node, uncGroup, rho_scalar, rho_vec, rho_unc, wavelengths.tolist(),
                                                x_slice)
            elif not(ConfigFile.settings['SensorType'].lower() in ["trios", "trios es only"] and (ConfigFile.settings["bL1bCal"] == 1)):
                Utilities.writeLogFileAndPrint(f"ProcessL2.ensemblesReflectance: Instrument uncertainty processing failed. Aborting.")
                return False
        elif ConfigFile.settings["bL1bCal"] == 3:  # FRM-Sensor Specific
            x_slice.update(sensor.FRM(node, uncGroup, raw_groups, raw_slices, stats, wavelengths))
            x_unc = sensor.FRM_L2(rho_scalar, rho_vec, rho_unc, wavelengths, x_slice)
        Utilities.writeLogFileAndPrint(f"ProcessL2.ensemblesReflectance: Uncertainty Update Elapsed Time: {time.process_time() - tic:.1f} s")

        # Move uncertainties to x_unc and drop samples form x_slice
        if x_unc is not None:
            for k in list(x_slice.keys()):  # trick to del items while looping on the dict
                if "sample" in k.lower():
                    del x_slice[k]  # samples are no longer needed
                elif "unc" in k.lower():
                    x_unc[f"{k[0:2]}UNC_HYPER"] = x_slice.pop(k)  # transfer instrument uncs to xUNC

            # Extract uncertainties for convolving to satellite bands
            slice_unc = {k: v for k, v in x_unc.items() if k.endswith('UNC_HYPER')}
        else:
            slice_unc = None

        # %% Populate relevant fields in node
        if es_only:
            ProcessL2.spectralIrradiance(node, 'HYPER', timestamp_dict, x_slice, F0_hyper, F0_unc, wavelengths, x_unc)
        else:
            ProcessL2.spectralReflectance(node, 'HYPER', timestamp_dict, x_slice, F0_hyper, F0_unc,
                                          rho_scalar, rho_vec, wavelengths, x_unc)


        # %% Apply NIR Correction
        # Perform near-infrared residual correction to remove additional atmospheric and glint contamination
        rrs_nir_cor, nLw_nir_corr = None, None
        if ConfigFile.settings["bL2PerformNIRCorrection"] and not es_only:
            rrs_nir_cor, nLw_nir_corr = ProcessL2.nirCorrection(node, 'HYPER', F0_hyper)

        # %% Convolve to satellite bands
        satellite_slice_mean, satellite_slice_median, satellite_slice_remaining = {}, {}, {}
        for (sat, mean), median, remaining, std in zip(
                satellite_slice_mean.items(), satellite_slice_median.values(),
                satellite_slice_remaining.values(), satellite_slice_std.values()):
            for k in mean.keys():
                x_slice[k] = mean[k]
                x_slice[k + 'remaining'] = remaining[k]
                x_slice[k + 'median'] = median[k]
                x_slice[k + 'std'] = std[k]
            sat_rho_vec = None
            if zhang_rho:
                sat_rho_vec = convolve_to_satellite[sat](rho_vec)
                sat_rho_vec = {key: value[0] for key, value in sat_rho_vec.items()}  # drop one level of list
            # NOTE: According to AR, this may not be a robust way of estimating convolved uncertainties.
            # He has implemented another way, but it is very slow due to multiple MC runs. Comment this out
            # for now, but a sensitivity analysis may show it to be okay.
            # NOTE: 1/2024 Why is this not commented out if the slow, more accurate way is now implemented?
            if slice_unc:
                for k, v in slice_unc.items():
                    x_unc[f'{k[:2]}UNC'] = convolve_to_satellite[sat](v)
            if es_only:
                ProcessL2.spectralIrradiance(node, sat, timestamp_dict, x_slice,
                                             satellite_f0[sat], satellite_f0_unc[sat],
                                             satellite_bands_subset[sat], x_unc)
            else:
                ProcessL2.spectralReflectance(node, sat, timestamp_dict, x_slice,
                                              satellite_f0[sat], satellite_f0_unc[sat], rho_scalar,
                                              sat_rho_vec, satellite_bands_subset[sat], x_unc)
                if ConfigFile.settings["bL2PerformNIRCorrection"]:
                    # Can't apply good NIR corrections at satellite bands,
                    # so use the correction factors from the hyperspectral instead.
                    ProcessL2.nirCorrectionSatellite(node, sat, rrs_nir_cor, nLw_nir_corr)
        return True


    @staticmethod
    def calculate_rho_sky_for_ensemble(wavelengths, data_slice_mean, anc_slice):
        # Get Configuration
        rho_default = float(ConfigFile.settings["fL2RhoSky"])
        if int(ConfigFile.settings["bL23CRho"]):
            method = 'three_c_rho'
        elif int(ConfigFile.settings["bL2ZhangRho"]):
            method = 'zhang_rho'
        else:
            method = 'mobley_rho'

        # Calculate rho_sky for the ensemble
        if method == "three_c_rho":
            # NOTE: Placeholder for Groetsch et al. 2017

            li750 = ProcessL2.interpolateColumn(data_slice_mean['LI'], 750.0)
            es750 = ProcessL2.interpolateColumn(data_slice_mean['ES'], 750.0)
            sky750 = li750[0] / es750[0]

            rhoScalar, rhoUNC = RhoCorrections.threeCCorr(sky750, rho_default, anc_slice['WINDSPEED'])
            # The above is not wavelength dependent. No need for seperate values/vectors for satellites
            rhoVec = None
        elif method == "zhang_rho":
            # Zhang rho is based on Zhang et al. 2017 and calculates the wavelength-dependent rho vector
            # separated for sun and sky to include polarization factors.

            # Model limitations: AOD 0 - 0.2, Solar zenith 0-60 deg, Wavelength 350-1000 nm.

            # reduced number of draws because of how computationally intensive the Zhang method is
            rho_uncertainty_obj = Propagate(M=10, cores=1)

            # Need to limit the input for the model limitations. This will also mean cutting out Li, Lt, and Es
            # from non-valid wavebands.
            # NOTE: Need to update to 0.5 for new database
            for k, limit in [('AOD', 0.2), ('WINDSPEED', 15), ('SZA', 60)]:
                if anc_slice[k] > limit:
                    Utilities.writeLogFileAndPrint(
                        f'{k} = {anc_slice[k]:.3f}. Maximum {k}. Setting to {limit}. Expect larger, uncaptured errors.')
                    anc_slice[k] = limit
            if min(wavelengths) < 350 or max(wavelengths) > 1000:
                Utilities.writeLogFileAndPrint('Wavelengths extend beyond model limits. Truncating to 350 - 1000 nm.')
                wave_old = wavelengths.copy()
                wave_list = [(i, band) for i, band in enumerate(wave_old) if (band >= 350) and (band <= 1000)]
                wave_array = np.array(wave_list)
                # wavelength is now truncated to only valid wavebands for use in Zhang models
                wavelengths = wave_array[:, 1].tolist()

            SVA = ConfigFile.settings['fL2SVA']
            rhoVector, rhoUNC = RhoCorrections.ZhangCorr(anc_slice['WINDSPEED'], anc_slice['AOD'], anc_slice['CLOUD'],
                                                         anc_slice['SZA'], anc_slice['SST'], anc_slice['SALINITY'],
                                                         anc_slice['REL_AZ'],
                                                         SVA, wavelengths, rho_uncertainty_obj)

            rhoVec = {}
            for i, k in enumerate(wavelengths):
                rhoVec[str(k)] = rhoVector[i]
        elif method == "mobley_rho":
            # Full Mobley 1999 model from LUT
            rho_uncertainty_obj = Propagate(M=100, cores=1)  # Standard number of draws for reasonable uncertainty estimates
            if 'AOD' in anc_slice:
                rhoScalar, rhoUNC = RhoCorrections.M99Corr(anc_slice['WINDSPEED'], anc_slice['SZA'],
                                                           anc_slice['REL_AZ'],
                                                           rho_uncertainty_obj,
                                                           AOD=anc_slice['AOD'], cloud=anc_slice['CLOUD'],
                                                           wTemp=anc_slice['SST'],
                                                           sal=anc_slice['SALINITY'], waveBands=wavelengths)
            else:
                rhoScalar, rhoUNC = RhoCorrections.M99Corr(anc_slice['WINDSPEED'], anc_slice['SZA'],
                                                           anc_slice['REL_AZ'],
                                                           rho_uncertainty_obj)
            # Not wavelength dependent, so no need for rhoVec
            rhoVec = None

        return rhoScalar, rhoVec, rhoUNC


    @staticmethod
    def stationsEnsemblesReflectance(node, root, station=None):
        ''' Extract stations if requested, then pass to ensemblesReflectance for ensemble
            averages, rho calcs, Rrs, Lwn, NIR correction, satellite convolution, OC Products.'''

        print("stationsEnsemblesReflectance")

        root_group_ids = [g.id for g in root.groups]

        # Create a third HDF for copying root without altering it
        rootCopy = HDFRoot()
        rootCopy.addGroup("ANCILLARY")
        rootCopy.addGroup("IRRADIANCE")
        if 'RADIANCE' in root_group_ids:
            rootCopy.addGroup("RADIANCE")
        rootCopy.addGroup('SIXS_MODEL')

        rootCopy.getGroup('ANCILLARY').copy(root.getGroup('ANCILLARY'))
        rootCopy.getGroup('IRRADIANCE').copy(root.getGroup('IRRADIANCE'))
        if 'RADIANCE' in root_group_ids:
            rootCopy.getGroup('RADIANCE').copy(root.getGroup('RADIANCE'))

        sixS_available = False
        for gp in root.groups:
            if gp.id == 'SIXS_MODEL':
                sixS_available = True
                rootCopy.getGroup('SIXS_MODEL').copy(root.getGroup('SIXS_MODEL'))
                break

        if ConfigFile.settings['SensorType'].lower() == 'seabird':
            rootCopy.addGroup("ES_DARK_L1AQC")
            rootCopy.addGroup("ES_LIGHT_L1AQC")
            rootCopy.addGroup("LI_DARK_L1AQC")
            rootCopy.addGroup("LI_LIGHT_L1AQC")
            rootCopy.addGroup("LT_DARK_L1AQC")
            rootCopy.addGroup("LT_LIGHT_L1AQC")
            rootCopy.getGroup('ES_LIGHT_L1AQC').copy(root.getGroup('ES_LIGHT_L1AQC'))
            rootCopy.getGroup('ES_DARK_L1AQC').copy(root.getGroup('ES_DARK_L1AQC'))
            rootCopy.getGroup('LI_LIGHT_L1AQC').copy(root.getGroup('LI_LIGHT_L1AQC'))
            rootCopy.getGroup('LI_DARK_L1AQC').copy(root.getGroup('LI_DARK_L1AQC'))
            rootCopy.getGroup('LT_LIGHT_L1AQC').copy(root.getGroup('LT_LIGHT_L1AQC'))
            rootCopy.getGroup('LT_DARK_L1AQC').copy(root.getGroup('LT_DARK_L1AQC'))

            esRawGroup = {"LIGHT": rootCopy.getGroup('ES_LIGHT_L1AQC'), "DARK": rootCopy.getGroup('ES_DARK_L1AQC')}
            liRawGroup = {"LIGHT": rootCopy.getGroup('LI_LIGHT_L1AQC'), "DARK": rootCopy.getGroup('LI_DARK_L1AQC')}
            ltRawGroup = {"LIGHT": rootCopy.getGroup('LT_LIGHT_L1AQC'), "DARK": rootCopy.getGroup('LT_DARK_L1AQC')}

            sasGroup = rootCopy.getGroup("RADIANCE")
        elif ConfigFile.settings["SensorType"].lower() == "trios":
            rootCopy.addGroup("ES_L1AQC")
            rootCopy.addGroup("LI_L1AQC")
            rootCopy.addGroup("LT_L1AQC")
            rootCopy.getGroup('ES_L1AQC').copy(root.getGroup('ES_L1AQC'))
            rootCopy.getGroup('LI_L1AQC').copy(root.getGroup('LI_L1AQC'))
            rootCopy.getGroup('LT_L1AQC').copy(root.getGroup('LT_L1AQC'))

            esRawGroup = rootCopy.getGroup('ES_L1AQC')
            liRawGroup = rootCopy.getGroup('LI_L1AQC')
            ltRawGroup = rootCopy.getGroup('LT_L1AQC')

            sasGroup = rootCopy.getGroup("RADIANCE")
        elif ConfigFile.settings["SensorType"].lower() == "trios es only":
            rootCopy.addGroup("ES_L1AQC")
            rootCopy.getGroup('ES_L1AQC').copy(root.getGroup('ES_L1AQC'))
            esRawGroup = rootCopy.getGroup('ES_L1AQC')
            liRawGroup, ltRawGroup = None, None
            sasGroup = None

        # rootCopy will be manipulated in the making of node, but root will not
        referenceGroup = rootCopy.getGroup("IRRADIANCE")
        ancGroup = rootCopy.getGroup("ANCILLARY")
        if sixS_available:
            sixSGroup = rootCopy.getGroup("SIXS_MODEL")
        else:
            sixSGroup = None

        if ConfigFile.settings["bL1bCal"] >= 2 or ConfigFile.settings['SensorType'].lower() == 'seabird':
            rootCopy.addGroup("RAW_UNCERTAINTIES")
            rootCopy.getGroup('RAW_UNCERTAINTIES').copy(root.getGroup('RAW_UNCERTAINTIES'))
            uncGroup = rootCopy.getGroup("RAW_UNCERTAINTIES")
        # Only Factory-Trios has no unc
        else:
            uncGroup = None

        Utilities.rawDataAddDateTime(rootCopy) # For L1AQC data carried forward
        Utilities.rootAddDateTimeCol(rootCopy)

        ###############################################################################
        #
        # Stations
        #   Simplest approach is to run station extraction seperately from (i.e. in addition to)
        #   underway data. This means if station extraction is selected in the GUI, all non-station
        #   data will be discarded here prior to any further filtering or processing.

        if ConfigFile.settings["bL2Stations"]:
            msg = "Extracting station data only. All other records will be discarded."
            print(msg)
            Utilities.writeLogFile(msg)

            # If we are here, the station was already chosen in Controller
            try:
                stations = ancGroup.getDataset("STATION").columns["STATION"]
                dateTime = ancGroup.getDataset("STATION").columns["Datetime"]
            except Exception:
                msg = "No station data found in ancGroup. Aborting."
                print(msg)
                Utilities.writeLogFile(msg)
                return False

            badTimes = []
            start = False
            stop = False
            for index, stn in enumerate(stations):
                # print(f'index: {index}, station: {station}, datetime: {dateTime[index]}')
                # if np.isnan(station) and start == False:
                if (stn != station) and (start is False):
                    start = dateTime[index]
                # if not np.isnan(station) and not (start == False) and (stop == False):
                if not (stn!=station) and (start is not False) and (stop is False):
                    stop = dateTime[index-1]
                    badTimes.append([start, stop])
                    start = False
                    stop = False
                # End of file, no active station
                # if np.isnan(station) and not (start == False) and (index == len(stations)-1):
                if (stn != station) and not (start is False) and (index == len(stations)-1):
                    stop = dateTime[index]
                    badTimes.append([start, stop])

            if badTimes is not None and len(badTimes) != 0:
                print('Removing records...')
                check = ProcessL2.filterData(referenceGroup, badTimes)
                if check == 1.0:
                    msg = "100% of irradiance data removed. Abort."
                    print(msg)
                    Utilities.writeLogFile(msg)
                    return False
                if sasGroup is not None:
                    ProcessL2.filterData(sasGroup, badTimes)
                ProcessL2.filterData(ancGroup, badTimes)
                if sixS_available:
                    ProcessL2.filterData(sixSGroup, badTimes)

        #####################################################################
        #
        # Ensembles. Break up data into time intervals, and calculate averages and reflectances
        #
        esData = referenceGroup.getDataset("ES")
        esColumns = esData.columns
        timeStamp = esColumns["Datetime"]
        esLength = len(list(esColumns.values())[0])
        interval = float(ConfigFile.settings["fL2TimeInterval"])

        # interpolate Light/Dark data for Raw groups if HyperOCR data is being processed
        # NOTE: Why is this necessary? Aren't we interested in the variability of darks at native acquisition frequency? -DA
        if ConfigFile.settings['SensorType'].lower() == "seabird":
            # in seabird case interpolate dark data to light timer before breaking into stations
            instrument = HyperOCR()
            if not any([instrument.darkToLightTimer(esRawGroup, 'ES'),
                        instrument.darkToLightTimer(liRawGroup, 'LI'),
                        instrument.darkToLightTimer(ltRawGroup, 'LT')]):
                msg = "failed to interpolate dark data to light data timer"
                print(msg)

        if interval == 0:
            # Here, take the complete time series
            print("No time binning. This can take a moment.")
            progressBar = tqdm(total=esLength, unit_scale=True, unit_divisor=1)
            for i in range(0, esLength-1):
                progressBar.update(1)
                start = i
                end = i+1

                if not ProcessL2.ensemblesReflectance(node, sasGroup, referenceGroup, ancGroup, 
                                                            uncGroup, esRawGroup,liRawGroup, ltRawGroup,
                                                            sixSGroup, start, end):
                    msg = 'ProcessL2.ensemblesReflectance unsliced failed. Abort.'
                    print(msg)
                    Utilities.writeLogFile(msg)
                    continue
        else:
            msg = 'Binning datasets to ensemble time interval.'
            print(msg)
            Utilities.writeLogFile(msg)

            # Iterate over the time ensembles
            start = 0
            endTime = timeStamp[0] + datetime.timedelta(0,interval)
            endFileTime = timeStamp[-1]
            EndOfFileFlag = False
            # endTime is theoretical based on interval
            if endTime > endFileTime:
                endTime = endFileTime
                EndOfFileFlag = True # In case the whole file is shorter than the selected interval

            for i in range(0, esLength):
                timei = timeStamp[i]
                if (timei > endTime) or EndOfFileFlag: # end of increment reached
                    if EndOfFileFlag:
                        end = len(timeStamp)-1 # File shorter than interval; include all spectra
                        if not ProcessL2.ensemblesReflectance(node, sasGroup, referenceGroup, ancGroup, 
                                                            uncGroup, esRawGroup,liRawGroup, ltRawGroup,
                                                            sixSGroup, start, end):
                            msg = 'ProcessL2.ensemblesReflectance with slices failed. Continue.'
                            print(msg)
                            Utilities.writeLogFile(msg)
                            break # End of file reached. Safe to break

                        break # End of file reached. Safe to break
                    else:
                        endTime = timei + datetime.timedelta(0,interval) # increment for the next bin loop
                        end = i # end of the slice is up to and not including...so -1 is not needed

                    if endTime > endFileTime:
                        endTime = endFileTime
                        EndOfFileFlag = True

                    if not ProcessL2.ensemblesReflectance(node, sasGroup, referenceGroup, ancGroup,
                                                                 uncGroup, esRawGroup,liRawGroup, ltRawGroup,
                                                                 sixSGroup, start, end):
                        msg = 'ProcessL2.ensemblesReflectance with slices failed. Continue.'
                        print(msg)
                        Utilities.writeLogFile(msg)

                        start = i
                        continue # End of ensemble reached. Continue.
                    start = i

                    if EndOfFileFlag:
                        # No need to continue incrementing; all records captured in one ensemble
                        break

            # For the rare case where end of record is reached at, but not exceeding, endTime...
            if not EndOfFileFlag:
                end = i+1 # i is the index of end of record; plus one to include i due to -1 list slicing
                if not ProcessL2.ensemblesReflectance(node, sasGroup, referenceGroup, ancGroup, 
                                                            uncGroup, esRawGroup,liRawGroup, ltRawGroup,
                                                            sixSGroup, start, end):
                    msg = 'ProcessL2.ensemblesReflectance ender clause failed.'
                    print(msg)
                    Utilities.writeLogFile(msg)


        #
        # Reflectance calculations complete
        #

        # Filter reflectances for negative ensemble spectra
        # NOTE: Any spectrum that has any negative values between
        #  400 - 700ish (hard-coded below), remove the entire spectrum. Otherwise,
        # set negative bands to 0.

        if ConfigFile.settings["bL2NegativeSpec"] and ConfigFile.settings["SensorType"].lower() == "trios es only":
            Utilities.writeLogFileAndPrint("Filtering reflectance spectra for negative values"
                                           " is not supported for Trios ES only. Disabled feature.")
        elif ConfigFile.settings["bL2NegativeSpec"]:
            fRange = [400, 680]
            Utilities.writeLogFileAndPrint("Filtering reflectance spectra for negative values.")
            # newReflectanceGroup = node.groups[0]
            newReflectanceGroup = node.getGroup("REFLECTANCE")
            if not newReflectanceGroup.datasets:
                Utilities.writeLogFileAndPrint("Ensemble is empty. Aborting.")
                return False

            badTimes1 = ProcessL2.negReflectance(newReflectanceGroup, 'Rrs_HYPER', VIS = fRange)
            badTimes2 = ProcessL2.negReflectance(newReflectanceGroup, 'nLw_HYPER', VIS = fRange)

            badTimes = None
            if badTimes1 is not None and badTimes2 is not None:
                badTimes = np.append(badTimes1,badTimes2, axis=0)
            elif badTimes1 is not None:
                badTimes = badTimes1
            elif badTimes2 is not None:
                badTimes = badTimes2

            if badTimes is not None:
                print('Removing records...')

                # Even though HYPER is specified here, ALL data at badTimes in the group,
                # including satellite data, will be removed.
                check = ProcessL2.filterData(newReflectanceGroup, badTimes, sensor = "HYPER")
                if check > 0.99:
                    msg = "Too few spectra remaining. Abort."
                    print(msg)
                    Utilities.writeLogFile(msg)
                    return False
                ProcessL2.filterData(node.getGroup("IRRADIANCE"), badTimes, sensor = "HYPER")
                ProcessL2.filterData(node.getGroup("RADIANCE"), badTimes, sensor = "HYPER")
                ProcessL2.filterData(node.getGroup("ANCILLARY"), badTimes)
                if sixS_available:
                    ProcessL2.filterData(node.getGroup("SIXS_MODEL"), badTimes)

        return True

    @staticmethod
    def processL2(root,station=None):
        '''Calculates Rrs and nLw after quality checks and filtering, glint removal, residual
            subtraction. Weights for satellite bands, and outputs plots and SeaBASS datasets'''

        root_group_ids = [g.id for g in root.groups]

        # Root is the input from L1BQC, node is the output
        # Root should not be impacted by data reduction in node...
        node = HDFRoot()
        node.addGroup("ANCILLARY")
        if 'REFLECTANCE' in root_group_ids:
            node.addGroup("REFLECTANCE")
        node.addGroup("IRRADIANCE")
        if 'RADIANCE' in root_group_ids:
            node.addGroup("RADIANCE")
        node.addGroup("SIXS_MODEL")
        node.copyAttributes(root)
        node.attributes["PROCESSING_LEVEL"] = "2"
        # Remaining attributes managed below...

        # Copy attributes from root and for completeness, flip datasets into columns in all groups
        for grp in root.groups:
            for gp in node.groups:
                if gp.id == grp.id:
                    gp.copyAttributes(grp)
            for ds in grp.datasets:
                grp.datasets[ds].datasetToColumns()

            # Carry over L1AQC data for use in uncertainty budgets
            if grp.id.endswith('_L1AQC'): #or grp.id.startswith('SIXS_MODEL'):
                newGrp = node.addGroup(grp.id)
                newGrp.copy(grp)
                for ds in newGrp.datasets:
                    newGrp.datasets[ds].datasetToColumns()                    

        # Process stations, ensembles to reflectances, OC prods, etc.
        if not ProcessL2.stationsEnsemblesReflectance(node, root,station):
            return None

        # Reflectance
        gp = node.getGroup("REFLECTANCE")
        if gp:
            gp.attributes["Rrs_UNITS"] = "1/sr"
            gp.attributes["nLw_UNITS"] = "uW/cm^2/nm/sr"
            if ConfigFile.settings['bL23CRho']:
                gp.attributes['GLINT_CORR'] = 'Groetsch et al. 2017'
            if ConfigFile.settings['bL2ZhangRho']:
                gp.attributes['GLINT_CORR'] = 'Zhang et al. 2017'
            if ConfigFile.settings['bL2DefaultRho']:
                gp.attributes['GLINT_CORR'] = 'Mobley 1999'
            if ConfigFile.settings['bL2PerformNIRCorrection']:
                if ConfigFile.settings['bL2SimpleNIRCorrection']:
                    gp.attributes['NIR_RESID_CORR'] = 'Mueller and Austin 1995'
                if ConfigFile.settings['bL2SimSpecNIRCorrection']:
                    gp.attributes['NIR_RESID_CORR'] = 'Ruddick et al. 2005/2006'
            if ConfigFile.settings['bL2NegativeSpec']:
                gp.attributes['NEGATIVE_VALUE_FILTER'] = 'ON'

        # Root
        if ConfigFile.settings['bL2Stations']:
            node.attributes['STATION_EXTRACTION'] = 'ON'
        node.attributes['ENSEMBLE_DURATION'] = str(ConfigFile.settings['fL2TimeInterval']) + ' sec'

        # Check to insure at least some data survived quality checks
        if ConfigFile.settings["SensorType"].lower() == "trios es only" and node.getGroup("IRRADIANCE").getDataset("ES_HYPER").data is None:
            Utilities.writeLogFileAndPrint("All irradiance data appear to have been eliminated from the file. Aborting.")
            return None
        if ConfigFile.settings["SensorType"].lower() != "trios es only" and node.getGroup("REFLECTANCE").getDataset("Rrs_HYPER").data is None:
            Utilities.writeLogFileAndPrint("All reflectance data appear to have been eliminated from the file. Aborting.")
            return None

        # If requested, proceed to calculation of derived geophysical and
        # inherent optical properties
        totalProds = sum(list(ConfigFile.products.values()))
        if totalProds > 0:
            if ConfigFile.settings["SensorType"].lower() == "trios es only":
                Utilities.writeLogFileAndPrint("Calculating derived geophysical and inherent optical properties "
                                               "is not supported for Trios ES only. Skipping.")
            else:
                ProcessL2OCproducts.procProds(node)

        # If requested, process BRDF corrections to Rrs and nLw
        if ConfigFile.settings["SensorType"].lower() == "trios es only" and ConfigFile.settings["bL2BRDF"]:
            Utilities.writeLogFileAndPrint("Calculating derived geophysical and inherent optical properties "
                                           "is not supported for Trios ES only. Skipping.")
        elif ConfigFile.settings["bL2BRDF"]:
            if ConfigFile.settings['bL2BRDF_fQ']:
                msg = "Applying iterative Morel et al. 2002 BRDF correction to Rrs and nLw"
                print(msg)
                Utilities.writeLogFile(msg)
                ProcessL2BRDF.procBRDF(node, BRDF_option='M02')

            if ConfigFile.settings['bL2BRDF_IOP']:
                msg = "Applying Lee et al. 2011 BRDF correction to Rrs and nLw"
                print(msg)
                Utilities.writeLogFile(msg)
                ProcessL2BRDF.procBRDF(node, BRDF_option='L11')

            # if ConfigFile.settings['bL2BRDF_OXX']:
            #     msg = "Applying OXX BRDF correction to Rrs and nLw"
            #     print(msg)
            #     Utilities.writeLogFile(msg)
            #     ProcessL2BRDF.procBRDF(node, BRDF_option='OXX')


        # Strip out L1AQC data
        for gp in reversed(node.groups):
            if gp.id.endswith('_L1AQC'):
                node.removeGroup(gp)

        # In the case of TriOS Factory, strip out uncertainty datasets
        if ConfigFile.settings["SensorType"].lower() in ["trios", "trios es only"] and ConfigFile.settings['bL1bCal'] == 1:
            for gp in node.groups:
                if gp.id in ('IRRADIANCE', 'RADIANCE', 'REFLECTANCE'):
                    removeList = []
                    for dsName in reversed(gp.datasets):
                        if dsName.endswith('_unc'):
                            removeList.append(dsName)
                    for dsName in removeList:
                        gp.removeDataset(dsName)

        # Change _median nomiclature to _uncorr
        for gp in node.groups:
            if gp.id in ('IRRADIANCE', 'RADIANCE', 'REFLECTANCE'):
                changeList = []
                for dsName in gp.datasets:
                    if dsName.endswith('_median'):
                        changeList.append(dsName)
                for dsName in changeList:
                    gp.datasets[dsName].changeDatasetName(gp,dsName,dsName.replace('_median','_uncorr'))


        # Now strip datetimes from all datasets
        for gp in node.groups:
            for dsName in gp.datasets:
                ds = gp.datasets[dsName]
                if "Datetime" in ds.columns:
                    ds.columns.pop("Datetime")
                ds.columnsToDataset()

        now = datetime.datetime.now()
        timestr = now.strftime("%d-%b-%Y %H:%M:%S")
        node.attributes["FILE_CREATION_TIME"] = timestr

        return node
