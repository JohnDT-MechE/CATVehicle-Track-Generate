# CATVehicle-Track-Generate
Working to create dynamic encryption key generation based on the entropy of traffic within a shared FOV. This project is being done by students working at the University of Arizona CAT Vehicle REU under UROC.

The main.py file contains the main code to track and detect the passage of vehicles past the camera perspectives. This script uses configurations for the configurations.json file to access the correct video source with the correct parameters for detection, and saves the data obtained into a csv file specified in the configuration. The counter.py and tracker.py files contain helper classes used in the main.py script. Additionally, encode.py reads in data from CSV files using Pandas, and encodes this in various ways to obtain two different codes from opposite vehicle perspectives and compare their similarities.