#!/usr/bin/env bash


readonly CURRENT_DIR=$(dirname $(realpath $0))
readonly DATA_DIR=${CURRENT_DIR}"/../data/"


readonly DOWNLOAD_URL="https://raw.githubusercontent.com/ryuichiueda/LNPR_BOOK_CODES/master/sensor_data/"
readonly DOWNLOAD_FILES='
         sensor_data_200.txt
         sensor_data_280.txt
         sensor_data_300.txt
         sensor_data_400.txt
         sensor_data_500.txt
         sensor_data_600.txt
         sensor_data_700.txt
         sensor_data_1000.txt
'


for file in ${DOWNLOAD_FILES}; do
    echo -e "\ndownloading ${file}\n"
    wget --show-progress --quiet ${DOWNLOAD_URL}/${file}"?raw=true" -O ${DATA_DIR}/${file}
done
