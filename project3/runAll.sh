#!/bin/bash

########################################
############# CSCI 2951-O ##############
########################################
E_BADARGS=65
if [ $# -ne 3 ]
then
	echo "Usage: `basename $0` <inputFolder/> <timeLimit> <logFile>"
	echo "Description:"
	echo -e "\t This script make calls to ./run0.sh for all the files in the given inputFolder/"
	echo -e "\t Each run is subject to the given time limit in seconds."
	echo -e "\t Last line of each run is appended to the given logFile."
	echo -e "\t If a run fails, due to the time limit or other error, the file name is appended to the logFile with --'s as time and result. "
	echo -e "\t If the logFile already exists, the run is aborted."
	exit $E_BADARGS
fi

# Parameters
inputFolder=$1
timeLimit=$2
logFile=$3

# Append slash to the end of inputFolder if it does not have it
lastChar="${inputFolder: -1}"
if [ "$lastChar" != "/" ]; then
inputFolder=$inputFolder/
fi

# Terminate if the log file already exists
[ -f $logFile ] && echo "Logfile $logFile already exists, terminating." && exit 1

# Create the log file
touch $logFile

# Check for which timeout
if command -v timeout >/dev/null 2>&1; then
    TIMEOUT_CMD="timeout"
elif command -v gtimeout >/dev/null 2>&1; then
    TIMEOUT_CMD="gtimeout"
else
    TIMEOUT_CMD="perl_timeout"
fi

# Run on every file, get the last line, append to log file
for f in $inputFolder*.*
do
	fullFileName=$(realpath "$f")
	echo "Running $fullFileName"
	if [[ "$TIMEOUT_CMD" == "perl_timeout" ]]; then
		perl -e 'alarm shift; exec @ARGV' "$timeLimit" ./run.sh "$fullFileName" > output.tmp
		
	else
		$TIMEOUT_CMD "$timeLimit" ./run.sh "$fullFileName" > output.tmp
	fi
	returnValue="$?"
	if [[ "$returnValue" = 0 ]]; then 					# Run is successful
		cat output.tmp | tail -1 >> $logFile				# Record the last line as solution
	else 										# Run failed, record the instanceName with no solution
		echo Error
		instance=$(basename "$fullFileName")	
		echo "{\"Instance\": \"$instance\", \"Time\": \"--\", \"Result\": \"--\"}" >> $logFile	
	fi
	rm -f output.tmp
done
