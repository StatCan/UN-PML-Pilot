#!/bin/bash

echo 
echo "Launching a local FL test"

if test "$#" -ne 4; then
    echo "Not enough parameters";
    echo "Ex. ./launch-test.sh flower 25 1 serverip:port"
    echo "Runs this script for 25 rounds of training in debug mode"
    echo
    exit 1;
fi


echo "SERVER @ $4"
echo
if [[ $(tmux ls | grep -c ^"$1") == 1 ]]
    then 
        echo "tmux session exists $1";
        exit 1;
fi


tmux new-session -s "$1" -d

tmux send-keys -t "$1":0 "./harserver.py -s [::]:8080 -m 4 -M 4 -r$2 -T../../OUTPUT/3\ -\ STATCAN/train/3_ALL_train.csv -t../../OUTPUT/3\ -\ STATCAN/test/3_ALL_test.csv -d$3" C-m
sleep 2
tmux send-keys -t "$1":0 "export FLWR_TXSESSION=$1" C-m


tmux attach-session -t "$1"

