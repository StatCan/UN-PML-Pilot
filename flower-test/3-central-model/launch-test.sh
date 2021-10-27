#!/usr/bin/bash

echo 
echo "Launching a local FL test"
echo
if test "$#" -ne 3; then
    echo "Not enough parameters";
    echo "Ex. ./launch-test.sh flower 25 1"
    echo "Runs this script for 25 rounds of training in debug mode"
    echo
    exit 1;
fi

if [[ $(tmux ls | grep -c ^"$1") == 1 ]]
    then 
        echo "tmux session exists $1";
        exit 1;
fi


tmux new-session -s "$1" -d
tmux split-window -h -t "$1"
tmux split-window -v -t "$1"
tmux split-window -h -t "$1"
tmux select-pane -U -t "$1"
tmux split-window -h -t "$1"


tmux send-keys -t "$1":0.0 "./har-server.py -s [::]:8080 -m 4 -M 4 -r$2 -T../../OUTPUT/3\ -\ STATCAN/train/3_ALL_train.csv -t../../OUTPUT/3\ -\ STATCAN/test/3_ALL_test.csv -d$3" C-m
sleep 3
tmux send-keys -t "$1":0.1 "export FLWR_TXSESSION=$1" C-m
tmux send-keys -t "$1":0.1 "./har-client.py -s localhost:8080 -T../../OUTPUT/0\ -\ CBS/train/0_ALL_train.csv -t../../OUTPUT/0\ -\ CBS/test/0_ALL_test.csv -d$3" C-m
tmux send-keys -t "$1":0.2 "./har-client.py -s localhost:8080 -T../../OUTPUT/1\ -\ ISTAT/train/1_ALL_train.csv -t../../OUTPUT/1\ -\ ISTAT/test/1_ALL_test.csv -d$3" C-m
tmux send-keys -t "$1":0.3 "./har-client.py -s localhost:8080 -T../../OUTPUT/2\ -\ ONS/train/2_ALL_train.csv -t../../OUTPUT/2\ -\ ONS/test/2_ALL_test.csv -d$3" C-m
tmux send-keys -t "$1":0.4 "./har-client.py -s localhost:8080 -T../../OUTPUT/3\ -\ STATCAN/train/3_ALL_train.csv -t../../OUTPUT/3\ -\ STATCAN/test/3_ALL_test.csv -d$3" C-m


tmux attach-session -t "$1"

