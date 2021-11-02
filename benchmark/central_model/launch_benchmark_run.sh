#!/usr/bin/bash

NUMBER_OF_CLIENTS="$1"
SESSION_NAME="$2"

echo 
echo "Launching benchmark run $SESSION_NAME with $NUMBER_OF_CLIENTS clients"
echo

if [[ $(tmux ls | grep -c ^"$SESSION_NAME") == 1 ]]
    then 
        echo "tmux session exists $SESSION_NAME, killing session";
        tmux kill-session -t $SESSION_NAME;
fi


echo "Creating new session with name $SESSION_NAME"
tmux new-session -s "$SESSION_NAME" -d
tmux select-layout -t "$SESSION_NAME" tiled
for ((CLIENT=1; CLIENT<=$NUMBER_OF_CLIENTS;CLIENT++))
do
    tmux select-layout -t "$SESSION_NAME" tiled
    tmux split-window -h -t "$SESSION_NAME"
done
# tmux split-window -h -t "$SESSION_NAME"
# tmux select-pane -U -t "$SESSION_NAME"
# tmux split-window -h -t "$SESSION_NAME"

tmux send-keys -t "$SESSION_NAME":0.0 "./har_server.py -s [::]:8080 -r 40 -m $NUMBER_OF_CLIENTS -M $NUMBER_OF_CLIENTS -T../benchmark_data/0/train/ALL_train.csv -t../benchmark_data/0/test/ALL_test.csv" C-m
sleep 3
tmux send-keys -t "$SESSION_NAME":0.1 "export FLWR_TXSESSION=$SESSION_NAME" C-m
for ((CLIENT=1; CLIENT<=$NUMBER_OF_CLIENTS;CLIENT++))
do
    tmux send-keys -t "$SESSION_NAME":0."$CLIENT" "./har_client.py -s localhost:8080 -T../benchmark_data/$CLIENT/train/ALL_train.csv -t../benchmark_data/$CLIENT/test/ALL_test.csv" C-m
done
tmux send-keys -t "$SESSION_NAME":0."$NUMBER_OF_CLIENTS" "tmux detach" C-m
tmux attach-session -t "$SESSION_NAME"
