#!/bin/bash

# this whole setup is not the most elegant. There has to be a bert-serving server instance running for the code to work. Once this is up, the flask app can be started. Training is then done once this is up and running. And since the whole thing would otherwise terminate when this expect script is finished, a dummy script which just never terminates is called to keep the flask app up. #!/usr/bin/expect -f


#set timeout -1
#spawn bert-serving-start -model_dir /gsdp/bert-base-german-tf-version/ -num_worker=4 -max_seq_len=52

#expect "all set, ready to serve request"

#cd gsdp

#spawn python Parser.py

#expect "Running on http://0.0.0.0:5000/ (Press CTRL+C to quit)"

#spawn curl localhost:5000/train

#expect "Successfully trained models"

#spawn ./sleep.sh

#expect "something"
bert-serving-start -model_dir /gsdp/bert-base-german-tf-version/ -num_worker=4 -max_seq_len=52 &
cd gsdp
python Parser.py

# for ELG version, I can already run curl localhost:5000/load so that everything is pre-loaded. Then there is a probe endpoint. If this returns 200, all is good to go. If it returns 500, the bert-serving-server is not (yet) up and running.
