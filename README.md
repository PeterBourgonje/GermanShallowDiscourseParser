# GermanShallowDiscourseParser

This Shallow Discourse Parser for German is the (practical) outcome of a PhD Thesis on that very topic.


## License

This parser is released under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
You can find a human-readable summary of the licence agreement here:

https://creativecommons.org/licenses/by-nc-sa/4.0/

If you use this parser for your research, please cite the following:
TBD

## Installation & Usage

There are two ways to get this parser up and running. The easy way is by building and running the docker version. The slightly more elaborate way is by downloading and installing all requirements yourself. Both are described below.

### Docker (easy)
- Clone this repository (`git clone https://github.com/PeterBourgonje/GermanShallowDiscourseParser`)
- `cd` into the cloned folder, then build the Docker container (`docker build -t gsdp .`), where `gsdp` is the container name, i.e. can be anything you want, as long as it matches this when running the container.
- After a successful build, start the container (`sudo docker run -p5500:5000 -it gsdp`). Running the container starts the bert-serving server that is required, and starts the Flask app that exposes the two endpoints; one for training and one for parsing. This (esp. the bert-serving server) takes a few seconds to start. Wait for the message `all set, ready to serve request!` to show up in your terminal.
- Before you can start parsing, you need to train the parser. This is best done with curl. As per the command above, the flask app is exposed through the docker container at port 5500, with this the command to train the parser is `curl localhost:5500/train`. This takes a few minutes (2 minutes on a laptop/CPU with 2.2GhZ and 24GB RAM), wait for the response message `INFO: Successfully trained models.`
- The parser is now trained and ready to go. The following curl command parses the input file located at `<path/to/local/file.txt>`, and writes the output to `<output.json>`: `curl -X POST -F input=@<path/to/local/file.txt> localhost:5500/parse -o <output.json>`

Parsing is not particularly fast (ca. 6.5 tokens/second on a laptop/CPU with 2.2GhZ and 24GB RAM), so please be patient.


### Manual (less easy)
Should the Docker version not work for some reason, here is how to get it up and running manually:

- Clone this repository (`git clone https://github.com/PeterBourgonje/GermanShallowDiscourseParser`)
- Download the Stanford Parser (`wget https://nlp.stanford.edu/software/stanford-parser-full-2018-10-17.zip`) and unzip it to a local folder on your system.
- Download the German Bert model (`wget https://int-deepset-models-bert.s3.eu-central-1.amazonaws.com/tensorflow/bert-base-german-cased.zip`) and unzip it to a local folder on your system.
- Clone the DiMLex repository to your local system (`git clone https://github.com/discourse-lab/dimlex`)
- Clone the PCC repository to your local system (`git clone https://github.com/PeterBourgonje/pcc2.2`)
- Install all required python packages (`pip install -r requirements.txt`)
- Modify the paths in `config.ini` to match your system configuration. The variables you have to modify are `pccdir`, `dimlexdir`, `parserdir` and `modeldir`. Make sure these point to the locations where you have just downloaded/unzipped/cloned the respective modules.
- Manually start a bert-serving server (`bert-serving-start -model_dir <location/to/bert/model> -num_worker=4 -max_seq_len=52`), where `<location/to/bert/model>` points to where you just unzipped the Bert model. You can use a longer `max_seq_len` value if you wish (this merely controls the number of tokens after which input to vector representation is cut off). Wait for the message `all set, ready to serve request!` to show up in your terminal.
- Start the flask app (`python3 Parser.py`)
You can specify a port number (optionally) with the `--port` flag (followed by a whitespace, then the desired port number; by default 5000 is taken).
- Before you can start parsing, you need to train the parser. This is best done with curl: `curl localhost:5000/train`. This takes a few minutes (2 minutes on a laptop/CPU with 2.2GhZ and 24GB RAM), wait for the response message `INFO: Successfully trained models.`
- The parser is now trained and ready to go. The following curl command parses the input file located at `<path/to/local/file.txt>`, and writes the output to `<output.json>`: `curl -X POST -F input=@<path/to/local/file.txt> localhost:5000/parse -o <output.json>`

